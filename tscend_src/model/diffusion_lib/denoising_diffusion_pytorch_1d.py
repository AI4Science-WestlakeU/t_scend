import math
import os 
import sys
from glob import glob
import collections
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from tabulate import tabulate

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce,repeat
from einops.layers.torch import Rearrange
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from tqdm.auto import tqdm

import os.path as osp
import time
import yaml
import numpy as np
import pandas as pd
from multiprocessing import Pool
# from tscend_src.utils.utils_sudoku_discrete import MCTSDiscrete,NodeSudokuDiscrete,plot_energy_vs_distance, process_tensor_sudoku
from tscend_src.utils.utils import set_seed
from tscend_src.utils.utils_sudoku_continuous import MCTSContinuous,NodeSudokuContinuous
from tscend_src.utils.utils_maze_continuous import NodeMazeContinuous
from tscend_src.utils.utils import p, get_entropy
from tscend_src.data.data_maze import MazeData,reconstruct_maze_solved,calculate_path_conformity,calculate_path_continuity,maze_accuracy,normalize_last_dim,plot_maze,maze_accuracy_batch
def _custom_exception_hook(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, ipdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        # ...then start the debugger in post-mortem mode.
        ipdb.post_mortem(tb)


def hook_exception_ipdb():
    """Add a hook to ipdb when an exception is raised."""
    if not hasattr(_custom_exception_hook, 'origin_hook'):
        _custom_exception_hook.origin_hook = sys.excepthook
        sys.excepthook = _custom_exception_hook


def unhook_exception_ipdb():
    """Remove the hook to ipdb when an exception is raised."""
    assert hasattr(_custom_exception_hook, 'origin_hook')
    sys.excepthook = _custom_exception_hook.origin_hook

hook_exception_ipdb()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    val: float = 0
    avg: float = 0
    sum: float = 0
    sum2: float = 0
    std: float = 0
    count: float = 0
    tot_count: float = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum2 = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum2 += val * val * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count
        self.std = (self.sum2 / self.count - self.avg * self.avg) ** 0.5

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        supervise_energy_landscape = True,
        use_innerloop_opt = True,
        innerloop_opt_steps = 0,
        show_inference_tqdm = True,
        baseline = False,
        sudoku = False,
        continuous = False,
        connectivity = False,
        shortest_path = False,
        args = None,
    ):
        super().__init__()
        self.model = model
        self.inp_dim = self.model.inp_dim
        self.out_dim = self.model.out_dim
        self.out_shape = (self.out_dim, )
        self.self_condition = False
        self.supervise_energy_landscape = supervise_energy_landscape
        self.use_innerloop_opt = innerloop_opt_steps > 0

        self.seq_length = seq_length
        self.objective = objective
        self.show_inference_tqdm = show_inference_tqdm
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.args = args
        self.baseline = baseline
        self.sudoku = sudoku
        self.connectivity = connectivity
        self.continuous = continuous
        self.shortest_path = shortest_path
        self.data_label = None
        self.std_list = []
        self.consistency_list = []
        self.path_length_std_list = []

        # sampling related parameters

        # self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = sampling_timesteps
        # assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.times_diffusion = reversed(torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1))

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Step size for optimizing
        register_buffer('opt_step_size', betas * torch.sqrt( 1 / (1 - alphas_cumprod)))
        # register_buffer('opt_step_size', 0.25 * torch.sqrt(alphas_cumprod) * torch.sqrt(1 / alphas_cumprod -1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)
        # whether to autonormalize

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, inp, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        with torch.enable_grad():
            model_output = self.model(inp, x, t) # [batch_size, 729] for sudoku; [B,N,N,4],[B,8,N,1]

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, cond, x, t, x_self_cond = None, clip_denoised = False):
        preds = self.model_predictions(cond, x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            # x_start.clamp_(-6, 6)

            if self.continuous:
                sf = 2.0
            else:
                sf = 1.0

            x_start.clamp_(-sf, sf)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, cond, x, t, x_self_cond = None, clip_denoised = True, with_noise=False, scale=False, enable_grad=False):

        '''
        Sudoku:
            cond: [batch_size, 729]
            x: [batch_size, 729]
            t: int

        '''
        b, *_, device = *x.shape, x.device

        if type(t) == int:
            batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
            noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        else:
            batched_times = t
            noise = torch.randn_like(x)

        if enable_grad:
            with torch.enable_grad():
                model_mean, _, model_log_variance, x_start = self.p_mean_variance(cond, x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        else:
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(cond, x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)

        # Don't scale inputs by expansion factor (Do that later)
        if not scale:
            if enable_grad:
                with torch.enable_grad():
                    model_mean = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape) * x_start
            else:
                model_mean = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape) * x_start


        if with_noise:
            if enable_grad:
                with torch.enable_grad():
                    pred_img = model_mean  + (0.5 * model_log_variance).exp() * noise
            else:
                pred_img = model_mean  + (0.5 * model_log_variance).exp() * noise
        else:
            pred_img = model_mean #  + (0.5 * model_log_variance).exp() * noise

        return pred_img, x_start

    def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0, detach=True):
        with torch.enable_grad():
            for i in range(step):
                energy, grad = self.model(inp, img, t, return_both=True)
                img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf  # / (i + 1) ** 0.5

                if mask is not None and self.args.dataset != 'maze':
                    img_new = img_new * (1 - mask) + mask * data_cond
                elif self.args.dataset == 'maze':
                    img_new = apply_mask_maze(img_new, mask)
                if self.continuous:
                    sf = 2.0
                else:
                    sf = 1.0

                max_val = extract(self.sqrt_alphas_cumprod, t, img_new.shape)[0, 0] * sf
                img_new = torch.clamp(img_new, -max_val, max_val)

                energy_new = self.model(inp, img_new, t, return_energy=True)
                if len(energy_new.shape) == 2:
                    bad_step = (energy_new > energy)[:, 0]
                elif len(energy_new.shape) == 1:
                    bad_step = (energy_new > energy)
                else:
                    raise ValueError('Bad shape!!!')

                # print("step: ", i, bad_step.float().mean())
                img_new[bad_step] = img[bad_step]

                if eval:
                    img = img_new.detach()
                else:
                    img = img_new

        return img

    @torch.no_grad()
    def p_sample_loop(self, batch_size, shape, inp, cond, mask, return_traj=False, enable_grad_steps=-1, end_step=0):
        '''
        Sudoku:
            inp: [batch_size,729]
            mask: [batch_size, 729]
            return_traj: False
            enable_grad_steps: number of last steps that enables gradient
            end_step: the inference step where the p_sample_loop stops generating. Default 0, meaning going full generation process.
        '''
        
        device = self.betas.device

        if hasattr(self.model, 'randn'):
            img = self.model.randn(batch_size, shape, inp, device)
        else:
            img = torch.randn((batch_size, *shape), device=device)

        x_start = None


        if self.show_inference_tqdm:
            iterator = tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps)
        else:
            iterator = reversed(range(0, self.num_timesteps))

        preds = []

        for t in iterator:
            self_cond = x_start if self.self_condition else None
            batched_times = torch.full((img.shape[0],), t, device = inp.device, dtype = torch.long)

            cond_val = None
            if mask is not None and self.args.dataset!='maze':
                cond_val = self.q_sample(x_start = inp, t = batched_times, noise = torch.zeros_like(inp))
                img = img * (1 - mask) + cond_val * mask
            elif mask is not None and self.args.dataset=='maze':
                img = apply_mask_maze(img, mask)

            if enable_grad_steps == -1 or t >= enable_grad_steps + end_step:
                img, x_start = self.p_sample(inp, img, t, self_cond, scale=False, with_noise=self.baseline)
            else:
                img, x_start = self.p_sample(inp, img, t, self_cond, scale=False, with_noise=self.baseline, enable_grad=True)
            if mask is not None and self.args.dataset!='maze':
                if enable_grad_steps == -1 or t >= enable_grad_steps + end_step:
                    img = img * (1 - mask) + cond_val * mask # impainting method to do conditional sampling
                else:
                    with torch.enable_grad():
                        img = img * (1 - mask) + cond_val * mask # impainting method to do conditional sampling
            elif mask is not None and self.args.dataset=='maze':
                if enable_grad_steps == -1 or t >= enable_grad_steps + end_step:
                    img = apply_mask_maze(img, mask)
                else:
                    with torch.enable_grad():
                        img = apply_mask_maze(img, mask)
            # if t < 50:

            step = self.args.innerloop_opt_steps
            if self.use_innerloop_opt:
                if enable_grad_steps == -1 or t >= enable_grad_steps + end_step:
                    if t < 1:
                        img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)
                    else:
                        img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)

                    img = img.detach()
                else:
                    with torch.enable_grad():
                        if t < 1:
                            img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, eval=False, sf=1.0)
                        else:
                            img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, eval=False, sf=1.0)

            if self.continuous:
                sf = 2.0
            elif self.shortest_path:
                sf = 0.1
            else:
                sf = 1.0

            # This clip threshold needs to be adjust to be larger for generalizations settings
            # import pdb; pdb.set_trace()
            max_val = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape)[0, 0] * sf

            if enable_grad_steps == -1 or t >= enable_grad_steps + end_step:
                img = torch.clamp(img, -max_val, max_val)

                # Correctly scale output
                img_unscaled = self.predict_start_from_noise(img, batched_times, torch.zeros_like(img))
                preds.append(img_unscaled)

                batched_times_prev = batched_times - 1

                if t != 0:
                    img = extract(self.sqrt_alphas_cumprod, batched_times_prev, img_unscaled.shape) * img_unscaled
            else:
                with torch.enable_grad():
                    img = torch.clamp(img, -max_val, max_val)

                    # Correctly scale output
                    img_unscaled = self.predict_start_from_noise(img, batched_times, torch.zeros_like(img))
                    preds.append(img_unscaled)

                    batched_times_prev = batched_times - 1

                    if t != 0:
                        img = extract(self.sqrt_alphas_cumprod, batched_times_prev, img_unscaled.shape) * img_unscaled
            # img, _, _ = self.q_posterior(img_unscaled, img, batched_times)

            # Early stop the generation process at end_step:
            if t <= end_step:
                break

        if return_traj:
            return torch.stack(preds, dim=0)
        else:
            return img
        
    @torch.no_grad()
    def ddim_sample(self, 
                    batch_size: int, 
                    shape: tuple, 
                    inp: torch.Tensor, 
                    cond: torch.Tensor, 
                    mask: torch.Tensor, 
                    return_traj: bool=False, 
                    clip_denoised: bool=True,
                    step_now: int = 0,
                    sampling_timesteps_ddim = None,
                    time_now =None,
                    img = None,
                    ):
        """
        DDIM sampling adapted to the same interface and logic as `p_sample_loop`.
        
        Args:
            batch_size (int): how many samples to draw in parallel
            shape (tuple): shape of the sampling output, e.g. (729,)
            inp (torch.Tensor): same `inp` as in p_sample_loop (e.g. partial Sudoku board), conditions
            cond (torch.Tensor): typically ignored in this code if `self.model` doesn't use it, 
                                but kept for symmetry with p_sample_loop signature
            mask (torch.Tensor): mask to enforce partial constraints (e.g. Sudoku clues)
            return_traj (bool): if True, return the entire trajectory of unscaled predictions
            clip_denoised (bool): whether to clamp the predicted x0 in [-1, 1] (DDIM-style)
            
        Returns:
            (torch.Tensor): final sample [batch_size, *shape], or entire trajectory if return_traj=True
        """
        device = self.betas.device
        if time_now is not None:
            total_timesteps = time_now + 1
        else:
            total_timesteps = self.num_timesteps
        if sampling_timesteps_ddim is not None:
            sampling_timesteps = sampling_timesteps_ddim
        else:
            sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        # prepare the time steps for DDIM sampling
        # e.g. times = list(reversed(range(T))) when sampling_timesteps == T
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = time_pairs[step_now:]

        # initialize the sample
        if img is None:
            if hasattr(self.model, 'randn'):
                img = self.model.randn(batch_size, shape, inp, device)
            else:
                img = torch.randn((batch_size, *shape), device=device)

        x_start = None
        preds = []

        # main DDIM sampling loop
        # import pdb; pdb.set_trace()
        for time, time_next in tqdm(time_pairs, desc='DDIM sampling loop time step', disable=not self.show_inference_tqdm):
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)

            # inpainting: if mask is provided, inject the known portion from `inp`
            cond_val = None
            if mask is not None and self.args.dataset!='maze':
                cond_val = self.q_sample(x_start=inp, t=time_cond, noise=torch.zeros_like(inp))
                img = img * (1 - mask) + cond_val * mask
            elif mask is not None and self.args.dataset=='maze':
                cond_val = self.q_sample(x_start=inp, t=time_cond, noise=torch.zeros_like(inp))
                img = apply_mask_maze(img, mask)

            # 1) forward pass to predict noise or x_start
            model_pred = self.model_predictions(inp, img, time_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = model_pred.pred_noise, model_pred.pred_x_start
            # if we've reached t < 0, simply set the final image to x_start
            if time_next < 0:
                img = x_start
            else:
                # standard DDIM update
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1. - alpha / alpha_next) * (1. - alpha_next) / (1. - alpha)).sqrt()
                c = (1. - alpha_next - sigma**2).sqrt()

                noise = torch.randn_like(img)  # DDIM sample does usually use noise here
                img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            # re-apply mask after the update step if needed
            if mask is not None and self.args.dataset!='maze':
                img = img * (1 - mask) + cond_val * mask
            elif mask is not None and self.args.dataset=='maze':
                img = apply_mask_maze(img, mask)

            # possibly do an inner-loop energy-based refinement step
            step = self.args.innerloop_opt_steps

            if self.use_innerloop_opt:
                # import pdb; pdb.set_trace()
                time_next_cond = torch.full((batch_size,), time_next, device=device, dtype=torch.long)
                if time_next < 0:
                    img = self.opt_step(inp, img, time_next_cond*0, mask, cond_val, step=step, sf=1.0)
                else:
                    img = self.opt_step(inp, img, time_next_cond, mask, cond_val, step=step, sf=1.0)
                img = img.detach()

            # clamp for e.g. continuous or shortest path
            if self.continuous:
                sf = 2.0
            elif self.shortest_path:
                sf = 0.1
            else:
                sf = 1.0
            if time_next < 0:
                max_val = extract(self.sqrt_alphas_cumprod, time_cond*0, x_start.shape)[0, 0] * sf
            else:
                max_val = extract(self.sqrt_alphas_cumprod, time_cond, x_start.shape)[0, 0] * sf
            # img = torch.clamp(img, -max_val, max_val)
            max_val_xt = img.abs().max()
            scale_val = max_val / max_val_xt
            img = img * scale_val
            # energy_sample = self.model(inp, img, time_cond, return_energy=True).squeeze(-1)
            # print(f"mean of energy at time {time}", energy_sample.mean())
            # with torch.enable_grad():
            #     grad = self.model(inp, x_start, torch.zeros(inp.shape[0],device = inp.device), return_energy=False)
            #     mean_grad = grad.abs().mean()
            #     print(f"mean of abs noise at time {time}", grad.abs().mean())
            # collect the unscaled sample if needed
            img_unscaled = self.predict_start_from_noise(img, time_cond, torch.zeros_like(img))
            if return_traj:
                preds.append(img_unscaled)
        if step_now != 0 or sampling_timesteps_ddim is not None:
            with torch.enable_grad():
                if self.sudoku:
                    img_processed = process_tensor_sudoku(img)
                elif self.args.dataset == 'maze':
                    img_processed = normalize_last_dim(img).float()
                else:
                    img_processed = img
                energy, grad = self.model(inp, img_processed, time_cond*0, return_both=True)
                energy = energy.detach()
                img = img.detach()
                del grad
        # energy_sample = self.model(inp, img, torch.zeros(inp.shape[0],device = inp.device), return_energy=True).squeeze(-1)
        # print(f"??????????mean of energy at time {time}", energy_sample.mean())
        # return full trajectory or final sample
        if return_traj:
            return torch.stack(preds, dim=0)
        else:
            if step_now != 0 or sampling_timesteps_ddim is not None:
                return img, energy
            else:
                return img
    @torch.no_grad()
    def ddim_sample_baseline(self, 
                    batch_size: int, 
                    shape: tuple, 
                    inp: torch.Tensor, 
                    cond: torch.Tensor, 
                    mask: torch.Tensor, 
                    return_traj: bool=False, 
                    clip_denoised: bool=True,
                    step_now: int = 0,
                    enable_grad_steps: int=-1,
                    end_step: int=0,
                    mcts_start_step: int = -1,
                    ):
        """
        DDIM sampling adapted to the same interface and logic as `p_sample_loop` with the same forward process of model
        
        Args:
            batch_size (int): how many samples to draw in parallel
            shape (tuple): shape of the sampling output, e.g. (729,)
            inp (torch.Tensor): same `inp` as in p_sample_loop (e.g. partial Sudoku board), condition
            cond (torch.Tensor): typically ignored in this code if `self.model` doesn't use it, 
                                but kept for symmetry with p_sample_loop signature
            mask (torch.Tensor): mask to enforce partial constraints (e.g. Sudoku clues)
            return_traj (bool): if True, return the entire trajectory of unscaled predictions
            clip_denoised (bool): whether to clamp the predicted x0 in [-1, 1] (DDIM-style)
            
        Returns:
            (torch.Tensor): final sample [batch_size, *shape], or entire trajectory if return_traj=True
        """
        device = self.betas.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        if self.args.noise_type == 'gaussian':
            eta = self.args.mcts_noise_scale
        else:
            eta = 0.0

        # prepare the time steps for DDIM sampling
        # e.g. times = list(reversed(range(T))) when sampling_timesteps == T
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = time_pairs[step_now:]

        # initialize the sample
        K = self.args.K + 1
        batch_size = batch_size * K
        inp = inp.repeat(K, *([1] * (len(inp.shape)-1)))
        if cond is not None:
            cond = cond.repeat(K, *([1] * (len(cond.shape)-1)))
        if mask is not None and self.args.dataset!='maze':
            mask = mask.repeat(K, *([1] * (len(mask.shape)-1)))
        elif mask is not None and self.args.dataset=='maze':
            mask = mask.repeat(K, *([1] * (len(mask.shape)-1)))
        # import pdb; pdb.set_trace()


        if hasattr(self.model, 'randn'):
            img = self.model.randn(batch_size, shape, inp, device)
        else:
            img = torch.randn((batch_size, *shape), device=device)

        x_start = None
        preds = []

        # main DDIM sampling loop
        # import pdb; pdb.set_trace()
        step_now = 0
        for time, time_next in tqdm(time_pairs, desc='DDIM sampling loop time step', disable=not self.show_inference_tqdm):
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)

            # inpainting: if mask is provided, inject the known portion from `inp`
            cond_val = None
            if mask is not None and self.args.dataset!='maze':
                cond_val = self.q_sample(x_start=inp, t=time_cond, noise=torch.zeros_like(inp))
                img = img * (1 - mask) + cond_val * mask
            elif mask is not None and self.args.dataset=='maze':
                img = apply_mask_maze(img, mask)
            # 1) forward pass to predict noise or x_start
            model_pred = self.model_predictions(inp, img, time_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = model_pred.pred_noise, model_pred.pred_x_start
            # if we've reached t < 0, simply set the final image to x_start
            if time_next < 0:
                img = x_start
            else:
                # standard DDIM update
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1. - alpha / alpha_next) * (1. - alpha_next) / (1. - alpha)).sqrt()
                c = (1. - alpha_next - sigma**2).sqrt()

                noise = torch.randn_like(img)  # DDIM sample does usually use noise here
                img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            # re-apply mask after the update step if needed
            if mask is not None and self.args.dataset!='maze':
                img = img * (1 - mask) + cond_val * mask
            elif mask is not None and self.args.dataset=='maze':
                img = apply_mask_maze(img, mask)

            # possibly do an inner-loop energy-based refinement step
            step = self.args.innerloop_opt_steps

            if self.use_innerloop_opt:
                # import pdb; pdb.set_trace()
                img = self.opt_step(inp, img, time_cond, mask, cond_val, step=step, sf=1.0)
                img = img.detach()

            # clamp for e.g. continuous or shortest path
            if self.continuous:
                sf = 2.0
            elif self.shortest_path:
                sf = 0.1
            else:
                sf = 1.0

            max_val = extract(self.sqrt_alphas_cumprod, time_cond, x_start.shape)[0, 0] * sf #[1,1]
            # img = torch.clamp(img, -max_val, max_val)
            # max_val_xt = img.abs().max()
            # scale_val = max_val / max_val_xt
            # img = img * scale_val
            if not ('denoise' in self.args.model):
                if self.args.dataset == 'maze':
                    max_val_xt = img.abs().amax(dim=[1, 2, 3], keepdim=True)  
                    scale_val = max_val.squeeze() / max_val_xt  #  [B, 1, 1, 1]
                    img = img * scale_val
                else:
                    max_val_xt = img.abs().amax(dim=1, keepdim=True)
                    scale_val = max_val.squeeze() / max_val_xt
                    img = img * scale_val
            # print(f"mean of energy at time {time}", energy.abs().mean())
            # print(f"mean of energy at time {time}", grad.abs().mean())
            # collect the unscaled sample if needed
            if self.args.inference_method == 'mixed_inference' and self.args.mcts_start_step!=0:
                step_now += 1
            if time_next == self.args.mcts_start_step and self.args.inference_method == 'mixed_inference' and self.args.mcts_start_step!=0:
                energy_x0_hat_t = self.model(inp, img, time_cond,return_energy = True).squeeze(-1)
                return inp,img,cond,mask,energy_x0_hat_t,step_now
  
            img_unscaled = self.predict_start_from_noise(img, time_cond, torch.zeros_like(img))
            if return_traj:
                preds.append(img_unscaled)
        with torch.enable_grad():
            energy, grad = self.model(inp, img, time_cond, return_both=True)
            energy = energy.detach()
            img = img.detach()
            del grad
         # select the best branch
        batch_size_origin = batch_size // K
        energy = energy.squeeze(-1)
        # energy = rearrange(energy, '(k b) -> b k', b = batch_size_origin, k = K)
        if self.args.J_type == 'J_defined':
            pred = img.view(-1, 9, 9, 9).argmax(dim=-1)
            board_accuracy = sudoku_score(pred, reduction='none') 
            reward = board_accuracy
        elif self.args.J_type == 'energy_learned':
            reward = -energy
        elif self.args.J_type == 'mixed':
            pred = img.view(-1, 9, 9, 9).argmax(dim=-1)
            board_accuracy = sudoku_score(pred, reduction='none') 
            reward = -energy + board_accuracy*100
        elif self.args.J_type == 'GD_accuracy' and self.args.dataset == 'sudoku':
            pred = img.view(-1, 9, 9, 9).argmax(dim=-1)
            label = self.data_label.view(-1, 9, 9, 9).argmax(dim=-1)
            label = label.repeat(K, *([1] * (len(label.shape)-1)))

            correct = (pred == label).float()
            mask = mask.view(-1, 9, 9, 9)[:, :, :, 0]
            mask_inverse = 1 - mask

            if mask_inverse.sum()<1.0:
                accuracy = torch.ones(1)
            else:
                accuracy = (correct * mask_inverse).view(-1, 81).sum(dim=-1) / mask_inverse.view(-1, 81).sum(dim=-1)
            # print("accuracy",accuracy)
            reward = accuracy * 100.0
        elif self.args.J_type == 'GD_accuracy' and self.args.dataset == 'maze':
            x0_hat = normalize_last_dim(img)
            label = self.data_label
            label = label.repeat(K, *([1] * (len(label.shape)-1)))
            label = label.to(x0_hat.device)
            summary = maze_accuracy_batch(maze_cond=inp,maze_solution=x0_hat,mask=mask,label=label)

            # accuracy = (summary['rate_success']+0.1)(summary['path_precision']+summary['path_recall']+summary['path_f1']+(summary["path_length_GD"]/summary["path_length"])+0.1)/4
            rate_success = summary['rate_success'].to(x0_hat.device)
            path_precision = summary['path_precision'].to(x0_hat.device)
            path_recall = summary['path_recall'].to(x0_hat.device)
            path_f1 = summary['path_f1'].to(x0_hat.device)
            path_length_GD = summary["path_length_GD"].to(x0_hat.device)
            path_length = summary["path_length"].to(x0_hat.device)

            accuracy = (rate_success+0.1)*(path_precision+path_recall+path_f1+(path_length_GD/path_length)+0.1)/4
            reward = accuracy * 100.0
                    
        
        reward = rearrange(reward, '(k b) -> b k', b = batch_size_origin, k = K)
        # entropy=get_entropy(img.flatten(start_dim=1), K=1, NN=100, stop_grad_reference=True) # img [K,H,W,2]
        std = torch.std(img.flatten(start_dim=1), dim=0).mean()
        img = rearrange(img, '(k b) ... -> b k ...', b = batch_size_origin, k = K)
        self.std_list.append(std.cpu().numpy())
        reward_energy = -energy
        reward_energy = rearrange(reward_energy, '(k b) -> b k', b = batch_size_origin, k = K)
        topk_energy = torch.sort(reward_energy, dim=1, descending=True)
        topk_reward = torch.sort(reward, dim=1, descending=True)
        # if self.sampling_timesteps ==10 and img.shape[0]>1:
        if False:
            consistency = calculate_order_consistency(topk_energy.indices.squeeze(0).cpu().numpy(),topk_reward.indices.squeeze(0).cpu().numpy())
        else:
            consistency = 0
        self.consistency_list.append(consistency)
        # path_length_std = torch.std(path_length.flatten())
        path_length_std = 0
        self.path_length_std_list.append(0)
        # best_idxs = torch.argmin(energy, dim=1)  # [B]
        best_idxs = torch.argmax(reward, dim=1)
        img = img[torch.arange(batch_size_origin), best_idxs]
         # return full trajectory or final sample
        if return_traj:
            return torch.stack(preds, dim=0)
        else:
            if step_now != 0:
                return img, energy
            else:
                return img


    @torch.no_grad()
    def ddim_sample_mcts(self, 
                    batch_size: int, 
                    shape: tuple, 
                    inp: torch.Tensor, 
                    cond: torch.Tensor, 
                    mask: torch.Tensor, 
                    return_traj: bool=False, 
                    clip_denoised: bool=True,
                    enable_grad_steps: int=-1,
                    end_step: int=0,
                    img = None,
                    step_now = 0,
        ):
        """
        DDIM sampling adapted to the same interface and logic as `p_sample_loop`.
        
        Args:
            batch_size (int): how many samples to draw in parallel
            shape (tuple): shape of the sampling output, e.g. (729,)
            inp (torch.Tensor): same `inp` as in p_sample_loop (e.g. partial Sudoku board)
            cond (torch.Tensor): typically ignored in this code if `self.model` doesn't use it, 
                                but kept for symmetry with p_sample_loop signature
            mask (torch.Tensor): mask to enforce partial constraints (e.g. Sudoku clues)
            return_traj (bool): if True, return the entire trajectory of unscaled predictions
            clip_denoised (bool): whether to clamp the predicted x0 in [-1, 1] (DDIM-style)
            
        Returns:
            (torch.Tensor): final sample [batch_size, *shape], or entire trajectory if return_traj=True
        """

        device = self.betas.device
        if self.args.inference_method == 'mixed_inference' and self.args.mcts_start_step!=9:
            total_timesteps =  self.args.mcts_start_step +1 # ddpm step start from 0 
            batch_size_origin = batch_size
            batch_size = inp.shape[0]
            sampling_timesteps = self.sampling_timesteps-step_now
        else:
            total_timesteps = self.num_timesteps
            sampling_timesteps = self.sampling_timesteps
        eta = self.args.mcts_noise_scale

        # prepare the time steps for DDIM sampling
        # e.g. times = list(reversed(range(T))) when sampling_timesteps == T
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # initialize the sample
        if img is None:
            if hasattr(self.model, 'randn'):
                img = self.model.randn(batch_size, shape, inp, device)
            else:
                img = torch.randn((batch_size, *shape), device=device)

        x_start = None
        preds = []

        # main DDIM sampling loop
        # import pdb; pdb.set_trace()
        for time, time_next in tqdm(time_pairs, desc='DDIM sampling loop time step', disable=not self.show_inference_tqdm):
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)

            # inpainting: if mask is provided, inject the known portion from `inp`
            cond_val = None
            if mask is not None and self.args.dataset!='maze':
                cond_val = self.q_sample(x_start=inp, t=time_cond, noise=torch.zeros_like(inp))
                img = img * (1 - mask) + cond_val * mask
            elif mask is not None and self.args.dataset=='maze':
                img = apply_mask_maze(img, mask)

            # 1) forward pass to predict noise or x_start
            model_pred = self.model_predictions(inp, img, time_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = model_pred.pred_noise, model_pred.pred_x_start

            # if we've reached t < 0, simply set the final image to x_start
            if time_next < 0:
                img = x_start
            else:
                # standard DDIM update
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1. - alpha / alpha_next) * (1. - alpha_next) / (1. - alpha)).sqrt()
                c = (1. - alpha_next - sigma**2).sqrt()
                if False:
                    noise = torch.randn_like(img)  # DDIM sample does usually use noise here
                    img = x_start * alpha_next.sqrt() + c * pred_noise 
                else:
                    coef_alpha = alpha_next.sqrt()
                    coef_c = c
                    coef_sigma = sigma
                    if self.args.mcts_type == 'continuous':
                        noise = torch.randn_like(img)  # DDIM sample does usually use noise here
                        img = x_start * alpha_next.sqrt() + c * pred_noise
                        # print("percentage of mask",mask.float().mean())

                        inp,img,mask = self.mcts_search_continuous(img,# TODO
                                                        time,
                                                        step_now,
                                                        pred_noise,
                                                        inp,
                                                        cond,
                                                        mask,
                                                        cond_val,
                                                        clip_denoised,
                                                        K = self.args.K,
                                                        coef_alpha = coef_alpha,
                                                        coef_c = coef_c,
                                                        coef_sigma = coef_sigma)
                    else:
                        noise = torch.randn_like(img)  # DDIM sample does usually use noise here
                        img = x_start * alpha_next.sqrt() + c * pred_noise
                        # print("percentage of mask",mask.float().mean())
                        inp,img,mask = self.mcts_search_discrete(img,# TODO
                                                        time,
                                                        step_now,
                                                        pred_noise,
                                                        inp,
                                                        cond,
                                                        mask,
                                                        cond_val,
                                                        clip_denoised,
                                                        K = self.args.K,
                                                        coef_alpha = coef_alpha,
                                                        coef_c = coef_c,
                                                        coef_sigma = coef_sigma)
                        if mask is not None:
                            cond_val = self.q_sample(x_start=inp, t=time_cond, noise=torch.zeros_like(inp))
            step_now += 1
                        
            # re-apply mask after the update step if needed
            if mask is not None and self.args.dataset!='maze':
                img = img * (1 - mask) + cond_val * mask
            elif mask is not None and self.args.dataset=='maze':
                img = apply_mask_maze(img, mask)

            # possibly do an inner-loop energy-based refinement step
            step = self.args.innerloop_opt_steps

            if self.use_innerloop_opt:
                # import pdb; pdb.set_trace()
                img = self.opt_step(inp, img, time_cond, mask, cond_val, step=step, sf=1.0)
                img = img.detach()
            # clamp for e.g. continuous or shortest path
            if self.continuous:
                sf = 2.0
            elif self.shortest_path:
                sf = 0.1
            else:
                sf = 1.0
            max_val = extract(self.sqrt_alphas_cumprod, time_cond, x_start.shape)[0, 0] * sf
            # img = torch.clamp(img, -max_val, max_val)
            max_val_xt = img.abs().max()
            scale_val = max_val / max_val_xt
            img = img * scale_val
            if self.args.inference_method == 'mixed_inference' and  self.args.max_root_num>1 and self.args.mcts_start_step!=9:
                topk = max(int((self.args.K+1) * (self.args.num_rood_decay**(step_now+1)),1))
                topk = min(topk,self.args.max_root_num)
                with torch.no_grad():
                    energy_x0_hat_t = self.model(inp, x_start, time_cond*0,return_energy = True)
                K_remain = int(batch_size/batch_size_origin)
                img = rearrange(img, '(k b) ... -> b k ...', b = batch_size_origin,k = K_remain) # [B, K_remain, ...]
                inp =  rearrange(inp, '(k b) ... -> b k ...', b = batch_size_origin, k = K_remain) # [B, K_remain, ...]
                if mask is not None:
                    mask =  rearrange(mask, '(k b) ... -> b k ...', b = batch_size, k = K_remain) # [B, K_remain, ...]

                sorted_values, sorted_indices = torch.sort(energy_x0_hat_t, dim=1)
                best_idxs_topk = sorted_indices[:, :topk]
                
                img = img[torch.arange(batch_size), best_idxs_topk] # [B, topk,...]
                inp = inp[torch.arange(batch_size), best_idxs_topk] # [B, topk,...]
                if mask is not None:
                    mask = mask[torch.arange(batch_size), best_idxs_topk] # [B, topk,...]

                img = rearrange(img, 'b k ... -> (k b) ...') # [B*topk, ...]
                inp = rearrange(inp, 'b k ... -> (k b) ...') # [B*topk, ...]
                if mask is not None:
                    mask = rearrange(mask, 'b k ... -> (k b) ...') # [B*topk, ...]
                batch_size = inp.shape[0]
            # collect the unscaled sample if needed
            img_unscaled = self.predict_start_from_noise(img, time_cond, torch.zeros_like(img))
            if return_traj:
                preds.append(img_unscaled)

        # return full trajectory or final sample
        if return_traj:
            return torch.stack(preds, dim=0)
        else:
            return img
        
    @torch.no_grad()
    def mcts_search_continuous(self,xt,time,step_now,pred_noise,inp,cond,mask,cond_val,clip_denoised,K=10,coef_alpha = 1.0,coef_c = 1.0,coef_sigma = 1.0): # TODO
        """
        expand K branches and select the best one
            args:
                x_start: [B,h]
        """
        mcts = MCTSContinuous(diffusion=self)
        noise_K = torch.randn((self.args.K, *xt[0].shape), device=xt.device)
        state = {
                "xt": xt,                 
                "time": time,             
                "step_now": step_now,    
                "pred_noise": pred_noise,
                "inp": inp, 
                "cond": cond,
                "mask": mask,
                "noise_K": noise_K,
        }
        if 'sudoku' in self.args.dataset:
            node_init = NodeSudokuContinuous(state,K=self.args.K,noise_K=noise_K,pred_noise = pred_noise)
        elif 'shortest-path' in self.args.dataset:
            node_init = NodeShortestPathContinuous(state,K=self.args.K,noise_K=noise_K,pred_noise = pred_noise)
        elif 'maze' in self.args.dataset:
            node_init = NodeMazeContinuous(state,K=self.args.K,noise_K=noise_K,pred_noise = pred_noise)
        else:
            assert False, "Not implemented yet"
        # try:
        for  _ in range (self.args.steps_rollout):
            mcts.do_rollout(node_init)
        best_node = mcts.choose(node_init)
        print("MCTS search over")
        return  best_node.state["inp"],best_node.state["xt"], best_node.state["mask"]
        
    @torch.no_grad()
    def mcts_search_discrete(self,xt,time,step_now,pred_noise,inp,cond,mask,cond_val,clip_denoised,K=10,coef_alpha = 1.0,coef_c = 1.0,coef_sigma = 1.0): # TODO
        mcts = MCTSDiscrete(diffusion=self)
        noise_K = torch.randn((self.args.K, *xt[0].shape), device=xt.device) # [K, 1, 729]
        # import pdb; pdb.set_trace()
        state = {
                "xt": xt,                 
                "time": time,             
                "step_now": step_now,    
                "pred_noise": pred_noise,
                "inp": inp, 
                "cond": cond,
                "mask": mask,}
        if 'sudoku' in self.args.dataset:
            node_init = NodeSudokuDiscrete(state,K=self.args.K,noise_K=noise_K,pred_noise = pred_noise)
        elif 'shortest-path' in self.args.dataset:
            raise ValueError('Not implemented yet')
        elif 'maze' in self.args.dataset:
            assert False, "Not implemented yet"
        # try:
        for  _ in range (self.args.steps_rollout):
            mcts.do_rollout(node_init)
        best_node = mcts.choose(node_init)
        print("MCTS search over")
        return  best_node.state["inp"],best_node.state["xt"], best_node.state["mask"]


    def mixed_sample(self, 
                    batch_size: int, 
                    shape: tuple, 
                    inp: torch.Tensor, 
                    cond: torch.Tensor, 
                    mask: torch.Tensor, 
                    return_traj: bool=False, 
                    clip_denoised: bool=True,
                    enable_grad_steps: int=-1,
                    end_step: int=0,
        ):
        '''
        combine the MCTS and DDIM samplinbaseline search methods to get the best sample
        '''
        inp,img,cond,mask,energy_x0_hat_t,step_now= self.ddim_sample_baseline(batch_size = batch_size,
                                        shape = shape,
                                        inp = inp,
                                        cond = cond,
                                        mask = mask,
                                        return_traj = return_traj,
                                        clip_denoised = clip_denoised,
                                        enable_grad_steps = enable_grad_steps,
                                        end_step = end_step,
                                        mcts_start_step=self.args.mcts_start_step
        )

        topk = max(int((self.args.K+1) * self.args.num_rood_decay),1)
        topk = min(topk,self.args.max_root_num)
        energy_x0_hat_t = rearrange(energy_x0_hat_t, '(k b) -> b k', b = batch_size, k = self.args.K+1) # [B, K]
        img = rearrange(img, '(k b) ... -> b k ...', b = batch_size, k = self.args.K+1)  # [B, K, ...]
        inp =  rearrange(inp, '(k b) ... -> b k ...', b = batch_size, k = self.args.K+1) # [B, K, ...]
        if mask is not None:
            mask =  rearrange(mask, '(k b) ... -> b k ...', b = batch_size, k = self.args.K+1)  # [B, K, ...]
        if cond is not None:
            cond =  rearrange(cond, '(k b) ... -> b k ...', b = batch_size, k = self.args.K+1)

        sorted_values, sorted_indices = torch.sort(energy_x0_hat_t, dim=1)
        best_idxs_topk = sorted_indices[:, :topk]
        
        img = img[torch.arange(batch_size), best_idxs_topk] # [B, topk,...]
        inp = inp[torch.arange(batch_size), best_idxs_topk] # [B, topk,...]
        if mask is not None:
            mask = mask[torch.arange(batch_size), best_idxs_topk] # [B, topk,...]
        if cond is not None:
            cond = cond[torch.arange(batch_size), best_idxs_topk]

        img = rearrange(img, 'b k ... -> (k b) ...') # [B*topk, ...]
        inp = rearrange(inp, 'b k ... -> (k b) ...') # [B*topk, ...]
        if mask is not None:
            mask = rearrange(mask, 'b k ... -> (k b) ...') # [B*topk, ...]
        if cond is not None:
            cond = rearrange(cond, 'b k ... -> (k b) ...')

        ## use mcts to search the best sample
        preds= self.ddim_sample_mcts(batch_size = batch_size,
                                        shape = shape,
                                        inp = inp,
                                        cond  = cond,
                                        mask = mask,
                                        return_traj = return_traj,
                                        clip_denoised = clip_denoised,
                                        enable_grad_steps = enable_grad_steps,
                                        end_step = end_step,
                                        img = img,
                                        step_now = step_now
        )
        if return_traj:
            return torch.stack(preds, dim=0)
        else:
            img = preds
            return img
        
    @torch.no_grad()
    def sample(self, x, label, mask, batch_size = 16, return_traj = False, enable_grad_steps = -1, end_step=0):
        """
        Args:
            x: [B, F], condition, F = H x W x num_category
            label: [B, F], label
            mask: [B, F], if 1, that element is provided as condition.
        """
        # seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.args.inference_method == 'mcts':
            print('??????? Using MCTS ?????????')
            sample_fn = self.ddim_sample_mcts
        elif self.args.inference_method == 'diffusion_baseline':
            print('??????? Using Diffusion Baseline ?????????')
            sample_fn = self.ddim_sample_baseline
        elif self.args.inference_method == 'mixed_inference' and self.args.mcts_start_step==9:
            print('??????? Using MCTS ?????????')
            sample_fn = self.ddim_sample_mcts
        elif self.args.inference_method == 'mixed_inference' and self.args.mcts_start_step==0:
            print('??????? Using Baseline ?????????')
            sample_fn = self.ddim_sample_baseline
        elif self.args.inference_method == 'mixed_inference': ## mcts_start_step \in [1,8]
            print('??????? Using Mixed Inference ?????????')
            sample_fn = self.mixed_sample
        self.out_shape = x.shape[1:]
        return sample_fn(batch_size, self.out_shape, x, label, mask, return_traj=return_traj, enable_grad_steps=enable_grad_steps, end_step=end_step)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, inp, x_start, mask, t, noise = None, is_kl_loss = True):
        b, *c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)  # [B, H, W, C] for maze

        if mask is not None and self.args.dataset != 'maze':
            # Mask out inputs
            x_cond = self.q_sample(x_start = inp, t = t, noise = torch.zeros_like(noise))
            x = x * (1 - mask) + mask * x_cond
        elif mask is not None and self.args.dataset == 'maze':
            x = apply_mask_maze(x, mask)

        # predict and take gradient step

        model_out = self.model(inp, x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if mask is not None and self.args.dataset != 'maze':
            # Mask out targets
            model_out = model_out * (1 - mask) + mask * target
        elif mask is not None and self.args.dataset == 'maze':
            model_out[mask] = target[mask] # do not caculate loss on startpoint and endpoint
        if self.args.loss_type == 'mse':
            loss = F.mse_loss(model_out, target, reduction = 'none')
        elif self.args.loss_type == 'mse_mae':
            loss = F.mse_loss(model_out, target, reduction = 'none') + F.l1_loss(model_out, target, reduction = 'none')


        if self.shortest_path:
            mask1 = (x_start > 0)
            mask2 = torch.logical_not(mask1)
            # mask1, mask2 = mask1.float(), mask2.float()
            weight = mask1 * 10 + mask2 * 0.5
            # loss = (loss * weight) / weight.sum() * target.numel()
            loss = loss * weight

        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = (loss * extract(self.loss_weight, t, loss.shape)).mean()
        loss_mse = loss

        loss_dict = {"loss_denoise": loss_mse.item()}
        if self.supervise_energy_landscape:
            noise = torch.randn_like(x_start)
            data_sample = self.q_sample(x_start = x_start, t = t, noise = noise)

            if mask is not None and self.args.dataset != 'maze':
                data_cond = self.q_sample(x_start = x_start, t = t, noise = torch.zeros_like(noise))
                data_sample = data_sample * (1 - mask) + mask * data_cond
            elif mask is not None and self.args.dataset == 'maze':
                data_sample = apply_mask_maze(data_sample, mask)
            # Add a noise contrastive estimation term with samples drawn from the data distribution
            #noise = torch.randn_like(x_start)

            # Optimize a sample using gradient descent on energy landscape
            xmin_noise = self.q_sample(x_start = x_start, t = t, noise = 3.0 * noise)

            if mask is not None and self.args.dataset != 'maze':
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
            elif mask is not None and self.args.dataset == 'maze':
                xmin_noise = apply_mask_maze(xmin_noise, mask)
            else:
                data_cond = None

            if self.sudoku:
                s = x_start.size()
                x_start_im = x_start.view(-1, 9, 9, 9).argmax(dim=-1)
                randperm = torch.randint(0, 9, x_start_im.size(), device=x_start_im.device)

                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()

                xmin_noise_im = x_start_im * (1 - rand_mask) + randperm * (rand_mask)

                xmin_noise_im = F.one_hot(xmin_noise_im.long(), num_classes=9)
                xmin_noise_im = (xmin_noise_im - 0.5) * 2

                xmin_noise_rescale = xmin_noise_im.view(-1, 729)

                loss_opt = torch.ones(1)

                loss_scale = 0.05
            elif self.args.dataset == 'maze':
                x_start_im = x_start # [B,H,W,2]
                x_start_im = x_start_im.argmax(dim=-1) # [B,H,W]


                randperm = torch.randint(0, 2, x_start_im.size(), device=x_start_im.device)

                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()

                xmin_noise_im = x_start_im * (1 - rand_mask) + randperm * (rand_mask)

                xmin_noise_im = F.one_hot(xmin_noise_im.long(), num_classes=2)
                xmin_noise_im = (xmin_noise_im - 0.5) * 2

                xmin_noise_rescale = xmin_noise_im

                loss_opt = torch.ones(1)

                loss_scale = 0.05

            elif self.connectivity:
                s = x_start.size()
                x_start_im = x_start.view(-1, 12, 12)
                randperm = (torch.randint(0, 1, x_start_im.size(), device=x_start_im.device) - 0.5) * 2

                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()

                xmin_noise_rescale = x_start_im * (1 - rand_mask) + randperm * (rand_mask)

                loss_opt = torch.ones(1)

                loss_scale = 0.05
            elif self.shortest_path:
                x_start_list = x_start.argmax(dim=2)
                classes = x_start.size(2)
                rand_vals = torch.randint(0, classes, x_start_list.size()).to(x_start.device)

                x_start_neg = torch.cat([rand_vals[:, :1], x_start_list[:, 1:]], dim=1)
                x_start_neg_oh = F.one_hot(x_start_neg[:, :, 0].long(), num_classes=classes)[:, :, :, None]
                xmin_noise_rescale = (x_start_neg_oh - 0.5) * 2

                loss_opt = torch.ones(1)

                loss_scale = 0.5
            else:

                xmin_noise = self.opt_step(inp, xmin_noise, t, mask, data_cond, step=2, sf=1.0)
                xmin = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                loss_opt = torch.pow(xmin_noise - xmin, 2).mean()

                xmin_noise = xmin_noise.detach()
                xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t, torch.zeros_like(xmin_noise))
                xmin_noise_rescale = torch.clamp(xmin_noise_rescale, -2, 2)

                # loss_opt = torch.ones(1)


                # rand_mask = (torch.rand(x_start.size(), device=x_start.device) < 0.2).float()

                # xmin_noise_rescale =  x_start * (1 - rand_mask) + rand_mask * x_start_noise

                # nrep = 1


                loss_scale = 0.5

            xmin_noise = self.q_sample(x_start=xmin_noise_rescale, t=t, noise=noise)

            if mask is not None and self.args.dataset != 'maze':
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
            elif mask is not None and self.args.dataset == 'maze':
                xmin_noise = apply_mask_maze(xmin_noise, mask)

            # Compute energy of both distributions
            inp_concat = torch.cat([inp, inp], dim=0)
            x_concat = torch.cat([data_sample, xmin_noise], dim=0)
            # x_concat = torch.cat([xmin, xmin_noise_min], dim=0)
            t_concat = torch.cat([t, t], dim=0)
            energy = self.model(inp_concat, x_concat, t_concat, return_energy=True)

            # Compute noise contrastive energy loss
            energy_real, energy_fake = torch.chunk(energy, 2, 0)
            energy_stack = torch.cat([energy_real, energy_fake], dim=-1)
            target = torch.zeros(energy_real.size(0)).to(energy_stack.device)
            loss_energy = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')[:, None].mean()

            # loss_energy = energy_real.mean() - energy_fake.mean()# loss_energy.mean()

            loss_dict["loss_energy"] = loss_energy.item()
            loss_dict["loss_opt"] = loss_opt.item()
            loss = loss_mse + loss_scale * loss_energy # + 0.001 * loss_opt
        else:
            loss = loss_mse
            loss_dict["loss_energy"] = -1
            loss_dict["loss_opt"] = -1

        if is_kl_loss and self.args.kl_coef > 0:
            t_end_step = torch.randint(0, self.args.kl_max_end_step+1, (1,), device=t.device)
            samples = self.sample(inp, None, mask, batch_size=x_start.shape[0], return_traj=False, enable_grad_steps=self.args.kl_enable_grad_steps, end_step=t_end_step.item())

            # Compute energy for the samples:
            t_end_step = t_end_step.repeat(t.shape[0])
            self.model.requires_grad_(False)
            energy_samples = self.model(inp, samples, t_end_step, return_energy=True)
            self.model.requires_grad_(True)
            loss_kl = energy_samples.mean()
            loss_dict["loss_kl"] = loss_kl.item()
            loss = loss + loss_kl * self.args.kl_coef
        else:
            loss_dict["loss_kl"] = 0

        if is_kl_loss and self.args.entropy_coef > 0: # TODO
            ##############-----------------Maximum entropy Loss training pipeline-----------------##############
            # Compute energy for the samples:
            entropy_samples = get_entropy(samples.flatten(start_dim=1), K=self.args.entropy_k_nearest_neighbor, NN=100, stop_grad_reference=True)

            # Compute entropy loss (energy_samples.max() is for offset):
            loss_entropy = -entropy_samples.mean()
            loss_dict["loss_entropy"] = loss_entropy.item()
            loss = loss + loss_entropy * self.args.entropy_coef
        else:
            loss_dict["loss_entropy"] = 0

            ##############-----------------Maximum entropy Loss training pipeline-----------------##############
        loss_neg_x0 = torch.tensor(0., dtype=torch.float32, device = x_start.device)
        loss_neg_xt = torch.tensor(0., dtype=torch.float32, device = x_start.device)
        loss_dict["loss_neg_x0"] = 0
        loss_dict["loss_neg_xt"] = 0
        if self.args.neg_contrast_coef > 0: # TODO
            ##############-----------------Negative Contrastive Loss training pipeline-----------------##############
            # print("Use negative contrastive loss")
            ## only consider do negative contrastive loss for x0
            # p_close= np.random.uniform(0, 1.0)
            # p_far = np.random.uniform(p_close, 1.0)
            # if self.args.neg_contrast_coef_x0>0:
            if True:
                if self.sudoku:
                    # resolution = 1.0/81.0
                    # p_close = np.random.uniform(0, self.args.max_strength_permutation_x0)
                    # p_far = np.random.uniform(p_close, self.args.max_strength_permutation_x0)
                    # p_close= p_close ** 2
                    # p_far = p_far ** 2
                    # p_close_num_entry = int(p_close / resolution)
                    # p_far_num_entry = int(p_far / resolution)
                    # p_gap_max = np.random.uniform(self.args.min_gap_x0, self.args.max_gap_x0)
                    # p_gap_max = p_gap_max ** 2
                    # gap_num_entry_max = max(int(p_gap_max / resolution),1)
                    # if p_close_num_entry == p_far_num_entry or (p_far_num_entry - p_close_num_entry) > gap_num_entry_max:
                    #     p_far_num_entry= p_close_num_entry + gap_num_entry_max
                    # p_close = p_close_num_entry * resolution
                    # p_far = p_far_num_entry * resolution
                    if self.args.num_distance_neg_contrast == 1:
                        if self.args.diverse_gap_batch:
                            p_close,p_far = generate_p_close_p_far(batch_size = x_start.size(0),
                                                            max_strength_permutation = self.args.max_strength_permutation_x0,
                                                            max_gap = self.args.max_gap_x0,
                                                            min_gap = self.args.min_gap_x0,
                                                            device = x_start.device
                                                            ) # [B,],[B,]
                        else:
                            p_close,p_far = generate_p_close_p_far(batch_size = 1,
                                                            max_strength_permutation = self.args.max_strength_permutation_x0,
                                                            max_gap = self.args.max_gap_x0,
                                                            min_gap = self.args.min_gap_x0,
                                                            device = x_start.device
                                                            )
                        x_start_close = random_permute_last_dim(x_start, p_close.unsqueeze(-1).unsqueeze(-1))
                        x_start_far = random_permute_last_dim(x_start, p_far.unsqueeze(-1).unsqueeze(-1))
                        x_start_repeat = x_start
                        inp_repeat = inp
                        mask_repeat = mask
                        t_repeat = t
                    elif self.args.num_distance_neg_contrast > 1:
                        batch_szie = x_start.size(0)
                        p_stack = generate_p_stack(batch_size=x_start.shape[0], 
                                                   max_strength_permutation=self.args.max_strength_permutation_x0, 
                                                   min_gap = self.args.min_gap_x0,
                                                   max_gap =self.args.max_gap_x0,
                                                   num_distance_neg_contrast = self.args.num_distance_neg_contrast,
                                                   device=x_start.device
                        ) # [num_distance_neg_contrast, B]
                        p_stack = p_stack.repeat(1,self.args.num_sample_each_neg_contrast) # [num_distance_neg_contrast, B*num_sample_each_neg_contrast]
                        p_close = p_stack[:-1].reshape(-1)
                        p_far = p_stack[1:].reshape(-1)
                        p_stack_flaten = p_stack.reshape(-1) # [num_distance_neg_contrast*B*num_sample_each_neg_contrast]

                        x_start_repeat = x_start.repeat(self.args.num_sample_each_neg_contrast*self.args.num_distance_neg_contrast,1) # [num_distance_neg_contrast*B*num_sample_each_neg_contrast*B, 729]
                        x_start_neg = random_permute_last_dim(x_start_repeat, p_stack_flaten.unsqueeze(-1).unsqueeze(-1)) # [num_distance_neg_contrast*B*num_sample_each_neg_contrast*B, 729]
                        x_start_neg = x_start_neg.reshape(self.args.num_distance_neg_contrast,batch_szie*self.args.num_sample_each_neg_contrast,-1) # [num_distance_neg_contrast, B*num_sample_each_neg_contrast, 729]

                        x_start_close = x_start_neg[:-1].reshape(-1,x_start_neg.size(-1)) # [(num_distance_neg_contrast-1)*B*num_sample_each_neg_contrast, 729]
                        x_start_far = x_start_neg[1:].reshape(-1,x_start_neg.size(-1)) # [(num_distance_neg_contrast-1)*B*num_sample_each_neg_contrast, 729]

                        x_start_repeat = x_start_repeat.reshape(self.args.num_distance_neg_contrast,batch_szie*self.args.num_sample_each_neg_contrast,-1) # [num_distance_neg_contrast, B*num_sample_each_neg_contrast, 729]
                        x_start_repeat = x_start_repeat[1:] # [(num_distance_neg_contrast-1),B*num_sample_each_neg_contrast, 729]
                        x_start_repeat = x_start_repeat.reshape(-1,x_start_repeat.size(-1)) # [(num_distance_neg_contrast-1)*B*num_sample_each_neg_contrast, 729]
                        inp_repeat = inp.repeat(self.args.num_sample_each_neg_contrast,1).repeat(self.args.num_distance_neg_contrast-1,1)
                        mask_repeat = mask.repeat(self.args.num_sample_each_neg_contrast,1).repeat(self.args.num_distance_neg_contrast-1,1)
                        t_repeat = t.repeat(self.args.num_sample_each_neg_contrast).repeat(self.args.num_distance_neg_contrast-1)
                elif self.args.dataset == 'maze':
                    # resolution = 1.0/81.0
                    # p_close = np.random.uniform(0, self.args.max_strength_permutation_x0)
                    # p_far = np.random.uniform(p_close, self.args.max_strength_permutation_x0)
                    # p_close= p_close ** 2
                    # p_far = p_far ** 2
                    # p_close_num_entry = int(p_close / resolution)
                    # p_far_num_entry = int(p_far / resolution)
                    # p_gap_max = np.random.uniform(self.args.min_gap_x0, self.args.max_gap_x0)
                    # p_gap_max = p_gap_max ** 2
                    # gap_num_entry_max = max(int(p_gap_max / resolution),1)
                    # if p_close_num_entry == p_far_num_entry or (p_far_num_entry - p_close_num_entry) > gap_num_entry_max:
                    #     p_far_num_entry= p_close_num_entry + gap_num_entry_max
                    # p_close = p_close_num_entry * resolution
                    # p_far = p_far_num_entry * resolution
                    if self.args.num_distance_neg_contrast == 1:
                        if self.args.diverse_gap_batch:
                            p_close,p_far = generate_p_close_p_far(batch_size = x_start.size(0),
                                                            max_strength_permutation = self.args.max_strength_permutation_x0,
                                                            max_gap = self.args.max_gap_x0,
                                                            min_gap = self.args.min_gap_x0,
                                                            device = x_start.device
                                                            ) # [B,],[B,]
                        else:
                            p_close,p_far = generate_p_close_p_far(batch_size = 1,
                                                            max_strength_permutation = self.args.max_strength_permutation_x0,
                                                            max_gap = self.args.max_gap_x0,
                                                            min_gap = self.args.min_gap_x0,
                                                            device = x_start.device
                                                            )
                        x_start_close = random_permute_last_dim(x_start, p_close.unsqueeze(-1).unsqueeze(-1))
                        x_start_far = random_permute_last_dim(x_start, p_far.unsqueeze(-1).unsqueeze(-1))
                        x_start_repeat = x_start # [B, H, W, 2]
                        inp_repeat = inp # [B, H, W, 5]
                        mask_repeat = mask # [B,H,W]
                        t_repeat = t
                    elif self.args.num_distance_neg_contrast > 1:
                        batch_szie = x_start.size(0)
                        p_stack = generate_p_stack(batch_size=x_start.shape[0], 
                                                   max_strength_permutation=self.args.max_strength_permutation_x0, 
                                                   min_gap = self.args.min_gap_x0,
                                                   max_gap =self.args.max_gap_x0,
                                                   num_distance_neg_contrast = self.args.num_distance_neg_contrast,
                                                   device=x_start.device
                        ) # [num_distance_neg_contrast, B]
                        p_stack = p_stack.repeat(1,self.args.num_sample_each_neg_contrast) # [num_distance_neg_contrast, B*num_sample_each_neg_contrast]
                        p_close = p_stack[:-1].reshape(-1)
                        p_far = p_stack[1:].reshape(-1)
                        p_stack_flaten = p_stack.reshape(-1) # [num_distance_neg_contrast*B*num_sample_each_neg_contrast]

                        x_start_repeat = x_start.repeat(self.args.num_sample_each_neg_contrast*self.args.num_distance_neg_contrast,[1] * (len(x_start) - 2)) # [num_distance_neg_contrast*B*num_sample_each_neg_contrast*B, 729]
                        x_start_neg = random_permute_last_dim(x_start_repeat, p_stack_flaten.unsqueeze(-1).unsqueeze(-1)) # [num_distance_neg_contrast*B*num_sample_each_neg_contrast*B, 729]
                        x_start_neg = x_start_neg.reshape(self.args.num_distance_neg_contrast,batch_szie*self.args.num_sample_each_neg_contrast,*x_start.size()[1:])

                        x_start_close = x_start_neg[:-1].reshape(-1,*x_start.size()[2:]) # [(num_distance_neg_contrast-1)*B*num_sample_each_neg_contrast, 729]
                        x_start_far = x_start_neg[1:].reshape(-1,*x_start.size()[2:])

                        x_start_repeat = x_start_repeat.reshape(self.args.num_distance_neg_contrast,batch_szie*self.args.num_sample_each_neg_contrast,*x_start.size()[1:])
                        x_start_repeat = x_start_repeat[1:] # [(num_distance_neg_contrast-1),B*num_sample_each_neg_contrast, 729]
                        x_start_repeat = x_start_repeat.reshape(-1,*x_start.size()[2:]) # [(num_distance_neg_contrast-1)*B*num_sample_each_neg_contrast, 729]
                        inp_repeat = inp.repeat(self.args.num_sample_each_neg_contrast,[1] * (len(x_start) - 1)).repeat(self.args.num_distance_neg_contrast-1,[1] * (len(x_start) - 1))
                        mask_repeat = mask.repeat(self.args.num_sample_each_neg_contrast,[1] * (len(x_start) - 1)).repeat(self.args.num_distance_neg_contrast-1,[1] * (len(x_start) - 1))
                        t_repeat = t.repeat(self.args.num_sample_each_neg_contrast).repeat(self.args.num_distance_neg_contrast-1)
                else:
                    raise ValueError('Not implemented yet')

                if mask is not None and self.args.dataset != 'maze':
                    x_start_close = x_start_close * (1 - mask_repeat) + mask_repeat * x_start_repeat
                    x_start_far = x_start_far * (1 - mask_repeat) + mask_repeat * x_start_repeat
                elif mask is not None and self.args.dataset == 'maze':
                    x_start_close = apply_mask_maze(x_start_close, mask_repeat)
                    x_start_far = apply_mask_maze(x_start_far, mask_repeat)
                inp_concat_neg_x0 = torch.cat([inp_repeat, inp_repeat, inp_repeat], dim=0)
                x_concat_neg_x0 = torch.cat([x_start_repeat, x_start_close, x_start_far], dim=0)
                t_concat_neg_x0 = torch.cat([t_repeat, t_repeat,t_repeat], dim=0)*0.0
                energy_neg_x0 = self.model(inp_concat_neg_x0, x_concat_neg_x0, t_concat_neg_x0, return_energy=True)

                # Compute noise contrastive energy loss
                energy_x0, energy_close_x0, energy_far_x0 = torch.chunk(energy_neg_x0, 3, 0)
                if self.args.monotonicity_landscape_loss:
                    k_energy_x0,mse_fit_x0 = linear_fit_three_points(p_close = p_close.unsqueeze(-1), E_close=energy_close_x0, p_far=p_far.unsqueeze(-1), E_far=energy_far_x0, E_0=energy_x0)
                    k_loss_x0 = torch.maximum(torch.zeros_like(k_energy_x0), self.args.k_min_monotonicity_landscape_x0-k_energy_x0)
                    loss_neg_x0 = self.args.monotonicity_landscape_k_loss_coef_x0*k_loss_x0.mean() + self.args.monotonicity_landscape_fit_loss_coef_x0 * mse_fit_x0.mean()
                else:
                    energy_stack_contrast_x0 = torch.cat([energy_close_x0, energy_far_x0], dim=-1)
                    energy_stack_close_x0 = torch.cat([energy_x0, energy_close_x0], dim=-1)
                    energy_stack_far_x0 = torch.cat([energy_x0, energy_far_x0], dim=-1)
                    
                    target_neg_contrast_x0 = torch.zeros(energy_close_x0.size(0)).to(energy_stack_contrast_x0.device)
                    target_neg_close_x0 = torch.zeros(energy_x0.size(0)).to(energy_stack_close_x0.device)
                    target_neg_far_x0 = torch.zeros(energy_x0.size(0)).to(energy_stack_far_x0.device)

                    weight_neg_contrast_x0 = generate_weights(energy_close_x0,energy_far_x0,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_x0) *((p_far-p_close)/self.args.min_gap_x0).unsqueeze(-1)
                    weight_neg_close_x0 = generate_weights(energy_x0,energy_close_x0,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_x0)*(p_close/self.args.min_gap_x0).unsqueeze(-1)
                    weight_neg_far_x0 = generate_weights(energy_x0,energy_far_x0,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_x0)*(p_far/self.args.min_gap_x0).unsqueeze(-1)
                    
                    loss_neg_contrast_x0 = F.cross_entropy(-1 * energy_stack_contrast_x0, target_neg_contrast_x0.long(), reduction='none')[:, None]
                    loss_neg_close_x0 = F.cross_entropy(-1 * energy_stack_close_x0, target_neg_close_x0.long(), reduction='none')[:, None]
                    loss_neg_far_x0 = F.cross_entropy(-1 * energy_stack_far_x0, target_neg_far_x0.long(), reduction='none')[:, None]

                    loss_neg_contrast_x0 = loss_neg_contrast_x0 * weight_neg_contrast_x0
                    loss_neg_close_x0 = loss_neg_close_x0 * weight_neg_close_x0
                    loss_neg_far_x0 = loss_neg_far_x0 * weight_neg_far_x0
                    
                    loss_neg_x0 = (loss_neg_contrast_x0 + loss_neg_close_x0 + loss_neg_far_x0).mean()
                if self.args.use_monotonicity_and_constrat_both:
                    energy_stack_contrast_x0 = torch.cat([energy_close_x0, energy_far_x0], dim=-1)
                    energy_stack_close_x0 = torch.cat([energy_x0, energy_close_x0], dim=-1)
                    energy_stack_far_x0 = torch.cat([energy_x0, energy_far_x0], dim=-1)
                    
                    target_neg_contrast_x0 = torch.zeros(energy_close_x0.size(0)).to(energy_stack_contrast_x0.device)
                    target_neg_close_x0 = torch.zeros(energy_x0.size(0)).to(energy_stack_close_x0.device)
                    target_neg_far_x0 = torch.zeros(energy_x0.size(0)).to(energy_stack_far_x0.device)

                    weight_neg_contrast_x0 = generate_weights(energy_close_x0,energy_far_x0,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_x0) *((p_far-p_close)/self.args.min_gap_x0).unsqueeze(-1)
                    weight_neg_close_x0 = generate_weights(energy_x0,energy_close_x0,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_x0)*(p_close/self.args.min_gap_x0).unsqueeze(-1)
                    weight_neg_far_x0 = generate_weights(energy_x0,energy_far_x0,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_x0)*(p_far/self.args.min_gap_x0).unsqueeze(-1)
                    
                    loss_neg_contrast_x0 = F.cross_entropy(-1 * energy_stack_contrast_x0, target_neg_contrast_x0.long(), reduction='none')[:, None]
                    loss_neg_close_x0 = F.cross_entropy(-1 * energy_stack_close_x0, target_neg_close_x0.long(), reduction='none')[:, None]
                    loss_neg_far_x0 = F.cross_entropy(-1 * energy_stack_far_x0, target_neg_far_x0.long(), reduction='none')[:, None]

                    loss_neg_contrast_x0 = loss_neg_contrast_x0 * weight_neg_contrast_x0
                    loss_neg_close_x0 = loss_neg_close_x0 * weight_neg_close_x0
                    loss_neg_far_x0 = loss_neg_far_x0 * weight_neg_far_x0
                    
                    loss_neg_x0 = self.args.coef_naive_contrast*(loss_neg_contrast_x0 + loss_neg_close_x0 + loss_neg_far_x0).mean() + loss_neg_x0

                loss_dict["loss_neg_x0"] = loss_neg_x0.item()
            if self.args.neg_contrast_coef_xt>0:
                # if self.sudoku:
                    # distance_close = np.random.uniform(0.02, self.args.max_strength_permutation_xt)
                    # distance_far = np.random.uniform(distance_close+0.01, self.args.max_strength_permutation_xt+0.01)
                    # distance_close = distance_close ** 2
                    # distance_far = distance_far ** 2
                    # max_gap = np.random.uniform(self.args.max_gap_xt, self.args.max_gap_xt)
                    # max_gap = max_gap ** 2
                    # if (distance_far - distance_close) > max_gap:
                    #     distance_far += max_gap
                    # if self.args.diverse_gap_batch:
                    #     distance_close, distance_far = generate_distances(batch_size=x_start.size(0), 
                    #                        max_strength_permutation_xt=self.args.max_strength_permutation_xt,
                    #                          max_gap=self.args.max_gap_xt, 
                    #                          min_gap=self.args.min_gap_xt, 
                    #                          device=x_start.device)
                    # else:
                    #     distance_close, distance_far = generate_distances(batch_size=1, 
                    #                        max_strength_permutation_xt=self.args.max_strength_permutation_xt,
                    #                          max_gap=self.args.max_gap_xt, 
                    #                          min_gap=self.args.min_gap_xt, 
                    #                          device=x_start.device)
                    # # import pdb; pdb.set_trace()
                    # xt_close = add_l2_noise(data_sample, distance_close.unsqueeze(-1))
                    # xt_far = add_l2_noise(data_sample, distance_far.unsqueeze(-1))
                # else:
                #     raise ValueError('Not implemented yet')
                xt_close = self.q_sample(x_start = x_start_close, t = t, noise = noise)
                xt_far = self.q_sample(x_start = x_start_far, t = t, noise = noise)
                if mask is not None and self.args.dataset != 'maze':
                    xt_close = xt_close * (1 - mask) + mask * data_sample
                    xt_far = xt_far * (1 - mask) + mask * data_sample
                elif mask is not None and self.args.dataset == 'maze':
                    xt_close = apply_mask_maze(xt_close, mask)
                    xt_far = apply_mask_maze(xt_far, mask)
                inp_concat_neg_xt = torch.cat([inp, inp, inp], dim=0)
                x_concat_neg_xt = torch.cat([data_sample, xt_close, xt_far], dim=0)
                t_concat_neg_xt = torch.cat([t, t,t], dim=0)
                energy_neg_xt = self.model(inp_concat_neg_xt, x_concat_neg_xt, t_concat_neg_xt, return_energy=True)

                # Compute noise contrastive energy loss
                energy_xt, energy_close_xt, energy_far_xt = torch.chunk(energy_neg_xt, 3, 0)
                if self.args.monotonicity_landscape_loss:
                    k_energy_xt,mse_fit_xt = linear_fit_three_points(p_close = p_close.unsqueeze(-1), E_close=energy_close_xt, p_far=p_far.unsqueeze(-1), E_far=energy_far_xt, E_0=energy_xt)
                    k_loss_xt = torch.maximum(torch.zeros_like(k_energy_xt), self.args.k_min_monotonicity_landscape_xt-k_energy_xt) #TODO,1. remove mse_loss at t 2. gamma-k_nenergy
                    loss_neg_xt = self.args.monotonicity_landscape_k_loss_coef_xt*k_loss_xt.mean() + self.args.monotonicity_landscape_fit_loss_coef_xt * mse_fit_xt.mean()
                else:
                    energy_stack_contrast_xt = torch.cat([energy_close_xt, energy_far_xt], dim=-1)
                    energy_stack_close_xt = torch.cat([energy_xt, energy_close_xt], dim=-1)
                    energy_stack_far_xt = torch.cat([energy_xt, energy_far_xt], dim=-1)
                    
                    target_neg_contrast_xt = torch.zeros(energy_close_xt.size(0)).to(energy_stack_contrast_xt.device)
                    target_neg_close_xt = torch.zeros(energy_xt.size(0)).to(energy_stack_close_xt.device)
                    target_neg_far_xt = torch.zeros(energy_xt.size(0)).to(energy_stack_far_xt.device)
                    
                    weight_neg_contrast_xt = generate_weights(energy_close_xt,energy_far_xt,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_xt) * ((distance_far-distance_close)/self.args.min_gap_xt).unsqueeze(-1)
                    weight_neg_close_xt = generate_weights(energy_xt,energy_close_xt,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_xt) * (distance_close/self.args.min_gap_xt).unsqueeze(-1)
                    weight_neg_far_xt = generate_weights(energy_xt,energy_far_xt,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_xt) * (distance_far/self.args.min_gap_xt).unsqueeze(-1)
                    
                    loss_neg_contrast_xt = F.cross_entropy(-1 * energy_stack_contrast_xt, target_neg_contrast_xt.long(), reduction='none')[:, None]
                    loss_neg_close_xt = F.cross_entropy(-1 * energy_stack_close_xt, target_neg_close_xt.long(), reduction='none')[:, None]
                    loss_neg_far_xt = F.cross_entropy(-1 * energy_stack_far_xt, target_neg_far_xt.long(), reduction='none')[:, None]

                    loss_neg_contrast_xt = loss_neg_contrast_xt * weight_neg_contrast_xt
                    loss_neg_close_xt = loss_neg_close_xt * weight_neg_close_xt
                    loss_neg_far_xt = loss_neg_far_xt * weight_neg_far_xt

                    loss_neg_xt = (loss_neg_contrast_xt + loss_neg_close_xt + loss_neg_far_xt).mean()
                if self.args.use_monotonicity_and_constrat_both:
                    energy_stack_contrast_xt = torch.cat([energy_close_xt, energy_far_xt], dim=-1)
                    energy_stack_close_xt = torch.cat([energy_xt, energy_close_xt], dim=-1)
                    energy_stack_far_xt = torch.cat([energy_xt, energy_far_xt], dim=-1)
                    
                    target_neg_contrast_xt = torch.zeros(energy_close_xt.size(0)).to(energy_stack_contrast_xt.device)
                    target_neg_close_xt = torch.zeros(energy_xt.size(0)).to(energy_stack_close_xt.device)
                    target_neg_far_xt = torch.zeros(energy_xt.size(0)).to(energy_stack_far_xt.device)
                    
                    weight_neg_contrast_xt = generate_weights(energy_close_xt,energy_far_xt,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_xt) * ((distance_far-distance_close)/self.args.min_gap_xt).unsqueeze(-1)
                    weight_neg_close_xt = generate_weights(energy_xt,energy_close_xt,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_xt) * (distance_close/self.args.min_gap_xt).unsqueeze(-1)
                    weight_neg_far_xt = generate_weights(energy_xt,energy_far_xt,min_weight_neg_contrat_x = self.args.min_weight_neg_contrat_xt) * (distance_far/self.args.min_gap_xt).unsqueeze(-1)
                    
                    loss_neg_contrast_xt = F.cross_entropy(-1 * energy_stack_contrast_xt, target_neg_contrast_xt.long(), reduction='none')[:, None]
                    loss_neg_close_xt = F.cross_entropy(-1 * energy_stack_close_xt, target_neg_close_xt.long(), reduction='none')[:, None]
                    loss_neg_far_xt = F.cross_entropy(-1 * energy_stack_far_xt, target_neg_far_xt.long(), reduction='none')[:, None]

                    loss_neg_contrast_xt = loss_neg_contrast_xt * weight_neg_contrast_xt
                    loss_neg_close_xt = loss_neg_close_xt * weight_neg_close_xt
                    loss_neg_far_xt = loss_neg_far_xt * weight_neg_far_xt

                    loss_neg_xt = self.args.coef_naive_contrast**(loss_neg_contrast_xt + loss_neg_close_xt + loss_neg_far_xt).mean() + loss_neg_xt
                loss_dict["loss_neg_xt"] = loss_neg_xt.item()
            loss = loss + self.args.neg_contrast_coef * (loss_neg_x0* self.args.neg_contrast_coef_x0 + loss_neg_xt * self.args.neg_contrast_coef_xt)
        return loss, loss_dict

    


    def forward(self, inp, target, mask, *args, **kwargs):
        b, *c = target.shape
        device = target.device
        if len(c) == 1:
            self.out_dim = c[0]
            self.out_shape = c
        else:
            self.out_dim = c[-1]
            self.out_shape = c

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(inp, target, mask, t, *args, **kwargs)

# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        validation_batch_size = None,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        data_workers = None,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        metric = 'mse',
        cond_mask = False,
        validation_dataset = None,
        extra_validation_datasets = None,
        extra_validation_every_mul = 10,
        evaluate_first = False,
        latent = False,
        autoencode_model = None,
        exp_hash_code = None,
        train_dataset_medium =None,
        validation_dataset_medium = None,
        extra_validation_datasets_medium =  None,
        train_dataset_hard =  None,
        validation_dataset_hard =  None,
        extra_validation_datasets_hard =  None,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model

        # Conditioning on mask

        self.cond_mask = cond_mask

        # Whether to do reasoning in the latent space

        self.latent = latent

        if autoencode_model is not None:
            self.autoencode_model = autoencode_model.cuda()
        
        self.exp_hash_code = exp_hash_code

        # sampling and training hyperparameters
        self.out_dim = self.model.out_dim

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.extra_validation_every_mul = extra_validation_every_mul

        self.batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size if validation_batch_size is not None else train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # Evaluation metric.
        self.metric = metric
        self.data_workers = data_workers
        self.best_loss = float('inf')
        self.best_metric = 0.0
        self.best_metric_harder_data = 0.0

        if self.data_workers is None:
            self.data_workers = cpu_count()

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = False, num_workers = self.data_workers)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        if train_dataset_medium is not None:
            dl_medium = DataLoader(train_dataset_medium, batch_size = train_batch_size, shuffle = True, pin_memory = False, num_workers = self.data_workers)
            dl_medium = self.accelerator.prepare(dl_medium)
            self.dl_medium = cycle(dl_medium)
        if train_dataset_hard is not None:
            dl_hard = DataLoader(train_dataset_hard, batch_size = train_batch_size, shuffle = True, pin_memory = False, num_workers = self.data_workers)
            dl_hard = self.accelerator.prepare(dl_hard)
            self.dl_hard = cycle(dl_hard)

        self.validation_dataset = validation_dataset

        if self.validation_dataset is not None:
            dl = DataLoader(self.validation_dataset, batch_size = validation_batch_size, shuffle=False, pin_memory=False, num_workers = self.data_workers)
            dl = self.accelerator.prepare(dl)
            self.validation_dl = dl
        else:
            self.validation_dl = None

        self.extra_validation_datasets = extra_validation_datasets

        if self.extra_validation_datasets is not None:
            self.extra_validation_dls = dict()
            for key, dataset in self.extra_validation_datasets.items():
                dl = DataLoader(dataset, batch_size = validation_batch_size, shuffle=False, pin_memory=False, num_workers = self.data_workers)
                dl = self.accelerator.prepare(dl)
                self.extra_validation_dls[key] = dl
        else:
            self.extra_validation_dls = None

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.evaluate_first = evaluate_first

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone,metric=None):
        if not self.accelerator.is_local_main_process:
            return

        # Remove prev best ckpt if 'best' is in the name
        if type(milestone)==str and 'model-best' in milestone and metric is None:
            for f in self.results_folder.iterdir():
                if 'model-best' in f.name:
                    f.unlink()

        data = {
            'step': self.step,
            'best_loss': self.best_loss,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        if type(milestone)==str and 'best_metric_val' in milestone:
            for f in self.results_folder.iterdir():
                if 'best_metric_val' in f.name:
                    f.unlink()
        if type(milestone)==str and 'best_metric_harder' in milestone:
            for f in self.results_folder.iterdir():
                if 'best_metric_harder' in f.name:
                    f.unlink()
            torch.save(data, str(self.results_folder / f'model-{milestone}-{metric:.4f}.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))


    def load(self, milestone):
        if osp.isfile(milestone):
            milestone_file = milestone
        else:
            milestone_file = sorted([f for f in self.results_folder.iterdir() if f.name.startswith('model')])[-1]
        data = torch.load(milestone_file)

        self.best_loss = data.get('best_loss', 1e9)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        args_dict = vars(self.model.args)
        with open(self.results_folder / "config.yml", "w") as f:
            yaml.dump(args_dict, f)

        if self.evaluate_first:
            milestone = self.step // self.save_and_sample_every
            self.evaluate(device, milestone)
            self.evaluate_first = False  # hack: later we will use this flag as a bypass signal to determine whether we want to run extra validation.

        end_time = time.time()
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process, dynamic_ncols = True) as pbar:
            total_loss = 0.
            loss_record_steps = 1000
            loss_sum_list = []
            loss_denoise_list = []
            loss_energy_list = []
            loss_opt_list = []
            loss_kl_list = []
            loss_entropy_list = []
            loss_neg_x0_list = []
            loss_neg_xt_list = []
            dataloader_id = 0
            if self.train_num_steps==-1:
                self.save('best-0')
            while self.step < self.train_num_steps:
                end_time = time.time()
                for _ in range(self.gradient_accumulate_every):
                    if self.model.args.num__size_training ==1:
                        data = next(self.dl)
                    elif self.model.args.num__size_training ==2:
                        if self.step%2 == 0:
                            data = next(self.dl)
                        else:
                            data = next(self.dl_medium)
                    elif self.model.args.num__size_training ==3:
                        if self.step%3 == 0:
                            data = next(self.dl)
                        elif self.step%3 == 1:
                            data = next(self.dl_medium)
                        else:
                            data = next(self.dl_hard)

                    if self.cond_mask:
                        inp, label, mask = data
                        if self.model.args.dataset == 'sudoku':
                            if self.model.args.proportion_added_entry<0:
                                p_added = np.random.uniform(0, 1)
                                inp,mask = mask_label_add_entry(label.view(-1, 9,9,9),p = p_added,mask_origin = mask.view(-1, 9,9,9))
                            else:
                                inp,mask = mask_label_add_entry(label.view(-1, 9,9,9),p = self.model.args.proportion_added_entry,mask_origin = mask.view(-1, 9,9,9))
                            inp,mask = mask_label_del_entry(label.view(-1, 9,9,9),p = self.model.args.proportion_deleted_entry,mask_origin = mask.view(-1, 9,9,9))
                            inp, label, mask = inp.float().to(device), label.float().to(device), mask.float().to(device)
                        elif self.model.args.dataset=='maze':
                            inp, label, mask = inp.float().to(device), label.float().to(device), mask.to(device)
                    elif self.latent:
                        inp, label, label_gt, mask_latent = data
                        mask_latent = mask_latent.float().to(device)
                        inp, label, label_gt = inp.float().to(device), label.float().to(device), label_gt.float().to(device)
                        mask = None
                    else:
                        inp, label = data
                        inp, label = inp.float().to(device), label.float().to(device)
                        mask = None

                    data_time = time.time() - end_time; end_time = time.time()

                    with self.accelerator.autocast():
                        loss, loss_dict = self.model(
                            inp,
                            label,
                            mask,
                            is_kl_loss=self.step % self.model.args.kl_interval == 0,
                        )
                        loss_denoise = loss_dict["loss_denoise"]
                        loss_energy = loss_dict["loss_energy"]
                        loss_opt = loss_dict["loss_opt"]
                        loss_kl = loss_dict["loss_kl"]
                        loss_entropy = loss_dict["loss_entropy"]
                        if loss_kl != 0:
                            loss_kl_show = loss_kl
                            loss_entropy_show = loss_entropy
                        else:
                            if self.model.args.entropy_coef == 0:
                                loss_kl_show = 0
                                loss_entropy_show = 0
                        loss_neg_x0 = loss_dict["loss_neg_x0"]
                        loss_neg_xt = loss_dict["loss_neg_xt"]
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                nn_time = time.time() - end_time; end_time = time.time()
                pbar.set_description(f'{self.exp_hash_code} l:{loss.item():.4f} l_mse:{loss_denoise:.4f} l_E:{loss_energy:.3f} l_kl: {loss_kl_show:.3f} l_ent:{loss_entropy_show:.3f} l_nx0:{loss_neg_x0:.3f} l_nxt:{loss_neg_xt:.4f}')
                if self.step % loss_record_steps == 0 and self.model.args.save_loss_curve:
                    loss_sum_list.append(loss.detach().cpu().item())
                    loss_denoise_list.append(loss_denoise)
                    loss_energy_list.append(loss_energy)
                    loss_opt_list.append(loss_opt)
                    loss_kl_list.append(loss_kl)
                    loss_entropy_list.append(loss_entropy_show)
                    loss_neg_x0_list.append(loss_neg_x0)
                    loss_neg_xt_list.append(loss_neg_xt)
                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    # if True:
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        if self.model.args.test_during_training:
                            self.inference_select(device=inp.device)
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        total_loss /= self.save_and_sample_every
                        if total_loss < self.best_loss:
                            self.best_loss = total_loss
                            self.save(f'best-{milestone}')
                        total_loss = 0
                        if self.latent:
                            self.evaluate(device, milestone, inp=inp, label=label_gt, mask=mask_latent)
                        else:
                            self.evaluate(device, milestone, inp=inp, label=label, mask=mask)


                pbar.update(1)
                if (self.step % loss_record_steps*10 == 0 or self.step == self.train_num_steps-1) and self.model.args.save_loss_curve:
                    np.save(self.results_folder / 'loss_sum.npy', np.array(loss_sum_list))
                    np.save(self.results_folder / 'loss_denoise.npy', np.array(loss_denoise_list))
                    np.save(self.results_folder / 'loss_energy.npy', np.array(loss_energy_list))
                    np.save(self.results_folder / 'loss_opt.npy', np.array(loss_opt_list))
                    np.save(self.results_folder / 'loss_kl.npy', np.array(loss_kl_list))
                    np.save(self.results_folder / 'loss_entropy.npy', np.array(loss_entropy_list))
                    np.save(self.results_folder / 'loss_neg_x0.npy', np.array(loss_neg_x0_list))
                    np.save(self.results_folder / 'loss_neg_xt.npy', np.array(loss_neg_xt_list))
                    plt.figure()
                    plt.plot(np.log(np.array(loss_sum_list)),label='loss_sum')
                    plt.plot(np.log(np.array(loss_denoise_list)),label='loss_denoise')
                    plt.plot(np.log(np.array(loss_energy_list)),label='loss_energy')
                    plt.plot(np.log(np.array(loss_opt_list)),label='loss_opt')
                    plt.plot(np.log(np.array(loss_entropy_list)),label='loss_entropy')
                    plt.plot(np.log(np.array(loss_neg_x0_list)),label='loss_neg_x0')
                    plt.plot(np.log(np.array(loss_neg_xt_list)),label='loss_neg_xt')
                    plt.legend()
                    plt.xlabel(f'steps / {loss_record_steps}')
                    plt.ylabel('log(loss)')
                    plt.title('Loss Curve')
                    plt.savefig(self.results_folder / 'loss_curve.png')
        accelerator.print('training complete, results saved to', self.results_folder)
    def inference_select(self,device):
        start_time = time.time()
        self.ema.ema_model.eval()
        for key, extra_dl in self.extra_validation_dls.items():
            break 
        for i, data in enumerate(tqdm(self.validation_dl, total=len(self.validation_dl), desc=f'running on the validation dataset ')):
            break
        if self.cond_mask and self.model.args.dataset != 'maze':
            inp, label, mask = map(lambda x: x.float().to(device), data)
            inp,mask = mask_label_add_entry(label.view(-1, 9,9,9),p = self.model.args.proportion_added_entry,mask_origin = mask.view(-1, 9,9,9))
            inp,mask = mask_label_del_entry(label.view(-1, 9,9,9),p = self.model.args.proportion_deleted_entry,mask_origin = mask.view(-1, 9,9,9))
        elif self.cond_mask and self.model.args.dataset == 'maze':
            inp, label, mask =data
            inp, label, mask = inp.float().to(device), label.float().to(device), mask.to(device)
        elif self.latent:
            inp, label, label_gt, mask = map(lambda x: x.float().to(device), data)
        else:
            inp, label = map(lambda x: x.float().to(device), data)
            mask = None
        samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0)) # inp,label,mask are all [B,729] for Sudoku
        if self.metric == 'sudoku':
            samples_processed = process_tensor_sudoku(samples)
            energy_sample = self.ema.ema_model.model(inp, samples_processed, torch.zeros(samples.shape[0]).to(device), return_energy=True).squeeze(-1)
            energy_gd = self.ema.ema_model.model(inp, label, torch.zeros(label.shape[0]).to(device), return_energy=True).squeeze(-1)
            summary = sudoku_accuracy(samples, label, mask,energy_sample=energy_sample,energy_gd=energy_gd)
            metric = summary['accuracy']
        elif self.metric == 'maze':
            samples_processed = normalize_last_dim(samples).float()
            energy_sample = self.ema.ema_model.model(inp, samples_processed, torch.zeros(samples.shape[0]).to(device), return_energy=True).squeeze(-1)
            energy_gd = self.ema.ema_model.model(inp, label, torch.zeros(label.shape[0]).to(device), return_energy=True).squeeze(-1)
            summary = maze_accuracy(maze_cond = inp,maze_solution=samples,mask = mask,label = label,energy_sample=energy_sample,energy_gd=energy_gd)
            metric = summary['accuracy']
        else:
            raise ValueError('Not implemented yet')
        energy_permutation_list = []
        for k in range(1,82):
            p_permutation = float(k/81)
            x0_permutation = random_permute_last_dim(label,p = p_permutation)
            energy_permutation = self.ema.ema_model.model(inp, x0_permutation, torch.zeros(x0_permutation.shape[0]).to(device), return_energy=True).squeeze(-1).detach().cpu().numpy()
            energy_permutation_list.append(energy_permutation)
        energy_permutation_tensor = torch.tensor(energy_permutation_list).permute(1,0)
        monotonicity_energy = calculate_monotonicity(energy_permutation_tensor) # [B]
        num_row = 3
        num_col = 3
        fig, axs = plt.subplots(num_row, num_col, figsize=(15, 15))
        fig.suptitle('Monotonicity of energy landscape')  
        for i in range(num_row):
            for j in range(num_col):
                index = i * num_col + j 
                if index < num_row * num_col: 
                    ax = axs[i, j] 
                    ax.plot(torch.log(energy_permutation_tensor[index, :]).cpu().numpy())  
                    ax.set_xlabel('strength of permutation') 
                    ax.set_ylabel('log(energy) of permutated x0')  
                    ax.set_title(f'Example {index + 1}')  
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
        plt.savefig(self.results_folder / f'val_energy_landscape_x0_permutation-{self.step}.png')
        plot_maze(maze_cond=inp,maze_solution=samples, num_plot=16, file_path=self.results_folder / f'val_maze_solved-{self.step}.png')
        if metric > self.best_metric:
            self.best_metric = metric
            self.save('best_metric_val',metric)
        rows = [[k, v] for k, v in summary.items()]
        # save rows to a txt in self.results_folder
        with open(str(self.results_folder) +f'/val_best_metric_results.txt', 'a') as f:
            f.write(f'#####################\n')
            f.write(f'steps: {self.step}\n')
            for row in rows:
                f.write(f'{row[0]}: {row[1]}\n')
            f.write(f'monotonicity_energy: {monotonicity_energy.mean()}\n')
            f.write(f'\n')
        # test on harder data           
        for i, data in enumerate(tqdm(extra_dl, total=len(extra_dl), desc=f'running on the validation dataset (ID: {key})')):
            break
        if self.cond_mask and self.model.args.dataset != 'maze':
            inp, label, mask = map(lambda x: x.float().to(device), data)
            inp,mask = mask_label_add_entry(label.view(-1, 9,9,9),p = self.model.args.proportion_added_entry,mask_origin = mask.view(-1, 9,9,9))
            inp,mask = mask_label_del_entry(label.view(-1, 9,9,9),p = self.model.args.proportion_deleted_entry,mask_origin = mask.view(-1, 9,9,9))
        elif self.cond_mask and self.model.args.dataset == 'maze':
            inp, label, mask =data
            inp, label, mask = inp.float().to(device), label.float().to(device), mask.to(device)
        elif self.latent:
            inp, label, label_gt, mask = map(lambda x: x.float().to(device), data)
        else:
            inp, label = map(lambda x: x.float().to(device), data)
            mask = None
        samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0)) # inp,label,mask are all [B,729] for Sudoku
        energy_permutation_list = []
        for k in range(1,82):
            p_permutation = float(k/81)
            x0_permutation = random_permute_last_dim(label,p = p_permutation)
            energy_permutation = self.ema.ema_model.model(inp, x0_permutation, torch.zeros(x0_permutation.shape[0]).to(device), return_energy=True).squeeze(-1).detach().cpu().numpy()
            energy_permutation_list.append(energy_permutation)
        energy_permutation_tensor = torch.tensor(energy_permutation_list).permute(1,0)
        monotonicity_energy = calculate_monotonicity(energy_permutation_tensor) # [B]
        fig, axs = plt.subplots(num_row, num_col, figsize=(15, 15))
        fig.suptitle('Monotonicity of energy landscape')  
        for i in range(num_row):
            for j in range(num_col):
                index = i * num_col + j 
                if index < num_row * num_col: 
                    ax = axs[i, j] 
                    ax.plot(torch.log(energy_permutation_tensor[index, :]).cpu().numpy())  
                    ax.set_xlabel('strength of permutation') 
                    ax.set_ylabel('log(energy) of permutated x0')  
                    ax.set_title(f'Example {index + 1}')  
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
        plt.savefig(self.results_folder / f'harder_energy_landscape_x0_permutation-{self.step}.png')
        plot_maze(maze_cond=inp,maze_solution=samples, num_plot=16, file_path=self.results_folder / f'harder_maze_solved-{self.step}.png')
        if self.metric == 'sudoku':
            samples_processed = process_tensor_sudoku(samples)
            energy_sample = self.ema.ema_model.model(inp, samples_processed, torch.zeros(samples.shape[0]).to(device), return_energy=True).squeeze(-1)
            energy_gd = self.ema.ema_model.model(inp, label, torch.zeros(label.shape[0]).to(device), return_energy=True).squeeze(-1)
            summary = sudoku_accuracy(samples, label, mask,energy_sample=energy_sample,energy_gd=energy_gd)
            metric = summary['accuracy']
        elif self.metric == 'maze':
            # samples_processed = normalize_last_dim(samples).float() # TODO
            samples_processed = samples
            energy_sample = self.ema.ema_model.model(inp, samples_processed, torch.zeros(samples.shape[0]).to(device), return_energy=True).squeeze(-1)
            energy_gd = self.ema.ema_model.model(inp, label, torch.zeros(label.shape[0]).to(device), return_energy=True).squeeze(-1)
            summary = maze_accuracy(maze_cond = inp,maze_solution=samples,mask = mask,label = label,energy_sample=energy_sample,energy_gd=energy_gd)
            metric = summary['accuracy']
        else:
            raise ValueError('Not implemented yet')
        if metric > self.best_metric_harder_data:
            self.best_metric_harder_data = metric
            self.save('best_metric_harder',metric)
        rows = [[k, v] for k, v in summary.items()]
        # save rows to a txt in self.results_folder
        with open(str(self.results_folder) +f'/harder_best_metric_results.txt', 'a') as f:
            f.write(f'#####################\n')
            f.write(f'steps: {self.step}\n')
            for row in rows:
                f.write(f'{row[0]}: {row[1]}\n')
            f.write(f'monotonicity_energy: {monotonicity_energy.mean()}\n')
            f.write(f'\n')
        print(f'Inference time: {time.time()-start_time}')
        return
    def evaluate(self, device, milestone, inp=None, label=None, mask=None):
        print('Running Evaluation...')
        self.ema.ema_model.eval()

        if inp is not None and label is not None:
            with torch.no_grad():
                # batches = num_to_groups(self.num_samples, self.batch_size)

                if self.latent:
                    all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0)), range(1)))
                else:
                    all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0)), range(1)))
                    # all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0), return_traj=True), range(1)))
                # all_samples_list = list(map(lambda n: self.model.sample(inp, label, mask, batch_size=inp.size(0)), range(1)))
                # all_samples_list = [self.model.sample(inp, batch_size=inp.size(0))]

                all_samples = torch.cat(all_samples_list, dim = 0)

                print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (Train)')
                if self.metric == 'mse':
                    all_samples = torch.cat(all_samples_list, dim = 0)
                    mse_error = (all_samples - label).pow(2).mean()
                    rows = [('mse_error', mse_error)]
                    print(tabulate(rows))
                elif self.metric == 'bce':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                elif self.metric == 'sudoku':
                    assert len(all_samples_list) == 1
                    summary = sudoku_accuracy(all_samples_list[0], label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                elif self.metric == 'maze':
                    # assert len(all_samples_list) == 1
                    # summary = maze_accuracy(all_samples_list[0], label, mask)
                    # rows = [[k, v] for k, v in summary.items()]
                    # print(tabulate(rows))
                    print('Not implemented yet')
                elif self.metric == 'sort':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    summary.update(sort_accuracy(all_samples_list[0], label, mask))
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'sort-2':
                    assert len(all_samples_list) == 1
                    summary = sort_accuracy_2(all_samples_list[0], label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'shortest-path-1d':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    summary.update(shortest_path_1d_accuracy(all_samples_list[0], label, mask, inp))
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'sudoku_latent':
                    sample = all_samples_list[0].view(-1, 9, 9, 3).permute(0, 3, 1, 2).contiguous() * 4
                    prediction = self.autoencode_model.decode(sample)
                    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 729)

                    assert len(all_samples_list) == 1
                    summary = sudoku_accuracy(prediction, label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                else:
                    raise NotImplementedError()

        if self.validation_dl is not None:
            self._run_validation(self.validation_dl, device, milestone, prefix = 'Validation')

        if (self.step % (self.save_and_sample_every * self.extra_validation_every_mul) == 0 and self.extra_validation_dls is not None) or self.evaluate_first:
            for key, extra_dl in self.extra_validation_dls.items():
                self._run_validation(extra_dl, device, milestone, prefix = key)

    def _run_validation(self, dl, device, milestone, prefix='Validation'):
        meters = collections.defaultdict(AverageMeter)
        with torch.no_grad():
            for i, data in enumerate(tqdm(dl, total=len(dl), desc=f'running on the validation dataset (ID: {prefix})')):
                if self.cond_mask and self.model.args.dataset != 'maze':
                    inp, label, mask = map(lambda x: x.float().to(device), data)
                    inp,mask = mask_label_add_entry(label.view(-1, 9,9,9),p = self.model.args.proportion_added_entry,mask_origin = mask.view(-1, 9,9,9))
                    inp,mask = mask_label_del_entry(label.view(-1, 9,9,9),p = self.model.args.proportion_deleted_entry,mask_origin = mask.view(-1, 9,9,9))
                elif self.cond_mask and self.model.args.dataset == 'maze':
                    inp, label, mask =data
                    inp, label, mask = inp.float().to(device), label.float().to(device), mask.to(device)
                elif self.latent:
                    inp, label, label_gt, mask = map(lambda x: x.float().to(device), data)
                else:
                    inp, label = map(lambda x: x.float().to(device), data)
                    mask = None

                if self.latent:
                    # Masking doesn't make sense in the latent space
                    # samples = self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0))
                    samples = self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0))
                else:
                    # samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))
                    # samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))
                    samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))
                # np.savez("sudoku.npz", inp=inp.detach().cpu().numpy(), label=label.detach().cpu().numpy(), mask=mask.detach().cpu().numpy(), samples=samples.detach().cpu().numpy())
                # import pdb
                # pdb.set_trace()
                # print("here")
                if self.metric == 'sudoku':
                    # samples_traj = samples
                    samples_processed = process_tensor_sudoku(samples)
                    energy_sample = self.ema.ema_model.model(inp, samples_processed, torch.zeros(samples.shape[0]).to(device), return_energy=True).squeeze(-1)
                    energy_gd = self.ema.ema_model.model(inp, label, torch.zeros(label.shape[0]).to(device), return_energy=True).squeeze(-1)
                    summary = sudoku_accuracy(samples, label, mask, energy_sample=energy_sample,energy_gd=energy_gd)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 2:
                        break
                elif self.metric == 'maze':
                    samples_processed = normalize_last_dim(samples).float()
                    energy_sample = self.ema.ema_model.model(inp, samples_processed, torch.zeros(samples.shape[0]).to(device), return_energy=True).squeeze(-1)
                    energy_gd = self.ema.ema_model.model(inp, label, torch.zeros(label.shape[0]).to(device), return_energy=True).squeeze(-1)
                    summary = maze_accuracy(maze_cond=inp,maze_solution=samples,mask=mask,label=label,energy_sample=energy_sample,energy_gd=energy_gd)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 2:
                        break
                elif self.metric == 'sudoku_latent':
                    sample = samples.view(-1, 9, 9, 3).permute(0, 3, 1, 2).contiguous() * 4
                    prediction = self.autoencode_model.decode(sample)
                    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 729)
                    summary = sudoku_accuracy(prediction, label_gt, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                elif self.metric == 'sort':
                    summary = binary_classification_accuracy_4(samples, label)
                    summary.update(sort_accuracy(samples, label, mask))
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'sort-2':
                    summary = sort_accuracy_2(samples, label, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'shortest-path-1d':
                    summary = binary_classification_accuracy_4(samples, label)
                    summary.update(shortest_path_1d_accuracy(samples, label, mask, inp))
                    # summary.update(shortest_path_1d_accuracy_closed_loop(samples, label, mask, inp, self.ema.ema_model.sample))
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'mse':
                    # all_samples = torch.cat(all_samples_list, dim = 0)
                    mse_error = (samples - label).pow(2).mean()
                    meters['mse'].update(mse_error, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'bce':
                    summary = binary_classification_accuracy_4(samples, label)
                    for k, v in summary.items():
                        meters[k].update(v, n=samples.shape[0])
                    if i > 20:
                        break
                else:
                    raise NotImplementedError()

            rows = [[k, v.avg] for k, v in meters.items()]
            print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (ID: {prefix})')
            print(tabulate(rows))

    def inference(self, dl, seed, device, milestone=-1, prefix='inference'):
        # if self.metric == 'shortest-path-1d':
        #     for key, extra_dl in self.extra_validation_dls.items():
        #         dl = extra_dl
        #         break
        set_seed(seed)
        milestone = os.path.basename(self.model.args.ckpt)
        # results_path_inference = str(self.results_folder) + f'/{milestone}-{self.model.args.inference_method}-innerloop_opt_steps-{self.model.args.innerloop_opt_steps}_{self.model.args.task_difficulty}_{self.model.args.J_type}_K-{self.model.args.K}_steps_rollout-{self.model.args.steps_rollout}_{self.model.args.noise_type}_mcts_noise_scale-{self.model.args.mcts_noise_scale}_exploration_weight-{self.model.args.exploration_weight}_b{self.model.args.batch_size}n{self.model.args.num_batch}_{self.exp_hash_code}'
        results_path_inference = str(self.results_folder) + f'/{milestone}-{self.model.args.inference_method}-innerloop_opt_steps-{self.model.args.innerloop_opt_steps}_{self.model.args.task_difficulty}_{self.model.args.J_type}_K-{self.model.args.K}_steps_rollout-{self.model.args.steps_rollout}_{self.model.args.noise_type}_mcts_noise_scale-{self.model.args.mcts_noise_scale}_exploration_weight-{self.model.args.exploration_weight}_b{self.model.args.batch_size}n{self.model.args.num_batch}_{self.exp_hash_code}_ddim_sanple_step-{self.model.args.sampling_timesteps}'
        if self.model.args.inference_method == 'mixed_inference':
            results_path_inference = str(self.results_folder) + f'/{milestone}-{self.model.args.inference_method}_mcts_start_step-{self.model.args.mcts_start_step}-innerloop_opt_steps-{self.model.args.innerloop_opt_steps}-{self.model.args.task_difficulty}_{self.model.args.J_type}_K-{self.model.args.K}_steps_rollout-{self.model.args.steps_rollout}_{self.model.args.noise_type}_mcts_noise_scale-{self.model.args.mcts_noise_scale}_exploration_weight-{self.model.args.exploration_weight}_b{self.model.args.batch_size}n{self.model.args.num_batch}_{self.exp_hash_code}'
        if not os.path.exists(results_path_inference):
            os.makedirs(results_path_inference)
        meters = collections.defaultdict(AverageMeter)
        self.model.eval()
        sample_list = []
        inp_list = []
        mask_list = []
        metrics_list = []
        accuracy_list = []
        with torch.no_grad():
            batch_count = 0
            for i, data in enumerate(tqdm(dl, total=len(dl), desc=f'{self.exp_hash_code} running on the validation dataset (ID: {prefix})')):
                batch_count += 1
                # if batch_count < 30:
                # if batch_count not in [31,  2, 49, 12,  8, 60, 61, 19, 54, 46]:
                #     continue
                # if batch_count - 100  > self.model.args.num_batch:
                #     break
                if batch_count > self.model.args.num_batch:
                    break
                # self.ema.ema_model.args.XXX = 1.0 - float((batch_count*5-5)/81)
                if self.ema.ema_model.args.test_condition_generalization:
                    self.ema.ema_model.args.proportion_deleted_entry = max(1.0 - float((batch_count/6)),0.0)
                    self.ema.ema_model.args.proportion_added_entry = max(0.0,float((batch_count-6)/10))
                    print("proportion_added_entry,",self.ema.ema_model.args.proportion_added_entry)
                    print("proportion_deleted_entry",self.ema.ema_model.args.proportion_deleted_entry)
                if self.cond_mask and self.model.args.dataset != 'maze':
                    inp, label, mask = map(lambda x: x.float().to(device), data)
                    self.ema.ema_model.data_label = label
                    inp,mask = mask_label_add_entry(label.view(-1, 9,9,9),p = self.ema.ema_model.args.proportion_added_entry,mask_origin = mask.view(-1, 9,9,9))
                    inp,mask = mask_label_del_entry(label.view(-1, 9,9,9),p = self.ema.ema_model.args.proportion_deleted_entry,mask_origin = mask.view(-1, 9,9,9))
                elif self.cond_mask and self.model.args.dataset == 'maze':
                    inp, label, mask =data
                    self.ema.ema_model.data_label = label
                    inp, label, mask = inp.float().to(device), label.float().to(device), mask.to(device)
                elif self.latent:
                    inp, label, label_gt, mask = map(lambda x: x.float().to(device), data)
                else:
                    inp, label = map(lambda x: x.float().to(device), data)
                    mask = None
                if self.latent:
                    # Masking doesn't make sense in the latent space
                    # samples = self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0))
                    samples = self.model.sample(inp, label, None, batch_size=inp.size(0))
                else:
                    # samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))
                    # samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))
                    # if self.model.args.mcts_type == 'discrete' and self.model.args.inference_method == 'mcts':
                    if False: #TODO
                    # if inp.shape[0] !=1:
                        samples = parallel_sample(self.ema.ema_model, inp, label, mask, chunk_size=1, num_workers=inp.shape[0])
                        samples = samples.to(inp.device)
                    else:
                        if self.model.args.plt_energy_landscape_permutation_noised:
                            energy_permutation_list = []
                            for k in range(1,82):
                                p_permutation = float(k/81)
                                x0_permutation = random_permute_last_dim(label,p = p_permutation)
                                energy_permutation = self.ema.ema_model.model(inp, x0_permutation, torch.zeros(x0_permutation.shape[0]).to(device), return_energy=True).squeeze(-1).detach().cpu().numpy()
                                energy_permutation_list.append(energy_permutation)
                            np.save(results_path_inference +f'/{prefix}_energy_permutation_list_case-{i}.npy',np.array(energy_permutation_list))
                            print("plot energy landscape")   
                        
                        samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0)) # inp,label,mask are all [B,729] for Sudoku
                sample_list.append(samples)
                inp_list.append(inp)
                mask_list.append(mask)
                # import pdb; pdb.set_trace()
                # for i in range(inp.shape[0]):
                #     plot_energy_vs_distance(inp[[i]],label[[i]], self.ema.ema_model.model, max_radius=30.0, n_levels=30, n_samples_per_level=1000,results_path = self.results_folder / f'{prefix}_energy_vs_distance_case-{i}.png',mask=mask[[i]],label_gd=label[[i]])
                # np.savez("sudoku.npz", inp=inp.detach().cpu().numpy(), label=label.detach().cpu().numpy(), mask=mask.detach().cpu().numpy(), samples=samples.detach().cpu().numpy())
                if self.metric == 'sudoku':
                    # samples_traj = samples
                    samples_processed = process_tensor_sudoku(samples)
                    energy_sample = self.ema.ema_model.model(inp, samples_processed, torch.zeros(samples.shape[0]).to(device), return_energy=True).squeeze(-1)
                    energy_gd = self.ema.ema_model.model(inp, label, torch.zeros(label.shape[0]).to(device), return_energy=True).squeeze(-1)
                    summary = sudoku_accuracy(samples, label, mask,energy_sample=energy_sample,energy_gd=energy_gd)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                elif self.metric == 'maze':
                    # samples_processed = normalize_last_dim(samples).float() # TODO
                    samples_processed = samples
                    energy_sample = self.ema.ema_model.model(inp, samples_processed, torch.zeros(samples.shape[0]).to(device), return_energy=True).squeeze(-1)
                    energy_gd = self.ema.ema_model.model(inp, label, torch.zeros(label.shape[0]).to(device), return_energy=True).squeeze(-1)
                    summary = maze_accuracy(maze_cond=inp,maze_solution=samples,mask=mask,label=label,energy_gd=energy_gd,energy_sample=energy_sample)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                elif self.metric == 'sudoku_latent':
                    sample = samples.view(-1, 9, 9, 3).permute(0, 3, 1, 2).contiguous() * 4
                    prediction = self.autoencode_model.decode(sample)
                    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 729)
                    summary = sudoku_accuracy(prediction, label_gt, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                elif self.metric == 'sort':
                    summary = binary_classification_accuracy_4(samples, label)
                    summary.update(sort_accuracy(samples, label, mask))
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'sort-2':
                    summary = sort_accuracy_2(samples, label, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'shortest-path-1d':
                    summary = binary_classification_accuracy_4(samples, label)
                    summary.update(shortest_path_1d_accuracy(samples, label, mask, inp))
                    # summary.update(shortest_path_1d_accuracy_closed_loop(samples, label, mask, inp, self.ema.ema_model.sample))
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    # if i > 20:
                    #     break
                elif self.metric == 'mse':
                    # all_samples = torch.cat(all_samples_list, dim = 0)
                    mse_error = (samples - label).pow(2).mean()
                    meters['mse'].update(mse_error, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'bce':
                    summary = binary_classification_accuracy_4(samples, label)
                    for k, v in summary.items():
                        meters[k].update(v, n=samples.shape[0])
                    if i > 20:
                        break
                else:
                    raise NotImplementedError()
                metrics_list.append(summary)
                accuracy_list.append(summary['accuracy'])
                # rows = [[k, v.avg] for k, v in meters.items()]
                rows = [[k, v] for k, v in summary.items()]
                # save rows to a txt in self.results_folder
                with open(results_path_inference +f'/{prefix}_result_seed{seed}.txt', 'a') as f:
                    f.write(f'\n')
                    f.write(f'batch-{i}\n')
                    for row in rows:
                        f.write(f'{row[0]}: {row[1]}\n')
                print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (ID: {prefix})')
                print("Evaluated on task  task_difficulty: ",self.model.args.task_difficulty)
                print(tabulate(rows))
                if i <16 and self.model.args.dataset == 'maze':
                    samples_plot = normalize_last_dim(samples)
                    plot_maze(maze_cond=inp.cpu(),maze_solution=samples_plot.cpu(), num_plot=1, file_path=results_path_inference+ f'/harder_maze_solved-{i}.png')
                # plot_maze(maze_cond=inp.cpu(),maze_solution=label.cpu(), num_plot=1, file_path=results_path_inference+ f'/harder_maze_solved-{i}.png')
                
                # save inp,samples,label,mask to a npz file for each batch
            mean_std = np.mean(self.ema.ema_model.std_list)
            mean_consistency = np.mean(self.ema.ema_model.consistency_list)
            mean_path_length_std = np.mean(self.ema.ema_model.path_length_std_list)
            # save mean_entropy and mean_consistency to a npy file
            np.save(results_path_inference +f'/{prefix}_mean_std_seed{seed}.npy',np.array(self.ema.ema_model.std_list))
            np.save(results_path_inference +f'/{prefix}_mean_consistency_seed{seed}.npy',np.array(self.ema.ema_model.consistency_list))
            np.save(results_path_inference +f'/{prefix}_mean_path_length_std_seed{seed}.npy',np.array(self.ema.ema_model.path_length_std_list))
            with open(results_path_inference +f'/{prefix}_result_seed{seed}.txt', 'a') as f:
                f.write(f'#####################\n')
                # f.write(f'mean_std: {mean_std}\n')
                # f.write(f'mean_consistency: {mean_consistency}\n')
                # f.write(f'mean_path_length_std: {mean_path_length_std}\n')
                f.write(f"mean accuracy of {self.model.args.num_batch} batches with batch_size {label.shape[0]} is {np.mean(accuracy_list)}\n")
                metrics={}
                for key in metrics_list[0].keys():
                    metrics[key] =[]
                for i in range(len(metrics_list)):
                    for key in metrics_list[i].keys():
                        metrics[key].append(metrics_list[i][key])
                for key in metrics.keys():
                    f.write(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}\n') 

            # Save metrics' mean and standard deviation to an Excel file with two rows
            metrics_summary = {}
            for key in metrics.keys():
                mean = np.mean(metrics[key])
                std = np.std(metrics[key])
                metrics_summary[key] = f"{mean:.4f} ± {std:.4f}"

            # Create a DataFrame with two rows: one for metric names, one for values
            metrics_df = pd.DataFrame(
                [list(metrics_summary.keys()), list(metrics_summary.values())]
            )

            # Save to Excel file
            excel_path = results_path_inference + f'/{prefix}_metrics_summary_seed{seed}.xlsx'
            metrics_df.to_excel(excel_path, index=False, header=False)
                
            # if 'accuracy' in summary.keys():
            #     print(f"mean accuracy of {self.model.args.num_batch} batches with batch_size {label.shape[0]} is {np.mean(metrics_list[:]['accuracy'])}")
            print(f"mean accuracy of {self.model.args.num_batch} batches with batch_size {label.shape[0]} is {np.mean(accuracy_list)}")
            # accuracy_np = np.array(accuracy_list)
            # np.save(str(self.results_folder)+f'/{prefix}_accuracy'+self.model.args.inference_method+'.npy', accuracy_np)
            # print(f'accuracy_np saved to {str(self.results_folder)+f"/{prefix}_accuracy"+self.model.args.inference_method+".npy"}')
            samples = torch.cat(sample_list, dim=0).cpu()
            inps = torch.cat(inp_list, dim=0).cpu()
            masks = torch.cat(mask_list, dim=0)
            num_plot = min(16, samples.size(0))
            if self.model.args.dataset == 'maze':
                samples_plot = normalize_last_dim(samples)
                plot_maze(maze_cond=inps,maze_solution=samples_plot, num_plot=num_plot, file_path=results_path_inference+ f'/harder_maze_solved.png')
            torch.save(samples.cpu(), results_path_inference + f'/samples_seed{seed}.pt')
            torch.save(metrics_list, results_path_inference + f'/metrics_list_seed{seed}.pt')
            # save self.model.args key and value
            args_dict = vars(self.model.args)
            if seed == self.model.args.seed: # only save args once
                with open(results_path_inference +f'/{prefix}_args.txt', 'a') as f:
                    for key, value in args_dict.items():
                        f.write(f'{key}: {value}\n')
            print(f'inferece results saved to {results_path_inference}')


as_float = lambda x: float(x.item())


@torch.no_grad()
def binary_classification_accuracy(pred: torch.Tensor, label: torch.Tensor, name: str = '', saturation: bool = True) -> dict[str, float]:
    r"""Compute the accuracy of binary classification.

    Args:
        pred: the prediction, of the same shape as ``label``.
        label: the label, of the same shape as ``pred``.
        name: the name of this monitor.
        saturation: whether to check the saturation of the prediction. Saturation
            is defined as :math:`1 - \min(pred, 1 - pred)`

    Returns:
        a dict of monitor values.
    """
    if name != '':
        name = '/' + name
    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    acc = label.float().eq((pred > 0.5).float())
    if saturation:
        sat = 1 - (pred - (pred > 0.5).float()).abs()
        return {
            prefix: as_float(acc.float().mean()),
            prefix + '/saturation/mean': as_float(sat.mean()),
            prefix + '/saturation/min': as_float(sat.min())
        }
    return {prefix: as_float(acc.float().mean())}


@torch.no_grad()
def binary_classification_accuracy_4(pred: torch.Tensor, label: torch.Tensor, name: str = '') -> dict[str, float]:
    if name != '':
        name = '/' + name

    # table = list()
    # table.append(('pred', pred[0].squeeze()))
    # table.append(('label', label[0].squeeze()))
    # print(tabulate(table))
    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    numel = pred.numel()

    gt_0_pred_0 = ((label < 0.0) & (pred < 0.0)).sum() / numel
    gt_0_pred_1 = ((label < 0.0) & (pred >= 0.0)).sum() / numel
    gt_1_pred_0 = ((label > 0.0) & (pred < 0.0)).sum() / numel
    gt_1_pred_1 = ((label > 0.0) & (pred >= 0.0)).sum() / numel

    accuracy = gt_0_pred_0 + gt_1_pred_1
    balanced_accuracy = sum([
        gt_0_pred_0 / ((label < 0.0).float().sum() / numel),
        gt_1_pred_1 / ((label >= 0.0).float().sum() / numel),
    ]) / 2

    return {
        prefix + '/gt_0_pred_0': as_float(gt_0_pred_0),
        prefix + '/gt_0_pred_1': as_float(gt_0_pred_1),
        prefix + '/gt_1_pred_0': as_float(gt_1_pred_0),
        prefix + '/gt_1_pred_1': as_float(gt_1_pred_1),
        prefix + '/accuracy': as_float(accuracy),
        prefix + '/balance_accuracy': as_float(balanced_accuracy),
    }


@torch.no_grad()
def sudoku_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = '', energy_sample=None, energy_gd= None) -> dict[str, float]:
    if name != '':
        name = '/' + name

    pred = pred.view(-1, 9, 9, 9).argmax(dim=-1)
    label = label.view(-1, 9, 9, 9).argmax(dim=-1)

    correct = (pred == label).float()
    mask = mask.view(-1, 9, 9, 9)[:, :, :, 0]
    mask_inverse = 1 - mask

    consistency,consistency_batch= sudoku_consistency(pred)
    if mask_inverse.sum()<1.0:
        accuracy = torch.ones(1)
    else:
        accuracy = (correct * mask_inverse).sum() / mask_inverse.sum()
    try:
        # import pdb; pdb.set_trace()
        energy_consitency = (
        torch.logical_and(energy_sample <= energy_gd, consistency_batch).float() +
        torch.logical_and(energy_sample > energy_gd, torch.logical_not(consistency_batch)).float()
        )
        energy_consitency = energy_consitency.mean()
        return {
            'accuracy': as_float(accuracy),
            'consistency': as_float(consistency),
            'board_accuracy': as_float(sudoku_score(pred)),
            'energy_consistency': as_float(energy_consitency),
            'mean of sample energy': as_float(energy_sample.mean()),
            'mean of gd energy': as_float(energy_gd.mean()),
    }
    except:
        return {
            'accuracy': as_float(accuracy),
            'consistency': as_float(consistency),
            'board_accuracy': as_float(sudoku_score(pred)),
        }


def sudoku_consistency(pred: torch.Tensor) -> bool:
    pred_onehot = F.one_hot(pred, num_classes=9)

    all_row_correct = (pred_onehot.sum(dim=1) == 1).all(dim=-1).all(dim=-1)
    all_col_correct = (pred_onehot.sum(dim=2) == 1).all(dim=-1).all(dim=-1)

    blocked = pred_onehot.view(-1, 3, 3, 3, 3, 9)
    all_block_correct = (blocked.sum(dim=(2, 4)) == 1).all(dim=-1).all(dim=-1).all(dim=-1)

    return (all_row_correct & all_col_correct & all_block_correct).float().mean(), (all_row_correct & all_col_correct & all_block_correct)


def sudoku_score(pred: torch.Tensor, reduction='mean') -> bool:
    valid_mask = torch.ones_like(pred)

    pred_sum_axis_1 = pred.sum(dim=1, keepdim=True)
    pred_sum_axis_2 = pred.sum(dim=2, keepdim=True)

    # Use the sum criteria from the SAT-Net paper
    axis_1_mask = (pred_sum_axis_1 == 36)
    axis_2_mask = (pred_sum_axis_2 == 36)

    valid_mask = valid_mask * axis_1_mask.float() * axis_2_mask.float()

    valid_mask = valid_mask.view(-1, 3, 3, 3, 3)
    grid_mask = pred.view(-1, 3, 3, 3, 3).sum(dim=(2, 4), keepdim=True) == 36

    valid_mask = valid_mask * grid_mask.float()
    if reduction == 'mean':
        return valid_mask.mean()
    elif reduction == 'none':
        return valid_mask.view(-1, 81).mean(-1)


def sort_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name

    array = (label[:, 0, ..., 2] * 0.5 + 0.5).sum(dim=-1).cpu()
    pred = pred.cpu()
    for t in range(pred.shape[1]):
        pred_xy = pred[:, t, ..., -1].reshape(pred.shape[0], -1).argmax(dim=-1)
        pred_x = torch.div(pred_xy, pred.shape[2], rounding_mode='floor')
        pred_y = pred_xy % pred.shape[2]
        # swap x and y
        next_array = array.clone()
        next_array.scatter_(1, pred_y.unsqueeze(1), array.gather(1, pred_x.unsqueeze(1)))
        next_array.scatter_(1, pred_x.unsqueeze(1), array.gather(1, pred_y.unsqueeze(1)))
        array = next_array

    ground_truth = torch.arange(pred.shape[2] - 1, -1, -1, device=array.device).unsqueeze(0).repeat(pred.shape[0], 1)
    elem_close = (array - ground_truth).abs() < 0.1
    element_correct = elem_close.float().mean()
    array_correct = elem_close.all(dim=-1).float().mean()
    return {
        'element_correct': as_float(element_correct),
        'array_correct': as_float(array_correct),
    }


def sort_accuracy_2(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name

    array = label[:, 0, :, 0].clone().cpu()  # B x N
    pred = pred.cpu()
    for t in range(pred.shape[1]):
        pred_x = pred[:, t, :, 1].argmax(dim=-1)  # B x N
        pred_y = pred[:, t, :, 2].argmax(dim=-1)  # B x N
        # swap x and y
        next_array = array.clone()
        next_array.scatter_(1, pred_y.unsqueeze(1), array.gather(1, pred_x.unsqueeze(1)))
        next_array.scatter_(1, pred_x.unsqueeze(1), array.gather(1, pred_y.unsqueeze(1)))
        array = next_array

    # stupid_impl_array = label[:, 0, :, 0].clone()  # B x N
    # for b in range(pred.shape[0]):
    #     for t in range(pred.shape[1]):
    #         pred_x = pred[b, t, :, 1].argmax(dim=-1)2
    #         pred_y = pred[b, t, :, 2].argmax(dim=-1)
    #         # swap x and y
    #         u, v = stupid_impl_array[b, pred_y].clone(), stupid_impl_array[b, pred_x].clone()
    #         stupid_impl_array[b, pred_x], stupid_impl_array[b, pred_y] = u, v

    # assert (array == stupid_impl_array).all(), 'Inconsistent implementation'
    # print('Consistent implementation!!')

    elem_close = torch.abs(array - label[:, -1, :, 0].cpu()) < 1e-5
    element_correct = elem_close.float().mean()
    array_correct = elem_close.all(dim=-1).float().mean()

    pred_first_action = pred[:, 0, :, 1:3].argmax(dim=-2).cpu()
    label_first_action = label[:, 0, :, 1:3].argmax(dim=-2).cpu()
    first_action_correct = (pred_first_action == label_first_action).all(dim=-1).float().mean()

    return {
        'element_accuracy' + name: as_float(element_correct),
        'array_accuracy' + name: as_float(array_correct),
        'first_action_accuracy' + name: as_float(first_action_correct)
    }


def shortest_path_1d_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, inp: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name
    pred_argmax = pred[:, :, :, -1].argmax(-1)
    label_argmax = label[:, :, :, -1].argmax(-1)

    argmax_accuracy = (pred_argmax == label_argmax).float().mean()

    # vis_array = torch.stack([pred_argmax, label_argmax], dim=1)
    # table = list()
    # for i in range(len(vis_array)):
    #     table.append((vis_array[i, 0].cpu().tolist(), vis_array[i, 1].cpu().tolist()))
    # print(tabulate(table))

    pred_argmax_first = pred_argmax[:, 0]
    label_argmax_first = label_argmax[:, 0]

    first_action_accuracy = (pred_argmax_first == label_argmax_first).float().mean()

    first_action_s = inp[:, :, 0, 1].argmax(dim=-1)
    first_action_t = pred_argmax_first
    first_action_feasibility = (inp[
        torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
        first_action_s,
        first_action_t,
        0
    ] > 0).float().cpu()

    final_t = label_argmax[:, -1]
    first_action_accuracy_2 = first_action_distance_accuracy(inp[..., 0], first_action_s, final_t, first_action_t).float().cpu()
    first_action_accuracy_2 = first_action_accuracy_2 * first_action_feasibility

    return {
        'argmax_accuracy' + name: as_float(argmax_accuracy),
        'first_action_accuracy' + name: as_float(first_action_accuracy),
        'first_action_feasibility' + name: as_float(first_action_feasibility.mean()),
        'first_action_accuracy_2' + name: as_float(first_action_accuracy_2.mean()),
    }


def get_shortest_batch(edges: torch.Tensor) -> torch.Tensor:
    """ Return the length of shortest path between nodes. """
    b = edges.shape[0]
    n = edges.shape[1]

    # n + 1 indicates unreachable.
    shortest = torch.ones((b, n, n), dtype=torch.float32, device=edges.device) * (n + 1)
    shortest[torch.where(edges == 1)] = 1
    # Make sure that shortest[x, x] = 0
    shortest -= shortest * torch.eye(n).unsqueeze(0).to(shortest.device)
    shortest = shortest

    # Floyd Algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != j:
                    shortest[:, i, j] = torch.min(shortest[:, i, j], shortest[:, i, k] + shortest[:, k, j])
    return shortest


def first_action_distance_accuracy(edge: torch.Tensor, s: torch.Tensor, t: torch.Tensor, pred: torch.Tensor):
    shortest = get_shortest_batch(edge.detach().cpu()).cuda()
    b = edge.shape[0]
    b_arrange = torch.arange(b, dtype=torch.int64, device=edge.device)
    return shortest[b_arrange, pred, t] < shortest[b_arrange, s, t]


def shortest_path_1d_accuracy_closed_loop(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, inp: torch.Tensor, sample_fn, name: str = '', execution_steps: int = 1):
    assert execution_steps in (1, 2), 'Only 1-step and 2-step execution is supported'
    b, t, n, _ = pred.shape
    failed = torch.zeros(b, dtype=torch.bool, device='cpu')
    succ = torch.zeros(b, dtype=torch.bool, device='cpu')

    for i in range(8 // execution_steps):
        pred_argmax = pred[:, :, :, -1].argmax(-1)
        pred_argmax_first = pred_argmax[:, 0]
        pred_argmax_second = pred_argmax[:, 1]
        target_argmax = inp[:, :, 0, 3].argmax(dim=-1)

        first_action_s = inp[:, :, 0, 1].argmax(dim=-1)
        first_action_t = pred_argmax_first
        first_action_feasibility = (inp[
            torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
            first_action_s,
            first_action_t,
            0
        ] > 0).cpu()
        last_t = first_action_t

        failed |= ~(first_action_feasibility.to(torch.bool))
        succ |= (first_action_t == target_argmax).cpu() & ~failed

        print(f'Step {i} (F) s={first_action_s[0].item()}, t={first_action_t[0].item()}, goal={target_argmax[0].item()}, feasible={first_action_feasibility[0].item()}')

        if execution_steps >= 2:
            second_action_s = first_action_t
            second_action_t = pred_argmax_second
            second_action_feasibility = (inp[
                torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
                second_action_s,
                second_action_t,
                0
            ] > 0).cpu()
            failed |= ~(second_action_feasibility.to(torch.bool))
            succ |= (second_action_t == target_argmax).cpu() & ~failed
            last_t = second_action_t

            print(f'Step {i} (S) s={second_action_s[0].item()}, t={second_action_t[0].item()}, goal={target_argmax[0].item()}, feasible={second_action_feasibility[0].item()}')

        inp_clone = inp.clone()
        inp_clone[:, :, :, 1] = 0
        inp_clone[torch.arange(b, dtype=torch.int64, device=inp.device), last_t, :, 1] = 1
        inp = inp_clone
        pred = sample_fn(inp, label, mask, batch_size=inp.size(0))

    return {
        'closed_loop_success_rate' + name: as_float(succ.float().mean()),
    }
import copy
def _sample_chunk(args):
    start_idx, end_idx, model, inp, label, mask = args
    # model_new = copy.deepcopy(model)
    model_new = model
    inp_chunk = inp[start_idx:end_idx]
    label_chunk = label[start_idx:end_idx]
    mask_chunk = mask[start_idx:end_idx]

    batch_size_chunk = inp_chunk.size(0)
    samples_chunk = model_new.sample(
        inp_chunk,
        label_chunk,
        mask_chunk,
        batch_size=batch_size_chunk
    ).detach()
    del inp_chunk, label_chunk, mask_chunk,model_new
    torch.cuda.empty_cache()
    return samples_chunk.cpu()

def parallel_sample(model, inp, label, mask, chunk_size=8, num_workers=2):
    N = inp.size(0)
    tasks = [
        (i, min(i + chunk_size, N), model, inp, label, mask)
        for i in range(0, N, chunk_size)
    ]

    with Pool(processes=num_workers) as p:
        results = p.map(_sample_chunk, tasks)

    samples = torch.cat(results, dim=0)
    return samples

import torch
def mask_label_add_entry(label, p,mask_origin):
    '''
    add more given entry for Sudoku problem
    '''
    B, h, w, num_class = label.shape
    device = label.device
    rand_tensor = torch.rand((B, h, w, num_class), device=device, dtype=torch.float32)
    mask_positions = (label == 1) & (rand_tensor < p)  & (mask_origin == 0)
    mask_positions_cond = (label == 1) & (mask_origin == 1)

    mask_final = mask_positions_cond | mask_positions

    masked_label = label.clone()
    masked_label[~mask_final] = -1


    mask = torch.zeros((B, h, w, 9), device=device, dtype=torch.float32)
    mask_final = mask_final.float()
    mask_final = mask_final.mean(-1,keepdim=True)
    mask_positions_final = mask_final.repeat(1,1,1,9) > mask

    mask[mask_positions_final] = 1.0
    inp = rearrange(masked_label, 'b h w c -> b (h w c)')
    mask = rearrange(mask, 'b h w c -> b (h w c)')

    return inp, mask

def mask_label_del_entry(label, p,mask_origin):
    '''
    decrease the given entry for Sudoku problem
    '''
    
    B, h, w, num_class = label.shape
    device = label.device
    rand_tensor = torch.rand((B, h, w, num_class), device=device, dtype=torch.float32)

    mask_final = (rand_tensor <(1-p)) & (label == 1) & (mask_origin == 1)
    masked_label = label.clone()
    masked_label[~mask_final] = -1


    mask = torch.zeros((B, h, w, 9), device=device, dtype=torch.float32)
    mask_final = mask_final.float()
    mask_final = mask_final.mean(-1,keepdim=True)
    mask_positions_final = mask_final.repeat(1,1,1,9) > mask

    mask[mask_positions_final] = 1.0
    inp = rearrange(masked_label, 'b h w c -> b (h w c)')
    mask = rearrange(mask, 'b h w c -> b (h w c)')
    return inp, mask

def random_replace(x, p, val_origin, val_replace):
    """
    This function takes a tensor `x` of shape [B,...], a probability `p`, a value `val_origin`, and a value `val_replace` as input.
    It randomly replaces elements in `x` that are close to `val_origin` with `val_replace` with a probability of `p`.

    Args:
        x (torch.Tensor): Input tensor of shape [B,...].
        p (float): Probability of replacement. Must be in the range [0, 1].
        val_origin: The original value to be replaced.
        val_replace: The value to replace with.

    Returns:
        torch.Tensor: The modified tensor after random replacement.
    """
    # Create a boolean mask indicating elements close to val_origin
    mask = torch.isclose(x, torch.tensor(val_origin))
    random_tensor = torch.rand(*x.shape)  # Generate random tensor with the same shape as x
    replace_mask = (random_tensor < p) & mask  # Create a mask for replacement based on p and val_origin
    x[replace_mask] = val_replace  # Replace elements where replace_mask is True
    return x

def random_permute_last_dim(X, p):
    """
    Randomly permutes the elements in the last dimension of X with probability p.
    Each [b, i, j] element has an independent probability p to be permuted.
    
    Args:
        X (torch.Tensor): Input tensor of shape [B, H, W, num_class].
        p (float): Probability of permuting the last dimension for each element.
    
    Returns:
        torch.Tensor: Tensor after applying random permutations.
    """
    # Ensure X is a floating tensor for sorting operations
    if X.shape[-1] ==729:
        X = X.view(-1,9,9,9)
    
    B, H, W, num_class = X.shape
    
    # Generate a mask with shape [B, H, W] where each element is True with probability p
    mask = torch.rand(B, H, W, device=X.device) < p  # [B, H, W]
    
    # Reshape X to [B*H*W, num_class] for easier processing
    X_reshaped = X.view(-1, num_class)  # [B*H*W, num_class]
    
    # Generate random scores for each element to create random permutations
    random_scores = torch.rand_like(X_reshaped)  # [B*H*W, num_class]
    
    # Get permutation indices by sorting the random scores
    perm_indices = torch.argsort(random_scores, dim=1)  # [B*H*W, num_class]
    
    # Create identity indices for cases where permutation is not applied
    identity_indices = torch.arange(num_class, device=X.device).unsqueeze(0).expand_as(perm_indices)  # [B*H*W, num_class]
    
    # Expand mask to match the permutation indices shape
    mask_flat = mask.view(-1, 1).expand(-1, num_class)  # [B*H*W, num_class]
    
    # Select permutation indices where mask is True, else use identity
    final_indices = torch.where(mask_flat, perm_indices, identity_indices)  # [B*H*W, num_class]
    
    # Gather the elements based on the final indices to get the permuted tensor
    X_permuted = torch.gather(X_reshaped, dim=1, index=final_indices)  # [B*H*W, num_class]
    
    # Reshape back to original shape [B, H, W, num_class]
    X_output = X_permuted.view(B, H, W, num_class)
    if X.shape[-1] ==729:
        X_output = rearrange(X_output, 'b h w c -> b (h w c)')
    
    return X_output

def add_l2_noise(x: torch.Tensor, distance: float) -> torch.Tensor:
    """
    Adds L2 noise to each example in the input tensor such that the L2 distance
    between the noised example and the original example equals the specified distance.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, ...].
        distance (float): Desired L2 distance between each noised example and the original example.

    Returns:
        torch.Tensor: Tensor of the same shape as `x` with added noise.
    """
    # Ensure distance is a positive scalar
    # assert distance >= 0, "Distance must be non-negative."
    
    # Generate random noise with the same shape as `x`
    noise = torch.randn_like(x)
    
    # Normalize the noise to have an L2 norm of 1 for each example
    noise_norms = torch.norm(noise.view(noise.size(0), -1), dim=1, keepdim=True)  # L2 norms of noise
    noise = noise / noise_norms.view(noise.size(0), *([1] * (len(x.shape) - 1)))  # Reshape to match x
    
    # Scale the noise to have the desired L2 distance
    noise = noise * distance
    
    # Add noise to the original tensor
    x_noised = x + noise
    
    return x_noised

import torch

def generate_p_close_p_far(batch_size, max_strength_permutation, min_gap, max_gap, device='cpu'):
    """
    Generate a batch of p_close and p_far values satisfying specific conditions.

    Args:
        batch_size (int): Number of p_close, p_far pairs to generate.
        max_strength_permutation (float): Maximum strength for permutation.
        min_gap (float): Minimum gap between p_close and p_far.
        max_gap (float): Maximum gap between p_close and p_far.
        device (str): Device to perform the computation ('cpu' or 'cuda').

    Returns:
        tuple: Two torch tensors (p_close, p_far) of shape (batch_size,).
    """
    resolution = 1.0 / 81.0

    # Generate random values for p_close and p_far
    p_close = torch.rand(batch_size, device=device) * max_strength_permutation + min_gap
    p_far = torch.rand(batch_size, device=device) * max_strength_permutation + min_gap

    # Ensure p_close <= p_far
    p_close, p_far = torch.min(p_close, p_far), torch.max(p_close, p_far)

    # Apply square transformation
    p_close = p_close ** 2
    p_far = p_far ** 2

    # Convert to number of entries
    p_close_num_entry = (p_close / resolution).long()
    p_close_num_entry = torch.max(p_close_num_entry, torch.tensor(3, device=device))
    p_far_num_entry = (p_far / resolution).long()

    # Generate random maximum gap
    p_gap_max = torch.rand(batch_size, device=device) * (max_gap - min_gap) + min_gap
    p_gap_max = p_gap_max ** 2
    gap_num_entry_max = torch.max((p_gap_max / resolution).long(), torch.tensor(8, device=device))
    gap_num_entry_min = torch.max((torch.tensor(min_gap ** 2, device=device) / resolution).long(), torch.tensor(3, device=device))

    # Adjust p_far_num_entry to meet conditions using vectorized operations
    condition_1 = (p_far_num_entry - p_close_num_entry) < gap_num_entry_min
    condition_2 = (p_far_num_entry - p_close_num_entry) > gap_num_entry_max

    adjust_indices = condition_1 | condition_2
    p_far_num_entry = torch.where(adjust_indices, p_close_num_entry + gap_num_entry_min, p_far_num_entry)

    # Convert back to p_close and p_far
    p_close = p_close_num_entry * resolution
    p_far = p_far_num_entry * resolution

    return p_close, p_far
import torch

def generate_distances(batch_size, max_strength_permutation_xt, min_gap, max_gap, device='cpu'):
    """
    Generate a batch of distance_close and distance_far values satisfying specific conditions.

    Args:
        batch_size (int): Number of distance_close and distance_far pairs to generate.
        max_strength_permutation_xt (float): Maximum strength for permutation xt.
        min_gap_xt (float): Minimum gap between distance_close and distance_far.
        max_gap_xt (float): Maximum gap between distance_close and distance_far.
        device (str): Device to perform the computation ('cpu' or 'cuda').

    Returns:
        tuple: Two torch tensors (distance_close, distance_far) of shape (batch_size,).
    """
    # Generate random values for distance_close
    distance_close = torch.rand(batch_size, device=device) * (max_strength_permutation_xt-max_gap) + min_gap
    distance_far = torch.rand(batch_size, device=device) * (max_strength_permutation_xt - distance_close) + distance_close + 0.01

    # Apply square transformation
    distance_close = distance_close ** 2
    distance_far = distance_far ** 2

    # Generate max_gap and min_gap
    max_gap = max_gap ** 2
    min_gap = min_gap ** 2

    # Apply constraints on distance_far
    gap = distance_far - distance_close
    distance_far = torch.where(gap > max_gap, distance_close + max_gap, distance_far)  # Enforce max_gap
    distance_far = torch.where(gap < min_gap, distance_close + min_gap, distance_far)  # Enforce min_gap

    return distance_close, distance_far

def calculate_monotonicity(data):
    """
    Analyzes the monotonicity of a batch of sequences using PyTorch.

    Args:
        data (torch.Tensor): A 2D tensor of shape [batch_size, num_steps] where
                             each row is a sequence of numbers.

    Returns:
        torch.Tensor: A 1D tensor of shape [batch_size,] containing monotonicity
                      scores for each sequence in the batch. The score ranges
                      between 0 and 1, with 1 indicating perfect monotonic increase.
    """
    # Compute differences between consecutive elements along each sequence
    diffs = data[:, 1:] - data[:, :-1]

    # Count the number of non-negative differences (indicating monotonic increase)
    positive_counts = torch.sum(diffs >= 0, dim=1)

    # Normalize by the total number of differences to get a score between 0 and 1
    monotonicity = positive_counts / (data.size(1) - 1)

    return monotonicity

def generate_weights(energy_x, energy_close_x, min_weight_neg_contrat_x, max_weight_neg_contrat_x=1.0):
    """
    Generate weights based on conditions on energy_x0 and energy_close_x0.

    Args:
        energy_x0 (torch.Tensor): Tensor of shape [Batch_size, 1].
        energy_close_x0 (torch.Tensor): Tensor of shape [Batch_size, 1].
        min_weight_neg_contrat_x0 (float): Weight for elements where energy_x0 < energy_close_x0.
        max_weight_neg_contrat_x0 (float): Weight for other elements.

    Returns:
        torch.Tensor: Weights tensor of shape [Batch_size, 1].
    """
    # Condition: energy_x0 < energy_close_x0
    condition = energy_x < energy_close_x

    # Generate weights based on the condition
    weights = torch.where(condition, 
                          torch.full_like(energy_x, min_weight_neg_contrat_x), 
                          torch.full_like(energy_x, max_weight_neg_contrat_x))
    
    return weights


def generate_p_stack(batch_size, max_strength_permutation, min_gap, max_gap, num_distance_neg_contrast, device='cpu'):
    """
    Generate a tensor `p_stack` with shape `[num_distance_neg_contrast, batch_size]`, 
    where values along the first dimension `p` increase quadratically for each case independently. 
    The growth rate is a quadratic function ensuring that:
    - Minimum gap between consecutive values is `min_gap`.
    - Maximum gap between consecutive values is `max_gap`.
    - Maximum value is capped at `max_strength_permutation`.

    The second dimension values are independent from each other.

    Args:
        batch_size (int): The size of the batch (second dimension).
        max_strength_permutation (float): Maximum allowable value for `p`.
        min_gap (float): Minimum gap between consecutive values in the first dimension.
        max_gap (float): Maximum gap between consecutive values in the first dimension.
        num_distance_neg_contrast (int): Number of values along the first dimension.
        device (str): Device to place the tensor (`cpu` or `cuda`).

    Returns:
        torch.Tensor: A tensor of shape `[num_distance_neg_contrast, batch_size]` with the described properties.
    """
    # Randomize base indices per batch
    base_indices = torch.arange(num_distance_neg_contrast, device=device, dtype=torch.float32).unsqueeze(1)
    # random_factors = torch.rand((num_distance_neg_contrast, batch_size), device=device)
    
    # Generate quadratic gaps independently for each case in the batch
    quadratic_gaps = min_gap + (max_gap-min_gap) * ((base_indices / (num_distance_neg_contrast - 1))**2)
    
    # Cumulatively sum gaps to create a quadratic growth
    p_stack = quadratic_gaps.clamp(max=max_strength_permutation).repeat(1, batch_size)
    
    # Clamp values to ensure they don't exceed max_strength_permutation
    p_stack = torch.clamp(p_stack, max=max_strength_permutation)

    return p_stack

def apply_mask_maze(x,mask):
    '''
    Apply the mask to tensor `x` such that:
    - x[i, j, k, 0] = -1 when mask[i, j, k] is True
    - x[i, j, k, 1] = 1 when mask[i, j, k] is True
    
    Args:
        x (torch.Tensor): The input tensor with shape [64, 9, 9, 2].
        mask (torch.Tensor): The mask with shape [64, 9, 9], where True/False determines which elements to update.
    
    Returns:
        torch.Tensor: The masked tensor `x_masked` with the updated values.
    '''
    x_1 = x[...,0].clone() #
    x_minus_1 = x[...,1].clone() #
    x_1[mask] = -1.0
    x_minus_1[mask] = 1.0
    x_masked = torch.stack([x_1, x_minus_1], dim=-1)

    return x_masked

import torch

def linear_fit_three_points(
    p_close:  torch.Tensor,  # shape: [batch_size, 1]
    E_close:  torch.Tensor,  # shape: [batch_size, 1]
    p_far:    torch.Tensor,  # shape: [batch_size, 1]
    E_far:    torch.Tensor,  # shape: [batch_size, 1]
    E_0:      torch.Tensor,  # shape: [batch_size, 1]
):
    """
    Given three points for each sample in the batch:
        (0, E_0[i]), (p_close[i], E_close[i]), (p_far[i], E_far[i]),
    we fit a line E[i] = k[i] * p[i] + b[i] (no for-loop) and return:
        - k:        shape [batch_size]
        - mse_fit:  shape [batch_size], mean squared error over the 3 points
    """

    # 1. Construct p, E as shape [batch_size, 3]
    #    We'll have p = [0, p_close, p_far], and E = [E_0, E_close, E_far].
    p_zero = torch.zeros_like(p_close)                     # shape: [batch_size, 1]
    p      = torch.cat([p_zero, p_close, p_far], dim=1)    # shape: [batch_size, 3]
    E      = torch.cat([E_0,    E_close, E_far],  dim=1)   # shape: [batch_size, 3]

    # 2. Compute mean p and mean E over the three points (dim=1)
    p_mean = p.mean(dim=1, keepdim=True)  # shape: [batch_size, 1]
    E_mean = E.mean(dim=1, keepdim=True)  # shape: [batch_size, 1]

    # 3. Compute slope k using the standard least-squares formula:
    #    k = sum((p - p_mean)*(E - E_mean)) / sum((p - p_mean)^2)
    #    We'll add a small value (1e-12) to avoid division by zero.
    numerator   = ((p - p_mean) * (E - E_mean)).sum(dim=1)       # shape: [batch_size]
    denominator = ((p - p_mean) ** 2).sum(dim=1)                 # shape: [batch_size]
    k = numerator / (denominator + 1e-12)                         # shape: [batch_size]

    # 4. Compute intercept b:
    #    b = E_mean - k * p_mean
    #    p_mean, E_mean are shape [batch_size, 1], we convert them to [batch_size] using [:, 0].
    b = E_mean[:, 0] - k * p_mean[:, 0]                           # shape: [batch_size]

    # 5. Compute predictions E_pred = k * p + b for the 3 points, then compute MSE over them
    #    E_pred will be shape [batch_size, 3].
    E_pred = k.unsqueeze(1) * p + b.unsqueeze(1)                  # broadcast to shape [batch_size, 3]

    # mse_fit is the mean squared error across the 3 points, shape: [batch_size]
    mse_fit = ((E - E_pred) ** 2).mean(dim=1)

    return k, mse_fit

import numpy as np
from scipy.stats import kendalltau, spearmanr

def similarity(X, Y,algorithm='kendall'):
    """
    Calculate the Kendall's Tau similarity between two lists X and Y.
    alrogithm: 'kendall' or 'spearman'
    Parameters:
    X : list of int
        A list of length N representing the first ordering.
    Y : list of int
        A list of length N representing the second ordering.
        
    Returns:
    float
        Kendall's Tau similarity between the two orderings.
    """
    # Ensure X and Y are numpy arrays for easier manipulation
    X = np.array(X)
    Y = np.array(Y)

    # Compute Kendall's Tau
    if algorithm == 'kendall':
        tau, _ = kendalltau(X, Y)
    elif algorithm == 'spearman':
        tau, _ = spearmanr(X, Y)

    return tau

def calculate_order_consistency(X, Y):
    """
    Calculate the proportion of consistent pairs (i.e., pairs with the same relative order) 
    between the two sorting results X and Y.

    Parameters:
    X : ndarray, the index array of the first sorting result, shape (N,)
    Y : ndarray, the index array of the second sorting result, shape (N,)

    Returns:
    float : proportion of consistent pairs (i.e., same relative order)
    """
    n = len(X)
    consistent_pairs = 0
    total_pairs = 0

    # Iterate over all element pairs (i, j), where i < j
    for i in range(n):
        index_of_i_X = np.where(X == i)[0][0]
        index_of_i_Y = np.where(Y == i)[0][0]
        for j in range(i + 1, n):
            total_pairs += 1
            index_of_j_X = np.where(X == j)[0][0]
            index_of_j_Y = np.where(Y == j)[0][0]
            # Check if the relative order of (i, j) is consistent between X and Y
            if (index_of_i_X < index_of_j_X and index_of_i_Y < index_of_j_Y) or (index_of_i_X > index_of_j_X and index_of_i_Y > index_of_j_Y):
                consistent_pairs += 1
    # Calculate the proportion of consistent pairs
    return consistent_pairs / total_pairs

def process_tensor_sudoku(x):
    """
    This function processes the input tensor x.
    The input tensor has the shape [B, 9, 9, 9].
    It sets the position of the max element in the last dimension to 1, and sets the other positions to -1.

    Parameters:
    x (torch.Tensor): The input tensor with shape [B, 9, 9, 9]

    Returns:
    torch.Tensor: The processed tensor with shape [B, 9, 9, 9]
    """
    # Find the indices of the maximum elements along the last dimension
    x = x.view(-1,9,9,9)
    max_indices = torch.argmax(x, dim=-1)
    # Create a tensor with the same shape as x, filled with -1
    result = torch.full_like(x, -1.0)
    # Create index tensors for the first three dimensions
    b_indices = torch.arange(x.shape[0]).view(-1, 1, 1)
    i_indices = torch.arange(x.shape[1]).view(1, -1, 1)
    j_indices = torch.arange(x.shape[2]).view(1, 1, -1)
    # Expand the dimension of max_indices
    max_indices = max_indices
    # Set the position of the maximum element to 1
    result[b_indices, i_indices, j_indices, max_indices] = 1.0
    
    result = rearrange(result, 'b h w c -> b (h w c)')
    return result