import os
import os.path as osp
import time
# Prevent numpy over multithreading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from tscend_src.model.diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D
from tscend_src.model.models import EBM, DiffusionWrapper
from tscend_src.model.models import SudokuEBM, SudokuTransformerEBM, SudokuDenoise, SudokuLatentEBM, AutoencodeModel
from tscend_src.data.sat_dataset import SATNetDataset, SudokuDataset, SudokuRRNDataset, SudokuRRNLatentDataset
import torch
import numpy as np
import argparse
import hashlib

from tscend_src.utils.utils import set_seed
from tscend_src.filepath import EXP_PATH,SRC_PATH
import torch.multiprocessing as mp

# mp.set_start_method('spawn', force=True)
# try:
#     import mkl
#     mkl.set_num_threads(1)
# except ImportError:
#     print('Warning: MKL not initialized.')


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ['0', 'n', 'f']:
        return False
    elif x[0] in ['1', 'y', 't']:
        return True
    raise ValueError('Invalid value: {}'.format(x))


parser = argparse.ArgumentParser(description='Train Diffusion Reasoning Model')
parser.add_argument('--dataset', default='inverse', type=str, help='dataset to evaluate')
parser.add_argument('--inspect-dataset', action='store_true', help='run an IPython embed interface after loading the dataset')
parser.add_argument('--model', default='mlp', type=str, choices=['mlp', 'mlp-reverse', 'sudoku', 'sudoku-latent', 'sudoku-transformer', 'sudoku-reverse', 'gnn', 'gnn-reverse', 'gnn-conv', 'gnn-conv-1d', 'gnn-conv-1d-v2', 'gnn-conv-1d-v2-reverse'])
parser.add_argument('--ckpt', type=str, default=None, help='directory to load model and save results')
parser.add_argument('--batch_size', default=2048, type=int, help='size of batch of input to use')
parser.add_argument('--diffusion_steps', default=10, type=int, help='number of diffusion time steps (default: 10)')
parser.add_argument('--rank', default=20, type=int, help='rank of matrix to use')
parser.add_argument('--data-workers', type=int, default=None, help='number of workers to use for data loading')
parser.add_argument('--supervise-energy-landscape', type=str2bool, default=False)
parser.add_argument('--use-innerloop-opt', type=str2bool, default=False, help='use inner loop optimization if innerloop_opt_steps > 0')
parser.add_argument('--innerloop_opt_steps', type=int, default=20, help='number of inner loop optimization steps')
parser.add_argument('--cond_mask', type=str2bool, default=False)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--latent', action='store_true', default=False)
parser.add_argument('--ood', action='store_true', default=False)
parser.add_argument('--baseline', action='store_true', default=False)
parser.add_argument('--data_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_seed', type=int, default=1)

# MCTS config
parser.add_argument('--inference_method', type=str, default='mcts', choices=['mcts', 'diffusion_baseline','naive_diffusion','mixed_inference'])
parser.add_argument('--mcts_start_step', type=int, default=0, help='start step for mcts')
parser.add_argument('--max_root_num', type=int, default=1, help='number of root nodes,9 means mcts and 0 means diffusion baseline')
parser.add_argument('--num_rood_decay', type=float, default=1.0, help='decay rate of num of root nodes')
parser.add_argument('--mcts_type', type=str, default='continuous', choices=['continuous', 'discrete'])
parser.add_argument('--K', type=int, default=10, help='number of search braches')
parser.add_argument('--mcts_noise_scale', type=float, default=10., help='noise scale for mcts')
parser.add_argument('--exploration_weight', type=float, default=1.0,help='exploration weight for mcts')
parser.add_argument('--steps_rollout', type=int, default=200, help='steps for rollout MCTS')
parser.add_argument('--noise_scale', type=float, default=1.0, help='noise scale to get the action')
parser.add_argument('--p_h', type=float, default=0.4, help='ratio of cols to select action')
parser.add_argument('--p_w', type=float, default=0.4, help='ratio of rows to select action')
parser.add_argument('--topk', type=int, default=25,help='select topk children nodes as the whole children nodes')
parser.add_argument('--J_type', type=str, default='energy_learned', choices=['energy_learned', 'J_defined','mixed','GD_accuracy','path_f1'])
parser.add_argument('--noise_type', type=str, default='gaussian', choices=['gaussian', 'uniform','permutation'])

# diffusion
parser.add_argument('--sampling_timesteps', type=int, default=5)
parser.add_argument('--show_inference_tqdm', type=str2bool, default=False)

# Experiment config
parser.add_argument('--exp_name', type=str, default='sudoku-rrn')
parser.add_argument('--datetime', type=str, default='2021-10-12-20-00-00')
parser.add_argument('--results_name', type=str, default='results')
parser.add_argument('--task_difficulty', type=str, default='easy', choices=['same', 'harder'])
parser.add_argument('--num_batch', type=int, default=100)

parser.add_argument('--proportion_added_entry', type=float, default=0.0)
parser.add_argument('--proportion_deleted_entry', type=float, default=0.0)
parser.add_argument('--test_condition_generalization',type=str2bool,default =False)
parser.add_argument('--plt_energy_landscape_permutation_noised',type=str2bool,default =False)


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    set_seed(FLAGS.seed)
    FLAGS.datetime = time.strftime('%Y-%m-%d-%H-%M-%S')
    # if FLAGS.ckpt file path , get the directory path
    args_dict = vars(FLAGS)
    sorted_args = {k: args_dict[k] for k in sorted(args_dict)}
    data_to_hash = str(sorted_args)
    hash_value = hashlib.sha256(data_to_hash.encode()).hexdigest()[:8] # TODO only use the first 8 characters
    if osp.isfile(FLAGS.ckpt):
        FLAGS.results_path = osp.dirname(FLAGS.ckpt)
    else:
        FLAGS.results_path = EXP_PATH + '/results/' + f'ds_{FLAGS.dataset}' + '/' + FLAGS.ckpt + '/'+str(hash_value)
    validation_dataset = None
    extra_validation_datasets = dict()
    extra_validation_every_mul = 10
    save_and_sample_every = 1000
    validation_batch_size = 256
    if FLAGS.dataset == 'sudoku':
        train_dataset = SudokuDataset(FLAGS.dataset, split='train')
        validation_dataset = SudokuDataset(FLAGS.dataset, split='val')
        extra_validation_datasets = {'sudoku-rrn-test': SudokuRRNDataset('sudoku-rrn', split='test')}
        dataset = train_dataset
        metric = 'sudoku'
        validation_batch_size = 64
        assert FLAGS.cond_mask
    elif FLAGS.dataset == 'sudoku-rrn':
        train_dataset = SudokuRRNDataset(FLAGS.dataset, split='train')
        validation_dataset = SudokuRRNDataset(FLAGS.dataset, split='test')
        save_and_sample_every = 10000
        dataset = train_dataset
        metric = 'sudoku'
        assert FLAGS.cond_mask
    elif FLAGS.dataset == 'sudoku-rrn-latent':
        train_dataset = SudokuRRNLatentDataset(FLAGS.dataset, split='train')
        validation_dataset = SudokuRRNLatentDataset(FLAGS.dataset, split='validation')
        save_and_sample_every = 10000
        dataset = train_dataset
        metric = 'sudoku_latent'
    else:
        assert False

    if FLAGS.inspect_dataset:
        from IPython import embed
        embed()
        exit()
    if FLAGS.model == 'sudoku':
        model = SudokuEBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
        model = DiffusionWrapper(model)
    elif FLAGS.model == 'sudoku-latent':
        model = SudokuLatentEBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
        model = DiffusionWrapper(model)
    else:
        assert False

    kwargs = dict()
    if FLAGS.baseline:
        kwargs['baseline'] = True

    if FLAGS.dataset in ['addition', 'inverse', 'lowrank']:
        kwargs['continuous'] = True

    if FLAGS.dataset in ['sudoku', 'sudoku_latent', 'sudoku-rrn', 'sudoku-rrn-latent']:
        kwargs['sudoku'] = True

    if FLAGS.dataset in ['connectivity', 'connectivity-2']:
        kwargs['connectivity'] = True

    if FLAGS.dataset in ['shortest-path', 'shortest-path-1d']:
        kwargs['shortest_path'] = True

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 32,
        objective = 'pred_noise',  # Alternative pred_x0
        timesteps = FLAGS.diffusion_steps,  # number of steps
        sampling_timesteps = FLAGS.sampling_timesteps,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper]),
        supervise_energy_landscape = FLAGS.supervise_energy_landscape,
        use_innerloop_opt = FLAGS.use_innerloop_opt,
        show_inference_tqdm = FLAGS.show_inference_tqdm,
        args = FLAGS,
        innerloop_opt_steps = FLAGS.innerloop_opt_steps,
        **kwargs
    )
    
    if FLAGS.latent:
        # Load the decoder
        autoencode_model = AutoencodeModel(729, 729)
        ckpt = torch.load("results/autoencode_sudoku-rrn/model_mlp_diffsteps_10/model-1.pt")
        model_ckpt = ckpt['model']
        autoencode_model.load_state_dict(model_ckpt)
    else:
        autoencode_model = None
    trainer = Trainer1D(
        diffusion,
        dataset,
        train_batch_size = FLAGS.batch_size,
        validation_batch_size = validation_batch_size,
        train_lr = 1e-4,
        train_num_steps = 1300000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        data_workers = FLAGS.data_workers,
        amp = False,                      # turn on mixed precision
        metric = metric,
        results_folder = FLAGS.results_path,
        cond_mask = FLAGS.cond_mask,
        validation_dataset = validation_dataset,
        extra_validation_datasets = extra_validation_datasets,
        extra_validation_every_mul = extra_validation_every_mul,
        save_and_sample_every = save_and_sample_every,
        evaluate_first = FLAGS.evaluate,  # run one evaluation first
        latent = FLAGS.latent,  # whether we are doing reasoning in the latent space
        autoencode_model = autoencode_model,
        exp_hash_code=hash_value
    )

    load_milestone = FLAGS.ckpt
    trainer.load(load_milestone)
    diffusion = trainer.model
    print(f'Loaded model from {FLAGS.ckpt}')
    diffusion.model.eval()
    
    # load data
    if FLAGS.dataset == 'sudoku':
        test_dataset = SudokuDataset(FLAGS.dataset, split='val')
        extra_validation_datasets = SudokuRRNDataset('sudoku-rrn', split='test')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.data_workers,pin_memory=True)
        extra_validation_dataloaders = torch.utils.data.DataLoader(extra_validation_datasets, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.data_workers,pin_memory=True)
    else:
        assert 'not implemented'
    # trainer.inference(test_dataloader, trainer.device)
    import time
    start = time.time()
    seed = FLAGS.seed
    for idx_seed in range(FLAGS.n_seed):
        print(f'??????? Seed: {seed} ???????')
        if FLAGS.task_difficulty == 'harder':
            trainer.inference(extra_validation_dataloaders, seed, trainer.device)
        else:
            trainer.inference(test_dataloader, seed, trainer.device)
        seed += 1
    print(f'Inference time: {time.time() - start}')
