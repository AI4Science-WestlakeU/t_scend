import debugpy

import os
import os.path as osp

# Prevent numpy over multithreading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from tscend_src.model.diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D
from tscend_src.model.models import EBM, DiffusionWrapper
from tscend_src.model.models import SudokuEBM, SudokuDenoise, AutoencodeModel, MazeDenoise, MazeEBM
from tscend_src.data.sat_dataset import SATNetDataset, SudokuDataset, SudokuRRNDataset, SudokuRRNLatentDataset
import torch
import hashlib
from tscend_src.data.data_maze import MazeData,reconstruct_maze_solved,calculate_path_conformity,calculate_path_continuity,maze_accuracy,normalize_last_dim

import argparse

from tscend_src.utils.utils import set_seed
from tscend_src.filepath import EXP_PATH,SRC_PATH

try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    print('Warning: MKL not initialized.')


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
parser.add_argument('--exp_name', default='default', type=str, help='name of experiment')
parser.add_argument('--dataset', default='inverse', type=str, help='dataset to evaluate')
parser.add_argument('--inspect-dataset', action='store_true', help='run an IPython embed interface after loading the dataset')
parser.add_argument('--model', default='mlp', type=str, choices=['mlp', 'mlp-reverse', 'sudoku', 'sudoku-latent', 'sudoku-transformer', 'sudoku-reverse', 'gnn', 'gnn-reverse', 'gnn-conv', 'gnn-conv-1d', 'gnn-conv-1d-v2', 'gnn-conv-1d-v2-reverse', 'maze-EBM','maze-denoise'], help='model to use')
parser.add_argument('--load-milestone', type=str, default=None, help='load a model from a milestone')
parser.add_argument('--batch_size', default=2048, type=int, help='size of batch of input to use')
parser.add_argument('--train_num_steps', type=int, default=100000)
parser.add_argument('--save_and_sample_every', type=int, default=10000)
parser.add_argument('--save_loss_curve', type=str2bool, default=False,help='whether to save loss curve')
parser.add_argument('--diffusion_steps', default=10, type=int, help='number of diffusion time steps (default: 10)')
parser.add_argument('--rank', default=20, type=int, help='rank of matrix to use')
parser.add_argument('--data-workers', type=int, default=None, help='number of workers to use for data loading')
parser.add_argument('--supervise-energy-landscape', type=str2bool, default=False)
parser.add_argument('--use-innerloop-opt', type=str2bool, default=False)
parser.add_argument('--innerloop_opt_steps', type=int, default=20, help='number of inner loop optimization steps')
parser.add_argument('--cond_mask', type=str2bool, default=False)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--latent', action='store_true', default=False)
parser.add_argument('--ood', action='store_true', default=False)
parser.add_argument('--baseline', action='store_true', default=False)
parser.add_argument('--inference_method', type=str, default='naive_diffusion', choices=['mcts', 'diffusion_baseline','naive_diffusion'])
parser.add_argument('--train_data_path', type=str, default=None)
parser.add_argument('--val_data_path', type=str, default=None)
parser.add_argument('--test_data_path', type=str, default=None)
parser.add_argument('--num__size_training', type=int, default=1)

parser.add_argument('--train_data_path_medium', type=str, default=None)
parser.add_argument('--val_data_path_medium', type=str, default=None)
parser.add_argument('--test_data_path_medium', type=str, default=None)

parser.add_argument('--train_data_path_hard', type=str, default=None)
parser.add_argument('--val_data_path_hard', type=str, default=None)
parser.add_argument('--test_data_path_hard', type=str, default=None)


# more training configurations
parser.add_argument('--proportion_added_entry', type=float, default=0.0,help='proportion of added entry')
parser.add_argument('--proportion_deleted_entry', type=float, default=0.0,help='proportion of deleted entry')

## advanced training configurations
# KL loss:
parser.add_argument('--kl_coef', type=float, default=0.0, help='coefficient for kl regularization')
parser.add_argument('--kl_max_end_step', type=int, default=0, help='maximum end_step for kl')
parser.add_argument('--kl_interval', type=int, default=50, help='Compute the kl/entropy loss every kl_interval steps')
parser.add_argument('--kl_enable_grad_steps', type=int, default=1, help='number of gradient steps enabled for kl/entropy loss')

# entropy loss:
parser.add_argument('--entropy_coef', type=float, default=0.0,help='coefficient for maximum entropy regularization')
parser.add_argument('--entropy_k_nearest_neighbor', type=int, default=3,help="Choosing K'th nearest neighbor when computing entropy")

# contrastive loss configurations
parser.add_argument('--neg_contrast_coef', type=float, default=0.0,help='coefficient for negative contrastive loss')
parser.add_argument('--neg_contrast_coef_x0', type=float, default=0.0,help='coefficient for negative contrastive loss at x0')
parser.add_argument('--neg_contrast_coef_xt', type=float, default=0.0,help='coefficient for negative contrastive loss at xt')
parser.add_argument('--max_strength_permutation_x0', type=float, default=0.7,help='maximum strength for permutation')
parser.add_argument('--max_strength_permutation_xt', type=float, default=0.7,help='maximum strength for permutation')
parser.add_argument('--max_gap_x0', type=float, default=0.7,help='maximum gap for x0')
parser.add_argument('--min_gap_x0', type=float, default=0.1,help='minimum gap for x0')
parser.add_argument('--max_gap_xt', type=float, default=0.7,help='maximum gap for xt')
parser.add_argument('--min_gap_xt', type=float, default=0.07,help='minimum gap for xt')
parser.add_argument('--diverse_gap_batch', type=str2bool, default=False,help='whether to use diverse gap in the same batch')
parser.add_argument('--min_weight_neg_contrat_x0', type=float, default=1.0,help='minimum weight for negative contrastive loss at x0 for any pair')
parser.add_argument('--min_weight_neg_contrat_xt', type=float, default=1.0,help='minimum weight for negative contrastive loss at xt for any pair')
parser.add_argument('--num_distance_neg_contrast', type=int, default=1,help='number of distance for negative sample')
parser.add_argument('--num_sample_each_neg_contrast', type=int, default=1,help='number of samples for each distance')
parser.add_argument('--monotonicity_landscape_loss', type=str2bool, default=False,help='whether to use monotonicity_lanscape_loss')
parser.add_argument('--monotonicity_landscape_k_loss_coef_x0', type=float, default=1.0,help='coefficient for monotonicity_lanscape_loss')
parser.add_argument('--monotonicity_landscape_fit_loss_coef_x0', type=float, default=1.0,help='coefficient for monotonicity_lanscape_loss')
parser.add_argument('--monotonicity_landscape_k_loss_coef_xt', type=float, default=1.0,help='coefficient for monotonicity_lanscape_loss')
parser.add_argument('--monotonicity_landscape_fit_loss_coef_xt', type=float, default=1.0,help='coefficient for monotonicity_lanscape_loss')
parser.add_argument('--k_min_monotonicity_landscape_x0', type=float, default=1.0,help='minimum value for monotonicity_lanscape_loss')
parser.add_argument('--k_min_monotonicity_landscape_xt', type=float, default=1.0,help='minimum value for monotonicity_lanscape_loss')
parser.add_argument('--use_monotonicity_and_constrat_both',type=str2bool, default=False)
parser.add_argument('--coef_naive_contrast',type= float,default = 0.01)

# plt configurations
parser.add_argument('--test_during_training', type=str2bool, default=True,help='whether to test during training')

# speccial configurations for maze
parser.add_argument('--loss_type', type=str, default='mse',help='loss type for maze')

if __name__ == "__main__":
    FLAGS = parser.parse_args()

    validation_dataset = None
    extra_validation_datasets = dict()
    extra_validation_every_mul = 10
    save_and_sample_every = 10
    validation_batch_size = 256
    
    train_dataset_medium = None
    validation_dataset_medium = None
    extra_validation_datasets_medium = None
    train_dataset_hard = None
    validation_dataset_hard = None
    extra_validation_datasets_hard = None
    if FLAGS.dataset == 'sudoku':
        train_dataset = SudokuDataset(FLAGS.dataset, split='train')
        validation_dataset = SudokuDataset(FLAGS.dataset, split='val')
        extra_validation_datasets = {'sudoku-rrn-test': SudokuRRNDataset('sudoku-rrn', split='test')}
        dataset = train_dataset
        metric = 'sudoku'
        validation_batch_size = 128
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
    elif FLAGS.dataset == 'maze':
        num_train_datapoint = eval(FLAGS.train_data_path.split('N-')[1])
        num_val_datapoint = eval(FLAGS.val_data_path.split('N-')[1])
        num_test_datapoint = eval(FLAGS.test_data_path.split('N-')[1])
        train_dataset = MazeData(
                        dataset_name="Maze",
                        dataset_path=FLAGS.train_data_path,
                        mode = 'train',# 'test'
                        num_datapoints = num_train_datapoint
                        )
        validation_dataset = MazeData(
                        dataset_name="Maze",
                        dataset_path=FLAGS.val_data_path,
                        mode = 'val',# 'test'
                        num_datapoints = num_val_datapoint
                        )
        extra_validation_datasets = {
            'maze-test': MazeData(
                        dataset_name="Maze",
                        dataset_path=FLAGS.test_data_path,
                        mode = 'test',
                        num_datapoints = num_test_datapoint
                        )
        }
        train_dataset_medium = None
        validation_dataset_medium = None
        extra_validation_datasets_medium = None
        train_dataset_hard = None
        validation_dataset_hard = None
        extra_validation_datasets_hard = None
        if FLAGS.num__size_training >1:
            num_train_datapoint_medium = eval(FLAGS.train_data_path_medium.split('N-')[1])
            # num_val_datapoint_medium = eval(FLAGS.val_data_path_medium.split('N-')[1])
            # num_test_datapoint_medium = eval(FLAGS.test_data_path_medium.split('N-')[1])
            train_dataset_medium = MazeData(
                        dataset_name="Maze",
                        dataset_path=FLAGS.train_data_path_medium,
                        mode = 'train',# 'test'
                        num_datapoints = num_train_datapoint_medium
                        )
            # validation_dataset_medium = MazeData(
            #             dataset_name="Maze",
            #             dataset_path=FLAGS.val_data_path_medium,
            #             mode = 'val',# 'test'
            #             num_datapoints = num_val_datapoint_medium
            #             )
            # extra_validation_datasets_medium = {
            #     'maze-test': MazeData(
            #                 dataset_name="Maze",
            #                 dataset_path=FLAGS.test_data_path_medium,
            #                 mode = 'test',
            #                 num_datapoints = num_test_datapoint_medium
            #                 )
            # }
            if FLAGS.num__size_training >2:
                num_train_datapoint_hard = eval(FLAGS.train_data_path_hard.split('N-')[1])
                # num_val_datapoint_hard = eval(FLAGS.val_data_path_hard.split('N-')[1])
                # num_test_datapoint_hard = eval(FLAGS.test_data_path_hard.split('N-')[1])
                train_dataset_hard = MazeData(
                            dataset_name="Maze",
                            dataset_path=FLAGS.train_data_path_hard,
                            mode = 'train',# 'test'
                            num_datapoints = num_train_datapoint_hard
                            )
                # validation_dataset_hard = MazeData(
                #             dataset_name="Maze",
                #             dataset_path=FLAGS.val_data_path_hard,
                #             mode = 'val',# 'test'
                #             num_datapoints = num_val_datapoint_hard
                #             )
                # extra_validation_datasets_hard = {
                #     'maze-test': MazeData(
                #                 dataset_name="Maze",
                #                 dataset_path=FLAGS.test_data_path_hard,
                #                 mode = 'test',
                #                 num_datapoints = num_test_datapoint_hard
                #                 )
                # }
        dataset = train_dataset
        metric = 'maze'
        validation_batch_size =128
        assert FLAGS.cond_mask
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
    elif FLAGS.model == 'sudoku-reverse':
        model = SudokuDenoise(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
    elif FLAGS.model == 'maze-EBM':
        model = MazeEBM(
        )
        model = DiffusionWrapper(model)
    elif FLAGS.model == 'maze-denoise':
        model = MazeDenoise(
            inp_dim = 5,
            out_dim =2
        )
    else:
        assert False

    kwargs = dict()
    if FLAGS.dataset in ['sudoku', 'sudoku_latent', 'sudoku-rrn', 'sudoku-rrn-latent']:
        kwargs['sudoku'] = True


    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 32,
        objective = 'pred_noise',  # Alternative pred_x0
        timesteps = FLAGS.diffusion_steps,  # number of steps
        sampling_timesteps = FLAGS.diffusion_steps,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper]),
        supervise_energy_landscape = FLAGS.supervise_energy_landscape,
        use_innerloop_opt = FLAGS.use_innerloop_opt,
        show_inference_tqdm = False,
        args = FLAGS,
        innerloop_opt_steps = FLAGS.innerloop_opt_steps,
        **kwargs
    )
    dataset2name = {'sudoku': 'SAT'}
    args_dict = vars(FLAGS)
    sorted_args = {k: args_dict[k] for k in sorted(args_dict)}
    data_to_hash = str(sorted_args)
    hash_value = hashlib.sha256(data_to_hash.encode()).hexdigest()[:8]   # only use the first 8 characters ? TODO
    if FLAGS.kl_coef > 0:
        kl_str = f"_kl-coef_{FLAGS.kl_coef}_kl-interval_{FLAGS.kl_interval}_kl-step_{FLAGS.kl_enable_grad_steps}_kl-max-end_{FLAGS.kl_max_end_step}_ent-coef_{FLAGS.entropy_coef}_ent-K_{FLAGS.entropy_k_nearest_neighbor}"
    else:
        kl_str = ""
    result_dir = EXP_PATH + '/results/'+ f'ds_{FLAGS.dataset}/'+f'{FLAGS.dataset}_diffsteps_{FLAGS.diffusion_steps}/{FLAGS.exp_name}{kl_str}-'+str(hash_value)
    os.makedirs(result_dir, exist_ok=True)
    
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
        train_num_steps = FLAGS.train_num_steps, # number of training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        data_workers = FLAGS.data_workers,
        amp = False,                      # turn on mixed precision
        metric = metric,
        results_folder = result_dir,
        cond_mask = FLAGS.cond_mask,
        validation_dataset = validation_dataset,
        extra_validation_datasets = extra_validation_datasets,
        extra_validation_every_mul = extra_validation_every_mul,
        save_and_sample_every = FLAGS.save_and_sample_every,
        evaluate_first = FLAGS.evaluate,  # run one evaluation first
        latent = FLAGS.latent,  # whether we are doing reasoning in the latent space
        autoencode_model = autoencode_model,
        exp_hash_code = hash_value,
        train_dataset_medium = train_dataset_medium,
        validation_dataset_medium = validation_dataset_medium,
        extra_validation_datasets_medium = extra_validation_datasets_medium,
        train_dataset_hard = train_dataset_hard,
        validation_dataset_hard = validation_dataset_hard,
        extra_validation_datasets_hard = extra_validation_datasets_hard,
    )

    if FLAGS.load_milestone is not None:
        trainer.load(FLAGS.load_milestone)

    trainer.train()

