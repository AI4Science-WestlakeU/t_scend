source /opt/conda/bin/activate
conda activate tscendEnv

export CUDA_VISIBLE_DEVICES=1
train_data_path='./dataset/maze/Maze-traingrid_n-4_n_mazes-16500_min_length-5_max_length-20N-10219'
val_data_path='./dataset/maze/Maze-testgrid_n-6_n_mazes-1000_min_length-5_max_length-20N-837'
test_data_path='./dataset/maze/Maze-testgrid_n-7_n_mazes-1000_min_length-5_max_length-20N-888'

train_data_path_medium='./dataset/maze/Maze-traingrid_n-5_n_mazes-12500_min_length-5_max_length-20N-9394'
val_data_path_medium='None'
test_data_path_medium='None'

train_data_path_hard='./dataset/maze/Maze-traingrid_n-6_n_mazes-12500_min_length-5_max_length-20N-10295'
val_data_path_hard='None'
test_data_path_hard='None'







################## main training command ##################
# untrained model
# python3 tscend_src/train/train.py --dataset 'maze' --batch_size 64 --model maze-EBM  --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 20 --diffusion_steps 10 --exp_name "origin_SAT_naive_training" --train_num_steps -1 --save_and_sample_every 1 --save_loss_curve True --train_data_path $train_data_path --val_data_path $val_data_path --test_data_path $test_data_path --train_data_path_medium $train_data_path_medium --val_data_path_medium $val_data_path_medium --test_data_path_medium $test_data_path_medium --train_data_path_hard $train_data_path_hard --val_data_path_hard $val_data_path_hard --test_data_path_hard $test_data_path_hard  --data-workers 12 --num__size_training 3


### Naive training pipeline
python3 tscend_src/train/train.py --dataset 'maze' --batch_size 64 --model maze-EBM  --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 20 --diffusion_steps 10 --exp_name "origin_SAT_naive_training" --train_num_steps 300000 --save_and_sample_every 10000 --save_loss_curve True --train_data_path $train_data_path --val_data_path $val_data_path --test_data_path $test_data_path --train_data_path_medium $train_data_path_medium --val_data_path_medium $val_data_path_medium --test_data_path_medium $test_data_path_medium --train_data_path_hard $train_data_path_hard --val_data_path_hard $val_data_path_hard --test_data_path_hard $test_data_path_hard  --data-workers 12 --num__size_training 3



### Maximum Entropy Loss training pipeline
python3 tscend_src/train/train.py --dataset 'maze' --batch_size 64 --model maze-EBM  --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 20 --diffusion_steps 10 --exp_name "0119-origin_SAT_naive_training" --train_num_steps 300000 --save_and_sample_every 10000 --save_loss_curve True --train_data_path $train_data_path --val_data_path $val_data_path --test_data_path $test_data_path --train_data_path_medium $train_data_path_medium --val_data_path_medium $val_data_path_medium --test_data_path_medium $test_data_path_medium --train_data_path_hard $train_data_path_hard --val_data_path_hard $val_data_path_hard --test_data_path_hard $test_data_path_hard  --data-workers 12 --num__size_training 3 --kl_coef 0.001 --kl_interval 50 --kl_enable_grad_steps 1 --kl_max_end_step 8 --entropy_coef 0 --entropy_k_nearest_neighbor 1

### Negative contrastive loss training pipeline
python3 tscend_src/train/train.py --dataset 'maze' --batch_size 64 --model maze-EBM  --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 20 --diffusion_steps 10 --exp_name "origin_SAT_naive_training" --train_num_steps 300000 --save_and_sample_every 10000 --save_loss_curve True --train_data_path $train_data_path --val_data_path $val_data_path --test_data_path $test_data_path --train_data_path_medium $train_data_path_medium --val_data_path_medium $val_data_path_medium --test_data_path_medium $test_data_path_medium --train_data_path_hard $train_data_path_hard --val_data_path_hard $val_data_path_hard --test_data_path_hard $test_data_path_hard  --data-workers 12 --num__size_training 3 --neg_contrast_coef 0.1  --neg_contrast_coef_x0 0.5 --neg_contrast_coef_xt 0.01 --max_strength_permutation_x0 0.9 --max_strength_permutation_xt 0.8 --max_gap_x0 0.51 --max_gap_xt 0.5 --diverse_gap_batch True --data-workers 12 --min_gap_x0 0.12 --min_gap_xt 0.11 --min_weight_neg_contrat_x0 0.2 --min_weight_neg_contrat_xt 0.2 --monotonicity_landscape_loss True --k_min_monotonicity_landscape_x0 1.0 --k_min_monotonicity_landscape_xt 1.0 --monotonicity_landscape_fit_loss_coef_xt 0.0  --k_min_monotonicity_landscape_x0 1.0 --k_min_monotonicity_landscape_xt 1.0 --monotonicity_landscape_fit_loss_coef_xt 0.0


### Maximum Entropy Loss + Negative contrastive loss training pipeline
python3 tscend_src/train/train.py --dataset 'maze' --batch_size 64 --model maze-EBM  --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 20 --diffusion_steps 10 --exp_name "origin_SAT_naive_training" --train_num_steps 300000 --save_and_sample_every 10000 --save_loss_curve True --train_data_path $train_data_path --val_data_path $val_data_path --test_data_path $test_data_path --train_data_path_medium $train_data_path_medium --val_data_path_medium $val_data_path_medium --test_data_path_medium $test_data_path_medium --train_data_path_hard $train_data_path_hard --val_data_path_hard $val_data_path_hard --test_data_path_hard $test_data_path_hard  --data-workers 12 --num__size_training 3 --neg_contrast_coef 0.01  --neg_contrast_coef_x0 0.5 --neg_contrast_coef_xt 0.02 --max_strength_permutation_x0 0.9 --max_strength_permutation_xt 0.8 --max_gap_x0 0.51 --max_gap_xt 0.5 --diverse_gap_batch True --data-workers 12 --min_gap_x0 0.12 --min_gap_xt 0.11 --min_weight_neg_contrat_x0 0.2 --min_weight_neg_contrat_xt 0.2 --monotonicity_landscape_loss True --k_min_monotonicity_landscape_x0 1.0 --k_min_monotonicity_landscape_xt 1.0 --monotonicity_landscape_fit_loss_coef_xt 0.0  --k_min_monotonicity_landscape_x0 1.0 --k_min_monotonicity_landscape_xt 1.0 --monotonicity_landscape_fit_loss_coef_xt 0.0 --kl_coef 0.001 --kl_interval 50 --kl_enable_grad_steps 1 --kl_max_end_step 8 --entropy_coef 0 --entropy_k_nearest_neighbor 1