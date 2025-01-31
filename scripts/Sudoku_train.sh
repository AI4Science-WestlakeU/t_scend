#activate the conda environment
source /opt/conda/bin/activate
conda activate tscendEnv

################## main training command ##################
### Naive training pipeline
python3 X_src/train/train.py --dataset 'sudoku' --batch_size 64 --model sudoku   --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 20 --diffusion_steps 10 --exp_name "origin_SAT_naive_training" --train_num_steps 300000 --save_and_sample_every 10000 --save_loss_curve True

### Maximum Entropy Loss training pipeline
python3 X_src/train/train.py --dataset 'sudoku' --batch_size 64 --model sudoku   --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 0 --diffusion_steps 10 --exp_name "origin_SAT_KL_training_entropy" --train_num_steps 300000 --save_and_sample_every 10000 --save_loss_curve True --kl_coef 0.001 --kl_interval 50 --kl_enable_grad_steps 1 --kl_max_end_step 8 --entropy_coef 0.001 --entropy_k_nearest_neighbor 1 

### Negative contrastive loss training pipeline
python3 X_src/train/train.py --dataset 'sudoku' --batch_size 64 --model sudoku   --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 20 --diffusion_steps 10 --exp_name "origin_SAT_neg_contrast_training_x0" --train_num_steps 300000 --save_and_sample_every 10000 --save_loss_curve True --neg_contrast_coef 1.0  --neg_contrast_coef_x0 0.5 --neg_contrast_coef_xt 0.0 --max_strength_permutation_x0 0.7 --max_strength_permutation_xt 0.8 --max_gap_x0 0.51 --max_gap_xt 0.5 --diverse_gap_batch True --data-workers 12 --min_gap_x0 0.12 --min_gap_xt 0.11 --min_weight_neg_contrat_x0 0.2 --min_weight_neg_contrat_xt 0.2

### Maximum Entropy Loss + Negative contrastive loss training pipeline
python3 X_src/train/train.py --dataset 'sudoku' --batch_size 64 --model sudoku   --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 0 --diffusion_steps 10 --exp_name "origin_SAT_entropy_neg_contrast" --train_num_steps 300000 --save_and_sample_every 10000 --save_loss_curve True --kl_coef 0.001 --kl_interval 50 --kl_enable_grad_steps 1 --kl_max_end_step 8 --entropy_coef 0.001 --entropy_k_nearest_neighbor 1 --neg_contrast_coef 1.0  --neg_contrast_coef_x0 0.5 --neg_contrast_coef_xt 0.0 --max_strength_permutation_x0 0.7 --max_strength_permutation_xt 0.8 --max_gap_x0 0.51 --max_gap_xt 0.5 --diverse_gap_batch True --data-workers 12 --min_gap_x0 0.12 --min_gap_xt 0.11 --min_weight_neg_contrat_x0 0.2 --min_weight_neg_contrat_xt 0.2
