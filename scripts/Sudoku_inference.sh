#activate the conda environment
source /opt/conda/bin/activate
conda activate tscendEnv
filename="results/checkpoint/Sudoku/Original/Naive_training.pt"

# Run the python script
## Naive inference for model base
python3 tscend_src/inference/inference_Sudoku.py --dataset sudoku --batch_size 128 --model sudoku   --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 20 --ckpt $filename --data_workers 16  --use-innerloop-opt 'True' --diffusion_steps 10 --sampling_timesteps 10 --inference_method 'diffusion_baseline' --mcts_type 'continuous' --K 0 --mcts_noise_scale 0.01 --task_difficulty 'harder' --num_batch 1 --n_seed 1
## diffusion baseline with same model forward computing budget (Random search)
python3 tscend_src/inference/inference_Sudoku.py --dataset sudoku --batch_size 1 --model sudoku   --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 20 --ckpt $filename --data_workers 16  --use-innerloop-opt 'True' --diffusion_steps 10 --sampling_timesteps 10 --inference_method 'diffusion_baseline' --mcts_type 'continuous' --K 40 --mcts_noise_scale 0.5 --task_difficulty 'harder' --num_batch 128 --n_seed 1


## MCTS expanded by gaussian noise
python3 tscend_src/inference/inference_Sudoku.py --dataset sudoku \
    --batch_size 1 \
    --exp_name "mcts_rnn_trained_model" \
    --model sudoku \
    --cond_mask True \
    --supervise-energy-landscape True \
    --innerloop_opt_steps 20 \
    --ckpt $filename \
    --data_workers 16 \
    --diffusion_steps 10 \
    --sampling_timesteps 10 \
    --inference_method 'mcts' \
    --mcts_type 'continuous' \
    --results_name 'mcts_continuous_15_1.5' \
    --task_difficulty 'harder' \
    --J_type 'energy_learned'  \
    --num_batch 128 \
    --K 40 \
    --steps_rollout 40 \
    --noise_type 'gaussian' \
    --mcts_noise_scale 0.5 \
    --exploration_weight 100