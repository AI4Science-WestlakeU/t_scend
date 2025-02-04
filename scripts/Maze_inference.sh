#activate the conda environment
# echo "Start running the script"
# source /opt/conda/bin/activate
# conda activate tscendEnv
export CUDA_VISIBLE_DEVICES=0

filename="results/checkpoint/Maze/Original/model-9.pt"
train_data_path="./dataset/maze/Maze-train-grid_n-4_n_mazes-50000_min_length-5_max_length-20N-31274"
val_data_path="./dataset/maze/Maze-test-grid_n-4_n_mazes-5000_min_length-5_max_length-20N-3081"
# test_data_path="./dataset/Maze-test-grid_n-5_n_mazes-3000_min_length-5_max_length-20N-2256"
# test_data_path="./dataset/Maze-test-grid_n-6_n_mazes-3000_min_length-5_max_length-20N-2498"

# test_data_path="./dataset/maze/Maze-testgrid_n-7_n_mazes-1000_min_length-5_max_length-20N-888"
# maze_grid_size=7

# test_data_path="./dataset/maze/Maze-testgrid_n-8_n_mazes-1000_min_length-5_max_length-20N-898"
# maze_grid_size=8

# test_data_path="./dataset/maze/Maze-testgrid_n-9_n_mazes-1000_min_length-5_max_length-20N-920"
# maze_grid_size=9

# test_data_path="./dataset/maze/Maze-testgrid_n-10_n_mazes-1000_min_length-5_max_length-20N-948"
# maze_grid_size=10

# test_data_path="./dataset/maze/Maze-testgrid_n-11_n_mazes-1000_min_length-5_max_length-20N-940"
# maze_grid_size=11

# test_data_path="./dataset/maze/Maze-testgrid_n-12_n_mazes-1000_min_length-5_max_length-20N-960"
# maze_grid_size=12

# test_data_path="./dataset/maze/Maze-testgrid_n-13_n_mazes-1000_min_length-5_max_length-20N-968"
# maze_grid_size=13

# test_data_path="./dataset/maze/Maze-testgrid_n-14_n_mazes-1000_min_length-5_max_length-20N-965"
# maze_grid_size=14

test_data_path="./dataset/maze/Maze-testgrid_n-15_n_mazes-1000_min_length-5_max_length-20N-975"
maze_grid_size=15

# Run the python script
## Naive inference for model base

python3 tscend_src/inference/inference_Maze.py --dataset maze --batch_size 128 --model maze-EBM   --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 20 --ckpt $filename --data_workers 16  --use-innerloop-opt 'True' --diffusion_steps 10 --sampling_timesteps 10 --inference_method 'diffusion_baseline' --mcts_type 'continuous' --K 0 --mcts_noise_scale 0.5 --task_difficulty 'harder' --num_batch 1 --n_seed 1 --train_data_path $train_data_path --val_data_path $val_data_path --test_data_path $test_data_path --maze_grid_size $maze_grid_size
## diffusion baseline with same model forward computing budget (Random search)
python3 tscend_src/inference/inference_Maze.py --dataset maze --batch_size 1 --model maze-EBM   --cond_mask True  --supervise-energy-landscape True   --innerloop_opt_steps 20 --ckpt $filename --data_workers 16  --use-innerloop-opt 'True' --diffusion_steps 10 --sampling_timesteps 10 --inference_method 'diffusion_baseline' --mcts_type 'continuous' --K 40 --mcts_noise_scale 0.5 --task_difficulty 'harder' --num_batch 128 --n_seed 1 --train_data_path $train_data_path --val_data_path $val_data_path --test_data_path $test_data_path --maze_grid_size $maze_grid_size


# for K in 80 160 320 # $(awk 'BEGIN {for (i=0.1; i<1.0; i+=0.1) printf "%.1f ", i}')
for K in 40
do
    ## MCTS expanded by gaussian noise
    python3 tscend_src/inference/inference_Maze.py --dataset maze \
        --batch_size 1 \
        --exp_name "mcts_rnn_trained_model" \
        --model maze-EBM \
        --cond_mask True \
        --supervise-energy-landscape True \
        --innerloop_opt_steps 20 \
        --ckpt $filename \
        --train_data_path $train_data_path \
        --val_data_path $val_data_path \
        --test_data_path $test_data_path \
        --maze_grid_size $maze_grid_size \
        --data_workers 16 \
        --diffusion_steps 10 \
        --sampling_timesteps 10 \
        --inference_method 'mcts' \
        --mcts_type 'continuous' \
        --results_name 'mcts_continuous_15_1.5' \
        --task_difficulty 'harder' \
        --J_type 'energy_learned'  \
        --num_batch 128 \
        --K $K \
        --steps_rollout $K \
        --noise_type 'gaussian' \
        --mcts_noise_scale 0.5 \
        --exploration_weight 100
done