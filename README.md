# T-SCEND: Test-time Scalable MCTS-enhanced Diffusion Model
Here is the official implementation for **T-SCEND: Test-time Scalable MCTS-enhanced Diffusion Model**. 

[[arXiv](https://arxiv.org/abs/2502.01989)]

We introduce Test-time Scalable MCTS-enhanced Diffusion Model (T-SCEND), a novel framework that significantly improves diffusion model’s reasoning capabilities with better energy-based training and scaling up test-time computation.

**Trained with Maze tasks of up to 6x6, T-SCEND can generalize to solve much harder 15x15 Maze tasks, with larger test-time compute resulting in higher accuracy.:**
<a href="https://github.com/AI4Science-WestlakeU/t_scend/tree/main/assets/maze_scalability.png">
  <img src="https://raw.githubusercontent.com/AI4Science-WestlakeU/t_scend/main/assets/maze_scalability.png" align="center" width="600">
</a>

**Framework of T-SCEND:**
<a href="https://github.com/AI4Science-WestlakeU/t_scend/tree/main/assets/figure1.jpg">
  <img src="https://raw.githubusercontent.com/AI4Science-WestlakeU/t_scend/main/assets/figure1.jpg" align="center" width="800">
</a>
## 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate tscendEnv
pip install -e .
```

## 2. Dataset and checkpoints
The datasets and checkpoints can be downloaded from this [link](https://drive.google.com/drive/folders/1ZfPdkQ4DpEukOxRn6S47ADV3TXTnr6xk?usp=drive_link).

## 3. Training
To train the model, run the following command
```bash
sh scripts/Maze_train.sh
```
```bash
sh scripts/Sudoku_train.sh
```
## 4. Inference

```bash
sh scripts/Maze_inference.sh
```
```bash
sh scripts/Sudoku_inference.sh
```

## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@article{zhang2025tscend,
  title={T-SCEND: Test-time Scalable MCTS-enhanced Diffusion Model},
  author={Zhang, Tao and Pan, Jia-Shu and Feng, Ruiqi and Wu, Tailin},
  journal={arXiv preprint arXiv:2502.01989},
  year={2025}
}
```
