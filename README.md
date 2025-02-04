# T-SCEND: Test-time Scalable MCTS-enhanced Diffusion Model
Here is the official T-SCEND implementation. 

[arXiv]()

We introduce Test-time Scalable MCTS-enhanced Diffusion Model (T-SCEND), a novel framework that significantly improves diffusion model’s reasoning capabilities with better energy-based training and scaling up test-time computation.

**Visualizations of Maze training data and solutions generated by hMCTS denoising of our T-SCEND framework:**
<a href="https://github.com/AI4Science-WestlakeU/t_scend/tree/main/assets/maze_plot_train_hmcts_00.jpg">
  <img src="https://raw.githubusercontent.com/AI4Science-WestlakeU/t_scend/main/assets/maze_plot_train_hmcts_00.jpg" align="center" width="500">
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
