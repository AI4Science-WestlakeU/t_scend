# T-SCEND_anonymous
Here is the official T-SCEND implementation. 

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
