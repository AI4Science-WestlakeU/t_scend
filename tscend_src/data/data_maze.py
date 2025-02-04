
import sys, os
import scipy.io
import h5py
import pdb
import pickle

import torch
import numpy as np
import torch.nn as nn
from torch_geometric.data import Dataset

from typing import Tuple
from einops import rearrange, repeat
from IPython import embed
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
as_float = lambda x: float(x.item())
class MazeData(Dataset):
    '''
    x: u in Burgers' equation
    y: u_1, u_2, ..., u_t, (Nt, Nx, 1)
    '''
    def __init__(
        self,
        # basic arguments
        dataset_name="Maze",
        dataset_path=None,
        mode = 'train',# 'test'


        # specific arguments
        num_datapoints = 20000,
    ):
        # arguments
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.mode = mode
        print("Load dataset from ", self.dataset_path)
        
        # basic preprocessing
        self.num_datapoints = num_datapoints

        super(MazeData, self).__init__()

    def len(self): # must be implemented
        return self.num_datapoints

    def get(self, idx, use_normalized=True): # must be implemented
        '''
        data:
        - map: (Nx, Ny, 2)
        - goal: (Nx, Ny, 3)
        - path: (Nx, Ny, 3)

        '''
        # get id
        idx = idx%self.num_datapoints
        maze_solved = torch.tensor(np.load(self.dataset_path + '/maze_solved-{}.npy'.format(idx)),dtype=torch.long) # (Nx, Ny, 3)

        # get map, goal, path (solution)
        map = maze_solved[:,:,0]
        goal = maze_solved[:,:,1]   
        path = maze_solved[:,:,2]

        # sonvert the values to 0 and 1
        map = ((map+1)/2).long()
        goal = (goal+1).long()
        path = ((path+1)/2).long()
        mask = (goal != 0)
        # one-hot encoding
        map_one_hot = (F.one_hot(map, num_classes=2).float())*2 -1  # all elements are -1 or 1
        goal_one_hot = (F.one_hot(goal, num_classes=3).float())*2 -1
        path_one_hot = (F.one_hot(path, num_classes=2).float())*2-1

        maze_cond = torch.cat([map_one_hot, goal_one_hot], dim=-1) # (Nx, Ny, 5)
        maze_solution = path_one_hot # (Nx, Ny, 2)

        data = (
            maze_cond, # (Nx, Ny, 5)
            maze_solution, # (Nx, Ny, 2)
            mask # (Nx, Ny) bool value
        )

        return data

import torch

def reconstruct_maze_solved(maze_cond,maze_solution):
    """
    Reconstruct the `maze_solved` tensor from one-hot encoded inputs.

    Args:
    maze_cond (torch.Tensor): Tensor of shape (B, Nx, Ny, 5),
    maze_solution (torch.Tensor): Tensor of shape (B, Nx, Ny, 2).

    Returns:
        torch.Tensor: Reconstructed `maze_solved` tensor of shape (B, Nx, Ny, 3),
                      where the channels correspond to map, goal, and path.
    """
    maze_cond = normalize_last_dim(maze_cond)
    maze_solution = normalize_last_dim(maze_solution)
    map_one_hot = maze_cond[:,:,:,:2]
    goal_one_hot = maze_cond[:,:,:,2:]
    path_one_hot = maze_solution
    # Convert map_one_hot to original values (-1 for free space, 1 for obstacle)
    map_reconstructed = torch.argmax(map_one_hot, dim=-1) * 2 - 1  # Convert from [0, 1] to [-1, 1]

    # Convert goal_one_hot to original values (0 for start point, 1 for goal point, -1 for other)
    goal_reconstructed = torch.argmax(goal_one_hot, dim=-1) - 1  # Convert from [0, 1, 2] to [-1, 0, 1]

    # Convert path_one_hot to original values (-1 for no path, 1 for path)
    path_reconstructed = torch.argmax(path_one_hot, dim=-1) * 2 - 1  # Convert from [0, 1] to [-1, 1]

    # Combine the channels into a single tensor
    maze_solved = torch.stack([map_reconstructed, goal_reconstructed, path_reconstructed], dim=-1)

    return maze_solved


def calculate_path_continuity(maze_solution: torch.Tensor,maze_cond) -> torch.Tensor:
    """
    Calculate the continuity of paths represented by maze_solution without using a for loop.
    
    Args:
        maze_solution (torch.Tensor): Tensor of shape (B, H, W, 2), 
                                      where the last dimension indicates [-1 (no path), 1 (path)] in one-hot encoding.
        maze_cond (torch.Tensor): Tensor of shape (B, H, W, 5),
                                      
    Returns:
        torch.Tensor: A tensor of shape (B,) indicating the path continuity for each maze in the batch.
                      Continuity is defined as the fraction of valid path points that are connected.
    """
    # Convert one-hot to path binary mask (shape: [B, H, W])
    maze_solution = normalize_last_dim(maze_solution)
    maze_cond = normalize_last_dim(maze_cond)
    startpoint_mask = (maze_cond[..., 3] == 1).long()  # Start points have value 1 in the third channel
    endpoint_mask = (maze_cond[..., 4] == 1).long()  # End points have value 1 in the fourth channel [B, H, W]
    path_mask = (maze_solution[..., 1] == 1).long()  # Path points have value 1 in the second channel [B, H, W]
    num_path_points = (maze_solution[...,1]==1).sum(dim=(1, 2))  # Total number of path points per maze, [B,]

    continuity_scale = (num_path_points - 2)/num_path_points #[B,]
    continuity_mask = continuity_scale > 0
    continuity_scale[continuity_scale <= 0] = 1.0
    # 

    # Define all shifts for 4-connectivity in one operation
    rolled_up = torch.roll(path_mask, shifts=-1, dims=1)   # Shift up
    rolled_down = torch.roll(path_mask, shifts=1, dims=1)  # Shift down
    rolled_left = torch.roll(path_mask, shifts=-1, dims=2)  # Shift left
    rolled_right = torch.roll(path_mask, shifts=1, dims=2)  # Shift right

    # Find connected points
    connected_up = path_mask * rolled_up
    connected_down = path_mask * rolled_down
    connected_left = path_mask * rolled_left
    connected_right = path_mask * rolled_right

    # Combine all connections
    total_connections = ((connected_up + connected_down + connected_left + connected_right)/2).long()
    total_connections = total_connections.bool().long() # [B, H, W]
    path_end_success = 1-((total_connections * endpoint_mask) + (total_connections * startpoint_mask)).sum(dim=(1, 2)).bool().long()# [B,] , 1 for success to arrive the end point starting from start point

    # Sum connections per batch
    total_connected_points = total_connections.sum(dim=(1, 2)) # Divide by 2 to avoid double-counting
    total_path_points = path_mask.sum(dim=(1, 2))  # Total number of path points per maze

    # Calculate continuity
    continuity = total_connected_points / total_path_points
    continuity[total_path_points == 0] = 0  # Handle mazes with no path points

    continuity = continuity/continuity_scale

    return continuity,continuity_mask,path_end_success

def calculate_path_conformity(maze_cond: torch.Tensor, maze_solution: torch.Tensor) -> torch.Tensor:
    """
    Calculate the conformity of paths in maze_solution with respect to maze_cond.
    Conformity is the proportion of path points that do not overlap with walls.

    Args:
        maze_cond (torch.Tensor): Tensor of shape (B, H, W, 5), where the first two channels represent the map.
                                  Channel 0: free space (-1 or 1).
                                  Channel 1: obstacles (wall, -1 or 1).
        maze_solution (torch.Tensor): Tensor of shape (B, H, W, 2), where the second channel indicates the path
                                       (-1 for no path, 1 for path).

    Returns:
        torch.Tensor: A tensor of shape (B,), indicating the conformity for each case in the batch.
    """
    # Extract the "wall" channel from maze_cond (second channel in map)
    maze_cond = normalize_last_dim(maze_cond)
    maze_solution = normalize_last_dim(maze_solution)
    wall_mask = maze_cond[:, :, :, 1]  # Shape: (B, H, W)

    # Extract the "path" channel from maze_solution (second channel in path)
    path_mask = maze_solution[:, :, :, 1]  # Shape: (B, H, W)

    # Valid path points: path == 1 and wall != -1
    valid_path_points = (path_mask == 1) & (wall_mask != -1)  # Boolean mask of valid path points

    # Total path points: where path == 1
    total_path_points = (path_mask == 1).sum(dim=(1, 2))  # Sum over H and W for each batch

    # Valid path points count
    valid_path_count = valid_path_points.sum(dim=(1, 2))  # Sum over H and W for each batch

    # Calculate conformity: valid_path_count / total_path_points
    conformity = valid_path_count / total_path_points
    conformity[total_path_points == 0] = 0  # Handle cases where no path exists

    return conformity

@torch.no_grad()
def maze_accuracy(maze_cond,
                    maze_solution,
                    mask,
                    label,
                    energy_sample=None, 
                    energy_gd= None) -> dict[str, float]:
    '''
        args:
        - maze_cond: (B, Nx, Ny, 5)
        - maze_solution: (B, Nx, Ny, 2)
        - mask: (B, Nx, Ny)
        - label: (B, Nx, Ny, 2)
        - energy_sample: (B,)
        - energy_gd: (B,)
    '''
    maze_cond = normalize_last_dim(maze_cond)
    maze_solution = normalize_last_dim(maze_solution)
    mask = normalize_last_dim(mask)
    label = normalize_last_dim(label)
    path_continuity,continuity_mask,path_end_success = calculate_path_continuity(maze_solution=maze_solution, maze_cond=maze_cond)
    path_conformity = calculate_path_conformity(maze_cond, maze_solution)
    accuracy = (maze_solution == label).float().mean()
    pred_class = torch.argmax(maze_solution, dim=-1)  # Predicted class (shape [B, ...])
    label_class = torch.argmax(label, dim=-1)  # Ground truth class (shape [B, ...])
    
    # Flatten tensors to 1D
    pred_class_flat = pred_class.view(-1).cpu().numpy()
    label_class_flat = label_class.view(-1).cpu().numpy()
    
    # Calculate accuracy
    accuracy = accuracy_score(label_class_flat, pred_class_flat)
    
    # Calculate precision, recall, and F1 for class 1 (positive class is class 1)
    path_precision = precision_score(label_class_flat, pred_class_flat, pos_label=1)  # class 1 is labeled as 0
    path_recall = recall_score(label_class_flat, pred_class_flat, pos_label=1)      # class 1 is labeled as 0
    path_f1 = f1_score(label_class_flat, pred_class_flat, pos_label=1)  
    path_length = (maze_solution[...,1]==1).sum()/(maze_solution.shape[0])
    path_length_GD = (label[...,1]==1).sum()/(label.shape[0])
    rate_success = ((path_continuity > 1-1e-5) & (path_conformity > 1-1e-5) & continuity_mask & path_end_success)
    try:
        energy_consitency = (
        torch.logical_and(energy_sample <= energy_gd, rate_success).float() +
        torch.logical_and(energy_sample > energy_gd, torch.logical_not(rate_success)).float()
        )
        energy_consitency = energy_consitency.mean()
        return {
            'accuracy': accuracy,
            'path_precision': path_precision,
            'path_recall': path_recall,
            'path_f1': path_f1,
            'path_continuity': as_float(path_continuity.mean()),
            'path_conformity': as_float(path_conformity.mean()),
            'path_length': as_float(path_length),
            'path_length_GD': as_float(path_length_GD),
            'path_end_success': as_float(path_end_success.float().mean()),
            'rate_success': as_float(rate_success.float().mean()),
            'energy_consistency': as_float(energy_consitency),
            'mean of sample energy': as_float(energy_sample.mean()),
            'mean of gd energy': as_float(energy_gd.mean()),
    }
    except:
        return {
            'accuracy': accuracy,
            'path_precision': path_precision,
            'path_recall': path_recall,
            'path_f1': path_f1,
            'path_continuity': as_float(path_continuity.mean()),
            'path_conformity': as_float(path_conformity.mean()),
            'path_length': as_float(path_length),
            'path_length_GD': as_float(path_length_GD),
            'path_end_success': as_float(path_end_success.float().mean()),
            'rate_success': as_float(rate_success.float().mean()),
        }
def maze_accuracy_batch(maze_cond, maze_solution, mask, label, energy_sample=None, energy_gd=None):
    """
    Vectorized implementation of maze_accuracy.

    Args:
        maze_cond (torch.Tensor): Shape (B, Nx, Ny, 5)
        maze_solution (torch.Tensor): Shape (B, Nx, Ny, 2)
        mask (torch.Tensor): Shape (B, Nx, Ny)
        label (torch.Tensor): Shape (B, Nx, Ny, 2)
        energy_sample (torch.Tensor): Shape (B,), optional
        energy_gd (torch.Tensor): Shape (B,), optional

    Returns:
        dict[str, torch.Tensor]: Dictionary with metrics as tensors of shape [B,].
    """
    def normalize_last_dim(tensor):
        return torch.where(tensor > 0, 1, -1).long()

    maze_cond = normalize_last_dim(maze_cond)
    maze_solution = normalize_last_dim(maze_solution)
    mask = normalize_last_dim(mask)
    label = normalize_last_dim(label)

    path_continuity, continuity_mask, path_end_success = calculate_path_continuity(maze_solution=maze_solution, maze_cond=maze_cond)
    path_conformity = calculate_path_conformity(maze_cond, maze_solution)

    # Accuracy calculation
    accuracy = (maze_solution == label).float().mean(dim=(1, 2, 3))

    # Prediction and label class extraction
    pred_class = torch.argmax(maze_solution, dim=-1)  # Shape (B, Nx, Ny)
    label_class = torch.argmax(label, dim=-1)        # Shape (B, Nx, Ny)

    # Flatten for metrics computation
    pred_class_flat = pred_class.view(maze_solution.shape[0], -1).cpu().numpy()
    label_class_flat = label_class.view(label.shape[0], -1).cpu().numpy()

    # Compute precision, recall, and F1 for each batch
    path_precision = torch.tensor([precision_score(label_class_flat[b], pred_class_flat[b], pos_label=1) for b in range(maze_solution.shape[0])])
    path_recall = torch.tensor([recall_score(label_class_flat[b], pred_class_flat[b], pos_label=1) for b in range(maze_solution.shape[0])])
    path_f1 = torch.tensor([f1_score(label_class_flat[b], pred_class_flat[b], pos_label=1) for b in range(maze_solution.shape[0])])

    path_length = (maze_solution[..., 1] == 1).sum(dim=(1, 2)).float()  # Sum of path points per maze
    path_length_GD = (label[..., 1] == 1).sum(dim=(1, 2)).float()  # Sum of ground truth path points per maze


    rate_success = ((path_continuity > 1 - 1e-5) & (path_conformity > 1 - 1e-5) & continuity_mask & path_end_success)

    if energy_sample is not None and energy_gd is not None:
        energy_consistency = (
            torch.logical_and(energy_sample <= energy_gd, rate_success).float() +
            torch.logical_and(energy_sample > energy_gd, torch.logical_not(rate_success)).float()
        )
    else:
        energy_consistency = torch.tensor([0.0] * maze_solution.shape[0])

    return {
        'accuracy': accuracy,
        'path_precision': path_precision,
        'path_recall': path_recall,
        'path_f1': path_f1,
        'path_continuity': path_continuity,
        'path_conformity': path_conformity,
        'path_length': path_length,
        'path_length_GD': path_length_GD,
        'path_end_success': path_end_success.float(),
        'rate_success': rate_success.float(),
        'energy_consistency': energy_consistency if energy_sample is not None else None
    }

def normalize_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize the last dimension of a tensor:
    - Elements > 0 are set to 1.
    - Elements < 0 are set to -1.
    - Elements == 0 remain unchanged (optional, adjust as needed).

    Args:
        tensor (torch.Tensor): Input tensor of any shape.

    Returns:
        torch.Tensor: Tensor with normalized last dimension and type `long`.
    """
    # Normalize the last dimension
    normalized_tensor = torch.where(tensor > 0, 1, -1)

    # Convert to long type
    return normalized_tensor.long()

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_maze(maze_cond,maze_solution, num_plot=16, file_path='maze_plot.png'):
    """
    Plot a grid of `num_plot` maze examples and save the figure.
    
    Args:
        maze_cond (torch.Tensor): Tensor of shape (B, H, W, 5), 
        maze_solution (torch.Tensor): Tensor of shape (B, H, W, 2),
        num_plot (int): The number of mazes to plot.
        file_path (str): The path to save the plot.
    """
    maze_cond = maze_cond.cpu()
    maze_solution = maze_solution.cpu()
    maze_solved = reconstruct_maze_solved(maze_cond,maze_solution)
    # Ensure the input is a torch tensor and convert to numpy for plotting
    if isinstance(maze_solved, torch.Tensor):
        maze_solved = maze_solved.cpu().numpy()  # Convert to numpy array

    B, H, W, _ = maze_solved.shape

    # Calculate grid size (e.g., for num_plot=16, create a 4x4 grid)
    grid_size = int(np.ceil(np.sqrt(num_plot)))

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    # Flatten axes to iterate over them
    if num_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot up to num_plot mazes
    # Plot up to num_plot mazes
    for i in range(num_plot):
        # Get the corresponding maze (if i exceeds the available samples, break)
        if i >= B:
            break

        # Create an empty image with white background (H, W, 3)
        maze_img = np.ones((H, W, 3), dtype=np.float32)
        # 4. Plot the free space in white (map == 1 in maze_cond)
        maze_img[maze_cond[i, :, :, 0] == 1] = [1, 1, 1]  # Free space (white)

        # 1. Plot the walls in black (map == -1 in maze_cond)
        maze_img[maze_cond[i, :, :, 1] == -1] = [0, 0, 0]  # Walls

        # 5. Plot the path in blue (path == 1 in maze_solution)
        maze_img[maze_solution[i, :, :, 1] == 1] = [0, 0, 1]  # Path

        # 2. Plot the start points in green (start == 1 in maze_cond)
        maze_img[maze_cond[i, :, :, 3] == 1] = [0, 1, 0]  # Start point

        # 3. Plot the goal points in red (goal == 1 in maze_cond)
        maze_img[maze_cond[i, :, :, 4] == 1] = [1, 0, 0]  # Goal point

        

        # Plot the maze on the appropriate subplot
        axes[i].imshow(maze_img)
        axes[i].axis('off')  # Hide axis

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
if __name__ == '__main__':
    maze = MazeData(dataset_path='./dataset/Maze_testgrid_n-10_n_mazes-20000_min_length-15_max_length-20', num_datapoints=20000)
    data = maze.get(0)
    maze_cond, maze_solution, mask = data
    label = maze_solution
    summary = maze_accuracy(maze_cond.unsqueeze(0), maze_solution.unsqueeze(0), mask.unsqueeze(0), label.unsqueeze(0))
    print(summary)
