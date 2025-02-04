from abc import ABC, abstractmethod
from collections import defaultdict
import math
import torch
from torch.nn import functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from einops import rearrange, reduce,repeat
from tscend_src.utils.utils import set_seed,p
# from tscend_src.model.ired.diffusion_lib.denoising_diffusion_pytorch_1d import sudoku_score
def sudoku_score(pred: torch.Tensor) -> bool:
    valid_mask = torch.ones_like(pred)

    pred_sum_axis_1 = pred.sum(dim=1, keepdim=True)
    pred_sum_axis_2 = pred.sum(dim=2, keepdim=True)

    # Use the sum criteria from the SAT-Net paper
    axis_1_mask = (pred_sum_axis_1 == 36)
    axis_2_mask = (pred_sum_axis_2 == 36)

    valid_mask = valid_mask * axis_1_mask.float() * axis_2_mask.float()

    valid_mask = valid_mask.view(-1, 3, 3, 3, 3)
    grid_mask = pred.view(-1, 3, 3, 3, 3).sum(dim=(2, 4), keepdim=True) == 36

    valid_mask = valid_mask * grid_mask.float()

    return valid_mask.mean()

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
class NodeMazeContinuous(Node):
    def __init__(self, state,noise_scale=0.1,topk=10,K=10,noise_K = None,pred_noise=None):
        self.state = state
        self.noise_scale = noise_scale # the scale of the noise
        self.topk = topk # the number of the topk actions to be selected
        self.K = K # number of branches
        self.noise_K = noise_K # K noise to expand the tree
        self.pred_noise = pred_noise # the noise of the prediction
        

    def find_children(self,diffusion):
        if self.is_terminal():
            return set()
        batch_size= self.state['xt'].shape[0]
        inp = self.state['inp']
        xt = self.state['xt']
        mask = self.state['mask']
        cond = self.state['inp']
        step_now = self.state['step_now']
        time_now = self.state['time']
        time_cond = torch.full((batch_size,), time_now, device=xt.device, dtype=torch.long)
        if self.state['noise_K'] is not None:
            self.state['noise_K'] = torch.randn(batch_size, self.K, *xt.shape[1:], device=xt.device)
            self.noise_K = self.state['noise_K']
        # greedy policy to get all children children, topk probablity actions worked as the children
        node_list = []
        for i in range(self.K):
            state_next = self.state.copy()
            if self.pred_noise is not None: # if root node
                alpha = diffusion.alphas_cumprod[time_now]
                alpha_next = diffusion.alphas_cumprod[time_now-1]
                sigma = diffusion.args.mcts_noise_scale * ((1. - alpha / alpha_next) * (1. - alpha_next) / (1. - alpha)).sqrt()
                assert (1. - alpha_next - sigma**2)<0, "mcts_noise_scale is too large"
                
                c = (1. - alpha_next - sigma**2).sqrt()
                state_next['xt'] = state_next['xt'] + self.noise_K[i]*sigma
                if mask is not None:
                    state_next['xt'] = apply_mask_maze(state_next['xt'], mask)
                max_val = extract(diffusion.sqrt_alphas_cumprod, time_cond-1, inp.shape)[0, 0] * 1.0
                max_val_xt = state_next['xt'].abs().max()
                scale_val = max_val / max_val_xt
                state_next['xt'] = state_next['xt'] * scale_val
                state_next['step_now'] = step_now + 1 
                state_next['time'] = int(diffusion.times_diffusion[state_next['step_now']])
                state_next["noise_K"] = None
                node_k = NodeMazeContinuous(state_next, K=self.K,noise_K=self.noise_K)
            else:
                with torch.enable_grad():
                    model_pred = diffusion.model_predictions(inp, xt, time_cond, clip_x_start=True)
                    pred_noise, x_start = model_pred.pred_noise, model_pred.pred_x_start
                alpha = diffusion.alphas_cumprod[time_now]
                alpha_next = diffusion.alphas_cumprod[time_now-1]
                sigma = diffusion.args.mcts_noise_scale * ((1. - alpha / alpha_next) * (1. - alpha_next) / (1. - alpha)).sqrt()
                assert (1. - alpha_next - sigma**2)<0, "mcts_noise_scale is too large"
                c = (1. - alpha_next - sigma**2).sqrt()

                state_next['xt'] = x_start * alpha_next.sqrt() + c * pred_noise + self.noise_K[i]*sigma
                state_next['step_now'] = step_now + 1 
                state_next['time'] = int(diffusion.times_diffusion[state_next['step_now']])
                if mask is not None:
                    state_next['xt'] = apply_mask_maze(state_next['xt'], mask)
                max_val = extract(diffusion.sqrt_alphas_cumprod, time_cond-1, inp.shape)[0, 0] * 1.0
                max_val_xt = state_next['xt'].abs().max()
                scale_val = max_val / max_val_xt
                state_next['xt'] = state_next['xt'] * scale_val
                state_next["noise_K"] = None
                node_k = NodeMazeContinuous(state_next, K=self.K,noise_K=self.noise_K)
            node_list.append(node_k)
        return node_list

    def find_children_parallel(self, diffusion):
        if self.is_terminal():
            return set()
        
        batch_size = self.state['xt'].shape[0]
        inp = self.state['inp']
        xt = self.state['xt']
        mask = self.state['mask'] 
        step_now = self.state['step_now']
        time_now = self.state['time']
        
        time_cond = torch.full((batch_size,), time_now, device=xt.device, dtype=torch.long)
        
        alpha = diffusion.alphas_cumprod[time_now]
        alpha_next = diffusion.alphas_cumprod[int(diffusion.times_diffusion[step_now+1])]
        sigma = ((1. - alpha / alpha_next) * (1. - alpha_next) / (1. - alpha)).sqrt()
        if diffusion.args.noise_type == 'gaussian':
            sigma = diffusion.args.mcts_noise_scale * sigma
        elif diffusion.args.noise_type == 'permutation':
            sigma = 0.0 # No noise for permutation noise

        # Forward pass only once
        if self.pred_noise is not None:
            # Base state without noise
            xt_base = xt
        else:
            model_pred = diffusion.model_predictions(inp, xt, time_cond, clip_x_start=True)
            pred_noise, x_start = model_pred.pred_noise, model_pred.pred_x_start
            if (1. - alpha_next - sigma**2)<0:
                import pdb; pdb.set_trace()
                f"mcts_noise_scale is too large.{(1. - alpha_next - sigma**2)} is invalid"
            c = (1. - alpha_next - sigma**2).sqrt()
            xt_base = x_start * alpha_next.sqrt() + c * pred_noise # [B, HWC]
        if diffusion.args.noise_type == 'gaussian':
            xt_next = xt_base + self.noise_K * sigma # [BK, HWC]
        elif diffusion.args.noise_type == 'permutation':
            assert diffusion.args.mcts_noise_scale < 1.0, "mcts_noise_scale must be less than or equal to 1.0 for permutation noise"
            xt_base_repeat = xt_base.repeat(self.K, 1,1,1) # [BK, HWC]
            noise_schedule = diffusion.args.mcts_noise_scale * ((1.1-alpha)**2) # TODO to tune 
            xt_next = random_permute_last_dim(xt_base_repeat, p=noise_schedule) # [BK, HWC]
        else:
            raise ValueError(f"Invalid noise type: {diffusion.args.noise_type}")
        if mask is not None:
            xt_next = apply_mask_maze(xt_next, mask)
        # Clamp values
        max_val = extract(diffusion.sqrt_alphas_cumprod, time_cond-1, xt_next.shape)[0, 0] * 1.0
        max_val_xt = xt_next.abs().max(dim=-1, keepdim=True).values
        scale_val = max_val / max_val_xt
        xt_next = xt_next * scale_val

        # Create nodes for each branch using broadcasting
        node_list = []
        for i in range(self.K):
            state_next = self.state.copy()
            state_next['xt'] = xt_next[[i]]
            state_next['step_now'] = step_now + 1 
            state_next['time'] = int(diffusion.times_diffusion[state_next['step_now']])
                
            node_k = NodeMazeContinuous(state_next, K=self.K, noise_K=self.noise_K)
            node_list.append(node_k)

        return node_list

        # Add verification code to compare implementations

    def verify_implementations(self, diffusion):
        # Run serial version
        start_time = time.time()
        serial_results = self.find_children(diffusion)
        serial_time = time.time() - start_time

        # Run parallel version  
        start_time = time.time()
        parallel_results = self.find_children_parallel(diffusion)
        parallel_time = time.time() - start_time

        # Compare results
        for s_node, p_node in zip(serial_results, parallel_results):
            if not torch.allclose(s_node.state['xt'], p_node.state['xt']):
                print("Warning: Results differ between implementations!")
                return

        print(f"Serial implementation time: {serial_time:.4f}s")
        print(f"Parallel implementation time: {parallel_time:.4f}s") 
        print(f"Speedup: {serial_time/parallel_time:.2f}x")
        return parallel_results

    def find_random_child(self,diffusion):
        if self.is_terminal():
            return set()
        batch_size= self.state['xt'].shape[0]
        inp = self.state['inp']
        xt = self.state['xt']
        mask = self.state['mask']
        cond = self.state['inp']
        step_now = self.state['step_now']
        time_now = self.state['time']
        
        i = random.randint(0,self.K)
        time_cond = torch.full((batch_size,), time_now, device=xt.device, dtype=torch.long)
        state_next = self.state.copy()
        if self.prediction_noise is not None: # if root node
            alpha = diffusion.alphas_cumprod[time_now]
            alpha_next = diffusion.alphas_cumprod[time_now-1]
            sigma = diffusion.args.mcts_noise_scale * ((1. - alpha / alpha_next) * (1. - alpha_next) / (1. - alpha)).sqrt()
            assert (1. - alpha_next - sigma**2)<0, "mcts_noise_scale is too large"
            c = (1. - alpha_next - sigma**2).sqrt()
            
            state_next['xt'] = state_next['xt'] + self.noise_K[i]*sigma
            state_next['time_now'] = time_now - 1
            state_next['step_now'] = step_now + 1
            if mask is not None:
                state_next['xt'] = apply_mask_maze(state_next['xt'], mask)
            max_val = extract(diffusion.sqrt_alphas_cumprod, time_cond-1, x_start.shape)[0, 0] * 1.0
            state_next['xt'] = torch.clamp(state_next['xt'],-max_val,max_val)
            node_k = NodeMazeContinuous(state_next, K=self.K,noise_K=self.noise_K)
        else:
            with torch.enable_grad():
                model_pred = self.model_predictions(inp, xt, time_now, clip_x_start=True)
                pred_noise, x_start = model_pred.pred_noise, model_pred.pred_x_start
            alpha = diffusion.alphas_cumprod[time_now]
            alpha_next = diffusion.alphas_cumprod[time_now-1]
            # alpha_next = diffusion.alphas_cumprod[0]
            sigma = diffusion.args.mcts_noise_scale * ((1. - alpha / alpha_next) * (1. - alpha_next) / (1. - alpha)).sqrt()
            assert (1. - alpha_next - sigma**2)<0, "mcts_noise_scale is too large"
            c = (1. - alpha_next - sigma**2).sqrt()

            state_next['xt'] = x_start * alpha_next.sqrt() + c * pred_noise + self.noise_K[i]*sigma
            state_next['time_now'] = time_now - 1
            # state_next['time_now'] = 0
            state_next['step_now'] = step_now + 1
            if mask is not None:
                state_next['xt'] = apply_mask_maze(state_next['xt'], mask)
            max_val = extract(diffusion.sqrt_alphas_cumprod, time_cond-1, x_start.shape)[0, 0] * 1.0
            state_next['xt'] = torch.clamp(state_next['xt'],-max_val,max_val)
            node_k = NodeMazeContinuous(state_next, K=self.K,noise_K=self.noise_K)
        return node_k
        


    def is_terminal(self):
        if self.state['mask'].float().mean() >= 1-1e-5 or self.state['time']<=0:
            return True
        return False

    def reward(self):
        '''
            return: 1 if the board is solved, 0 otherwise
        '''
        pass

    # def __hash__(self):
    #     return hash(self.state['inp'])

    # def __eq__(self, other):
    #     return self.state['inp'] == other.state['inp']
    def __hash__(self):
        return hash(tuple(self.state['xt'].cpu().numpy().flatten()))

    def __eq__(self, other):
        return torch.equal(self.state['xt'], other.state['xt'])


def plot_energy_vs_distance(inp,x0, model, max_radius, n_levels=50, n_samples_per_level=1000,results_path = None):
    """
    Plots energy vs distance from x0 using the provided model.

    Args:
        x0 (torch.Tensor): Input tensor of shape [1, ...].
        model (nn.Module): Neural network model that takes x0 or x_noised as input and outputs energy.
        max_radius (float): Maximum distance allowed between x0 and x_noised.
        n_levels (int, optional): Number of discrete distance levels. Default is 50.
        n_samples_per_level (int, optional): Number of noise samples per distance level. Default is 100.
        device (str, optional): Device to perform computations on ('cpu' or 'cuda'). Default is 'cpu'.
    """
    device = x0.device
    B = x0.shape[0]
    if B != 1:
        raise ValueError(f"Batch size B must be 1, but got B={B}")
    
    # Determine the shape of noise
    noise_shape = x0.shape[1:]  # Exclude batch dimension
    noise_dim = x0.numel() // B  # Total number of elements per sample
    
    # Prepare distance levels
    distances = np.linspace(0, max_radius, n_levels)
    
    # Containers for results
    distance_list = []
    max_energy = []
    min_energy = []
    median_energy = []
    mean_energy = []
    
    max_grad = []
    min_grad = []
    median_grad = []
    mean_grad = []
    inp = inp.repeat(n_samples_per_level, 1,1,1)
    with torch.enable_grad():
        for distance in distances:
            if distance == 0:
                # If distance is zero, x_noised is x0
                x_noised = x0.repeat(n_samples_per_level, *([1]*(x0.dim()-1)))
            else:
                # Generate random directions
                # Sample from normal distribution and normalize
                noise = torch.randn(n_samples_per_level, *noise_shape, device=device)
                noise = noise.view(n_samples_per_level, -1)
                noise = noise / noise.norm(dim=1, keepdim=True)  # Normalize to unit vectors
                noise = noise.view(n_samples_per_level, *noise_shape)
                
                # Scale noise to the desired distance
                scaled_noise = noise * distance
                
                # Add noise to x0
                x_noised = x0.repeat(n_samples_per_level, *([1]*(x0.dim()-1))) + scaled_noise
            
            # Compute actual distances to verify
            with torch.enable_grad():
                # Flatten tensors for distance computation
                x0_flat = x0.view(1, -1)
                x_noised_flat = x_noised.view(n_samples_per_level, -1)
                distances_actual = torch.norm(x_noised_flat - x0_flat, dim=1).cpu().numpy()
                
            
            # Compute energy
            # import  pdb; pdb.set_trace()
            energies,grads= model(inp,x_noised,torch.zeros(inp.shape[0]).to(device), return_both=True)
            energies = energies.squeeze(1).detach().cpu().numpy()  # Shape [n_samples_per_level]
            grads = grads.abs().mean(-1).detach().cpu().numpy()
            
            # Compute statistics
            max_e = np.max(energies)
            min_e = np.min(energies)
            median_e = np.median(energies)
            mean_e = np.mean(energies)
            
            max_g = np.max(grads)
            min_g = np.min(grads)
            median_g = np.median(grads)
            mean_g = np.mean(grads)
            
            
            # Store results
            distance_list.append(distance)
            max_energy.append(max_e)
            min_energy.append(min_e)
            median_energy.append(median_e)
            mean_energy.append(mean_e)
            
            max_grad.append(max_g)
            min_grad.append(min_g)
            median_grad.append(median_g)
            mean_grad.append(mean_g)
            
    
    # Plotting
    plt.figure(figsize=(12, 8))
    # import pdb; pdb.set_trace()
    plt.plot(distance_list, mean_energy, label='Mean Energy', color='blue')
    plt.plot(distance_list, median_energy, label='Median Energy', color='green')
    plt.plot(distance_list, max_energy, label='Max Energy', color='red')
    plt.plot(distance_list, min_energy, label='Min Energy', color='orange')
    plt.title('Energy vs Distance from x0')
    plt.xlabel('Distance')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_path)
    print(f"Results saved to {results_path}")
    plt.show()
    
    plt.figure(figsize=(12, 8))
    plt.plot(distance_list, mean_grad, label='Mean grad', color='blue')
    plt.plot(distance_list, median_grad, label='Median grad', color='green')
    plt.plot(distance_list, max_grad, label='Max grad', color='red')
    plt.plot(distance_list, min_grad, label='Min grad', color='orange')
    plt.title('Gradient vs Distance from x0')
    plt.xlabel('Distance')
    plt.ylabel('Gradient')
    plt.legend()
    plt.grid(True)
    results_path = str(results_path).split(".")[0]+"_grad.png"
    plt.savefig(results_path)
    print(f"Results saved to {results_path}")
    plt.show()

import torch
import time


def process_tensor_sudoku(x):
    """
    This function processes the input tensor x.
    The input tensor has the shape [B, 9, 9, 9].
    It sets the position of the max element in the last dimension to 1, and sets the other positions to -1.

    Parameters:
    x (torch.Tensor): The input tensor with shape [B, 9, 9, 9]

    Returns:
    torch.Tensor: The processed tensor with shape [B, 9, 9, 9]
    """
    # Find the indices of the maximum elements along the last dimension
    x = x.view(-1,9,9,9)
    max_indices = torch.argmax(x, dim=-1)
    # Create a tensor with the same shape as x, filled with -1
    result = torch.full_like(x, -1.0)
    # Create index tensors for the first three dimensions
    b_indices = torch.arange(x.shape[0]).view(-1, 1, 1)
    i_indices = torch.arange(x.shape[1]).view(1, -1, 1)
    j_indices = torch.arange(x.shape[2]).view(1, 1, -1)
    # Expand the dimension of max_indices
    max_indices = max_indices
    # Set the position of the maximum element to 1
    result[b_indices, i_indices, j_indices, max_indices] = 1.0
    
    result = rearrange(result, 'b h w c -> b (h w c)')
    return result
def random_permute_last_dim(X, p):
    """
    Randomly permutes the elements in the last dimension of X with probability p.
    Each [b, i, j] element has an independent probability p to be permuted.
    
    Args:
        X (torch.Tensor): Input tensor of shape [B, H, W, num_class].
        p (float): Probability of permuting the last dimension for each element.
    
    Returns:
        torch.Tensor: Tensor after applying random permutations.
    """
    # Ensure X is a floating tensor for sorting operations
    
    B, H, W, num_class = X.shape
    
    # Generate a mask with shape [B, H, W] where each element is True with probability p
    mask = torch.rand(B, H, W, device=X.device) < p  # [B, H, W]
    
    # Reshape X to [B*H*W, num_class] for easier processing
    X_reshaped = X.view(-1, num_class)  # [B*H*W, num_class]
    
    # Generate random scores for each element to create random permutations
    random_scores = torch.rand_like(X_reshaped)  # [B*H*W, num_class]
    
    # Get permutation indices by sorting the random scores
    perm_indices = torch.argsort(random_scores, dim=1)  # [B*H*W, num_class]
    
    # Create identity indices for cases where permutation is not applied
    identity_indices = torch.arange(num_class, device=X.device).unsqueeze(0).expand_as(perm_indices)  # [B*H*W, num_class]
    
    # Expand mask to match the permutation indices shape
    mask_flat = mask.view(-1, 1).expand(-1, num_class)  # [B*H*W, num_class]
    
    # Select permutation indices where mask is True, else use identity
    final_indices = torch.where(mask_flat, perm_indices, identity_indices)  # [B*H*W, num_class]
    
    # Gather the elements based on the final indices to get the permuted tensor
    X_permuted = torch.gather(X_reshaped, dim=1, index=final_indices)  # [B*H*W, num_class]
    
    # Reshape back to original shape [B, H, W, num_class]
    X_output = X_permuted.view(B, H, W, num_class)
    return X_output
def apply_mask_maze(x,mask):
    '''
    Apply the mask to tensor `x` such that:
    - x[i, j, k, 0] = -1 when mask[i, j, k] is True
    - x[i, j, k, 1] = 1 when mask[i, j, k] is True
    
    Args:
        x (torch.Tensor): The input tensor with shape [64, 9, 9, 2].
        mask (torch.Tensor): The mask with shape [64, 9, 9], where True/False determines which elements to update.
    
    Returns:
        torch.Tensor: The masked tensor `x_masked` with the updated values.
    '''
    if mask.shape[0] == 1:
        mask = mask.repeat(x.shape[0], 1, 1)
    x_1 = x[...,0].clone() #
    x_minus_1 = x[...,1].clone() #
    x_1[mask] = -1.0
    x_minus_1[mask] = 1.0
    x_masked = torch.stack([x_1, x_minus_1], dim=-1)

    return x_masked
    
if __name__ == '__main__':
    inp = torch.rand(2,9,9,9)
    mask = torch.rand(2,9,9,9)
    mask = mask > 0.9
    xt = torch.rand(2,9,9,9)
    state = {'inp': inp, 'xt': xt, 'mask': mask}
    node_test = NodeMazeContinuous(state)
    node_test.find_children()