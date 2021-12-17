import torch
import math

def compute_asm_loss(ypred, ytrue, device):
    """ 
    bone constraint loss function 
     
    input:
        ypred = predicted 3D hand skeletonization - torch.Size([batch_size x 21 x 3])
        ytrue = true 3D hand skeletonization - torch.Size([batch_size x 21 x 3])
         
    output:
        L_len = bone length loss - torch.Size([1])
        L_dir = bone direction loss  - torch.Size([1])
    """

    # asm loss
    num_comp = 6
    ytrue_flatten = torch.reshape(ytrue, (ytrue.shape[0], -1))
    ypred_flatten = torch.reshape(ypred, (ypred.shape[0], -1))

    sample_mean = torch.mean(ytrue_flatten, 0)

    U, S, V = torch.pca_lowrank(ytrue_flatten)

    # compute b
    diff = ytrue_flatten - sample_mean
    b = torch.matmul(diff, V)
    # clamp b
    threshold = 3 * torch.abs(S) / math.sqrt(ytrue.shape[0] - 1)
    b = torch.clamp(b, -threshold, threshold)
    # generate asm ground truth
    y_new = sample_mean + torch.matmul(b, torch.transpose(V, 0, 1))
    # compute asm loss
    L_asm = torch.square(ypred_flatten - y_new)
    
    return torch.sum(L_asm)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    