import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons



#------------------------------------------------------------------------------#
# Implement some helper functions
# from original torch_cfm repo 

def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()

#------------------------------------------------------------------------------#
#general method to get samples for ODE sim 
#can be used for toys or HD case -- 
#output samples are in [bs, dim] shape and in ES. 

def get_ODE_gen_samples(flow_matcher, dset_kwargs, n, tmax, device):
    """
    Gets samples for generation (i.e. at t=t0 in cfm notation, 
                                 at t=tmax in ours)
    These are to be used along with calc trajectories... 
    """
    
    samples = torch.randn((n, dset_kwargs.working_data_dim)).type(torch.float32).to(device)
    if flow_matcher.__class__.__name__ == 'IFsConditionalFlowMatcher':
        #sample for IFs CFM 
        tmax_tensor = (torch.ones(n)*tmax).type(torch.float32).to(device)
        gamma_sqrd = flow_matcher.get_gamma_sqrd(tmax_tensor)
        if flow_matcher.ODE_type == 'scaled':
            #scaled ODE --> \Sigma_tmax = A(C + \Sigma_0)A^\top 
            alpha = flow_matcher.get_alpha(tmax_tensor) 
            scale = torch.sqrt((alpha**2)*(gamma_sqrd + flow_matcher.data_eigs[None, :]))
        else: 
            #unscaled ODE --> \Sigma_tmax = C + \Sigma_0 
            scale = torch.sqrt((gamma_sqrd + flow_matcher.data_eigs[None, :]))
        samples *= scale
        if flow_matcher.space=='IS':
            samples = torch.einsum('ij, bjk -> bik', flow_matcher.W, samples.unsqueeze(-1)).squeeze(-1)
    else: 
        #sample for OT CFM 
        if dset_kwargs.working_data_dim != dset_kwargs.dims_to_keep: 
            scale = torch.cat([torch.ones(dset_kwargs.dims_to_keep), \
                                   torch.ones(dset_kwargs.working_data_dim - dset_kwargs.dims_to_keep)*dset_kwargs.eps], \
                                  dim=0).type(torch.float32).to(device) 
            samples *= torch.sqrt(scale)  
    return samples 
        


#------------------------------------------------------------------------------#
#for our version of toy_cfm training 

class torch_wrapper(torch.nn.Module):
    """
    Wraps model to torchdyn compatible format.
    
    """

    def __init__(self, model, ifscfm_loss):
        super().__init__()
        self.model = model
        self.ifscfm_loss = ifscfm_loss

    def forward(self, t, x, *args, **kwargs):
        t = t.repeat(x.shape[0]) #bs 
        if self.ifscfm_loss.precond:
            #if applying precond, modify x input appropriately 
            cin = self.ifscfm_loss._get_Cin(t)
            if self.ifscfm_loss.flowmatcher.space=='IS': 
                x = torch.einsum('ij, bjk -> bik', self.ifscfm_loss.flowmatcher.W.T, \
                                 x.unsqueeze(-1)).squeeze(-1)
                x *= cin 
                x = torch.einsum('ij, bjk -> bik', self.ifscfm_loss.flowmatcher.W, \
                                 x.unsqueeze(-1)).squeeze(-1)
            else:
                x *= cin 
        
        #get net output
        vt = self.model(x, t)
        
        if self.ifscfm_loss.precond:
            #if applying precond, modify netout accordingly
            cout = self.ifscfm_loss._get_Cout(t)
            if self.ifscfm_loss.flowmatcher.space=='IS':
                vt = torch.einsum('ij, bjk -> bik', self.ifscfm_loss.flowmatcher.W.T, \
                                      vt.unsqueeze(-1)).squeeze(-1)
                vt *= torch.reciprocal(cout)
                vt = torch.einsum('ij, bjk -> bik', self.ifscfm_loss.flowmatcher.W, \
                                  vt.unsqueeze(-1)).squeeze(-1)
            else: 
                vt *= torch.reciprocal(cout)
        
        return vt 
    
def calc_trajectories(model, dset_kwargs, loss_fn, device, n=1024, nt=100):
    """
    Computes ODE trajectories for plotting/checking
    model during training.
    
    Args
    -----
    model: torch.nn.Module. Instance of IFsCFMToyNet class. 
    dset_kwargs: dict. Contains original args for data specification.
    loss_fn: instance of IFsCFMToyLoss class. 
    n:int. Number of samples to simulate.
    nt: number of time pts to use for trajectory integration.
    device: instance of torch.device.
    """
    node = NeuralODE(torch_wrapper(model, loss_fn), solver='dopri5', \
                     sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    #get gen samples
    samples = get_ODE_gen_samples(loss_fn.flowmatcher, dset_kwargs, n, loss_fn.t_max, device)
    #get ts 
    ts = torch.linspace(0, loss_fn.t_max, nt).to(device)
    if loss_fn.flowmatcher.__class__.__name__ == 'IFsConditionalFlowMatcher':
        #flip times 
        ts = loss_fn.t_max - ts 
    #now sim ODE (gen direction)
    with torch.no_grad():
        traj = node.trajectory(samples, ts)
    return traj

    
def plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :, 0], traj[0, :, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :, 0], traj[:, :, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :, 0], traj[-1, :, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    return fig 

#--------------------------------------------------------------------------------#
#for our version of HD CFM training 

def calc_hd_trajectories(model, device, pfODEsim_kwargs, loss_fn, W, n=64, nt=100):
    """
    Similar to above toy option BUT simpler and using 
    less samples for memory saving.
    """
    #get samples
    sample_shape = (n, pfODEsim_kwargs.ch, pfODEsim_kwargs.img_res, pfODEsim_kwargs.img_res)
    samples = get_ODE_gen_samples(loss_fn.flowmatcher, pfODEsim_kwargs, n, 1.0, device) #bs, dim - ES
    samples = torch.einsum('ij, bjk -> bik', W, samples.unsqueeze(-1)).squeeze(-1) #bs, dim - IS 
    #get ts
    ts = torch.linspace(0, 1.0, nt).to(device)
    
    node_ = NeuralODE(torch_wrapper(model, loss_fn), solver='euler', sensitivity='adjoint')
    with torch.no_grad():
        traj = node_.trajectory(samples.reshape(sample_shape), \
                                ts)
        traj = traj[-1, :, :, :]
        traj = (traj * pfODEsim_kwargs.scale) + pfODEsim_kwargs.mean #undo center scaling 
        traj = np.clip(traj.cpu().numpy(), 0, 255).astype(np.uint8)
    return traj
        
    
def plot_imgs(imgs):
    """Similar to code we have to plot a grid of 64 
    generated imgs"""
    fig, axs = plt.subplots(8,8, figsize=(28, 28))
    for i in range(8):
        for j in range(8):
            idx = 8*i + j
            curr_img = imgs[idx, :, :, :]
            axs[i, j].imshow(curr_img.transpose(1, 2, 0))
    return fig    


