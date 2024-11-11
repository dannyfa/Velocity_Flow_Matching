#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to simulate a couple of trajectories 
using dynamics network. 

For now, am doing so ONLY in data space. But will add 
option to run it on latent space as well. 
"""

#general dependencies 
import torch 
import numpy as np 
import pickle
import os
import click
import json 

#repo specific dependencies
import dnnlib
from torch_utils import distributed as dist

@click.command()
#device
#network
#dataset specs -- ntraj, proj specs ....
#number of points to sim per traj
#output dir 

# Main options.
@click.option('--outdir',                  help='Where to save the results', metavar='DIR',                                                                type=str, required=True)
@click.option('--network',                 help='Network to use when simulating trajs', metavar='STR',                                                     type=str, required=True)
@click.option('--data_name',               help='Name of toy dset to use', metavar='STR',                                                                  type=str, required=True)
@click.option('--data_dim',                help='Number of dimensions in original dset (w/out projection)', metavar='INT',                                 type=int, required=True)
@click.option('--dims_to_keep',            help='Number of dimensions to keep', metavar='INT',                                                             type=int, required=True)
@click.option('--n_trajs',                 help='Number of trajectories to simulate.', metavar='INT',                                                      type=int, default=100, show_default=True)
@click.option('--end_t',                   help='End time for each sampled trajectory.', metavar='FLOAT',                                                  type=float, default=1.0, show_default=True)
@click.option('--dt',                      help='Time interval between successive pts in sampled trajectories', metavar='FLOAT',                           type=float, default=1e-3, show_default=True)
@click.option('--sigma_dset',              help='Std for noise used to sample trajectories.', metavar='FLOAT',                                             type=float, default=0.25, show_default=True)
@click.option('--project',                 help='Project original dset to higher dimensional space',                                                       is_flag=True)
@click.option('--project_to',              help='Dimensionality we wish to achieve after projecting data.', metavar='INT',                                 type=int, default=3, show_default=True)
@click.option('--project_type',            help='Non-linearity used to construct projections', metavar='DIR',                                              type=str, default='double swish', show_default=True)
@click.option('--project_temp',            help='Temperature param for non-linearity used in projection.', metavar='FLOAT',                                type=float, default=1.2, show_default=True)
@click.option('--device',                  help='Name of device to use', metavar='STR',                                                                    type=str, default='cuda:0', show_default=True)
@click.option('--n_pts',                   help='Number of points to simulate per trajectory.', metavar='INT',                                             type=int, default=1000, show_default=True)


def main(**kwargs):
    """
    Runs actual trajectory simulations....
    """
    
    #get dict with our args 
    opts = dnnlib.EasyDict(kwargs)
    
    #init distributed mode
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    
    #setup device 
    device = torch.device(opts.device)
    
    #setup data_dim to match augmented dims (if using this option)
    working_data_dim = opts.project_to if opts.project else opts.data_dim
    
    #set up save dir and output fname
    out_dir = os.path.join(opts.outdir, 'dynamics_traj_net_sims')
    if dist.get_rank() == 0:
        os.makedirs(out_dir, exist_ok=True)    
    
    out_fname = f'{opts.data_name}_PRRfrom{working_data_dim}D_to{opts.dims_to_keep}D.npz' if working_data_dim != opts.dims_to_keep else f'{opts.data_name}_PRP.npz'
    
    #ok now load network 
    if dist.get_rank() != 0: 
        torch.distributed.barrier() 
    
    dist.print0(f'Loading network from "{opts.network}"...') 
    with dnnlib.util.open_url(opts.network, verbose=(dist.get_rank() == 0)) as f: 
        net = pickle.load(f)['ema'].to(device) 
        dyn_net = net.vnet_model #get ONLY dyn net! 
    
    # Other ranks follow.
    if dist.get_rank() == 0: 
        torch.distributed.barrier()
        
    
    #now construct set of GT trajs using dset options given
    #will use only starting pts of these ... 
    proj_specs = dnnlib.EasyDict(project_to=opts.project_to, proj_type=opts.project_type, temp=opts.project_temp) if opts.project else None 
    _, dset_samples = dnnlib.util.get_toy_dynamicdset(opts.data_name, opts.n_trajs, opts.end_t, opts.dt, opts.sigma_dset, project=opts.project, proj_specs=proj_specs)
    data = np.array(dset_samples) #n_trajs, traj_len, dims
    
    #ok now simulate our trajs 
    starting_pts = torch.from_numpy(data[:, 0, :]).unsqueeze(1).type(torch.float32).to(device)
    curr_pts = starting_pts 
    ds_dyn_trajs = []
    ds_dyn_trajs.append(starting_pts.cpu().numpy()) 
    
    for tp in range(opts.n_pts): 
        print('*'*40) 
        print('Simulating time pt:{}'.format(tp)) 
        print('*'*40) 
        v_trajs = dnnlib.util.calc_dyn_trajectories(dyn_net, curr_pts.squeeze(1)) 
        v_trajs = v_trajs[-1].unsqueeze(1) 
        curr_pts = v_trajs 
        ds_dyn_trajs.append(curr_pts.cpu().numpy())    
    
    ds_dyn_trajs = np.concatenate(ds_dyn_trajs, axis=1) #cat along time dim -- ntrajs, npts, dim 
    
    #save this to output dir 
    #save also a json with all of our config/arg choices 
    if dist.get_rank() ==0:
        np.savez(os.path.join(out_dir, out_fname), trajs=ds_dyn_trajs)
        with open(os.path.join(out_dir, 'sim_options.json'), 'wt') as f:
            json.dump(opts, f, indent=2)        
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
    
    
    
    