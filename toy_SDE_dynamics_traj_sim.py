#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small script to simulate dynamics trajectories at multiple taus
using SDE instead of ODE for dynamics component.
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


#----------------------------------------------------------------------------
# Parse a comma separated list of floats and return a list of floats.
# Example: '0.0, 0.25, 0.50' returns [0.0,0.25,0.50]

def parse_float_list(s):
    if isinstance(s, list): return s
    ranges = []
    for p in s.split(','):
        ranges.append(float(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

#General Options
@click.option('--outdir',                  help='Where to save the results', metavar='DIR',                                                                type=str, required=True)
@click.option('--network',                 help='Network to use when simulating trajs', metavar='STR',                                                     type=str, required=True)
@click.option('--traj_len',                help='Number of points to simulate per trajectory.', metavar='INT',                                             type=int, default=1000, show_default=True)
@click.option('--taus',                    help='Tau values to simulate full trajs for  [default: varies]', metavar='LIST',                                type=parse_float_list)
@click.option('--device',                  help='Name of device to use', metavar='STR',                                                                    type=str, default='cuda:0', show_default=True)
@click.option('--sigma_sde',               help='Sigma value to use when constructing G for SDE simulation.', metavar='FLOAT',                             type=float, default=9e-3, show_default=True)
@click.option('--start_time',              help='Start time for each \delta t interval to be integrated over.', metavar='FLOAT',                           type=float, default=0.0, show_default=True)
@click.option('--end_time',                help='End time for each \delta t interval to be integrated over', metavar='FLOAT',                              type=float, default=1.0, show_default=True)
@click.option('--dt_sde',                  help='Step size to be used when integrating SDE with EM method.', metavar='FLOAT',                              type=float, default=1e-2, show_default=True)


# Main Dset Options
@click.option('--data_name',               help='Name of toy dset to use', metavar='STR',                                                                  type=str, required=True)
@click.option('--data_dim',                help='Number of dimensions in original dset (w/out projection)', metavar='INT',                                 type=int, required=True)
@click.option('--dims_to_keep',            help='Number of dimensions to keep', metavar='INT',                                                             type=int, required=True)
@click.option('--n_trajs',                 help='Number of trajectories to simulate.', metavar='INT',                                                      type=int, default=100, show_default=True)
@click.option('--end_t',                   help='End time for each sampled trajectory.', metavar='FLOAT',                                                  type=float, default=1.0, show_default=True)
@click.option('--dt_dset',                 help='Time interval between successive pts in sampled trajectories', metavar='FLOAT',                           type=float, default=1e-3, show_default=True)
@click.option('--sigma_dset',              help='Std for noise used to sample trajectories.', metavar='FLOAT',                                             type=float, default=0.25, show_default=True)

#Projection Options 
@click.option('--project',                 help='Project original dset to higher dimensional space',                                                       is_flag=True)
@click.option('--project_to',              help='Dimensionality we wish to achieve after projecting data.', metavar='INT',                                 type=int, default=3, show_default=True)
@click.option('--project_type',            help='Non-linearity used to construct projections', metavar='DIR',                                              type=str, default='double swish', show_default=True)
@click.option('--project_temp',            help='Temperature param for non-linearity used in projection.', metavar='FLOAT',                                type=float, default=1.2, show_default=True)

#Dset options (if using balls case)
@click.option('--data_imgshape',           help='Shape for img if using toy image data (balls)', metavar='INT',                                            type=int, default=32, show_default=True)
@click.option('--data_radius',             help='Radius for balls to be created (if using balls dset)', metavar='INT',                                     type=int, default=3, show_default=True)
@click.option('--data_inch',               help='Nuber of channels in toy img data (if using balls dset)', metavar='INT',                                  type=int, default=1, show_default=True)
@click.option('--data_blur',               help='Whether or not to add small blur to created balls',                                                       is_flag=True)


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
    out_dir = os.path.join(opts.outdir, 'SDE_dynamics_traj_net_sims')
    if dist.get_rank() == 0:
        os.makedirs(out_dir, exist_ok=True)    
    
    out_fname_root = f'{opts.data_name}_PRRfrom{working_data_dim}D_to{opts.dims_to_keep}D' if working_data_dim != opts.dims_to_keep else f'{opts.data_name}_PRP'
    
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
    
    #get flow and dynamics nets from the above
    flow_net = net.unet_model
    dyn_net = net.vnet_model
    
    #now construct set of GT trajs using dset options given
    #will use only starting pts of these ... 
    proj_specs = dnnlib.EasyDict(project_to=opts.project_to, proj_type=opts.project_type, temp=opts.project_temp) if opts.project else None 
    
    balls_dset_specs = dnnlib.EasyDict(img_shape=[opts.data_imgshape, opts.data_imgshape], radius=opts.data_radius, blur=opts.data_blur) \
        if opts.data_name.lower()=='balls' else None

    _, dset_samples, _ = dnnlib.util.get_toy_dynamicdset(opts.data_name, opts.n_trajs, opts.end_t, opts.dt_dset, opts.sigma_dset, \
                                                        project=opts.project, proj_specs=proj_specs, balls_dset_specs=balls_dset_specs)    
    
    data = np.array(dset_samples) #n_trajs, traj_len, dims
    data = np.reshape(data, (data.shape[0], data.shape[1], -1)) #make sure this is flat on dims! 
    
    starting_pts = torch.from_numpy(data[:, 0, :]).unsqueeze(1).type(torch.float32).to(device) #ntrajs,1,dim
    
    #for each tau, 
    #pass initial data pts to desired flow time
    #take these and simulate full dynamics trajectories
    for tau in opts.taus: 
        dist.print0('*'*40)
        dist.print0(f'Simulating SDE dynamics for {tau} tau')
        dist.print0('*'*40)
        curr_tau_trajs = []
        if tau != 1.0:
            #use flow net to map starting pts from DS (tau==1.0) to desired tau
            #this step is still just a pfODE!! 
            curr_tau_starting_pts = dnnlib.util.calc_flow_trajectories(flow_net, starting_pts.squeeze(1), 1.0, tau)
            #init curr_tau_pts
            curr_tau_pts = curr_tau_starting_pts[-1].unsqueeze(1) 
        else: 
            curr_tau_pts =  starting_pts
        curr_tau_trajs.append(curr_tau_pts.cpu().numpy())
        for i in range(opts.traj_len):
            dist.print0('*'*40)
            dist.print0(f'Processing step: {i}')
            dist.print0('*'*40)
            curr_tau_pts = dnnlib.util.int_dyn_SDE(curr_tau_pts.squeeze(1), dyn_net, tau, opts.sigma_sde, \
                                                   opts.start_time, opts.end_time, opts.dt_sde)
            curr_tau_pts = torch.from_numpy(curr_tau_pts[-1]).unsqueeze(1).type(torch.float32).to(device)
            curr_tau_trajs.append(curr_tau_pts.cpu().numpy())
        curr_tau_trajs = np.concatenate(curr_tau_trajs, axis=1) #n_trajs, traj_len+1, dim
        #save this out...
        if dist.get_rank() ==0:
            curr_tau_out_fname = out_fname_root + f'_{tau}tau.npz'
            np.savez(os.path.join(out_dir, curr_tau_out_fname), trajs=curr_tau_trajs)
    
    #now save sim options...
    if dist.get_rank() ==0:
        with open(os.path.join(out_dir, 'sim_options.json'), 'wt') as f:
            json.dump(opts, f, indent=2)     
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
    
    
    
    

