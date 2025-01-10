#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to simulate integrate flow and dynamics networks 
in alternating fashion, s.t. we simulate dynamics for some pts
and then move some in tau (flow dimension) before simulating more dynamics...

Idea here is to move slower in flow time than in dynamics time, s.t. we can 
have a sense for how dynamics changes as we change flow/flow time.
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
@click.option('--tps_per_tau',             help='Number of t steps to take per tau', metavar='INT',                                                        type=int, default=200, show_default=True)
@click.option('--num_taus',                help='Number of linearly space taus to simulate dynamics for', metavar='INT',                                   type=int, default=5, show_default=True)

#Dset options (if using balls case)
@click.option('--data_imgshape',           help='Shape for img if using toy image data (balls)', metavar='INT',                                            type=int, default=28, show_default=True)
@click.option('--data_radius',             help='Radius for balls to be created (if using balls dset)', metavar='INT',                                     type=int, default=3, show_default=True)
@click.option('--data_inch',               help='Nuber of channels in toy img data (if using balls dset)', metavar='INT',                                  type=int, default=1, show_default=True)
@click.option('--data_blur',               help='Whether or not to add small blur to created balls',                                                       is_flag=True)

def main(**kwargs):
    """
    Runs actual alternating flow/dynamics integration.
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
    out_dir = os.path.join(opts.outdir, 'alternating_flow_dyn_traj_sims')
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
    balls_dset_specs = dnnlib.EasyDict(img_shape=[opts.data_imgshape, opts.imgshape], radius=opts.data_radius, blur=opts.data_blur) \
        if opts.data_name.lower()=='balls' else None
    _, dset_samples, _ = dnnlib.util.get_toy_dynamicdset(opts.data_name, opts.n_trajs, opts.end_t, opts.dt, opts.sigma_dset, \
                                                         project=opts.project, proj_specs=proj_specs, balls_dset_specs=balls_dset_specs)
    data = np.array(dset_samples) #n_trajs, traj_len, dims
    data = np.reshape(data, (data.shape[0], data.shape[1], -1)) #make sure this is flat on dims! 
    
    #init curr_tau_pts
    starting_pts = torch.from_numpy(data[:, 0, :]).unsqueeze(1).type(torch.float32).to(device) #ntrajs,1,dim
    curr_tau_pts = starting_pts
    
    #setup init_tau to 1.0 --> always take starting pts from DS
    init_tau=1.0 
    
    #get taus to sim partial trajs for
    taus_to_sim = np.linspace(0, 1.0, opts.num_taus) 
    
    for tau in taus_to_sim: 
        dist.print0('*'*40) 
        dist.print0(f'Processing traj for tau:{tau}') 
        dist.print0('*'*40)
        curr_tau_traj = []
        #sim flow up to desired tau
        curr_tau_pts = dnnlib.util.calc_flow_trajectories(flow_net, curr_tau_pts.squeeze(1), init_tau, tau)
        curr_tau_pts = curr_tau_pts[-1].unsqueeze(1) #ntrajsx1xdim in curr tau space 
        curr_tau_traj.append(curr_tau_pts.cpu().numpy())
        #sim dyn for some steps, at this tau... 
        for s in range(opts.tps_per_tau): 
            dist.print0('*'*40) 
            dist.print0(f'Processing step:{s}') 
            dist.print0('*'*40)
            curr_tau_pts = dnnlib.util.calc_dyn_trajectories(dyn_net, curr_tau_pts.squeeze(1), tau)
            curr_tau_pts = curr_tau_pts[-1].unsqueeze(1)
            curr_tau_traj.append(curr_tau_pts.cpu().numpy())
        curr_tau_traj = np.concatenate(curr_tau_traj, axis=1)
        #save steps for curr_tau 
        if dist.get_rank() ==0:
            curr_tau_out_fname = out_fname_root + f'_{tau}tau_{opts.tps_per_tau}timepts.npz'
            np.savez(os.path.join(out_dir, curr_tau_out_fname), trajs=curr_tau_traj) 
        #update starting tau ... 
        init_tau = tau 

    #now save sim options...
    if dist.get_rank() ==0:
        with open(os.path.join(out_dir, 'sim_options.json'), 'wt') as f:
            json.dump(opts, f, indent=2)     
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------