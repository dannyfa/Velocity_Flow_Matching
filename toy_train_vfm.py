#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implements VFM model training for toy datasets 

Supports original ToyConvUNet and simpler/smaller
ToyMLP archs.

"""

import os
import re 
import json 
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import toy_training_loop_vfm

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.


@click.command()

# Main options.
@click.option('--outdir',                  help='Where to save the results', metavar='DIR',                                                                type=str, required=True)
@click.option('--data_name',               help='Name of toy dset to use', metavar='STR',                                                                  type=str, required=True)
@click.option('--data_dim',                help='Number of dimensions in original dset (w/out projection)', metavar='INT',                                 type=int, required=True)
@click.option('--dims_to_keep',            help='Number of dimensions to keep', metavar='INT',                                                             type=int, required=True)
@click.option('--n_trajs',                 help='Number of trajectories to sample when creating dset.', metavar='INT',                                     type=int, default=50, show_default=True)
@click.option('--end_t',                   help='End time for each sampled trajectory.', metavar='FLOAT',                                                  type=float, default=1.0, show_default=True)
@click.option('--dt',                      help='Time interval between successive pts in sampled trajectories', metavar='FLOAT',                           type=float, default=1e-3, show_default=True)
@click.option('--sigma_dset',              help='Std for noise used to sample trajectories.', metavar='FLOAT',                                             type=float, default=0.25, show_default=True)
@click.option('--project',                 help='Project original dset to higher dimensional space',                                                       is_flag=True)
@click.option('--project_to',              help='Dimensionality we wish to achieve after projecting data.', metavar='INT',                                 type=int, default=3, show_default=True)
@click.option('--project_type',            help='Non-linearity used to construct projections', metavar='DIR',                                              type=str, default='double swish', show_default=True)
@click.option('--project_temp',            help='Temperature param for non-linearity used in projection.', metavar='FLOAT',                                type=float, default=1.2, show_default=True)
@click.option('--flow_matcher_type',       help='Flow matching implementation to use.', metavar='regular|exactot',                                         type=click.Choice(['regular', 'exactot']), default='regular', show_default=True)
@click.option('--sigma_fm',                help='Sigma val for flow matcher class', metavar='FLOAT',                                                       type=float, default=0.1, show_default=True)
@click.option('--eps',                     help='Variance for compressed dimnensions in LS. Used to sample x0s', metavar='FLOAT',                          type=float, default=1.0, show_default=True)
@click.option('--arch',                    help='Network architecture to use. This is same for u,v nets.', metavar='ToyConvUNet|ToyMLP',                   type=click.Choice(['ToyConvUNet', 'ToyMLP']), default='ToyConvUNet', show_default=True)
@click.option('--encoder_depth',           help='Number of hidden layers in MLP encoder', metavar='INT',                                                   type=int, default=2, show_default=True)
@click.option('--encoder_width',           help='Width of each hidden layer in MLP encoder', metavar='INT',                                                type=int, default=10, show_default=True)


# Hyperparameters.
@click.option('--duration',               help='Training duration', metavar='MIMG',                                                                        type=click.FloatRange(min=0, min_open=True), default=7000, show_default=True)
@click.option('--batch',                  help='Total batch size', metavar='INT',                                                                          type=click.IntRange(min=1), default=8192, show_default=True)
@click.option('--batch-gpu',              help='Limit batch size per GPU', metavar='INT',                                                                  type=click.IntRange(min=1), default=1024, show_default=True)
@click.option('--lr',                     help='Learning rate', metavar='FLOAT',                                                                           type=click.FloatRange(min=0, min_open=True), default=1e-5, show_default=True)
@click.option('--use_ema',                help='Whether or not to apply EMA to model params',                                                              is_flag=True)
@click.option('--ema',                    help='EMA half-life (if using EMA)', metavar='MIMG',                                                             type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--alpha',                  help='Scale for flow net component of loss', metavar='FLOAT',                                                    type=float, default=1.0, show_default=True)
@click.option('--beta',                   help='Scale for dynamics net component of loss', metavar='FLOAT',                                                type=float, default=1.0, show_default=True)
@click.option('--gamma',                  help='Scale for Lie derivative component of loss', metavar='FLOAT',                                              type=float, default=1.0, show_default=True)
@click.option('--grad_clip',              help='Whether or not to clip model gradient norm.',                                                              is_flag=True)
@click.option('--grad_clip_val',          help='Max value model gradients should be clipped to.', metavar='FLOAT',                                         type=float, default=1.0, show_default=True)


# Performance-related.
@click.option('--ls',                     help='Loss scaling', metavar='FLOAT',                                                                            type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',                  help='Enable cuDNN benchmarking', metavar='BOOL',                                                                type=bool, default=True, show_default=True)
@click.option('--cache',                  help='Cache dataset in CPU memory', metavar='BOOL',                                                              type=bool, default=True, show_default=True)
@click.option('--workers',                help='DataLoader worker processes', metavar='INT',                                                               type=click.IntRange(min=1), default=1, show_default=True)


# I/O-related.
@click.option('--desc',                   help='String to include in result dir name', metavar='STR',                                                      type=str)
@click.option('--nosubdir',               help='Do not create a subdirectory for results',                                                                 is_flag=True)
@click.option('--tick',                   help='How often to print progress', metavar='KIMG',                                                              type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',                   help='How often to save snapshots', metavar='TICKS',                                                             type=click.IntRange(min=1), default=250, show_default=True)
@click.option('--dump',                   help='How often to dump state', metavar='TICKS',                                                                 type=click.IntRange(min=1), default=250, show_default=True)
@click.option('--seed',                   help='Random seed  [default: random]', metavar='INT',                                                            type=int)
@click.option('--resume',                 help='Resume from previous training state', metavar='PT',                                                        type=str)
@click.option('-n', '--dry-run',          help='Print training options and exit',                                                                          is_flag=True)


def main(**kwargs):
    """
    Sets up main args needed for toy_training_loop_vfm.py 
    """
    
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    
    #setup device
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    
    #setup data_dim to match augmented dims (if using this option)
    working_data_dim = opts.project_to if opts.project else opts.data_dim
    

    # Initialize config dict.
    # Set up simplified dset, optim, loss, network kwargs 
    c = dnnlib.EasyDict()
    
    #setup dset args
    proj_specs = dnnlib.EasyDict(project_to=opts.project_to, proj_type=opts.project_type, temp=opts.project_temp) if opts.project else None
    c.dataset_kwargs = dnnlib.EasyDict(dset_name = opts.data_name, n_trajs=opts.n_trajs, T=opts.end_t, \
                                       dt=opts.dt, sigma=opts.sigma_dset, project=opts.project, proj_specs=proj_specs)
        
    #setup dataloder kwargs 
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)

    
    #set up additional args to sample x0s
    c.x0_sampler_kwargs = dnnlib.EasyDict(data_dim=opts.data_dim, working_data_dim=working_data_dim, \
                                          dims_to_keep=opts.dims_to_keep, eps=opts.eps)
    
    #setup optimizer kwargs 
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    
    #setup loss kwargs
    c.loss_kwargs = dnnlib.EasyDict(flow_matcher_type=opts.flow_matcher_type, 
                                        sigma=opts.sigma_fm, class_name='training.loss.VFMToyLoss')
    
    #setup net kwargs 
    if opts.arch == "ToyConvUNet": 
        c.network_kwargs = dnnlib.EasyDict(model_type=opts.arch, channels=[32, 64, 128, 256], fc_embed_dim=2, conv_embed_dim=256, \
                                               data_dim=working_data_dim, dims_to_keep=opts.dims_to_keep, \
                                                   out_ch=1, class_name='training.networks.VFMToyNet') 
    elif opts.arch=='ToyMLP': 
        c.network_kwargs = dnnlib.EasyDict(model_type=opts.arch, data_dim=working_data_dim, dims_to_keep=opts.dims_to_keep, \
                                           class_name='training.networks.VFMToyNet')

    else:
        raise NotImplementedError('Only ToyConvUNet and ToyMLP architectures supported!') 
    
    c.network_kwargs.update(depth_encoder=opts.encoder_depth, width_encoder=opts.encoder_width, cd_eps=opts.eps)
        
    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.use_ema = opts.use_ema
    c.ema_halflife_kimg = int(opts.ema * 1000) #only used if use_ema==True 
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)
    c.update(alpha=opts.alpha, beta=opts.beta, gamma=opts.gamma)
    c.update(grad_clip=opts.grad_clip, grad_clip_val=opts.grad_clip_val)
    
    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=device)
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Resume learning
    if opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    schedule_type_str = 'prp' if working_data_dim == opts.dims_to_keep else 'prr' 
    desc = f'{opts.data_name}-{schedule_type_str}-uncond-{opts.arch}-{opts.flow_matcher_type}FM-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-fp32'

    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset name:            {c.dataset_kwargs.dset_name}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0()
        
    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    

    # Train.
    toy_training_loop_vfm.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
