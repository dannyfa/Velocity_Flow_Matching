#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implements VFM model training 

"""


import os
import re
import json
import click
import torch
import dnnlib
import numpy as np
from torch_utils import distributed as dist
from training import toy_training_loop_vfm_noDataGen

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.


@click.command()

# Main Options (adapted for loading data)
@click.option('--data_path',               help='Path to the folder containing dataset_samples.npz', metavar='DIR', type=str, required=True)
@click.option('--data_name',               help='Name of the dataset', metavar='STR', type=str, required=True)
@click.option('--dims_to_keep',            help='Number of dimensions to keep', metavar='INT', type=int, required=True)
@click.option('--dt',                      help='Time interval between successive pts in sampled trajectories', metavar='FLOAT', type=float, default=1e-3, show_default=True)

# Balls dset options (kept for compatibility, but unused if not 'balls')
@click.option('--data_imgshape',           help='Shape for img if using toy image data (balls)', metavar='INT', type=int, default=28, show_default=True)
@click.option('--data_radius',             help='Radius for balls to be created (if using balls dset)', metavar='INT', type=int, default=3, show_default=True)
@click.option('--data_inch',               help='Number of channels in toy img data (if using balls dset)', metavar='INT', type=int, default=1, show_default=True)
@click.option('--data_blur',               help='Whether or not to add small blur to created balls', is_flag=True)

# FM options
@click.option('--flow_matcher_type',       help='Flow matching implementation to use.', metavar='regular|exactot', type=click.Choice(['regular', 'exactot']), default='regular', show_default=True)
@click.option('--sigma_dyn_fm',            help='Sigma val for dynamics flow matcher class', metavar='FLOAT', type=float, default=0.1, show_default=True)
@click.option('--sigma_comp_fm',           help='Sigma val for compression flow matcher class', metavar='FLOAT', type=float, default=0.1, show_default=True)
@click.option('--eps',                     help='Variance for compressed dimensions in LS. Used to sample x0s', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--d_min',                   help='Minimum variance for any/all dimensions in LS. Used to sample x0s.', metavar='FLOAT', type=float, default=1e-15, show_default=True)

# Arch Options
@click.option('--dyn_arch',                help='Dynamics net arch to use.', metavar='ToyConvUNet|ToyMLP|Adapted_ToyConvUNet', type=click.Choice(['ToyConvUNet', 'ToyMLP', 'Adapted_ToyConvUNet']), default='ToyMLP', show_default=True)
@click.option('--flow_arch',               help='Flow net arch to use.', metavar='ToyConvUNet|ToyMLP|Adapted_ToyConvUNet', type=click.Choice(['ToyConvUNet', 'ToyMLP', 'Adapted_ToyConvUNet']), default='ToyMLP', show_default=True)
@click.option('--encoder_arch',            help='Network architecture to use for encoder.', metavar='Latent_MLP_VAE|Latent_CNN_VAE|Latent_LargeCNN_VAE|Adapted_Latent_LargeCNN_VAE', type=click.Choice(['Latent_MLP_VAE', 'Latent_CNN_VAE', 'Latent_LargeCNN_VAE', 'Adapted_Latent_LargeCNN_VAE']), default='Latent_MLP_VAE', show_default=True)
@click.option('--encoder_depth',           help='Number of hidden layers in MLP encoder', metavar='INT', type=int, default=2, show_default=True)
@click.option('--encoder_width',           help='Width of each hidden layer in MLP encoder', metavar='INT', type=int, default=10, show_default=True)
@click.option('--mlp_depth',               help='Number of hidden layers in MLP flow, dyn nets', metavar='INT', type=int, default=2, show_default=True)
@click.option('--mlp_width',               help='Width of each hidden layer in MLP flow,dyn nets', metavar='INT', type=int, default=64, show_default=True)

# Training Hyperparameters.
@click.option('--duration',                help='Training duration', metavar='MIMG', type=click.FloatRange(min=0, min_open=True), default=7000, show_default=True)
@click.option('--batch',                   help='Total batch size', metavar='INT', type=click.IntRange(min=1), default=8192, show_default=True)
@click.option('--batch-gpu',               help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1), default=1024, show_default=True)
@click.option('--lr',                      help='Learning rate', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1e-5, show_default=True)
@click.option('--use_ema',                 help='Whether or not to apply EMA to model params', is_flag=True)
@click.option('--ema',                     help='EMA half-life (if using EMA)', metavar='MIMG', type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--alpha',                   help='Scale for flow net component of loss', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--beta',                    help='Scale for dynamics net component of loss', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--gamma',                   help='Scale for Lie derivative component of loss', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--eta',                     help='Scale for encoder reconstruction component of loss', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--grad_clip',               help='Whether or not to clip model gradient norm.', is_flag=True)
@click.option('--grad_clip_val',           help='Max value model gradients should be clipped to.', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--norm_lie',                help='Whether or not to normalize Lie derivative.', is_flag=True)
@click.option('--pre_train',               help='Whether or not to pre-train nets.', is_flag=True)
@click.option('--pre_train_kimgs',         help='Number of Kimgs to pre-train nets for', metavar='INT', type=int, default=0, show_default=True)

# Performance-related.
@click.option('--ls',                      help='Loss scaling', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',                   help='Enable cuDNN benchmarking', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--cache',                   help='Cache dataset in CPU memory', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--workers',                 help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--outdir',                  help='Where to save the results', metavar='DIR', type=str, required=True)
@click.option('--desc',                    help='String to include in result dir name', metavar='STR', type=str)
@click.option('--nosubdir',                help='Do not create a subdirectory for results', is_flag=True)
@click.option('--tick',                    help='How often to print progress', metavar='KIMG', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',                    help='How often to save snapshots', metavar='TICKS', type=click.IntRange(min=1), default=250, show_default=True)
@click.option('--dump',                    help='How often to dump state', metavar='TICKS', type=click.IntRange(min=1), default=250, show_default=True)
@click.option('--seed',                    help='Random seed  [default: random]', metavar='INT', type=int)
@click.option('--resume',                  help='Resume from previous training state', metavar='PT', type=str)
@click.option('-n', '--dry-run',           help='Print training options and exit', is_flag=True)


def main(**kwargs):
    """
    Sets up main args needed for toy_training_loop_vfm_noDataGen.py
    """
    
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    
    # Setup device
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    
    # Load data
    dataset_samples = np.load(os.path.join(opts.data_path, 'dataset_samples.npz'), allow_pickle=True)
    dset_samples_raw = dataset_samples['samples']
    dset_samples = [np.array(dset_samples_raw[ii]) for ii in range(len(dset_samples_raw))]
    dataset_obj = dnnlib.util.ToyDsetDynamics(dset_samples, opts.dt, nForward=1)
    
    # Infer data_dim from loaded data
    opts.data_dim = dset_samples[0][0].shape[0]
    working_data_dim = opts.data_dim
    
    # Infer n_trajs
    inferred_n_trajs = len(dset_samples)
    
    # Initialize config dict.
    c = dnnlib.EasyDict()
    
    # Setup dataset kwargs
    balls_dset_specs = dnnlib.EasyDict(img_shape=[opts.data_imgshape, opts.data_imgshape], radius=opts.data_radius, blur=opts.data_blur) if opts.data_name.lower()=='balls' else None
    c.dataset_kwargs = dnnlib.EasyDict(dset_name=opts.data_name, dt=opts.dt, balls_dset_specs=balls_dset_specs)
    c.dataset_kwargs.n_trajs = inferred_n_trajs
    c.dataset_obj = dataset_obj
    c.dset_samples = dset_samples
    
    # Setup dataloader kwargs
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    
    # Setup optimizer kwargs
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    
    # Setup loss kwargs
    c.loss_kwargs = dnnlib.EasyDict(flow_matcher_type=opts.flow_matcher_type, 
                                    sigma_dynamics=opts.sigma_dyn_fm, sigma_compression=opts.sigma_comp_fm, 
                                    normalize_lie=opts.norm_lie, class_name='training.loss.VFMToyLoss')
    
    # Setup network kwargs
    c.network_kwargs = dnnlib.EasyDict(dyn_model_type=opts.dyn_arch, flow_model_type=opts.flow_arch, encoder_type=opts.encoder_arch, channels=[32, 64, 128, 256], conv_embed_dim=256, 
                                       data_dim=working_data_dim, dims_to_keep=opts.dims_to_keep, depth_mlp=opts.mlp_depth, width_mlp=opts.mlp_width, depth_encoder=opts.encoder_depth, 
                                       width_encoder=opts.encoder_width, cd_eps=opts.eps, d_min=opts.d_min, img_size=opts.data_imgshape, in_ch=opts.data_inch, class_name='training.networks_v2.VFMToyNet')
    
    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.use_ema = opts.use_ema
    c.ema_halflife_kimg = int(opts.ema * 1000) # only used if use_ema==True 
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)
    c.update(alpha=opts.alpha, beta=opts.beta, gamma=opts.gamma, eta=opts.eta)
    c.update(grad_clip=opts.grad_clip, grad_clip_val=opts.grad_clip_val)
    c.update(pre_train=opts.pre_train, pre_train_kimgs=opts.pre_train_kimgs)
    
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
    desc = f'{opts.data_name}-{schedule_type_str}-uncond-{opts.flow_matcher_type}FM-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-fp32'

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

    # Define serializable version of c (exclude non-JSON-friendly keys)
    serial_c = {k: v for k, v in c.items() if k not in ['dataset_obj', 'dset_samples']}

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(serial_c, indent=2))
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
        c.network_kwargs.update(save_dir=c.run_dir) # add save_dir to network kwargs 
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(serial_c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    toy_training_loop_vfm_noDataGen.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------