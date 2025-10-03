#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implements VFM model training 

"""


import os
import gc
import re
import json
import click
import torch
import dnnlib
import numpy as np
from torch_utils import distributed as dist
from training import toy_training_loop_vfm_v6
import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

def _to_list_of_arrays(x):
    """Normalize npz-loaded 'samples' into a Python list of arrays."""
    if isinstance(x, list):
        return [np.asarray(a) for a in x]
    x = np.asarray(x, dtype=object)
    if x.dtype == object:                  # object array of trials
        return [np.asarray(a) for a in x.tolist()]
    if x.ndim == 3:                        # (n_trial, T, D) or (n_trial, T, H, W)
        return [x[i] for i in range(x.shape[0])]
    if x.ndim == 5:                        # (n_trial, T, C, H, W)
        return [x[i] for i in range(x.shape[0])]
    raise RuntimeError(f"Unsupported dataset shape: {getattr(x,'shape',None)}")

def _is_vector_trials(trials):
    return all(arr.ndim == 2 for arr in trials)        # (T, D)

def _is_movie_trials(trials):
    return all(arr.ndim in (3, 4) for arr in trials)   # (T,H,W) or (T,C,H,W)

def _ensure_TCHW(trials, dtype=np.float32):
    """Ensure each trial is (T, C, H, W) and dtype=float32. No value scaling."""
    out = []
    for a in trials:
        if a.ndim == 3:               # (T, H, W) -> add channel
            a = a[:, None, :, :]
        elif a.ndim == 4:             # already (T, C, H, W)
            pass
        else:
            raise RuntimeError(f"Expected image trial, got {a.shape}")
        out.append(a.astype(dtype, copy=False))
    return out

def _infer_img_ch_hw(first_tchw):
    """first_tchw: array shaped (T, C, H, W). Returns (C,H,W)."""
    _, C, H, W = first_tchw.shape
    return int(C), int(H), int(W)


def _try_load_cov(path_npy, path_npz_key='samples'):
    if os.path.isfile(path_npy):
        arr = np.load(path_npy, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            try:
                return arr.tolist()
            except:
                return arr
        return arr
    path_npz = os.path.splitext(path_npy)[0] + '.npz'
    if os.path.isfile(path_npz):
        arr = np.load(path_npz, allow_pickle=True)[path_npz_key]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            try:
                return arr.tolist()
            except:
                return arr
        return arr
    return None


def _make_lag_cov_list(trajs, k_):
    lag_list = []
    for X in trajs:
        X = np.asarray(X)
        if X.ndim != 2:
            raise click.ClickException("lag_k covariates are only supported for vector trials (T, D).")
        T, D = X.shape
        if k_ == 0:
            lag = X.copy()
        else:
            blocks = [X[k_-j : T-j] for j in range(k_ + 1)]  # [X_{t-k}, ..., X_t]
            lag = np.concatenate(blocks, axis=1)
        lag_list.append(lag)
    return lag_list


def _make_lag_cov_list_images(trajs, k_):
    """For image trials shaped (T, C, H, W), return list of (T-k, C*(k+1), H, W)
    by stacking frames [t-k, ..., t] along the channel axis."""
    lag_list = []
    for X in trajs:
        X = np.asarray(X)
        if X.ndim != 4:
            raise click.ClickException("Image lag cov requires trials shaped (T, C, H, W).")
        T, C, H, W = X.shape
        if k_ == 0:
            lag = X.copy()
        else:
            # gather [X_{t-k}, ..., X_t] and concat over channel dimension
            blocks = [X[k_-j : T-j] for j in range(k_ + 1)]  # k_+1 tensors of shape (T-k, C, H, W)
            lag = np.concatenate(blocks, axis=1)             # (T-k, C*(k+1), H, W)
        lag_list.append(lag)
    return lag_list

def _flatten_timewise(arr):
    """Flatten spatial/channel dims, keep time: (T, ...) -> (T, prod(...))."""
    a = np.asarray(arr)
    return a.reshape(a.shape[0], -1)



@click.command()

# Main Options (adapted for loading data)
@click.option('--data_path',               help='Path to the folder containing dataset_samples.npz', metavar='DIR', type=str, required=True)
@click.option('--data_name',               help='Name of the dataset', metavar='STR', type=str, required=True)
@click.option('--dims_to_keep',            help='Number of dimensions to keep', metavar='INT', type=int, required=True)
@click.option('--dt',                      help='Time interval between successive pts in sampled trajectories', metavar='FLOAT', type=float, default=1e-3, show_default=True)

# Covariate options
@click.option('--include_x0_tau',          help='Whether to include x0_tau in dynamics net inputs', is_flag=True)
@click.option('--use_dynamic_covariates',  help='Whether to use dynamic covariates (if available)', is_flag=True)
@click.option('--use_static_covariates',   help='Whether to use static covariates (if available)', is_flag=True)
@click.option('--dim_cov_dynamic',         help='Dimension of dynamic covariates [default: inferred from data]', metavar='INT', type=int)
@click.option('--dim_cov_static',          help='Dimension of static covariates [default: inferred from data]', metavar='INT', type=int)
@click.option('--lag_k',                   help='Lag order k to build lag-k covariate matrix (>=0)', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--hist_noise_std',          help='Std of Gaussian noise added to lag-history features during training (0 disables).', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0, show_default=True)


@click.option('--data_imgshape',           help='Shape for img if using toy image data (balls)', metavar='INT', type=int, default=28, show_default=True)
@click.option('--data_radius',             help='Radius for balls to be created (if using balls dset)', metavar='INT', type=int, default=3, show_default=True)
@click.option('--data_inch',               help='Number of channels in toy img data (if using balls dset)', metavar='INT', type=int, default=1, show_default=True)
@click.option('--data_blur',               help='Whether or not to add small blur to created balls', is_flag=True)


# FM options
@click.option('--flow_matcher_type',       help='Flow matching implementation to use.', metavar='regular|exactot|sinkhorn', type=click.Choice(['regular', 'exactot', 'sinkhorn']), default='regular', show_default=True)
@click.option('--sigma_dyn_fm',            help='Sigma val for dynamics flow matcher class', metavar='FLOAT', type=float, default=0.1, show_default=True)
@click.option('--sigma_comp_fm',           help='Sigma val for compression flow matcher class', metavar='FLOAT', type=float, default=0.1, show_default=True)

@click.option('--cmp-is/--no-cmp-is', default=True, show_default=True,
              help='Use importance sampling over τ for compression (flow) loss.')
@click.option('--dyn-is/--no-dyn-is', default=True, show_default=True,
              help='Use importance sampling over (τ, t_dyn) for dynamics loss.')

@click.option('--is-bins-tau', type=int, default=16, show_default=True,
              help='Number of τ bins for importance sampling.')
@click.option('--is-bins-t', type=int, default=16, show_default=True,
              help='Number of t_dyn bins per τ-bin for importance sampling.')
@click.option('--is-ema', type=float, default=0.05, show_default=True,
              help='EMA step for second-moment tables used by IS (0<is_ema≤1).')
@click.option('--is-eps', type=float, default=1e-8, show_default=True,
              help='Tiny numerical floor to keep proposals/weights well-defined.')



@click.option('--eps',                     help='Variance for compressed dimensions in LS. Used to sample x0s', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--d_min',                   help='Minimum variance for any/all dimensions in LS. Used to sample x0s.', metavar='FLOAT', type=float, default=1e-15, show_default=True)

# Arch Options
@click.option('--dyn_arch',                help='Dynamics net arch to use.', metavar='ToyConvUNet|ToyMLP|Adapted_ToyConvUNet', type=click.Choice(['ToyConvUNet', 'ToyMLP', 'Adapted_ToyConvUNet']), default='ToyMLP', show_default=True)
@click.option('--use-t-dyn/--no-use-t-dyn', help='Include normalized dynamics time t_dyn=ts/dt in v-net inputs', default=True, show_default=True)

@click.option('--flow_arch',               help='Flow net arch to use.', metavar='ToyConvUNet|ToyMLP|Adapted_ToyConvUNet', type=click.Choice(['ToyConvUNet', 'ToyMLP', 'Adapted_ToyConvUNet']), default='ToyMLP', show_default=True)
@click.option('--encoder_arch',            help='Network architecture to use for encoder.', metavar='Latent_MLP_VAE|Latent_CNN_VAE|Latent_LargeCNN_VAE|Adapted_Latent_LargeCNN_VAE', type=click.Choice(['Latent_MLP_VAE', 'Latent_CNN_VAE', 'Latent_LargeCNN_VAE', 'Adapted_Latent_LargeCNN_VAE']), default='Latent_MLP_VAE', show_default=True)
@click.option('--encoder_depth',           help='Number of hidden layers in MLP encoder', metavar='INT', type=int, default=2, show_default=True)
@click.option('--encoder_width',           help='Width of each hidden layer in MLP encoder', metavar='INT', type=int, default=10, show_default=True)
@click.option('--encoder-rank', 'encoder_rank', help='Rank r for encoder U (default: None → 2*dims_to_keep)', metavar='INT', type=int, default=None, show_default=True)

@click.option('--init_latent',             help='Latent init mode', type=click.Choice(['random','pca']), default='random', show_default=True)
@click.option('--init_latent_steps',       help='Warmup steps for PCA init', type=int, default=50000, show_default=True)
@click.option('--init_latent_lr',          help='LR for PCA init', type=float, default=1e-3, show_default=True)
@click.option('--init_latent_max_samples', help='Max samples to fit PCA', type=int, default=100000, show_default=True)
@click.option('--init_latent_tol',          help='Relative tol for PCA warm-up early stop', type=float, default=1e-5, show_default=True)
@click.option('--init_latent_patience',     help='Stop after this many consecutive small-improvement steps', type=int, default=100, show_default=True)
@click.option('--init_latent_min_steps',    help='Do at least this many warm-up steps before checking tol', type=int, default=1000, show_default=True)

@click.option('--mlp_depth_cmp',               help='Number of hidden layers in MLP, compression flow nets', metavar='INT', type=int, default=2, show_default=True)
@click.option('--mlp_width_cmp',               help='Width of each hidden layer in MLP, compression flow nets', metavar='INT', type=int, default=64, show_default=True)
@click.option('--mlp_depth_dyn',               help='Number of hidden layers in MLP, dynamic flow nets', metavar='INT', type=int, default=2, show_default=True)
@click.option('--mlp_width_dyn',               help='Width of each hidden layer in MLP, dynamic flow nets', metavar='INT', type=int, default=64, show_default=True)


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
@click.option('--alpha_mu',           help='Weifht of KL loss on mu', metavar='FLOAT', type=float, default=0.1, show_default=True)
@click.option('--kl-eps', type=float, default=1e-6, show_default=True,
              help='Jitter/epsilon for KL Cholesky and variance clamps.')
@click.option('--kl-use-dims-to-keep/--no-kl-use-dims-to-keep',
              default=False, show_default=True,
              help='If set, KL uses first dims_to_keep as full-cov head; else full-cov over all d.')
@click.option('--kl-warmup-kimg', metavar='KIMG', type=click.FloatRange(min=0), default=1000, show_default=True,
              help='Warmup period (in kimg) before KL ramp starts.')
@click.option('--kl-ramp-kimg', metavar='KIMG', type=click.FloatRange(min=0), default=5000, show_default=True,
              help='Ramp duration (in kimg) for KL weight.')

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
@click.option('--eta_post', help='Post-ramp target for eta (encoder loss weight)', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--eta_hold_kimg', help='Hold period for eta (in kimg) before ramp starts', metavar='KIMG', type=click.FloatRange(min=0), default=1000, show_default=True)
@click.option('--eta_ramp_kimg', help='Ramp duration for eta (in kimg) from eta -> eta_post', metavar='KIMG', type=click.FloatRange(min=0), default=5000, show_default=True)



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
    
    # Load main data
    dataset_samples = np.load(os.path.join(opts.data_path, 'dataset_samples.npz'), allow_pickle=True)
    dset_samples_raw = dataset_samples['samples']
    dset_samples = _to_list_of_arrays(dset_samples_raw)

    if _is_movie_trials(dset_samples):
        dset_samples = _ensure_TCHW(dset_samples)
    
    dset_samples_orig = [np.asarray(a) for a in dset_samples]

    
    # Load covariate data if available
    cov_dynamic_samples = None
    cov_static_samples = None
    
    # Try to load dynamic covariates
    if opts.use_dynamic_covariates:
        cov_dynamic_path = os.path.join(opts.data_path, 'cov_dynamic_samples.npy')
        cov_dynamic_samples = _try_load_cov(cov_dynamic_path)
    
    # Load static covariates only if flag is set
    if opts.use_static_covariates:
        cov_static_path = os.path.join(opts.data_path, 'cov_static_samples.npy')
        cov_static_samples = _try_load_cov(cov_static_path)


    # Build lag-k covs, truncate, and concatenate
    k = int(max(0, opts.lag_k))
    min_T = min(np.asarray(tr).shape[0] for tr in dset_samples)
    if k >= min_T:
        raise click.ClickException(f'lag_k={k} must be < min trajectory length ({min_T}).')

    is_image = _is_movie_trials(dset_samples)
    if is_image:
        _T, _C, _H, _W = np.asarray(dset_samples[0]).shape
        opts.data_inch = _C
        opts.data_imgshape = _H  # assumes square

    
    # 1) construct lag-k cov matrix from original samples
    if is_image:
        # dset_samples: list of (T, C, H, W) — build (T-k, C*(k+1), H, W)
        lag_cov_list = None
    else:
        # dset_samples: list of (T, D) — build (T-k, (k+1)*D)
        lag_cov_list = _make_lag_cov_list(dset_samples, k)

    
    # 2) truncate dset_samples
    dset_samples_trunc = [np.asarray(X)[k:] for X in dset_samples]
    
    # 3) truncate cov_dynamic
    cov_dynamic_trunc = None
    if cov_dynamic_samples is not None:
        if isinstance(cov_dynamic_samples, list):
            cov_dynamic_trunc = []
            for C in cov_dynamic_samples:
                C = np.asarray(C)        # (T, Sd) or (T, C, H, W) or (T,)
                Ck = C[k:]               # time-align
                if Ck.ndim >= 2:
                    Ck = Ck.reshape(Ck.shape[0], -1)  # (T-k, Sd_vec)
                else:
                    Ck = Ck[:, None]                   # (T-k,) -> (T-k,1)
                cov_dynamic_trunc.append(Ck)
        elif isinstance(cov_dynamic_samples, np.ndarray):
            # ndarray: either (n_trial, T, Sd) or (n_trial, T, C, H, W)
            if cov_dynamic_samples.ndim == 3:
                # vector covs
                cov_dynamic_trunc = [cov_dynamic_samples[i, k:]                   # (T-k, Sd)
                                     for i in range(cov_dynamic_samples.shape[0])]
            elif cov_dynamic_samples.ndim == 5:
                # image-like covs
                n_trial, T, C, H, W = cov_dynamic_samples.shape
                cov_dynamic_trunc = [cov_dynamic_samples[i, k:].reshape(T - k, -1)  # (T-k, C*H*W)
                                     for i in range(n_trial)]
            else:
                raise click.ClickException("Unsupported cov_dynamic array shape; expected 3D or 5D.")
        else:
            raise click.ClickException("Unsupported cov_dynamic type; expected list or ndarray.")

    # 4) truncate cov_static if it is a list (time-indexed); if ndarray (per-traj), broadcast
    if cov_static_samples is None:
        # Case 1: lag + static=None
        cov_static_new = None
        cov_static_trunc_only = []
        if is_image:
            # stream per-trial: avoid materializing lag_cov_list
            for X in dset_samples:                      # X: (T, C, H, W)
                X = np.asarray(X)
                T = X.shape[0]
                cov_static_trunc_only.append(np.zeros((T - k, 0), dtype=X.dtype))
        else:
            cov_static_new = []
            cov_static_trunc_only = []
            for L in lag_cov_list:
                L_vec = np.asarray(L)                   # (T-k, (k+1)·D)
                cov_static_new.append(L_vec)            # keep legacy behavior for vectors
                cov_static_trunc_only.append(np.zeros((L_vec.shape[0], 0), dtype=L_vec.dtype))
    
    elif isinstance(cov_static_samples, list):
        # Case 2 & 3: time-indexed static; may be vector (T,S) or image-like (T,Cs,Hs,Ws)
        cov_static_new = None
        cov_static_trunc_only = []
        if is_image:
            for S, X in zip(cov_static_samples, dset_samples):
                S = np.asarray(S)
                # keep only time-aligned true static, flattened
                S_trunc = S[k:]
                if S_trunc.ndim >= 3: S_trunc = S_trunc.reshape(S_trunc.shape[0], -1)
                elif S_trunc.ndim == 1: S_trunc = S_trunc[:, None]
                cov_static_trunc_only.append(S_trunc)
        else:
            for S, L in zip(cov_static_samples, lag_cov_list):
                S = np.asarray(S)
                S_trunc = S[k:]
                if S_trunc.ndim == 1: S_trunc = S_trunc[:, None]
                cov_static_trunc_only.append(S_trunc)
                cov_static_new.append(np.concatenate([S_trunc, np.asarray(L)], axis=1))
        
    else:
        cov_static_new = None
        cov_static_trunc_only = []
        if is_image:
            for i, X in enumerate(dset_samples):
                X = np.asarray(X)
                T = X.shape[0]
                Prow = np.asarray(cov_static_samples[i])
                if Prow.ndim >= 2: Prow = Prow.reshape(-1)
                if Prow.ndim == 0: Prow = Prow[None]
                Tk = T - k
                P_b = np.repeat(Prow[None, :], Tk, axis=0)   # (T-k, Ss)
                cov_static_trunc_only.append(P_b)
        else:
            for i, L in enumerate(lag_cov_list):
                Prow = np.asarray(cov_static_samples[i])
                if Prow.ndim >= 2: Prow = Prow.reshape(-1)
                if Prow.ndim == 0: Prow = Prow[None]
                Tk = np.asarray(L).shape[0]
                P_b = np.repeat(Prow[None, :], Tk, axis=0)
                cov_static_trunc_only.append(P_b)
                cov_static_new.append(np.concatenate([P_b, np.asarray(L)], axis=1))

    if is_image:
        del lag_cov_list
        gc.collect()
        
    
    data_outdir = os.path.join(opts.outdir, 'data')
    os.makedirs(data_outdir, exist_ok=True)
    np.savez(os.path.join(data_outdir, 'dataset_samples.npz'), samples=np.array(dset_samples_trunc, dtype=object))
    if cov_dynamic_trunc is not None:
        np.savez(os.path.join(data_outdir, 'cov_dynamic_samples.npz'), samples=np.array(cov_dynamic_trunc, dtype=object))


    if is_image:
        # For image data: save ONLY compact static (no lag) for downstream analysis, to avoid huge files / RAM spikes
        np.savez(os.path.join(data_outdir, 'cov_static_trunc_only.npz'),
                 samples=np.array(cov_static_trunc_only, dtype=object))
    else:
        # For non-image data: usually save is not very painful
        np.savez(os.path.join(data_outdir, 'lag_cov_list.npz'),
                 samples=np.array(lag_cov_list, dtype=object))
        np.savez(os.path.join(data_outdir, 'cov_static_new.npz'),
                 samples=np.array(cov_static_new, dtype=object))
    
    dist.print0(f"Saved processed data to: {data_outdir}")
        
    dset_samples = dset_samples_trunc
    cov_dynamic_samples = cov_dynamic_trunc
    cov_static_samples = cov_static_new
    
    # Infer data_dim from loaded data
    if is_image:
        # dset_samples_trunc element is (T, C, H, W)
        _, C, H, W = np.asarray(dset_samples_trunc[0]).shape
        opts.data_dim = int(C * H * W)
    else:
        # vectors: (T, D)
        opts.data_dim = int(np.asarray(dset_samples_trunc[0]).shape[1])
    working_data_dim = opts.data_dim

    if opts.dim_cov_dynamic is None:
        opts.dim_cov_dynamic = (cov_dynamic_samples[0].shape[1]
                                if cov_dynamic_samples is not None else 0)


    if opts.dim_cov_static is None:
        if is_image:
            # static (time-varying) part we kept is cov_static_trunc_only; lag is on-the-fly
            static_dim = (cov_static_trunc_only[0].shape[1] if len(cov_static_trunc_only) > 0 else 0)
            _T, _C, _H, _W = np.asarray(dset_samples_trunc[0]).shape
            lag_dim = (k + 1) * _C * _H * _W
            opts.dim_cov_static = static_dim + lag_dim
        else:
            if cov_static_samples is not None:
                opts.dim_cov_static = cov_static_samples[0].shape[1]
            else:
                opts.dim_cov_static = 0

    if is_image:
        cov_static_samples_for_dataset = cov_static_trunc_only
    else:
        cov_static_samples_for_dataset = cov_static_samples


    # Infer n_trajs
    inferred_n_trajs = len(dset_samples)
    
    # Initialize config dict.
    c = dnnlib.EasyDict()
    
    # Setup dataset kwargs
    balls_dset_specs = dnnlib.EasyDict(img_shape=[opts.data_imgshape, opts.data_imgshape], radius=opts.data_radius, blur=opts.data_blur) if opts.data_name.lower()=='balls' else None
    c.dataset_kwargs = dnnlib.EasyDict(dset_name=opts.data_name, dt=opts.dt, balls_dset_specs=balls_dset_specs)
    c.dataset_kwargs.n_trajs = inferred_n_trajs

    # Create dataset object with covariate support
    dataset_obj = dnnlib.util_v6.ToyDsetDynamics(
        data=dset_samples_trunc, dt=opts.dt, nForward=1,
        cov_dynamic_data=cov_dynamic_trunc,
        cov_static_data=cov_static_samples_for_dataset,
        # NEW args used only for images; no-ops for vectors
        image_lag_source=(dset_samples_orig if is_image else None),
        lag_k=(k if is_image else 0)
    )

    c.dataset_obj = dataset_obj
    c.dset_samples = dset_samples_trunc
    c.cov_dynamic_samples = cov_dynamic_trunc
    c.cov_static_samples = cov_static_samples_for_dataset
    
    # Setup dataloader kwargs
    c.data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True, 
        num_workers=opts.workers, 
        prefetch_factor=2
    )
    
    # Setup optimizer kwargs
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    
    # Setup loss kwargs
    c.loss_kwargs = dnnlib.EasyDict(flow_matcher_type=opts.flow_matcher_type, 
                                    sigma_dynamics=opts.sigma_dyn_fm, 
                                    sigma_compression=opts.sigma_comp_fm, 
                                    normalize_lie=opts.norm_lie,
                                    cmp_is=opts.cmp_is,
                                    dyn_is=opts.dyn_is,
                                    is_bins_tau=opts.is_bins_tau,
                                    is_bins_t=opts.is_bins_t,
                                    is_ema=opts.is_ema,
                                    is_eps=opts.is_eps,
                                    class_name='training.loss_v6.VFMToyLoss')
    
    # Setup network kwargs
    c.network_kwargs = dnnlib.EasyDict(
        dyn_model_type=opts.dyn_arch, 
        use_t_dyn=opts.use_t_dyn,
        flow_model_type=opts.flow_arch, 
        encoder_type=opts.encoder_arch, 
        channels=[32, 64, 128, 256], 
        conv_embed_dim=256, 
        data_dim=working_data_dim, 
        dims_to_keep=opts.dims_to_keep, 
        depth_mlp_cmp=opts.mlp_depth_cmp, 
        width_mlp_cmp=opts.mlp_width_cmp, 
        depth_mlp_dyn=opts.mlp_depth_dyn, 
        width_mlp_dyn=opts.mlp_width_dyn,
        depth_encoder=opts.encoder_depth, 
        width_encoder=opts.encoder_width, 
        cd_eps=opts.eps, 
        d_min=opts.d_min, 
        img_size=opts.data_imgshape, 
        in_ch=opts.data_inch,
        include_x0_tau=opts.include_x0_tau,
        dim_cov_dynamic=opts.dim_cov_dynamic,
        dim_cov_static=opts.dim_cov_static,
        encoder_rank=opts.encoder_rank,
        class_name='training.networks_v6.VFMToyNet'
    )
    
    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    hist_lag_dim = 0
    hist_static_dim = 0
    if (getattr(opts, 'lag_k', 0) or 0) >= 0 and (getattr(opts, 'dim_cov_static', 0) or 0) > 0:
        k = int(opts.lag_k)
        if _is_movie_trials(dset_samples_trunc):
            _T, _C, _H, _W = np.asarray(dset_samples_trunc[0]).shape
            base = _C * _H * _W
        else:
            base = int(opts.data_dim)
        
        hist_lag_dim    = (k + 1) * base
        hist_static_dim = max(0, int(opts.dim_cov_static or 0) - hist_lag_dim)

        expected = int(opts.dim_cov_static or 0)
        actual   = hist_static_dim + hist_lag_dim
        if actual != expected:
            raise ValueError(f"dim_cov_static mismatch: expected {expected}, got {actual}")
    
    # Pass knobs to training loop (via **c)
    c.hist_noise_std = float(getattr(opts, 'hist_noise_std', 0.0) or 0.0)
    c.hist_lag_dim = int(hist_lag_dim)
    c.hist_static_dim = int(hist_static_dim)
    
    c.use_ema = opts.use_ema
    c.ema_halflife_kimg = int(opts.ema * 1000) # only used if use_ema==True 
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)
    c.update(alpha=opts.alpha, beta=opts.beta, gamma=opts.gamma, eta=opts.eta)
    c.update(eta_post=opts.eta_post)
    c.update(hold_kimg=opts.eta_hold_kimg, ramp_kimg=opts.eta_ramp_kimg)
    c.update(grad_clip=opts.grad_clip, grad_clip_val=opts.grad_clip_val, alpha_mu=opts.alpha_mu, 
             kl_eps=opts.kl_eps, kl_use_dims_to_keep=opts.kl_use_dims_to_keep,
             kl_warmup_kimg=opts.kl_warmup_kimg, kl_ramp_kimg=opts.kl_ramp_kimg)
    c.update(pre_train=opts.pre_train, pre_train_kimgs=opts.pre_train_kimgs)

    # PCA init knobs → training_loop
    c.init_latent             = opts.init_latent
    c.init_latent_steps       = opts.init_latent_steps
    c.init_latent_lr          = opts.init_latent_lr
    c.init_latent_max_samples = opts.init_latent_max_samples
    c.init_latent_tol          = opts.init_latent_tol
    c.init_latent_patience     = opts.init_latent_patience
    c.init_latent_min_steps    = opts.init_latent_min_steps
    
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
    serial_c = {k: v for k, v in c.items() if k not in ['dataset_obj', 'dset_samples', 
                                                        'cov_dynamic_samples', 'cov_static_samples']}

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(serial_c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset name:            {c.dataset_kwargs.dset_name}')
    dist.print0(f'Data dimension:          {opts.data_dim}')
    dist.print0(f'Dynamic covariate dimension:     {opts.dim_cov_dynamic}')
    dist.print0(f'use_t_dyn={opts.use_t_dyn}')
    dist.print0(f'Static covariate dimension:      {opts.dim_cov_static}')
    dist.print0(f'Lag k used:               {opts.lag_k}')
    dist.print0(f'Include x0_tau:          {opts.include_x0_tau}')
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
        dnnlib.util_v6.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)


    # Train.
    toy_training_loop_vfm_v6.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------