#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actually does training for toy VFM models.

"""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from torch.utils.tensorboard import SummaryWriter
from torch_cfm import conditional_flow_matching as cfm
import torch.serialization
import inspect
from torch_utils import persistence
import torch.nn as nn
import gc

def g_only(old_const, grad_term):
    return (old_const - grad_term).detach() + grad_term

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for constructing dataloader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    use_ema             = True,     # Whether or not to apply EMA to model weights.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    alpha               = 1.0,      #scale for flow net loss 
    beta                = 1.0,      #scale for dynamics net loss
    gamma               = 1.0,      #scale for Lie derivative loss
    eta                 = 1.0,      #scale for encoder loss 
    grad_clip           = False,    #whether or no to apply grad norm clipping to model params
    grad_clip_val       = None,     #val to clip model grad norms to.
    pre_train           = False,    #whether or not to pre-train nets.
    pre_train_kimgs     = 0,        # how many Kimgs to run pre-training for 
    dataset_obj         = None,     # Pre-generated dataset object.
    dset_samples        = None,     # Pre-generated dataset samples.
    cov_dynamic_samples = None,     # ADDED: To match notebook call (ignored, as dataset_obj handles covs)
    cov_static_samples  = None,
    alpha_mu = .1,
    kl_use_dims_to_keep=False,
    kl_eps=1e-6,
    kl_warmup_kimg = 1000,
    kl_ramp_kimg = 5000,
    init_latent='random',
    init_latent_steps=50000,
    init_latent_lr=1e-3,
    init_latent_max_samples=100000,
    init_latent_tol=1e-5,
    init_latent_patience=100,
    init_latent_min_steps=1000,
    eta_post = 1.0,
    hold_kimg = 1000, 
    ramp_kimg = 5000,
    hist_noise_std=0.0,
    hist_lag_dim=0,
    hist_static_dim=0,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


    kl_stats = {'sum1': None, 'sum2': None, 'Sxxk': None, 'Sxxt': None, 'B': None}
    beta_ema = 0.90  # strong smoothing, keeps 0.1 gradient from current batch
    def blend(curr, prev):
        return curr if prev is None else (1.0 - beta_ema)*curr + beta_ema*prev.detach()

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    
    # Set up our TB writer
    dist.print0('Creating TB dir and writer...')
    if dist.get_rank() == 0:
        writer_dir = os.path.join(run_dir, 'TB_logs')
        os.makedirs(writer_dir, exist_ok=True) 
        writer = SummaryWriter(log_dir = writer_dir)
        
    # Use provided dataset_obj and dset_samples
    dist.print0('Using provided dataset...')
    
    
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)  
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, \
                                                        batch_size=batch_gpu, **data_loader_kwargs))
    
    # Construct u, v networks 
    dist.print0('Constructing network...')
    net = dnnlib.util_v6.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module
    
    net.train().requires_grad_(True).to(device)

    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.data_dim], device=device)
            ts = torch.ones([batch_gpu], device=device)
            test_flow_matcher = cfm.ConditionalFlowMatcher(sigma=0.1)

            
            cov_dynamic_0 = None
            cov_dynamic_dt = None
            cov_static = None
            if (hasattr(net, 'dim_cov_dynamic') and net.dim_cov_dynamic > 0) or \
               (hasattr(net, 'dim_cov_static') and net.dim_cov_static > 0):
                if hasattr(net, 'dim_cov_dynamic') and net.dim_cov_dynamic > 0:
                    cov_dynamic_0 = torch.zeros([batch_gpu, net.dim_cov_dynamic], device=device)
                    cov_dynamic_dt = torch.zeros([batch_gpu, net.dim_cov_dynamic], device=device)
                if hasattr(net, 'dim_cov_static') and net.dim_cov_static > 0:
                    cov_static = torch.zeros([batch_gpu, net.dim_cov_static], device=device)
                misc.print_module_summary(net, [images, images, ts, ts, 1e-3, test_flow_matcher, test_flow_matcher,
                                                cov_dynamic_0, cov_dynamic_dt, cov_static], max_nesting=2)  
            else:
                misc.print_module_summary(net, [images, images, ts, ts, 1e-3, test_flow_matcher, test_flow_matcher],
                                          max_nesting=2)


    # -------------------------------
    # Optional: PCA-based latent init
    # -------------------------------
    if init_latent == 'pca' and init_latent_steps > 0 and dset_samples is not None:
        dist.print0(f'[PCA init] collecting samples (≤ {init_latent_max_samples})...')
        flat = []
        took = 0
        D = int(getattr(net, 'data_dim', None) or (dset_samples[0].shape[-1] if dset_samples else 0))
        for arr in dset_samples:
            a = np.asarray(arr)
            if a.ndim == 4:  # (T, C, H, W) -> flatten per-frame
                T, C, H, W = a.shape
                a = a.reshape(T, C * H * W)
            elif a.ndim == 3:  # (T, H, W) -> add channel then flatten
                T, H, W = a.shape
                a = a.reshape(T, 1 * H * W)
            # else assume (T, D)
            a = a.astype(np.float32)
            need = max(0, min(init_latent_max_samples - took, a.shape[0]))
            if need > 0:
                flat.append(a[:need, :D])
                took += need
            if took >= init_latent_max_samples:
                break
                
        if took > 0:
            X = np.concatenate(flat, axis=0).astype(np.float32)
            # K = int(getattr(net, 'dims_to_keep', min(D, 2)))
            K = int(getattr(net, 'dims_to_keep', min(D, 2)))
            K = max(1, min(K, min(X.shape[0], X.shape[1])))
            
            # Xm = X.mean(0, keepdims=True)
            # Xc = X - Xm
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            PCs = Vt[:K]               # (K, D)
            target_proj = X @ PCs.T    # (N, K)
            
            enc = net.encoder
            
            # -- device helpers
            cap   = enc.d_max_diag.squeeze(0).to(device)    # [d_total]
            floor = enc.d_min_diag.squeeze(0).to(device)    # [d_total]
            d_total = int(cap.numel())
            assert d_total >= K, "encoder latent dim must be >= K"

            Y_t = torch.from_numpy(target_proj.astype(np.float32)).to(device)[:, :K]  # [N,K]
            Ntot = max(Y_t.shape[0], 1)
            eps = 1e-8
            
            # Σ ≈ (Y^T Y)/N (uncentered)
            SigmaK = (Y_t.T @ Y_t) / float(Ntot)
            SigmaK = SigmaK + eps * torch.eye(K, device=device, dtype=Y_t.dtype)
            
            # Cholesky: Σ = Lc_k Lc_k^T
            Lc_k   = torch.linalg.cholesky(SigmaK)          # [K,K] lower
            diagLc = torch.diagonal(Lc_k)                   # [K]
            D_head = (diagLc ** 2).clamp_min(eps)           # principal variances
            inv_d  = (1.0 / diagLc).clamp_max(1e12)
            L_head_unit = Lc_k @ torch.diag(inv_d)          # unit-lower version
            
            # Write into encoder: D and (unit-lower) L into enc.U (strictly lower)
            
            with torch.no_grad():
                D_full = floor.clone()
                D_full[:K] = torch.clamp(D_head, min=floor[:K], max=cap[:K])
                gap = (cap[K:] - floor[K:]).clamp_min(0)
                tau = 1e-2  # small fraction into the allowed range; try 1e-3 or 1e-2
                D_tail = floor[K:] + tau * gap
                D_full[K:] = torch.where(gap > 0, D_tail, floor[K:])  # if cap==floor, we can't go higher
                
                enc.d.copy_(torch.log(D_full))
            
                if hasattr(enc, 'U') and isinstance(enc.U, torch.nn.Parameter):
                    Uh, Uw = enc.U.shape
                    rows = min(Uh, K); cols = min(Uw, K)
                    U_fill = torch.zeros_like(enc.U)
                    U_fill[:rows, :cols] = torch.tril(L_head_unit[:rows, :cols], diagonal=-1)
                    enc.U.copy_(U_fill)
            
            # Targets for μ from non-centered relation Y ≈ Lc_k sqrt(D_head) μ  ⇒ μ = Lc_k^{-1} Y
            MU_t = torch.zeros((Y_t.shape[0], d_total), device=device, dtype=Y_t.dtype)
            MU_t[:, :K] = torch.linalg.solve(Lc_k, Y_t.T).T
            
            # Fit ONLY the mu branch to MU_t (L & D stay as initialized above)
            for p in enc.parameters():
                p.requires_grad_(False)
            mu_params = []
            for n, p in enc.named_parameters():
                if n.startswith('mu') or '.mu.' in n:
                    p.requires_grad_(True)
                    mu_params.append(p)
            pca_opt = torch.optim.Adam(mu_params, lr=init_latent_lr)
            
            # Prepare inputs for the loop
            X_t = torch.from_numpy(X).to(device)                 # [N,D]
            n   = X_t.shape[0]
            ptr = 0
            enc.train()
            ema = None; alpha_ema = 0.1; step = 0
            max_steps = int(init_latent_steps) if init_latent_steps is not None else 0
            bs = int(min(1024, max(32, batch_gpu or 32)))
            print_every = max(1, (max_steps // 100) if max_steps else 50)
            min_steps  = int(init_latent_min_steps)
            tol        = float(init_latent_tol)
            patience   = int(init_latent_patience)
            
            
            while True:
                if ptr >= n:
                    ptr = 0
                j  = min(bs, n - ptr)
                xb = X_t[ptr:ptr + j]
                tb = MU_t[ptr:ptr + j]
                ptr += j
            
                pca_opt.zero_grad(set_to_none=True)
                mu_pred, _, _ = enc.encode(xb)                  # only μ has grads
                loss_pca = torch.mean((mu_pred[:, :K] - tb[:, :K]) ** 2)
                loss_pca.backward()
                pca_opt.step()
            
                step += 1
                val = float(loss_pca.item())
                ema = val if ema is None else (alpha_ema * val + (1.0 - alpha_ema) * ema)
                if step % print_every == 0 or step == 1:
                    dist.print0(f'[PCA init] step {step}: mse={val:.6g}, ema={ema:.6g}')
                    
                if max_steps > 0 and step >= max_steps:
                    dist.print0(f'[PCA init] reached step cap {max_steps}')
                    break
            
            # 4) Re-enable grads & report
            for p in enc.parameters():
                p.requires_grad_(True)
            
            with torch.no_grad():
                d_vec = torch.exp(enc.d).to(device).clamp(min=floor, max=cap)
                if dist.get_rank() == 0:
                    dist.print0(f'[PCA init] d_vec head stats: min={d_vec[:K].min().item():.3e}, max={d_vec[:K].max().item():.3e}')
                    dist.print0(f'[PCA init] d_vec tail stats: min={d_vec[K:].min().item():.3e}, max={d_vec[K:].max().item():.3e}')
            dist.print0('[PCA init] done.')
                        

    gc.collect()
    torch.cuda.empty_cache()
    
    # Setup optimizer and lossfn
    dist.print0('Setting up optimizer and loss fn...')
    loss_fn = dnnlib.util_v6.construct_class_by_name(**loss_kwargs) 
    optimizer = dnnlib.util_v6.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    
    #init Dist mode for nets
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[torch.cuda.current_device()],
                                                    broadcast_buffers=False, find_unused_parameters=True)

    
    #copy original model weights into EMA (if using it)
    if use_ema: 
        ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        
        if use_ema: 
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False) 
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        else:
            misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
            
        del data # conserve memory
        
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')

        all_nn_classes = [obj for name, obj in inspect.getmembers(nn) if inspect.isclass(obj) and obj.__module__.startswith('torch.nn')]
        torch.serialization.add_safe_globals(all_nn_classes + [persistence._reconstruct_persistent_obj])
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    #make sure pre_train flag is correctly set if resuming exp
    if (cur_nimg >= pre_train_kimgs * 1000):
        pre_train=False 
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    gs = 0 
    loss_scales = torch.Tensor([alpha, beta, eta, gamma]).type(torch.float32).to(device)
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        tot_scalar_loss = 0
        tot_separate_losses = torch.zeros(4).type(torch.float32).to(device)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)): 
                batch_data = next(dataset_iterator)

                x0_1 = batch_data[0].type(torch.float32).reshape(batch_data[0].shape[0], -1).to(device)
                xdt_1 = batch_data[1].type(torch.float32).reshape(batch_data[1].shape[0], -1).to(device)
                # dt is batch_data[2], already handled in loss function

                idx = 3
                cov_dynamic_0 = None
                cov_dynamic_dt = None
                cov_static = None

                if hasattr(net, 'dim_cov_dynamic') and net.dim_cov_dynamic > 0:
                    cov_dynamic_0 = batch_data[idx].type(torch.float32).to(device)
                    cov_dynamic_dt = batch_data[idx+1].type(torch.float32).to(device)
                    idx += 2

                # if hasattr(net, 'dim_cov_static') and net.dim_cov_static > 0:
                #     cov_static = batch_data[idx].type(torch.float32).to(device)
                if hasattr(net, 'dim_cov_static') and net.dim_cov_static > 0:
                    cov_static = batch_data[idx].type(torch.float32).to(device)
                    if hist_noise_std > 0 and hist_lag_dim > 0:
                        start = hist_static_dim
                        cov_static[:, start:start+hist_lag_dim] += torch.randn_like(
                            cov_static[:, start:start+hist_lag_dim]
                        ) * float(hist_noise_std)
                
                
                loss = loss_fn(net=ddp, x0_1=x0_1, xdt_1=xdt_1, dt=dataset_kwargs.dt, 
                      pre_training=pre_train, 
                      cov_dynamic_0=cov_dynamic_0, cov_dynamic_dt=cov_dynamic_dt,
                      cov_static=cov_static)

                # 09/14 -----------------------------------------------
                if pre_train:
                    eta_curr = float(eta)
                else:
                    kimg_after = max(cur_nimg - pre_train_kimgs * 1000, 0) / 1000.0
                    if kimg_after <= hold_kimg:
                        eta_curr = float(eta)
                    else:
                        t = min(1.0, (kimg_after - hold_kimg) / max(ramp_kimg, 1e-8))
                        eta_curr = (1.0 - t) * float(eta) + t * float(eta_post)
        
                loss_scales[2] = eta_curr
                if dist.get_rank() == 0:
                    writer.add_scalar('eta/weight', eta_curr, gs)
                # -----------------------------------------------------

                
                # Scale losses
                loss = loss * loss_scales[:, None, None] 
                
                # Log losses
                training_stats.report('Loss/loss', torch.sum(loss.detach(), dim=0))
                
                # Compute scalar loss for backprop
                round_scalar_loss = loss.sum() * (loss_scaling / batch_gpu_total)


                # KL of empirical mu distribution to N(0,1)
                # goal: 1) decorrelate latent & 2) control magnitude of latent, so that it won't compensate for d
                # 2 options:
                # 1) full: brute force implementation for low-D data
                # 2) kl_use_dims_to_keep: full covariance KL for the active dims_to_keep, diagonal cov KL for surpressed & de-correlate blocks
                enc = getattr(ddp.module, 'encoder', ddp.module)
                mu_pre, _, _ = enc.encode(x0_1)                       # [B,d]
                B_local, d = mu_pre.shape
                dev, dtype = mu_pre.device, mu_pre.dtype
                
                # choose K = dims_to_keep if provided, else all
                K = d if not kl_use_dims_to_keep else int(network_kwargs.get('dims_to_keep', d))
                K = max(0, min(K, d))
                shrink = 0.05
                eps = float(kl_eps)

                # -----------------------------------------------------
                # # old version: works, but warning...

                # sum1_local = mu_pre.sum(dim=0)                                     # [d]
                # sum2_local = (mu_pre * mu_pre).sum(dim=0)                          # [d]
                # Sxxk_local = mu_pre[:, :K].T @ mu_pre[:, :K] if K > 0 else None    # [K,K]
                # Sxxt_local = (mu_pre[:, :K].T @ mu_pre[:, K:]) if (0 < K < d) else None
                # B_local_t  = torch.tensor([B_local], device=dev, dtype=torch.float32)
                
                # # all-reduce to global
                # if torch.distributed.is_available() and torch.distributed.is_initialized():
                #     torch.distributed.all_reduce(B_local_t)
                #     torch.distributed.all_reduce(sum1_local)
                #     torch.distributed.all_reduce(sum2_local)
                #     if Sxxk_local is not None: torch.distributed.all_reduce(Sxxk_local)
                #     if Sxxt_local is not None: torch.distributed.all_reduce(Sxxt_local)
                
                # prev_sum1 = None if kl_stats['sum1'] is None else kl_stats['sum1'].detach()
                # prev_sum2 = None if kl_stats['sum2'] is None else kl_stats['sum2'].detach()
                # prev_Sxxk = None if kl_stats['Sxxk'] is None else kl_stats['Sxxk'].detach()
                # prev_Sxxt = None if kl_stats['Sxxt'] is None else kl_stats['Sxxt'].detach()
                # prev_B    = None if kl_stats['B']    is None else kl_stats['B'].detach()
                
                # sum1_ema = blend(sum1_local, prev_sum1)
                # sum2_ema = blend(sum2_local, prev_sum2)
                # Sxxk_ema = None if Sxxk_local is None else blend(Sxxk_local, prev_Sxxk)
                # Sxxt_ema = None if Sxxt_local is None else blend(Sxxt_local, prev_Sxxt)
                # B_ema_t  = blend(B_local_t, prev_B)
                
                # # write DETACHED copies into state to avoid graph retention
                # kl_stats['sum1'] = sum1_ema.detach()
                # kl_stats['sum2'] = sum2_ema.detach()
                # if Sxxk_ema is not None: kl_stats['Sxxk'] = Sxxk_ema.detach()
                # if Sxxt_ema is not None: kl_stats['Sxxt'] = Sxxt_ema.detach()
                # kl_stats['B'] = B_ema_t.detach()

                # new version: no warning...
                # local stats WITH grad (for the KL gradient path)
                sum1_loc_g = mu_pre.sum(dim=0)                                     # [d]
                sum2_loc_g = (mu_pre * mu_pre).sum(dim=0)                          # [d]
                Sxxk_loc_g = mu_pre[:, :K].T @ mu_pre[:, :K] if K > 0 else None    # [K,K]
                Sxxt_loc_g = (mu_pre[:, :K].T @ mu_pre[:, K:]) if (0 < K < d) else None
                B_local_t  = torch.tensor([B_local], device=dev, dtype=torch.float32)
                
                # detached COPIES for cross-GPU comms (to avoid autograd warning)
                sum1_local = sum1_loc_g.detach().clone()
                sum2_local = sum2_loc_g.detach().clone()
                Sxxk_local = None if Sxxk_loc_g is None else Sxxk_loc_g.detach().clone()
                Sxxt_local = None if Sxxt_loc_g is None else Sxxt_loc_g.detach().clone()
                
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    with torch.no_grad():
                        torch.distributed.all_reduce(B_local_t)
                        torch.distributed.all_reduce(sum1_local)
                        torch.distributed.all_reduce(sum2_local)
                        if Sxxk_local is not None: torch.distributed.all_reduce(Sxxk_local)
                        if Sxxt_local is not None: torch.distributed.all_reduce(Sxxt_local)
                
                # previous EMA states (constants)
                prev_sum1 = None if kl_stats['sum1'] is None else kl_stats['sum1'].detach()
                prev_sum2 = None if kl_stats['sum2'] is None else kl_stats['sum2'].detach()
                prev_Sxxk = None if kl_stats['Sxxk'] is None else kl_stats['Sxxk'].detach()
                prev_Sxxt = None if kl_stats['Sxxt'] is None else kl_stats['Sxxt'].detach()
                prev_B    = None if kl_stats['B']    is None else kl_stats['B'].detach()
                
                # EMA as CONSTANTS (forward values)
                sum1_ema_c = sum1_local if prev_sum1 is None else (1.0 - beta_ema)*sum1_local + beta_ema*prev_sum1
                sum2_ema_c = sum2_local if prev_sum2 is None else (1.0 - beta_ema)*sum2_local + beta_ema*prev_sum2
                Sxxk_ema_c = None if Sxxk_local is None else (Sxxk_local if prev_Sxxk is None else (1.0 - beta_ema)*Sxxk_local + beta_ema*prev_Sxxk)
                Sxxt_ema_c = None if Sxxt_local is None else (Sxxt_local if prev_Sxxt is None else (1.0 - beta_ema)*Sxxt_local + beta_ema*prev_Sxxt)
                B_ema_c    = B_local_t if prev_B is None else (1.0 - beta_ema)*B_local_t + beta_ema*prev_B
                
                # stop-grad glue: forward equals EMA constants, gradient equals (1-β)*local stats
                
                sum1_ema = g_only(sum1_ema_c, (1.0 - beta_ema)*sum1_loc_g)
                sum2_ema = g_only(sum2_ema_c, (1.0 - beta_ema)*sum2_loc_g)
                Sxxk_ema = None if Sxxk_ema_c is None else g_only(Sxxk_ema_c, (1.0 - beta_ema)*Sxxk_loc_g)
                Sxxt_ema = None if Sxxt_ema_c is None else g_only(Sxxt_ema_c, (1.0 - beta_ema)*Sxxt_loc_g)
                B_ema_t  = B_ema_c.detach()  # B has no grad path anyway

                
                # write DETACHED copies for next step's state (no graph retention)
                kl_stats['sum1'] = sum1_ema_c.detach()
                kl_stats['sum2'] = sum2_ema_c.detach()
                if Sxxk_ema_c is not None: kl_stats['Sxxk'] = Sxxk_ema_c.detach()
                if Sxxt_ema_c is not None: kl_stats['Sxxt'] = Sxxt_ema_c.detach()
                kl_stats['B'] = B_ema_c.detach()

                # ----------------------------------------------------------------------------------

                Btot = max(1.0, float(B_ema_t.item()))
                m    = (sum1_ema / Btot)

                kl = mu_pre.new_tensor(0.0)
                
                # ---- full-cov KL on first K dims (global) ----
                if K > 0:
                    Ekxx = (Sxxk_ema / Btot)                                     # E[xxᵀ]
                    mk = m[:K]
                    Ck = Ekxx - mk[:, None] @ mk[None, :]                          # Cov
                    Ck = 0.5 * (Ck + Ck.T)                                         # symmetrize
                    trCk = torch.trace(Ck)
                    # Ledoit-Wolf style fixed shrinkage
                    eyeK = torch.eye(K, device=dev, dtype=dtype)
                    avg = (trCk / K).clamp_min(eps)
                    Ck = (1.0 - shrink) * Ck + shrink * avg * eyeK
                
                    # robust logdet with cholesky + jitter escalate
                    jitter = eps
                    max_retries = 5
                    ok = False
                    for _ in range(max_retries + 1):
                        try:
                            L = torch.linalg.cholesky(Ck + jitter * eyeK)
                            logdet_Ck = 2.0 * torch.log(torch.diag(L)).sum()
                            ok = True
                            break
                        except RuntimeError:
                            jitter *= 10.0
                    if not ok:
                        eigvals = torch.linalg.eigvalsh(Ck).clamp_min(eps)
                        logdet_Ck = torch.log(eigvals).sum()
                
                    kl = kl + 0.5 * (trCk + (mk @ mk) - K - logdet_Ck)
                
                # ---- diagonal KL on remaining dims (global) ----
                if K < d:
                    mr = m[K:]
                    Erx2 = (sum2_ema[K:] / Btot)
                    vr = (Erx2 - mr * mr).clamp_min(eps)
                    # mild shrinkage of tail variances toward their mean
                    vr_mean = vr.mean().clamp_min(eps)
                    vr = (1.0 - shrink) * vr + shrink * vr_mean
                    kl = kl + 0.5 * (vr + mr*mr - 1.0 - torch.log(vr)).sum()
                
                # ---- block decorrelation (centered, global) ----
                if 0 < K < d:
                    Cht = (Sxxt_ema / Btot) - m[:K, None] @ m[None, K:]          # [K, d-K]
                    kl = kl + 0.5 * (Cht * Cht).sum()

                
                # keep your existing schedule/weighting exactly
                kl_weight = float(alpha_mu) * max(0.0, min((cur_nimg - kl_warmup_kimg*1000) / (kl_ramp_kimg*1000), 1.0))
                round_scalar_loss = round_scalar_loss + kl_weight * kl

                # -----------------------------------------------------------------------------
                
                # Get separate losses for TB logging
                separate_losses = torch.sum(torch.sum(loss.detach(), dim=2), dim=1) * (loss_scaling / batch_gpu_total)
                tot_separate_losses += separate_losses
                
                # Accumulate loss
                tot_scalar_loss += round_scalar_loss.item()
                
                # Backpropagate
                round_scalar_loss.backward()


        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        
        #make sure all grads are numbers
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        
        #clip model gradient norms, if desired
        if grad_clip: 
            assert grad_clip_val != None, 'Need a value to clip grads to!'
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
        
        #then step 
        optimizer.step()

        # Update EMA, if using it
        if use_ema: 
            ema_halflife_nimg = ema_halflife_kimg * 1000 
            if ema_rampup_ratio is not None: 
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8)) 
            for p_ema, p_net in zip(ema.parameters(), net.parameters()): 
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))           
        
        #Log loss to TB
        if dist.get_rank()==0: 
            writer.add_scalar('tot_train_loss', tot_scalar_loss, gs)
            writer.add_scalar('flow_train_loss', tot_separate_losses[0].item(), gs)
            writer.add_scalar('dyn_train_loss', tot_separate_losses[1].item(), gs)
            writer.add_scalar('enc_recon_loss', tot_separate_losses[2].item(), gs)
            writer.add_scalar('lie_derivative_loss', tot_separate_losses[3].item(), gs)
            writer.add_scalar('Kimgs', cur_nimg/1000, gs)
            
            # 09/14:add
            writer.add_scalar('kl_mu/weight', kl_weight, gs)
            writer.add_scalar('kl_mu/value', float(kl.item()), gs)
            gs+=1

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        
        #check if pre-training time is done... 
        if (cur_nimg >= pre_train_kimgs * 1000):
            pre_train = False
        
        
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        # fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        # fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        # fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        
        fields += [f"cmp_loss {tot_separate_losses[0].item():<7.3f}"]
        fields += [f"dyn_loss {tot_separate_losses[1].item():<7.3f}"]
        fields += [f"con_loss {tot_separate_losses[2].item():<7.3f}"]
        fields += [f"total_loss {tot_scalar_loss:<7.3f}"]
        
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            if use_ema: 
                data = dict(ema=ema, loss_fn=loss_fn, dataset_kwargs=dict(dataset_kwargs))
            else:
                data = dict(net=net, loss_fn=loss_fn, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):

                    m = copy.deepcopy(value).eval().requires_grad_(False).to(device)
                    try:
                        enc = getattr(m, 'encoder', None)
                        if enc is not None and hasattr(enc, 'compute_sort_perm') and hasattr(enc, 'apply_perm_inplace'):
                            dev = next(enc.parameters()).device  # encoder param device (CUDA)
                            # Ensure non-buffer tensors used during perm live on the same device
                            for name in ('d_max_diag', 'd_min_diag'):
                                if hasattr(enc, name):
                                    t = getattr(enc, name)
                                    if isinstance(t, torch.Tensor) and t.device != dev:
                                        setattr(enc, name, t.to(dev))
                    
                            perm = enc.compute_sort_perm(topk=getattr(enc, 'dims_to_keep', None))
                            if perm.device != dev:
                                perm = perm.to(dev)
                            enc.apply_perm_inplace(perm)
                    except Exception as e:
                        if dist.get_rank() == 0:
                            print(f"[warn] sort-on-save skipped: {type(e).__name__}: {e}")

                    misc.check_ddp_consistency(m)
                    data[key] = m.cpu()
                    del m
                else:
                    data[key] = value
                    
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        #log net param grads and ODE_sim/trajs to TB 
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and dist.get_rank() == 0:
            #log grads 
            for tag, value in net.named_parameters():
                if value.grad is not None:
                    writer.add_histogram(tag+"/grad", value.grad.cpu(), gs) 
            #log encoded trajs, if working with balls dset
            if dataset_kwargs.balls_dset_specs != None:
                gt_trajs = np.array(dset_samples) #pass gt_trajs to numpy
                gt_trajs = np.reshape(gt_trajs, (gt_trajs.shape[0], gt_trajs.shape[1], -1)) #flatten across img dims
                orig_traj, enc_traj = dnnlib.util_v6.sim_encoded_trajs(gt_trajs, net.encoder, device)
                orig_fig, enc_fig = dnnlib.util_v6.plot_encoded_trajs(orig_traj, enc_traj, \
                                                                   dataset_kwargs.balls_dset_specs.img_shape[0])
                
                writer.add_figure("gt_traj", orig_fig, gs)
                writer.add_figure("encoded_traj", enc_fig, gs)
                writer.flush()

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------