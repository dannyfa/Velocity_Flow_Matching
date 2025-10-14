#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actually does training for toy VFM models.

"""

import os
import re
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


 # -----------------------------------------------------------------------------------------------------

def training_loop(
    # a) output & runtime
    run_dir             = '.',      # Output directory.,
    seed                = 0,        # Global random seed.,
    device              = torch.device('cuda'),
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?,
    kimg_per_tick       = 50,       # Interval of progress prints.,
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.,
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.,
    total_kimg          = 200000,   # Training duration, in thousands of training images.,
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.,
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.,
    resume_kimg         = 0,        # Start from the given training progress.,
    use_ema             = True,     # Whether or not to apply EMA to model weights.,
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.,
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.,
    
    # b)data & loading
    dataset_kwargs      = {},       # Options for training set.,
    data_loader_kwargs  = {},       # Options for constructing dataloader.,
    dataset_obj         = None,     # Pre-generated dataset object.,
    dset_samples        = None,     # Pre-generated dataset samples.,
    cov_dynamic_samples = None,     # to match notebook call (ignored, as dataset_obj handles covs),
    cov_static_samples  = None,
    batch_size          = 512,      # Total batch size for one training iteration.,
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.,
    
    # c) networks, loss, optimizer
    network_kwargs      = {},       # Options for model and preconditioning.,
    loss_kwargs         = {},       # Options for loss function.,
    optimizer_kwargs    = {},       # Options for optimizer.,
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.,
    grad_clip           = False,    #whether or no to apply grad norm clipping to model params,
    grad_clip_val       = None,     #val to clip model grad norms to.,
    
    # d) phases & ramps
    pre_train           = False,    #whether or not to pre-train nets.,
    pre_train_kimgs     = 0,        # how many Kimgs to run pre-training for,
    pre_T_abs           = None,
    dip_thresh_scale    = 10.0,
    lie_thresh_scale    = 10.0,
    flow_thresh_scale   = 10.0,
    pretrain_early_on   = False,
    pre_lr_ramp_kimg    = None,     # Pre-train LR warmup length,
    pre_param_kimg      = 0,
    train_param_ramp_kimg= 0,
    pre_no_lie          = False,
    lr_pre              = None,
    lr_floor_pre_abs    = None,
    
    
    # e) weights / penalties
    alpha               = 1.0,      #scale for flow net loss,
    beta                = 1.0,      #scale for dynamics net loss,
    gamma               = 1.0,      #scale for Lie derivative loss,
    eta                 = 1.0,      #scale for encoder loss,
    eta_post            = 1.0,
    alpha_mu            = .1,
    dip_use_dims_to_keep= False,
    dip_eps             = 1e-6,
    
    # f) init-latent & history
    init_latent         = 'random', # can be PCA
    init_latent_steps   = 50000,
    init_latent_lr      = 1e-3,
    init_latent_max_samples= 100000,
    init_latent_tol     = 1e-5,
    init_latent_patience= 100,
    init_latent_min_steps= 1000,
    hist_noise_std      = 0.0,
    hist_lag_dim        = 0,
    hist_static_dim     = 0,
):
     # -----------------------------------------------------------------------------------------------------
    # some helper functions
    def g_only(old_const, grad_term):
        return (old_const - grad_term).detach() + grad_term
    
    def _eff_start(early_on, start_nimg):
        return 0 if early_on else (start_nimg if start_nimg is not None else None)
    
    def _ramp_t(start_nimg, dur_kimg, cur_nimg):
        dur = float(dur_kimg or 0.0) * 1000.0
        if dur <= 0: 
            return 1.0
        if start_nimg is None:
            return 0.0
        return max(0.0, min(1.0, (cur_nimg - start_nimg) / max(dur, 1e-8)))

    def _t_pre_if_active(armed_at, active_in_pre, pre_lr_done_nimg):
        if not active_in_pre:
            return 0.0
        # start point: respect early_on vs arming, but never before LR warmup end
        start_raw = 0 if pretrain_early_on else (armed_at if armed_at is not None else None)
        if start_raw is None:
            return 0.0
        start_eff = max(int(pre_lr_done_nimg or 0), int(start_raw))
        return _ramp_t(start_eff, pre_param_kimg, cur_nimg)
        
     # -----------------------------------------------------------------------------------------------------

    # (1) Loop entry preparation
    # (1a) Initialize
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    dip_stats = {'sum1': None, 'sum2': None, 'Sxxk': None, 'B': None}
    beta_ema = 0.90  # strong smoothing, keeps 0.1 gradient from current batch

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
    
    # (1b) Data
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)  
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, \
                                                        batch_size=batch_gpu, **data_loader_kwargs))

    # (1c) Build network & summary
    # u (compression flow) and v (dynamic flow) 
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
                misc.print_module_summary(net, [images, images, ts, ts, 1e-3,
                                                test_flow_matcher, test_flow_matcher,
                                                cov_dynamic_0, cov_dynamic_dt, cov_static],
                                          max_nesting=2)  
            else:
                misc.print_module_summary(net, [images, images, ts, ts, 1e-3,
                                                test_flow_matcher, test_flow_matcher],
                                          max_nesting=2)

    # (1d) initialization of encoder: PCA or random
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
            K = int(getattr(net, 'dims_to_keep', min(D, 2)))
            K = max(1, min(K, min(X.shape[0], X.shape[1])))
            
            Xm = X.mean(0, keepdims=True).astype(np.float32)
            Xc = X - Xm
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            PCs = Vt[:K]               # (K, D)
            target_proj = Xc @ PCs.T   # (N, K)
            
            enc = net.encoder
            
            # device helpers
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
                tau = 1e-2  
                D_tail = floor[K:] + tau * gap
                D_full[K:] = torch.where(gap > 0, D_tail, floor[K:])  # if cap==floor, we can't go higher
                enc.d.copy_(torch.log(D_full))
            
                if hasattr(enc, 'U') and isinstance(enc.U, torch.nn.Parameter):
                    Uh, Uw = enc.U.shape
                    rows = min(Uh, K); cols = min(Uw, K)
                    # Make a PSD matrix S_K so that tril(S_K) off-diag equals desired strictly-lower
                    L_strict_K = torch.tril(L_head_unit[:rows, :rows], diagonal=-1)  # [rows, rows]
                    I_K  = torch.eye(rows, device=device, dtype=Y_t.dtype)
                    S_K  = L_strict_K + L_strict_K.T + I_K
                    U_K  = torch.linalg.cholesky(S_K)  # lower-triangular [rows, rows]
                    U_fill = torch.zeros_like(enc.U)
                    U_fill[:rows, :rows] = U_K
                    enc.U.copy_(U_fill)

            # Build A = L sqrt(D) exactly like encode() would, then solve A \mu \approx X
            X_t = torch.from_numpy(X).to(device)  # [N, D]
            
            with torch.no_grad():
                d_vec = torch.exp(enc.d).to(device)
                d_vec = torch.clamp(d_vec, min=floor.squeeze(0), max=cap.squeeze(0))
                Up = enc.U.to(device)
                L_raw = torch.tril(Up @ Up.T)
                L_mat = L_raw - torch.diag_embed(torch.diag(L_raw))
                L_mat = L_mat + torch.eye(L_raw.size(0), device=device, dtype=L_raw.dtype)

            sqrtD = torch.sqrt(d_vec).clamp_min(1e-8)
            
            A = L_mat.contiguous()
            B = X_t.T.contiguous()
            yT = torch.linalg.solve(A, B)
            torch.cuda.synchronize()  # force any kernel error to show here
            MU_t = (yT.T / sqrtD.unsqueeze(0))
            
            # yT = torch.linalg.solve(L_mat, X_t.T)
            # MU_t = (yT.T / sqrtD.unsqueeze(0))
            
            for p in enc.parameters():
                p.requires_grad_(True)
            if hasattr(enc, 'd') and isinstance(enc.d, torch.nn.Parameter):
                enc.d.requires_grad_(False)
            if hasattr(enc, 'U') and isinstance(enc.U, torch.nn.Parameter):
                enc.U.requires_grad_(False)
            
            pca_opt = torch.optim.Adam([p for p in enc.parameters() if p.requires_grad], lr=init_latent_lr)

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

            # PCA initialization loop
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
            
            # Re-enable grads & report
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
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for t in net.state_dict().values():
            if torch.is_tensor(t):
                torch.distributed.broadcast(t, src=0)

    # (1e) Loss, optimizer & learning rate schedule
    # Setup optimizer and lossfn
    dist.print0('Setting up optimizer and loss fn...')
    loss_fn = dnnlib.util_v6.construct_class_by_name(**loss_kwargs)
    
    try:
        if resume_pkl is not None:
            with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
                _snap = pickle.load(f)
            lf = _snap.get('loss_fn', None)
            if lf is not None and hasattr(lf, 'm_tau') and hasattr(lf, 'm_dyn'):
                if lf.m_tau.shape == loss_fn.m_tau.shape and lf.m_dyn.shape == loss_fn.m_dyn.shape:
                    loss_fn.m_tau.data.copy_(lf.m_tau.to(loss_fn.m_tau.device))
                    loss_fn.m_dyn.data.copy_(lf.m_dyn.to(loss_fn.m_dyn.device))
                    dist.print0('[resume] Restored IS tables (m_tau/m_dyn) from snapshot.')
            del _snap
    except Exception as e:
        dist.print0(f'[resume] IS-table restore skipped: {repr(e)}')

    abs_floor_train = float(optimizer_kwargs.pop('lr_floor_abs', 1e-4))
    optimizer = dnnlib.util_v6.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer

    #init Dist mode for nets
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[torch.cuda.current_device()],
                                                    broadcast_buffers=False, find_unused_parameters=True)

    # Adaptive LR + Pre-train controller
    base_lr_train = float(optimizer_kwargs.get('lr', 1e-3))
    base_lr_pre = float(lr_pre) if (lr_pre is not None) else (10.0 * base_lr_train)
    abs_floor_pre = float(lr_floor_pre_abs) if (lr_floor_pre_abs is not None) else (10.0 * abs_floor_train)

    base_lr = (base_lr_pre if pre_train else base_lr_train)
    abs_floor = (abs_floor_pre if pre_train else abs_floor_train)
    lr_mult = 1.0
    rel_floor = 1e-5
    lr_floor = max(rel_floor * base_lr, abs_floor)
    if abs_floor >= base_lr and dist.get_rank() == 0:
        dist.print0(f"[warn] lr_floor_abs ({abs_floor:.3e}) >= base lr ({base_lr:.3e}); decay will floor immediately.")

    max_reductions, reductions_done = 10, 0
    patience_ticks = 10
    cooldown_ticks = 5
    cooldown = 0
    ema_beta_loss = 0.95
    loss_ema = None
    best_ema = None
    no_improve_ticks = 0
    
    # Pre-train threshold & EMA
    recon_ema = None
    dip_pre_start_nimg = None
    dip_trigger_logged = False
    dip_end_logged  = False
    lie_pre_start_nimg = None
    lie_trigger_logged = False
    lie_end_logged  = False
    flow_pre_start_nimg = None
    flow_trigger_logged = False
    flow_end_logged = False
    force_pretrain_snapshot = False
    pre_end_nimg = None
    
    recon_good_ticks = 0
    min_pre_kimg = 0.5
    
    # Tick accumulators
    tick_total_sum = 0.0
    tick_recon_sum = 0.0
    tick_steps = 0

    #copy original model weights into EMA (if using it)
    if use_ema: 
        ema = copy.deepcopy(net).eval().requires_grad_(False)

    
    # (1f) Resume training from previous snapshot.
    # (1f-1) Load training state: net + optimizer (authoritative for resuming)
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        all_nn_classes = [obj for name, obj in inspect.getmembers(nn) if inspect.isclass(obj) and obj.__module__.startswith('torch.nn')]
        torch.serialization.add_safe_globals(all_nn_classes + [persistence._reconstruct_persistent_obj])
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))

        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])

        sched = data.get("sched_state", None)
        if sched is not None:
            lr_mult = float(sched.get("lr_mult", lr_mult))
            reductions_done  = int(sched.get("reductions_done", reductions_done))
            no_improve_ticks = int(sched.get("no_improve_ticks", no_improve_ticks))
            cooldown         = int(sched.get("cooldown", cooldown))
            best_ema         = sched.get("best_ema", best_ema)

        ph = data.get("phase_state", None)
        if ph is not None:
            pre_train              = bool(ph.get("pre_train", pre_train))
            pre_end_nimg           = ph.get("pre_end_nimg", pre_end_nimg)
            dip_pre_start_nimg     = ph.get("dip_pre_start_nimg", dip_pre_start_nimg)
            lie_pre_start_nimg     = ph.get("lie_pre_start_nimg", lie_pre_start_nimg)
            flow_pre_start_nimg    = ph.get("flow_pre_start_nimg", flow_pre_start_nimg)
            dip_trigger_logged     = bool(ph.get("dip_trigger_logged",  dip_trigger_logged))
            lie_trigger_logged     = bool(ph.get("lie_trigger_logged",  lie_trigger_logged))
            flow_trigger_logged    = bool(ph.get("flow_trigger_logged", flow_trigger_logged))
            dip_end_logged         = bool(ph.get("dip_end_logged",      dip_end_logged))
            lie_end_logged         = bool(ph.get("lie_end_logged",      lie_end_logged))
            flow_end_logged        = bool(ph.get("flow_end_logged",     flow_end_logged))
            recon_ema              = ph.get("recon_ema", recon_ema)
            recon_good_ticks       = int(ph.get("recon_good_ticks", recon_good_ticks))
            pretrain_early_on      = bool(ph.get("pretrain_early_on", pretrain_early_on))
            pre_lr_ramp_kimg       = ph.get("pre_lr_ramp_kimg", pre_lr_ramp_kimg)
            pre_param_kimg         = ph.get("pre_param_kimg", pre_param_kimg)
            train_param_ramp_kimg  = ph.get("train_param_ramp_kimg", train_param_ramp_kimg)
        
        # restore forward flags that are not in state_dict
        _src_net = data['net']
        for _name in ('use_t_dyn', 'include_x0_tau', 'dim_cov_dynamic', 'dim_cov_static'):
            if hasattr(_src_net, _name) and hasattr(net, _name):
                setattr(net, _name, getattr(_src_net, _name))
        
        del data  # conserve memory
        dist.print0('[resume] Restored net + optimizer + forward flags from training-state dump.')
        
    
    # (1f-2) Optional: hydrate EMA ONLY from .pkl (leave net untouched)
    if resume_pkl is not None and use_ema:
        dist.print0(f'Loading EMA weights from "{resume_pkl}" (net untouched)...')
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data
        dist.print0('[resume] EMA set from .pkl.')

    # -----------------------------------------------------------------------------------------------------
    # (2) Training
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = int(resume_kimg) * 1000
        
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    gs = 0 
    loss_scales = torch.Tensor([alpha, beta, eta, gamma]).type(torch.float32).to(device)

    # Initialize ramp-state flags from the *current* resume point to avoid fake transitions.
    kimg_after0      = max(cur_nimg - pre_train_kimgs * 1000, 0) / 1000.0
    in_train_ramp0   = ((not pre_train) and (kimg_after0 < float(train_param_ramp_kimg or 0)))
    
    # prev_in_pre_warmup = in_pre_warmup0
    prev_in_train_ramp = in_train_ramp0
    prev_pre_train     = pre_train
    prev_in_pre_warmup = None

    # enter training loop!
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        tot_scalar_loss = 0
        tot_separate_losses = torch.zeros(4).type(torch.float32).to(device)

        # (2a) Schedules & per-step constants
        # (2a-1) Per-tick constant
        active_flow_in_pre = (getattr(loss_fn, "pre_no_flow", False) == False)

        # (2a-2) Compute schedules once per tick
        t_train = 0.0 if pre_train else _ramp_t(pre_end_nimg, train_param_ramp_kimg, cur_nimg)
        pre_lr_done_nimg = int((pre_lr_ramp_kimg or 0) * 1000)
        t_pre_dip  = _t_pre_if_active(dip_pre_start_nimg, True, pre_lr_done_nimg)

        # Lie is active in pre-train only if pre_no_lie == False
        t_pre_lie  = _t_pre_if_active(lie_pre_start_nimg, not pre_no_lie, pre_lr_done_nimg)
        t_pre_flow = _t_pre_if_active(flow_pre_start_nimg, active_flow_in_pre, pre_lr_done_nimg)
        
        t_dip  = t_pre_dip  + (1.0 - t_pre_dip)  * t_train
        t_lie  = t_pre_lie  + (1.0 - t_pre_lie)  * t_train
        t_flow = t_pre_flow + (1.0 - t_pre_flow) * t_train
        
        # one-time “end of ramp” logs when each pre-train ramp hits 100%
        if (dip_pre_start_nimg  is not None) and (not dip_end_logged)  and (t_pre_dip  >= 1.0):
            dist.print0(f"[DIP] DIP ramp complete at {cur_nimg/1000:.3f} kimg");   dip_end_logged  = True
        if (lie_pre_start_nimg  is not None) and (not lie_end_logged)  and (t_pre_lie  >= 1.0):
            dist.print0(f"[Lie] Lie ramp complete at {cur_nimg/1000:.3f} kimg");   lie_end_logged  = True
        if (flow_pre_start_nimg is not None) and (not flow_end_logged) and (t_pre_flow >= 1.0):
            dist.print0(f"[Flow] Flow ramp complete at {cur_nimg/1000:.3f} kimg"); flow_end_logged = True
        
        alpha_curr = alpha * t_flow
        beta_curr  = beta  * t_flow
        eta_curr   = (1.0 - t_train) * eta + t_train * eta_post
        gamma_curr = gamma * t_lie
        dip_weight = alpha_mu * t_dip
    
        loss_scales[0] = float(alpha_curr)
        loss_scales[1] = float(beta_curr)
        loss_scales[2] = eta_curr
        loss_scales[3] = gamma_curr
        if dist.get_rank() == 0:
            writer.add_scalar('flow_train_loss/weight', float(alpha_curr), gs)
            writer.add_scalar('dyn_train_loss/weight',  float(beta_curr),  gs)
            writer.add_scalar('enc_recon_loss/weight', float(eta_curr), gs)
            writer.add_scalar('lie_derivative_loss/weight', float(gamma_curr), gs)

        # (2b) Now run the micro-batch accumulation loop
        dip_step_sum = 0.0
        for round_idx in range(num_accumulation_rounds):
            # (2b-1) Fetch batch & upack covariates
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)): 
                batch_data = next(dataset_iterator)

                x0_1 = batch_data[0].type(torch.float32).reshape(batch_data[0].shape[0], -1).to(device)
                xdt_1 = batch_data[1].type(torch.float32).reshape(batch_data[1].shape[0], -1).to(device)
                # dt is batch_data[2], already handled in loss function

                # (2b-2) Infer pre-train recon threshold pre_T_abs (if needed)
                if pre_train and (pre_T_abs is None):
                    x = x0_1
                    B = (x * x).mean()
                    A = torch.quantile(x.abs().view(-1), 0.99)
                    m_var  = 0.01 * B  # criteria 1: signal power
                    m_psnr = (A * A) * (10.0 ** (-30.0 / 10.0))  # criteria 2: PSNR, 30 dB default, don't be too bad, that's enough
                    m_star = torch.maximum(m_var, m_psnr)
                    N = x.numel()
                    eta_curr_tmp = float(eta)  # pre-train uses base eta
                    # pre_T_abs = float(eta_curr_tmp * m_star * N * (loss_scaling / batch_gpu_total))
                    pre_T_abs = float(eta_curr_tmp * m_star * N * (loss_scaling / batch_gpu_total) * num_accumulation_rounds)
                    # pre_T_abs = float(eta_curr_tmp * m_star * net.data_dim * loss_scaling)
                    
                    
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        val = torch.tensor([pre_T_abs if dist.get_rank()==0 else 0.0],
                                           device=device, dtype=torch.float32)
                        torch.distributed.broadcast(val, src=0)
                        pre_T_abs = float(val.item())
                    if dist.get_rank() == 0:
                        dist.print0(f"[pre-train] inferred reconstruction loss threshold={pre_T_abs:.3e}")
                

                idx = 3
                cov_dynamic_0 = None
                cov_dynamic_dt = None
                cov_static = None

                if hasattr(net, 'dim_cov_dynamic') and net.dim_cov_dynamic > 0:
                    cov_dynamic_0 = batch_data[idx].type(torch.float32).to(device)
                    cov_dynamic_dt = batch_data[idx+1].type(torch.float32).to(device)
                    idx += 2

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
                
                
                loss = loss * loss_scales[:, None, None] # Scale losses
                training_stats.report('Loss/loss', torch.sum(loss.detach(), dim=0)) # Log losses
                round_scalar_loss = loss.sum() * (loss_scaling / batch_gpu_total) # Compute scalar loss for backprop

                # (2b-4) DIP: empirical first 2 moment control of mu (mean = 0, Cov = I)
                dip = torch.zeros((), device=x0_1.device)
                if dip_weight > 0:
                
                    enc = getattr(ddp.module, 'encoder', ddp.module)
                    mu_pre, _, _ = enc.encode(x0_1)                       # [B,d]
                    B_local, d = mu_pre.shape
                    dev, dtype = mu_pre.device, mu_pre.dtype
                    
                    # choose K = dims_to_keep if provided, else all
                    K = d if not dip_use_dims_to_keep else int(network_kwargs.get('dims_to_keep', d))
                    K = max(0, min(K, d))
    
                    # local stats with grad (for the DIP gradient path)
                    sum1_loc_g = mu_pre.sum(dim=0)                                     # [d]
                    sum2_loc_g = (mu_pre * mu_pre).sum(dim=0)                          # [d]
                    Sxxk_loc_g = mu_pre[:, :K].T @ mu_pre[:, :K] if K > 0 else None    # [K,K]
                    B_local_t  = torch.tensor([B_local], device=dev, dtype=torch.float32)
                    
                    # detached COPIES for cross-GPU comms (to avoid autograd warning)
                    sum1_local = sum1_loc_g.detach().clone()
                    sum2_local = sum2_loc_g.detach().clone()
                    Sxxk_local = None if Sxxk_loc_g is None else Sxxk_loc_g.detach().clone()
                    
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        with torch.no_grad():
                            torch.distributed.all_reduce(B_local_t)
                            torch.distributed.all_reduce(sum1_local)
                            torch.distributed.all_reduce(sum2_local)
                            if Sxxk_local is not None: torch.distributed.all_reduce(Sxxk_local)
                    
                    # previous EMA states (constants)
                    prev_sum1 = None if dip_stats['sum1'] is None else dip_stats['sum1'].detach()
                    prev_sum2 = None if dip_stats['sum2'] is None else dip_stats['sum2'].detach()
                    prev_Sxxk = None if dip_stats['Sxxk'] is None else dip_stats['Sxxk'].detach()
                    prev_B    = None if dip_stats['B']    is None else dip_stats['B'].detach()
                    
                    # EMA as constant (forward values)
                    sum1_ema_c = sum1_local if prev_sum1 is None else (1.0 - beta_ema)*sum1_local + beta_ema*prev_sum1
                    sum2_ema_c = sum2_local if prev_sum2 is None else (1.0 - beta_ema)*sum2_local + beta_ema*prev_sum2
                    Sxxk_ema_c = None if Sxxk_local is None else (Sxxk_local if prev_Sxxk is None else (1.0 - beta_ema)*Sxxk_local + beta_ema*prev_Sxxk)
                    B_ema_c    = B_local_t if prev_B is None else (1.0 - beta_ema)*B_local_t + beta_ema*prev_B
                    
                    # stop-grad glue: forward equals EMA constants, gradient equals (1-β)*local stats
                    sum1_ema = g_only(sum1_ema_c, (1.0 - beta_ema)*sum1_loc_g)
                    sum2_ema = g_only(sum2_ema_c, (1.0 - beta_ema)*sum2_loc_g)
                    Sxxk_ema = None if Sxxk_ema_c is None else g_only(Sxxk_ema_c, (1.0 - beta_ema)*Sxxk_loc_g)
                    B_ema_t  = B_ema_c.detach()  # B has no grad path anyway
    
                    # write detached copies for next step's state
                    dip_stats['sum1'] = sum1_ema_c.detach()
                    dip_stats['sum2'] = sum2_ema_c.detach()
                    if Sxxk_ema_c is not None: dip_stats['Sxxk'] = Sxxk_ema_c.detach()
                    dip_stats['B'] = B_ema_c.detach()
    
                    Btot = max(1.0, float(B_ema_t.item()))
                    m    = (sum1_ema / Btot)
                    dip = mu_pre.new_tensor(0.0)
                    
                    # ----- DIP-VAE-I (EMA, head–tail) -----
                    # Centered covariance pieces from EMAs
                    Erx2 = (sum2_ema / Btot)                 # E[x_i^2]
                    v    = (Erx2 - m * m).clamp_min(1e-12)   # diag(C) for all dims [d]
                    
                    dip_var = 0.25*((v - 1.0) ** 2).sum()        # variance→1 (all dims)
                    dip_mean = 0.5*(m * m).sum()
                    
                    dip_off = mu_pre.new_tensor(0.0)
                    if K > 1:
                        Ekxx = (Sxxk_ema / Btot)             # E[x x^T] on head [K,K]
                        mk   = m[:K]
                        Ck   = Ekxx - mk[:, None] @ mk[None, :]
                        Ck   = 0.5 * (Ck + Ck.T)             # symmetrize
                        off  = Ck - torch.diag_embed(torch.diag(Ck))
                        dip_off = 0.25*(off * off).sum()
                        
                    dip = dip_off + dip_var + dip_mean
                    dip_step_sum += float(B_local) * float(dip.item())
                    round_scalar_loss = round_scalar_loss + (loss_scaling / batch_gpu_total) * dip_weight * (B_local * dip)

                # (2b-5) Aggregate TB losses / totals / backprop
                # Get separate losses for TB logging
                separate_losses = torch.sum(torch.sum(loss.detach(), dim=2), dim=1) * (loss_scaling / batch_gpu_total)
                tot_separate_losses += separate_losses
                
                # Accumulate loss
                tot_scalar_loss += round_scalar_loss.item()
                
                # Backpropagate
                round_scalar_loss.backward()


        # Update weights.
        eff_lr_ramp_kimg = float(pre_lr_ramp_kimg or 0.0)
        if pre_train:  warmup = min(cur_nimg / max(eff_lr_ramp_kimg * 1000, 1e-8), 1.0)
        else: warmup = 1.0
        for g in optimizer.param_groups:
            g['lr'] = base_lr * warmup * lr_mult
        
        #make sure all grads are numbers
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        
        #clip model gradient norms, if desired
        if grad_clip: 
            assert grad_clip_val != None, 'Need a value to clip grads to!'
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
        
        # (2c) optimizer/ EMA/ LR
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
            writer.add_scalar('flow_train_loss/value', tot_separate_losses[0].item(), gs)
            writer.add_scalar('dyn_train_loss/value', tot_separate_losses[1].item(), gs)
            writer.add_scalar('enc_recon_loss/value', tot_separate_losses[2].item(), gs)
            writer.add_scalar('lie_derivative_loss/value', tot_separate_losses[3].item(), gs)
            writer.add_scalar('Kimgs', cur_nimg/1000, gs)
            
            # 09/14:add
            writer.add_scalar('dip_mu/weight', dip_weight, gs)
            writer.add_scalar('dip_mu/value', (loss_scaling / batch_gpu_total) * dip_weight * dip_step_sum, gs)
            gs+=1

        # Perform maintenance tasks once per tick.
        tick_steps += 1
        tick_total_sum += float(tot_scalar_loss)
        tick_recon_sum += float(tot_separate_losses[2].item())
        cur_nimg += batch_size
        
        done = (cur_nimg >= total_kimg * 1000)
        
        want = 1 if ((not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick*1000)) else 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([want], device=device, dtype=torch.int32)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            want = int(t.item() > 0)  # if ANY rank wants to continue, ALL continue
        if want:
            continue

        # if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
        #    continue

        tick_total_mean = tick_total_sum / max(1, tick_steps)
        tick_recon_mean = tick_recon_sum / max(1, tick_steps)
        tick_total_sum = 0.0; tick_recon_sum = 0.0; tick_steps = 0
        
        # # local version
        # metric = tick_recon_mean if pre_train else tick_total_mean
        
        # DCC version
        metric_val = tick_recon_mean if pre_train else tick_total_mean
        metric_t = torch.tensor([metric_val], device=device, dtype=torch.float32)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(metric_t, op=torch.distributed.ReduceOp.AVG)
        metric = float(metric_t.item())
        
        if loss_ema is None:
            loss_ema = metric
            best_ema = metric
        else:
            loss_ema = ema_beta_loss * loss_ema + (1.0 - ema_beta_loss) * metric

        # (2d) Pre-train maintenance & transitions
        # pre-train maintenance: arm (by threshold or time), then end (no waiting; ramps finish in training)
        if pre_train:
            # (2d-1) Update recon EMA and arm by threshold (if spec'd and min kimg reached)
            if (pre_T_abs is not None) and (cur_nimg >= int(min_pre_kimg * 1000)):
                recon_ema = tick_recon_mean if recon_ema is None else (
                    ema_beta_loss * recon_ema + (1.0 - ema_beta_loss) * tick_recon_mean
                )
            
                recon_t = torch.tensor([float(recon_ema)], device=device, dtype=torch.float32)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(recon_t, op=torch.distributed.ReduceOp.AVG)
                recon_ema = float(recon_t.item())
            
                # separate thresholds
                dip_T_abs = float(dip_thresh_scale) * float(pre_T_abs)
                lie_T_abs = float(lie_thresh_scale) * float(pre_T_abs)
                flow_T_abs = float(flow_thresh_scale) * float(pre_T_abs)
            
                # Arm DIP first, Lie later (each triggers an immediate snapshot of the pre-change state)
                if (recon_ema <= dip_T_abs) and (dip_pre_start_nimg is None) and (not pretrain_early_on) and not (cur_nimg < int((pre_lr_ramp_kimg or 0) * 1000)):
                    dip_pre_start_nimg = cur_nimg
                    force_pretrain_snapshot = True
                    if dist.get_rank() == 0 and not dip_trigger_logged:
                        dist.print0(f"[DIP] DIP ramp armed (×{dip_thresh_scale:g}) at {cur_nimg/1000:.3f} kimg")
                        dip_trigger_logged = True
                        
                if (not pre_no_lie) and (t_pre_dip >= 1.0) and \
                   (recon_ema <= lie_T_abs) and (lie_pre_start_nimg is None) and (not pretrain_early_on):
                    lie_pre_start_nimg = cur_nimg
                    force_pretrain_snapshot = True
                    if dist.get_rank() == 0 and not lie_trigger_logged:
                        dist.print0(f"[Lie] Lie ramp armed (×{lie_thresh_scale:g}) at {cur_nimg/1000:.3f} kimg")
                        lie_trigger_logged = True
            
                # flow (may need change later)
                # if active_flow_in_pre and (recon_ema <= dip_T_abs) and (flow_pre_start_nimg is None) and (not pretrain_early_on):
                if (active_flow_in_pre) and (t_pre_dip >= 1.0) and \
                   (recon_ema <= flow_T_abs) and (flow_pre_start_nimg is None) and (not pretrain_early_on):
                    flow_pre_start_nimg = cur_nimg
                    force_pretrain_snapshot = True
                    if dist.get_rank() == 0 and not flow_trigger_logged:
                        dist.print0(f"[Flow] Flow ramp armed (×{flow_thresh_scale:g}) at {cur_nimg/1000:.3f} kimg")
                        flow_trigger_logged = True
            
                # consecutive-good ticks bookkeeping (unchanged)
                recon_good_ticks = (recon_good_ticks + 1) if (recon_ema <= float(pre_T_abs)) else 0

            # (2d-2) Arm at time-limit only if needed, so training can finish any remaining ramps
            hit_time_limit = (cur_nimg >= pre_train_kimgs * 1000)
            dip_start_eff = _eff_start(pretrain_early_on, dip_pre_start_nimg)
            lie_start_eff = _eff_start((not pre_no_lie) and pretrain_early_on, lie_pre_start_nimg)
            flow_start_eff = _eff_start(active_flow_in_pre and pretrain_early_on, flow_pre_start_nimg)
            
            if hit_time_limit and not (cur_nimg < int((pre_lr_ramp_kimg or 0) * 1000)):
                if dip_start_eff is None:
                    dip_pre_start_nimg = cur_nimg
                    if dist.get_rank() == 0: dist.print0(f"[DIP] armed at time-limit @ {cur_nimg/1000:.3f} kimg")

                # Lie only time-arms if DIP was already active before this tick
                if (not pre_no_lie) and (lie_start_eff is None) and (dip_pre_start_nimg is not None) and \
                   (cur_nimg >= dip_pre_start_nimg + int(kimg_per_tick * 1000)):
                    lie_pre_start_nimg = cur_nimg
                    if dist.get_rank() == 0: dist.print0(f"[Lie] armed at time-limit @ {cur_nimg/1000:.3f} kimg")
                if active_flow_in_pre and (flow_start_eff is None):
                    flow_pre_start_nimg = cur_nimg
                    if dist.get_rank() == 0: dist.print0(f"[Flow] armed at time-limit @ {cur_nimg/1000:.3f} kimg")

            # (2d-3) End rule: ("good enough" OR time-limit); do not wait for pre-ramps to finish
            good_enough = (pre_T_abs is not None) and (recon_good_ticks >= 5)
            if (good_enough or hit_time_limit):
                pre_train = False
                pre_end_nimg = cur_nimg
                force_pretrain_snapshot = True
                if dist.get_rank() == 0:
                    reason = "EMA recon ≤ threshold" if good_enough else "time-limit"
                    dist.print0(f"pre-train end @ {cur_nimg/1000:.3f} kimg ({reason})")

        pre_lr_warmup = pre_train and (cur_nimg < int((pre_lr_ramp_kimg or 0) * 1000))

        # "pre-param ramping" means any of DIP/Lie/Flow has t_pre in (0,1)
        pre_param_ramping = pre_train and (
            (0.0 < t_pre_dip  < 1.0) or
            (0.0 < t_pre_lie  < 1.0) or
            (0.0 < t_pre_flow < 1.0)
        )
        in_train_ramp = (not pre_train) and (t_train < 1.0)
        pause_adaptive_decay = pre_lr_warmup or pre_param_ramping or in_train_ramp


        if prev_in_pre_warmup is None and pre_lr_warmup and pre_train:
            dist.print0('[ramp] start of pre-train ramp-up')
        if prev_in_pre_warmup and not pre_lr_warmup:
            dist.print0('[ramp] end of pre-train ramp-up')
        if prev_pre_train and (not pre_train) and (not in_train_ramp):
            dist.print0('[ramp] end of pre-train (no training ramp)')
            if pre_end_nimg is None:
                pre_end_nimg = cur_nimg
            if resume_kimg <= 0:  # fresh run only — do not clobber LR on resume
                lr_mult = 1.0
                reductions_done = 0
                no_improve_ticks = 0
                cooldown = 0
                best_ema = None
                warmup = 1.0
                base_lr   = base_lr_train
                abs_floor = abs_floor_train
                lr_floor  = max(rel_floor * base_lr, abs_floor)
                for g in optimizer.param_groups:
                    g['lr'] = base_lr * warmup * lr_mult

                if dist.get_rank() == 0 and cur_tick == 0:
                    dist.print0(f"[sanity] lr={optimizer.param_groups[0]['lr']:.3e}")
                if dist.get_rank() == 0:
                    dist.print0(f"[lr-reset] lr={optimizer.param_groups[0]['lr']:.3e}, "
                                f"lr_mult={lr_mult:.2f}, budget={reductions_done}/{max_reductions} "
                                f"(left={max_reductions - reductions_done})")


        if (prev_in_train_ramp is None or not prev_in_train_ramp) and in_train_ramp:
            dist.print0('[ramp] end of pre-train / start of training ramp-up')
            lr_mult = 1.0
            reductions_done = 0
            no_improve_ticks = 0
            cooldown = 0
            best_ema = None          # set fresh baseline at ramp end
            base_lr   = base_lr_train
            abs_floor = abs_floor_train
            lr_floor  = max(rel_floor * base_lr, abs_floor)
            
            # reset all param group LRs to the max (base) right away
            for g in optimizer.param_groups:
                g['lr'] = base_lr

            if dist.get_rank() == 0:
                dist.print0(
                    f"[lr-reset] entering training ramp: "
                    f"set lr=base_lr={base_lr:.3e}, lr_mult={lr_mult:.2f}, "
                    f"budget={reductions_done}/{max_reductions} "
                    f"(left={max_reductions - reductions_done})"
                )

        if prev_in_train_ramp and not in_train_ramp:
            dist.print0('[ramp] end of training ramp-up')
            best_ema = loss_ema      # set post-ramp baseline
            no_improve_ticks = 0     # clear counters so decay can trigger
            cooldown = 0
        
        prev_in_pre_warmup = pre_lr_warmup
        prev_in_train_ramp = in_train_ramp
        prev_pre_train = pre_train
        
        
        # Global LR adaptation: simple plateau rule with floor & cooldown
        if not pause_adaptive_decay:
            # improved = (best_ema is None) or (loss_ema < best_ema * (1.0 - 0.015))
            improved = (best_ema is None) or (loss_ema < best_ema * (1.0 - 0.005))
            if improved:
                best_ema = loss_ema
                no_improve_ticks = 0
            else:
                no_improve_ticks += 1
        
            if cooldown > 0:
                cooldown -= 1
        
            # if no_improve_ticks >= 2 and cooldown == 0 and reductions_done < max_reductions:
            if no_improve_ticks >= patience_ticks and cooldown == 0 and reductions_done < max_reductions:
                new_lr_mult = max(lr_mult * 0.7, lr_floor / base_lr)
                if new_lr_mult < lr_mult - 1e-12:
                    lr_mult = new_lr_mult
                    reductions_done += 1
                    # cooldown = 1
                    cooldown = cooldown_ticks
                    eff_lr = base_lr * warmup * lr_mult
                    if dist.get_rank() == 0:
                        dist.print0(f"LR to {eff_lr:.3e}")

        
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
        # if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
        if (snapshot_ticks is not None) and (done or force_pretrain_snapshot or cur_tick % snapshot_ticks == 0):
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

                    # misc.check_ddp_consistency(m)
                    data[key] = m.cpu()
                    del m
                else:
                    data[key] = value
                    
                del value # conserve memory
            if dist.get_rank() == 0:
                if force_pretrain_snapshot: network_snap_name = f'network-snapshot-{cur_nimg//1000:06d}-pretrain.pkl'
                else: network_snap_name = f'network-snapshot-{cur_nimg//1000:06d}.pkl'
                with open(os.path.join(run_dir, network_snap_name), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory
            
        # Save full dump of the training state.
        if dist.get_rank() == 0 and (done or force_pretrain_snapshot or (state_dump_ticks is not None and cur_tick % state_dump_ticks == 0)):

            if force_pretrain_snapshot: train_state_name = f"training-state-{cur_nimg//1000:06d}-pretrain.pt"
            else: train_state_name = f"training-state-{cur_nimg//1000:06d}.pt"
            
            torch.save(
                {
                    "net": net,
                    "optimizer_state": optimizer.state_dict(),
                    "sched_state": {
                        "lr_mult": lr_mult,
                        "reductions_done": reductions_done,
                        "no_improve_ticks": no_improve_ticks,
                        "cooldown": cooldown,
                        "best_ema": best_ema,
                    },
                    "phase_state": {
                        "pre_train": pre_train,
                        "pre_end_nimg": pre_end_nimg,
                        "dip_pre_start_nimg": dip_pre_start_nimg,
                        "lie_pre_start_nimg": lie_pre_start_nimg,
                        "flow_pre_start_nimg": flow_pre_start_nimg,
                        "dip_trigger_logged":  dip_trigger_logged,
                        "lie_trigger_logged":  lie_trigger_logged,
                        "flow_trigger_logged": flow_trigger_logged,
                        "dip_end_logged":      dip_end_logged,
                        "lie_end_logged":      lie_end_logged,
                        "flow_end_logged":     flow_end_logged,
                        "recon_ema": recon_ema,
                        "recon_good_ticks": recon_good_ticks,
                        "pretrain_early_on": pretrain_early_on,
                        "pre_lr_ramp_kimg": pre_lr_ramp_kimg,
                        "pre_param_kimg": pre_param_kimg,
                        "train_param_ramp_kimg": train_param_ramp_kimg,
                    },
                },
                os.path.join(run_dir, train_state_name),
            )
            
        if force_pretrain_snapshot:
            force_pretrain_snapshot = False
            

        # log net param grads and ODE_sim/trajs to TB 
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
                

        # (2f) logging
        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(),
                                              timestamp=time.time())) + '\n')
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