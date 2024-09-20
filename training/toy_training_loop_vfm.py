#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_loop for toy dsets using CFM 

Essentially same as toy_training_loop
but using CFM loss, net, and etc.

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
from torch_cfm.utils import calc_trajectories, plot_trajectories 

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
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
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    
    # Set up our TB writer
    dist.print0('Creating TB dir and writter...')
    if dist.get_rank() == 0:
        writer_dir = os.path.join(run_dir, 'TB_logs')
        os.makedirs(writer_dir, exist_ok=True) 
        writer = SummaryWriter(log_dir = writer_dir)

    # Compute eigen-decomp. for dataset if using IFs flow_matcher
    if loss_kwargs.flow_matcher_type == 'ifs': 
        dist.print0('Computing eigen-decomposition for ifs FM...')
        large_sample = dnnlib.util.get_toy_dset(dataset_kwargs.dset_name, 70000, augment_to=dataset_kwargs.augment_to)[0] 
        data_eigs, _, W = dnnlib.util.get_eigenvals_basis(large_sample, n_comp=dataset_kwargs.working_data_dim) 
        data_eigs = torch.from_numpy(data_eigs).type(torch.float32).to(device) 
        W = torch.from_numpy(W).type(torch.float32).to(device)
        loss_kwargs.update(data_eigs=data_eigs, W=W)
        del large_sample #conserve mem 

    # Construct network.
    dist.print0('Constructing network...')
    net = dnnlib.util.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, dataset_kwargs.working_data_dim], device=device)
            ts = torch.ones([batch_gpu], device=device)
            misc.print_module_summary(net, [images, ts], max_nesting=2)

    # Setup optimizer and lossfn
    dist.print0('Setting up optimizer and loss fn...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) 
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    
    #init Dist mode for net 
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    
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
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    gs = 0 
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        tot_scalar_loss = 0
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):  
                W = loss_fn.flowmatcher.W if loss_kwargs.flow_matcher_type=='ifs' else None 
                x1, x0 = dnnlib.util.get_cfm_samples(dataset_kwargs, batch_gpu, device, \
                                                     W, loss_fn.flow_matcher_type)
                loss = loss_fn(net=ddp, x1=x1, x0=x0) #bs, dim 
                #log in using original training stats
                training_stats.report('Loss/loss', loss)
                #compute scalar (final) loss
                round_scalar_loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                #accumulate loss for rounds 
                tot_scalar_loss += round_scalar_loss.item()
                #accumulate grads 
                round_scalar_loss.backward()


        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        
        #make sure all grads are numbers
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                
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
        writer.add_scalar('train_loss', tot_scalar_loss/num_accumulation_rounds, gs)
        gs+=1

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
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
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
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
            #calc and log trajs 
            traj = calc_trajectories(net, dataset_kwargs, loss_fn, device)
            traj_fig = plot_trajectories(traj.cpu().numpy())
            writer.add_figure('ODE_trajs', traj_fig, gs, close=True)
            
            
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



