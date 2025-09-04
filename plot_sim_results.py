import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import torch
import dnnlib.util_v4 as util
from tqdm import tqdm


def plotTraj(data, data_sim = None, title = None):
    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min
    offset = data_range * 1.1

    time = np.arange(data.shape[0])
    if data_sim is not None:
        time_sim = np.arange(data_sim.shape[0])
    
    # Plot the trajectories
    plt.figure(figsize=(6, 4))  # Adjust figure size as needed
    for i in range(data.shape[1]):
        adjusted_y = data[:, i] + i * offset
        plt.plot(time, adjusted_y, label=f'Trajectory {i+1}', linewidth=0.5, c='red') 
        if data_sim is not None:
            adjusted_y_sim = data_sim[:, i] + i * offset
            plt.plot(time_sim, adjusted_y_sim, label=f'Trajectory {i+1}', linewidth=0.5, c='black')
        
    # Customize the plot
    plt.xlabel('Time')
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    plt.show()


def generate_traj(dyn_net, dset_samples, n_step, tau, lag_samples, device, lag,
                  include_x0_tau = False,
                  oracle = False, lag_cov_list_pre = None, flow_net = None, clamp_range = None):

    if clamp_range is not None:
        if len(clamp_range.shape) == 2:
            clamp_range = torch.from_numpy(clamp_range).to(device).float()
            full_min = clamp_range[:, 0].tile(lag+1).unsqueeze(0)
            full_max = clamp_range[:, 1].tile(lag+1).unsqueeze(0)
        
    lag_cov = torch.stack([
        torch.tensor(lag_samples[j][0], dtype=torch.float32)
        for j in range(len(lag_samples))
    ], dim=0).to(device)
    
    starting_pts = np.zeros((len(dset_samples), dset_samples[0].shape[1]))
    for ii in range(len(dset_samples)): starting_pts[ii,:] = dset_samples[ii][0,:]
    starting_pts = torch.from_numpy(starting_pts).unsqueeze(1).type(torch.float32).to(device) #ntrajs,1,dim
    if tau < 1.0:
        print('project starting points to latent')
        curr_tau_starting_pts = util.calc_flow_trajectories(flow_net,
                                                            starting_pts.squeeze(1), 1.0, tau, nt=2)
        curr_tau_pts = curr_tau_starting_pts[-1].unsqueeze(1)
    else:
        curr_tau_pts = starting_pts

    B, _, D = curr_tau_pts.shape

    curr_tau_trajs = []
    curr_tau_trajs.append(curr_tau_pts.cpu().numpy())
    lag_cov_list = []
    for ii in tqdm(range(n_step-1)):

        if oracle:
            lag_cov = torch.stack([
                torch.as_tensor(lag_samples[j][ii], dtype=torch.float32, device=device)
                if ii < lag_samples[j].shape[0]
                else torch.as_tensor(lag_samples[j][-1], dtype=torch.float32, device=device)  # hold last
                for j in range(len(lag_samples))
            ], dim=0)
        
            x0_tmp = torch.stack([
                torch.as_tensor(dset_samples[j][ii], dtype=torch.float32, device=device)
                if ii < dset_samples[j].shape[0]
                else torch.as_tensor(dset_samples[j][-1], dtype=torch.float32, device=device)  # hold last
                for j in range(len(dset_samples))
            ], dim=0)

        else:
            x0_tmp = curr_tau_pts.squeeze(1)
            if lag_cov_list_pre is None:
                lag_cov_list.append(lag_cov)
            else:
                assert tau < 1.0, "lag_cov_list_pre is only expected for latent τ."
                lag_cov = lag_cov_list_pre[ii]
        if clamp_range is not None:
            if len(clamp_range.shape) == 1:
                lag_cov = lag_cov.clamp(min=clamp_range[0], max=clamp_range[1])
            else:
                lag_cov = torch.clamp(lag_cov, min=full_min, max=full_max)
        
        traj = util.calc_dyn_trajectories(dyn_net, x0_tmp,
                                          tau, x0_tau=None, 
                                          covariates=None,
                                          next_covariates=None,
                                          covariates_static=lag_cov,
                                          include_x0_tau=include_x0_tau, nt=2)

        x_next = traj[-1]                    # [B,D]
        if clamp_range is not None:
            if len(clamp_range.shape) == 1:
                x_next = x_next.clamp(min=clamp_range[0], max=clamp_range[1])
            else:
                x_next = torch.clamp(x_next, min=clamp_range[:, 0].unsqueeze(0), max=clamp_range[:,1].unsqueeze(0))
        curr_tau_pts = x_next.unsqueeze(1)   # [B,1,D]
        curr_tau_trajs.append(curr_tau_pts.cpu().numpy())
        if (not oracle) and (lag_cov_list_pre is None):
            lag_cov = torch.cat([x_next, lag_cov[:, :lag*D]], dim=-1)
       
    curr_tau_trajs = np.concatenate(curr_tau_trajs, axis=1)
    curr_tau_trajs_trunc = []
    for ii in range(curr_tau_trajs.shape[0]):
        T_tmp = dset_samples[ii].shape[0]
        curr_tau_trajs_trunc.append(curr_tau_trajs[ii,:T_tmp,:])

    return curr_tau_trajs_trunc, lag_cov_list

def plot_latent_time_series(
    mu_traj_np,             # list of [T_i, dim_preserve]
    dims=(0,),              # which latent dims to plot (0-based)
    trial_indices=None,
    labels=None,            # None -> black; else categorical (array-like or {idx: label})
    dt=None,
    linewidth=1.2,
    alpha=0.9,
    fig_w=8,
    fig_h_per=2,
):
    if trial_indices is None:
        trial_indices = list(range(len(mu_traj_np)))
    dims = (dims,) if isinstance(dims, (int, np.integer)) else list(dims)

    # collect non-empty trials
    data = {i: mu_traj_np[i] for i in trial_indices if len(mu_traj_np[i]) > 0}
    if not data:
        raise ValueError("No non-empty trials to plot.")

    # categorical colors
    use_cats = labels is not None
    if use_cats:
        if isinstance(labels, dict):
            lab_for = {i: labels.get(i) for i in trial_indices}
        else:
            arr = np.asarray(labels, dtype=object)
            lab_for = {i: arr[i] for i in trial_indices}
        cats = []
        for i in trial_indices:
            if i in data:
                v = lab_for[i]
                if v not in cats:
                    cats.append(v)
        K = max(1, len(cats))
        hsv = plt.get_cmap("hsv", K)
        cmap = ListedColormap([hsv(i) for i in range(K)])
        norm = BoundaryNorm(np.arange(K+1)-0.5, K)
        cat_to_i = {c: i for i, c in enumerate(cats)}

    # figure
    fig_h = fig_h_per * len(dims)
    fig, axes = plt.subplots(len(dims), 1, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.ravel()

    for ax, d in zip(axes, dims):
        for i in trial_indices:
            if i not in data:
                continue
            y = data[i][:, d]
            x = np.arange(len(y)) if dt is None else np.arange(len(y)) * dt
            color = "k" if not use_cats else cmap(norm(cat_to_i[lab_for[i]]))
            ax.plot(x, y, lw=linewidth, alpha=alpha, color=color)
        ax.set_ylabel(f"μ{d+1}")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (step)" if dt is None else "Time")

    if use_cats:
        fig.subplots_adjust(bottom=0.18)
        cax = fig.add_axes([0.12, 0.08, 0.76, 0.04])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks(range(len(cats)))
        cbar.set_ticklabels([str(c) for c in cats])
        cbar.set_label("Trial category")

    plt.tight_layout(rect=[0.02, 0.18 if use_cats else 0.02, 0.98, 0.98])
    plt.show()

def plot_mu_trajectories_gradient_steps_3d(
    mu_traj_np,                   # list of arrays [T_i, D]
    trial_indices=None,           # iterable of trial ids to plot; None => all
    dims=(0, 1, 2),               # which μ dims (0-based)
    start_color="#fdae61",
    end_color="#313695",
    linewidth=2.0,
    alpha=0.5,                    # line alpha
    start_marker_size=36,
    start_marker_alpha=0.95,
    end_arrow_alpha=0.95,
    arrow_head_frac=0.03,         # head length as fraction of per-trial diagonal
    arrow_length_ratio=0.6,
    title="Latent trajectories",
    fig_w=7, fig_h=6,
    decimate=1,                   # plot every k-th step along each trial (speed-up)
    z_scale='auto',               # 'auto', 'none', or a number to multiply z values by
    aspect_ratio=(1, 1, 1),       # visual aspect ratio for the box
    elev_azim = None,
):
   
    # ----- choose trials -----
    if trial_indices is None:
        trial_indices = range(len(mu_traj_np))

    # ----- gather trajectories -----
    trajs, max_len = [], 0
    for idx in trial_indices:
        S = mu_traj_np[idx]
        if S is None or S.size == 0:
            continue
        trajs.append((idx, S))
        max_len = max(max_len, S.shape[0])
    if not trajs:
        raise ValueError("No non-empty trials to plot.")
        
    # ----- compute data ranges for auto-scaling -----
    all_x = []
    all_y = []
    all_z = []
    for _, S in trajs:
        all_x.extend(S[:, dims[0]].tolist())
        all_y.extend(S[:, dims[1]].tolist())
        all_z.extend(S[:, dims[2]].tolist())
    
    x_range = max(all_x) - min(all_x) if all_x else 1
    y_range = max(all_y) - min(all_y) if all_y else 1
    z_range = max(all_z) - min(all_z) if all_z else 1
    
    # ----- determine z scaling factor -----
    if z_scale == 'auto':
        # Scale z to have visual range comparable to x and y
        xy_avg_range = (x_range + y_range) / 2
        if z_range > 0:
            scale_factor = xy_avg_range / z_range
        else:
            scale_factor = 1.0
    elif z_scale == 'none':
        scale_factor = 1.0
    else:
        scale_factor = float(z_scale)
    
    # ----- setup color map -----
    cmap = LinearSegmentedColormap.from_list("o2b", [start_color, end_color])
    norm = Normalize(vmin=0, vmax=(1 if max_len <= 1 else max_len - 1))

    # ----- figure & axes -----
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111, projection="3d")

    # ----- track global data limits (with scaled z) -----
    xmin = ymin = zmin_scaled = np.inf
    xmax = ymax = zmax_scaled = -np.inf
    
    for _, S in trajs:
        # pick dims and cast to float
        x = np.asarray(S[:, dims[0]], dtype=float)
        y = np.asarray(S[:, dims[1]], dtype=float)
        z_raw = np.asarray(S[:, dims[2]], dtype=float)
        z = z_raw * scale_factor  # Apply scaling to z
        
        # optional decimation for speed
        if decimate > 1:
            x = x[::decimate]; y = y[::decimate]; z = z[::decimate]; z_raw = z_raw[::decimate]
        T = x.shape[0]

        # update bounds on the scaled data
        if T > 0:
            xmin = min(xmin, x.min()); xmax = max(xmax, x.max())
            ymin = min(ymin, y.min()); ymax = max(ymax, y.max())
            zmin_scaled = min(zmin_scaled, z.min()); zmax_scaled = max(zmax_scaled, z.max())

        if T == 0:
            continue

        # start marker (using scaled z)
        ax.scatter(x[0], y[0], z[0], s=start_marker_size, marker="o",
                   facecolor=cmap(norm(0)), edgecolor="black",
                   linewidths=0.5, alpha=start_marker_alpha, zorder=3)

        if T == 1:
            continue

        # segments colored by step index (using scaled z)
        pts = np.column_stack([x, y, z])                   # (T, 3)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)       # (T-1, 2, 3)
        seg_steps = np.arange(segs.shape[0])               # 0..T-2
        colors = cmap(norm(seg_steps))

        lc = Line3DCollection(segs, linewidths=linewidth, alpha=alpha)
        lc.set_colors(colors)
        ax.add_collection3d(lc)

        # end arrow (using scaled z)
        dx, dy, dz = x[-1]-x[-2], y[-1]-y[-2], z[-1]-z[-2]
        end_col = cmap(norm(T-1))
        rng = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()])
        diag = float(np.linalg.norm(rng))
        head_len = max(1e-12, arrow_head_frac * (diag if diag > 0 else 1.0))

        uv = np.array([dx, dy, dz], float)
        nrm = float(np.linalg.norm(uv))
        if nrm == 0:
            uv = np.array([1.0, 0.0, 0.0]); nrm = 1.0
        u_hat = uv / nrm
        tail = np.array([x[-1], y[-1], z[-1]]) - u_hat * head_len

        ax.quiver(
            tail[0], tail[1], tail[2],
            u_hat[0], u_hat[1], u_hat[2],
            length=head_len, normalize=False,
            color=end_col, linewidth=linewidth*1.2,
            arrow_length_ratio=arrow_length_ratio,
            alpha=end_arrow_alpha
        )

    # ----- labels (show scaling info if applied) -----
    ax.set_xlabel(f"μ{dims[0]+1}")
    ax.set_ylabel(f"μ{dims[1]+1}")
    # if scale_factor != 1.0:
    #     ax.set_zlabel(f"μ{dims[2]+1} (×{scale_factor:.1f})")
    # else:
    #     ax.set_zlabel(f"μ{dims[2]+1}")
    ax.set_zlabel(f"μ{dims[2]+1}")
    ax.set_title(title)

    # ----- set limits with padding -----
    rx = max(1e-12, xmax - xmin)
    ry = max(1e-12, ymax - ymin)
    rz_scaled = max(1e-12, zmax_scaled - zmin_scaled)
    
    ax.set_xlim(xmin - 0.02*rx, xmax + 0.02*rx)
    ax.set_ylim(ymin - 0.02*ry, ymax + 0.02*ry)
    ax.set_zlim(zmin_scaled - 0.02*rz_scaled, zmax_scaled + 0.02*rz_scaled)
    
    # ----- set aspect ratio -----
    ax.set_box_aspect(aspect_ratio)
    if elev_azim is not None:
        ax.view_init(elev=elev_azim[0], azim=elev_azim[1]) 
    
    if scale_factor != 1.0:
        zticks = ax.get_zticks()
        ax.set_zticks(zticks)  # Explicitly set the positions first
        ax.set_zticklabels([f"{z/scale_factor:.1f}" for z in zticks])

    # ----- colorbar on the right (shared step index) -----
    cax = fig.add_axes([0.88, 0.18, 0.03, 0.64])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Step index")
    ticks = [0, 1] if max_len <= 2 else [0, (max_len - 1)//2, max_len - 1]
    cbar.set_ticks(ticks); cbar.set_ticklabels([str(t) for t in ticks])

    plt.show()

def generate_traj_cov(dyn_net, dset_samples, n_step, tau, lag_samples, device, lag,
                      include_x0_tau = False, covStat_samples = None,
                      oracle = False, lag_cov_list_pre = None, flow_net = None, clamp_range = None):

    if clamp_range is not None:
        if len(clamp_range.shape) == 2:
            clamp_range = torch.from_numpy(clamp_range).to(device).float()
            full_min = clamp_range[:, 0].tile(lag+1).unsqueeze(0)
            full_max = clamp_range[:, 1].tile(lag+1).unsqueeze(0)
        
    lag_cov = torch.stack([
        torch.tensor(lag_samples[j][0], dtype=torch.float32)
        for j in range(len(lag_samples))
    ], dim=0).to(device)
    
    starting_pts = np.zeros((len(dset_samples), dset_samples[0].shape[1]))
    for ii in range(len(dset_samples)): starting_pts[ii,:] = dset_samples[ii][0,:]
    starting_pts = torch.from_numpy(starting_pts).unsqueeze(1).type(torch.float32).to(device) #ntrajs,1,dim
    if tau < 1.0:
        print('project starting points to latent')
        curr_tau_starting_pts = util.calc_flow_trajectories(flow_net,
                                                            starting_pts.squeeze(1), 1.0, tau, nt=2)
        curr_tau_pts = curr_tau_starting_pts[-1].unsqueeze(1)
    else:
        curr_tau_pts = starting_pts

    B, _, D = curr_tau_pts.shape

    curr_tau_trajs = []
    curr_tau_trajs.append(curr_tau_pts.cpu().numpy())
    lag_cov_list = []
    for ii in tqdm(range(n_step-1)):

        if oracle:
            lag_cov = torch.stack([
                torch.as_tensor(lag_samples[j][ii], dtype=torch.float32, device=device)
                if ii < lag_samples[j].shape[0]
                else torch.as_tensor(lag_samples[j][-1], dtype=torch.float32, device=device)  # hold last
                for j in range(len(lag_samples))
            ], dim=0)
        
            x0_tmp = torch.stack([
                torch.as_tensor(dset_samples[j][ii], dtype=torch.float32, device=device)
                if ii < dset_samples[j].shape[0]
                else torch.as_tensor(dset_samples[j][-1], dtype=torch.float32, device=device)  # hold last
                for j in range(len(dset_samples))
            ], dim=0)

        else:
            x0_tmp = curr_tau_pts.squeeze(1)
            if lag_cov_list_pre is None:
                lag_cov_list.append(lag_cov)
            else:
                assert tau < 1.0, "lag_cov_list_pre is only expected for latent τ."
                lag_cov = lag_cov_list_pre[ii]
        if clamp_range is not None:
            if len(clamp_range.shape) == 1:
                lag_cov = lag_cov.clamp(min=clamp_range[0], max=clamp_range[1])
            else:
                lag_cov = torch.clamp(lag_cov, min=full_min, max=full_max)

        if covStat_samples is not None:
            covStat_tmp = torch.stack([
                torch.as_tensor(covStat_samples[j][ii], dtype=torch.float32, device=device)
                if ii < covStat_samples[j].shape[0]
                else torch.as_tensor(covStat_samples[j][-1], dtype=torch.float32, device=device)  # hold last
                for j in range(len(covStat_samples))
                ], dim=0)
            covStat_all = torch.cat((covStat_tmp, lag_cov), dim = 1)
        else:
            covStat_all = lag_cov
        
        traj = util.calc_dyn_trajectories(dyn_net, x0_tmp,
                                          tau, x0_tau=None, 
                                          covariates=None,
                                          next_covariates=None,
                                          covariates_static=covStat_all,
                                          include_x0_tau=include_x0_tau, nt=2)

        x_next = traj[-1]                    # [B,D]
        if clamp_range is not None:
            if len(clamp_range.shape) == 1:
                x_next = x_next.clamp(min=clamp_range[0], max=clamp_range[1])
            else:
                x_next = torch.clamp(x_next, min=clamp_range[:, 0].unsqueeze(0), max=clamp_range[:,1].unsqueeze(0))
        curr_tau_pts = x_next.unsqueeze(1)   # [B,1,D]
        curr_tau_trajs.append(curr_tau_pts.cpu().numpy())
        if (not oracle) and (lag_cov_list_pre is None):
            lag_cov = torch.cat([x_next, lag_cov[:, :lag*D]], dim=-1)
       
    curr_tau_trajs = np.concatenate(curr_tau_trajs, axis=1)
    curr_tau_trajs_trunc = []
    for ii in range(curr_tau_trajs.shape[0]):
        T_tmp = dset_samples[ii].shape[0]
        curr_tau_trajs_trunc.append(curr_tau_trajs[ii,:T_tmp,:])

    return curr_tau_trajs_trunc, lag_cov_list


def proj_to_latent(dset_samples, flow_net, tau_latent, device):
    latent_traj_list = []
    for ii in tqdm(range(len(dset_samples))):
        latent_traj_tmp = util.calc_flow_trajectories(flow_net,
                                                      torch.from_numpy(dset_samples[ii]).type(torch.float32).to(device),
                                                      1.0, tau_latent, nt=2)
        latent_traj_list.append(latent_traj_tmp[-1,:,:].cpu().numpy())
    return latent_traj_list

def latent_mu(latent_traj_list, encoder, dim_preserve, device):
    mu_traj = []
    encoder.eval()
    with torch.inference_mode():                      
        for ii in tqdm(range(len(latent_traj_list))):
            X = torch.from_numpy(latent_traj_list[ii]).to(device=device, dtype=torch.float32)  # [T, in_dim]
            mu = encoder.to(device).mu(X)                         # [T, out_dim]
            mu_traj_tmp = mu[:, :dim_preserve].detach().cpu()   # [T, dim_preserve]
            mu_traj.append(mu_traj_tmp)
    return mu_traj



