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
    mu_traj_np,                 # list of [T_i, dim_preserve]
    trial_indices,              # iterable of trial ids to plot
    dims=(0, 1, 2),             # which μ dims (0-based)
    start_color="#fdae61",
    end_color="#313695",
    linewidth=2.0,
    alpha=0.5,                  # line alpha
    start_marker_size=36,
    start_marker_alpha=0.95,
    end_arrow_alpha=0.95,
    arrow_head_frac=0.03,       # head length as fraction of per-trial diagonal
    arrow_length_ratio=0.6,
    title=None,
    zlabel_on_left=True,
    fig_w=6,
    fig_h=6,
):
    # collect selected trials; find max length for shared color scale
    trajs, max_len = [], 0
    for idx in trial_indices:
        S = mu_traj_np[idx]
        if S.size == 0:
            continue
        trajs.append((idx, S))
        max_len = max(max_len, S.shape[0])
    if not trajs:
        raise ValueError("No non-empty trials to plot.")

    cmap = LinearSegmentedColormap.from_list("o2b", [start_color, end_color])
    norm = Normalize(vmin=0, vmax=(1 if max_len <= 1 else max_len - 1))

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.08, top=0.92)
    ax = fig.add_subplot(111, projection="3d")

    for idx, S in trajs:
        if S.shape[0] == 1:
            x0, y0, z0 = S[0, dims[0]], S[0, dims[1]], S[0, dims[2]]
            ax.scatter(x0, y0, z0, s=start_marker_size, marker="o",
                       facecolor=cmap(norm(0)), edgecolor="black",
                       linewidths=0.5, alpha=start_marker_alpha, zorder=3)
            continue

        x, y, z = S[:, dims[0]], S[:, dims[1]], S[:, dims[2]]
        pts = np.column_stack([x, y, z])               # (T, 3)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)   # (T-1, 2, 3)
        seg_steps = np.arange(segs.shape[0])           # 0..T-2
        colors = cmap(norm(seg_steps))

        lc = Line3DCollection(segs, linewidths=linewidth, alpha=alpha)
        lc.set_colors(colors)
        ax.add_collection3d(lc)

        # start marker
        ax.scatter(x[0], y[0], z[0], s=start_marker_size, marker="o",
                   facecolor=cmap(norm(0)), edgecolor="black",
                   linewidths=0.5, alpha=start_marker_alpha, zorder=3)

        # end arrow (scale head by overall data extent)
        dx, dy, dz = x[-1]-x[-2], y[-1]-y[-2], z[-1]-z[-2]
        end_col = cmap(norm(S.shape[0]-1))
        rng = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()])
        diag = float(np.linalg.norm(rng))
        head_len = max(1e-12, arrow_head_frac * (diag if diag > 0 else 1.0))

        uv = np.array([dx, dy, dz], dtype=float)
        nrm = float(np.linalg.norm(uv))
        if nrm == 0:
            uv = np.array([1.0, 0.0, 0.0]); nrm = 1.0
        u_hat = uv / nrm
        tail = np.array([x[-1], y[-1], z[-1]]) - u_hat * head_len

        ax.quiver(
            tail[0], tail[1], tail[2],
            u_hat[0], u_hat[1], u_hat[2],
            length=head_len, normalize=True,
            color=end_col, linewidth=linewidth*1.2,
            arrow_length_ratio=arrow_length_ratio,
            alpha=end_arrow_alpha
        )

    # axis labels for μ dims
    def lab(i): return f"μ{dims[i]+1}"
    ax.set_xlabel(lab(0))
    ax.set_ylabel(lab(1))
    z_text = lab(2)
    if zlabel_on_left:
        ax.set_zlabel("")
        fig.text(0.035, 0.52, z_text, rotation=90, va="center", ha="center")
    else:
        ax.set_zlabel(z_text, labelpad=12)

    ax.set_title(title or "3D latent (μ) trajectories — color = step index")
    ax.set_box_aspect((1, 1, 1))
    # ax.view_init(elev=20, azim=35)  # optional fixed view

    # colorbar on right
    cax = fig.add_axes([0.88, 0.18, 0.03, 0.64])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Step index")
    if max_len <= 1:
        ticks = [0, 1]
    elif max_len == 2:
        ticks = [0, 1]
    else:
        ticks = [0, (max_len - 1)//2, max_len - 1]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])

    ax.autoscale_view()
    plt.show()


