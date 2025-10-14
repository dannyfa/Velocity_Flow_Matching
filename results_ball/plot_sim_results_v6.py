import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import torch
import dnnlib.util_v6 as util
from tqdm import tqdm
import pickle
from matplotlib.ticker import FuncFormatter


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
    # plt.show()


def generate_traj(dyn_net, dset_samples, n_step, tau, lag_samples, device, lag,
                  include_x0_tau=False,
                  oracle=False, lag_cov_list_pre=None, flow_net=None, clamp_range=None,
                  reuse = False):
    """
    Vector rollout with lag history.
    reuse=False: roll at τ, back-project each step to data (τ→1) to update lag.
    reuse=True: use lag_cov_list_pre each step (fast but conceptually wrong).
    """
    # clamp prep
    full_min = full_max = None
    if clamp_range is not None and len(clamp_range.shape) == 2:
        clamp_range = torch.from_numpy(clamp_range).to(device).float()
        full_min = clamp_range[:, 0].tile(lag + 1).unsqueeze(0)
        full_max = clamp_range[:, 1].tile(lag + 1).unsqueeze(0)

    # init lag (first row from provided lag_samples)
    lag_cov = torch.stack([
        torch.tensor(lag_samples[j][0], dtype=torch.float32)
        for j in range(len(lag_samples))
    ], dim=0).to(device)

    # x(τ) at first point
    starting_pts = np.stack([dset_samples[ii][0, :] for ii in range(len(dset_samples))], 0)
    starting_pts = torch.from_numpy(starting_pts).unsqueeze(1).float().to(device)  # (B,1,D)
    if tau < 1.0:
        assert flow_net is not None, "flow_net required when tau<1.0."
        curr_tau_pts = util.calc_flow_trajectories(flow_net, starting_pts.squeeze(1), 1.0, tau, nt=2)[-1].unsqueeze(1)
    else:
        curr_tau_pts = starting_pts

    B, _, D = curr_tau_pts.shape
    curr_tau_trajs, lag_cov_list = [curr_tau_pts.cpu().numpy()], []

    for ii in tqdm(range(n_step - 1)):
        # source state and lag
        if oracle:
            lag_cov = torch.stack([
                torch.as_tensor(lag_samples[j][ii], dtype=torch.float32, device=device)
                if ii < lag_samples[j].shape[0] else
                torch.as_tensor(lag_samples[j][-1], dtype=torch.float32, device=device)
                for j in range(len(lag_samples))
            ], dim=0)
            x0_tmp = torch.stack([
                torch.as_tensor(dset_samples[j][ii], dtype=torch.float32, device=device)
                if ii < dset_samples[j].shape[0] else
                torch.as_tensor(dset_samples[j][-1], dtype=torch.float32, device=device)
                for j in range(len(dset_samples))
            ], dim=0)
        else:
            x0_tmp = curr_tau_pts.squeeze(1)
            if reuse and (lag_cov_list_pre is not None):
                assert tau < 1.0, "lag_cov_list_pre is only expected for latent τ."
                lag_cov = lag_cov_list_pre[ii]  # precomputed τ=1 lag for this step

        # clamp lag
        if clamp_range is not None:
            if len(clamp_range.shape) == 1:
                lag_cov = lag_cov.clamp(min=clamp_range[0], max=clamp_range[1])
            else:
                lag_cov = torch.clamp(lag_cov, min=full_min, max=full_max)

        # one step in τ
        traj = util.calc_dyn_trajectories(
            dyn_net, x0_tmp, tau, x0_tau=None,
            covariates=None, next_covariates=None,
            covariates_static=lag_cov,
            include_x0_tau=include_x0_tau, nt=2
        )
        x_next = traj[-1]  # (B,D)

        # clamp state
        if clamp_range is not None:
            if len(clamp_range.shape) == 1:
                x_next = x_next.clamp(min=clamp_range[0], max=clamp_range[1])
            else:
                x_next = torch.clamp(x_next, min=clamp_range[:, 0].unsqueeze(0), max=clamp_range[:, 1].unsqueeze(0))

        # push time
        curr_tau_pts = x_next.unsqueeze(1)
        curr_tau_trajs.append(curr_tau_pts.cpu().numpy())

        # roll lag when NOT reusing and not oracle
        if (not oracle) and (not reuse):
            if tau < 1.0:
                assert flow_net is not None, "flow_net required when tau<1.0 and reuse=False."
                x_next_for_lag = util.calc_flow_trajectories(flow_net, x_next, tau, 1.0, nt=2)[-1]
            else:
                x_next_for_lag = x_next
            lag_cov = torch.cat([lag_cov[:, D:], x_next_for_lag], dim=-1)

        # record lag used this step only when building here
        if not reuse:
            lag_cov_list.append(lag_cov)

    curr_tau_trajs = np.concatenate(curr_tau_trajs, axis=1)
    curr_tau_trajs_trunc = [curr_tau_trajs[i, :dset_samples[i].shape[0], :] for i in range(curr_tau_trajs.shape[0])]
    return curr_tau_trajs_trunc, lag_cov_list


def plot_latent_time_series(
    mu_traj_np,             # list of [T_i, dim_preserve]
    dims=(0,),              # which latent dims to plot (0-based)
    trial_indices=None,
    labels=None,            # None -> black; numeric -> depends on label_mode
    label_mode='auto',      # 'auto' | 'categorical' | 'continuous'
    dt=None,
    linewidth=1.2,
    alpha=0.9,
    fig_w=8,
    fig_h_per=2,
    cont_cmap="twilight", # "twilight_shifted"
    cont_vrange=None        # (vmin, vmax)
):
    if trial_indices is None:
        trial_indices = list(range(len(mu_traj_np)))
    dims = (dims,) if isinstance(dims, (int, np.integer)) else list(dims)

    # collect non-empty trials
    data = {i: mu_traj_np[i] for i in trial_indices if len(mu_traj_np[i]) > 0}
    if not data:
        raise ValueError("No non-empty trials to plot.")

    # build label lookup
    lab_for = None
    use_labels = labels is not None
    if use_labels:
        if isinstance(labels, dict):
            lab_for = {i: labels.get(i) for i in trial_indices if i in data}
        else:
            arr = np.asarray(labels)
            lab_for = {i: (arr[i] if i < arr.shape[0] else None) for i in trial_indices if i in data}

        # determine mode
        if label_mode == 'auto':
            vals = [lab_for[i] for i in lab_for if lab_for[i] is not None]
            # if all ints -> categorical; else continuous if all numeric
            all_int = all(isinstance(v, (int, np.integer)) for v in vals)
            all_num = all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals)
            use_cont = (not all_int) and all_num
        elif label_mode == 'continuous':
            use_cont = True
        else:
            use_cont = False
    else:
        use_cont = False

    # set up color mapping
    cmap = None
    norm = None
    cat_to_i = None
    cats = None

    if not use_labels:
        pass
    elif use_cont:
        vals = np.array([float(lab_for[i]) for i in lab_for], dtype=float)
        if cont_vrange is None:
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        else:
            vmin, vmax = cont_vrange

        default_name = plt.rcParams.get('image.cmap', 'viridis')
        cmap = plt.get_cmap(default_name if cont_cmap is None else cont_cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        # categorical palette (keep your original hsv style)
        cats = []
        for i in trial_indices:
            if i in data:
                v = lab_for.get(i, None)
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

            if not use_labels:
                color = "k"
            elif use_cont:
                v = lab_for.get(i, np.nan)
                color = cmap(norm(v)) if np.isfinite(v) else (0.5, 0.5, 0.5, 1.0)  # gray if missing
            else:
                color = cmap(norm(cat_to_i[lab_for[i]]))

            ax.plot(x, y, lw=linewidth, alpha=alpha, color=color)

        ax.set_ylabel(f"μ{d+1}")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (step)" if dt is None else "Time")

    # colorbar
    if use_labels:
        fig.subplots_adjust(bottom=0.18)
        cax = fig.add_axes([0.12, 0.08, 0.76, 0.04])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        if use_cont:
            cbar.set_label("Angle (deg)")
            if np.isclose(norm.vmin, -180) and np.isclose(norm.vmax, 180):
                cbar.set_ticks([-180, -90, 0, 90, 180])
        else:
            cbar.set_ticks(range(len(cats)))
            cbar.set_ticklabels([str(c) for c in cats])
            cbar.set_label("Trial category")

    plt.tight_layout(rect=[0.02, 0.18 if use_labels else 0.02, 0.98, 0.98])
    # plt.show()


# def plot_latent_time_series(
#     mu_traj_np,             # list of [T_i, dim_preserve]
#     dims=(0,),              # which latent dims to plot (0-based)
#     trial_indices=None,
#     labels=None,            # None -> black; else categorical (array-like or {idx: label})
#     dt=None,
#     linewidth=1.2,
#     alpha=0.9,
#     fig_w=8,
#     fig_h_per=2,
# ):
#     if trial_indices is None:
#         trial_indices = list(range(len(mu_traj_np)))
#     dims = (dims,) if isinstance(dims, (int, np.integer)) else list(dims)

#     # collect non-empty trials
#     data = {i: mu_traj_np[i] for i in trial_indices if len(mu_traj_np[i]) > 0}
#     if not data:
#         raise ValueError("No non-empty trials to plot.")

#     # categorical colors
#     use_cats = labels is not None
#     if use_cats:
#         if isinstance(labels, dict):
#             lab_for = {i: labels.get(i) for i in trial_indices}
#         else:
#             arr = np.asarray(labels, dtype=object)
#             lab_for = {i: arr[i] for i in trial_indices}
#         cats = []
#         for i in trial_indices:
#             if i in data:
#                 v = lab_for[i]
#                 if v not in cats:
#                     cats.append(v)
#         K = max(1, len(cats))
#         hsv = plt.get_cmap("hsv", K)
#         cmap = ListedColormap([hsv(i) for i in range(K)])
#         norm = BoundaryNorm(np.arange(K+1)-0.5, K)
#         cat_to_i = {c: i for i, c in enumerate(cats)}

#     # figure
#     fig_h = fig_h_per * len(dims)
#     fig, axes = plt.subplots(len(dims), 1, figsize=(fig_w, fig_h), squeeze=False)
#     axes = axes.ravel()

#     for ax, d in zip(axes, dims):
#         for i in trial_indices:
#             if i not in data:
#                 continue
#             y = data[i][:, d]
#             x = np.arange(len(y)) if dt is None else np.arange(len(y)) * dt
#             color = "k" if not use_cats else cmap(norm(cat_to_i[lab_for[i]]))
#             ax.plot(x, y, lw=linewidth, alpha=alpha, color=color)
#         ax.set_ylabel(f"μ{d+1}")
#         ax.grid(True, alpha=0.25)

#     axes[-1].set_xlabel("Time (step)" if dt is None else "Time")

#     if use_cats:
#         fig.subplots_adjust(bottom=0.18)
#         cax = fig.add_axes([0.12, 0.08, 0.76, 0.04])
#         sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
#         sm.set_array([])
#         cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
#         cbar.set_ticks(range(len(cats)))
#         cbar.set_ticklabels([str(c) for c in cats])
#         cbar.set_label("Trial category")

#     plt.tight_layout(rect=[0.02, 0.18 if use_cats else 0.02, 0.98, 0.98])
#     plt.show()

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
    elev_azim=None,               # tuple (elev, azim)
    # NEW:
    labels=None,                  # per-trial values -> color each whole trajectory
    label_mode='auto',            # 'auto' | 'continuous' | 'categorical'
    cmap=None,                    # None -> Matplotlib default (e.g., 'viridis') for continuous; tab10 for categorical
    vrange=None,                  # (vmin, vmax) for continuous labels
    cbar_label=None,               # override colorbar label (continuous labels)
    circmap = 'twilight',
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
    all_x, all_y, all_z = [], [], []
    for _, S in trajs:
        all_x.extend(S[:, dims[0]].tolist())
        all_y.extend(S[:, dims[1]].tolist())
        all_z.extend(S[:, dims[2]].tolist())
    x_range = (max(all_x) - min(all_x)) if all_x else 1
    y_range = (max(all_y) - min(all_y)) if all_y else 1
    z_range = (max(all_z) - min(all_z)) if all_z else 1

    # ----- determine z scaling factor -----
    if z_scale == 'auto':
        xy_avg_range = (x_range + y_range) / 2
        scale_factor = (xy_avg_range / z_range) if z_range > 0 else 1.0
    elif z_scale == 'none':
        scale_factor = 1.0
    else:
        scale_factor = float(z_scale)

    # ===== label mode setup (if labels given) =====
    use_labels = labels is not None
    lab_for = {}
    use_cont = False
    cats = None
    cmap_obj = None
    norm = None

    if use_labels:
        # build per-trial label lookup
        if isinstance(labels, dict):
            for idx, _S in trajs:
                if idx in labels:
                    lab_for[idx] = labels[idx]
        else:
            arr = np.asarray(labels)
            for idx, _S in trajs:
                if idx < arr.shape[0]:
                    lab_for[idx] = arr[idx]

        vals_exist = [lab_for[i] for i, _ in trajs if i in lab_for]
        if len(vals_exist) == 0:
            raise ValueError("Provided labels are empty or misaligned with trials.")

        # decide continuous vs categorical
        if label_mode == 'continuous':
            use_cont = True
        elif label_mode == 'categorical':
            use_cont = False
        else:  # 'auto'
            all_int = all(isinstance(v, (int, np.integer)) for v in vals_exist)
            all_num = all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals_exist)
            use_cont = (not all_int) and all_num

        if use_cont:
            vals = np.array(vals_exist, dtype=float)
            if vrange is None:
                vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                # if looks like degrees in [-180, 180], lock it
                if (vmin >= -180-1e-6) and (vmax <= 180+1e-6):
                    vmin, vmax = -180.0, 180.0
            else:
                vmin, vmax = vrange

            if np.isclose(vmin, -180.0) and np.isclose(vmax, 180.0) and cmap is None:
                cmap_obj = plt.get_cmap(circmap)
            else:
                default_name = plt.rcParams.get('image.cmap', 'viridis')
                cmap_obj = plt.get_cmap(default_name if cmap is None else cmap)
            
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            # categorical palette
            cats = []
            for idx, _ in trajs:
                if idx in lab_for:
                    v = lab_for[idx]
                    if v not in cats:
                        cats.append(v)
            if cmap is None:
                base = plt.get_cmap('tab10')
                cmap_obj = ListedColormap([base(i % base.N) for i in range(len(cats))])
            else:
                cmap_obj = plt.get_cmap(cmap)
            norm = BoundaryNorm(np.arange(len(cats)+1)-0.5, len(cats))
            cat_to_i = {c: i for i, c in enumerate(cats)}
    else:
        # no labels -> we'll use a time-gradient colormap per your original code
        cmap_obj = LinearSegmentedColormap.from_list("o2b", [start_color, end_color])
        norm = Normalize(vmin=0, vmax=(1 if max_len <= 1 else max_len - 1))

    # ----- figure & axes -----
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111, projection="3d")

    # ----- track global data limits (with scaled z) -----
    xmin = ymin = zmin_scaled = np.inf
    xmax = ymax = zmax_scaled = -np.inf

    # ===== plot =====
    for idx, S in trajs:
        # dims + scale z
        x = np.asarray(S[:, dims[0]], dtype=float)
        y = np.asarray(S[:, dims[1]], dtype=float)
        z_raw = np.asarray(S[:, dims[2]], dtype=float)
        z = z_raw * scale_factor

        # decimate
        if decimate > 1:
            x = x[::decimate]; y = y[::decimate]; z = z[::decimate]; z_raw = z_raw[::decimate]
        T = x.shape[0]
        if T == 0:
            continue

        # update global bounds (scaled z)
        xmin = min(xmin, x.min()); xmax = max(xmax, x.max())
        ymin = min(ymin, y.min()); ymax = max(ymax, y.max())
        zmin_scaled = min(zmin_scaled, z.min()); zmax_scaled = max(zmax_scaled, z.max())

        # start marker
        if not use_labels:
            start_col = cmap_obj(norm(0))
        else:
            if idx in lab_for:
                if use_cont:
                    start_col = cmap_obj(norm(float(lab_for[idx])))
                else:
                    start_col = cmap_obj(norm(cat_to_i[lab_for[idx]]))
            else:
                start_col = (0.6, 0.6, 0.6, 1.0)  # gray when missing
        ax.scatter(x[0], y[0], z[0], s=start_marker_size, marker="o",
                   facecolor=start_col, edgecolor="black",
                   linewidths=0.5, alpha=start_marker_alpha, zorder=3)

        if T == 1:
            continue

        if not use_labels:
            # time-gradient segments
            pts = np.column_stack([x, y, z])             # (T, 3)
            segs = np.stack([pts[:-1], pts[1:]], axis=1) # (T-1, 2, 3)
            seg_steps = np.arange(segs.shape[0])         # 0..T-2
            colors = cmap_obj(norm(seg_steps))
            lc = Line3DCollection(segs, linewidths=linewidth, alpha=alpha)
            lc.set_colors(colors)
            ax.add_collection3d(lc)
            end_col = cmap_obj(norm(T-1))
        else:
            # solid color per trajectory from labels
            if idx in lab_for:
                if use_cont:
                    color = cmap_obj(norm(float(lab_for[idx])))
                else:
                    color = cmap_obj(norm(cat_to_i[lab_for[idx]]))
            else:
                color = (0.6, 0.6, 0.6, 1.0)
            ax.plot(x, y, z, lw=linewidth, alpha=alpha, color=color)
            end_col = color

        # end arrow
        dx, dy, dz = x[-1]-x[-2], y[-1]-y[-2], z[-1]-z[-2]
        rng = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()])
        diag = float(np.linalg.norm(rng))
        head_len = max(1e-12, arrow_head_frac * (diag if diag > 0 else 1.0))

        uv = np.array([dx, dy, dz], float)
        nrm = float(np.linalg.norm(uv))

        # Use a slightly larger minimum so the head is visible
        min_head = 0.02 * max(rng.max(), 1e-12)   # 2% of local span
        head_len = max(head_len, min_head)
        
        # If the last step is (near) zero, just place an end marker instead of an arrow
        if nrm < 1e-12:
            ax.scatter(x[-1], y[-1], z[-1],
                       s=start_marker_size*0.9,
                       marker="^", facecolor=end_col, edgecolor="black",
                       linewidths=0.5, alpha=end_arrow_alpha, zorder=3)
        else:
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

    # ----- labels & view -----
    ax.set_xlabel(f"μ{dims[0]+1}")
    ax.set_ylabel(f"μ{dims[1]+1}")
    ax.set_zlabel(f"μ{dims[2]+1}")
    ax.set_title(title)

    # limits + aspect
    # --- Let Matplotlib autoscale using the exact points we plotted (with z scaling) ---
    xs, ys, zs = [], [], []
    for idx, S in trajs:
        x = np.asarray(S[:, dims[0]], float)
        y = np.asarray(S[:, dims[1]], float)
        z = np.asarray(S[:, dims[2]], float) * scale_factor
        if decimate > 1:
            x = x[::decimate]; y = y[::decimate]; z = z[::decimate]
        if x.size:
            xs.append(x); ys.append(y); zs.append(z)
    
    if xs:  # concatenate and hand to the built-in autoscaler
        xs = np.concatenate(xs); ys = np.concatenate(ys); zs = np.concatenate(zs)
        ax.auto_scale_xyz(xs, ys, zs, had_data=False)
    
        # Optional: tiny uniform pad so points aren't glued to the box
        dx = max(1e-12, xs.max() - xs.min())
        dy = max(1e-12, ys.max() - ys.min())
        dz = max(1e-12, zs.max() - zs.min())
        pad = 0.03  # 3% visual padding
        ax.set_xlim(xs.min() - pad*dx, xs.max() + pad*dx)
        ax.set_ylim(ys.min() - pad*dy, ys.max() + pad*dy)
        ax.set_zlim(zs.min() - pad*dz, zs.max() + pad*dz)
    
    # Aspect: keep data aspect unless you explicitly pass something else
    try:
        if aspect_ratio == (1, 1, 1):
            ax.set_box_aspect('auto')   # natural data aspect (no forced cube)
        else:
            ax.set_box_aspect(aspect_ratio)
    except Exception:
        pass

    if elev_azim is not None:
        ax.view_init(elev=elev_azim[0], azim=elev_azim[1])

    # rescale tick labels on z back (if we visually scaled z)
    if scale_factor != 1.0:
        ax.zaxis.set_major_formatter(
            FuncFormatter(lambda val, pos: f"{val/scale_factor:.2f}")
        )

    # ----- colorbar -----
    sm = None
    if not use_labels:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm); sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.08)
        cbar.set_label("Step index")
        ticks = [0, 1] if max_len <= 2 else [0, (max_len - 1)//2, max_len - 1]
        cbar.set_ticks(ticks); cbar.set_ticklabels([str(t) for t in ticks])
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm); sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.08)
        if use_cont:
            lbl = cbar_label if cbar_label is not None else "Label"
            # nice default for angles
            if isinstance(vrange, tuple):
                vmin, vmax = vrange
            else:
                vmin, vmax = norm.vmin, norm.vmax
            if np.isclose(vmin, -180) and np.isclose(vmax, 180) and cbar_label is None:
                lbl = "Angle (deg)"
                cbar.set_ticks([-180, -90, 0, 90, 180])
            cbar.set_label(lbl)
        else:
            cbar.set_ticks(range(len(cats)))
            cbar.set_ticklabels([str(c) for c in cats])
            cbar.set_label("Category")

    # plt.show()


# def plot_mu_trajectories_gradient_steps_3d(
#     mu_traj_np,                   # list of arrays [T_i, D]
#     trial_indices=None,           # iterable of trial ids to plot; None => all
#     dims=(0, 1, 2),               # which μ dims (0-based)
#     start_color="#fdae61",
#     end_color="#313695",
#     linewidth=2.0,
#     alpha=0.5,                    # line alpha
#     start_marker_size=36,
#     start_marker_alpha=0.95,
#     end_arrow_alpha=0.95,
#     arrow_head_frac=0.03,         # head length as fraction of per-trial diagonal
#     arrow_length_ratio=0.6,
#     title="Latent trajectories",
#     fig_w=7, fig_h=6,
#     decimate=1,                   # plot every k-th step along each trial (speed-up)
#     z_scale='auto',               # 'auto', 'none', or a number to multiply z values by
#     aspect_ratio=(1, 1, 1),       # visual aspect ratio for the box
#     elev_azim = None,
# ):
   
#     # ----- choose trials -----
#     if trial_indices is None:
#         trial_indices = range(len(mu_traj_np))

#     # ----- gather trajectories -----
#     trajs, max_len = [], 0
#     for idx in trial_indices:
#         S = mu_traj_np[idx]
#         if S is None or S.size == 0:
#             continue
#         trajs.append((idx, S))
#         max_len = max(max_len, S.shape[0])
#     if not trajs:
#         raise ValueError("No non-empty trials to plot.")
        
#     # ----- compute data ranges for auto-scaling -----
#     all_x = []
#     all_y = []
#     all_z = []
#     for _, S in trajs:
#         all_x.extend(S[:, dims[0]].tolist())
#         all_y.extend(S[:, dims[1]].tolist())
#         all_z.extend(S[:, dims[2]].tolist())
    
#     x_range = max(all_x) - min(all_x) if all_x else 1
#     y_range = max(all_y) - min(all_y) if all_y else 1
#     z_range = max(all_z) - min(all_z) if all_z else 1
    
#     # ----- determine z scaling factor -----
#     if z_scale == 'auto':
#         # Scale z to have visual range comparable to x and y
#         xy_avg_range = (x_range + y_range) / 2
#         if z_range > 0:
#             scale_factor = xy_avg_range / z_range
#         else:
#             scale_factor = 1.0
#     elif z_scale == 'none':
#         scale_factor = 1.0
#     else:
#         scale_factor = float(z_scale)
    
#     # ----- setup color map -----
#     cmap = LinearSegmentedColormap.from_list("o2b", [start_color, end_color])
#     norm = Normalize(vmin=0, vmax=(1 if max_len <= 1 else max_len - 1))

#     # ----- figure & axes -----
#     fig = plt.figure(figsize=(fig_w, fig_h))
#     ax = fig.add_subplot(111, projection="3d")

#     # ----- track global data limits (with scaled z) -----
#     xmin = ymin = zmin_scaled = np.inf
#     xmax = ymax = zmax_scaled = -np.inf
    
#     for _, S in trajs:
#         # pick dims and cast to float
#         x = np.asarray(S[:, dims[0]], dtype=float)
#         y = np.asarray(S[:, dims[1]], dtype=float)
#         z_raw = np.asarray(S[:, dims[2]], dtype=float)
#         z = z_raw * scale_factor  # Apply scaling to z
        
#         # optional decimation for speed
#         if decimate > 1:
#             x = x[::decimate]; y = y[::decimate]; z = z[::decimate]; z_raw = z_raw[::decimate]
#         T = x.shape[0]

#         # update bounds on the scaled data
#         if T > 0:
#             xmin = min(xmin, x.min()); xmax = max(xmax, x.max())
#             ymin = min(ymin, y.min()); ymax = max(ymax, y.max())
#             zmin_scaled = min(zmin_scaled, z.min()); zmax_scaled = max(zmax_scaled, z.max())

#         if T == 0:
#             continue

#         # start marker (using scaled z)
#         ax.scatter(x[0], y[0], z[0], s=start_marker_size, marker="o",
#                    facecolor=cmap(norm(0)), edgecolor="black",
#                    linewidths=0.5, alpha=start_marker_alpha, zorder=3)

#         if T == 1:
#             continue

#         # segments colored by step index (using scaled z)
#         pts = np.column_stack([x, y, z])                   # (T, 3)
#         segs = np.stack([pts[:-1], pts[1:]], axis=1)       # (T-1, 2, 3)
#         seg_steps = np.arange(segs.shape[0])               # 0..T-2
#         colors = cmap(norm(seg_steps))

#         lc = Line3DCollection(segs, linewidths=linewidth, alpha=alpha)
#         lc.set_colors(colors)
#         ax.add_collection3d(lc)

#         # end arrow (using scaled z)
#         dx, dy, dz = x[-1]-x[-2], y[-1]-y[-2], z[-1]-z[-2]
#         end_col = cmap(norm(T-1))
#         rng = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()])
#         diag = float(np.linalg.norm(rng))
#         head_len = max(1e-12, arrow_head_frac * (diag if diag > 0 else 1.0))

#         uv = np.array([dx, dy, dz], float)
#         nrm = float(np.linalg.norm(uv))
#         if nrm == 0:
#             uv = np.array([1.0, 0.0, 0.0]); nrm = 1.0
#         u_hat = uv / nrm
#         tail = np.array([x[-1], y[-1], z[-1]]) - u_hat * head_len

#         ax.quiver(
#             tail[0], tail[1], tail[2],
#             u_hat[0], u_hat[1], u_hat[2],
#             length=head_len, normalize=False,
#             color=end_col, linewidth=linewidth*1.2,
#             arrow_length_ratio=arrow_length_ratio,
#             alpha=end_arrow_alpha
#         )

#     # ----- labels (show scaling info if applied) -----
#     ax.set_xlabel(f"μ{dims[0]+1}")
#     ax.set_ylabel(f"μ{dims[1]+1}")
#     # if scale_factor != 1.0:
#     #     ax.set_zlabel(f"μ{dims[2]+1} (×{scale_factor:.1f})")
#     # else:
#     #     ax.set_zlabel(f"μ{dims[2]+1}")
#     ax.set_zlabel(f"μ{dims[2]+1}")
#     ax.set_title(title)

#     # ----- set limits with padding -----
#     rx = max(1e-12, xmax - xmin)
#     ry = max(1e-12, ymax - ymin)
#     rz_scaled = max(1e-12, zmax_scaled - zmin_scaled)
    
#     ax.set_xlim(xmin - 0.02*rx, xmax + 0.02*rx)
#     ax.set_ylim(ymin - 0.02*ry, ymax + 0.02*ry)
#     ax.set_zlim(zmin_scaled - 0.02*rz_scaled, zmax_scaled + 0.02*rz_scaled)
    
#     # ----- set aspect ratio -----
#     ax.set_box_aspect(aspect_ratio)
#     if elev_azim is not None:
#         ax.view_init(elev=elev_azim[0], azim=elev_azim[1]) 
    
#     if scale_factor != 1.0:
#         zticks = ax.get_zticks()
#         ax.set_zticks(zticks)  # Explicitly set the positions first
#         ax.set_zticklabels([f"{z/scale_factor:.1f}" for z in zticks])

#     # ----- colorbar on the right (shared step index) -----
#     cax = fig.add_axes([0.88, 0.18, 0.03, 0.64])
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
#     cbar = fig.colorbar(sm, cax=cax)
#     cbar.set_label("Step index")
#     ticks = [0, 1] if max_len <= 2 else [0, (max_len - 1)//2, max_len - 1]
#     cbar.set_ticks(ticks); cbar.set_ticklabels([str(t) for t in ticks])

#     plt.show()


def generate_traj_cov(dyn_net, dset_samples, n_step, tau, lag_samples, device, lag,
                      include_x0_tau=False, covStat_samples=None,
                      oracle=False, lag_cov_list_pre=None, flow_net=None, clamp_range=None,
                      reuse = False):
    """
    Vector rollout with extra static covs (prepended before lag).
    reuse flag as above.
    """
    # clamp prep
    full_min = full_max = None
    if clamp_range is not None and len(clamp_range.shape) == 2:
        clamp_range = torch.from_numpy(clamp_range).to(device).float()
        full_min = clamp_range[:, 0].tile(lag + 1).unsqueeze(0)
        full_max = clamp_range[:, 1].tile(lag + 1).unsqueeze(0)

    # init lag (first row)
    lag_cov = torch.stack([
        torch.tensor(lag_samples[j][0], dtype=torch.float32)
        for j in range(len(lag_samples))
    ], dim=0).to(device)

    # x(τ) at first point
    starting_pts = np.stack([dset_samples[ii][0, :] for ii in range(len(dset_samples))], 0)
    starting_pts = torch.from_numpy(starting_pts).unsqueeze(1).float().to(device)  # (B,1,D)
    if tau < 1.0:
        assert flow_net is not None, "flow_net required when tau<1.0."
        curr_tau_pts = util.calc_flow_trajectories(flow_net, starting_pts.squeeze(1), 1.0, tau, nt=2)[-1].unsqueeze(1)
    else:
        curr_tau_pts = starting_pts

    B, _, D = curr_tau_pts.shape
    curr_tau_trajs, lag_cov_list = [curr_tau_pts.cpu().numpy()], []

    for ii in tqdm(range(n_step - 1)):
        # state & lag
        if oracle:
            lag_cov = torch.stack([
                torch.as_tensor(lag_samples[j][ii], dtype=torch.float32, device=device)
                if ii < lag_samples[j].shape[0] else
                torch.as_tensor(lag_samples[j][-1], dtype=torch.float32, device=device)
                for j in range(len(lag_samples))
            ], dim=0)
            x0_tmp = torch.stack([
                torch.as_tensor(dset_samples[j][ii], dtype=torch.float32, device=device)
                if ii < dset_samples[j].shape[0] else
                torch.as_tensor(dset_samples[j][-1], dtype=torch.float32, device=device)
                for j in range(len(dset_samples))
            ], dim=0)
        else:
            x0_tmp = curr_tau_pts.squeeze(1)
            if reuse and (lag_cov_list_pre is not None):
                assert tau < 1.0, "lag_cov_list_pre is only expected for latent τ."
                lag_cov = lag_cov_list_pre[ii]

        # clamp lag
        if clamp_range is not None:
            if len(clamp_range.shape) == 1:
                lag_cov = lag_cov.clamp(min=clamp_range[0], max=clamp_range[1])
            else:
                lag_cov = torch.clamp(lag_cov, min=full_min, max=full_max)

        # prepend covStat if provided
        if covStat_samples is not None:
            covStat_tmp = torch.stack([
                torch.as_tensor(covStat_samples[j][ii], dtype=torch.float32, device=device)
                if ii < covStat_samples[j].shape[0] else
                torch.as_tensor(covStat_samples[j][-1], dtype=torch.float32, device=device)
                for j in range(len(covStat_samples))
            ], dim=0)
            covStat_all = torch.cat((covStat_tmp, lag_cov), dim=1)
        else:
            covStat_all = lag_cov

        # one step in τ
        traj = util.calc_dyn_trajectories(
            dyn_net, x0_tmp, tau, x0_tau=None,
            covariates=None, next_covariates=None,
            covariates_static=covStat_all,
            include_x0_tau=include_x0_tau, nt=2
        )
        x_next = traj[-1]  # (B,D)

        # clamp state
        if clamp_range is not None:
            if len(clamp_range.shape) == 1:
                x_next = x_next.clamp(min=clamp_range[0], max=clamp_range[1])
            else:
                x_next = torch.clamp(x_next, min=clamp_range[:, 0].unsqueeze(0), max=clamp_range[:, 1].unsqueeze(0))

        # push time
        curr_tau_pts = x_next.unsqueeze(1)
        curr_tau_trajs.append(curr_tau_pts.cpu().numpy())

        # roll lag when NOT reusing and not oracle
        if (not oracle) and (not reuse):
            if tau < 1.0:
                assert flow_net is not None, "flow_net required when tau<1.0 and reuse=False."
                x_next_for_lag = util.calc_flow_trajectories(flow_net, x_next, tau, 1.0, nt=2)[-1]
            else:
                x_next_for_lag = x_next
            lag_cov = torch.cat([lag_cov[:, D:], x_next_for_lag], dim=-1)

        # record lag used this step only when building here
        if not reuse:
            lag_cov_list.append(lag_cov)

    curr_tau_trajs = np.concatenate(curr_tau_trajs, axis=1)
    curr_tau_trajs_trunc = [curr_tau_trajs[i, :dset_samples[i].shape[0], :] for i in range(curr_tau_trajs.shape[0])]
    return curr_tau_trajs_trunc, lag_cov_list




def proj_to_latent(dset_samples, flow_net, tau_latent, device):
    latent_traj_list = []
    for ii in tqdm(range(len(dset_samples))):
        latent_traj_tmp = util.calc_flow_trajectories(flow_net,
                                                      torch.from_numpy(dset_samples[ii]).type(torch.float32).to(device),
                                                      1.0, tau_latent, nt=2)
        latent_traj_list.append(latent_traj_tmp[-1,:,:].cpu().numpy())
    return latent_traj_list

# def latent_mu(latent_traj_list, encoder, dim_preserve, device, fullRes = False):
#     mu_traj = []
#     encoder.eval()
#     with torch.inference_mode():
#         X = torch.from_numpy(latent_traj_list[0]).to(device=device, dtype=torch.float32)
#         _, d, L = encoder.to(device).encode(X)
#         loading = L @ torch.sqrt(d)
        
#         for ii in tqdm(range(len(latent_traj_list))):
#             X = torch.from_numpy(latent_traj_list[ii]).to(device=device, dtype=torch.float32)  # [T, in_dim]
#             mu = (torch.linalg.solve(loading, X.T)).T
#             mu_traj_tmp = mu[:, :dim_preserve].detach().cpu()   # [T, dim_preserve]
#             mu_traj.append(mu_traj_tmp)
#     if fullRes:
#         return mu_traj, d, L, loading
#     else:
#         return mu_traj

def latent_mu(latent_traj_list, encoder, dim_preserve, device, fullRes=False,
              min_var=1e-20, rcond=1e-20):
    """
    Stable computation of latent means given X ≈ (L sqrt(D)) mu.
    - min_var: floor for diagonal entries before rsqrt to avoid NaNs
    - rcond:   cutoff for pseudoinverse fallback
    """
    import torch
    from torch import nn
    encoder = encoder.to(device).eval()

    mu_traj = []
    with torch.inference_mode():
        # Get L and D once from any batch (they're global params)
        X0 = torch.from_numpy(latent_traj_list[0]).to(device=device, dtype=torch.float32)
        _, D, L = encoder.encode(X0)

        # Extract diagonal as a vector and make it safe
        d_vec = torch.diagonal(D) if D.ndim == 2 else D
        d_safe = torch.clamp(d_vec, min=float(min_var))
        inv_sqrt_d = d_safe.rsqrt()  # 1 / sqrt(d_safe)

        # Helper: forward-substitution solve (unit lower-triangular)
        def solve_mu(X_batch):
            # X_batch: [T, in_dim]
            XT = X_batch.T  # [in_dim, T]
            try:
                # Prefer torch.linalg.solve_triangular if available
                Y = torch.linalg.solve_triangular(L, XT, upper=False, unitriangular=True)
            except AttributeError:
                # Fallback for older PyTorch
                Y = torch.triangular_solve(XT, L, upper=False, unitriangular=True).solution
            MUt = inv_sqrt_d.unsqueeze(1) * Y  # diag(inv_sqrt_d) @ Y
            return MUt.T                        # [T, in_dim]

        # Try triangular solves first
        bad = False
        for arr in latent_traj_list:
            X = torch.from_numpy(arr).to(device=device, dtype=torch.float32)  # [T, in_dim]
            mu_full = solve_mu(X)                                             # [T, in_dim]
            if torch.isnan(mu_full).any() or torch.isinf(mu_full).any():
                bad = True
                break
            mu_traj.append(mu_full[:, :dim_preserve].detach().cpu())

        # Robust fallback: pseudoinverse of loading if anything went bad
        loading = None
        if bad:
            # Build a *safe* loading with clamped D
            loading = L @ torch.diag(d_safe.sqrt())
            W = torch.linalg.pinv(loading, rcond=float(rcond))  # [in_dim, in_dim]
            mu_traj = []  # recompute all with pinv
            for arr in latent_traj_list:
                X = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
                mu_full = (W @ X.T).T
                mu_traj.append(mu_full[:, :dim_preserve].detach().cpu())

    if fullRes:
        # Return D as a vector for clarity, plus L and (optional) loading if built
        return mu_traj, d_vec, L, (loading if loading is not None else L @ torch.diag(d_vec.sqrt()))
    else:
        return mu_traj




# def latent_mu(latent_traj_list, encoder, dim_preserve, device):
#     mu_traj = []
#     encoder.eval()
#     with torch.inference_mode():                      
#         for ii in tqdm(range(len(latent_traj_list))):
#             X = torch.from_numpy(latent_traj_list[ii]).to(device=device, dtype=torch.float32)  # [T, in_dim]
#             mu = encoder.to(device).mu(X)                         # [T, out_dim]
#             mu_traj_tmp = mu[:, :dim_preserve].detach().cpu()   # [T, dim_preserve]
#             mu_traj.append(mu_traj_tmp)
#     return mu_traj

def list_mse_equal_weight(data_list, sim_list):
    """
    Returns:
      overall_mse: float                      # mean of per-trial scalar MSEs
      per_trial:   (n_trial,) array           # scalar MSE per trial
      per_trial_dim: (n_trial, D) array       # MSE per trial per dimension
    """
    assert len(data_list) == len(sim_list), "Trial count mismatch"
    per_trial = []
    per_trial_dim = []

    D_ref = None
    for X, Y in zip(data_list, sim_list):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.shape != Y.shape:
            raise ValueError(f"Shape mismatch in a trial: {X.shape} vs {Y.shape}")
        if D_ref is None:
            D_ref = X.shape[1]
        elif X.shape[1] != D_ref:
            raise ValueError(f"Inconsistent dims across trials: expected {D_ref}, got {X.shape[1]}")

        diff2 = (X - Y) ** 2            # [T, D]
        per_trial.append(diff2.mean())  # mean over time and dims
        per_trial_dim.append(diff2.mean(axis=0))  # mean over time -> [D]

    per_trial = np.asarray(per_trial)                  # [n_trial]
    per_trial_dim = np.vstack(per_trial_dim)          # [n_trial, D]
    overall_mse = float(per_trial.mean())             # scalar
    overall_mse_dim = np.mean(per_trial_dim, axis=0)
    
    return overall_mse, per_trial, overall_mse_dim, per_trial_dim


def mean_traj_by_labels(traj_array, label_array, round_labels=None):
    N, T, D = traj_array.shape
    labels = label_array
    if round_labels is not None:
        labels = np.round(labels.astype(float), round_labels)

    unique_labels, inv, counts = np.unique(labels, axis=0, return_inverse=True, return_counts=True)
    G = unique_labels.shape[0]

    sums = np.zeros((G, T, D), dtype=np.float64)
    np.add.at(sums, inv, traj_array)            # accumulate trial-wise
    means = sums / counts[:, None, None]        # broadcast divide

    return unique_labels, counts, means, inv



def get_theta_deg(XY):
    x = XY[:, 0]
    y = XY[:, 1]
    r = np.hypot(x, y)
    theta = np.arctan2(x, y)              # <-- note: x first, y second (swapped)
    theta = (theta + np.pi) % (2*np.pi) - np.pi
    theta[np.isclose(theta, np.pi)] = 0.0
    theta_deg = np.degrees(theta)
    return theta_deg

def model_gen(folder_data, folder, chk_pts, dim_preserve, clamp_range, device, useCov = False):
    pkl_path = folder + chk_pts
    dataset_samples = np.load(folder_data + "/dataset_samples.npz", allow_pickle=True)
    lagRead_samples = np.load(folder_data + "/lag_cov_list.npz", allow_pickle=True)
    if useCov:
        cov_static_all = np.load(folder_data + "/cov_static_new.npz", allow_pickle=True)
    
    data_raw = dataset_samples['samples']
    lag_raw = lagRead_samples['samples']
    dset_samples = []
    lag_samples = []

    if useCov:
        cov_static_raw = cov_static_all['samples']
        covStat_samples = []
    
    for ii in range(len(data_raw)):
        dset_samples.append(np.asarray(data_raw[ii], dtype=np.float64))
        lag_samples.append(np.asarray(lag_raw[ii], dtype=np.float64))
        if useCov:
            covStat_samples.append(np.asarray(cov_static_raw[ii][:, 0:(cov_static_raw[ii].shape[1] - lag_raw[ii].shape[1])], dtype=np.float64))
    
    
    lag = lag_samples[0].shape[1] // dset_samples[0].shape[1] - 1
    T_min = min(arr.shape[0] for arr in dset_samples)
    T_max = max(arr.shape[0] for arr in dset_samples)
    
    with open(pkl_path, 'rb') as f:
        model_all = pickle.load(f)
    if 'ema' in model_all: net = model_all['ema']  # EMA-stabilized model (recommended for inference/simulation)
    else: net = model_all['net']  # Raw trained model
    flow_net = net.unet_model  # Compressive flow (u_theta)
    dyn_net = net.vnet_model   # Dynamics flow (v_theta)
    encoder = net.encoder      # For latent proposals
    
    
    _, d, L, loading = latent_mu(dset_samples, encoder, dim_preserve, device, fullRes = True)
    
    if useCov:
        traj_sim_tau1, lag_cov_list = generate_traj_cov(dyn_net.to(device), dset_samples,
                                                        n_step = T_max, tau = 1.0,
                                                        lag_samples = lag_samples,
                                                        device = device, lag = lag,
                                                        include_x0_tau = False,
                                                        covStat_samples = covStat_samples,
                                                        oracle = False, lag_cov_list_pre = None,
                                                        clamp_range = clamp_range)
        traj_sim_tau0, _ = generate_traj_cov(dyn_net.to(device), dset_samples,
                                             n_step = T_max, tau = 0.0,
                                             lag_samples = lag_samples,
                                             device = device, lag = lag,
                                             include_x0_tau = False,
                                             covStat_samples = covStat_samples,
                                             oracle = False, lag_cov_list_pre = lag_cov_list,
                                             flow_net = flow_net.to(device), clamp_range = clamp_range)
    else:
        traj_sim_tau1, lag_cov_list = generate_traj(dyn_net.to(device), dset_samples,
                                                    n_step = T_max, tau = 1.0,
                                                    lag_samples = lag_samples, device = device, lag = lag,
                                                    include_x0_tau = False,
                                                    oracle = False, lag_cov_list_pre = None,
                                                    clamp_range = clamp_range)
        
        traj_sim_tau0, _ = generate_traj(dyn_net.to(device), dset_samples,
                                         n_step = T_max, tau = 0.0,
                                         lag_samples = lag_samples, device = device, lag = lag,
                                         include_x0_tau = False,
                                         oracle = False, lag_cov_list_pre = lag_cov_list,
                                         flow_net = flow_net.to(device), clamp_range = clamp_range)

    if useCov:
        return dset_samples, lag_samples, covStat_samples, flow_net, dyn_net, encoder, d, L, loading, traj_sim_tau1, traj_sim_tau0
    else:
        return dset_samples, lag_samples, flow_net, dyn_net, encoder, d, L, loading, traj_sim_tau1, traj_sim_tau0



def plot_mu_trajectories_gradient_steps_2d(
    mu_traj_np,                   # list of arrays [T_i, D]
    trial_indices=None,           # iterable of trial ids to plot; None => all
    dims=(0, 1),                  # which μ dims (0-based)
    start_color="#fdae61",
    end_color="#313695",
    linewidth=2.0,
    alpha=0.5,                    # line alpha
    start_marker_size=36,
    start_marker_alpha=0.95,
    end_arrow_alpha=0.95,
    arrow_head_frac=0.03,         # head length as fraction of per-trial diagonal (data units)
    arrow_length_ratio=0.6,       # kept for API symmetry (not used by quiver directly)
    title="Latent trajectories (2D)",
    fig_w=6, fig_h=6,
    decimate=1,                   # plot every k-th step along each trial (speed-up)
    aspect_equal=False,            # equal axes by default
    # Labels (same semantics as your 3D version)
    labels=None,                  # per-trial values -> color each whole trajectory
    label_mode='auto',            # 'auto' | 'continuous' | 'categorical'
    cmap=None,                    # None -> default (e.g., 'viridis') for continuous; tab10 for categorical
    vrange=None,                  # (vmin, vmax) for continuous labels
    cbar_label=None,              # override colorbar label (continuous labels)
    circmap='twilight'            # used for angles if [-180, 180]
):
    # ----- choose trials -----
    if trial_indices is None:
        trial_indices = range(len(mu_traj_np))

    # ----- gather trajectories -----
    trajs, max_len = [], 0
    for idx in trial_indices:
        S = mu_traj_np[idx]
        if S is None or getattr(S, "size", 0) == 0:
            continue
        trajs.append((idx, S))
        max_len = max(max_len, S.shape[0])
    if not trajs:
        raise ValueError("No non-empty trials to plot.")

    # ===== label mode setup (if labels given) =====
    use_labels = labels is not None
    lab_for = {}
    use_cont = False
    cats = None
    cmap_obj = None
    norm = None

    if use_labels:
        # map trial index -> label
        if isinstance(labels, dict):
            for idx, _S in trajs:
                if idx in labels:
                    lab_for[idx] = labels[idx]
        else:
            arr = np.asarray(labels)
            for idx, _S in trajs:
                if idx < arr.shape[0]:
                    lab_for[idx] = arr[idx]

        vals_exist = [lab_for[i] for i, _ in trajs if i in lab_for]
        if len(vals_exist) == 0:
            raise ValueError("Provided labels are empty or misaligned with trials.")

        # decide continuous vs categorical
        if label_mode == 'continuous':
            use_cont = True
        elif label_mode == 'categorical':
            use_cont = False
        else:  # 'auto'
            all_int = all(isinstance(v, (int, np.integer)) for v in vals_exist)
            all_num = all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals_exist)
            use_cont = (not all_int) and all_num

        if use_cont:
            vals = np.array(vals_exist, dtype=float)
            if vrange is None:
                vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                # lock to degrees if it fits [-180, 180]
                if (vmin >= -180-1e-6) and (vmax <= 180+1e-6):
                    vmin, vmax = -180.0, 180.0
            else:
                vmin, vmax = vrange

            if np.isclose(vmin, -180.0) and np.isclose(vmax, 180.0) and cmap is None:
                cmap_obj = plt.get_cmap(circmap)
            else:
                default_name = plt.rcParams.get('image.cmap', 'viridis')
                cmap_obj = plt.get_cmap(default_name if cmap is None else cmap)

            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            # categorical palette
            cats = []
            for idx, _ in trajs:
                if idx in lab_for:
                    v = lab_for[idx]
                    if v not in cats:
                        cats.append(v)
            if cmap is None:
                base = plt.get_cmap('tab10')
                cmap_obj = ListedColormap([base(i % base.N) for i in range(len(cats))])
            else:
                cmap_obj = plt.get_cmap(cmap)
            norm = BoundaryNorm(np.arange(len(cats)+1)-0.5, len(cats))
            cat_to_i = {c: i for i, c in enumerate(cats)}
    else:
        # no labels -> time-gradient colormap per trajectory
        cmap_obj = LinearSegmentedColormap.from_list("o2b", [start_color, end_color])
        norm = Normalize(vmin=0, vmax=(1 if max_len <= 1 else max_len - 1))

    # ----- figure & axes -----
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ----- global limits -----
    xmin = ymin = np.inf
    xmax = ymax = -np.inf

    # ===== plot =====
    for idx, S in trajs:
        x = np.asarray(S[:, dims[0]], dtype=float)
        y = np.asarray(S[:, dims[1]], dtype=float)

        # decimate
        if decimate > 1:
            x = x[::decimate]
            y = y[::decimate]
        T = x.shape[0]
        if T == 0:
            continue

        # update bounds
        xmin = min(xmin, np.min(x))
        xmax = max(xmax, np.max(x))
        ymin = min(ymin, np.min(y))
        ymax = max(ymax, np.max(y))

        # start marker color
        if not use_labels:
            start_col = cmap_obj(norm(0))
        else:
            if idx in lab_for:
                if use_cont:
                    start_col = cmap_obj(norm(float(lab_for[idx])))
                else:
                    start_col = cmap_obj(norm(cat_to_i[lab_for[idx]]))
            else:
                start_col = (0.6, 0.6, 0.6, 1.0)
        ax.scatter(x[0], y[0], s=start_marker_size, marker="o",
                   facecolor=start_col, edgecolor="black",
                   linewidths=0.5, alpha=start_marker_alpha, zorder=3)

        if T == 1:
            continue

        if not use_labels:
            # time-gradient colored segments
            pts = np.column_stack([x, y])               # (T, 2)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)# (T-1, 2, 2)
            seg_steps = np.arange(segs.shape[0])        # 0..T-2
            colors = cmap_obj(norm(seg_steps))
            lc = LineCollection(segs, linewidths=linewidth, alpha=alpha)
            lc.set_colors(colors)
            ax.add_collection(lc)
            end_col = cmap_obj(norm(T-1))
        else:
            # solid color per trajectory
            if idx in lab_for:
                if use_cont:
                    color = cmap_obj(norm(float(lab_for[idx])))
                else:
                    color = cmap_obj(norm(cat_to_i[lab_for[idx]]))
            else:
                color = (0.6, 0.6, 0.6, 1.0)
            ax.plot(x, y, lw=linewidth, alpha=alpha, color=color)
            end_col = color

        # end arrow
        dx, dy = x[-1] - x[-2], y[-1] - y[-2]
        uv = np.array([dx, dy], float)
        nrm = float(np.linalg.norm(uv))

        # compute diagonal span for head length in data units
        rng = np.array([x.max()-x.min(), y.max()-y.min()], dtype=float)
        diag = float(np.linalg.norm(rng))
        head_len = max(1e-12, arrow_head_frac * (diag if diag > 0 else 1.0))

        # ensure visibility (2% of local span)
        min_head = 0.02 * max(rng.max(), 1e-12)
        head_len = max(head_len, min_head)

        if nrm < 1e-12:
            ax.scatter(x[-1], y[-1],
                       s=start_marker_size*0.9, marker="^",
                       facecolor=end_col, edgecolor="black",
                       linewidths=0.5, alpha=end_arrow_alpha, zorder=3)
        else:
            u_hat = uv / nrm
            # draw arrow of length=head_len in data units
            ax.quiver(
                x[-1] - u_hat[0]*head_len, y[-1] - u_hat[1]*head_len,
                u_hat[0]*head_len, u_hat[1]*head_len,
                angles='xy', scale_units='xy', scale=1.0,
                color=end_col, alpha=end_arrow_alpha, width=0.003
            )

    # ----- axes cosmetics -----
    ax.set_xlabel(f"μ{dims[0]+1}")
    ax.set_ylabel(f"μ{dims[1]+1}")
    ax.set_title(title)
    # limits + aspect
    if not np.isfinite([xmin, xmax, ymin, ymax]).all():
        xmin, xmax, ymin, ymax = -1, 1, -1, 1
    pad_x = 0.02 * max(1e-12, xmax - xmin)
    pad_y = 0.02 * max(1e-12, ymax - ymin)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    if aspect_equal:
        ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)

    # ----- colorbar -----
    sm = None
    if not use_labels:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm); sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
        cbar.set_label("Step index")
        ticks = [0, 1] if max_len <= 2 else [0, (max_len - 1)//2, max_len - 1]
        cbar.set_ticks(ticks); cbar.set_ticklabels([str(t) for t in ticks])
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm); sm.set_array([])
        if use_cont:
            cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
            lbl = cbar_label if cbar_label is not None else "Label"
            if isinstance(vrange, tuple):
                vmin, vmax = vrange
            else:
                vmin, vmax = norm.vmin, norm.vmax
            if np.isclose(vmin, -180) and np.isclose(vmax, 180) and cbar_label is None:
                lbl = "Angle (deg)"
                cbar.set_ticks([-180, -90, 0, 90, 180])
            cbar.set_label(lbl)
        else:
            # bottom categorical legend-style colorbar (like your PCA example)
            fig.subplots_adjust(bottom=0.2)
            cax = fig.add_axes([0.12, 0.08, 0.76, 0.05])  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cbar.set_ticks(range(len(cats)))
            cbar.set_ticklabels([str(c) for c in cats])
            cbar.set_label("Category")

    return fig, ax

