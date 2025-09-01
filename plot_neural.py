import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap, BoundaryNorm

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


#### 1. PCA related
## 1.1 PCA
def pca_stream(trials, n_components=None, dtype=np.float64):
    """
    trials: list of arrays with shapes [T_k, d] (d=137)
    n_components: keep top-k PCs (default: all)
    returns:
        mean (d,),
        components (d, k)  # columns are PC directions
        explained_variance (k,),
        explained_variance_ratio (k,),
        transform_fn  # function to stream-project new trials
    """
    # --- accumulate mean and M2 (sum of centered outer products) in a single pass
    N = 0
    d = trials[0].shape[1]
    mean = np.zeros(d, dtype=dtype)
    M2 = np.zeros((d, d), dtype=dtype)

    for X in trials:
        if X.size == 0: 
            continue
        X = np.asarray(X, dtype=dtype)
        m = X.shape[0]
        # block stats
        block_mean = X.mean(axis=0)
        Xc = X - block_mean
        block_M2 = Xc.T @ Xc  # d x d

        # combine A (N, mean, M2) with B (m, block_mean, block_M2)
        delta = block_mean - mean
        newN = N + m
        if newN > 0:
            mean = mean + delta * (m / newN)
        M2 = M2 + block_M2 + np.outer(delta, delta) * (N * m / max(newN, 1))
        N = newN

    if N < 2:
        raise ValueError("Not enough total samples for PCA.")

    # sample covariance
    cov = M2 / (N - 1)

    # eigendecomposition (cov is symmetric)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # keep top-k
    if n_components is None or n_components > d:
        n_components = d
    evals_k = np.clip(evals[:n_components], 0, None)
    comps_k = evecs[:, :n_components]
    evr_k = evals_k / np.maximum(evals.sum(), 1e-12)

    # transformer that streams projections trial-by-trial
    def transform(trial_iterable):
        """
        Yields projected scores for each trial in trial_iterable
        without stacking, each as [T_k, n_components].
        """
        for X in trial_iterable:
            X = np.asarray(X, dtype=dtype)
            if X.size == 0:
                yield np.empty((0, n_components), dtype=dtype)
            else:
                yield (X - mean) @ comps_k

    return mean, comps_k, evals_k, evr_k, transform


## 1.2 scree plot
def plot_scree(explained_variance, explained_variance_ratio, fig_w = 10, fig_h = 4):
    k = len(explained_variance)
    xs = np.arange(1, k+1)
    fig, ax = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    ax[0].plot(xs, explained_variance_ratio, marker='o')
    ax[0].set_xlabel("PC")
    ax[0].set_ylabel("Explained variance ratio")
    ax[0].set_title("Scree")
    ax[1].plot(xs, np.cumsum(explained_variance_ratio), marker='o')
    ax[1].set_xlabel("PC")
    ax[1].set_ylabel("Cumulative EVR")
    ax[1].set_title("Cumulative variance")
    ax[1].set_ylim(0,1.01)
    plt.show()

## 1.3 2D projection
def plot_trial_trajectories_gradient_steps(
    transform,
    trials,
    trial_indices,
    comps=(0, 1),
    evr=None,                    # explained_variance_ratio from pca_stream
    start_color="#fdae61",       # orange
    end_color="#313695",         # blue
    linewidth=2.0,
    start_marker_size=36,
    alpha=0.5,
    start_marker_alpha=0.8,
    end_arrow_alpha=0.8,
    title=None,
    fig_w = 6,
    fig_h = 6
):
    """
    Plot full 2D PC trajectories with:
      • start circle
      • end arrow
      • per-segment color mapped to the ACTUAL step index
      • shared colorbar across trials (0 .. max_step)

    transform: callable from pca_stream(...), yields [T_k, n_components] arrays
    trials:    list of [T_k, d]
    trial_indices: iterable of trial indices to plot
    comps:     (pc_x, pc_y)
    evr:       explained_variance_ratio array (to annotate axis labels)
    """

    # 1) Project selected trials (once) and find the global max length
    proj = []
    max_len = 0
    for idx in trial_indices:
        S = next(transform([trials[idx]]))  # [T_k, n_components]
        if S.size == 0: 
            continue
        proj.append((idx, S))
        max_len = max(max_len, S.shape[0])

    if len(proj) == 0 or max_len < 1:
        raise ValueError("No non-empty trials to plot.")

    # 2) Colormap + normalization by ACTUAL step index
    cmap = LinearSegmentedColormap.from_list("o2b", [start_color, end_color])
    norm = Normalize(vmin=0, vmax=max_len - 1)

    # 3) Figure and axes
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for idx, S in proj:
        if S.shape[0] == 1:
            # single point: just draw start marker with its color
            x0, y0 = S[0, comps[0]], S[0, comps[1]]
            ax.scatter(x0, y0, s=start_marker_size, facecolor=cmap(norm(0)),
                       edgecolor="black", linewidths=0.5, zorder=3, label=f"trial {idx}")
            continue

        x, y = S[:, comps[0]], S[:, comps[1]]
        pts = np.column_stack([x, y])
        # segments: N-1 segments each colored by its START step index (0..len-2)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_steps = np.arange(segs.shape[0])  # 0..T_k-2

        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=linewidth, alpha=alpha)
        lc.set_array(seg_steps)               # <-- actual step index drives color
        ax.add_collection(lc)

        # start marker (circle) at step 0 color
        ax.scatter(x[0], y[0], s=start_marker_size, facecolor=cmap(norm(0)),
                   edgecolor="black", linewidths=0.5, zorder=3, alpha=start_marker_alpha)

        # end arrow colored by its actual last step index (T_k-1)
        end_color_i = cmap(norm(S.shape[0]-1))
        ax.annotate(
            "",
            xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
            arrowprops=dict(arrowstyle="->", color=end_color_i, lw=linewidth*1.2,
                            shrinkA=0, shrinkB=0, alpha=end_arrow_alpha),
            zorder=4
        )

    # 4) Labels (with explained variance if provided)
    xlab = f"PC{comps[0]+1}"
    ylab = f"PC{comps[1]+1}"
    if evr is not None:
        try:
            xlab += f" ({evr[comps[0]]*100:.1f}% EV)"
            ylab += f" ({evr[comps[1]]*100:.1f}% EV)"
        except Exception:
            pass
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if title is None:
        title = "PC trajectories (color = step index)"
    ax.set_title(title)

    ax.axhline(0, lw=0.5, color="k", alpha=0.3)
    ax.axvline(0, lw=0.5, color="k", alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    # 5) Shared colorbar (0 .. max_len-1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("Step index")
    ticks = [0, (max_len - 1) // 2, max_len - 1] if max_len > 2 else [0, max_len - 1]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])

    plt.tight_layout()
    plt.show()

## 1.4 3D projection
def plot_trial_trajectories_gradient_steps_3d(
    transform,
    trials,
    trial_indices,
    comps=(0, 1, 2),
    evr=None,
    start_color="#fdae61",     # orange
    end_color="#313695",       # blue
    linewidth=2.0,
    alpha=0.5,                # line alpha
    start_marker_size=36,
    start_marker_alpha=0.95,   # start circle alpha
    end_arrow_alpha=0.95,      # end arrow alpha
    arrow_head_frac=0.03,      # head length as fraction of per-trial diagonal
    arrow_length_ratio=0.6,    # quiver head/body ratio
    title=None,
    zlabel_on_left=True,       # <--- render the Z label on left margin
    fig_w = 6,
    fig_h = 6
):
    """
    3D PC trajectories with:
      • start circle
      • end arrow (visible even if last step is tiny)
      • per-segment color = actual step index (shared 0..max_step)
      • colorbar on the RIGHT (own 2D axes)
      • Z label on the LEFT margin (no overlap), with EVR if provided
    """
    # --- Project selected trials; find max length for shared color scale
    proj, max_len = [], 0
    for idx in trial_indices:
        S = next(transform([trials[idx]]))  # [T_k, n_components]
        if S.size == 0:
            continue
        proj.append((idx, S))
        max_len = max(max_len, S.shape[0])
    if not proj:
        raise ValueError("No non-empty trials to plot.")

    # --- Colormap + normalization by ACTUAL step index (robust for max_len==1)
    cmap = LinearSegmentedColormap.from_list("o2b", [start_color, end_color])
    norm = Normalize(vmin=0, vmax=(1 if max_len <= 1 else max_len - 1))

    # --- Figure + 3D axes
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.08, top=0.92)
    ax = fig.add_subplot(111, projection="3d")

    for idx, S in proj:
        if S.shape[0] == 1:
            x0, y0, z0 = S[0, comps[0]], S[0, comps[1]], S[0, comps[2]]
            ax.scatter(x0, y0, z0, s=start_marker_size, marker="o",
                       facecolor=cmap(norm(0)), edgecolor="black",
                       linewidths=0.5, alpha=start_marker_alpha, zorder=3)
            continue

        x, y, z = S[:, comps[0]], S[:, comps[1]], S[:, comps[2]]
        pts = np.column_stack([x, y, z])                # (T_k, 3)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)    # (T_k-1, 2, 3)
        seg_steps = np.arange(segs.shape[0])            # 0..T_k-2
        colors = cmap(norm(seg_steps))                  # (T_k-1, 4)

        lc = Line3DCollection(segs, linewidths=linewidth, alpha=alpha)
        lc.set_colors(colors)
        ax.add_collection3d(lc)

        # start marker (step 0 color)
        ax.scatter(x[0], y[0], z[0], s=start_marker_size, marker="o",
                   facecolor=cmap(norm(0)), edgecolor="black",
                   linewidths=0.5, alpha=start_marker_alpha, zorder=3)

        # end arrow (scale head to data, not to tiny last step)
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

    # --- Labels (with EVR)
    def lab(i):
        base = f"PC{comps[i]+1}"
        if evr is not None and len(evr) > comps[i]:
            base += f" ({evr[comps[i]]*100:.1f}% EV)"
        return base
    ax.set_xlabel(lab(0))
    ax.set_ylabel(lab(1))

    # Use normal Z label OR a left-margin fig label
    z_text = lab(2)
    if zlabel_on_left:
        # hide default zlabel and place a figure-level rotated label on the left margin
        ax.set_zlabel("")  # no overlap possible now
        fig.text(0.035, 0.52, z_text, rotation=90, va="center", ha="center")
    else:
        ax.set_zlabel(z_text, labelpad=12)

    ax.set_title(title or "3D PC trajectories (color = step index)")
    ax.set_box_aspect((1, 1, 1))
    # optional: consistent view
    # ax.view_init(elev=20, azim=35)

    # --- Colorbar on the RIGHT in its own 2D axes (never overlaps)
    cax = fig.add_axes([0.88, 0.18, 0.03, 0.64])   # [left, bottom, width, height]
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

## 1.4 trajectory for each PC
def plot_pc_time_series(
    transform,
    trials,
    pcs=(0,),
    trial_indices=None,
    labels=None,           # None -> black; else categorical (array-like or {idx: label})
    evr=None,
    dt=None,
    linewidth=1.2,
    alpha=0.9,
    fig_w=8,
    fig_h_per = 2,
):
    # pick trials
    if trial_indices is None:
        trial_indices = list(range(len(trials)))
    pcs = (pcs,) if isinstance(pcs, (int, np.integer)) else list(pcs)

    # project once per trial
    proj = {}
    for idx in trial_indices:
        S = next(transform([trials[idx]]))
        if S.size > 0:
            proj[idx] = S
    if not proj:
        raise ValueError("No non-empty trials to plot.")

    # categorical colors (unique color per unique label)
    use_cats = labels is not None
    if use_cats:
        if isinstance(labels, dict):
            lab_for = {i: labels.get(i) for i in trial_indices}
        else:
            arr = np.asarray(labels, dtype=object)
            lab_for = {i: arr[i] for i in trial_indices}
        # preserve first-seen order
        cats = []
        for i in trial_indices:
            if i in proj:
                v = lab_for[i]
                if v not in cats:
                    cats.append(v)
        K = max(1, len(cats))
        # evenly spaced hues → distinct colors even for many categories
        hsv = plt.get_cmap("hsv", K)
        cmap = ListedColormap([hsv(i) for i in range(K)])
        norm = BoundaryNorm(np.arange(K+1)-0.5, K)
        cat_to_i = {c:i for i,c in enumerate(cats)}

    # figure & axes
    fig_h = fig_h_per * len(pcs)
    fig, axes = plt.subplots(len(pcs), 1, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.ravel()

    for ax, pc in zip(axes, pcs):
        for idx in trial_indices:
            if idx not in proj:
                continue
            y = proj[idx][:, pc]
            x = np.arange(len(y)) if dt is None else np.arange(len(y)) * dt
            if use_cats:
                color = cmap(norm(cat_to_i[lab_for[idx]]))
            else:
                color = "k"
            ax.plot(x, y, lw=linewidth, alpha=alpha, color=color)

        ylab = f"PC{pc+1}"
        if evr is not None and pc < len(evr):
            ylab += f" ({evr[pc]*100:.1f}% EV)"
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (step)" if dt is None else "Time")

    # put a horizontal categorical colorbar at the very bottom (no overlap)
    if use_cats:
        fig.subplots_adjust(bottom=0.18)                # reserve space
        cax = fig.add_axes([0.12, 0.08, 0.76, 0.04])    # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks(range(len(cats)))
        cbar.set_ticklabels([str(c) for c in cats])
        cbar.set_label("Trial category")

    plt.tight_layout(rect=[0.02, 0.18 if use_cats else 0.02, 0.98, 0.98])
    plt.show()





