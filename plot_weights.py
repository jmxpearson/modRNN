"""Preview script for weight matrix visualization."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from model import RateRNN


def _spectral_order(similarity: np.ndarray) -> np.ndarray:
    """Compute spectral reordering from a similarity matrix."""
    degree = np.diag(np.abs(similarity).sum(axis=1))
    laplacian = degree - similarity
    _, eigvecs = np.linalg.eigh(laplacian)
    return np.argsort(eigvecs[:, 1])


def cluster_and_reorder(
    W_rec: np.ndarray, method: str = "ward"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster the recurrent weight matrix and return separate row/column orderings.

    Args:
        W_rec: Recurrent weight matrix (hidden_size, hidden_size)
        method: Clustering method - "ward" (hierarchical) or "spectral"

    Returns:
        row_order: Permutation indices for rows (clustering rows as observations)
        col_order: Permutation indices for columns (clustering columns as observations)
    """
    if method == "ward":
        # Cluster rows: each row is an observation (incoming weights to each neuron)
        Z_rows = linkage(W_rec, method="ward")
        row_order = leaves_list(Z_rows)
        # Cluster columns: each column is an observation (outgoing weights from each neuron)
        Z_cols = linkage(W_rec.T, method="ward")
        col_order = leaves_list(Z_cols)
    elif method == "spectral":
        row_order = _spectral_order(W_rec @ W_rec.T)
        col_order = _spectral_order(W_rec.T @ W_rec)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return row_order, col_order


def _symmetric_clim(*arrays, percentile=99):
    """Compute symmetric color limits from weight arrays."""
    combined = np.concatenate([a.flatten() for a in arrays])
    vmax = float(np.percentile(np.abs(combined), percentile))
    return -vmax, vmax


def _has_neuromodulation(model) -> bool:
    """Check if model has neuromodulation parameters."""
    return hasattr(model, 'U') and hasattr(model, 'V') and hasattr(model, 'nm_rank')


def _compute_gain_masks(U: np.ndarray, V: np.ndarray) -> list[np.ndarray]:
    """Compute log10 of gain masks G_k at s=1 for all ranks.

    The gain mask formula is: G = softplus(1 + U diag(s - 0.5) V^T)
    At s=1, this becomes: G_k = softplus(1 + 0.5 * U[:,k] @ V[:,k].T)
    Returns log10(G_k) for more informative visualization.
    """
    nm_rank = U.shape[1]
    masks = []
    for k in range(nm_rank):
        uv_outer = np.outer(U[:, k], V[:, k])
        G_k = np.log1p(np.exp(1 + 0.5 * uv_outer))  # softplus
        masks.append(np.log10(G_k))  # log10 for visualization
    return masks


def _mask_clim(*masks, percentile=99):
    """Compute symmetric color limits centered at 0 for log10 gain masks.

    Uses percentile to avoid saturation from outliers.
    """
    combined = np.concatenate([m.flatten() for m in masks])
    max_dev = float(np.percentile(np.abs(combined), percentile))
    return -max_dev, max_dev


def _plot_heatmap(ax, data, vmin, vmax, cmap="RdBu_r", **kwargs):
    """Plot a weight matrix heatmap with common defaults."""
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for k, v in kwargs.items():
        getattr(ax, f"set_{k}")(v)
    return im


def _extract_weights(model, method):
    """Extract and optionally cluster weight matrices from model.

    Returns:
        dict with keys: W_in, W_rec, W_out, hidden_size, input_size,
        dale_active, dale_ratio, n_exc, n_input_e, n_input_i,
        n_input_neurons, method
    """
    W_in = model.w_in.weight.detach().cpu().numpy()  # (hidden, input)
    W_rec = model.w_rec.weight.detach().cpu().numpy()  # (hidden, hidden)
    W_out = model.w_out.weight.detach().cpu().numpy()  # (1, hidden)

    hidden_size, input_size = W_in.shape

    dale_ratio = model.dale_ratio
    dale_active = dale_ratio is not None
    n_exc = int(dale_ratio * hidden_size) if dale_ratio is not None else 0
    n_input_e = model.n_input_e if model.n_input_e is not None else 0
    n_input_i = model.n_input_i if model.n_input_i is not None else 0

    # Determine how many neurons receive input (non-Dale case)
    if model.input_fraction is not None:
        n_input_neurons = int(model.input_fraction * hidden_size)
    elif not dale_active:
        n_input_neurons = hidden_size
    else:
        n_input_neurons = 0  # handled by Dale split

    # Cluster the recurrent matrix (non-Dale case only)
    if not dale_active:
        row_order, col_order = cluster_and_reorder(W_rec, method=method)
        W_rec = W_rec[np.ix_(row_order, col_order)]
        W_in = W_in[col_order, :]
        W_out = W_out[:, row_order]

    # Check for neuromodulation parameters
    has_neuromod = _has_neuromodulation(model)
    nm_rank = model.nm_rank if has_neuromod else 0
    U = model.U.detach().cpu().numpy() if has_neuromod else None
    V = model.V.detach().cpu().numpy() if has_neuromod else None

    return {
        "W_in": W_in,
        "W_rec": W_rec,
        "W_out": W_out,
        "hidden_size": hidden_size,
        "input_size": input_size,
        "dale_active": dale_active,
        "dale_ratio": dale_ratio,
        "n_exc": n_exc,
        "n_input_e": n_input_e,
        "n_input_i": n_input_i,
        "n_input_neurons": n_input_neurons,
        "method": method,
        "has_neuromod": has_neuromod,
        "nm_rank": nm_rank,
        "U": U,
        "V": V,
    }


def _plot_dale_layout(
    W_in,
    W_rec,
    W_out,
    vmin,
    vmax,
    input_labels,
    hidden_size,
    input_size,
    n_exc,
    n_input_e,
    n_input_i,
    gain_masks=None,
):
    """Create the Dale's law weight matrix layout with optional neuromodulation masks.

    Layout without neuromod:
        [Out_E] [gap] [E->E] [gap] [I->E]
        [Out_I] [gap] [E->I] [gap] [I->I]
                      [E_in]       [I_in]

    Layout with neuromod (nm_rank masks to the right):
        [Out_E] [E->E] [I->E] |cbar| <nm_gap> [G1_EE] [G1_IE] ... |mask_cbar|
        [Out_I] [E->I] [I->I]                 [G1_EI] [G1_II]
                [E_in] [I_in]
    """
    nm_rank = len(gain_masks) if gain_masks is not None else 0

    margin_left = 0.08
    margin_right = 0.12
    margin_bottom = 0.08
    margin_top = 0.08
    gap = 0.10

    total_height = 1.0 - margin_bottom - margin_top

    output_rel = 0.5
    input_height_rel = 1.5
    rec_height_rel = 8.0

    n_inh = hidden_size - n_exc
    rec_e_rel = 8.0 * (n_exc / hidden_size)
    rec_i_rel = 8.0 * (n_inh / hidden_size)
    inner_gap_rel = gap * 0.5
    nm_gap_rel = gap * 2.0  # Larger gap before modulator section

    # Calculate total width including modulator blocks
    base_width_rel = output_rel + gap + rec_e_rel + inner_gap_rel + rec_i_rel
    mod_block_rel = rec_e_rel + inner_gap_rel + rec_i_rel  # Each mod block same as rec
    total_width_rel = base_width_rel + nm_rank * (nm_gap_rel + mod_block_rel)

    # Scale figure width based on content
    fig_width = 10.0 * (1 + 0.5 * nm_rank)
    total_width = 1.0 - margin_left - margin_right

    output_w = (output_rel / total_width_rel) * total_width
    rec_e_w = (rec_e_rel / total_width_rel) * total_width
    rec_i_w = (rec_i_rel / total_width_rel) * total_width
    gap_w = (gap / total_width_rel) * total_width
    inner_gap_w = (inner_gap_rel / total_width_rel) * total_width
    nm_gap_w = (nm_gap_rel / total_width_rel) * total_width

    output_left = margin_left
    rec_e_left = output_left + output_w + gap_w
    rec_i_left = rec_e_left + rec_e_w + inner_gap_w

    # Vertical layout: split recurrent height into E-target and I-target rows
    rec_e_h_rel = rec_height_rel * (n_exc / hidden_size)
    rec_i_h_rel = rec_height_rel * (n_inh / hidden_size)
    total_height_rel = (
        rec_e_h_rel + inner_gap_rel + rec_i_h_rel + gap + input_height_rel
    )
    rec_e_h = (rec_e_h_rel / total_height_rel) * total_height
    rec_i_h = (rec_i_h_rel / total_height_rel) * total_height
    inner_gap_h = (inner_gap_rel / total_height_rel) * total_height
    input_h = (input_height_rel / total_height_rel) * total_height
    gap_h = (gap / total_height_rel) * total_height

    # Vertical positions (bottom to top)
    input_bottom = margin_bottom
    rec_i_bottom = input_bottom + input_h + gap_h
    rec_e_bottom = rec_i_bottom + rec_i_h + inner_gap_h

    fig_height = fig_width * (total_height_rel / total_width_rel) * 1.1
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Split recurrent matrix into 4 quadrants
    # W_rec[i, j] = connection from j to i
    # Rows = "to" neuron, Cols = "from" neuron
    W_ee = W_rec[:n_exc, :n_exc]  # from E to E
    W_ie = W_rec[:n_exc, n_exc:]  # from I to E
    W_ei = W_rec[n_exc:, :n_exc]  # from E to I
    W_ii = W_rec[n_exc:, n_exc:]  # from I to I

    # Split output weights: W_out has shape (1, hidden)
    W_out_e = W_out[:, :n_exc]  # reading from E neurons
    W_out_i = W_out[:, n_exc:]  # reading from I neurons

    # --- Top row: "to excitatory" ---
    ax_out_e = fig.add_axes((output_left, rec_e_bottom, output_w, rec_e_h))
    _plot_heatmap(
        ax_out_e, W_out_e.T, vmin, vmax, ylabel="To E", title="Output\nWeights"
    )

    ax_ee = fig.add_axes((rec_e_left, rec_e_bottom, rec_e_w, rec_e_h))
    im_rec = _plot_heatmap(ax_ee, W_ee, vmin, vmax, title="From Excitatory")

    ax_ie = fig.add_axes((rec_i_left, rec_e_bottom, rec_i_w, rec_e_h))
    _plot_heatmap(ax_ie, W_ie, vmin, vmax, title="From Inhibitory")

    # --- Bottom row: "to inhibitory" ---
    ax_out_i = fig.add_axes((output_left, rec_i_bottom, output_w, rec_i_h))
    _plot_heatmap(ax_out_i, W_out_i.T, vmin, vmax, ylabel="To I")

    ax_ei = fig.add_axes((rec_e_left, rec_i_bottom, rec_e_w, rec_i_h))
    _plot_heatmap(ax_ei, W_ei, vmin, vmax)

    ax_ii = fig.add_axes((rec_i_left, rec_i_bottom, rec_i_w, rec_i_h))
    _plot_heatmap(ax_ii, W_ii, vmin, vmax)

    # --- Input weights (below bottom row) ---
    if n_input_e > 0:
        W_in_e = W_in[0:n_input_e, :].T
        input_e_w = rec_e_w * (n_input_e / n_exc)
        ax_in_e = fig.add_axes((rec_e_left, input_bottom, input_e_w, input_h))
        _plot_heatmap(
            ax_in_e, W_in_e, vmin, vmax, xlabel="Excitatory", ylabel="Input Weights"
        )
        ax_in_e.set_yticks(range(input_size))
        ax_in_e.set_yticklabels(input_labels, fontsize=8)

    if n_input_i > 0:
        W_in_i = W_in[n_exc : n_exc + n_input_i, :].T
        input_i_w = rec_i_w * (n_input_i / n_inh)
        ax_in_i = fig.add_axes((rec_i_left, input_bottom, input_i_w, input_h))
        _plot_heatmap(ax_in_i, W_in_i, vmin, vmax, xlabel="Inhibitory")

    # Colorbar for recurrent weights (right of recurrent block)
    cbar_left = rec_i_left + rec_i_w + 0.01
    cbar_ax = fig.add_axes(
        (cbar_left, rec_i_bottom, 0.02, rec_e_h + inner_gap_h + rec_i_h)
    )
    cbar = fig.colorbar(im_rec, cax=cbar_ax)
    cbar.set_label("Weight value", fontsize=10)

    # Label for recurrent weights (centered below the block)
    rec_center_x = (rec_e_left + rec_i_left + rec_i_w) / 2
    label_y = rec_i_bottom - 0.03
    fig.text(rec_center_x, label_y, "Recurrent Weights", ha="center", va="top", fontsize=11)

    # --- Receptor ratio (if two modulators present) ---
    if gain_masks is not None and len(gain_masks) >= 2:
        # Compute log ratio: log10(G1/G2) = log10(G1) - log10(G2)
        ratio_mask = gain_masks[0] - gain_masks[1]

        # Split ratio into quadrants
        R_ee = ratio_mask[:n_exc, :n_exc]
        R_ie = ratio_mask[:n_exc, n_exc:]
        R_ei = ratio_mask[n_exc:, :n_exc]
        R_ii = ratio_mask[n_exc:, n_exc:]

        # Position for ratio block (after colorbar + larger gap)
        mod_base_left = cbar_left + 0.06 + nm_gap_w
        mod_e_left = mod_base_left
        mod_i_left = mod_e_left + rec_e_w + inner_gap_w

        # Top row (to E)
        ax_ree = fig.add_axes((mod_e_left, rec_e_bottom, rec_e_w, rec_e_h))
        _plot_heatmap(ax_ree, R_ee, None, None, cmap="viridis", title="From E")

        ax_rie = fig.add_axes((mod_i_left, rec_e_bottom, rec_i_w, rec_e_h))
        _plot_heatmap(ax_rie, R_ie, None, None, cmap="viridis", title="From I")

        # Bottom row (to I)
        ax_rei = fig.add_axes((mod_e_left, rec_i_bottom, rec_e_w, rec_i_h))
        _plot_heatmap(ax_rei, R_ei, None, None, cmap="viridis")

        ax_rii = fig.add_axes((mod_i_left, rec_i_bottom, rec_i_w, rec_i_h))
        im_ratio = _plot_heatmap(ax_rii, R_ii, None, None, cmap="viridis")

        # Label for receptor ratio (centered below the block)
        mod_center_x = (mod_e_left + mod_i_left + rec_i_w) / 2
        fig.text(mod_center_x, label_y, "Receptor ratio", ha="center", va="top", fontsize=11)

        # Colorbar for ratio
        ratio_cbar_left = mod_i_left + rec_i_w + 0.01
        ratio_cbar_ax = fig.add_axes(
            (ratio_cbar_left, rec_i_bottom, 0.02, rec_e_h + inner_gap_h + rec_i_h)
        )
        ratio_cbar = fig.colorbar(im_ratio, cax=ratio_cbar_ax)
        ratio_cbar.set_label("log10(G1/G2)", fontsize=10)

    return fig


def _plot_standard_layout(
    W_in,
    W_rec,
    W_out,
    vmin,
    vmax,
    input_labels,
    hidden_size,
    input_size,
    n_input_neurons,
    method,
):
    """Create the standard (non-Dale) weight matrix layout.

    Layout: [Output] [gap] [Recurrent]
                     [Input]
    """
    margin_left = 0.08
    margin_right = 0.12
    margin_bottom = 0.08
    margin_top = 0.08
    gap = 0.10

    total_width = 1.0 - margin_left - margin_right
    total_height = 1.0 - margin_bottom - margin_top

    output_rel = 0.5
    input_height_rel = 1.5
    rec_height_rel = 8.0
    fig_width = 10.0

    total_height_rel = rec_height_rel + gap + input_height_rel
    rec_h = (rec_height_rel / total_height_rel) * total_height
    input_h = (input_height_rel / total_height_rel) * total_height
    gap_h = (gap / total_height_rel) * total_height

    fig_height = fig_width * (total_height_rel / (output_rel + gap + 8.0)) * 1.1

    rec_bottom = margin_bottom + input_h + gap_h
    input_bottom = margin_bottom

    rec_rel = 8.0
    total_width_rel = output_rel + gap + rec_rel
    output_w = (output_rel / total_width_rel) * total_width
    rec_w = (rec_rel / total_width_rel) * total_width
    gap_w = (gap / total_width_rel) * total_width
    input_w = rec_w * (n_input_neurons / hidden_size)

    output_left = margin_left
    rec_left = output_left + output_w + gap_w

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Output weights
    ax_out = fig.add_axes((output_left, rec_bottom, output_w, rec_h))
    _plot_heatmap(
        ax_out,
        W_out.T,
        vmin,
        vmax,
        ylabel="To neuron (clustered)",
        title="Output\nWeights",
    )

    # Recurrent weights
    ax_rec = fig.add_axes((rec_left, rec_bottom, rec_w, rec_h))
    im_rec = _plot_heatmap(
        ax_rec, W_rec, vmin, vmax, title=f"Recurrent Weights (clustered: {method})"
    )

    # Input weights
    ax_in = fig.add_axes((rec_left, input_bottom, input_w, input_h))
    W_in_display = W_in[:n_input_neurons, :].T
    _plot_heatmap(
        ax_in,
        W_in_display,
        vmin,
        vmax,
        xlabel="To neuron (clustered)",
        ylabel="Input Weights",
    )
    ax_in.set_yticks(range(input_size))
    ax_in.set_yticklabels(input_labels, fontsize=8)

    # Colorbar
    cbar_ax = fig.add_axes((rec_left + rec_w + 0.01, rec_bottom, 0.02, rec_h))
    cbar = fig.colorbar(im_rec, cax=cbar_ax)
    cbar.set_label("Weight value", fontsize=10)

    plt.suptitle("Weight Matrices Visualization", fontsize=14)
    return fig


def plot_weight_matrices(
    model: RateRNN,
    method: str = "ward",
    save_path: str = "./plots/weight_matrices.png",
    show: bool = True,
):
    """
    Plot the three weight matrices (input, recurrent, output) as heatmaps.

    The recurrent weights are clustered, and the input/output matrices are
    permuted accordingly so edges align when displayed together.

    Args:
        model: Trained RateRNN model
        method: Clustering method - "ward" (hierarchical) or "spectral"
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    w = _extract_weights(model, method)
    vmin, vmax = _symmetric_clim(w["W_rec"], w["W_in"], w["W_out"])

    # Compute gain masks if model has neuromodulation
    gain_masks = None
    if w["has_neuromod"] and w["nm_rank"] > 0:
        gain_masks = _compute_gain_masks(w["U"], w["V"])

    # Input feature labels
    input_labels = ["fixation", "test_side", "go_cue", "hold_cue"] + [
        f"chk{i}" for i in range(w["input_size"] - 4)
    ]

    if w["dale_active"]:
        fig = _plot_dale_layout(
            w["W_in"],
            w["W_rec"],
            w["W_out"],
            vmin,
            vmax,
            input_labels,
            w["hidden_size"],
            w["input_size"],
            w["n_exc"],
            w["n_input_e"],
            w["n_input_i"],
            gain_masks=gain_masks,
        )
    else:
        fig = _plot_standard_layout(
            w["W_in"],
            w["W_rec"],
            w["W_out"],
            vmin,
            vmax,
            input_labels,
            w["hidden_size"],
            w["input_size"],
            w["n_input_neurons"],
            w["method"],
        )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved weight matrices plot to {save_path}")

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    device = "cpu"
    checkpoint = torch.load("./checkpoints/best_model.pt", map_location=device)

    config = checkpoint["model_config"]
    config.pop("device", None)
    model = RateRNN(**config, device=device).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    if model.dale_ratio is not None:
        # Dale's law: no clustering, so one plot suffices
        plot_weight_matrices(model, save_path="./plots/weight_matrices.png", show=False)
    else:
        # Plot with both clustering methods
        plot_weight_matrices(
            model,
            method="ward",
            save_path="./plots/weight_matrices_ward.png",
            show=False,
        )
        plot_weight_matrices(
            model,
            method="spectral",
            save_path="./plots/weight_matrices_spectral.png",
            show=False,
        )
