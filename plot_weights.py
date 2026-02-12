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


def _plot_heatmap(ax, data, vmin, vmax, **kwargs):
    """Plot a weight matrix heatmap with common defaults."""
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
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
):
    """Create the Dale's law weight matrix layout.

    Layout: [Out_E] [gap] [E->E] [gap] [I->E]
            [gap]         [gap]        [gap]
            [Out_I] [gap] [E->I] [gap] [I->I]
                          [E_in]       [I_in]
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

    n_inh = hidden_size - n_exc
    rec_e_rel = 8.0 * (n_exc / hidden_size)
    rec_i_rel = 8.0 * (n_inh / hidden_size)
    inner_gap_rel = gap * 0.5

    # Horizontal layout
    total_width_rel = output_rel + gap + rec_e_rel + inner_gap_rel + rec_i_rel
    output_w = (output_rel / total_width_rel) * total_width
    rec_e_w = (rec_e_rel / total_width_rel) * total_width
    rec_i_w = (rec_i_rel / total_width_rel) * total_width
    gap_w = (gap / total_width_rel) * total_width
    inner_gap_w = (inner_gap_rel / total_width_rel) * total_width

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

    # Colorbar (right of top-right block)
    cbar_ax = fig.add_axes(
        (
            rec_i_left + rec_i_w + 0.01,
            rec_i_bottom,
            0.02,
            rec_e_h + inner_gap_h + rec_i_h,
        )
    )
    cbar = fig.colorbar(im_rec, cax=cbar_ax)
    cbar.set_label("Weight value", fontsize=10)

    plt.suptitle("Weight Matrices", fontsize=14)
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
