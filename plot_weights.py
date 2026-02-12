"""Preview script for weight matrix visualization."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from model import RateRNN


def cluster_and_reorder(W_rec: np.ndarray, method: str = "ward") -> tuple[np.ndarray, np.ndarray]:
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
        # Spectral reordering for rows
        similarity_rows = W_rec @ W_rec.T  # Row similarity based on outgoing patterns
        degree = np.diag(np.abs(similarity_rows).sum(axis=1))
        laplacian = degree - similarity_rows
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        row_order = np.argsort(eigvecs[:, 1])

        # Spectral reordering for columns
        similarity_cols = W_rec.T @ W_rec  # Column similarity based on incoming patterns
        degree = np.diag(np.abs(similarity_cols).sum(axis=1))
        laplacian = degree - similarity_cols
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        col_order = np.argsort(eigvecs[:, 1])
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return row_order, col_order


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

    Layout:
        [Output] [    Recurrent     ]
                 [Input][   empty   ]

    The input matrix only spans the first n_input_neurons columns,
    corresponding to neurons that receive task input.

    Args:
        model: Trained RateRNN model
        method: Clustering method - "ward" (hierarchical) or "spectral"
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    # Extract weight matrices
    W_in = model.w_in.weight.detach().cpu().numpy()    # (hidden, input)
    W_rec = model.w_rec.weight.detach().cpu().numpy()  # (hidden, hidden)
    W_out = model.w_out.weight.detach().cpu().numpy()  # (1, hidden)

    hidden_size, input_size = W_in.shape

    # Determine Dale's law configuration
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
        n_input_neurons = 0  # handled by Dale split below

    # Cluster the recurrent matrix (non-Dale case only)
    if not dale_active:
        row_order, col_order = cluster_and_reorder(W_rec, method=method)
        W_rec = W_rec[np.ix_(row_order, col_order)]
        W_in = W_in[col_order, :]
        W_out = W_out[:, row_order]

    # Compute color scale limits (symmetric around 0)
    all_weights = np.concatenate([W_rec.flatten(),
                                   W_in.flatten(),
                                   W_out.flatten()])
    vmax = float(np.percentile(np.abs(all_weights), 99))
    vmin = -vmax

    # Input feature labels
    input_labels = ["fixation", "test_side", "go_cue", "hold_cue"] + [f"chk{i}" for i in range(input_size - 4)]

    # Create figure with manual axes positioning for precise alignment
    # All positions in normalized figure coordinates [left, bottom, width, height]
    margin_left = 0.08
    margin_right = 0.12  # space for colorbar
    margin_bottom = 0.08
    margin_top = 0.08
    gap = 0.10  # gap between subplots

    # Calculate available space
    total_width = 1.0 - margin_left - margin_right
    total_height = 1.0 - margin_bottom - margin_top

    # Relative sizes (in arbitrary units, will be normalized)
    output_rel = 0.5
    input_height_rel = 1.5
    rec_height_rel = 8.0

    # Normalize heights (same for both layouts)
    total_height_rel = rec_height_rel + gap + input_height_rel
    rec_h = (rec_height_rel / total_height_rel) * total_height
    input_h = (input_height_rel / total_height_rel) * total_height
    gap_h = (gap / total_height_rel) * total_height

    fig_width = 10.0
    fig_height = fig_width * (total_height_rel / (output_rel + gap + 8.0)) * 1.1

    rec_bottom = margin_bottom + input_h + gap_h
    input_bottom = margin_bottom

    if dale_active:
        # Layout: [Out_E] [gap] [E→E] [gap] [I→E]
        #         [gap]         [gap]        [gap]
        #         [Out_I] [gap] [E→I] [gap] [I→I]
        #                       [E_in]      [I_in]
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
        total_height_rel = rec_e_h_rel + inner_gap_rel + rec_i_h_rel + gap + input_height_rel
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
        W_ee = W_rec[:n_exc, :n_exc]       # from E to E
        W_ie = W_rec[:n_exc, n_exc:]       # from I to E
        W_ei = W_rec[n_exc:, :n_exc]       # from E to I
        W_ii = W_rec[n_exc:, n_exc:]       # from I to I

        # Split output weights: W_out has shape (1, hidden)
        W_out_e = W_out[:, :n_exc]         # reading from E neurons
        W_out_i = W_out[:, n_exc:]         # reading from I neurons

        # --- Top row: "to excitatory" ---
        # Output weights (E targets)
        ax_out_e = fig.add_axes((output_left, rec_e_bottom, output_w, rec_e_h))
        ax_out_e.imshow(W_out_e.T, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax_out_e.set_ylabel("To E", fontsize=10)
        ax_out_e.set_title("Output\nWeights", fontsize=10)
        ax_out_e.set_xticks([])
        ax_out_e.set_yticks([])

        # E→E
        ax_ee = fig.add_axes((rec_e_left, rec_e_bottom, rec_e_w, rec_e_h))
        im_rec = ax_ee.imshow(W_ee, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax_ee.set_title("From Excitatory", fontsize=11)
        ax_ee.set_xticks([])
        ax_ee.set_yticks([])

        # I→E
        ax_ie = fig.add_axes((rec_i_left, rec_e_bottom, rec_i_w, rec_e_h))
        ax_ie.imshow(W_ie, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax_ie.set_title("From Inhibitory", fontsize=11)
        ax_ie.set_xticks([])
        ax_ie.set_yticks([])

        # --- Bottom row: "to inhibitory" ---
        # Output weights (I targets)
        ax_out_i = fig.add_axes((output_left, rec_i_bottom, output_w, rec_i_h))
        ax_out_i.imshow(W_out_i.T, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax_out_i.set_ylabel("To I", fontsize=10)
        ax_out_i.set_xticks([])
        ax_out_i.set_yticks([])

        # E→I
        ax_ei = fig.add_axes((rec_e_left, rec_i_bottom, rec_e_w, rec_i_h))
        ax_ei.imshow(W_ei, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax_ei.set_xticks([])
        ax_ei.set_yticks([])

        # I→I
        ax_ii = fig.add_axes((rec_i_left, rec_i_bottom, rec_i_w, rec_i_h))
        ax_ii.imshow(W_ii, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax_ii.set_xticks([])
        ax_ii.set_yticks([])

        # --- Input weights (below bottom row) ---
        if n_input_e > 0:
            W_in_e = W_in[0:n_input_e, :].T
            input_e_w = rec_e_w * (n_input_e / n_exc)
            ax_in_e = fig.add_axes((rec_e_left, input_bottom, input_e_w, input_h))
            ax_in_e.imshow(W_in_e, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax_in_e.set_xlabel("Excitatory", fontsize=9)
            ax_in_e.set_xticks([])
            ax_in_e.set_yticks(range(input_size))
            ax_in_e.set_yticklabels(input_labels, fontsize=8)
            ax_in_e.set_ylabel("Input Weights", fontsize=10)

        if n_input_i > 0:
            W_in_i = W_in[n_exc:n_exc+n_input_i, :].T
            input_i_w = rec_i_w * (n_input_i / n_inh)
            ax_in_i = fig.add_axes((rec_i_left, input_bottom, input_i_w, input_h))
            ax_in_i.imshow(W_in_i, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax_in_i.set_xlabel("Inhibitory", fontsize=9)
            ax_in_i.set_xticks([])
            ax_in_i.set_yticks([])

        # Colorbar (right of top-right block)
        cbar_ax = fig.add_axes((rec_i_left + rec_i_w + 0.01, rec_i_bottom, 0.02, rec_e_h + inner_gap_h + rec_i_h))
        cbar = fig.colorbar(im_rec, cax=cbar_ax)
        cbar.set_label("Weight value", fontsize=10)

        plt.suptitle("Weight Matrices", fontsize=14)

    else:
        # Non-Dale layout: [Output] [gap] [Recurrent]
        #                            [Input]
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
        ax_out.imshow(W_out.T, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax_out.set_ylabel("To neuron (clustered)", fontsize=10)
        ax_out.set_title("Output\nWeights", fontsize=10)
        ax_out.set_xticks([])
        ax_out.set_yticks([])

        # Recurrent weights
        ax_rec = fig.add_axes((rec_left, rec_bottom, rec_w, rec_h))
        im_rec = ax_rec.imshow(W_rec, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax_rec.set_title(f"Recurrent Weights (clustered: {method})", fontsize=12)
        ax_rec.set_xticks([])
        ax_rec.set_yticks([])

        # Input weights
        ax_in = fig.add_axes((rec_left, input_bottom, input_w, input_h))
        W_in_display = W_in[:n_input_neurons, :].T
        ax_in.imshow(W_in_display, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax_in.set_xlabel("To neuron (clustered)", fontsize=10)
        ax_in.set_xticks([])
        ax_in.set_yticks(range(input_size))
        ax_in.set_yticklabels(input_labels, fontsize=8)
        ax_in.set_ylabel("Input Weights", fontsize=10)

        # Colorbar
        cbar_ax = fig.add_axes((rec_left + rec_w + 0.01, rec_bottom, 0.02, rec_h))
        cbar = fig.colorbar(im_rec, cax=cbar_ax)
        cbar.set_label("Weight value", fontsize=10)

        plt.suptitle("Weight Matrices Visualization", fontsize=14)

    # Ensure directory exists
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved weight matrices plot to {save_path}")

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    device = "cpu"
    n_checkerboard_channels = 10
    input_size = 4 + n_checkerboard_channels

    # Create model with same architecture
    model = RateRNN(
        input_size=input_size,
        hidden_size=256,
        dt=20.0,
        tau=100.0,
        activation="relu",
        noise_std=0.01,
        dale_ratio=0.8,
        n_input_e=32,
        n_input_i=32,
        alpha=1e-5,
        device=device,
    ).to(device)

    # Load best model weights (strict=False to allow input_mask mismatch)
    checkpoint = torch.load("./checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Plot with both clustering methods
    plot_weight_matrices(model, method="ward", save_path="./plots/weight_matrices_ward.png", show=False)
    plot_weight_matrices(model, method="spectral", save_path="./plots/weight_matrices_spectral.png", show=False)
