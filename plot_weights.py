"""Preview script for weight matrix visualization."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
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
    output_size = W_out.shape[0]

    # Determine how many neurons receive input
    if model.input_fraction is not None:
        n_input_neurons = int(model.input_fraction * hidden_size)
    else:
        n_input_neurons = hidden_size

    # Cluster the recurrent matrix - separate orderings for rows and columns
    row_order, col_order = cluster_and_reorder(W_rec, method=method)

    # Reorder matrices according to clustering
    # Recurrent: rows reordered by row_order, cols reordered by col_order
    W_rec_ordered = W_rec[np.ix_(row_order, col_order)]
    # Input weights: rows of W_in are hidden neurons, cols are input features
    # Permute rows by col_order so input matrix columns align with recurrent matrix columns
    W_in_ordered = W_in[col_order, :]
    # Output weights: rows of W_out read from hidden neurons
    # So permute cols of W_out by row_order (to align with recurrent rows when transposed left)
    W_out_ordered = W_out[:, row_order]

    # Compute color scale limits (symmetric around 0)
    all_weights = np.concatenate([W_rec_ordered.flatten(),
                                   W_in_ordered.flatten(),
                                   W_out_ordered.flatten()])
    vmax = np.percentile(np.abs(all_weights), 99)
    vmin = -vmax

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
    rec_rel = 8.0
    input_height_rel = 1.5
    rec_height_rel = 8.0

    # Normalize widths
    total_width_rel = output_rel + gap + rec_rel
    output_w = (output_rel / total_width_rel) * total_width
    rec_w = (rec_rel / total_width_rel) * total_width
    gap_w = (gap / total_width_rel) * total_width

    # Normalize heights
    total_height_rel = rec_height_rel + gap + input_height_rel
    rec_h = (rec_height_rel / total_height_rel) * total_height
    input_h = (input_height_rel / total_height_rel) * total_height
    gap_h = (gap / total_height_rel) * total_height

    # Input width is fraction of recurrent width
    input_w = rec_w * (n_input_neurons / hidden_size)

    # Calculate positions
    output_left = margin_left
    rec_left = output_left + output_w + gap_w
    rec_bottom = margin_bottom + input_h + gap_h
    input_bottom = margin_bottom

    fig_width = 10.0
    fig_height = fig_width * (total_height_rel / total_width_rel) * 1.1
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Output weights (left of recurrent)
    ax_out = fig.add_axes([output_left, rec_bottom, output_w, rec_h])
    im_out = ax_out.imshow(
        W_out_ordered.T,  # (hidden, 1)
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax_out.set_ylabel("To neuron (clustered)", fontsize=10)
    ax_out.set_title("Output\nWeights", fontsize=10)
    ax_out.set_xticks([])
    ax_out.set_yticks([])

    # Recurrent weights
    ax_rec = fig.add_axes([rec_left, rec_bottom, rec_w, rec_h])
    im_rec = ax_rec.imshow(
        W_rec_ordered,
        aspect="auto",  # use auto to fill axes exactly
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax_rec.set_title(f"Recurrent Weights (clustered: {method})", fontsize=12)
    ax_rec.set_xticks([])
    ax_rec.set_yticks([])

    # Input weights (below left portion of recurrent) - aligned with rec_left
    ax_in = fig.add_axes([rec_left, input_bottom, input_w, input_h])
    W_in_display = W_in_ordered[:n_input_neurons, :].T  # (input_size, n_input_neurons)
    im_in = ax_in.imshow(
        W_in_display,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax_in.set_xlabel("To neuron (clustered)", fontsize=10)
    ax_in.set_xticks([])
    # Label input features on y-axis
    input_labels = ["fixation", "test_side", "go_cue", "hold_cue"] + [f"chk{i}" for i in range(input_size - 4)]
    ax_in.set_yticks(range(input_size))
    ax_in.set_yticklabels(input_labels, fontsize=8)
    ax_in.set_ylabel("Input Weights", fontsize=10)

    # Add colorbar (positioned relative to recurrent matrix)
    cbar_ax = fig.add_axes([rec_left + rec_w + 0.01, rec_bottom, 0.02, rec_h])
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
        dale_ratio=None,
        input_fraction=0.125,  # 1/8 of neurons receive task input
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
