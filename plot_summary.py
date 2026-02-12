"""
Summary plot combining training metrics and neural analysis for neuromodulated RNN.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from checkerboard import (
    DelayedMatchToEvidenceDataset,
    create_dataloaders,
)
from neuromod_model import NeuromodRNN


def plot_summary(
    checkpoint_path: str = "./checkpoints_neuromod/best_model.pt",
    save_path: str = "./plots/summary.pdf",
    n_trials: int = 20,
    show: bool = True,
):
    """
    Generate a 2x2 summary figure with training metrics and neural analysis.

    Args:
        checkpoint_path: Path to model checkpoint
        save_path: Where to save the figure
        n_trials: Number of trials to plot in readout/modulator panels
        show: Whether to display the plot
    """
    device = "cpu"

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    history = checkpoint["history"]
    config = checkpoint["model_config"]

    # Load model
    config.pop("device", None)
    model = NeuromodRNN(**config, device=device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # Create validation dataset with same parameters as training
    dataset_kwargs = checkpoint.get("dataset_kwargs", {})
    _, val_loader, _ = create_dataloaders(
        train_trials=100,  # minimal, not used
        val_trials=200,
        test_trials=100,
        batch_size=32,
        **dataset_kwargs,
    )
    val_dataset = val_loader.dataset
    assert isinstance(val_dataset, DelayedMatchToEvidenceDataset)

    # Collect model outputs on validation set
    all_outputs = []
    all_targets = []
    all_s_traj = []
    all_lengths = []
    all_infos = []

    with torch.no_grad():
        for batch_inputs, batch_targets, batch_lengths in val_loader:
            batch_inputs = batch_inputs.to(device)
            outputs, _, _states, _z_traj, s_traj = model(
                batch_inputs, return_states=True, return_neuromod=True
            )
            outputs = outputs.cpu().numpy()
            s_traj = s_traj.cpu().numpy()
            batch_targets = batch_targets.numpy()
            batch_lengths = batch_lengths.numpy()

            for i in range(outputs.shape[0]):
                seq_len = int(batch_lengths[i])
                all_outputs.append(outputs[i, :seq_len])
                all_targets.append(batch_targets[i, :seq_len])
                all_s_traj.append(s_traj[i, :seq_len])
                all_lengths.append(seq_len)

    for idx in range(len(val_dataset)):
        all_infos.append(val_dataset.get_trial_info(idx))

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Panel [0,0]: Task Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_task_loss"], label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Task Loss (MSE)")
    ax.set_title("Task Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel [0,1]: Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["train_accuracy"], label="Train", linewidth=2)
    ax.plot(epochs, history["val_accuracy"], label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Select trials for bottom panels
    trial_indices = np.random.choice(
        len(all_outputs), size=min(n_trials, len(all_outputs)), replace=False
    )

    # Panel [1,0]: Model Readouts
    ax = axes[1, 0]
    for ti in trial_indices:
        t_ax = np.arange(all_lengths[ti]) * val_dataset.time_bin_size
        target_sign = np.sign(all_targets[ti][-1])
        color = "tab:blue" if target_sign < 0 else "tab:red"
        linestyle = "-" if all_infos[ti]["empirical_predom_color"] == "black" else "--"
        ax.plot(t_ax, all_outputs[ti], color=color, ls=linestyle, lw=1, alpha=0.6)
    ax.plot([], [], color="tab:blue", lw=2, label="Target = Left")
    ax.plot([], [], color="tab:red", lw=2, label="Target = Right")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Model output")
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"Model readouts ({len(trial_indices)} trials)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel [1,1]: Neuromodulator Dynamics
    ax = axes[1, 1]
    for ti in trial_indices:
        t_ax = np.arange(all_lengths[ti]) * val_dataset.time_bin_size
        target_sign = np.sign(all_targets[ti][-1])
        color = "tab:blue" if target_sign < 0 else "tab:red"
        linestyle = "-" if all_infos[ti]["empirical_predom_color"] == "black" else "--"
        s_ti = all_s_traj[ti]
        ax.plot(t_ax, s_ti[:, 0], color=color, ls=linestyle, lw=1, alpha=0.6)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Modulator value")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Neuromodulator dynamics")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Add panel labels
    panel_labels = [("A", axes[0, 0]), ("B", axes[0, 1]), ("C", axes[1, 0]), ("D", axes[1, 1])]
    for label, ax in panel_labels:
        ax.text(
            -0.12, 1.08, label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            fontfamily="Arial",
            va="top",
            ha="left",
        )

    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


if __name__ == "__main__":
    plot_summary(show=False)
