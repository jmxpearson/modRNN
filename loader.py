import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, Any


class AccumulationTaskDataset(Dataset):
    """
    PyTorch Dataset for the accumulation-of-evidence task from Pinto et al. (2018).
    
    Mice navigate a virtual T-maze and accumulate visual evidence (towers) 
    appearing on left/right sides to make a decision.
    
    Task structure:
    - Cue period: 200 cm where towers appear as brief pulses
    - Delay period: 100 cm with no cues
    - Towers appear with Poisson statistics (7.7/m on rewarded side, 2.3/m on other)
    - Each tower is visible for 200ms
    - Refractory period: 12 cm between towers
    """
    
    def __init__(
        self,
        n_trials: int = 1000,
        cue_period_length: int = 200,  # cm
        delay_period_length: int = 100,  # cm
        time_bin_size: float = 0.1,  # seconds
        avg_speed: float = 60.0,  # cm/s
        tower_duration: float = 0.2,  # seconds
        rewarded_rate: float = 7.7,  # towers per meter
        minority_rate: float = 2.3,  # towers per meter
        refractory_period: float = 12.0,  # cm
        seed: Optional[int] = None
    ):
        """
        Args:
            n_trials: Number of trials to generate
            cue_period_length: Length of cue region in cm
            delay_period_length: Length of delay region in cm
            time_bin_size: Time resolution for neural data (seconds)
            avg_speed: Average running speed in cm/s
            tower_duration: Duration each tower is visible (seconds)
            rewarded_rate: Tower rate on rewarded side (per meter)
            minority_rate: Tower rate on minority side (per meter)
            refractory_period: Minimum distance between towers (cm)
            seed: Random seed for reproducibility
        """
        self.n_trials = n_trials
        self.cue_period_length = cue_period_length
        self.delay_period_length = delay_period_length
        self.total_length = cue_period_length + delay_period_length
        self.time_bin_size = time_bin_size
        self.avg_speed = avg_speed
        self.tower_duration = tower_duration
        self.rewarded_rate = rewarded_rate / 100  # convert to per cm
        self.minority_rate = minority_rate / 100  # convert to per cm
        self.refractory_period = refractory_period
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate all trials
        self.trials = self._generate_trials()
    
    def _generate_poisson_towers(
        self, 
        rate: float, 
        length: float
    ) -> np.ndarray:
        """Generate tower positions using Poisson process with refractory period."""
        positions = []
        current_pos = 10.0  # First tower can appear at 10 cm
        
        while current_pos < length:
            # Draw inter-tower interval from exponential distribution
            interval = np.random.exponential(1.0 / rate)
            # Enforce refractory period
            interval = max(interval, self.refractory_period)
            current_pos += interval
            
            if current_pos < length:
                positions.append(current_pos)
        
        return np.array(positions)
    
    def _generate_single_trial(self) -> Dict[str, Any]:
        """Generate a single trial with towers and task structure."""
        # Randomly choose rewarded side (0=left, 1=right)
        rewarded_side = np.random.randint(0, 2)
        
        # Generate tower positions for each side in cue period
        left_towers = self._generate_poisson_towers(
            self.rewarded_rate if rewarded_side == 0 else self.minority_rate,
            self.cue_period_length
        )
        right_towers = self._generate_poisson_towers(
            self.rewarded_rate if rewarded_side == 1 else self.minority_rate,
            self.cue_period_length
        )
        
        # Calculate trial duration in time
        trial_duration = self.total_length / self.avg_speed
        n_timesteps = int(trial_duration / self.time_bin_size)
        
        # Initialize input arrays (time x features)
        left_cues = np.zeros(n_timesteps)
        right_cues = np.zeros(n_timesteps)
        hold_cue = np.zeros(n_timesteps)
        side_cue = np.zeros(n_timesteps)
        target_output = np.zeros(n_timesteps)
        
        # Convert spatial positions to time indices
        for pos in left_towers:
            time_idx = int((pos / self.avg_speed) / self.time_bin_size)
            duration_bins = int(self.tower_duration / self.time_bin_size)
            end_idx = min(time_idx + duration_bins, n_timesteps)
            left_cues[time_idx:end_idx] = 1.0
        
        for pos in right_towers:
            time_idx = int((pos / self.avg_speed) / self.time_bin_size)
            duration_bins = int(self.tower_duration / self.time_bin_size)
            end_idx = min(time_idx + duration_bins, n_timesteps)
            right_cues[time_idx:end_idx] = 1.0
        
        # Hold cue active during delay period
        delay_start_time = self.cue_period_length / self.avg_speed
        delay_start_idx = int(delay_start_time / self.time_bin_size)
        hold_cue[delay_start_idx:] = 1.0
        
        # Side cue indicates rewarded side (constant throughout trial)
        side_cue[:] = rewarded_side
        
        # Target output: 0 during cue period, -1 (left) or +1 (right) during delay
        if rewarded_side == 0:  # Left rewarded
            target_output[delay_start_idx:] = -1.0
        else:  # Right rewarded
            target_output[delay_start_idx:] = 1.0
        
        return {
            'left_cues': left_cues,
            'right_cues': right_cues,
            'hold_cue': hold_cue,
            'side_cue': side_cue,
            'target_output': target_output,
            'rewarded_side': rewarded_side,
            'n_left': len(left_towers),
            'n_right': len(right_towers),
            'delta': len(right_towers) - len(left_towers)
        }
    
    def _generate_trials(self):
        """Generate all trials for the dataset."""
        return [self._generate_single_trial() for _ in range(self.n_trials)]
    
    def __len__(self) -> int:
        return self.n_trials
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs: Tensor of shape (time, 4) containing:
                - left_cues (binary)
                - right_cues (binary)
                - hold_cue (binary, active during delay)
                - side_cue (binary, indicates correct side)
            target: Tensor of shape (time,) containing desired output:
                - 0 during cue period
                - -1 (left rewarded) or +1 (right rewarded) during delay period
        """
        trial = self.trials[idx]
        
        # Stack all input features
        inputs = np.stack([
            trial['left_cues'],
            trial['right_cues'],
            trial['hold_cue'],
            trial['side_cue']
        ], axis=1)
        
        inputs = torch.FloatTensor(inputs)
        target = torch.FloatTensor(trial['target_output'])
        
        return inputs, target
    
    def get_trial_info(self, idx: int) -> Dict[str, int]:
        """Get detailed information about a specific trial."""
        trial = self.trials[idx]
        return {
            'n_left': trial['n_left'],
            'n_right': trial['n_right'],
            'delta': trial['delta'],
            'rewarded_side': trial['rewarded_side'],
            'difficulty': abs(trial['delta'])
        }


def create_dataloaders(
    train_trials: int = 800,
    val_trials: int = 100,
    test_trials: int = 100,
    batch_size: int = 32,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_trials: Number of training trials
        val_trials: Number of validation trials
        test_trials: Number of test trials
        batch_size: Batch size for dataloaders
        **dataset_kwargs: Additional arguments passed to AccumulationTaskDataset
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = AccumulationTaskDataset(
        n_trials=train_trials, 
        seed=0, 
        **dataset_kwargs
    )
    val_dataset = AccumulationTaskDataset(
        n_trials=val_trials, 
        seed=1, 
        **dataset_kwargs
    )
    test_dataset = AccumulationTaskDataset(
        n_trials=test_trials, 
        seed=2, 
        **dataset_kwargs
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def plot_trials(
    dataset: AccumulationTaskDataset,
    trial_indices: list = [0, 1, 2, 3],
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot multiple trials as heatmaps with time on x-axis and variables on y-axis.
    
    Args:
        dataset: AccumulationTaskDataset instance
        trial_indices: List of trial indices to plot
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    n_trials = len(trial_indices)
    fig, axes = plt.subplots(n_trials, 1, figsize=figsize)
    
    if n_trials == 1:
        axes = [axes]
    
    feature_names = ['Left Cues', 'Right Cues', 'Hold Cue', 'Side Cue', 'Target Output']
    
    for ax_idx, trial_idx in enumerate(trial_indices):
        inputs, target = dataset[trial_idx]
        info = dataset.get_trial_info(trial_idx)
        
        # Combine inputs and target for plotting
        # Transpose to get (features, time) for plotting
        inputs_np = inputs.numpy().T
        target_np = target.numpy()[np.newaxis, :]  # Add feature dimension
        data = np.vstack([inputs_np, target_np])
        
        # Create heatmap
        im = axes[ax_idx].imshow(
            data,
            aspect='auto',
            cmap='RdYlGn',
            interpolation='nearest',
            vmin=-1,
            vmax=1
        )
        
        # Set y-axis labels
        axes[ax_idx].set_yticks(range(len(feature_names)))
        axes[ax_idx].set_yticklabels(feature_names)
        
        # Add vertical line to indicate cue/delay boundary
        cue_duration = dataset.cue_period_length / dataset.avg_speed
        delay_start_idx = int(cue_duration / dataset.time_bin_size)
        axes[ax_idx].axvline(
            x=delay_start_idx, 
            color='blue', 
            linestyle='--', 
            linewidth=2, 
            alpha=0.7,
            label='Delay Start'
        )
        
        # Add title with trial information
        rewarded_side_str = 'Right' if info['rewarded_side'] == 1 else 'Left'
        title = (f"Trial {trial_idx}: Rewarded={rewarded_side_str} | "
                f"#L={info['n_left']}, #R={info['n_right']}, "
                f"Δ={info['delta']:+d}")
        axes[ax_idx].set_title(title, fontsize=11, pad=10)
        
        # Add x-axis label only for bottom plot
        if ax_idx == n_trials - 1:
            axes[ax_idx].set_xlabel('Time (bins)', fontsize=11)
        else:
            axes[ax_idx].set_xticks([])
        
        # Add legend for first plot
        if ax_idx == 0:
            axes[ax_idx].legend(loc='upper right')
    
    plt.tight_layout()
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes((0.94, 0.15, 0.02, 0.7))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Activity', rotation=270, labelpad=20)
    
    return fig


def plot_trial_detailed(
    dataset: AccumulationTaskDataset,
    trial_idx: int = 0,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot a single trial with separate subplots for each variable.
    
    Args:
        dataset: AccumulationTaskDataset instance
        trial_idx: Index of trial to plot
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    
    inputs, target = dataset[trial_idx]
    info = dataset.get_trial_info(trial_idx)
    inputs_np = inputs.numpy()
    target_np = target.numpy()
    
    # Create time axis in seconds
    time_axis = np.arange(len(inputs_np)) * dataset.time_bin_size
    
    # Calculate cue/delay boundary
    cue_duration = dataset.cue_period_length / dataset.avg_speed
    
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
    
    feature_names = ['Left Cues', 'Right Cues', 'Hold Cue', 'Side Cue', 'Target Output']
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    
    # Plot first 4 input features
    for i, (name, color) in enumerate(zip(feature_names[:4], colors[:4])):
        axes[i].plot(time_axis, inputs_np[:, i], color=color, linewidth=2)
        axes[i].fill_between(
            time_axis, 
            0, 
            inputs_np[:, i], 
            color=color, 
            alpha=0.3
        )
        axes[i].set_ylabel(name, fontsize=11, fontweight='bold')
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].grid(True, alpha=0.3)
        
        # Add vertical line for delay start
        axes[i].axvline(
            x=cue_duration, 
            color='black', 
            linestyle='--', 
            linewidth=1.5, 
            alpha=0.5
        )
        
        # Shade cue and delay regions
        axes[i].axvspan(0, cue_duration, alpha=0.1, color='yellow', label='Cue Period')
        axes[i].axvspan(
            cue_duration, 
            time_axis[-1], 
            alpha=0.1, 
            color='cyan', 
            label='Delay Period'
        )
        
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=9)
    
    # Plot target output
    axes[4].plot(time_axis, target_np, color=colors[4], linewidth=2)
    axes[4].fill_between(
        time_axis, 
        0, 
        target_np, 
        color=colors[4], 
        alpha=0.3
    )
    axes[4].set_ylabel(feature_names[4], fontsize=11, fontweight='bold')
    axes[4].set_ylim(-1.2, 1.2)
    axes[4].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axes[4].grid(True, alpha=0.3)
    
    # Add vertical line for delay start
    axes[4].axvline(
        x=cue_duration, 
        color='black', 
        linestyle='--', 
        linewidth=1.5, 
        alpha=0.5
    )
    
    # Shade cue and delay regions
    axes[4].axvspan(0, cue_duration, alpha=0.1, color='yellow')
    axes[4].axvspan(cue_duration, time_axis[-1], alpha=0.1, color='cyan')
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    
    rewarded_side_str = 'Right' if info['rewarded_side'] == 1 else 'Left'
    fig.suptitle(
        f"Trial {trial_idx}: Rewarded Side = {rewarded_side_str} | "
        f"Left Cues = {info['n_left']}, Right Cues = {info['n_right']}, "
        f"Δ = {info['delta']:+d}",
        fontsize=13,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    return fig


def plot_psychometric_curve(
    dataset: AccumulationTaskDataset,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot psychometric curve showing choice accuracy as a function of evidence.
    
    Args:
        dataset: AccumulationTaskDataset instance
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    # Collect choices by delta
    choices_by_delta = defaultdict(list)
    
    for idx in range(len(dataset)):
        info = dataset.get_trial_info(idx)
        delta = info['delta']
        # "Choose right" if right was rewarded
        chose_correctly = 1
        choices_by_delta[delta].append(chose_correctly)
    
    # Calculate proportion choosing right for each delta
    deltas = sorted(choices_by_delta.keys())
    prop_correct = []
    counts = []
    
    for delta in deltas:
        choices = choices_by_delta[delta]
        prop_correct.append(np.mean(choices))
        counts.append(len(choices))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot as a line with error bars (binomial confidence intervals)
    errors = [1.96 * np.sqrt(p * (1 - p) / n) 
              for p, n in zip(prop_correct, counts)]
    
    ax.errorbar(
        deltas, 
        prop_correct, 
        yerr=errors,
        marker='o', 
        markersize=8,
        linewidth=2,
        capsize=5,
        label='Dataset Performance'
    )
    
    # Add horizontal line at 50%
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    # Add vertical line at delta=0
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Evidence Difference (#Right - #Left)', fontsize=12)
    ax.set_ylabel('Proportion Correct', fontsize=12)
    ax.set_title('Psychometric Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create dataset
    dataset = AccumulationTaskDataset(n_trials=200, seed=42)
    
    # Get a single trial
    inputs, target = dataset[0]
    print(f"Input shape: {inputs.shape}")  # (time_steps, 4)
    print(f"Target shape: {target.shape}")  # (time_steps,)
    print(f"Target values (first 5, last 5): {target[:5].numpy()}, {target[-5:].numpy()}")
    
    # Get trial details
    info = dataset.get_trial_info(0)
    print(f"Trial info: {info}")
    
    # Plot multiple trials as heatmaps
    fig1 = plot_trials(dataset, trial_indices=[0, 5, 10, 15])
    plt.savefig('trials_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved trials_heatmap.png")
    
    # Plot single trial with detailed view
    fig2 = plot_trial_detailed(dataset, trial_idx=0)
    plt.savefig('trial_detailed.png', dpi=150, bbox_inches='tight')
    print("Saved trial_detailed.png")
    
    # Plot psychometric curve
    fig3 = plot_psychometric_curve(dataset)
    plt.savefig('psychometric_curve.png', dpi=150, bbox_inches='tight')
    print("Saved psychometric_curve.png")
    
    plt.show()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_trials=800,
        val_trials=100,
        test_trials=100,
        batch_size=32
    )
    
    # Iterate through a batch
    for batch_inputs, batch_targets in train_loader:
        print(f"\nBatch input shape: {batch_inputs.shape}")  # (batch, time, 4)
        print(f"Batch target shape: {batch_targets.shape}")  # (batch, time)
        break