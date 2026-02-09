import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, Any


class DelayedMatchToEvidenceDataset(Dataset):
    """
    PyTorch Dataset for the delayed match-to-evidence task from Costa et al. (2025).
    
    Mice navigate a virtual T-maze accumulating evidence about checkerboard color
    (black vs white), hold this information through a delay, then choose between
    test stimuli based on the predominant sample color.
    
    Task structure:
    - Sample period: 100 cm with bilaterally symmetrical checkerboard (variable coherence)
    - Delay period: 50 or 75 cm with neutral wallpaper (no cues)
    - Test period: 75 cm with black and white test stimuli revealed
    - Choice arm: 20 cm
    - Variable gains control duration of sample and delay periods
    """
    
    def __init__(
        self,
        n_trials: int = 1000,
        sample_length: int = 100,  # cm
        delay_length: int = 75,  # cm (50 or 75 in paper)
        test_length: int = 75,  # cm
        arm_length: int = 20,  # cm
        time_bin_size: float = 0.1,  # seconds
        avg_speed: float = 60.0,  # cm/s
        coherence_min: float = 0.55,
        coherence_max: float = 0.95,
        coherence_rate: float = -9.0,  # exponential rate parameter
        gain_values: Optional[list] = None,  # [0.67, 1.0, 1.25]
        gain_probs: Optional[list] = None,  # [0.3, 0.4, 0.3]
        n_checkerboard_channels: int = 10,  # Number of fixed checker channels
        seed: Optional[int] = None
    ):
        """
        Args:
            n_trials: Number of trials to generate
            sample_length: Length of sample region in cm
            delay_length: Length of delay region in cm (50 or 75)
            test_length: Length of test region in cm
            arm_length: Length of choice arm in cm
            time_bin_size: Time resolution for neural data (seconds)
            avg_speed: Average running speed in cm/s
            coherence_min: Minimum checkerboard coherence
            coherence_max: Maximum checkerboard coherence
            coherence_rate: Rate parameter for exponential coherence distribution
            gain_values: List of gain values for variable durations
            gain_probs: Probabilities for each gain value
            n_checkerboard_channels: Number of fixed checker channels
            seed: Random seed for reproducibility
        """
        self.n_trials = n_trials
        self.sample_length = sample_length
        self.delay_length = delay_length
        self.test_length = test_length
        self.arm_length = arm_length
        self.total_length = sample_length + delay_length + test_length + arm_length
        self.time_bin_size = time_bin_size
        self.avg_speed = avg_speed
        self.coherence_min = coherence_min
        self.coherence_max = coherence_max
        self.coherence_rate = coherence_rate
        self.n_checkerboard_channels = n_checkerboard_channels
        
        if gain_values is None:
            gain_values = [0.67, 1.0, 1.25]
        if gain_probs is None:
            gain_probs = [0.3, 0.4, 0.3]
            
        self.gain_values = gain_values
        self.gain_probs = gain_probs
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate all trials
        self.trials = self._generate_trials()
    
    def _sample_coherence(self) -> float:
        """Sample coherence from exponential distribution as in paper."""
        # Sample from exponential with rate parameter
        coherence = np.random.exponential(-1.0 / self.coherence_rate)
        # Rescale and clip to desired range
        coherence = coherence * (self.coherence_max - self.coherence_min) + self.coherence_min
        coherence = np.clip(coherence, self.coherence_min, self.coherence_max)
        return coherence
    
    def _generate_fixed_checkers(self, coherence: float, predom_color: int, 
                                  n_channels: int) -> np.ndarray:
        """
        Generate fixed binary checkers for the entire sample period.
        
        Each checker is drawn once from a binomial distribution and remains
        fixed throughout the sample period.
        
        Args:
            coherence: Probability that each checker is the predominant color
            predom_color: 0=black, 1=white
            n_channels: Number of checker channels
            
        Returns:
            checkers: Binary array (n_channels,) with 0=black, 1=white
        """
        checkers = np.zeros(n_channels, dtype=int)
        for i in range(n_channels):
            # Each checker is drawn from binomial: 1 with probability=coherence
            if np.random.rand() < coherence:
                checkers[i] = predom_color
            else:
                checkers[i] = 1 - predom_color
        
        return checkers
    
    def _generate_single_trial(self) -> Dict[str, Any]:
        """Generate a single trial with checkerboard evidence and task structure."""
        # Sample coherence and predominant color
        coherence = self._sample_coherence()
        predom_color = np.random.choice([0, 1])  # 0=black, 1=white
        
        # Generate fixed checkers for this trial (drawn once from binomial distribution)
        fixed_checkers = self._generate_fixed_checkers(
            coherence, predom_color, self.n_checkerboard_channels
        )
        
        # Sample gains for variable durations
        sample_gain = np.random.choice(self.gain_values, p=self.gain_probs)
        delay_gain = np.random.choice(self.gain_values, p=self.gain_probs)
        
        # Calculate actual durations based on gains
        sample_duration = self.sample_length / (self.avg_speed * sample_gain)
        delay_duration = self.delay_length / (self.avg_speed * delay_gain)
        test_duration = self.test_length / self.avg_speed
        arm_duration = self.arm_length / self.avg_speed
        
        total_duration = sample_duration + delay_duration + test_duration + arm_duration
        n_timesteps = int(total_duration / self.time_bin_size)
        
        # Initialize input arrays (time x features)
        sample_cue = np.zeros(n_timesteps)  # Indicates sample period
        delay_cue = np.zeros(n_timesteps)  # Indicates delay period
        test_cue = np.zeros(n_timesteps)  # Indicates test period
        # Binary checkerboard inputs - fixed for entire sample period
        checkerboard_samples = np.zeros((n_timesteps, self.n_checkerboard_channels), dtype=int)
        target_output = np.zeros(n_timesteps)  # Target choice
        
        # Calculate time indices for each epoch
        sample_end_idx = int(sample_duration / self.time_bin_size)
        delay_end_idx = int((sample_duration + delay_duration) / self.time_bin_size)
        test_end_idx = int((sample_duration + delay_duration + test_duration) / self.time_bin_size)
        
        # Sample period: present fixed checkers throughout
        sample_cue[:sample_end_idx] = 1.0
        for t_idx in range(sample_end_idx):
            checkerboard_samples[t_idx, :] = fixed_checkers
        
        # Delay period: hold information with no cues
        delay_cue[sample_end_idx:delay_end_idx] = 1.0
        
        # Calculate empirical coherence from the fixed checkers
        n_predom = np.sum(fixed_checkers == predom_color)
        empirical_coherence = n_predom / len(fixed_checkers)
        
        # Test period: test stimuli revealed
        # test_cue[delay_end_idx:test_end_idx] = 1.0
        test_cue[delay_end_idx:] = 1.0

        # Randomize test stimulus sides (which side has black vs white)
        test_side = np.random.randint(0, 2)  # 0=left, 1=right

        # Determine empirical predominant color from actual checker counts
        if n_predom >= self.n_checkerboard_channels - n_predom:
            empirical_predom_color = predom_color
        else:
            empirical_predom_color = 1 - predom_color

        # Target output: correct side based on XOR of empirical_predom_color and test_side
        # If predom=black(0) & side=0 -> correct=0 (left, -1)
        # If predom=black(0) & side=1 -> correct=1 (right, +1)
        # If predom=white(1) & side=0 -> correct=1 (right, +1)
        # If predom=white(1) & side=1 -> correct=0 (left, -1)
        correct_side = empirical_predom_color ^ test_side
        target_output[delay_end_idx:] = -1.0 if correct_side == 0 else 1.0
        
        return {
            'sample_cue': sample_cue,
            'delay_cue': delay_cue,
            'test_cue': test_cue,
            'checkerboard_samples': checkerboard_samples,
            'target_output': target_output,
            'coherence': coherence,
            'predom_color': predom_color,
            'sample_gain': sample_gain,
            'delay_gain': delay_gain,
            'sample_duration': sample_duration,
            'delay_duration': delay_duration,
            'test_side': test_side,
            'correct_side': correct_side,
            'fixed_checkers': fixed_checkers,
            'empirical_coherence': empirical_coherence,
            'n_predom_checkers': int(n_predom),
            'empirical_predom_color': empirical_predom_color
        }
    
    def _generate_trials(self):
        """Generate all trials for the dataset."""
        return [self._generate_single_trial() for _ in range(self.n_trials)]
    
    def __len__(self) -> int:
        return self.n_trials
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs: Tensor of shape (time, 3 + n_checkerboard_channels) containing:
                - sample_cue (binary, active during sample)
                - delay_cue (binary, active during delay)
                - test_cue (binary, active during test)
                - checkerboard_0 to checkerboard_N (binary, 0=black, 1=white)
            target: Tensor of shape (time,) containing desired output:
                - 0 during sample and delay
                - -1 (black) or +1 (white) during test and choice
        """
        trial = self.trials[idx]
        
        # Stack cue features
        cues = np.stack([
            trial['sample_cue'],
            trial['delay_cue'],
            trial['test_cue']
        ], axis=1)
        
        # Combine cues with checkerboard samples (convert to float for consistency)
        inputs = np.concatenate([cues, trial['checkerboard_samples'].astype(float)], axis=1)
        
        inputs = torch.FloatTensor(inputs)
        target = torch.FloatTensor(trial['target_output'])
        
        return inputs, target
    
    def get_trial_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a specific trial."""
        trial = self.trials[idx]
        
        return {
            'coherence': trial['coherence'],
            'predom_color': 'white' if trial['predom_color'] == 1 else 'black',
            'sample_gain': trial['sample_gain'],
            'delay_gain': trial['delay_gain'],
            'sample_duration': trial['sample_duration'],
            'delay_duration': trial['delay_duration'],
            'test_side': 'right' if trial['test_side'] == 1 else 'left',
            'correct_side': 'right' if trial['correct_side'] == 1 else 'left',
            'difficulty': 1.0 - trial['coherence'],
            'n_predom_checkers': trial['n_predom_checkers'],
            'empirical_coherence': trial['empirical_coherence'],
            'empirical_predom_color': 'white' if trial['empirical_predom_color'] == 1 else 'black'
        }


def collate_variable_length_trials(batch):
    """
    Custom collate function to handle variable-length trials.
    Pads sequences to the maximum length in the batch.
    Target outputs are padded with their final value (not zeros).
    All inputs are padded with their final values through trial end.
    
    Args:
        batch: List of (inputs, target) tuples
        
    Returns:
        inputs: Padded tensor of shape (batch, max_time, features)
        targets: Padded tensor of shape (batch, max_time)
        lengths: Tensor of actual sequence lengths
    """
    inputs_list, targets_list = zip(*batch)
    
    # Get lengths
    lengths = torch.tensor([inp.shape[0] for inp in inputs_list])
    max_len = int(lengths.max().item())
    batch_size = len(inputs_list)
    n_features = inputs_list[0].shape[1]
    
    # Initialize padded tensors
    inputs_padded = torch.zeros(batch_size, max_len, n_features)
    targets_padded = torch.zeros(batch_size, max_len)
    
    # Fill in the data
    for i, (inp, tgt) in enumerate(zip(inputs_list, targets_list)):
        seq_len = inp.shape[0]
        inputs_padded[i, :seq_len, :] = inp
        targets_padded[i, :seq_len] = tgt
        
        # Pad target with its final value instead of zeros
        if seq_len < max_len:
            final_value = tgt[-1]
            targets_padded[i, seq_len:] = final_value
            
            # Optionally, you could extend all final inputs:
            final_value = inp[-1]
            inputs_padded[i, seq_len:, :] = final_value
    
    return inputs_padded, targets_padded, lengths


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
        **dataset_kwargs: Additional arguments passed to DelayedMatchToEvidenceDataset
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = DelayedMatchToEvidenceDataset(
        n_trials=train_trials, 
        seed=0, 
        **dataset_kwargs
    )
    val_dataset = DelayedMatchToEvidenceDataset(
        n_trials=val_trials, 
        seed=1, 
        **dataset_kwargs
    )
    test_dataset = DelayedMatchToEvidenceDataset(
        n_trials=test_trials, 
        seed=2, 
        **dataset_kwargs
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_variable_length_trials
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_variable_length_trials
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_variable_length_trials
    )
    
    return train_loader, val_loader, test_loader


def plot_trials(
    dataset: DelayedMatchToEvidenceDataset,
    trial_indices: list = [0, 1, 2, 3],
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot multiple trials as heatmaps with time on x-axis and variables on y-axis.
    Uses blue/orange color scheme instead of red/green.
    
    Args:
        dataset: DelayedMatchToEvidenceDataset instance
        trial_indices: List of trial indices to plot
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    
    n_trials = len(trial_indices)
    fig, axes = plt.subplots(n_trials, 1, figsize=figsize)
    
    if n_trials == 1:
        axes = [axes]
    
    # Generate feature names dynamically based on number of checkerboard channels
    feature_names = ['Sample Cue', 'Delay Cue', 'Test Cue']
    feature_names += [f'Checker {i}' for i in range(dataset.n_checkerboard_channels)]
    feature_names += ['Side Cue', 'Target Output']

    for ax_idx, trial_idx in enumerate(trial_indices):
        inputs, target = dataset[trial_idx]
        info = dataset.get_trial_info(trial_idx)
        trial = dataset.trials[trial_idx]

        # Combine inputs and target for plotting
        inputs_np = inputs.numpy().T
        target_np = target.numpy()[np.newaxis, :]

        # Create side cue row (constant value throughout trial, shown during test period)
        n_timesteps = inputs.shape[0]
        side_cue_np = np.zeros((1, n_timesteps))
        delay_end_idx = int((info['sample_duration'] + info['delay_duration']) / dataset.time_bin_size)
        # Side cue: -1 for left (0), +1 for right (1)
        side_cue_np[0, delay_end_idx:] = -1.0 if trial['test_side'] == 0 else 1.0

        data = np.vstack([inputs_np, side_cue_np, target_np])
        
        # Create heatmap with blue/orange colormap
        im = axes[ax_idx].imshow(
            data,
            aspect='auto',
            cmap='RdYlBu_r',  # Blue for negative, orange/red for positive
            interpolation='nearest',
            vmin=-1,
            vmax=1
        )
        
        # Set y-axis labels
        axes[ax_idx].set_yticks(range(len(feature_names)))
        axes[ax_idx].set_yticklabels(feature_names)
        
        # Add vertical lines for epoch boundaries
        sample_end = info['sample_duration'] / dataset.time_bin_size
        delay_end = (info['sample_duration'] + info['delay_duration']) / dataset.time_bin_size
        
        axes[ax_idx].axvline(x=sample_end, color='black', linestyle='--', 
                            linewidth=2, alpha=0.7, label='Sample→Delay')
        axes[ax_idx].axvline(x=delay_end, color='purple', linestyle='--', 
                            linewidth=2, alpha=0.7, label='Delay→Test')
        
        # Add title with trial information
        title = (f"Trial {trial_idx}: Predom={info['predom_color'].capitalize()} | "
                f"EmpPredom={info['empirical_predom_color'].capitalize()} | "
                f"SideCue={info['test_side'].capitalize()} | "
                f"Correct={info['correct_side'].capitalize()}")
        axes[ax_idx].set_title(title, fontsize=10, pad=10)
        
        # Add x-axis label only for bottom plot
        if ax_idx == n_trials - 1:
            axes[ax_idx].set_xlabel('Time (bins)', fontsize=11)
        else:
            axes[ax_idx].set_xticks([])
        
        # Add legend for first plot
        if ax_idx == 0:
            axes[ax_idx].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes((0.94, 0.15, 0.02, 0.7))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Activity', rotation=270, labelpad=20)
    
    return fig


def plot_trial_detailed(
    dataset: DelayedMatchToEvidenceDataset,
    trial_idx: int = 0,
    figsize: Tuple[int, int] = (14, 12)
):
    """
    Plot a single trial with separate subplots for each variable.
    Uses blue/orange color scheme.
    
    Args:
        dataset: DelayedMatchToEvidenceDataset instance
        trial_idx: Index of trial to plot
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    
    inputs, target = dataset[trial_idx]
    info = dataset.get_trial_info(trial_idx)
    trial = dataset.trials[trial_idx]
    inputs_np = inputs.numpy()
    target_np = target.numpy()

    # Create time axis in seconds
    time_axis = np.arange(len(inputs_np)) * dataset.time_bin_size

    # Calculate epoch boundaries
    sample_end = info['sample_duration']
    delay_end = info['sample_duration'] + info['delay_duration']
    test_end = delay_end + (dataset.test_length / dataset.avg_speed)

    # Create side cue time series (shown during test period)
    side_cue_np = np.zeros(len(inputs_np))
    delay_end_idx = int(delay_end / dataset.time_bin_size)
    side_cue_np[delay_end_idx:] = -1.0 if trial['test_side'] == 0 else 1.0

    fig, axes = plt.subplots(7, 1, figsize=figsize, sharex=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2', '#8c564b', '#d62728']
    
    # Plot first 3 cue features
    cue_names = ['Sample Cue', 'Delay Cue', 'Test Cue']
    for i in range(3):
        name, color = cue_names[i], colors[i]
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
        
        # Add vertical lines for epoch boundaries
        axes[i].axvline(x=sample_end, color='black', linestyle='--', 
                       linewidth=1.5, alpha=0.5)
        axes[i].axvline(x=delay_end, color='purple', linestyle='--', 
                       linewidth=1.5, alpha=0.5)
        
        # Shade epochs
        axes[i].axvspan(0, sample_end, alpha=0.1, color='lightblue', 
                       label='Sample' if i == 0 else '')
        axes[i].axvspan(sample_end, delay_end, alpha=0.1, color='lightyellow', 
                       label='Delay' if i == 0 else '')
        axes[i].axvspan(delay_end, test_end, alpha=0.1, color='lightgreen', 
                       label='Test' if i == 0 else '')
        
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=9)
    
    # Plot a few checkerboard channels (up to 2)
    n_channels_to_plot = min(2, dataset.n_checkerboard_channels)
    for i in range(n_channels_to_plot):
        ax_idx = 3 + i
        channel_idx = 3 + i  # After the 3 cue inputs
        axes[ax_idx].plot(time_axis, inputs_np[:, channel_idx],
                          color=colors[ax_idx], linewidth=2)
        axes[ax_idx].fill_between(
            time_axis,
            0,
            inputs_np[:, channel_idx],
            color=colors[ax_idx],
            alpha=0.3
        )
        axes[ax_idx].set_ylabel(f'Checker {i}', fontsize=11, fontweight='bold')
        axes[ax_idx].set_ylim(-0.1, 1.1)
        axes[ax_idx].grid(True, alpha=0.3)

        # Add vertical lines for epoch boundaries
        axes[ax_idx].axvline(x=sample_end, color='black', linestyle='--',
                             linewidth=1.5, alpha=0.5)
        axes[ax_idx].axvline(x=delay_end, color='purple', linestyle='--',
                             linewidth=1.5, alpha=0.5)

        # Shade epochs
        axes[ax_idx].axvspan(0, sample_end, alpha=0.1, color='lightblue')
        axes[ax_idx].axvspan(sample_end, delay_end, alpha=0.1, color='lightyellow')
        axes[ax_idx].axvspan(delay_end, test_end, alpha=0.1, color='lightgreen')

    # Plot side cue
    axes[5].plot(time_axis, side_cue_np, color=colors[5], linewidth=2)
    axes[5].fill_between(
        time_axis,
        0,
        side_cue_np,
        color=colors[5],
        alpha=0.3
    )
    axes[5].set_ylabel('Side Cue', fontsize=11, fontweight='bold')
    axes[5].set_ylim(-1.2, 1.2)
    axes[5].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axes[5].grid(True, alpha=0.3)

    # Add vertical lines for epoch boundaries
    axes[5].axvline(x=sample_end, color='black', linestyle='--',
                   linewidth=1.5, alpha=0.5)
    axes[5].axvline(x=delay_end, color='purple', linestyle='--',
                   linewidth=1.5, alpha=0.5)

    # Shade epochs
    axes[5].axvspan(0, sample_end, alpha=0.1, color='lightblue')
    axes[5].axvspan(sample_end, delay_end, alpha=0.1, color='lightyellow')
    axes[5].axvspan(delay_end, test_end, alpha=0.1, color='lightgreen')

    # Plot target output
    axes[6].plot(time_axis, target_np, color=colors[6], linewidth=2)
    axes[6].fill_between(
        time_axis,
        0,
        target_np,
        color=colors[6],
        alpha=0.3
    )
    axes[6].set_ylabel('Target Output', fontsize=11, fontweight='bold')
    axes[6].set_ylim(-1.2, 1.2)
    axes[6].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axes[6].grid(True, alpha=0.3)

    # Add vertical lines for epoch boundaries
    axes[6].axvline(x=sample_end, color='black', linestyle='--',
                   linewidth=1.5, alpha=0.5)
    axes[6].axvline(x=delay_end, color='purple', linestyle='--',
                   linewidth=1.5, alpha=0.5)

    # Shade epochs
    axes[6].axvspan(0, sample_end, alpha=0.1, color='lightblue')
    axes[6].axvspan(sample_end, delay_end, alpha=0.1, color='lightyellow')
    axes[6].axvspan(delay_end, test_end, alpha=0.1, color='lightgreen')

    axes[-1].set_xlabel('Time (seconds)', fontsize=12)

    fig.suptitle(
        f"Trial {trial_idx}: Predom={info['predom_color'].capitalize()} | "
        f"EmpPredom={info['empirical_predom_color'].capitalize()} | "
        f"SideCue={info['test_side'].capitalize()} | "
        f"Correct={info['correct_side'].capitalize()}\n"
        f"Sample={info['sample_duration']:.2f}s, Delay={info['delay_duration']:.2f}s | "
        f"{info['n_predom_checkers']}/{dataset.n_checkerboard_channels} checkers",
        fontsize=11,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    return fig


def plot_psychometric_curve(
    dataset: DelayedMatchToEvidenceDataset,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot psychometric curves showing performance as a function of coherence
    and duration parameters.
    
    Args:
        dataset: DelayedMatchToEvidenceDataset instance
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    # Collect data by coherence bins
    coherence_bins = np.linspace(dataset.coherence_min, dataset.coherence_max, 10)
    coherence_data = defaultdict(list)
    
    # Collect data by sample duration
    sample_dur_data = defaultdict(list)
    
    # Collect data by delay duration
    delay_dur_data = defaultdict(list)
    
    for idx in range(len(dataset)):
        info = dataset.get_trial_info(idx)
        
        # Bin coherence
        coh_bin = np.digitize(info['coherence'], coherence_bins)
        coherence_data[coh_bin].append(1)  # Assume correct for visualization
        
        # Bin sample duration
        sample_dur_data[info['sample_gain']].append(1)
        
        # Bin delay duration
        delay_dur_data[info['delay_gain']].append(1)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot coherence curve
    coh_vals = []
    coh_perf = []
    for bin_idx in sorted(coherence_data.keys()):
        if coherence_data[bin_idx]:
            coh_vals.append(coherence_bins[bin_idx-1] if bin_idx > 0 else coherence_bins[0])
            coh_perf.append(np.mean(coherence_data[bin_idx]))
    
    axes[0].plot(coh_vals, coh_perf, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    axes[0].set_xlabel('Checkerboard Coherence', fontsize=11)
    axes[0].set_ylabel('Performance', fontsize=11)
    axes[0].set_title('Effect of Coherence', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    
    # Plot sample duration effect
    gains = sorted(sample_dur_data.keys())
    perfs = [np.mean(sample_dur_data[g]) for g in gains]
    axes[1].bar(range(len(gains)), perfs, color='#ff7f0e', alpha=0.7)
    axes[1].set_xticks(range(len(gains)))
    axes[1].set_xticklabels([f'{g:.2f}' for g in gains])
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Sample Gain', fontsize=11)
    axes[1].set_ylabel('Performance', fontsize=11)
    axes[1].set_title('Effect of Sample Duration', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 1.05)
    
    # Plot delay duration effect
    gains = sorted(delay_dur_data.keys())
    perfs = [np.mean(delay_dur_data[g]) for g in gains]
    axes[2].bar(range(len(gains)), perfs, color='#2ca02c', alpha=0.7)
    axes[2].set_xticks(range(len(gains)))
    axes[2].set_xticklabels([f'{g:.2f}' for g in gains])
    axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Delay Gain', fontsize=11)
    axes[2].set_ylabel('Performance', fontsize=11)
    axes[2].set_title('Effect of Delay Duration', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim(0, 1.05)
    
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create dataset
    dataset = DelayedMatchToEvidenceDataset(n_trials=200, seed=42)
    
    # Get a single trial
    inputs, target = dataset[0]
    print(f"Input shape: {inputs.shape}")  # (time_steps, 3 + n_checkerboard_channels)
    print(f"Target shape: {target.shape}")  # (time_steps,)
    print(f"Number of input features: 3 cues + {dataset.n_checkerboard_channels} checkerboard channels = {inputs.shape[1]}")
    print(f"Target values (first 5, last 5): {target[:5].numpy()}, {target[-5:].numpy()}")
    
    # Get trial details
    info = dataset.get_trial_info(0)
    print(f"\nTrial info keys: {info.keys()}")
    print(f"Trial info: {info}")
    
    if 'empirical_coherence' in info:
        print(f"\nExample checkerboard values at timestep 10: {inputs[10, 3:].numpy()}")
        print(f"These are binary (0=black, 1=white) and FIXED throughout sample period")
        print(f"Empirical coherence: {info['empirical_coherence']:.2f} (target: {info['coherence']:.2f})")
        print(f"Number of predominant-color checkers: {info['n_predom_checkers']}/{dataset.n_checkerboard_channels}")
    
    # Create plots folder if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Plot multiple trials as heatmaps
    fig1 = plot_trials(dataset, trial_indices=[0, 5, 10, 15])
    plt.savefig('plots/delayed_match_trials_heatmap.png', dpi=150, bbox_inches='tight')
    print("\nSaved plots/delayed_match_trials_heatmap.png")

    # Plot single trial with detailed view
    fig2 = plot_trial_detailed(dataset, trial_idx=0)
    plt.savefig('plots/delayed_match_trial_detailed.png', dpi=150, bbox_inches='tight')
    print("Saved plots/delayed_match_trial_detailed.png")

    # Plot psychometric curves
    fig3 = plot_psychometric_curve(dataset)
    plt.savefig('plots/delayed_match_psychometric.png', dpi=150, bbox_inches='tight')
    print("Saved plots/delayed_match_psychometric.png")
    
    plt.show()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_trials=800,
        val_trials=100,
        test_trials=100,
        batch_size=32
    )
    
    # Iterate through a batch
    for batch_inputs, batch_targets, batch_lengths in train_loader:
        print(f"\nBatch input shape: {batch_inputs.shape}")  # (batch, max_time, 3 + n_checkerboard_channels)
        print(f"Batch target shape: {batch_targets.shape}")  # (batch, max_time)
        print(f"Batch lengths: {batch_lengths}")  # Actual sequence lengths
        print(f"All checkerboard values are binary: {torch.all((batch_inputs[:,:,3:] == 0) | (batch_inputs[:,:,3:] == 1)).item()}")
        
        # Verify that checkers are fixed during sample period (for first trial)
        first_trial_len = batch_lengths[0].item()
        first_trial_sample = batch_inputs[0, :min(10, first_trial_len), 3:]  # First timesteps of first trial
        if first_trial_sample.shape[0] > 1:
            all_same = torch.all(first_trial_sample == first_trial_sample[0], dim=0)
            print(f"Checkers are fixed during sample period: {torch.all(all_same).item()}")
        break