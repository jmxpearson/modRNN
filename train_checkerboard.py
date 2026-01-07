import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from tqdm import tqdm

# Import from other files
from checkerboard import DelayedMatchToEvidenceDataset, create_dataloaders
from model import RateRNN


class Trainer:
    """
    Trainer class for the rate-based RNN on the delayed match-to-evidence task.
    """
    
    def __init__(
        self,
        model: RateRNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        save_dir: str = './checkpoints'
    ):
        """
        Args:
            model: RateRNN model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            device: 'cpu' or 'cuda'
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function: MSE between model output and target
        self.criterion = nn.MSELoss(reduction='none')  # No reduction for masking
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_task_loss': [],
            'train_reg_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
    
    def masked_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked MSE loss that ignores padded timesteps.
        
        Args:
            outputs: (batch, time)
            targets: (batch, time)
            lengths: (batch,) actual sequence lengths
            
        Returns:
            loss: Scalar loss value
        """
        # Compute element-wise loss
        losses = self.criterion(outputs, targets)  # (batch, time)
        
        # Create mask
        batch_size, max_len = outputs.shape
        mask = torch.arange(max_len, device=outputs.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.float()  # (batch, max_len)
        
        # Apply mask and compute mean
        masked_losses = losses * mask
        total_loss = masked_losses.sum()
        total_valid = mask.sum()
        
        return total_loss / total_valid if total_valid > 0 else total_loss
    
    def train_epoch(self) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Returns:
            avg_total_loss: Average total loss
            avg_task_loss: Average task loss (MSE)
            avg_reg_loss: Average regularization loss
        """
        self.model.train()
        total_losses = []
        task_losses = []
        reg_losses = []
        
        for batch_inputs, batch_targets, batch_lengths in tqdm(self.train_loader, desc='Training'):
            # Move to device
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            batch_lengths = batch_lengths.to(self.device)
            # Note: we don't need predom_colors or empirical_coherences for training loss
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.model(batch_inputs)  # (batch, time)
            
            # Compute task loss with masking
            task_loss = self.masked_loss(outputs, batch_targets, batch_lengths)
            
            # Compute regularization loss
            reg_loss = self.model.compute_regularization_loss()
            
            # Total loss
            total_loss = task_loss + reg_loss
            
            # Backward pass
            total_loss.backward()
            
            # Apply Dale's law constraint if enabled
            if self.model.dale_mask is not None:
                self.model.apply_dale_constraint()
            
            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Record losses
            total_losses.append(total_loss.item())
            task_losses.append(task_loss.item())
            reg_losses.append(reg_loss.item())
        
        return float(np.mean(total_losses)), float(np.mean(task_losses)), float(np.mean(reg_losses))
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            avg_loss: Average validation loss
            accuracy: Classification accuracy (based on final output sign during test period)
        """
        self.model.eval()
        losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets, batch_lengths in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                batch_lengths = batch_lengths.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(batch_inputs)  # (batch, time)
                
                # Compute loss
                loss = self.masked_loss(outputs, batch_targets, batch_lengths)
                losses.append(loss.item())
                
                # Compute accuracy based on final output (during test/choice period)
                # Get the actual final timestep for each sequence
                for i in range(outputs.size(0)):
                    seq_len = batch_lengths[i].item()
                    final_output = outputs[i, seq_len - 1]
                    final_target = batch_targets[i, seq_len - 1]
                    
                    # Predictions: sign of final output should match sign of target
                    prediction = torch.sign(final_output)
                    target_sign = torch.sign(final_target)
                    
                    if prediction == target_sign:
                        correct += 1
                    total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return float(np.mean(losses)), float(accuracy)
    
    def train(
        self,
        n_epochs: int,
        save_every: int = 10,
        plot_every: int = 5,
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            n_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            plot_every: Plot training progress every N epochs
            scheduler: Optional learning rate scheduler
        """
        print(f"Starting training for {n_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            
            # Train
            train_loss, task_loss, reg_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_task_loss'].append(task_loss)
            self.history['train_reg_loss'].append(reg_loss)
            
            # Check for NaN
            if np.isnan(train_loss):
                print("ERROR: NaN loss detected! Stopping training.")
                break
            
            # Validate
            val_loss, val_accuracy = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f} (Task: {task_loss:.4f}, Reg: {reg_loss:.6f})")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Update learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss and not np.isnan(val_loss):
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch)
                print(f"Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint periodically
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch)
            
            # Plot progress
            if epoch % plot_every == 0:
                self.plot_training_history()
        
        print("\nTraining completed!")
        self.save_checkpoint('final_model.pt', n_epochs)
    
    def save_checkpoint(self, filename: str, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['epoch']
    
    def plot_training_history(self, save: bool = True):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot total loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot task loss
        axes[0, 1].plot(epochs, self.history['train_task_loss'], linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Task Loss (MSE)')
        axes[0, 1].set_title('Task Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot regularization loss
        axes[1, 0].plot(epochs, self.history['train_reg_loss'], linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Regularization Loss')
        axes[1, 0].set_title('L2 Regularization Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1, 1].plot(epochs, self.history['val_accuracy'], linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        
        plt.show()


def plot_example_predictions(
    model: RateRNN,
    dataset: DelayedMatchToEvidenceDataset,
    device: str,
    n_examples: int = 4,
    save_path: str = './checkpoints/example_predictions.png'
):
    """
    Plot example predictions from the model.
    
    Args:
        model: Trained RateRNN model
        dataset: Dataset to sample from
        device: Device to run model on
        n_examples: Number of examples to plot
        save_path: Path to save the figure
    """
    model.eval()
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for i, trial_idx in enumerate(np.random.choice(len(dataset), n_examples, replace=False)):
            # Get trial data
            inputs, target = dataset[trial_idx]
            info = dataset.get_trial_info(trial_idx)
            
            # Run model
            inputs_batch = inputs.unsqueeze(0).to(device)  # (1, time, n_features)
            output, _ = model(inputs_batch)
            output = output.squeeze(0).cpu().numpy()  # (time,)
            
            # Create time axis
            time_axis = np.arange(len(output)) * dataset.time_bin_size
            sample_end = info['sample_duration']
            delay_end = info['sample_duration'] + info['delay_duration']
            
            # Plot
            axes[i].plot(time_axis, target.numpy(), 'k-', linewidth=2, label='Target', alpha=0.7)
            axes[i].plot(time_axis, output, 'r-', linewidth=2, label='Model Output')
            axes[i].axvline(x=sample_end, color='black', linestyle='--', alpha=0.5, label='Sample→Delay')
            axes[i].axvline(x=delay_end, color='purple', linestyle='--', alpha=0.5, label='Delay→Test')
            axes[i].axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Shade regions
            axes[i].axvspan(0, sample_end, alpha=0.1, color='lightblue')
            axes[i].axvspan(sample_end, delay_end, alpha=0.1, color='lightyellow')
            axes[i].axvspan(delay_end, time_axis[-1], alpha=0.1, color='lightgreen')
            
            axes[i].set_ylabel('Output', fontsize=11)
            axes[i].set_ylim(-1.5, 1.5)
            axes[i].grid(True, alpha=0.3)
            
            axes[i].set_title(
                f"Trial {trial_idx}: Predom={info['predom_color'].capitalize()} | "
                f"Coherence={info['coherence']:.2f} (Empirical={info['empirical_coherence']:.2f})",
                fontsize=11
            )
            
            if i == 0:
                axes[i].legend(loc='upper right')
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main training script."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_trials=2000,
        val_trials=500,
        test_trials=500,
        batch_size=32,
        n_checkerboard_channels=10  # Number of binary checker inputs
    )
    
    # Determine input size (3 cues + n_checkerboard_channels)
    n_checkerboard_channels = 10
    input_size = 3 + n_checkerboard_channels
    
    # Create model
    print("Creating model...")
    model = RateRNN(
        input_size=input_size,
        hidden_size=256,
        dt=20.0,
        tau=100.0,
        activation='relu',
        noise_std=0.01,  # Reduced noise for stability
        dale_ratio=None,  # Disable Dale's law initially for stability
        alpha=1e-5,  # Reduced L2 regularization
        device=device
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer with lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir='./checkpoints'
    )
    
    # Train model
    trainer.train(
        n_epochs=100,
        save_every=10,
        plot_every=5,
        scheduler=scheduler
    )
    
    # Load best model
    print("\nLoading best model...")
    trainer.load_checkpoint('best_model.pt')
    
    # Plot example predictions
    print("Plotting example predictions...")
    # Type assertion for the dataset
    train_dataset = train_loader.dataset
    assert isinstance(train_dataset, DelayedMatchToEvidenceDataset)
    plot_example_predictions(
        model=model,
        dataset=train_dataset,
        device=device,
        n_examples=6
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    
    model.eval()
    test_losses = []
    test_correct = 0
    test_total = 0
    
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch_inputs, batch_targets, batch_lengths in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_lengths = batch_lengths.to(device)
            
            outputs, _ = model(batch_inputs)
            
            # Masked loss
            losses = criterion(outputs, batch_targets)
            batch_size, max_len = outputs.shape
            mask = torch.arange(max_len, device=outputs.device).unsqueeze(0) < batch_lengths.unsqueeze(1)
            mask = mask.float()
            masked_losses = losses * mask
            loss = masked_losses.sum() / mask.sum()
            test_losses.append(loss.item())
            
            # Accuracy
            for i in range(outputs.size(0)):
                seq_len = batch_lengths[i].item()
                final_output = outputs[i, seq_len - 1]
                final_target = batch_targets[i, seq_len - 1]
                prediction = torch.sign(final_output)
                target_sign = torch.sign(final_target)
                if prediction == target_sign:
                    test_correct += 1
                test_total += 1
    
    test_loss = np.mean(test_losses)
    test_accuracy = test_correct / test_total
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save final results
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'best_val_loss': float(trainer.best_val_loss),
        'final_train_loss': float(trainer.history['train_loss'][-1]),
        'final_val_accuracy': float(trainer.history['val_accuracy'][-1])
    }
    
    with open('./checkpoints/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete! Results saved to ./checkpoints/")


if __name__ == "__main__":
    main()