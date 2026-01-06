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
from loader import AccumulationTaskDataset, create_dataloaders
from model import RateRNN


class Trainer:
    """
    Trainer class for the rate-based RNN on the accumulation task.
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
        self.criterion = nn.MSELoss()
        
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
        
        for batch_inputs, batch_targets in tqdm(self.train_loader, desc='Training'):
            # Move to device
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.model(batch_inputs)  # (batch, time)
            
            # Compute task loss (MSE between model output and target)
            task_loss = self.criterion(outputs, batch_targets)
            
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
            accuracy: Classification accuracy (based on final output sign)
        """
        self.model.eval()
        losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(batch_inputs)  # (batch, time)
                
                # Compute loss
                loss = self.criterion(outputs, batch_targets)
                losses.append(loss.item())
                
                # Compute accuracy based on final output
                final_outputs = outputs[:, -1]  # (batch,)
                final_targets = batch_targets[:, -1]  # (batch,)
                
                # Predictions: sign of final output should match sign of target
                predictions = torch.sign(final_outputs)
                targets_sign = torch.sign(final_targets)
                
                correct += (predictions == targets_sign).sum().item()
                total += batch_targets.size(0)
        
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
    dataset: AccumulationTaskDataset,
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
            inputs_batch = inputs.unsqueeze(0).to(device)  # (1, time, 4)
            output, _ = model(inputs_batch)
            output = output.squeeze(0).cpu().numpy()  # (time,)
            
            # Create time axis
            time_axis = np.arange(len(output)) * dataset.time_bin_size
            cue_duration = dataset.cue_period_length / dataset.avg_speed
            
            # Plot
            axes[i].plot(time_axis, target.numpy(), 'k-', linewidth=2, label='Target', alpha=0.7)
            axes[i].plot(time_axis, output, 'r-', linewidth=2, label='Model Output')
            axes[i].axvline(x=cue_duration, color='blue', linestyle='--', alpha=0.5, label='Delay Start')
            axes[i].axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Shade regions
            axes[i].axvspan(0, cue_duration, alpha=0.1, color='yellow')
            axes[i].axvspan(cue_duration, time_axis[-1], alpha=0.1, color='cyan')
            
            axes[i].set_ylabel('Output', fontsize=11)
            axes[i].set_ylim(-1.5, 1.5)
            axes[i].grid(True, alpha=0.3)
            
            rewarded_side = 'Right' if info['rewarded_side'] == 1 else 'Left'
            axes[i].set_title(
                f"Trial {trial_idx}: {rewarded_side} | #L={info['n_left']}, #R={info['n_right']}, Δ={info['delta']:+d}",
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
        batch_size=32
        # Note: seed is handled internally by create_dataloaders
    )
    
    # Create model
    print("Creating model...")
    model = RateRNN(
        input_size=4,
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
    assert isinstance(train_dataset, AccumulationTaskDataset)
    plot_example_predictions(
        model=model,
        dataset=train_dataset,
        device=device,
        n_examples=6
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loader_eval = DataLoader(
        test_loader.dataset,
        batch_size=32,
        shuffle=False
    )
    
    model.eval()
    test_losses = []
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader_eval:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs, _ = model(batch_inputs)
            loss = nn.MSELoss()(outputs, batch_targets)
            test_losses.append(loss.item())
            
            # Accuracy
            final_outputs = outputs[:, -1]
            final_targets = batch_targets[:, -1]
            predictions = torch.sign(final_outputs)
            targets_sign = torch.sign(final_targets)
            test_correct += (predictions == targets_sign).sum().item()
            test_total += batch_targets.size(0)
    
    test_loss = np.mean(test_losses)
    test_accuracy = test_correct / test_total
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save final results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'best_val_loss': trainer.best_val_loss,
        'final_train_loss': trainer.history['train_loss'][-1],
        'final_val_accuracy': trainer.history['val_accuracy'][-1]
    }
    
    with open('./checkpoints/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete! Results saved to ./checkpoints/")


if __name__ == "__main__":
    main()