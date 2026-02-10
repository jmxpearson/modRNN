import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class RateRNN(nn.Module):
    """
    Rate-based recurrent neural network for cognitive tasks.
    
    Based on Yang et al. (2018) Nature Neuroscience:
    "Task representations in neural networks trained to perform many cognitive tasks"
    
    The network dynamics follow:
    τ * dr/dt = -r + f(W_rec * r + W_in * x + b_rec)
    
    where:
    - r is the firing rate of recurrent units
    - f is the activation function (ReLU, tanh, or softplus)
    - W_rec is the recurrent weight matrix
    - W_in is the input weight matrix
    - x is the external input
    - b_rec is the recurrent bias
    - τ is the time constant
    
    Output is a linear readout of firing rates at each timestep:
    y(t) = w_out^T * r(t) + b_out
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dt: float = 20.0,  # ms, discretization time step
        tau: float = 100.0,  # ms, time constant
        activation: str = 'relu',
        noise_std: float = 0.0,
        dale_ratio: Optional[float] = None,  # fraction of excitatory units (0.8 for Dale's law)
        alpha: Optional[float] = None,  # L2 regularization weight
        device: str = 'cpu'
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of recurrent units
            dt: Time step for discretization (ms)
            tau: Time constant (ms)
            activation: 'relu', 'tanh', or 'softplus'
            noise_std: Standard deviation of noise added to hidden units
            dale_ratio: If set, enforce Dale's law with this fraction of excitatory units
            alpha: L2 regularization strength for recurrent weights
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau
        self.alpha_rec = alpha if alpha is not None else 0.0
        self.noise_std = noise_std
        self.dale_ratio = dale_ratio
        self.device = device
        
        # Discretization: new_r = (1 - dt/tau) * r + (dt/tau) * f(...)
        self.alpha = dt / tau
        
        # Initialize activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input weights
        self.w_in = nn.Linear(input_size, hidden_size, bias=False)
        
        # Recurrent weights
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Output weights - linear readout producing scalar at each timestep
        self.w_out = nn.Linear(hidden_size, 1, bias=True)
        
        # Initialize weights
        self._initialize_weights()
        
        # Dale's law mask (if enabled)
        if dale_ratio is not None:
            self.register_buffer('dale_mask', self._create_dale_mask())
        else:
            self.dale_mask = None
    
    def _initialize_weights(self, spectral_radius: float = 1.2):
        """Initialize weights with spectral radius control for recurrent weights."""
        # Input weights: Gaussian with small variance
        nn.init.normal_(self.w_in.weight, mean=0.0, std=1.0 / np.sqrt(self.input_size))

        # Recurrent weights: random Gaussian, then rescale to target spectral radius
        nn.init.normal_(self.w_rec.weight, mean=0.0, std=1.0 / np.sqrt(self.hidden_size))
        with torch.no_grad():
            eigvals = torch.linalg.eigvals(self.w_rec.weight)
            current_radius = eigvals.abs().max().item()
            if current_radius > 0:
                self.w_rec.weight.mul_(spectral_radius / current_radius)
        nn.init.constant_(self.w_rec.bias, 0.0)

        # Output weights: scaled so initial output std is O(1)
        nn.init.normal_(self.w_out.weight, mean=0.0, std=1.0 / np.sqrt(self.hidden_size) * 2)
        nn.init.constant_(self.w_out.bias, 0.0)
    
    def _create_dale_mask(self) -> torch.Tensor:
        """
        Create mask to enforce Dale's law (neurons are either excitatory or inhibitory).
        
        Returns:
            Mask tensor of shape (hidden_size, hidden_size) with +1 for excitatory
            connections and -1 for inhibitory connections.
        """
        assert self.dale_ratio is not None, "dale_ratio must be set to create Dale mask"
        n_exc = int(self.dale_ratio * self.hidden_size)
        mask = torch.ones(self.hidden_size, self.hidden_size)
        mask[n_exc:, :] = -1  # Last (1-dale_ratio) fraction are inhibitory
        return mask
    
    def apply_dale_constraint(self):
        """Apply Dale's law constraint to recurrent weights."""
        if self.dale_mask is not None:
            with torch.no_grad():
                # Make weights non-negative then apply sign mask
                self.w_rec.weight.data = torch.abs(self.w_rec.weight.data) * self.dale_mask
    
    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN.

        Args:
            inputs: Input tensor of shape (batch, time, input_size)
            hidden: Initial hidden state of shape (batch, hidden_size)
                   If None, initialized to zeros
            return_states: If True, return full hidden state trajectory as third element

        Returns:
            outputs: Output time series of shape (batch, time) - one scalar per timestep
            hidden: Final hidden state of shape (batch, hidden_size)
            states: (only if return_states=True) Hidden states (batch, time, hidden_size)
        """
        batch_size, seq_len, _ = inputs.shape

        # Initialize hidden state if not provided
        h: torch.Tensor
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        else:
            h = hidden

        # Store outputs for all time steps
        outputs = []
        if return_states:
            states = []

        # Process each time step
        for t in range(seq_len):
            # Get input at current time step
            x_t = inputs[:, t, :]

            # Update hidden state using discretized dynamics
            # τ * dr/dt = -r + f(W_rec * r + W_in * x + b)
            # Discretized: r_new = (1 - α) * r + α * f(...)
            input_current = self.w_in(x_t)
            recurrent_current = self.w_rec(h)
            total_current = input_current + recurrent_current

            # Add noise if specified
            if self.noise_std > 0 and self.training:
                noise = torch.randn_like(total_current) * self.noise_std
                total_current = total_current + noise

            # Apply activation function
            activated = self.activation(total_current)

            # Update hidden state with discretized dynamics
            h = (1 - self.alpha) * h + self.alpha * activated

            # Compute linear readout at current time step
            output_t = self.w_out(h)  # (batch, 1)
            outputs.append(output_t.squeeze(-1))  # (batch,)
            if return_states:
                states.append(h)

        # Stack outputs into time series (batch, time)
        outputs_tensor = torch.stack(outputs, dim=1)

        if return_states:
            states_tensor = torch.stack(states, dim=1)  # (batch, time, hidden_size)
            return outputs_tensor, h, states_tensor

        return outputs_tensor, h
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute L2 regularization loss on recurrent weights.
        
        Returns:
            Regularization loss
        """
        if self.alpha_rec > 0:
            return self.alpha_rec * torch.sum(self.w_rec.weight ** 2)
        else:
            return torch.tensor(0.0, device=self.device)
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
        
        Returns:
            Hidden state tensor of shape (batch_size, hidden_size)
        """
        return torch.zeros(batch_size, self.hidden_size, device=self.device)


def train_step(
    model: RateRNN,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mask: Optional[torch.Tensor] = None
) -> Tuple[float, float]:
    """
    Perform one training step.
    
    Args:
        model: RateRNN model
        inputs: Input tensor (batch, time, input_size)
        targets: Target time series (batch, time) or (batch, 1) for final time point only
        optimizer: Optimizer
        criterion: Loss function
        mask: Optional mask for variable-length sequences (batch, time)
    
    Returns:
        task_loss: Task loss value
        reg_loss: Regularization loss value
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass - returns full time series
    outputs, _ = model(inputs)  # (batch, time)
    
    # Compute task loss
    if targets.dim() == 2 and targets.shape[1] == 1:
        # If target is only for final time point
        task_loss = criterion(outputs[:, -1], targets.squeeze(-1))
    else:
        # If target is a full time series
        if mask is not None:
            # Apply mask if provided
            task_loss = criterion(outputs * mask, targets * mask)
        else:
            task_loss = criterion(outputs, targets)
    
    # Compute regularization loss
    reg_loss = model.compute_regularization_loss()
    
    # Total loss
    total_loss = task_loss + reg_loss
    
    # Backward pass
    total_loss.backward()
    
    # Apply Dale's law constraint if enabled
    if model.dale_mask is not None:
        model.apply_dale_constraint()
    
    # Update weights
    optimizer.step()
    
    return task_loss.item(), reg_loss.item()


def evaluate(
    model: RateRNN,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    Evaluate model on a batch.
    
    Args:
        model: RateRNN model
        inputs: Input tensor (batch, time, input_size)
        targets: Target tensor (batch, 1) for final time point
        criterion: Loss function
    
    Returns:
        loss: Loss value
        accuracy: Classification accuracy (based on final output)
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass - returns full time series
        outputs, _ = model(inputs)  # (batch, time)
        
        # Use final time point for evaluation
        final_output = outputs[:, -1]  # (batch,)
        
        # Compute loss
        loss = criterion(final_output, targets.squeeze(-1))
        
        # Compute accuracy (for binary classification)
        predictions = (torch.sigmoid(final_output) > 0.5).long()
        accuracy = (predictions == targets.squeeze(-1)).float().mean()
    
    return loss.item(), accuracy.item()


# Example usage
if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = RateRNN(
        input_size=4,  # Matches accumulation task (left, right, hold, side cues)
        hidden_size=256,
        dt=20.0,
        tau=100.0,
        activation='relu',
        noise_std=0.05,
        dale_ratio=0.8,  # 80% excitatory neurons
        alpha=1e-4,  # L2 regularization
        device=device
    ).to(device)
    
    # Print model info
    print(f"\nModel architecture:")
    print(f"Input size: {model.input_size}")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Output: 1D time series (linear readout)")
    print(f"Time constant: {model.tau} ms")
    print(f"Time step: {model.dt} ms")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create dummy data
    batch_size = 32
    seq_len = 50
    dummy_input = torch.randn(batch_size, seq_len, 4).to(device)
    dummy_target = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    
    # Forward pass
    outputs, hidden = model(dummy_input)
    print(f"\nOutput shape: {outputs.shape}")  # (batch, time)
    print(f"Hidden state shape: {hidden.shape}")
    print(f"Sample output values (first trial, last 5 timesteps): {outputs[0, -5:].detach().cpu().numpy()}")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training step
    task_loss, reg_loss = train_step(
        model, dummy_input, dummy_target, optimizer, criterion
    )
    print(f"\nTask loss: {task_loss:.4f}")
    print(f"Regularization loss: {reg_loss:.4f}")
    
    # Evaluation
    loss, accuracy = evaluate(model, dummy_input, dummy_target, criterion)
    print(f"Eval loss: {loss:.4f}")
    print(f"Eval accuracy: {accuracy:.4f}")