import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


class NeuromodRNN(nn.Module):
    """
    Neuromodulated voltage-based recurrent neural network with Dale's Law.

    The network dynamics follow:

    Neuromodulator (z):
        z_{t+1} = (1 - α_z) z_t + α_z (W_zz f(z_t) + B_z x_t)

    Modulation signal:
        s = σ(M z + c)

    Gain mask (rank-R, constrained positive):
        G = softplus(1 + U diag(s - 0.5) V^T)

    Main network (v):
        v_{t+1} = (1 - α) v_t + α ((G ⊙ W_rec) f(v_t) + W_in x_t + b_rec)

    Output:
        y_t = w_out^T f(v_t) + b_out

    Dale's Law is enforced by: W_rec = |W_rec| ⊙ D where D is the sign mask.
    G > 0 via softplus ensures the effective weight preserves sign structure.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dt: float = 20.0,
        tau: float = 100.0,
        tau_z: float = 100.0,
        dim_z: int = 16,
        nm_rank: int = 1,
        activation: str = "tanh",
        noise_std: float = 0.0,
        dale_ratio: Optional[float] = None,
        input_fraction: Optional[float] = None,
        n_input_e: Optional[int] = None,
        n_input_i: Optional[int] = None,
        alpha: Optional[float] = None,
        alpha_nm: Optional[float] = None,
        device: str = "cpu",
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of recurrent units
            dt: Time step for discretization (ms)
            tau: Time constant for main network (ms)
            tau_z: Time constant for neuromodulator (ms)
            dim_z: Number of neuromodulator variables
            nm_rank: Rank of the gain modulation (number of s variables)
            activation: 'relu', 'tanh', 'softplus', or 'gelu'
            noise_std: Standard deviation of noise added to hidden units
            dale_ratio: If set, enforce Dale's law with this fraction of excitatory units
            input_fraction: If set, only this fraction of neurons receive task input
            n_input_e: Number of excitatory neurons receiving input (requires dale_ratio)
            n_input_i: Number of inhibitory neurons receiving input (requires dale_ratio)
            alpha: L2 regularization strength for recurrent weights
            alpha_nm: L2 regularization strength for U, V (gain modulation)
            device: 'cpu' or 'cuda'
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau
        self.tau_z = tau_z
        self.dim_z = dim_z
        self.nm_rank = nm_rank
        self.alpha_rec = alpha if alpha is not None else 0.0
        self.alpha_nm = alpha_nm if alpha_nm is not None else 0.0
        self.noise_std = noise_std
        self.dale_ratio = dale_ratio
        self.input_fraction = input_fraction
        self.n_input_e = n_input_e
        self.n_input_i = n_input_i
        self.device = device

        # Discretization constants
        self.alpha_v = dt / tau  # for main network
        self.alpha_z = dt / tau_z  # for neuromodulator

        # Initialize activation function
        self.activation_name = activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # === Main network parameters (same as RateRNN) ===
        self.w_in = nn.Linear(input_size, hidden_size, bias=False)
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=True)
        self.w_out = nn.Linear(hidden_size, 1, bias=True)

        # Dale's law mask (if enabled)
        if dale_ratio is not None:
            self.register_buffer("dale_mask", self._create_dale_mask())
        else:
            self.dale_mask = None

        # === Neuromodulator parameters ===
        # W_zz: recurrent weights for z (with interactions)
        self.W_zz = nn.Linear(dim_z, dim_z, bias=True)
        # B_z: input weights for z
        self.B_z = nn.Linear(input_size, dim_z, bias=False)
        # M, c: mapping from z to s
        self.M = nn.Parameter(torch.zeros(nm_rank, dim_z))
        self.c = nn.Parameter(torch.zeros(nm_rank))
        # U, V: low-rank factors for gain mask
        self.U = nn.Parameter(torch.zeros(hidden_size, nm_rank))
        self.V = nn.Parameter(torch.zeros(hidden_size, nm_rank))

        # Initialize all weights
        self._initialize_weights()

        # Input mask (if enabled)
        if n_input_e is not None and n_input_i is not None:
            assert dale_ratio is not None, (
                "n_input_e and n_input_i require dale_ratio to be set"
            )
            self.register_buffer("input_mask", self._create_dale_input_mask())
        elif input_fraction is not None:
            self.register_buffer("input_mask", self._create_input_mask())
        else:
            self.input_mask = None

    def _initialize_weights(self, spectral_radius: float = 1.8):
        """Initialize weights with spectral radius control for recurrent weights."""
        # === Main network initialization (same as RateRNN) ===
        nn.init.normal_(self.w_in.weight, mean=0.0, std=1.0 / np.sqrt(self.input_size))

        nn.init.normal_(
            self.w_rec.weight, mean=0.0, std=1.0 / np.sqrt(self.hidden_size)
        )
        with torch.no_grad():
            if self.dale_mask is not None:
                self.w_rec.weight.data = (
                    torch.abs(self.w_rec.weight.data) * self.dale_mask
                )
            eigvals = torch.linalg.eigvals(self.w_rec.weight)
            current_radius = eigvals.abs().max().item()
            if current_radius > 0:
                self.w_rec.weight.mul_(spectral_radius / current_radius)

        nn.init.constant_(self.w_rec.bias, 0.0)
        if self.dale_ratio is not None:
            n_exc = int(self.dale_ratio * self.hidden_size)
            with torch.no_grad():
                self.w_rec.bias[n_exc:] = 1.0

        nn.init.normal_(
            self.w_out.weight, mean=0.0, std=1.0 / np.sqrt(self.hidden_size) * 2
        )
        nn.init.constant_(self.w_out.bias, 0.0)

        # === Neuromodulator initialization ===
        # W_zz: small weights, spectral radius ~1
        nn.init.normal_(self.W_zz.weight, mean=0.0, std=1.0 / np.sqrt(self.dim_z))
        with torch.no_grad():
            eigvals_z = torch.linalg.eigvals(self.W_zz.weight)
            current_radius_z = eigvals_z.abs().max().item()
            if current_radius_z > 0:
                self.W_zz.weight.mul_(1.0 / current_radius_z)
        nn.init.constant_(self.W_zz.bias, 0.0)

        # B_z: same scale as w_in
        nn.init.normal_(self.B_z.weight, mean=0.0, std=1.0 / np.sqrt(self.input_size))

        # M: small Gaussian
        nn.init.normal_(self.M, mean=0.0, std=0.1)

        # c: zeros (so initial s ≈ 0.5)
        nn.init.constant_(self.c, 0.0)

        # U, V: small so initial G ≈ 1
        nn.init.normal_(self.U, mean=0.0, std=0.1 / np.sqrt(self.hidden_size))
        nn.init.normal_(self.V, mean=0.0, std=0.1 / np.sqrt(self.hidden_size))

    def _create_dale_mask(self) -> torch.Tensor:
        """Create mask to enforce Dale's law."""
        assert self.dale_ratio is not None
        n_exc = int(self.dale_ratio * self.hidden_size)
        mask = torch.ones(self.hidden_size, self.hidden_size)
        mask[:, n_exc:] = -1
        return mask

    def _create_input_mask(self) -> torch.Tensor:
        """Create mask so only a fraction of neurons receive task input."""
        assert self.input_fraction is not None
        n_input_neurons = int(self.input_fraction * self.hidden_size)
        mask = torch.zeros(self.hidden_size)
        mask[:n_input_neurons] = 1.0
        return mask

    def _create_dale_input_mask(self) -> torch.Tensor:
        """Create mask so specified numbers of E and I neurons receive task input."""
        assert self.dale_ratio is not None
        assert self.n_input_e is not None and self.n_input_i is not None

        n_exc = int(self.dale_ratio * self.hidden_size)
        n_inh = self.hidden_size - n_exc

        assert self.n_input_e <= n_exc
        assert self.n_input_i <= n_inh

        mask = torch.zeros(self.hidden_size)
        mask[: self.n_input_e] = 1.0
        mask[n_exc : n_exc + self.n_input_i] = 1.0
        return mask

    def get_config(self) -> dict:
        """Return constructor arguments needed to recreate this model."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "dt": self.dt,
            "tau": self.tau,
            "tau_z": self.tau_z,
            "dim_z": self.dim_z,
            "nm_rank": self.nm_rank,
            "activation": self.activation_name,
            "noise_std": self.noise_std,
            "dale_ratio": self.dale_ratio,
            "input_fraction": self.input_fraction,
            "n_input_e": self.n_input_e,
            "n_input_i": self.n_input_i,
            "alpha": self.alpha_rec,
            "alpha_nm": self.alpha_nm,
        }

    def apply_dale_constraint(self):
        """Apply Dale's law constraint to recurrent weights."""
        if self.dale_mask is not None:
            with torch.no_grad():
                self.w_rec.weight.data = (
                    torch.abs(self.w_rec.weight.data) * self.dale_mask
                )

    def compute_modulated_recurrent(
        self, s: torch.Tensor, rates: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficiently compute (G ⊙ W_rec) @ rates without materializing G.

        Uses the identity: (u v^T) ⊙ W @ r = u ⊙ (W @ (v ⊙ r))

        So: ((1 + Σ_k s_k * u_k v_k^T) ⊙ W) @ r
          = W @ r + Σ_k s_k * (u_k ⊙ (W @ (v_k ⊙ r)))

        Args:
            s: Modulation signals of shape (batch, nm_rank)
            rates: Firing rates of shape (batch, hidden_size)

        Returns:
            Modulated recurrent current of shape (batch, hidden_size)
        """
        # Base recurrent current: W_rec @ rates
        base_current = self.w_rec(rates)  # (batch, H) - includes bias

        # Modulation term: Σ_k (s_k - 0.5) * (U[:,k] ⊙ (W @ (V[:,k] ⊙ r)))
        s_centered = s - 0.5  # (batch, nm_rank)

        # For each rank component
        modulation = torch.zeros_like(base_current)
        for k in range(self.nm_rank):
            # v_k ⊙ r: element-wise multiply V[:,k] with rates
            v_r = self.V[:, k] * rates  # (batch, H)
            # W @ (v_k ⊙ r): apply recurrent weights (without bias)
            W_v_r = F.linear(v_r, self.w_rec.weight)  # (batch, H)
            # u_k ⊙ (W @ (v_k ⊙ r))
            u_W_v_r = self.U[:, k] * W_v_r  # (batch, H)
            # Scale by (s_k - 0.5) and accumulate
            modulation = modulation + s_centered[:, k : k + 1] * u_W_v_r

        return base_current + modulation

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_states: bool = False,
        return_neuromod: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Tuple[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        """
        Forward pass through the neuromodulated RNN.

        Args:
            inputs: Input tensor of shape (batch, time, input_size)
            hidden: Initial states as tuple (v, z) where:
                   v: voltage state of shape (batch, hidden_size)
                   z: neuromodulator state of shape (batch, dim_z)
                   If None, both initialized to zeros
            return_states: If True, return firing rate trajectory
            return_neuromod: If True, return z and s trajectories

        Returns:
            outputs: Output time series of shape (batch, time)
            hidden: Final states as tuple (v, z)
            states: (if return_states) Firing rates (batch, time, hidden_size)
            z_traj: (if return_neuromod) z trajectory (batch, time, dim_z)
            s_traj: (if return_neuromod) s trajectory (batch, time, nm_rank)
        """
        batch_size, seq_len, _ = inputs.shape

        # Initialize states if not provided
        if hidden is None:
            v = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
            z = torch.zeros(batch_size, self.dim_z, device=inputs.device)
        else:
            v, z = hidden

        # Store outputs and optional trajectories
        outputs = []
        if return_states:
            states = []
        if return_neuromod:
            z_traj = []
            s_traj = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]  # (batch, input_size)

            # === Update neuromodulator z ===
            z_rates = self.activation(z)  # f(z)
            z_drive = self.W_zz(z_rates) + self.B_z(x_t)
            z = (1 - self.alpha_z) * z + self.alpha_z * z_drive

            # === Compute modulation signal s ===
            s = torch.sigmoid(z @ self.M.T + self.c)  # (batch, nm_rank)

            # === Compute firing rates from voltage ===
            rates = self.activation(v)  # (batch, hidden_size)

            # === Compute modulated recurrent current (efficient, no G materialization) ===
            recurrent_current = self.compute_modulated_recurrent(s, rates)

            # === Compute input current ===
            input_current = self.w_in(x_t)
            if self.input_mask is not None:
                input_current = input_current * self.input_mask

            # === Total drive ===
            drive = input_current + recurrent_current

            # Add noise if specified
            if self.noise_std > 0 and self.training:
                noise = torch.randn_like(drive) * self.noise_std
                drive = drive + noise

            # === Update voltage ===
            v = (1 - self.alpha_v) * v + self.alpha_v * drive

            # === Compute output ===
            rates_new = self.activation(v)
            output_t = self.w_out(rates_new)
            outputs.append(output_t.squeeze(-1))

            if return_states:
                states.append(rates_new)
            if return_neuromod:
                z_traj.append(z)
                s_traj.append(s)

        # Stack outputs
        outputs_tensor = torch.stack(outputs, dim=1)  # (batch, time)

        result = [outputs_tensor, (v, z)]

        if return_states:
            states_tensor = torch.stack(states, dim=1)
            result.append(states_tensor)

        if return_neuromod:
            z_tensor = torch.stack(z_traj, dim=1)  # (batch, time, dim_z)
            s_tensor = torch.stack(s_traj, dim=1)  # (batch, time, nm_rank)
            result.append(z_tensor)
            result.append(s_tensor)

        return tuple(result)

    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss on weights.

        Returns:
            Regularization loss (L2 on W_rec and optionally U, V)
        """
        loss = torch.tensor(0.0, device=self.w_rec.weight.device)

        if self.alpha_rec > 0:
            loss = loss + self.alpha_rec * torch.sum(self.w_rec.weight**2)

        if self.alpha_nm > 0:
            loss = loss + self.alpha_nm * (torch.sum(self.U**2) + torch.sum(self.V**2))

        return loss

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden states.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (v, z) initial states
        """
        v = torch.zeros(batch_size, self.hidden_size, device=self.device)
        z = torch.zeros(batch_size, self.dim_z, device=self.device)
        return v, z


if __name__ == "__main__":
    # Test the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model = NeuromodRNN(
        input_size=14,
        hidden_size=256,
        dt=20.0,
        tau=100.0,
        tau_z=100.0,
        dim_z=16,
        nm_rank=1,
        activation="relu",
        noise_std=0.01,
        dale_ratio=0.8,
        n_input_e=32,
        n_input_i=32,
        alpha=1e-5,
        alpha_nm=1e-5,
        device=device,
    ).to(device)

    print(f"\nModel has {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 4
    seq_len = 50
    dummy_input = torch.randn(batch_size, seq_len, 14).to(device)

    print("\nTesting forward pass...")
    outputs, (v_final, z_final) = model(dummy_input)
    print(f"Output shape: {outputs.shape}")
    print(f"Final v shape: {v_final.shape}")
    print(f"Final z shape: {z_final.shape}")
    print(f"Output contains NaN: {torch.isnan(outputs).any().item()}")

    # Test with return_states and return_neuromod
    print("\nTesting with return_states and return_neuromod...")
    outputs, hidden, states, z_traj, s_traj = model(
        dummy_input, return_states=True, return_neuromod=True
    )
    print(f"States shape: {states.shape}")
    print(f"z trajectory shape: {z_traj.shape}")
    print(f"s trajectory shape: {s_traj.shape}")

    # Test modulation at s = 0.5 (should be ~identity)
    print("\nTesting modulation at s = 0.5 (should be ~identity)...")
    s_test = torch.ones(1, model.nm_rank, device=device) * 0.5
    rates_test = torch.randn(1, model.hidden_size, device=device)

    # Modulated current
    mod_current = model.compute_modulated_recurrent(s_test, rates_test)
    # Unmodulated current (just W_rec @ rates)
    unmod_current = model.w_rec(rates_test)

    diff = (mod_current - unmod_current).abs().mean().item()
    print(f"Mean abs difference from unmodulated: {diff:.6f} (should be ~0)")

    # Test regularization loss
    print(f"\nRegularization loss: {model.compute_regularization_loss().item():.6f}")

    print("\nAll tests passed!")
