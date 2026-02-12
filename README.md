# modRNN: Biologically-Constrained RNNs for Cognitive Tasks

Recurrent neural networks with biological constraints (Dale's Law, neuromodulation) trained to solve evidence accumulation and delayed match-to-sample tasks.

## Installation

```bash
uv sync
```

## Python Files

| File | Purpose |
|------|---------|
| `model.py` | **RateRNN** - Voltage-based RNN with Dale's Law (E/I populations) |
| `neuromod_model.py` | **NeuromodRNN** - Extends RateRNN with low-rank gain modulation |
| `loader.py` | **AccumulationTaskDataset** - Towers task from Pinto et al. (2018) |
| `checkerboard.py` | **DelayedMatchToEvidenceDataset** - Checkerboard task from Costa et al. (2025) |
| `train.py` | Trainer for RateRNN on accumulation task |
| `train_checkerboard.py` | Trainer for RateRNN on checkerboard task (with curriculum) |
| `train_neuromod_checkerboard.py` | Trainer for NeuromodRNN on checkerboard task |
| `plot_weights.py` | Weight matrix visualization with spectral/hierarchical clustering |
| `plot_summary.py` | 2x2 summary figure (training metrics + neural dynamics) |

## Training

**Accumulation task (RateRNN):**
```bash
uv run python train.py
```

**Checkerboard task (RateRNN):**
```bash
uv run python train_checkerboard.py
```

**Checkerboard task (NeuromodRNN):**
```bash
uv run python train_neuromod_checkerboard.py
```

Checkpoints saved to `./checkpoints/` or `./checkpoints_neuromod/`.

## Evaluation and Visualization

**Weight matrices:**
```bash
uv run python plot_weights.py
```

**Summary plot (metrics + neural activity):**
```bash
uv run python plot_summary.py
```

**Task visualization:**
```bash
uv run python loader.py       # Accumulation task samples
uv run python checkerboard.py # Checkerboard task samples
```

Plots saved to `./plots/`.

## Model Architecture

**RateRNN:**
```
Input → W_in → Recurrent: v = (1-α)v + α(W_rec·f(v) + W_in·x + b) → W_out → Output
```

**NeuromodRNN** (adds gain modulation):
```
Input → W_in, B_z
         ↓
Neuromodulator: z = (1-α_z)z + α_z(W_zz·f(z) + B_z·x)
         ↓
Modulation: s = σ(M·z + c)
         ↓
Gain: G = softplus(1 + U·diag(s-0.5)·V^T)   [low-rank]
         ↓
Main: v = (1-α)v + α((G⊙W_rec)·f(v) + W_in·x + b)
         ↓
Output: y = W_out·f(v) + b
```

## Biological Constraints

- **Dale's Law**: `dale_ratio` sets fraction of excitatory neurons (e.g., 0.8 = 80% E, 20% I)
- **Sparse input**: `input_fraction` or `n_input_e`/`n_input_i` control which neurons receive task input
- **Neuromodulation**: `nm_rank` sets rank of gain modulation (typically 1-2)

## Output Structure

```
modRNN/
├── checkpoints/           # RateRNN training outputs
├── checkpoints_neuromod/  # NeuromodRNN training outputs
└── plots/                 # Generated figures
```

## References

- [Pinto et al. (2018)](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2018.00036/full) - Accumulation task
- [Costa et al. (2025)](https://www.biorxiv.org/content/10.1101/2025.07.24.666672v1) - Delayed match-to-evidence checkerboard task
