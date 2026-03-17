# Hanabi Temporal KL Regularization

Pure Python implementation of Hanabi RL with temporal attention consistency regularization.

## Overview

This project implements the **temporal KL divergence regularization** for training Hanabi agents. The key idea is that since Hanabi game states change incrementally between turns, the model's internal attention (which sections of the game state it focuses on) should also change smoothly.

### The 7 Game State Sections

1. **Life tokens** - Remaining mistakes allowed
2. **Hint tokens** - Remaining hints available  
3. **Fireworks** - Progress on each color stack
4. **Own hand** - What the agent knows about its own cards (from hints)
5. **Opponent's hand** - Full information about partner's cards
6. **Discards** - Cards that have been discarded
7. **Last action** - Most recent move in the game

### Temporal KL Loss

The regularization loss encourages smooth attention changes:

```
L_temporal = E_t [ D_sym(A(s_t), A(s_{t+1})) ]

where D_sym(p, q) = KL(p || q) + KL(q || p)
```

### Training Objective

```
L = L_RL + λ * L_temporal
```

Where:
- `L_RL` is the standard TD error (Smooth L1 loss)
- `L_temporal` is the temporal attention consistency penalty
- `λ` (temporal_reg_lambda) controls regularization strength

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies (no CMake/pybind needed!):**
- PyTorch >= 2.0
- Transformers >= 4.30
- NumPy
- PyYAML
- tqdm

## Usage

### Single Training Run

```bash
python train.py --config config.yaml
```

### Override Lambda (for quick testing)

```bash
python train.py --config config.yaml --temporal_reg_lambda 0.1
```

### Run Ablation Study

```bash
# Default: test λ ∈ {0.0, 0.001, 0.01, 0.1, 1.0} with 3 seeds each
python run_ablation.py

# Custom values
python run_ablation.py --lambdas 0.0 0.05 0.1 --seeds 42 43 44 45
```

## Project Structure

```
nima-work/
├── env/
│   ├── __init__.py
│   └── hanabi_env.py        # Pure Python Hanabi environment
├── networks/
│   ├── __init__.py
│   └── q_net.py             # Network with section attention
├── losses/
│   ├── __init__.py
│   └── temporal_kl.py       # Temporal KL divergence loss
├── agent.py                 # R2D2 agent with temporal regularization
├── replay_buffer.py         # Pure Python replay buffer
├── train.py                 # Main training script
├── run_ablation.py          # Ablation study runner
├── config.yaml              # Default configuration
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Configuration

Key hyperparameters in `config.yaml`:

```yaml
# THE KEY HYPERPARAMETER FOR ABLATION
temporal_reg_lambda: 0.01  # Try: 0.0, 0.001, 0.01, 0.1, 1.0

# Model
model:
  hidden_dim: 512
  num_lstm_layers: 2
  pretrained_model: "cross-encoder/ms-marco-TinyBERT-L-2-v2"

# RL
rl:
  gamma: 0.99
  multi_step: 1
```

## Key Files

### `losses/temporal_kl.py`

The core contribution - implements symmetric KL divergence between consecutive timesteps:

```python
def temporal_kl_loss(attention_weights, mask=None, epsilon=1e-8):
    """
    attention_weights: [seq_len, batch, num_sections]
    Returns: scalar loss
    """
    attn_t = attention_weights[:-1]   # t
    attn_t1 = attention_weights[1:]   # t+1
    
    # Symmetric KL
    sym_kl = KL(attn_t || attn_t1) + KL(attn_t1 || attn_t)
    
    return sym_kl.mean()
```

### `networks/q_net.py`

Modified TextLSTMNet that:
1. Encodes each of the 7 sections separately using TinyBERT
2. Uses learned attention over sections
3. Returns attention weights for temporal KL loss

### `agent.py`

R2D2 agent with the combined loss:

```python
def loss(self, batch):
    td_error, attention_weights = self.compute_td_error(batch)
    
    rl_loss = smooth_l1_loss(td_error)
    temporal_loss = temporal_kl_loss(attention_weights)
    
    total_loss = rl_loss + self.temporal_reg_lambda * temporal_loss
    return total_loss
```

## Ablation Study Design

To study the effect of temporal KL regularization:

1. **Baseline (λ=0)**: No regularization
2. **Weak (λ=0.001)**: Minimal smoothing
3. **Medium (λ=0.01)**: Moderate smoothing  
4. **Strong (λ=0.1)**: Strong smoothing
5. **Very Strong (λ=1.0)**: Dominant smoothing term

For each λ, run 3+ seeds and report:
- Final score (mean ± std)
- Attention entropy over time
- Cross-play performance (if testing with different partners)

## Differences from Original Codebase

| Aspect | Original | This Implementation |
|--------|----------|-------------------|
| Hanabi env | C++ (hanalearn) | Pure Python |
| Replay buffer | C++ (rela) | Pure Python |
| Training loop | C++ threads | Pure Python |
| Network | TextLSTMNet | TextLSTMNetWithSectionAttention |
| Loss | RL only | RL + temporal KL |
| Dependencies | CMake, pybind11 | None (pure Python) |

## Notes

- Training is slower than the C++ version but requires no compilation
- The section attention mechanism makes attention weights explicit and interpretable
- Temporal KL loss is computed efficiently over batched trajectories
