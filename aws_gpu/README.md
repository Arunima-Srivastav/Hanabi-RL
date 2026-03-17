# Hanabi Temporal KL Training - AWS GPU Version

Self-contained package for running Hanabi RL training with temporal KL regularization on AWS EC2 GPU instances.

## Quick Start on AWS EC2

### 1. Upload this folder to your EC2 instance

```bash
scp -r aws_gpu/ ubuntu@your-ec2-ip:~/
```

### 2. SSH into your instance

```bash
ssh ubuntu@your-ec2-ip
cd aws_gpu
```

### 3. Run setup

```bash
chmod +x setup.sh run.sh run_baseline.sh
./setup.sh
```

### 4. Start training

```bash
./run.sh
```

Or run baseline for comparison:
```bash
./run_baseline.sh
```

## Files

```
aws_gpu/
├── setup.sh              # Install dependencies (run once)
├── run.sh                # Run training with temporal KL (λ=0.01)
├── run_baseline.sh       # Run baseline (λ=0.0)
├── config.yaml           # Configuration
├── requirements.txt      # Python dependencies
├── train.py              # Training script
├── agent.py              # R2D2 agent with temporal KL
├── replay_buffer.py      # Replay buffer
├── env/                  # Hanabi environment
├── networks/             # Q-network with section attention
└── losses/               # Temporal KL loss
```

## Key Hyperparameter

In `config.yaml`:
```yaml
temporal_reg_lambda: 0.01  # λ for temporal KL regularization
```

- `λ = 0.0` → Baseline (no regularization)
- `λ = 0.01` → Recommended (smooth attention changes)

## GPU Config (optimized for g4dn.xlarge or similar)

- Batch size: 64
- Burn-in: 1000 trajectories
- Buffer size: 20000 trajectories

## Monitor Training

Training logs print every 10 epochs:
```
Epoch  10 | RL: 0.1234 | Temporal: 0.0567 | Score: 12.3±3.2 | Perfect: 4.0%
```

Checkpoints saved to `experiments/run_TIMESTAMP_lambda_0.01/`
