#!/bin/bash
# Quick run script for training

# Activate virtual environment
source venv/bin/activate

# Check GPU
echo "Checking GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run training
echo "Starting training..."
python train.py --config config.yaml

# Or run with specific lambda:
# python train.py --config config.yaml --temporal_reg_lambda 0.01
