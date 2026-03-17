#!/bin/bash
# Run baseline (no temporal regularization) for comparison

source venv/bin/activate

echo "Running BASELINE (lambda=0.0)..."
python train.py --config config.yaml --temporal_reg_lambda 0.0
