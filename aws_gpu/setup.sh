#!/bin/bash
# Setup script for AWS EC2 GPU instance
# Run this after SSHing into your instance

set -e

echo "=== Hanabi Temporal KL Training Setup ==="

# Update system
echo "Updating system packages..."
sudo apt-get update -y

# Install Python if needed
if ! command -v python3 &> /dev/null; then
    echo "Installing Python..."
    sudo apt-get install -y python3 python3-pip python3-venv
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify CUDA
echo ""
echo "=== Verifying CUDA ==="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Setup Complete ==="
echo "To start training, run:"
echo "  source venv/bin/activate"
echo "  python train.py"
