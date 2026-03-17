"""
Run ablation study over different temporal_reg_lambda values.

Usage:
    python run_ablation.py
    python run_ablation.py --lambdas 0.0 0.01 0.1 --seeds 42 43 44
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_ablation(lambdas: list, seeds: list, config_path: str, device: str):
    """Run training for each combination of lambda and seed."""
    
    results_dir = Path("ablation_results")
    results_dir.mkdir(exist_ok=True)
    
    experiments = []
    for lam in lambdas:
        for seed in seeds:
            experiments.append((lam, seed))
    
    print(f"Running {len(experiments)} experiments:")
    print(f"  Lambda values: {lambdas}")
    print(f"  Seeds: {seeds}")
    print()
    
    for i, (lam, seed) in enumerate(experiments):
        print(f"[{i+1}/{len(experiments)}] Running lambda={lam}, seed={seed}")
        
        cmd = [
            sys.executable, "train.py",
            "--config", config_path,
            "--temporal_reg_lambda", str(lam),
            "--seed", str(seed),
            "--device", device,
        ]
        
        log_file = results_dir / f"log_lambda_{lam}_seed_{seed}.txt"
        
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        if result.returncode != 0:
            print(f"  WARNING: Experiment failed with return code {result.returncode}")
        else:
            print(f"  Completed. Log saved to {log_file}")
        
        print()
    
    print("Ablation study complete!")
    print(f"Results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run temporal KL ablation study")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0, 0.001, 0.01, 0.1, 1.0],
                        help="Lambda values to test")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44],
                        help="Random seeds for each lambda")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to base config file")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    run_ablation(args.lambdas, args.seeds, args.config, args.device)


if __name__ == "__main__":
    main()
