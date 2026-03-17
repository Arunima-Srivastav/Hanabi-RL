"""
Comprehensive logging and visualization for Hanabi Temporal KL training.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class TrainingLogger:
    """Logs training metrics and generates visualizations."""
    
    def __init__(self, save_dir: str, config: dict):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)
        self.attention_stats = defaultdict(list)
        
        # Save config
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def log_epoch(
        self,
        epoch: int,
        rl_loss: float,
        temporal_loss: float,
        total_loss: float,
        epsilon: float,
    ):
        """Log training metrics for an epoch."""
        self.metrics["epoch"].append(epoch)
        self.metrics["rl_loss"].append(rl_loss)
        self.metrics["temporal_loss"].append(temporal_loss)
        self.metrics["total_loss"].append(total_loss)
        self.metrics["epsilon"].append(epsilon)
    
    def log_eval(
        self,
        epoch: int,
        mean_score: float,
        std_score: float,
        perfect_rate: float,
        attention_entropy: Optional[float] = None,
        attention_smoothness: Optional[float] = None,
    ):
        """Log evaluation metrics."""
        self.eval_metrics["epoch"].append(epoch)
        self.eval_metrics["mean_score"].append(mean_score)
        self.eval_metrics["std_score"].append(std_score)
        self.eval_metrics["perfect_rate"].append(perfect_rate)
        if attention_entropy is not None:
            self.eval_metrics["attention_entropy"].append(attention_entropy)
        if attention_smoothness is not None:
            self.eval_metrics["attention_smoothness"].append(attention_smoothness)
    
    def log_attention(
        self,
        epoch: int,
        section_weights: np.ndarray,
        temporal_kl_values: List[float],
    ):
        """Log attention statistics."""
        self.attention_stats["epoch"].append(epoch)
        self.attention_stats["section_weights"].append(section_weights.tolist())
        self.attention_stats["temporal_kl_mean"].append(np.mean(temporal_kl_values))
        self.attention_stats["temporal_kl_std"].append(np.std(temporal_kl_values))
        
        # Compute entropy of attention distribution
        entropy = -np.sum(section_weights * np.log(section_weights + 1e-8))
        self.attention_stats["attention_entropy"].append(entropy)
    
    def save_metrics(self):
        """Save all metrics to JSON."""
        all_metrics = {
            "training": dict(self.metrics),
            "evaluation": dict(self.eval_metrics),
            "attention": dict(self.attention_stats),
        }
        with open(self.save_dir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
    
    def plot_training_curves(self):
        """Generate training visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Training Analysis (λ={self.config.get('temporal_reg_lambda', 'N/A')})", fontsize=14)
        
        epochs = self.metrics["epoch"]
        
        # 1. Training Loss Evolution
        ax = axes[0, 0]
        ax.plot(epochs, self.metrics["rl_loss"], label="RL Loss", color="blue")
        ax.plot(epochs, self.metrics["temporal_loss"], label="Temporal KL Loss", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Evaluation Score Progression
        ax = axes[0, 1]
        eval_epochs = self.eval_metrics["epoch"]
        scores = self.eval_metrics["mean_score"]
        stds = self.eval_metrics["std_score"]
        ax.plot(eval_epochs, scores, color="green", marker="o", markersize=3)
        ax.fill_between(eval_epochs, 
                        np.array(scores) - np.array(stds),
                        np.array(scores) + np.array(stds),
                        alpha=0.3, color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Score")
        ax.set_title("Evaluation Score Progression")
        ax.set_ylim(0, 25)
        ax.grid(True, alpha=0.3)
        
        # 3. Temporal KL Loss Evolution
        ax = axes[0, 2]
        ax.plot(epochs, self.metrics["temporal_loss"], color="red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Temporal KL Loss")
        ax.set_title("Attention Smoothness (Lower = Smoother)")
        ax.grid(True, alpha=0.3)
        
        # 4. Epsilon Decay
        ax = axes[1, 0]
        ax.plot(epochs, self.metrics["epsilon"], color="purple")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Epsilon")
        ax.set_title("Exploration Rate Decay")
        ax.grid(True, alpha=0.3)
        
        # 5. Perfect Game Rate
        ax = axes[1, 1]
        if self.eval_metrics["perfect_rate"]:
            ax.plot(eval_epochs, [r * 100 for r in self.eval_metrics["perfect_rate"]], 
                   color="gold", marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perfect Rate (%)")
        ax.set_title("Perfect Game Rate")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # 6. Attention Entropy (if logged)
        ax = axes[1, 2]
        if self.attention_stats["attention_entropy"]:
            attn_epochs = self.attention_stats["epoch"]
            ax.plot(attn_epochs, self.attention_stats["attention_entropy"], color="teal")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Attention Entropy")
        ax.set_title("Attention Distribution Entropy")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved training curves to {self.save_dir / 'training_curves.png'}")
    
    def plot_section_attention(self):
        """Plot attention weights per section over training."""
        if not self.attention_stats["section_weights"]:
            return
        
        section_names = ["Life", "Hints", "Fireworks", "Own Hand", "Opp Hand", "Discards", "Last Act"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        weights = np.array(self.attention_stats["section_weights"])
        epochs = self.attention_stats["epoch"]
        
        for i, name in enumerate(section_names):
            ax.plot(epochs, weights[:, i], label=name, marker="o", markersize=2)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Attention Weight")
        ax.set_title("Section Attention Weights Over Training")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "section_attention.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved section attention plot to {self.save_dir / 'section_attention.png'}")
    
    def plot_final_summary(self):
        """Generate final summary visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Final attention distribution (pie chart)
        ax = axes[0]
        if self.attention_stats["section_weights"]:
            final_weights = self.attention_stats["section_weights"][-1]
            section_names = ["Life", "Hints", "Fireworks", "Own Hand", "Opp Hand", "Discards", "Last Act"]
            colors = plt.cm.Set3(np.linspace(0, 1, 7))
            ax.pie(final_weights, labels=section_names, autopct='%1.1f%%', colors=colors)
            ax.set_title("Final Attention Distribution")
        
        # Score distribution (histogram from last few evaluations)
        ax = axes[1]
        if len(self.eval_metrics["mean_score"]) > 5:
            recent_scores = self.eval_metrics["mean_score"][-10:]
            ax.bar(range(len(recent_scores)), recent_scores, color="steelblue")
            ax.set_xlabel("Recent Evaluations")
            ax.set_ylabel("Score")
            ax.set_title("Recent Evaluation Scores")
            ax.set_ylim(0, 25)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "final_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved final summary to {self.save_dir / 'final_summary.png'}")
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        self.save_metrics()
        self.plot_training_curves()
        self.plot_section_attention()
        self.plot_final_summary()
        print(f"\nAll plots saved to: {self.save_dir}")


def compute_attention_entropy(attention_weights: np.ndarray) -> float:
    """Compute entropy of attention distribution."""
    return -np.sum(attention_weights * np.log(attention_weights + 1e-8))


def compute_attention_smoothness(attention_sequence: np.ndarray) -> float:
    """
    Compute smoothness of attention over a sequence.
    Lower values = smoother (less change between timesteps).
    """
    if len(attention_sequence) < 2:
        return 0.0
    
    diffs = np.diff(attention_sequence, axis=0)
    return np.mean(np.abs(diffs))
