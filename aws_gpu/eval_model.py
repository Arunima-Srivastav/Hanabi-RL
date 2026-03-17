"""
Evaluate a saved model and generate metrics/plots.

Usage:
    python eval_model.py --checkpoint experiments/run_XXX/best_model.pt
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path

from env.hanabi_env import TextHanabiEnv
from agent import R2D2Agent

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False


def evaluate_detailed(env, agent, num_games, device):
    """Run detailed evaluation with attention tracking."""
    scores = []
    game_lengths = []
    all_attention = []
    temporal_kl_per_game = []
    
    print(f"Running {num_games} evaluation games...")
    
    for game_idx in range(num_games):
        state = env.reset()
        hid = agent.get_h0(1)
        game_attention = []
        steps = 0
        
        while not state.terminal:
            section_texts = env.get_text_state(state.current_player)
            section_emb = agent.online_net.encode_sections_from_strings({k: [v] for k, v in section_texts.items()})
            
            legal_actions = env.get_legal_actions()
            legal_move = torch.zeros(1, env.num_actions(), device=device)
            for a in legal_actions:
                legal_move[0, a] = 1.0
            
            action, hid, attention = agent.act(section_emb, legal_move, hid, epsilon=0.0)
            game_attention.append(attention.cpu().numpy().flatten())
            
            state, _, _ = env.step(action.item())
            steps += 1
        
        scores.append(state.score)
        game_lengths.append(steps)
        
        # Compute temporal KL for this game
        if len(game_attention) > 1:
            game_attention = np.array(game_attention)
            all_attention.append(np.mean(game_attention, axis=0))
            
            game_kl = []
            for t in range(len(game_attention) - 1):
                p, q = game_attention[t], game_attention[t + 1]
                kl = np.sum(p * np.log((p + 1e-8) / (q + 1e-8))) + np.sum(q * np.log((q + 1e-8) / (p + 1e-8)))
                game_kl.append(kl)
            temporal_kl_per_game.append(np.mean(game_kl))
        
        if (game_idx + 1) % 20 == 0:
            print(f"  Completed {game_idx + 1}/{num_games} games...")
    
    return {
        "scores": scores,
        "game_lengths": game_lengths,
        "attention_weights": np.array(all_attention) if all_attention else None,
        "temporal_kl": temporal_kl_per_game,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num_games", type=int, default=100, help="Number of eval games")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    output_dir = checkpoint_path.parent
    
    # Detect device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"Using device: {args.device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    
    # Create environment
    env = TextHanabiEnv(num_players=2, hand_size=5, seed=42)
    
    # Create agent with default params (we'll load weights)
    agent = R2D2Agent(
        device=args.device,
        hidden_dim=512,
        num_actions=env.num_actions(),
        num_lstm_layers=2,
        gamma=0.99,
        multi_step=1,
        temporal_reg_lambda=0.01,
    )
    
    # Load weights
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    print("Model loaded successfully!")
    
    # Run evaluation
    results = evaluate_detailed(env, agent, args.num_games, args.device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Games played: {args.num_games}")
    print(f"Mean score: {np.mean(results['scores']):.2f} ± {np.std(results['scores']):.2f}")
    print(f"Max score: {max(results['scores'])}")
    print(f"Min score: {min(results['scores'])}")
    print(f"Perfect games (25): {sum(s == 25 for s in results['scores'])} ({100*sum(s == 25 for s in results['scores'])/len(results['scores']):.1f}%)")
    print(f"Mean game length: {np.mean(results['game_lengths']):.1f} steps")
    print(f"Mean temporal KL: {np.mean(results['temporal_kl']):.4f}")
    
    # Section attention
    if results['attention_weights'] is not None:
        print("\nSection Attention Weights (avg):")
        section_names = ["Life", "Hints", "Fireworks", "Own Hand", "Opp Hand", "Discards", "Last Act"]
        avg_attn = np.mean(results['attention_weights'], axis=0)
        for name, weight in zip(section_names, avg_attn):
            bar = "█" * int(weight * 40)
            print(f"  {name:12s}: {weight:.3f} {bar}")
    
    # Save results
    eval_results = {
        "num_games": args.num_games,
        "mean_score": float(np.mean(results['scores'])),
        "std_score": float(np.std(results['scores'])),
        "max_score": int(max(results['scores'])),
        "min_score": int(min(results['scores'])),
        "perfect_rate": float(sum(s == 25 for s in results['scores']) / len(results['scores'])),
        "mean_game_length": float(np.mean(results['game_lengths'])),
        "mean_temporal_kl": float(np.mean(results['temporal_kl'])),
        "scores": results['scores'],
    }
    
    eval_path = output_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to: {eval_path}")
    
    # Generate plots if matplotlib available
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Score distribution
        ax = axes[0]
        ax.hist(results['scores'], bins=range(0, 27), edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(results['scores']), color='red', linestyle='--', label=f"Mean: {np.mean(results['scores']):.1f}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution")
        ax.legend()
        
        # Attention weights
        ax = axes[1]
        if results['attention_weights'] is not None:
            avg_attn = np.mean(results['attention_weights'], axis=0)
            colors = plt.cm.Set3(np.linspace(0, 1, 7))
            ax.bar(section_names, avg_attn, color=colors)
            ax.set_ylabel("Attention Weight")
            ax.set_title("Section Attention Distribution")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Temporal KL distribution
        ax = axes[2]
        ax.hist(results['temporal_kl'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax.axvline(np.mean(results['temporal_kl']), color='red', linestyle='--', label=f"Mean: {np.mean(results['temporal_kl']):.4f}")
        ax.set_xlabel("Temporal KL")
        ax.set_ylabel("Count")
        ax.set_title("Attention Smoothness (per game)")
        ax.legend()
        
        plt.tight_layout()
        plot_path = output_dir / "eval_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plots saved to: {plot_path}")


if __name__ == "__main__":
    main()
