"""
Training script for Hanabi with Temporal KL Regularization.

This script trains an R2D2 agent with section attention and temporal KL
regularization. The key hyperparameter for ablation studies is `temporal_reg_lambda`.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --temporal_reg_lambda 0.1
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from env.hanabi_env import TextHanabiEnv
from networks.q_net import TextLSTMNetWithSectionAttention, SECTION_NAMES
from agent import R2D2Agent
from replay_buffer import ReplayBuffer, TrajectoryCollector, Trajectory


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_trajectory(
    env: TextHanabiEnv,
    agent: R2D2Agent,
    epsilon: float,
    device: str,
) -> Trajectory:
    """
    Collect a single game trajectory.
    
    Returns a Trajectory object containing section embeddings, actions, rewards, etc.
    """
    state = env.reset()
    
    collector = TrajectoryCollector(
        num_sections=7,
        hidden_dim=agent.online_net.hidden_dim,
        num_actions=env.num_actions(),
        max_len=200,
    )
    
    hid = agent.get_h0(1)
    total_reward = 0
    
    while not state.terminal:
        observer = state.current_player
        
        section_texts = env.get_text_state(observer)
        section_emb = agent.online_net.encode_sections_from_strings(
            {k: [v] for k, v in section_texts.items()}
        )  # [1, num_sections, hidden_dim]
        
        legal_actions = env.get_legal_actions()
        legal_move = torch.zeros(1, env.num_actions(), device=device)
        for a in legal_actions:
            legal_move[0, a] = 1.0
        
        action, hid, _ = agent.act(section_emb, legal_move, hid, epsilon=epsilon)
        action_idx = action.item()
        
        state, reward, done = env.step(action_idx)
        total_reward += reward
        
        collector.add(
            section_embedding=section_emb.squeeze(0),
            legal_move=legal_move.squeeze(0),
            action=action_idx,
            reward=reward,
            terminal=done,
        )
    
    return collector.get_trajectory()


def evaluate(
    env: TextHanabiEnv,
    agent: R2D2Agent,
    num_games: int,
    device: str,
) -> dict:
    """
    Evaluate agent performance.
    
    Returns dict with average score, perfect games, etc.
    """
    scores = []
    perfect = 0
    
    for _ in range(num_games):
        state = env.reset()
        hid = agent.get_h0(1)
        
        while not state.terminal:
            observer = state.current_player
            section_texts = env.get_text_state(observer)
            section_emb = agent.online_net.encode_sections_from_strings(
                {k: [v] for k, v in section_texts.items()}
            )
            
            legal_actions = env.get_legal_actions()
            legal_move = torch.zeros(1, env.num_actions(), device=device)
            for a in legal_actions:
                legal_move[0, a] = 1.0
            
            action, hid, _ = agent.act(section_emb, legal_move, hid, epsilon=0.0)
            state, _, _ = env.step(action.item())
        
        scores.append(state.score)
        if state.score == 25:
            perfect += 1
    
    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "perfect_rate": perfect / num_games,
        "max_score": max(scores),
        "min_score": min(scores),
    }


def train(config: dict, args):
    """Main training loop."""
    
    if args.temporal_reg_lambda is not None:
        config["temporal_reg_lambda"] = args.temporal_reg_lambda
    
    # Auto-detect device
    device_config = config.get("device", "auto")
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    print(f"Using device: {device}")
    seed = config.get("seed", 42)
    set_seed(seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config["logging"]["save_dir"]) / f"run_{timestamp}_lambda_{config['temporal_reg_lambda']}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    print(f"Saving to: {save_dir}")
    print(f"Temporal regularization lambda: {config['temporal_reg_lambda']}")
    
    env = TextHanabiEnv(
        num_players=config["env"]["num_players"],
        hand_size=config["env"]["hand_size"],
        num_colors=config["env"]["num_colors"],
        num_ranks=config["env"]["num_ranks"],
        max_hint_tokens=config["env"]["max_hint_tokens"],
        max_life_tokens=config["env"]["max_life_tokens"],
        seed=seed,
    )
    
    agent = R2D2Agent(
        device=device,
        hidden_dim=config["model"]["hidden_dim"],
        num_actions=env.num_actions(),
        num_lstm_layers=config["model"]["num_lstm_layers"],
        gamma=config["rl"]["gamma"],
        multi_step=config["rl"]["multi_step"],
        temporal_reg_lambda=config["temporal_reg_lambda"],
        pretrained_model=config["model"]["pretrained_model"],
        freeze_bert=config["model"]["freeze_bert"],
    )
    
    optimizer = torch.optim.Adam(
        agent.online_net.parameters(),
        lr=config["training"]["lr"],
        eps=config["training"]["eps"],
    )
    
    replay_buffer = ReplayBuffer(
        capacity=config["replay"]["buffer_size"],
        seed=seed,
    )
    
    print("Collecting initial trajectories...")
    epsilon = config["collect"]["epsilon_start"]
    while replay_buffer.size() < config["replay"]["burn_in_frames"]:
        traj = collect_trajectory(env, agent, epsilon, device)
        if traj is not None:
            replay_buffer.add(traj)
        print(f"  Buffer size: {replay_buffer.size()}", end="\r")
    print(f"\nBuffer warmed up with {replay_buffer.size()} trajectories")
    
    stats_history = []
    best_score = 0
    
    for epoch in range(config["training"]["num_epochs"]):
        epoch_start = time.time()
        epoch_stats = {
            "rl_loss": [],
            "temporal_loss": [],
            "total_loss": [],
        }
        
        decay_progress = min(1.0, epoch / config["collect"]["epsilon_decay_epochs"])
        epsilon = config["collect"]["epsilon_start"] + \
            (config["collect"]["epsilon_end"] - config["collect"]["epsilon_start"]) * decay_progress
        
        print(f"Epoch {epoch + 1}: Collecting trajectories...", end=" ", flush=True)
        for _ in range(config["collect"]["num_games_per_epoch"]):
            traj = collect_trajectory(env, agent, epsilon, device)
            if traj is not None:
                replay_buffer.add(traj)
        print("done. Training...", end=" ", flush=True)
        
        for batch_idx in range(config["training"]["epoch_len"]):
            batch = replay_buffer.sample(config["training"]["batch_size"], device)
            
            loss, stats = agent.loss_from_batch(batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.online_net.parameters(),
                config["training"]["grad_clip"],
            )
            optimizer.step()
            
            for k, v in stats.items():
                epoch_stats[k].append(v)
        
        epoch_time = time.time() - epoch_start
        print(f"done. ({epoch_time:.1f}s)")
        
        if (epoch + 1) % config["training"]["target_sync_freq"] == 0:
            agent.sync_target_with_online()
        
        avg_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
        stats_history.append(avg_stats)
        
        if (epoch + 1) % config["logging"]["log_freq"] == 0:
            eval_results = evaluate(env, agent, num_games=20, device=device)
            
            print(f"Epoch {epoch + 1:4d} | "
                  f"RL Loss: {avg_stats['rl_loss']:.4f} | "
                  f"Temporal Loss: {avg_stats['temporal_loss']:.4f} | "
                  f"Score: {eval_results['mean_score']:.2f} +/- {eval_results['std_score']:.2f} | "
                  f"Perfect: {eval_results['perfect_rate']*100:.1f}% | "
                  f"Epsilon: {epsilon:.3f}")
            
            if eval_results["mean_score"] > best_score:
                best_score = eval_results["mean_score"]
                torch.save({
                    "epoch": epoch,
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_results": eval_results,
                    "config": config,
                }, save_dir / "best_model.pt")
        
        if (epoch + 1) % config["logging"]["save_freq"] == 0:
            torch.save({
                "epoch": epoch,
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stats_history": stats_history,
                "config": config,
            }, save_dir / f"checkpoint_epoch_{epoch + 1}.pt")
    
    print(f"\nTraining complete. Best score: {best_score:.2f}")
    print(f"Results saved to: {save_dir}")
    
    return stats_history


def main():
    parser = argparse.ArgumentParser(description="Train Hanabi agent with temporal KL regularization")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--temporal_reg_lambda", type=float, default=None,
                        help="Override temporal_reg_lambda from config (for ablation studies)")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.device and args.device != "auto":
        config["device"] = args.device
    if args.seed:
        config["seed"] = args.seed
    
    train(config, args)


if __name__ == "__main__":
    main()
