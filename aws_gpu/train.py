"""
Training script for Hanabi with Temporal KL Regularization (GPU-optimized).

Usage:
    python train.py
    python train.py --temporal_reg_lambda 0.1
"""

import os
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

from env.hanabi_env import TextHanabiEnv
from agent import R2D2Agent
from replay_buffer import ReplayBuffer, TrajectoryCollector, Trajectory
from logger import TrainingLogger, compute_attention_entropy


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_trajectory(env, agent, epsilon, device):
    state = env.reset()
    collector = TrajectoryCollector(7, agent.online_net.hidden_dim, env.num_actions(), 200)
    hid = agent.get_h0(1)
    
    while not state.terminal:
        observer = state.current_player
        section_texts = env.get_text_state(observer)
        section_emb = agent.online_net.encode_sections_from_strings({k: [v] for k, v in section_texts.items()})
        
        legal_actions = env.get_legal_actions()
        legal_move = torch.zeros(1, env.num_actions(), device=device)
        for a in legal_actions:
            legal_move[0, a] = 1.0
        
        action, hid, _ = agent.act(section_emb, legal_move, hid, epsilon=epsilon)
        action_idx = action.item()
        state, reward, done = env.step(action_idx)
        
        collector.add(section_emb.squeeze(0), legal_move.squeeze(0), action_idx, reward, done)
    
    return collector.get_trajectory()


def evaluate(env, agent, num_games, device):
    """Evaluate agent and collect attention statistics."""
    scores = []
    all_attention_weights = []
    temporal_kl_values = []
    
    for _ in range(num_games):
        state = env.reset()
        hid = agent.get_h0(1)
        game_attention = []
        
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
        
        scores.append(state.score)
        
        # Compute temporal KL for this game
        if len(game_attention) > 1:
            game_attention = np.array(game_attention)
            for t in range(len(game_attention) - 1):
                p, q = game_attention[t], game_attention[t + 1]
                kl = np.sum(p * np.log((p + 1e-8) / (q + 1e-8))) + np.sum(q * np.log((q + 1e-8) / (p + 1e-8)))
                temporal_kl_values.append(kl)
            all_attention_weights.append(np.mean(game_attention, axis=0))
    
    # Average attention weights across games
    avg_attention = np.mean(all_attention_weights, axis=0) if all_attention_weights else None
    
    return {
        "mean_score": np.mean(scores), 
        "std_score": np.std(scores), 
        "perfect_rate": sum(s == 25 for s in scores) / len(scores),
        "attention_weights": avg_attention,
        "temporal_kl_values": temporal_kl_values,
    }


def train(config, args):
    if args.temporal_reg_lambda is not None:
        config["temporal_reg_lambda"] = args.temporal_reg_lambda
    
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed", 42)
    set_seed(seed)
    
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config["logging"]["save_dir"]) / f"run_{timestamp}_lambda_{config['temporal_reg_lambda']}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    print(f"Saving to: {save_dir}")
    print(f"Temporal regularization lambda: {config['temporal_reg_lambda']}")
    
    # Initialize logger
    logger = TrainingLogger(save_dir, config)
    
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
    
    optimizer = torch.optim.Adam(agent.online_net.parameters(), lr=config["training"]["lr"], eps=config["training"]["eps"])
    replay_buffer = ReplayBuffer(capacity=config["replay"]["buffer_size"], seed=seed)
    
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
        epoch_stats = {"rl_loss": [], "temporal_loss": [], "total_loss": []}
        
        decay_progress = min(1.0, epoch / config["collect"]["epsilon_decay_epochs"])
        epsilon = config["collect"]["epsilon_start"] + (config["collect"]["epsilon_end"] - config["collect"]["epsilon_start"]) * decay_progress
        
        print(f"Epoch {epoch + 1}: Collecting...", end=" ", flush=True)
        for _ in range(config["collect"]["num_games_per_epoch"]):
            traj = collect_trajectory(env, agent, epsilon, device)
            if traj is not None:
                replay_buffer.add(traj)
        print("Training...", end=" ", flush=True)
        
        for _ in range(config["training"]["epoch_len"]):
            batch = replay_buffer.sample(config["training"]["batch_size"], device)
            loss, stats = agent.loss_from_batch(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), config["training"]["grad_clip"])
            optimizer.step()
            for k, v in stats.items():
                epoch_stats[k].append(v)
        
        epoch_time = time.time() - epoch_start
        print(f"done. ({epoch_time:.1f}s)")
        
        if (epoch + 1) % config["training"]["target_sync_freq"] == 0:
            agent.sync_target_with_online()
        
        avg_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
        stats_history.append(avg_stats)
        
        # Log training metrics
        logger.log_epoch(
            epoch=epoch + 1,
            rl_loss=avg_stats["rl_loss"],
            temporal_loss=avg_stats["temporal_loss"],
            total_loss=avg_stats["total_loss"],
            epsilon=epsilon,
        )
        
        if (epoch + 1) % config["logging"]["log_freq"] == 0:
            eval_results = evaluate(env, agent, num_games=20, device=device)
            
            # Log evaluation metrics
            logger.log_eval(
                epoch=epoch + 1,
                mean_score=eval_results["mean_score"],
                std_score=eval_results["std_score"],
                perfect_rate=eval_results["perfect_rate"],
            )
            
            # Log attention statistics
            if eval_results["attention_weights"] is not None:
                logger.log_attention(
                    epoch=epoch + 1,
                    section_weights=eval_results["attention_weights"],
                    temporal_kl_values=eval_results["temporal_kl_values"],
                )
            
            print(f"  Epoch {epoch + 1:4d} | RL: {avg_stats['rl_loss']:.4f} | Temporal: {avg_stats['temporal_loss']:.4f} | "
                  f"Score: {eval_results['mean_score']:.2f}±{eval_results['std_score']:.2f} | Perfect: {eval_results['perfect_rate']*100:.1f}%")
            
            if eval_results["mean_score"] > best_score:
                best_score = eval_results["mean_score"]
                torch.save({"epoch": epoch, "agent_state_dict": agent.state_dict(), "eval_results": eval_results}, save_dir / "best_model.pt")
        
        if (epoch + 1) % config["logging"]["save_freq"] == 0:
            torch.save({"epoch": epoch, "agent_state_dict": agent.state_dict(), "stats_history": stats_history}, save_dir / f"checkpoint_{epoch + 1}.pt")
            # Generate plots periodically
            logger.generate_all_plots()
    
    # Final logging and plots
    logger.generate_all_plots()
    
    print(f"\nTraining complete. Best score: {best_score:.2f}")
    print(f"Results saved to: {save_dir}")
    print(f"Plots saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--temporal_reg_lambda", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.seed:
        config["seed"] = args.seed
    
    train(config, args)


if __name__ == "__main__":
    main()
