"""
Pure Python Replay Buffer for R2D2-style training.

This implements a sequence-based replay buffer suitable for training
recurrent agents on Hanabi trajectories.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class Trajectory:
    """A single game trajectory."""
    section_embeddings: torch.Tensor  # [seq_len, num_sections, hidden_dim]
    legal_moves: torch.Tensor         # [seq_len, num_actions]
    actions: torch.Tensor             # [seq_len]
    rewards: torch.Tensor             # [seq_len]
    terminals: torch.Tensor           # [seq_len]
    seq_len: int
    
    def to(self, device: str) -> "Trajectory":
        return Trajectory(
            section_embeddings=self.section_embeddings.to(device),
            legal_moves=self.legal_moves.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            terminals=self.terminals.to(device),
            seq_len=self.seq_len,
        )


@dataclass
class Batch:
    """A batch of trajectories for training."""
    section_embeddings: torch.Tensor  # [max_seq_len, batch, num_sections, hidden_dim]
    legal_moves: torch.Tensor         # [max_seq_len, batch, num_actions]
    actions: torch.Tensor             # [max_seq_len, batch]
    rewards: torch.Tensor             # [max_seq_len, batch]
    terminals: torch.Tensor           # [max_seq_len, batch]
    bootstrap: torch.Tensor           # [max_seq_len, batch]
    seq_lens: torch.Tensor            # [batch]
    mask: torch.Tensor                # [max_seq_len, batch]


class ReplayBuffer:
    """
    Replay buffer storing full game trajectories.
    
    Each trajectory contains:
    - Pre-computed section embeddings (to avoid re-encoding)
    - Legal moves at each timestep
    - Actions taken
    - Rewards received
    - Terminal flags
    """
    
    def __init__(
        self,
        capacity: int,
        seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)
    
    def add(self, trajectory: Trajectory):
        """Add a trajectory to the buffer."""
        self.buffer.append(trajectory)
    
    def sample(self, batch_size: int, device: str) -> Batch:
        """Sample a batch of trajectories."""
        trajectories = self.rng.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        return self._collate(trajectories, device)
    
    def _collate(self, trajectories: List[Trajectory], device: str) -> Batch:
        """Collate trajectories into a padded batch."""
        max_seq_len = max(t.seq_len for t in trajectories)
        batch_size = len(trajectories)
        
        num_sections = trajectories[0].section_embeddings.shape[1]
        hidden_dim = trajectories[0].section_embeddings.shape[2]
        num_actions = trajectories[0].legal_moves.shape[1]
        
        section_embeddings = torch.zeros(max_seq_len, batch_size, num_sections, hidden_dim)
        legal_moves = torch.zeros(max_seq_len, batch_size, num_actions)
        actions = torch.zeros(max_seq_len, batch_size, dtype=torch.long)
        rewards = torch.zeros(max_seq_len, batch_size)
        terminals = torch.zeros(max_seq_len, batch_size)
        mask = torch.zeros(max_seq_len, batch_size)
        seq_lens = torch.zeros(batch_size, dtype=torch.long)
        
        for i, traj in enumerate(trajectories):
            seq_len = traj.seq_len
            section_embeddings[:seq_len, i] = traj.section_embeddings
            legal_moves[:seq_len, i] = traj.legal_moves
            actions[:seq_len, i] = traj.actions
            rewards[:seq_len, i] = traj.rewards
            terminals[:seq_len, i] = traj.terminals
            mask[:seq_len, i] = 1.0
            seq_lens[i] = seq_len
        
        bootstrap = 1.0 - terminals
        
        return Batch(
            section_embeddings=section_embeddings.to(device),
            legal_moves=legal_moves.to(device),
            actions=actions.to(device),
            rewards=rewards.to(device),
            terminals=terminals.to(device),
            bootstrap=bootstrap.to(device),
            seq_lens=seq_lens.to(device),
            mask=mask.to(device),
        )
    
    def size(self) -> int:
        return len(self.buffer)
    
    def __len__(self) -> int:
        return len(self.buffer)


class TrajectoryCollector:
    """
    Collects transitions during environment interaction and forms trajectories.
    """
    
    def __init__(self, num_sections: int, hidden_dim: int, num_actions: int, max_len: int):
        self.num_sections = num_sections
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.max_len = max_len
        self.reset()
    
    def reset(self):
        """Reset the collector for a new episode."""
        self.section_embeddings = []
        self.legal_moves = []
        self.actions = []
        self.rewards = []
        self.terminals = []
    
    def add(
        self,
        section_embedding: torch.Tensor,
        legal_move: torch.Tensor,
        action: int,
        reward: float,
        terminal: bool,
    ):
        """Add a transition."""
        self.section_embeddings.append(section_embedding.detach().cpu())
        self.legal_moves.append(legal_move.detach().cpu())
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(float(terminal))
    
    def get_trajectory(self) -> Optional[Trajectory]:
        """Get the collected trajectory."""
        if len(self.actions) == 0:
            return None
        
        seq_len = len(self.actions)
        
        return Trajectory(
            section_embeddings=torch.stack(self.section_embeddings),
            legal_moves=torch.stack(self.legal_moves),
            actions=torch.tensor(self.actions, dtype=torch.long),
            rewards=torch.tensor(self.rewards, dtype=torch.float),
            terminals=torch.tensor(self.terminals, dtype=torch.float),
            seq_len=seq_len,
        )
