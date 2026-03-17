"""
Pure Python Replay Buffer for R2D2-style training.
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class Trajectory:
    section_embeddings: torch.Tensor
    legal_moves: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    seq_len: int


@dataclass
class Batch:
    section_embeddings: torch.Tensor
    legal_moves: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    bootstrap: torch.Tensor
    seq_lens: torch.Tensor
    mask: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)
    
    def add(self, trajectory: Trajectory):
        self.buffer.append(trajectory)
    
    def sample(self, batch_size: int, device: str) -> Batch:
        trajectories = self.rng.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        return self._collate(trajectories, device)
    
    def _collate(self, trajectories: List[Trajectory], device: str) -> Batch:
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
            sl = traj.seq_len
            section_embeddings[:sl, i] = traj.section_embeddings
            legal_moves[:sl, i] = traj.legal_moves
            actions[:sl, i] = traj.actions
            rewards[:sl, i] = traj.rewards
            terminals[:sl, i] = traj.terminals
            mask[:sl, i] = 1.0
            seq_lens[i] = sl
        
        return Batch(
            section_embeddings=section_embeddings.to(device),
            legal_moves=legal_moves.to(device),
            actions=actions.to(device),
            rewards=rewards.to(device),
            terminals=terminals.to(device),
            bootstrap=(1.0 - terminals).to(device),
            seq_lens=seq_lens.to(device),
            mask=mask.to(device),
        )
    
    def size(self) -> int:
        return len(self.buffer)


class TrajectoryCollector:
    def __init__(self, num_sections: int, hidden_dim: int, num_actions: int, max_len: int):
        self.num_sections = num_sections
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.max_len = max_len
        self.reset()
    
    def reset(self):
        self.section_embeddings = []
        self.legal_moves = []
        self.actions = []
        self.rewards = []
        self.terminals = []
    
    def add(self, section_embedding, legal_move, action, reward, terminal):
        self.section_embeddings.append(section_embedding.detach().cpu())
        self.legal_moves.append(legal_move.detach().cpu())
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(float(terminal))
    
    def get_trajectory(self) -> Optional[Trajectory]:
        if len(self.actions) == 0:
            return None
        return Trajectory(
            section_embeddings=torch.stack(self.section_embeddings),
            legal_moves=torch.stack(self.legal_moves),
            actions=torch.tensor(self.actions, dtype=torch.long),
            rewards=torch.tensor(self.rewards, dtype=torch.float),
            terminals=torch.tensor(self.terminals, dtype=torch.float),
            seq_len=len(self.actions),
        )
