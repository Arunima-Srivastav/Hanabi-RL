"""
R2D2 Agent with Temporal KL Regularization.

This agent uses the TextLSTMNetWithSectionAttention network and includes
the temporal KL loss in its training objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from networks.q_net import TextLSTMNetWithSectionAttention
from losses.temporal_kl import temporal_kl_loss_batched


class R2D2Agent(nn.Module):
    """
    R2D2 Agent with section attention and temporal KL regularization.
    
    Training objective:
        L = L_RL + lambda * L_temporal
    
    where:
        L_RL = TD error (smooth L1 loss)
        L_temporal = symmetric KL divergence between attention at consecutive timesteps
    """
    
    def __init__(
        self,
        device: str,
        hidden_dim: int = 512,
        num_actions: int = 21,
        num_lstm_layers: int = 2,
        gamma: float = 0.99,
        multi_step: int = 1,
        temporal_reg_lambda: float = 0.01,
        pretrained_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.multi_step = multi_step
        self.temporal_reg_lambda = temporal_reg_lambda
        
        self.online_net = TextLSTMNetWithSectionAttention(
            device=device,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_lstm_layers=num_lstm_layers,
            pretrained_model=pretrained_model,
            freeze_bert=freeze_bert,
        )
        
        self.target_net = TextLSTMNetWithSectionAttention(
            device=device,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_lstm_layers=num_lstm_layers,
            pretrained_model=pretrained_model,
            freeze_bert=freeze_bert,
        )
        
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        self.sync_target_with_online()
    
    def sync_target_with_online(self):
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def get_h0(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get initial hidden state."""
        return self.online_net.get_h0(batch_size)
    
    def act(
        self,
        section_embeddings: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
        epsilon: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Select action with epsilon-greedy exploration.
        
        Args:
            section_embeddings: [batch, num_sections, hidden_dim]
            legal_move: [batch, num_actions]
            hid: LSTM hidden state
            epsilon: exploration rate
            
        Returns:
            action: [batch] selected actions
            new_hid: updated hidden state
            attention: [batch, num_sections] attention weights
        """
        greedy_action, new_hid, attention = self.online_net.act(
            section_embeddings, legal_move, hid
        )
        
        if epsilon > 0:
            batch_size = greedy_action.shape[0]
            random_action = torch.randint(
                0, legal_move.shape[1], (batch_size,), device=self.device
            )
            
            num_legal = legal_move.sum(dim=1)
            random_idx = (torch.rand(batch_size, device=self.device) * num_legal).long()
            legal_indices = legal_move.cumsum(dim=1)
            random_action = (legal_indices > random_idx.unsqueeze(1)).float().argmax(dim=1)
            
            use_random = (torch.rand(batch_size, device=self.device) < epsilon).long()
            action = greedy_action * (1 - use_random) + random_action * use_random
        else:
            action = greedy_action
        
        return action, new_hid, attention
    
    def compute_td_error(
        self,
        section_embeddings: torch.Tensor,
        legal_moves: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute TD error for a batch of trajectories.
        
        Args:
            section_embeddings: [seq_len, batch, num_sections, hidden_dim]
            legal_moves: [seq_len, batch, num_actions]
            actions: [seq_len, batch]
            rewards: [seq_len, batch]
            bootstrap: [seq_len, batch] (1 - terminal)
            seq_lens: [batch]
            
        Returns:
            td_error: [seq_len, batch]
            attention_weights: [seq_len, batch, num_sections]
            q_values: [seq_len, batch, num_actions]
        """
        qa, greedy_a, online_q, lstm_o, attention = self.online_net(
            section_embeddings, legal_moves, actions, hid=None
        )
        
        with torch.no_grad():
            target_qa, _, _, _, _ = self.target_net(
                section_embeddings, legal_moves, greedy_a, hid=None
            )
        
        max_seq_len = section_embeddings.shape[0]
        
        target_qa_shifted = torch.cat([
            target_qa[self.multi_step:],
            torch.zeros(self.multi_step, target_qa.shape[1], device=self.device)
        ], dim=0)
        
        target = rewards + bootstrap * (self.gamma ** self.multi_step) * target_qa_shifted
        
        td_error = target.detach() - qa
        
        return td_error, attention, online_q
    
    def loss(
        self,
        section_embeddings: torch.Tensor,
        legal_moves: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        seq_lens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss = RL loss + temporal KL loss.
        
        Args:
            section_embeddings: [seq_len, batch, num_sections, hidden_dim]
            legal_moves: [seq_len, batch, num_actions]
            actions: [seq_len, batch]
            rewards: [seq_len, batch]
            bootstrap: [seq_len, batch]
            seq_lens: [batch]
            mask: [seq_len, batch]
            
        Returns:
            total_loss: scalar
            stats: dict with individual loss components
        """
        td_error, attention_weights, _ = self.compute_td_error(
            section_embeddings, legal_moves, actions, rewards, bootstrap, seq_lens
        )
        
        rl_loss = F.smooth_l1_loss(td_error * mask, torch.zeros_like(td_error), reduction="none")
        rl_loss = (rl_loss * mask).sum() / mask.sum().clamp(min=1)
        
        temporal_loss = temporal_kl_loss_batched(attention_weights, seq_lens)
        
        total_loss = rl_loss + self.temporal_reg_lambda * temporal_loss
        
        stats = {
            "rl_loss": rl_loss.item(),
            "temporal_loss": temporal_loss.item(),
            "total_loss": total_loss.item(),
        }
        
        return total_loss, stats
    
    def loss_from_batch(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Convenience method to compute loss from a Batch object."""
        return self.loss(
            section_embeddings=batch.section_embeddings,
            legal_moves=batch.legal_moves,
            actions=batch.actions,
            rewards=batch.rewards,
            bootstrap=batch.bootstrap,
            seq_lens=batch.seq_lens,
            mask=batch.mask,
        )
