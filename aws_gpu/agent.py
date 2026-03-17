"""
R2D2 Agent with Temporal KL Regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from networks.q_net import TextLSTMNetWithSectionAttention
from losses.temporal_kl import temporal_kl_loss_batched


class R2D2Agent(nn.Module):
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
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def get_h0(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batch_size)
    
    def act(self, section_embeddings, legal_move, hid, epsilon=0.0):
        greedy_action, new_hid, attention = self.online_net.act(section_embeddings, legal_move, hid)
        
        if epsilon > 0:
            batch_size = greedy_action.shape[0]
            num_legal = legal_move.sum(dim=1)
            random_idx = (torch.rand(batch_size, device=self.device) * num_legal).long()
            legal_indices = legal_move.cumsum(dim=1)
            random_action = (legal_indices > random_idx.unsqueeze(1)).float().argmax(dim=1)
            use_random = (torch.rand(batch_size, device=self.device) < epsilon).long()
            action = greedy_action * (1 - use_random) + random_action * use_random
        else:
            action = greedy_action
        
        return action, new_hid, attention
    
    def compute_td_error(self, section_embeddings, legal_moves, actions, rewards, bootstrap, seq_lens):
        qa, greedy_a, online_q, lstm_o, attention = self.online_net(
            section_embeddings, legal_moves, actions, hid=None
        )
        
        with torch.no_grad():
            target_qa, _, _, _, _ = self.target_net(section_embeddings, legal_moves, greedy_a, hid=None)
        
        target_qa_shifted = torch.cat([
            target_qa[self.multi_step:],
            torch.zeros(self.multi_step, target_qa.shape[1], device=self.device)
        ], dim=0)
        
        target = rewards + bootstrap * (self.gamma ** self.multi_step) * target_qa_shifted
        td_error = target.detach() - qa
        
        return td_error, attention, online_q
    
    def loss(self, section_embeddings, legal_moves, actions, rewards, bootstrap, seq_lens, mask):
        td_error, attention_weights, _ = self.compute_td_error(
            section_embeddings, legal_moves, actions, rewards, bootstrap, seq_lens
        )
        
        rl_loss = F.smooth_l1_loss(td_error * mask, torch.zeros_like(td_error), reduction="none")
        rl_loss = (rl_loss * mask).sum() / mask.sum().clamp(min=1)
        
        temporal_loss = temporal_kl_loss_batched(attention_weights, seq_lens)
        total_loss = rl_loss + self.temporal_reg_lambda * temporal_loss
        
        return total_loss, {
            "rl_loss": rl_loss.item(),
            "temporal_loss": temporal_loss.item(),
            "total_loss": total_loss.item(),
        }
    
    def loss_from_batch(self, batch):
        return self.loss(
            batch.section_embeddings, batch.legal_moves, batch.actions,
            batch.rewards, batch.bootstrap, batch.seq_lens, batch.mask
        )
