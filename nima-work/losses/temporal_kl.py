"""
Temporal KL Divergence Loss for Attention Consistency.

This module implements the temporal attention consistency regularization
described in the project proposal. The idea is that since Hanabi game states
change incrementally between turns, the model's attention distribution should
also change smoothly rather than abruptly.

We enforce this by adding a symmetric KL divergence penalty between attention
distributions at consecutive timesteps:

L_temporal = E_t [ sum_h D_sym(A^(h)(s_t), A^(h)(s_{t+1})) ]

where D_sym(p, q) = KL(p || q) + KL(q || p)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def symmetric_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute symmetric KL divergence between two distributions.
    
    D_sym(p, q) = KL(p || q) + KL(q || p)
    
    Args:
        p: [..., dim] probability distribution (should sum to 1 along last dim)
        q: [..., dim] probability distribution (should sum to 1 along last dim)
        epsilon: small constant for numerical stability
        
    Returns:
        [...] symmetric KL divergence for each distribution pair
    """
    p = p + epsilon
    q = q + epsilon
    
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    
    return kl_pq + kl_qp


def temporal_kl_loss(
    attention_weights: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute temporal KL consistency loss over attention weights.
    
    This encourages the attention distribution to change smoothly between
    consecutive timesteps, which aligns with the intuition that Hanabi game
    states change incrementally.
    
    Args:
        attention_weights: [seq_len, batch, num_sections] or [seq_len, batch, num_heads, num_sections]
            Attention weights over sections at each timestep.
            Should be normalized (sum to 1) along the last dimension.
        mask: [seq_len, batch] optional mask for valid timesteps
            1 = valid, 0 = padding
        epsilon: small constant for numerical stability
        reduction: "mean", "sum", or "none"
        
    Returns:
        Temporal KL loss (scalar if reduction != "none")
        
    Example:
        >>> attention = torch.softmax(torch.randn(10, 32, 7), dim=-1)  # [T, B, sections]
        >>> loss = temporal_kl_loss(attention)
    """
    if attention_weights.dim() == 4:
        seq_len, batch, num_heads, num_sections = attention_weights.shape
        has_heads = True
    else:
        seq_len, batch, num_sections = attention_weights.shape
        has_heads = False
    
    if seq_len < 2:
        return torch.tensor(0.0, device=attention_weights.device)
    
    attn_t = attention_weights[:-1]  # [T-1, batch, ...]
    attn_t1 = attention_weights[1:]   # [T-1, batch, ...]
    
    sym_kl = symmetric_kl_divergence(attn_t, attn_t1, epsilon)  # [T-1, batch] or [T-1, batch, heads]
    
    if has_heads:
        sym_kl = sym_kl.sum(dim=-1)  # Sum over heads: [T-1, batch]
    
    if mask is not None:
        pair_mask = mask[:-1] * mask[1:]  # [T-1, batch]
        sym_kl = sym_kl * pair_mask
        
        if reduction == "mean":
            num_valid = pair_mask.sum().clamp(min=1)
            return sym_kl.sum() / num_valid
        elif reduction == "sum":
            return sym_kl.sum()
        else:
            return sym_kl
    else:
        if reduction == "mean":
            return sym_kl.mean()
        elif reduction == "sum":
            return sym_kl.sum()
        else:
            return sym_kl


def temporal_kl_loss_batched(
    attention_weights: torch.Tensor,
    seq_lens: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute temporal KL loss with variable sequence lengths.
    
    Args:
        attention_weights: [seq_len, batch, num_sections]
        seq_lens: [batch] actual sequence lengths for each batch element
        epsilon: numerical stability constant
        
    Returns:
        Scalar loss averaged over valid timestep pairs
    """
    seq_len, batch, num_sections = attention_weights.shape
    device = attention_weights.device
    
    timesteps = torch.arange(seq_len, device=device).unsqueeze(1)  # [T, 1]
    mask = (timesteps < seq_lens.unsqueeze(0)).float()  # [T, batch]
    
    return temporal_kl_loss(attention_weights, mask=mask, epsilon=epsilon, reduction="mean")


class TemporalKLLoss(torch.nn.Module):
    """
    Module wrapper for temporal KL loss.
    
    Usage:
        loss_fn = TemporalKLLoss(lambda_weight=0.01)
        temporal_loss = loss_fn(attention_weights, seq_lens)
    """
    
    def __init__(
        self,
        lambda_weight: float = 0.01,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.epsilon = epsilon
    
    def forward(
        self,
        attention_weights: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted temporal KL loss.
        
        Args:
            attention_weights: [seq_len, batch, num_sections]
            seq_lens: [batch] sequence lengths (optional)
            mask: [seq_len, batch] validity mask (optional)
            
        Returns:
            lambda * temporal_kl_loss
        """
        if seq_lens is not None:
            loss = temporal_kl_loss_batched(attention_weights, seq_lens, self.epsilon)
        else:
            loss = temporal_kl_loss(attention_weights, mask=mask, epsilon=self.epsilon)
        
        return self.lambda_weight * loss
