"""
Temporal KL Divergence Loss for Attention Consistency.

L_temporal = E_t [ D_sym(A(s_t), A(s_{t+1})) ]
where D_sym(p, q) = KL(p || q) + KL(q || p)
"""

import torch
from typing import Optional


def symmetric_kl_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
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
    if attention_weights.dim() == 4:
        seq_len, batch, num_heads, num_sections = attention_weights.shape
        has_heads = True
    else:
        seq_len, batch, num_sections = attention_weights.shape
        has_heads = False
    
    if seq_len < 2:
        return torch.tensor(0.0, device=attention_weights.device)
    
    attn_t = attention_weights[:-1]
    attn_t1 = attention_weights[1:]
    sym_kl = symmetric_kl_divergence(attn_t, attn_t1, epsilon)
    
    if has_heads:
        sym_kl = sym_kl.sum(dim=-1)
    
    if mask is not None:
        pair_mask = mask[:-1] * mask[1:]
        sym_kl = sym_kl * pair_mask
        if reduction == "mean":
            return sym_kl.sum() / pair_mask.sum().clamp(min=1)
        elif reduction == "sum":
            return sym_kl.sum()
        return sym_kl
    else:
        if reduction == "mean":
            return sym_kl.mean()
        elif reduction == "sum":
            return sym_kl.sum()
        return sym_kl


def temporal_kl_loss_batched(
    attention_weights: torch.Tensor,
    seq_lens: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    seq_len, batch, num_sections = attention_weights.shape
    device = attention_weights.device
    timesteps = torch.arange(seq_len, device=device).unsqueeze(1)
    mask = (timesteps < seq_lens.unsqueeze(0)).float()
    return temporal_kl_loss(attention_weights, mask=mask, epsilon=epsilon, reduction="mean")
