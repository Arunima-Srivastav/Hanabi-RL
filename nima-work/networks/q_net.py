"""
Q-Network with Section Attention for Hanabi.

This network encodes each of the 7 game sections separately, then uses a learned
attention mechanism to weight them. The attention weights are returned for
temporal KL regularization.

7 Sections (from milestone):
1. Life tokens
2. Hint tokens
3. Fireworks
4. Own hand (agent's knowledge about own hand)
5. Opponent's hand
6. Discards
7. Last action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from transformers import BertModel, BertConfig, BertTokenizer


NUM_SECTIONS = 7
SECTION_NAMES = [
    "life_tokens",
    "hint_tokens", 
    "fireworks",
    "own_hand",
    "opponent_hand",
    "discards",
    "last_action",
]


class SectionEncoder(nn.Module):
    """Encodes a single text section using a shared BERT encoder."""
    
    def __init__(self, bert_model: nn.Module, hidden_dim: int):
        super().__init__()
        self.bert = bert_model
        self.projection = nn.Linear(bert_model.config.hidden_size, hidden_dim)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token IDs for this section
        Returns:
            [batch, hidden_dim] section embedding
        """
        outputs = self.bert(input_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.projection(cls_embedding)


class SectionAttention(nn.Module):
    """
    Learned attention over the 7 game sections.
    
    This produces attention weights that can be used for:
    1. Weighted combination of section embeddings for Q-value computation
    2. Temporal KL regularization (encouraging smooth changes across timesteps)
    """
    
    def __init__(self, hidden_dim: int, num_sections: int = NUM_SECTIONS):
        super().__init__()
        self.num_sections = num_sections
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
    
    def forward(self, section_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            section_embeddings: [batch, num_sections, hidden_dim]
        Returns:
            weighted_embedding: [batch, hidden_dim]
            attention_weights: [batch, num_sections] - normalized attention over sections
        """
        keys = self.key_proj(section_embeddings)  # [batch, num_sections, hidden_dim]
        
        scores = torch.einsum("bsh,h->bs", keys, self.query) / self.scale
        attention_weights = F.softmax(scores, dim=-1)  # [batch, num_sections]
        
        weighted_embedding = torch.einsum("bs,bsh->bh", attention_weights, section_embeddings)
        
        return weighted_embedding, attention_weights


class TextLSTMNetWithSectionAttention(nn.Module):
    """
    Text-based LSTM Q-Network with explicit section attention.
    
    Key difference from original TextLSTMNet: we encode each section separately
    and use a learned attention mechanism. The attention weights are returned
    for temporal KL regularization.
    """
    
    def __init__(
        self,
        device: str,
        hidden_dim: int = 512,
        num_actions: int = 21,  # 5 discard + 5 play + 5 color hints + 5 rank hints + noop
        num_lstm_layers: int = 2,
        pretrained_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_lstm_layers = num_lstm_layers
        
        config = BertConfig.from_pretrained(pretrained_model)
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        
        self.section_projection = nn.Linear(config.hidden_size, hidden_dim)
        
        self.section_attention = SectionAttention(hidden_dim, NUM_SECTIONS)
        
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=False,
        )
        
        self.fc_v = nn.Linear(hidden_dim, 1)
        self.fc_a = nn.Linear(hidden_dim, num_actions)
        
        self.to(device)
    
    def encode_sections(self, section_texts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode each section separately.
        
        Args:
            section_texts: Dict mapping section name to token IDs [batch, seq_len]
        Returns:
            section_embeddings: [batch, num_sections, hidden_dim]
        """
        embeddings = []
        for section_name in SECTION_NAMES:
            input_ids = section_texts[section_name]
            outputs = self.bert(input_ids)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch, bert_hidden]
            proj_emb = self.section_projection(cls_emb)  # [batch, hidden_dim]
            embeddings.append(proj_emb)
        
        return torch.stack(embeddings, dim=1)  # [batch, num_sections, hidden_dim]
    
    def encode_sections_from_strings(self, section_texts: Dict[str, list]) -> torch.Tensor:
        """
        Encode sections from raw strings (convenience method).
        
        Args:
            section_texts: Dict mapping section name to list of strings (batch)
        Returns:
            section_embeddings: [batch, num_sections, hidden_dim]
        """
        embeddings = []
        for section_name in SECTION_NAMES:
            texts = section_texts[section_name]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            
            outputs = self.bert(input_ids)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            proj_emb = self.section_projection(cls_emb)
            embeddings.append(proj_emb)
        
        return torch.stack(embeddings, dim=1)
    
    def get_h0(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Initialize LSTM hidden state."""
        shape = (self.num_lstm_layers, batch_size, self.hidden_dim)
        return {
            "h0": torch.zeros(*shape, device=self.device),
            "c0": torch.zeros(*shape, device=self.device),
        }
    
    def forward(
        self,
        section_embeddings: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning Q-values and attention weights.
        
        Args:
            section_embeddings: [seq_len, batch, num_sections, hidden_dim]
            legal_move: [seq_len, batch, num_actions]
            action: [seq_len, batch]
            hid: LSTM hidden state dict
            
        Returns:
            qa: [seq_len, batch] Q-values for taken actions
            greedy_action: [seq_len, batch] greedy action indices
            q: [seq_len, batch, num_actions] all Q-values
            lstm_out: [seq_len, batch, hidden_dim] LSTM outputs
            section_attention: [seq_len, batch, num_sections] attention weights per timestep
        """
        seq_len, batch, num_sections, hidden_dim = section_embeddings.shape
        
        all_attention_weights = []
        all_weighted_embeddings = []
        
        for t in range(seq_len):
            weighted_emb, attn_weights = self.section_attention(section_embeddings[t])
            all_weighted_embeddings.append(weighted_emb)
            all_attention_weights.append(attn_weights)
        
        x = torch.stack(all_weighted_embeddings, dim=0)  # [seq_len, batch, hidden_dim]
        section_attention = torch.stack(all_attention_weights, dim=0)  # [seq_len, batch, num_sections]
        
        if hid is None:
            lstm_out, _ = self.lstm(x)
        else:
            lstm_out, _ = self.lstm(x, (hid["h0"], hid["c0"]))
        
        v = self.fc_v(lstm_out)  # [seq_len, batch, 1]
        a = self.fc_a(lstm_out)  # [seq_len, batch, num_actions]
        
        legal_a = a * legal_move
        q = v + legal_a  # Dueling architecture
        
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)  # [seq_len, batch]
        
        legal_q = (1 + q - q.min()) * legal_move
        greedy_action = legal_q.argmax(2).detach()
        
        return qa, greedy_action, q, lstm_out, section_attention
    
    def act(
        self,
        section_embeddings: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Select action for a single timestep.
        
        Args:
            section_embeddings: [batch, num_sections, hidden_dim]
            legal_move: [batch, num_actions]
            hid: LSTM hidden state
            
        Returns:
            action: [batch] selected action indices
            new_hid: updated hidden state
            attention_weights: [batch, num_sections] attention over sections
        """
        with torch.no_grad():
            weighted_emb, attention_weights = self.section_attention(section_embeddings)
            x = weighted_emb.unsqueeze(0)  # [1, batch, hidden_dim]
            
            lstm_out, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
            
            a = self.fc_a(lstm_out.squeeze(0))  # [batch, num_actions]
            
            legal_a = (1 + a - a.min()) * legal_move
            action = legal_a.argmax(1).detach()
            
            new_hid = {"h0": h, "c0": c}
            
        return action, new_hid, attention_weights
