#!/usr/bin/env python3
"""
Interpretability analysis for R3D2 Hanabi agents (TextLSTMNet / drrn-lstm).

Architecture recap
------------------
TextLSTMNet encodes the game state as natural language via TinyBERT
(cross-encoder/ms-marco-TinyBERT-L-2-v2, trimmed to num_lm_layer=1 layers),
then maintains temporal memory with a 2-layer LSTM (hid_dim=128).
For drrn-lstm, Q(s,a) = fc_v(h) + fc_a(BERT(state) ⊙ BERT(action)).

Analyses
--------
  1. weights      — Parameter norms, LSTM gate magnitudes, fc_v vs fc_a directions.
                    Always available; no game environment needed.

  2. bert_attn    — TinyBERT attention maps on representative game-state texts.
                    Requires `transformers` (already in requirements).

  3. action_sim   — Pairwise cosine similarity of BERT action embeddings, showing
                    which actions the network considers structurally similar.

  4. hidden_pca   — PCA of LSTM hidden states collected over many live games,
                    coloured by timestep, action type, and cumulative score.
                    Requires the compiled C++ game environment (rela, hanalearn).

  5. probing      — Linear probes trained on hidden states to predict game
                    features (action type, score bucket).
                    Requires hidden_pca to have already run.

  6. qval         — Value and max-advantage trajectories plotted over game time,
                    showing how the agent's confidence evolves across an episode.
                    Requires the compiled C++ game environment.

Usage (run from pyhanabi/)
--------------------------
    python tools/interpretability.py \\
        --weight ../final_r3d2_ckpts/R3D2-2p/a/epoch3000.pthw \\
        --output_dir ../interpret_results \\
        --analyses weights,bert_attn,action_sim

    # With game environment compiled (adds hidden_pca, probing, qval):
    python tools/interpretability.py \\
        --weight ../final_r3d2_ckpts/R3D2-2p/a/epoch3000.pthw \\
        --output_dir ../interpret_results \\
        --analyses weights,bert_attn,action_sim,hidden_pca,probing,qval \\
        --num_games 200
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
# Insert pyhanabi/ so we can import r2d2, q_net, etc. directly.
# We intentionally do NOT import utils or create here because those modules
# chain into set_path.py which asserts a compiled build/ directory exists.
# That directory is only present when the C++ game environment has been built.
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, lib_path)


# ── Self-contained model loading (no dependency on create.py / set_path.py) ──

import pickle
from collections import OrderedDict

import r2d2 as r2d2_module


def _load_cfg(weight_file: str) -> dict:
    """Read the training config saved alongside the checkpoint."""
    cfg_file = weight_file + ".cfg"
    if os.path.exists(cfg_file):
        return pickle.load(open(cfg_file, "rb"))
    # Fallback: parse from train.log
    log = os.path.join(os.path.dirname(weight_file), "train.log")
    if not os.path.exists(log):
        raise FileNotFoundError(f"No .cfg or train.log found near {weight_file}")
    import json
    lines = open(log).readlines()
    config_lines, open_count = [], 0
    for line in lines:
        if line.strip() and line.strip()[0] == "{":
            open_count += 1
        if open_count:
            config_lines.append(line)
        if line.strip() and line.strip()[-1] == "}":
            open_count -= 1
        if open_count == 0 and config_lines:
            break
    raw = "".join(config_lines).replace("'", '"')
    for src, dst in (("True", "true"), ("False", "false"), ("None", "null")):
        raw = raw.replace(src, dst)
    return json.loads(raw)


def _load_weight(model: torch.nn.Module, weight_file: str, device: str) -> None:
    """Load a .pthw state dict into model, matching keys and skipping mismatches."""
    state_dict = torch.load(weight_file, map_location=device)
    # Strip torch.compile prefix if present
    state_dict = OrderedDict(
        (k.replace("_orig_mod.", ""), v) for k, v in state_dict.items()
    )
    target = model.state_dict()
    filtered = OrderedDict()
    for k, v in target.items():
        if k not in state_dict:
            print(f"  [load] warning: {k} not found in checkpoint — keeping init")
            filtered[k] = v
        elif state_dict[k].size() != v.size():
            print(f"  [load] warning: {k} size mismatch — keeping init")
            filtered[k] = v
        else:
            filtered[k] = state_dict[k]
    model.load_state_dict(filtered)


def load_model(weight_file: str, device: str = "cpu"):
    """
    Load an R3D2 checkpoint without touching create.py / set_path.py.
    Returns (agent, cfg).
    """
    cfg = _load_cfg(weight_file)
    net_type = cfg.get("net", "drrn-lstm")

    agent = r2d2_module.R2D2Agent(
        vdn=False,
        multi_step=cfg.get("multi_step", 1),
        gamma=cfg.get("gamma", 0.999),
        device=device,
        in_dim=None,            # TextLSTMNet stores but does not use in_dim
        hid_dim=cfg["rnn_hid_dim"],
        out_dim=1,              # always 1 for drrn-lstm (hardcoded in R2D2Agent)
        net=net_type,
        num_lstm_layer=cfg["num_lstm_layer"],
        lm_weights=cfg.get("lm_weights", "pretrained"),
        num_of_player=cfg["num_player"],
        num_of_additional_layer=cfg.get("num_of_additional_layer", 0),
        num_lm_layer=cfg.get("num_lm_layer", 1),
        lora_dim=cfg.get("lora_dim", 0),
    ).to(device)

    _load_weight(agent.online_net, weight_file, device)
    agent.sync_target_with_online()
    agent.train(False)
    return agent, cfg

# ── action labels for 2-player Hanabi (20 legal moves + 1 no-op) ─────────────
ACTION_LABELS_2P = (
    [f"D{i+1}" for i in range(5)]   # discard card 0-4
    + [f"P{i+1}" for i in range(5)] # play card 0-4
    + [f"C{c}" for c in "RYGWB"]    # hint colour to player 1
    + [f"R{r}" for r in "12345"]    # hint rank to player 1
    + ["noop"]
)

# Three representative game states (approximated — actual text comes from
# HanabiState::ToText(); see README for the reference implementation).
SAMPLE_GAME_TEXTS = {
    "early_game": (
        "Life tokens: 3. Information tokens: 8. "
        "Fireworks: R0 Y0 G0 W0 B0. "
        "Player 0 (me): ?? ?? ?? ?? ??. "
        "Player 1: R1 Y2 G3 W4 B1. "
        "Discards: none. "
        "Last action: none."
    ),
    "mid_game": (
        "Life tokens: 2. Information tokens: 3. "
        "Fireworks: R2 Y1 G2 W1 B1. "
        "Player 0 (me): ?? ?? ?? ?? ??. "
        "Player 1: R3 Y2 G3 W2 B2. "
        "Discards: R1 Y1 G1 W1. "
        "Last action: player 1 hinted rank 2."
    ),
    "late_game": (
        "Life tokens: 1. Information tokens: 1. "
        "Fireworks: R4 Y4 G4 W4 B3. "
        "Player 0 (me): ?? ?? ?? ?? ??. "
        "Player 1: R5 Y5 G5 W5 B5. "
        "Discards: R1 Y1 G1 W1 B1 R2 Y2 G2. "
        "Last action: player 1 played B3."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-8)).item()


def action_type(action_idx: int, num_actions: int = 20) -> str:
    """Map a 2p action index to 'discard', 'play', 'hint', or 'noop'."""
    if action_idx < 5:
        return "discard"
    if action_idx < 10:
        return "play"
    if action_idx < num_actions:
        return "hint"
    return "noop"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Weight Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_weights(net: torch.nn.Module, output_dir: str) -> None:
    """
    Three sub-analyses:
      a) Full parameter summary (shapes + L2 norms) written to a text file.
      b) LSTM gate weight norms and gate bias means — the forget-gate bias is
         particularly informative: a large positive mean indicates the LSTM
         prefers to retain information across turns (long memory).
      c) fc_v vs fc_a weight vectors: cosine similarity and scatter plot.
         In drrn-lstm, both are shape [1, 128]. High similarity suggests the
         value and advantage heads decode the same direction of the hidden state.
    """
    ensure_dir(output_dir)

    # ── 1a. Parameter summary ─────────────────────────────────────────────────
    lines = ["Parameter summary\n", "=" * 70 + "\n"]
    for name, param in net.named_parameters():
        lines.append(
            f"  {name:65s}  shape={str(list(param.shape)):20s}"
            f"  norm={param.detach().norm().item():.4f}\n"
        )
    summary_path = os.path.join(output_dir, "weight_norms.txt")
    with open(summary_path, "w") as f:
        f.writelines(lines)
    print(f"[weights] parameter summary → {summary_path}")

    # ── 1b. LSTM gate analysis ────────────────────────────────────────────────
    lstm = net.state_lstm
    num_layers = lstm.num_layers
    gate_names = ["input (i)", "forget (f)", "cell (g)", "output (o)"]

    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 5 * num_layers))
    if num_layers == 1:
        axes = axes[np.newaxis, :]

    for layer in range(num_layers):
        wih = getattr(lstm, f"weight_ih_l{layer}").detach()  # [4H, input]
        whh = getattr(lstm, f"weight_hh_l{layer}").detach()  # [4H, H]
        bih = getattr(lstm, f"bias_ih_l{layer}").detach()    # [4H]
        bhh = getattr(lstm, f"bias_hh_l{layer}").detach()    # [4H]

        ih_gates = wih.chunk(4, dim=0)
        hh_gates = whh.chunk(4, dim=0)
        bih_gates = bih.chunk(4, dim=0)
        bhh_gates = bhh.chunk(4, dim=0)

        ih_norms = [g.norm().item() for g in ih_gates]
        hh_norms = [g.norm().item() for g in hh_gates]
        # Total bias per gate (bih + bhh) — tells us the gate's prior activation
        bias_means = [
            (b1 + b2).mean().item()
            for b1, b2 in zip(bih_gates, bhh_gates)
        ]

        x = np.arange(4)
        axes[layer, 0].bar(x - 0.2, ih_norms, 0.4, label="weight_ih (input→gate)")
        axes[layer, 0].bar(x + 0.2, hh_norms, 0.4, label="weight_hh (hidden→gate)")
        axes[layer, 0].set_xticks(x)
        axes[layer, 0].set_xticklabels(gate_names)
        axes[layer, 0].set_ylabel("Frobenius norm")
        axes[layer, 0].set_title(f"LSTM layer {layer} — gate weight norms")
        axes[layer, 0].legend()

        bar_colors = ["steelblue" if v >= 0 else "coral" for v in bias_means]
        axes[layer, 1].bar(gate_names, bias_means, color=bar_colors)
        axes[layer, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[layer, 1].set_ylabel("Mean combined bias")
        axes[layer, 1].set_title(
            f"LSTM layer {layer} — gate bias means\n"
            "(positive forget bias → LSTM retains state longer)"
        )

    plt.tight_layout()
    gate_path = os.path.join(output_dir, "lstm_gate_analysis.png")
    plt.savefig(gate_path, dpi=120)
    plt.close()
    print(f"[weights] LSTM gate analysis → {gate_path}")

    # ── 1c. fc_v vs fc_a ─────────────────────────────────────────────────────
    fc_v_w = net.fc_v.weight.detach().float().squeeze()   # [128]
    fc_a_w = net.fc_a.weight.detach().float().squeeze()   # [128]
    sim = cosine_sim(fc_v_w, fc_a_w)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].bar(range(len(fc_v_w)), fc_v_w.numpy(), color="steelblue", alpha=0.8)
    axes[0].set_title("fc_v weight vector\n(LSTM hidden → state value)")
    axes[0].set_xlabel("Hidden dimension")
    axes[0].set_ylabel("Weight")

    axes[1].bar(range(len(fc_a_w)), fc_a_w.numpy(), color="coral", alpha=0.8)
    axes[1].set_title("fc_a weight vector\n(state ⊙ action embed → advantage)")
    axes[1].set_xlabel("Hidden dimension")
    axes[1].set_ylabel("Weight")

    axes[2].scatter(fc_v_w.numpy(), fc_a_w.numpy(), alpha=0.5, s=12, color="purple")
    axes[2].set_xlabel("fc_v weight")
    axes[2].set_ylabel("fc_a weight")
    axes[2].set_title(
        f"fc_v vs fc_a weight components\ncosine similarity = {sim:.4f}"
    )

    print(
        f"[weights] cosine(fc_v, fc_a) = {sim:.4f}  "
        f"({'shared direction' if abs(sim) > 0.5 else 'distinct directions'})"
    )

    plt.tight_layout()
    fc_path = os.path.join(output_dir, "fc_v_vs_fc_a.png")
    plt.savefig(fc_path, dpi=120)
    plt.close()
    print(f"[weights] fc_v vs fc_a → {fc_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BERT Attention Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_bert_attention(
    net: torch.nn.Module,
    full_state_dict: dict,
    output_dir: str,
    num_lm_layer: int = 1,
) -> None:
    """
    Copy the TinyBERT weights from the checkpoint into a fresh (non-traced)
    BertModel that returns attention weights, then plot mean attention heatmaps
    for the three representative game-state texts.

    The state dict keys for the BERT submodule are prefixed with
    "call_transformer." in the checkpoint file.
    """
    try:
        from transformers import BertConfig, BertModel, BertTokenizer
    except ImportError:
        print("[bert_attn] transformers not available — skipping.")
        return

    ensure_dir(output_dir)
    pretrained_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

    # Build fresh BertModel with output_attentions=True
    config = BertConfig.from_pretrained(pretrained_name)
    config.num_hidden_layers = num_lm_layer
    config.output_attentions = True
    fresh_bert = BertModel(config)

    # Extract BERT parameters from the full checkpoint state dict
    prefix = "call_transformer."
    bert_sd = {
        k[len(prefix):]: v
        for k, v in full_state_dict.items()
        if k.startswith(prefix)
    }
    if not bert_sd:
        print(
            "[bert_attn] no 'call_transformer.*' keys found in state dict. "
            "Skipping — make sure weight_file is a .pthw checkpoint."
        )
        return

    missing, unexpected = fresh_bert.load_state_dict(bert_sd, strict=False)
    if missing:
        print(f"[bert_attn] missing keys in fresh BertModel ({len(missing)} total).")
    fresh_bert.eval()

    tokenizer = BertTokenizer.from_pretrained(pretrained_name)
    n = len(SAMPLE_GAME_TEXTS)
    fig, axes = plt.subplots(n, 1, figsize=(15, 7 * n))
    if n == 1:
        axes = [axes]

    for ax, (scenario, text) in zip(axes, SAMPLE_GAME_TEXTS.items()):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

        with torch.no_grad():
            out = fresh_bert(**enc)

        # out.attentions: tuple of (1 per layer) of [1, num_heads, seq, seq]
        # Average across heads for a single summary view
        mean_attn = out.attentions[0][0].mean(0).numpy()  # [seq, seq]

        im = ax.imshow(mean_attn, cmap="Blues", aspect="auto", vmin=0)
        ax.set_title(
            f"TinyBERT mean attention (all heads, layer 0) — {scenario}",
            fontsize=11,
        )
        tick_pos = range(len(tokens))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tokens, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)

    plt.tight_layout()
    attn_path = os.path.join(output_dir, "bert_attention_maps.png")
    plt.savefig(attn_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[bert_attn] attention maps → {attn_path}")

    # Per-head attention for early_game scenario (richer view)
    scenario_key = "early_game"
    text = SAMPLE_GAME_TEXTS[scenario_key]
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    with torch.no_grad():
        out = fresh_bert(**enc)

    attn_heads = out.attentions[0][0].numpy()  # [num_heads, seq, seq]
    num_heads = attn_heads.shape[0]
    cols = min(num_heads, 4)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn_heads[h], cmap="Blues", aspect="auto", vmin=0)
        ax.set_title(f"Head {h}", fontsize=9)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=5)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=5)

    for h in range(num_heads, len(axes)):
        axes[h].axis("off")

    plt.suptitle(
        f"TinyBERT per-head attention — {scenario_key}", fontsize=12, y=1.01
    )
    plt.tight_layout()
    per_head_path = os.path.join(output_dir, "bert_attention_per_head.png")
    plt.savefig(per_head_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[bert_attn] per-head attention → {per_head_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Action Embedding Similarity
# ══════════════════════════════════════════════════════════════════════════════

def analyze_action_similarity(
    net: torch.nn.Module,
    output_dir: str,
    action_token_path: str = "action_tokens/2p_action_ids.json",
) -> None:
    """
    Pass the pre-tokenized action descriptions through the frozen BERT encoder
    and compute pairwise cosine similarities between action embeddings.

    In drrn-lstm, Q(s, a) = fc_v(h) + fc_a(BERT(state) ⊙ BERT(action)), so
    this matrix directly shows how the model's dot-product scoring groups actions.

    Expected similarities:
      - All "play" actions should cluster together (same verb, different slot).
      - All "discard" actions should cluster together.
      - "hint colour" and "hint rank" may share structure.
      - Play vs discard may differ more than within-group pairs.
    """
    ensure_dir(output_dir)

    if not os.path.exists(action_token_path):
        print(
            f"[action_sim] {action_token_path} not found. "
            "Run from pyhanabi/ directory. Skipping."
        )
        return

    with open(action_token_path) as f:
        data = json.load(f)

    act_toks = torch.tensor(data["input_ids"])  # [num_actions, seq_len]
    num_actions = act_toks.size(0)
    labels = ACTION_LABELS_2P[:num_actions]

    with torch.no_grad():
        out = net.call_transformer(act_toks)
    embeds = out["last_hidden_state"].mean(dim=1).float()  # [num_actions, 128]

    # Normalise and compute full cosine similarity matrix
    normed = embeds / embeds.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim_mat = (normed @ normed.T).numpy()

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(num_actions))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(range(num_actions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(
        "Pairwise cosine similarity of BERT action embeddings\n"
        "Q(s,a) = fc_v(h) + fc_a(BERT(s) ⊙ BERT(a))   [drrn-lstm]"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    sim_path = os.path.join(output_dir, "action_embedding_similarity.png")
    plt.savefig(sim_path, dpi=120)
    plt.close()
    print(f"[action_sim] similarity heatmap → {sim_path}")

    # ── Most and least similar pairs (upper triangle only, no self) ───────────
    rows, cols = np.triu_indices(num_actions, k=1)
    pair_sims = sim_mat[rows, cols]
    sorted_idx = np.argsort(pair_sims)

    print("[action_sim] Top-5 most similar pairs (excluding self):")
    for idx in sorted_idx[-5:][::-1]:
        i, j = rows[idx], cols[idx]
        print(f"  {labels[i]:6s} ↔ {labels[j]:6s}  sim={pair_sims[idx]:.4f}")

    print("[action_sim] Top-5 most dissimilar pairs:")
    for idx in sorted_idx[:5]:
        i, j = rows[idx], cols[idx]
        print(f"  {labels[i]:6s} ↔ {labels[j]:6s}  sim={pair_sims[idx]:.4f}")

    # ── Embedding PCA (2D) ────────────────────────────────────────────────────
    from numpy.linalg import svd
    centered = embeds.numpy() - embeds.numpy().mean(0)
    _, _, Vt = svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T  # [num_actions, 2]

    type_colors = {
        "discard": "steelblue",
        "play": "coral",
        "hint": "seagreen",
        "noop": "grey",
    }
    fig, ax = plt.subplots(figsize=(8, 7))
    for idx, label in enumerate(labels):
        atype = action_type(idx)
        color = type_colors[atype]
        ax.scatter(proj[idx, 0], proj[idx, 1], color=color, s=60, zorder=3)
        ax.annotate(
            label, (proj[idx, 0], proj[idx, 1]),
            fontsize=8, ha="center", va="bottom",
        )
    # Legend
    for atype, color in type_colors.items():
        ax.scatter([], [], color=color, label=atype, s=60)
    ax.legend(loc="best")
    ax.set_title("Action embeddings — PCA projection (first 2 principal components)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()
    pca_path = os.path.join(output_dir, "action_embedding_pca.png")
    plt.savefig(pca_path, dpi=120)
    plt.close()
    print(f"[action_sim] action embedding PCA → {pca_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Collect Hidden States via the Game Environment
# ══════════════════════════════════════════════════════════════════════════════

def _collect_game_data(weight_file: str, cfg: dict, num_games: int, device: str) -> dict:
    """
    Run `num_games` self-play games using the C++ Hanabi environment, store
    episodes in a replay buffer, then replay each episode through BERT + LSTM
    manually to obtain per-step hidden states, values, and advantages.

    Returns a dict with numpy arrays:
      hidden      [N, 128]  — LSTM output at each timestep
      values      [N]       — fc_v(hidden) scalar
      max_advs    [N]       — max_action fc_a(hidden ⊙ action_embed)
      actions     [N]       — greedy action index taken by player 0
      action_types[N]       — 0=discard, 1=play, 2=hint
      timesteps   [N]       — within-episode step index
      cum_scores  [N]       — cumulative score (sum of rewards) up to this step
    """
    import time

    import hanalearn
    import rela

    from create import create_envs, create_threads

    # ── Load model ────────────────────────────────────────────────────────────
    agent, _ = load_model(weight_file, device)
    net = agent.online_net

    num_player = cfg["num_player"]
    max_len = cfg["max_len"]
    hid_dim = cfg["rnn_hid_dim"]
    num_lstm_layer = cfg["num_lstm_layer"]

    # ── Pre-compute action embeddings once (for drrn-lstm) ───────────────────
    act_tok_path = "action_tokens/5p_action_ids.json"  # net always loads 5p
    with open(act_tok_path) as f:
        act_data = json.load(f)
    act_toks = torch.tensor(act_data["input_ids"])  # [num_actions_5p, seq_len]
    with torch.no_grad():
        act_out = net.call_transformer(act_toks)
    act_embeds = act_out["last_hidden_state"].mean(dim=1).float()  # [A, 128]

    # ── Fill replay buffer ────────────────────────────────────────────────────
    num_thread = min(10, num_games)
    game_per_thread = max(1, num_games // num_thread)
    total_games = num_thread * game_per_thread

    runner = rela.BatchRunner(agent, device, 1000, ["act"])
    replay_buffer = rela.RNNReplay(total_games, 1, 0)

    games = create_envs(
        total_games,
        seed=42,
        num_player=num_player,
        bomb=0,
        max_len=max_len,
        hand_size=cfg.get("hand_size", 5),
        num_color=cfg.get("num_color", 5),
        num_rank=cfg.get("num_rank", 5),
        num_hint=cfg.get("num_hint", 8),
    )

    actors = []
    seed = 100
    for t in range(num_thread):
        thread_actors = []
        for p in range(num_player):
            seed += 1
            actor = hanalearn.R2D2Actor(
                runner,
                seed,
                num_player,
                p,
                True,                     # vdn-style — both players share replay
                False,                    # sad
                False,                    # shuffle_color
                False,                    # hide_action
                hanalearn.AuxType.Null,
                replay_buffer,
                1,                        # multi_step
                max_len,
                cfg["gamma"],
            )
            thread_actors.append(actor)
        for k in range(num_player):
            partners = thread_actors[:]
            partners[k] = None
            thread_actors[k].set_partners(partners)
        actors.append([thread_actors])

    context, _ = create_threads(num_thread, game_per_thread, actors, games)
    runner.start()
    context.start()

    print(f"[hidden_pca] collecting {total_games} games ...", flush=True)
    while replay_buffer.size() < total_games:
        time.sleep(0.2)
    context.terminate()
    print(f"[hidden_pca] replay buffer: {replay_buffer.size()} episodes")

    # ── Replay episodes manually through BERT + LSTM ──────────────────────────
    all_hidden, all_values, all_max_advs = [], [], []
    all_actions, all_action_types = [], []
    all_timesteps, all_cum_scores = [], []

    for ep_idx in range(replay_buffer.size()):
        epsd = replay_buffer.get(ep_idx)
        seq_len = int(epsd.seq_len.item())
        if seq_len < 2:
            continue

        obs = epsd.obs
        if "priv_s_text" not in obs:
            continue

        # priv_s_text shape from the C++ actor: [max_len, num_words, num_player]
        # (transposed relative to how td_error processes it)
        raw_text = obs["priv_s_text"][:seq_len]  # trim to actual length

        # Select player 0's token sequences
        # Infer which dim is num_player and which is num_words
        if raw_text.dim() == 3:
            d1, d2, d3 = raw_text.shape
            # Heuristic: num_player is small (2), num_words is large
            if d3 == num_player:
                p0_toks = raw_text[:, :, 0].long()  # [seq_len, num_words]
            elif d2 == num_player:
                p0_toks = raw_text[:, 0, :].long()  # [seq_len, num_words]
            else:
                # Fallback: assume [seq_len, num_words, num_player]
                p0_toks = raw_text[:, :, 0].long()
        elif raw_text.dim() == 2:
            p0_toks = raw_text.long()  # [seq_len, num_words]
        else:
            continue

        # BERT: process all timesteps in one batch
        with torch.no_grad():
            bert_out = net.call_transformer(p0_toks)
        x = bert_out["last_hidden_state"].mean(dim=1).float()  # [seq_len, 128]

        # LSTM: sequential forward (carry hidden state across steps)
        h = torch.zeros(num_lstm_layer, 1, hid_dim)
        c = torch.zeros(num_lstm_layer, 1, hid_dim)
        hidden_seq = []
        value_seq = []
        max_adv_seq = []

        with torch.no_grad():
            for t in range(seq_len):
                xt = x[t].unsqueeze(0).unsqueeze(1)  # [1, 1, 128]
                o, (h, c) = net.state_lstm(xt, (h, c))
                ht = o[0, 0]  # [128]
                hidden_seq.append(ht.numpy())
                value_seq.append(net.fc_v(ht).item())

                # Advantage over all actions: fc_a(ht ⊙ act_embed)
                state_tiled = ht.unsqueeze(0) * act_embeds  # [A, 128]
                advs = net.fc_a(state_tiled).squeeze(-1)    # [A]
                max_adv_seq.append(advs.max().item())

        # Actions taken by player 0
        act_raw = epsd.action["a"][:seq_len]   # [seq_len, num_player] or [seq_len]
        if act_raw.dim() == 2:
            acts_p0 = act_raw[:, 0].numpy()
        else:
            acts_p0 = act_raw.numpy()

        rewards = epsd.reward[:seq_len].numpy()
        cum = np.cumsum(rewards)

        all_hidden.append(np.array(hidden_seq))
        all_values.append(np.array(value_seq))
        all_max_advs.append(np.array(max_adv_seq))
        all_actions.append(acts_p0)
        all_action_types.append(
            np.array([{"discard": 0, "play": 1, "hint": 2, "noop": 3}[action_type(a)] for a in acts_p0])
        )
        all_timesteps.append(np.arange(seq_len))
        all_cum_scores.append(cum)

    return {
        "hidden": np.concatenate(all_hidden, axis=0),
        "values": np.concatenate(all_values, axis=0),
        "max_advs": np.concatenate(all_max_advs, axis=0),
        "actions": np.concatenate(all_actions, axis=0),
        "action_types": np.concatenate(all_action_types, axis=0),
        "timesteps": np.concatenate(all_timesteps, axis=0),
        "cum_scores": np.concatenate(all_cum_scores, axis=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Hidden State PCA
# ══════════════════════════════════════════════════════════════════════════════

def visualize_hidden_pca(data: dict, output_dir: str) -> None:
    """
    Project the collected LSTM hidden states onto their first two principal
    components and colour them by (a) timestep, (b) action type, (c) score.

    Interpretation guide:
      - Clustering by timestep → LSTM tracks how far into the game we are.
      - Clustering by action type → LSTM separates play/discard/hint contexts.
      - Clustering by score → LSTM encodes whether the game is going well.
    """
    ensure_dir(output_dir)

    hidden = data["hidden"]                    # [N, 128]
    centered = hidden - hidden.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pc = centered @ Vt[:2].T                   # [N, 2]

    explained = np.var(centered @ Vt.T, axis=0)
    explained /= explained.sum()
    print(
        f"[hidden_pca] PC1 explains {100*explained[0]:.1f}%, "
        f"PC2 {100*explained[1]:.1f}% of variance"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Colour by timestep
    sc = axes[0].scatter(pc[:, 0], pc[:, 1], c=data["timesteps"], cmap="viridis",
                         s=4, alpha=0.4)
    axes[0].set_title("Hidden state PCA — coloured by timestep")
    plt.colorbar(sc, ax=axes[0], label="Timestep within episode")
    axes[0].set_xlabel(f"PC 1 ({100*explained[0]:.1f}%)")
    axes[0].set_ylabel(f"PC 2 ({100*explained[1]:.1f}%)")

    # (b) Colour by action type
    atype_labels = ["discard", "play", "hint", "noop"]
    atype_colors = ["steelblue", "coral", "seagreen", "grey"]
    for at_idx, (at_name, at_color) in enumerate(zip(atype_labels, atype_colors)):
        mask = data["action_types"] == at_idx
        if mask.sum() == 0:
            continue
        axes[1].scatter(pc[mask, 0], pc[mask, 1], color=at_color, s=4,
                        alpha=0.4, label=at_name)
    axes[1].set_title("Hidden state PCA — coloured by action type")
    axes[1].legend(loc="best", markerscale=3)
    axes[1].set_xlabel(f"PC 1 ({100*explained[0]:.1f}%)")
    axes[1].set_ylabel(f"PC 2 ({100*explained[1]:.1f}%)")

    # (c) Colour by cumulative score
    sc = axes[2].scatter(pc[:, 0], pc[:, 1], c=data["cum_scores"], cmap="RdYlGn",
                         s=4, alpha=0.4, vmin=0, vmax=25)
    axes[2].set_title("Hidden state PCA — coloured by cumulative score")
    plt.colorbar(sc, ax=axes[2], label="Cumulative score (0–25)")
    axes[2].set_xlabel(f"PC 1 ({100*explained[0]:.1f}%)")
    axes[2].set_ylabel(f"PC 2 ({100*explained[1]:.1f}%)")

    plt.tight_layout()
    pca_path = os.path.join(output_dir, "hidden_state_pca.png")
    plt.savefig(pca_path, dpi=150)
    plt.close()
    print(f"[hidden_pca] hidden state PCA → {pca_path}")

    # Save PC loadings (which hidden dimensions drive PC1/PC2)
    pc1_top = np.argsort(np.abs(Vt[0]))[::-1][:10]
    pc2_top = np.argsort(np.abs(Vt[1]))[::-1][:10]
    with open(os.path.join(output_dir, "pca_loadings.txt"), "w") as f:
        f.write("Top-10 hidden dimensions driving PC1:\n")
        for d in pc1_top:
            f.write(f"  dim {d:3d}  loading={Vt[0, d]:+.4f}\n")
        f.write("\nTop-10 hidden dimensions driving PC2:\n")
        for d in pc2_top:
            f.write(f"  dim {d:3d}  loading={Vt[1, d]:+.4f}\n")
    print(f"[hidden_pca] PCA loadings → {output_dir}/pca_loadings.txt")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Linear Probing
# ══════════════════════════════════════════════════════════════════════════════

def linear_probe(data: dict, output_dir: str) -> None:
    """
    Train L2-regularised linear classifiers on the collected hidden states.

    Probes:
      A) Action type (3-class: discard / play / hint) — tests whether the
         hidden state encodes what kind of action will be taken next.
      B) Score bucket (low ≤ 10 / mid 11-19 / high ≥ 20) — tests whether
         the hidden state encodes overall game progress.
      C) Early vs late game (timestep < 20 vs ≥ 20) — tests temporal encoding.

    Each probe is evaluated with 5-fold cross-validation. Accuracy well above
    chance indicates the feature is linearly decodable from the hidden state.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[probing] scikit-learn not available — skipping. Install with: pip install scikit-learn")
        return

    ensure_dir(output_dir)
    hidden = data["hidden"]                    # [N, 128]

    scaler = StandardScaler()
    H = scaler.fit_transform(hidden)

    def probe(X, y, label, chance, report_file):
        # Exclude noop (class 3) from action probes
        mask = y < 3
        X_filtered, y_filtered = X[mask], y[mask]
        if len(np.unique(y_filtered)) < 2:
            report_file.write(f"{label}: not enough classes\n")
            return

        clf = LogisticRegression(max_iter=1000, C=0.1, multi_class="auto")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(clf, X_filtered, y_filtered, cv=cv, scoring="accuracy")
        mean_acc = scores.mean()
        std_acc = scores.std()
        line = (
            f"{label}: acc = {mean_acc:.3f} ± {std_acc:.3f}  "
            f"(chance = {chance:.3f})\n"
        )
        print(f"[probing] {line.strip()}")
        report_file.write(line)

    with open(os.path.join(output_dir, "probing_results.txt"), "w") as f:
        f.write("Linear probing results (5-fold cross-validation)\n")
        f.write("=" * 55 + "\n\n")

        # A) Action type
        action_types = data["action_types"]
        counts = np.bincount(action_types[action_types < 3], minlength=3)
        chance_a = counts.max() / counts.sum()
        probe(H, action_types, "Action type (discard/play/hint)", chance_a, f)

        # B) Score bucket
        scores_arr = data["cum_scores"]
        buckets = np.where(scores_arr <= 10, 0, np.where(scores_arr <= 19, 1, 2))
        bcounts = np.bincount(buckets, minlength=3)
        chance_b = bcounts.max() / bcounts.sum()
        probe(H, buckets, "Score bucket (low/mid/high)", chance_b, f)

        # C) Early vs late game
        early_late = (data["timesteps"] >= 20).astype(int)
        chance_c = max(early_late.mean(), 1 - early_late.mean())
        probe(H, early_late, "Game phase (early t<20 / late t≥20)", chance_c, f)

    print(f"[probing] results → {output_dir}/probing_results.txt")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Q-value Decomposition
# ══════════════════════════════════════════════════════════════════════════════

def analyze_qvalues(data: dict, output_dir: str) -> None:
    """
    Plot the value head (fc_v) and max-advantage head (fc_a) as a function of
    game timestep, averaged over all collected episodes.

    Interpretation:
      - Value rising as the game progresses → agent becomes more certain of
        a good outcome as fireworks are built.
      - Large advantage spread → agent has a clear preferred action.
      - Advantage collapsing late → few legal moves remain.
    """
    ensure_dir(output_dir)

    max_t = 80
    val_by_t = [[] for _ in range(max_t)]
    adv_by_t = [[] for _ in range(max_t)]

    for t, v, a in zip(data["timesteps"], data["values"], data["max_advs"]):
        if t < max_t:
            val_by_t[t].append(v)
            adv_by_t[t].append(a)

    ts = [t for t in range(max_t) if val_by_t[t]]
    val_means = np.array([np.mean(val_by_t[t]) for t in ts])
    val_stds = np.array([np.std(val_by_t[t]) for t in ts])
    adv_means = np.array([np.mean(adv_by_t[t]) for t in ts])
    adv_stds = np.array([np.std(adv_by_t[t]) for t in ts])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(ts, val_means, color="steelblue", linewidth=1.5, label="Mean value")
    ax1.fill_between(ts, val_means - val_stds, val_means + val_stds,
                     alpha=0.25, color="steelblue")
    ax1.set_ylabel("fc_v(h)  — state value")
    ax1.set_title("Value head across timesteps  (mean ± 1 std over all episodes)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ts, adv_means, color="coral", linewidth=1.5, label="Mean max advantage")
    ax2.fill_between(ts, adv_means - adv_stds, adv_means + adv_stds,
                     alpha=0.25, color="coral")
    ax2.set_xlabel("Timestep within episode")
    ax2.set_ylabel("max_a  fc_a(h ⊙ BERT(a))  — best advantage")
    ax2.set_title("Max advantage across timesteps  (mean ± 1 std over all episodes)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    qval_path = os.path.join(output_dir, "qvalue_over_time.png")
    plt.savefig(qval_path, dpi=120)
    plt.close()
    print(f"[qval] Q-value decomposition → {qval_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpretability analysis for R3D2 Hanabi checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--weight", required=True,
        help="Path to the .pthw checkpoint file, e.g. "
             "../final_r3d2_ckpts/R3D2-2p/a/epoch3000.pthw",
    )
    parser.add_argument(
        "--output_dir", default="../interpret_results",
        help="Directory to write all output plots and text files.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device (cpu or cuda:0).",
    )
    parser.add_argument(
        "--analyses",
        default="weights,bert_attn,action_sim",
        help=(
            "Comma-separated list of analyses to run. "
            "Options: weights, bert_attn, action_sim, hidden_pca, probing, qval. "
            "hidden_pca/probing/qval require the compiled C++ game environment."
        ),
    )
    parser.add_argument(
        "--num_games", type=int, default=200,
        help="Number of games to collect for hidden_pca / probing / qval.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyses = {a.strip() for a in args.analyses.split(",")}
    ensure_dir(args.output_dir)

    print(f"Loading checkpoint: {args.weight}")
    agent, cfg = load_model(args.weight, args.device)
    agent.train(False)
    net = agent.online_net
    print(
        f"Model: net={cfg.get('net')}  "
        f"num_player={cfg.get('num_player')}  "
        f"rnn_hid_dim={cfg.get('rnn_hid_dim')}  "
        f"num_lstm_layer={cfg.get('num_lstm_layer')}  "
        f"num_lm_layer={cfg.get('num_lm_layer', 1)}  "
        f"lm_weights={cfg.get('lm_weights')}"
    )

    # Load raw state dict (needed for BERT weight extraction)
    full_state_dict = None
    if "bert_attn" in analyses:
        full_state_dict = torch.load(args.weight, map_location="cpu")
        # Handle checkpoints compiled with torch.compile
        from collections import OrderedDict
        cleaned = OrderedDict()
        for k, v in full_state_dict.items():
            cleaned[k.replace("_orig_mod.", "")] = v
        full_state_dict = cleaned

    # ── Analyses that do not require the game environment ─────────────────────

    if "weights" in analyses:
        print("\n── Analysis 1: Weight structure ──────────────────────────────")
        analyze_weights(net, args.output_dir)

    if "bert_attn" in analyses:
        print("\n── Analysis 2: BERT attention ────────────────────────────────")
        analyze_bert_attention(
            net, full_state_dict, args.output_dir,
            num_lm_layer=cfg.get("num_lm_layer", 1),
        )

    if "action_sim" in analyses:
        print("\n── Analysis 3: Action embedding similarity ───────────────────")
        analyze_action_similarity(net, args.output_dir)

    # ── Analyses that require the compiled C++ game environment ───────────────
    needs_game_env = {"hidden_pca", "probing", "qval"}
    if analyses & needs_game_env:
        try:
            import rela  # noqa: F401
            import hanalearn  # noqa: F401
        except ImportError:
            print(
                "\n[WARNING] rela / hanalearn C++ bindings not found. "
                "Skipping hidden_pca, probing, and qval analyses.\n"
                "Build the project with `make` from the repo root first."
            )
            return

        print("\n── Collecting game data ──────────────────────────────────────")
        game_data = _collect_game_data(
            args.weight, cfg, num_games=args.num_games, device=args.device
        )
        print(
            f"Collected {len(game_data['hidden'])} timesteps "
            f"from {args.num_games} games."
        )

        if "hidden_pca" in analyses:
            print("\n── Analysis 4: Hidden state PCA ──────────────────────────────")
            visualize_hidden_pca(game_data, args.output_dir)

        if "probing" in analyses:
            print("\n── Analysis 5: Linear probing ────────────────────────────────")
            linear_probe(game_data, args.output_dir)

        if "qval" in analyses:
            print("\n── Analysis 6: Q-value decomposition ────────────────────────")
            analyze_qvalues(game_data, args.output_dir)

    print(f"\nDone. All outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
