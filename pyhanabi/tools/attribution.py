#!/usr/bin/env python3
"""
Attribution analysis for R3D2 Hanabi agents (TextLSTMNet / drrn-lstm).

Answers the question:
  "How much does each part of the game state (opponent's hand, hint tokens,
  life tokens, fireworks, discards) influence the Q-value of each action?"

Methods
-------
  Saliency       — gradient of Q(s,a) w.r.t. the BERT input token embeddings,
                   then L2-normed over the embedding dimension per token:
                       saliency[t] = ||∂Q / ∂e_t||₂

  Integrated     — Riemann approximation of Sundararajan et al. (2017):
  Gradients (IG)     IG[t] = (e_t - e_t^0) · (1/N) Σ_k ∂Q/∂e_t |_{α_k}
                   Baseline e^0 = zero vector; N = 50 steps.
                   Attribution per token is the sum over embedding dims.
                   Signed, so positive = pushes Q up; negative = pushes Q down.

  Both methods operate on the BERT input embeddings (word + position +
  token-type), so the gradient flows:
      Q  →  fc_a / fc_v  →  LSTM hidden  →  LSTM  →  mean-pooled BERT output
         →  BERT encoder  →  BERT input embeddings

Figures produced
----------------
  fig1_token_heatmaps.png      — per-token saliency and IG for the greedy
                                  action in each game scenario
  fig2_component_bars.png      — section-level aggregated attribution for
                                  4 action types in each game scenario
  fig3_cross_scenario.png      — how each component's importance shifts from
                                  early → mid → late game (IG, greedy action)

Usage (run from pyhanabi/)
--------------------------
    python tools/attribution.py \\
        --weight ../final_r3d2_ckpts/R3D2-2p/a/epoch3000.pthw \\
        --output_dir ../attribution_results
"""

import argparse
import json
import os
import pickle
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, lib_path)

import r2d2 as r2d2_module

# ══════════════════════════════════════════════════════════════════════════════
# Model loading  (no dependency on create.py / set_path.py)
# ══════════════════════════════════════════════════════════════════════════════

def _load_cfg(weight_file: str) -> dict:
    cfg_file = weight_file + ".cfg"
    if os.path.exists(cfg_file):
        return pickle.load(open(cfg_file, "rb"))
    log = os.path.join(os.path.dirname(weight_file), "train.log")
    if not os.path.exists(log):
        raise FileNotFoundError(f"No .cfg or train.log found near {weight_file}")
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
    state_dict = torch.load(weight_file, map_location=device)
    state_dict = OrderedDict(
        (k.replace("_orig_mod.", ""), v) for k, v in state_dict.items()
    )
    target = model.state_dict()
    filtered = OrderedDict()
    for k, v in target.items():
        if k not in state_dict:
            filtered[k] = v
        elif state_dict[k].size() != v.size():
            filtered[k] = v
        else:
            filtered[k] = state_dict[k]
    model.load_state_dict(filtered)


def load_model(weight_file: str, device: str = "cpu"):
    """Load an R3D2 checkpoint. Returns (agent, cfg)."""
    cfg = _load_cfg(weight_file)
    agent = r2d2_module.R2D2Agent(
        vdn=False,
        multi_step=cfg.get("multi_step", 1),
        gamma=cfg.get("gamma", 0.999),
        device=device,
        in_dim=None,
        hid_dim=cfg["rnn_hid_dim"],
        out_dim=1,
        net=cfg.get("net", "drrn-lstm"),
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


# ══════════════════════════════════════════════════════════════════════════════
# Game state scenarios — each section is a separately-identifiable substring
# ══════════════════════════════════════════════════════════════════════════════

# Human-readable labels for the plot axes
SECTION_LABELS = OrderedDict([
    ("life_tokens", "Life tokens"),
    ("info_tokens", "Hint tokens"),
    ("fireworks",   "Fireworks"),
    ("own_hand",    "Own hand"),
    ("opp_hand",    "Opp. hand"),
    ("discards",    "Discards"),
    ("last_action", "Last action"),
])

# Each scenario is an OrderedDict so the full text can be built by joining
# the values, and span positions can be tracked by substring search.
GAME_SCENARIOS: dict[str, OrderedDict] = {
    "early_game": OrderedDict([
        ("life_tokens", "Life tokens: 3."),
        ("info_tokens", "Information tokens: 8."),
        ("fireworks",   "Fireworks: R0 Y0 G0 W0 B0."),
        ("own_hand",    "Player 0 (me): ?? ?? ?? ?? ??."),
        ("opp_hand",    "Player 1: R1 Y2 G3 W4 B1."),
        ("discards",    "Discards: none."),
        ("last_action", "Last action: none."),
    ]),
    "mid_game": OrderedDict([
        ("life_tokens", "Life tokens: 2."),
        ("info_tokens", "Information tokens: 3."),
        ("fireworks",   "Fireworks: R2 Y1 G2 W1 B1."),
        ("own_hand",    "Player 0 (me): ?? ?? ?? ?? ??."),
        ("opp_hand",    "Player 1: R3 Y2 G3 W2 B2."),
        ("discards",    "Discards: R1 Y1 G1 W1."),
        ("last_action", "Last action: player 1 hinted rank 2."),
    ]),
    "late_game": OrderedDict([
        ("life_tokens", "Life tokens: 1."),
        ("info_tokens", "Information tokens: 1."),
        ("fireworks",   "Fireworks: R4 Y4 G4 W4 B3."),
        ("own_hand",    "Player 0 (me): ?? ?? ?? ?? ??."),
        ("opp_hand",    "Player 1: R5 Y5 G5 W5 B5."),
        ("discards",    "Discards: R1 Y1 G1 W1 B1 R2 Y2 G2."),
        ("last_action", "Last action: player 1 played B3."),
    ]),
}

# Representative actions to analyse (2-player indices)
PROBE_ACTIONS = OrderedDict([
    ("greedy", None),   # filled in at runtime
    ("D1",     0),      # discard card 0
    ("P1",     5),      # play card 0
    ("CR",     10),     # hint colour Red to player 1
    ("R1",     15),     # hint rank 1 to player 1
])


# ══════════════════════════════════════════════════════════════════════════════
# BERT helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_fresh_bert(full_state_dict: dict, num_lm_layer: int = 1):
    """
    Copy TinyBERT weights from the checkpoint into a non-traced BertModel so
    we can compute gradients through it and pass `inputs_embeds` directly.
    """
    from transformers import BertConfig, BertModel, BertTokenizer

    pretrained_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    config = BertConfig.from_pretrained(pretrained_name)
    config.num_hidden_layers = num_lm_layer

    bert = BertModel(config)
    prefix = "call_transformer."
    bert_sd = {
        k[len(prefix):]: v
        for k, v in full_state_dict.items()
        if k.startswith(prefix)
    }
    bert.load_state_dict(bert_sd, strict=False)
    bert.eval()

    tokenizer = BertTokenizer.from_pretrained(pretrained_name)
    return bert, tokenizer


def build_fresh_heads(full_state_dict: dict, hid_dim: int, num_lstm_layer: int):
    """
    Extract the LSTM and output-head weights from the checkpoint state dict
    into plain nn.Module objects (not TorchScript).

    This is necessary because TextLSTMNet is a ScriptModule, and calling
    net.state_lstm / net.fc_v / net.fc_a directly in autograd context raises
    AttributeError('RecursiveScriptModule has no attribute forward').

    Returns: lstm, fc_v, fc_a  — all plain nn.Module, eval mode.
    """
    import torch.nn as nn

    lstm = nn.LSTM(hid_dim, hid_dim, num_layers=num_lstm_layer)
    lstm_sd = OrderedDict()
    for k, v in full_state_dict.items():
        if k.startswith("state_lstm."):
            lstm_sd[k[len("state_lstm."):]] = v
    lstm.load_state_dict(lstm_sd)
    lstm.eval()

    fc_v = nn.Linear(hid_dim, 1)
    fc_v.weight = torch.nn.Parameter(full_state_dict["fc_v.weight"].clone())
    fc_v.bias   = torch.nn.Parameter(full_state_dict["fc_v.bias"].clone())
    fc_v.eval()

    fc_a = nn.Linear(hid_dim, 1)
    fc_a.weight = torch.nn.Parameter(full_state_dict["fc_a.weight"].clone())
    fc_a.bias   = torch.nn.Parameter(full_state_dict["fc_a.bias"].clone())
    fc_a.eval()

    return lstm, fc_v, fc_a


def get_input_embeddings(bert, tokenizer, text: str):
    """
    Tokenize `text` and return the full BERT input embeddings (word +
    position + token-type), detached but with requires_grad set after the
    return so the caller can backpropagate through them.

    Returns:
        input_ids   : LongTensor [1, seq_len]
        tokens      : list[str]  length seq_len
        embeds      : FloatTensor [1, seq_len, 128]  — cloned, no grad yet
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    with torch.no_grad():
        embeds = bert.embeddings(input_ids).detach().clone()
    return input_ids, tokens, embeds


def forward_from_embeds(bert, lstm, fc_v, fc_a, act_embeds, embeds, action_idx,
                        num_lstm_layer: int, hid_dim: int):
    """
    Full forward pass starting from BERT input embeddings (after the
    embedding lookup), enabling gradient computation w.r.t. `embeds`.

    Pipeline:
        embeds  →  BERT encoder  →  mean-pool  →  LSTM  →  fc_v / fc_a  →  Q

    lstm, fc_v, fc_a are plain nn.Module objects (not TorchScript), extracted
    from the checkpoint via build_fresh_heads().

    Returns scalar Q-value for `action_idx`.
    """
    seq_len = embeds.size(1)
    attention_mask = torch.ones(1, seq_len)
    out = bert(inputs_embeds=embeds, attention_mask=attention_mask)
    pooled = out.last_hidden_state.mean(dim=1)          # [1, 128]

    x = pooled.unsqueeze(0)                             # [1, 1, 128]
    h0 = torch.zeros(num_lstm_layer, 1, hid_dim)
    c0 = torch.zeros(num_lstm_layer, 1, hid_dim)
    lstm_out, _ = lstm(x, (h0, c0))                    # [1, 1, 128]
    h = lstm_out[0, 0]                                  # [128]

    v = fc_v(h)                                         # [1]
    adv = fc_a(h * act_embeds[action_idx])              # [1]
    return (v + adv).squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# Span detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_spans(tokenizer, full_text: str, sections: OrderedDict) -> dict:
    """
    For each section text, encode it independently with the same tokenizer
    and search for the exact token-ID subsequence inside the full tokenisation.

    Returns dict {section_name: (start_idx, end_idx)} using Python slice
    conventions (end_idx is exclusive).  Missing sections map to None.
    """
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)
    spans = {}
    search_from = 1  # skip [CLS]

    for name, snippet in sections.items():
        snip_ids = tokenizer.encode(snippet, add_special_tokens=False)
        n = len(snip_ids)
        found = False
        for i in range(search_from, len(full_ids) - n + 1):
            if full_ids[i : i + n] == snip_ids:
                spans[name] = (i, i + n)
                search_from = i + n   # next section must start after this one
                found = True
                break
        if not found:
            spans[name] = None

    return spans


# ══════════════════════════════════════════════════════════════════════════════
# Saliency
# ══════════════════════════════════════════════════════════════════════════════

def compute_saliency(bert, lstm, fc_v, fc_a, act_embeds, embeds_base,
                     action_idx, num_lstm_layer: int, hid_dim: int) -> np.ndarray:
    """
    Gradient-based saliency: ||∂Q / ∂e_t||₂  for each token t.

    Returns: np.ndarray shape [seq_len], all non-negative.
    """
    embeds = embeds_base.detach().clone().requires_grad_(True)
    q = forward_from_embeds(bert, lstm, fc_v, fc_a, act_embeds, embeds,
                            action_idx, num_lstm_layer, hid_dim)
    q.backward()
    saliency = embeds.grad.norm(dim=-1).squeeze().detach().numpy()
    return saliency


# ══════════════════════════════════════════════════════════════════════════════
# Integrated Gradients
# ══════════════════════════════════════════════════════════════════════════════

def compute_ig(bert, lstm, fc_v, fc_a, act_embeds, embeds_base, action_idx,
               num_lstm_layer: int, hid_dim: int,
               n_steps: int = 50) -> np.ndarray:
    """
    Integrated Gradients with a zero-vector baseline.

    IG[t] = (e_t - 0) · (1/N) Σ_k  ∂Q/∂e_t  evaluated at  α_k · e_t

    Attribution per token is the sum over the embedding dimension (signed).
    Positive → pushes Q up; negative → pushes Q down.

    Returns: np.ndarray shape [seq_len], signed.
    """
    baseline = torch.zeros_like(embeds_base)
    e = embeds_base.detach()

    accumulated_grads = torch.zeros_like(e)
    for k in range(n_steps):
        alpha = k / (n_steps - 1)
        interp = (baseline + alpha * (e - baseline)).requires_grad_(True)
        q = forward_from_embeds(bert, lstm, fc_v, fc_a, act_embeds, interp,
                                action_idx, num_lstm_layer, hid_dim)
        q.backward()
        accumulated_grads += interp.grad.detach()

    # IG formula: (inputs - baseline) * mean gradient
    ig_per_dim = (e - baseline) * (accumulated_grads / n_steps)  # [1, seq, 128]
    ig = ig_per_dim.squeeze(0).sum(dim=-1).numpy()               # [seq_len], signed
    return ig


# ══════════════════════════════════════════════════════════════════════════════
# Greedy action (highest Q-value across all legal actions)
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_qvalues(bert, lstm, fc_v, fc_a, act_embeds, embeds_base,
                        num_lstm_layer: int, hid_dim: int) -> np.ndarray:
    """Return Q-values for all actions as a numpy array."""
    with torch.no_grad():
        seq_len = embeds_base.size(1)
        attention_mask = torch.ones(1, seq_len)
        out = bert(inputs_embeds=embeds_base, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        x = pooled.unsqueeze(0)
        h0 = torch.zeros(num_lstm_layer, 1, hid_dim)
        c0 = torch.zeros(num_lstm_layer, 1, hid_dim)
        lstm_out, _ = lstm(x, (h0, c0))
        h = lstm_out[0, 0]
        v = fc_v(h)
        qs = []
        for a in range(act_embeds.size(0)):
            adv = fc_a(h * act_embeds[a])
            qs.append((v + adv).item())
    return np.array(qs)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate attribution over token spans → section scores
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_by_section(attribution: np.ndarray, spans: dict) -> dict:
    """
    Sum |attribution| over each section's token span.
    Returns dict {section_name: float}.
    """
    scores = {}
    for name, span in spans.items():
        if span is None:
            scores[name] = 0.0
        else:
            s, e = span
            scores[name] = float(np.abs(attribution[s:e]).sum())
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Token-level heatmaps (saliency + IG, greedy action, 3 scenarios)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_token_bar(ax, tokens, values, title, color, spans=None):
    """Draw a horizontal bar chart of per-token attribution values."""
    n = len(tokens)
    y = np.arange(n)
    ax.barh(y, values, color=color, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(tokens, fontsize=6)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Attribution", fontsize=8)

    # Shade section backgrounds
    if spans:
        colors_bg = ["#ffe0e0", "#fff3e0", "#e0ffe0", "#e0e8ff",
                     "#f3e0ff", "#ffeedd", "#e8f8ff"]
        for (name, span), bg in zip(spans.items(), colors_bg):
            if span is None:
                continue
            s, e = span
            ax.axhspan(s - 0.5, e - 0.5, color=bg, alpha=0.4, zorder=0)


def plot_token_heatmaps(results: dict, output_dir: str) -> None:
    """
    Figure 1: for each scenario (row) × method (column), draw a horizontal
    bar chart showing per-token attribution for the greedy action.
    """
    scenarios = list(results.keys())
    n_rows = len(scenarios)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 8 * n_rows))

    method_colors = {"saliency": "steelblue", "ig": "coral"}

    for row, scenario in enumerate(scenarios):
        r = results[scenario]
        tokens = r["tokens"]
        greedy_idx = r["greedy_action_idx"]
        greedy_label = r["greedy_action_label"]
        spans = r["spans"]

        for col, (method, color) in enumerate(method_colors.items()):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            vals = r[method]["greedy"]
            # For saliency always positive; for IG show signed
            _draw_token_bar(
                ax, tokens, vals,
                title=(
                    f"{scenario}  |  {method.upper()}"
                    f"  |  greedy = {greedy_label} (idx {greedy_idx})"
                ),
                color=color,
                spans=spans,
            )

    plt.suptitle(
        "Per-token attribution for the greedy action\n"
        "(shaded bands = game-state sections)",
        fontsize=12, y=1.005,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "fig1_token_heatmaps.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[fig1] token heatmaps → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Section-level attribution by action type (one panel per scenario)
# ══════════════════════════════════════════════════════════════════════════════

def plot_component_bars(results: dict, output_dir: str) -> None:
    """
    Figure 2: for each scenario, show a grouped bar chart of section-level
    IG attribution for each probe action type.
    """
    scenarios = list(results.keys())
    n_scenarios = len(scenarios)
    section_names = list(SECTION_LABELS.keys())
    section_display = list(SECTION_LABELS.values())
    probe_names = ["greedy", "D1", "P1", "CR", "R1"]
    action_colors = {
        "greedy": "black",
        "D1": "steelblue",
        "P1": "coral",
        "CR": "seagreen",
        "R1": "purple",
    }

    fig, axes = plt.subplots(1, n_scenarios, figsize=(7 * n_scenarios, 6),
                             sharey=False)
    if n_scenarios == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        r = results[scenario]
        n_sections = len(section_names)
        n_probes = len(probe_names)
        width = 0.15
        x = np.arange(n_sections)

        for i, probe in enumerate(probe_names):
            scores = r["ig_by_section"][probe]       # dict {section: float}
            vals = np.array([scores.get(s, 0.0) for s in section_names])
            # Normalise per probe so all probes are on the same scale
            total = vals.sum()
            if total > 0:
                vals = vals / total
            ax.bar(
                x + (i - n_probes / 2 + 0.5) * width,
                vals,
                width,
                label=probe,
                color=action_colors[probe],
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(section_display, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{scenario}", fontsize=11)
        ax.set_ylabel("Normalised IG attribution", fontsize=9)
        ax.legend(title="Action", fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Section-level Integrated Gradients by action type\n"
        "(normalised so each action sums to 1.0)",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_component_bars.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[fig2] component bars → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Cross-scenario shift for the greedy action
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_scenario(results: dict, output_dir: str) -> None:
    """
    Figure 3: how much each section contributes to the greedy Q-value
    attribution across early / mid / late game.

    Left panel : absolute normalised IG per section (stacked area)
    Right panel: line chart — same data, one line per section
    """
    scenarios = list(results.keys())
    section_names = list(SECTION_LABELS.keys())
    section_display = list(SECTION_LABELS.values())

    # Build matrix [n_scenarios, n_sections]
    mat = np.zeros((len(scenarios), len(section_names)))
    for si, scenario in enumerate(scenarios):
        scores = results[scenario]["ig_by_section"]["greedy"]
        vals = np.array([scores.get(s, 0.0) for s in section_names])
        total = vals.sum()
        if total > 0:
            vals = vals / total
        mat[si] = vals

    section_colors = [
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
        "#f4a261", "#264653", "#a8dadc",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked area
    x = np.arange(len(scenarios))
    bottom = np.zeros(len(scenarios))
    for j, (name, color) in enumerate(zip(section_display, section_colors)):
        ax1.bar(x, mat[:, j], bottom=bottom, label=name,
                color=color, alpha=0.85, width=0.6)
        bottom += mat[:, j]
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.set_ylabel("Normalised IG attribution (stacked)")
    ax1.set_title("Section importance across game phases\n(greedy action, stacked)")
    ax1.legend(loc="upper right", fontsize=8)

    # Line chart
    for j, (name, color) in enumerate(zip(section_display, section_colors)):
        ax2.plot(scenarios, mat[:, j], marker="o", label=name,
                 color=color, linewidth=2)
    ax2.set_ylabel("Normalised IG attribution")
    ax2.set_title("Section importance across game phases\n(greedy action, per-section)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        "How each game-state component influences the greedy Q-value\n"
        "as the game progresses",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_cross_scenario.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[fig3] cross-scenario shift → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Signed IG: which sections push Q up vs down
# ══════════════════════════════════════════════════════════════════════════════

def plot_signed_ig(results: dict, output_dir: str) -> None:
    """
    Figure 4: signed (not absolute) section-level IG for the greedy action
    across scenarios.  Positive = section pushes Q up; negative = pushes down.
    """
    scenarios = list(results.keys())
    section_names = list(SECTION_LABELS.keys())
    section_display = list(SECTION_LABELS.values())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5),
                             sharey=False)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        scores = results[scenario]["ig_signed_by_section"]["greedy"]
        vals = np.array([scores.get(s, 0.0) for s in section_names])
        colors = ["coral" if v < 0 else "steelblue" for v in vals]
        x = np.arange(len(section_names))
        ax.bar(x, vals, color=colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(section_display, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{scenario}", fontsize=11)
        ax.set_ylabel("Signed IG sum")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Signed Integrated Gradients by section — greedy action\n"
        "(blue = increases Q-value, red = decreases Q-value)",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_signed_ig.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[fig4] signed IG → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main driver
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weight", required=True,
                   help="Path to .pthw checkpoint, e.g. "
                        "../final_r3d2_ckpts/R3D2-2p/a/epoch3000.pthw")
    p.add_argument("--output_dir", default="../attribution_results",
                   help="Directory for output figures.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ig_steps", type=int, default=50,
                   help="Number of steps for Integrated Gradients.")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.weight}")
    agent, cfg = load_model(args.weight, args.device)
    net = agent.online_net

    hid_dim = cfg["rnn_hid_dim"]
    num_lstm_layer = cfg["num_lstm_layer"]
    num_lm_layer = cfg.get("num_lm_layer", 1)

    # ── Load raw state dict for fresh BERT ─────────────────────────────────────
    raw_sd = torch.load(args.weight, map_location="cpu")
    full_sd = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in raw_sd.items())

    # ── Build fresh (non-traced) modules for gradient computation ──────────────
    print("Building fresh BertModel and head modules for gradient computation ...")
    bert, tokenizer = build_fresh_bert(full_sd, num_lm_layer=num_lm_layer)
    lstm, fc_v, fc_a = build_fresh_heads(full_sd, hid_dim, num_lstm_layer)

    # ── Pre-compute action embeddings (2p actions) ─────────────────────────────
    act_tok_path = "action_tokens/2p_action_ids.json"
    if not os.path.exists(act_tok_path):
        raise FileNotFoundError(
            f"{act_tok_path} not found. Run this script from pyhanabi/."
        )
    with open(act_tok_path) as f:
        act_data = json.load(f)
    act_toks = torch.tensor(act_data["input_ids"])  # [num_actions, seq_len]
    with torch.no_grad():
        act_out = bert(input_ids=act_toks)
    act_embeds = act_out.last_hidden_state.mean(dim=1).float()  # [num_actions, 128]

    num_actions = act_embeds.size(0)
    # Action labels: D1-D5, P1-P5, CR-CB, R1-R5, noop
    action_labels = (
        [f"D{i+1}" for i in range(5)]
        + [f"P{i+1}" for i in range(5)]
        + [f"C{c}" for c in "RYGWB"]
        + [f"R{r}" for r in "12345"]
        + ["noop"]
    )[:num_actions]

    # ── Run attribution for each scenario ──────────────────────────────────────
    results = {}

    for scenario, sections in GAME_SCENARIOS.items():
        print(f"\n── Scenario: {scenario} ──────────────────────────────────────")
        full_text = " ".join(sections.values())

        # Tokenize + spans
        input_ids, tokens, embeds_base = get_input_embeddings(bert, tokenizer, full_text)
        spans = detect_spans(tokenizer, full_text, sections)
        missing = [n for n, s in spans.items() if s is None]
        if missing:
            print(f"  [warn] could not locate sections: {missing}")

        # Greedy action
        qvals = compute_all_qvalues(bert, lstm, fc_v, fc_a, act_embeds,
                                    embeds_base, num_lstm_layer, hid_dim)
        greedy_idx = int(np.argmax(qvals))
        greedy_label = action_labels[greedy_idx] if greedy_idx < len(action_labels) else str(greedy_idx)
        print(f"  greedy action = {greedy_label} (idx {greedy_idx})  "
              f"Q = {qvals[greedy_idx]:.4f}")
        print(f"  Q-values: " + "  ".join(f"{action_labels[i]}={qvals[i]:.3f}"
                                           for i in range(min(21, len(qvals)))))

        # Build probe action map for this scenario
        probe_map = dict(PROBE_ACTIONS)
        probe_map["greedy"] = greedy_idx

        # Compute attributions for all probe actions
        saliency_results = {}
        ig_results = {}
        ig_signed_results = {}

        for probe_name, probe_idx in probe_map.items():
            if probe_idx is None or probe_idx >= num_actions:
                continue
            print(f"  computing saliency for {probe_name} (idx {probe_idx}) ...",
                  end=" ", flush=True)
            saliency_results[probe_name] = compute_saliency(
                bert, lstm, fc_v, fc_a, act_embeds, embeds_base, probe_idx,
                num_lstm_layer, hid_dim,
            )
            print("done  |  IG ...", end=" ", flush=True)
            ig_vals = compute_ig(
                bert, lstm, fc_v, fc_a, act_embeds, embeds_base, probe_idx,
                num_lstm_layer, hid_dim, n_steps=args.ig_steps,
            )
            ig_results[probe_name] = ig_vals
            ig_signed_results[probe_name] = ig_vals  # keep signed for fig4
            print("done")

        # Aggregate by section (absolute IG for fig2/fig3, signed for fig4)
        ig_by_section = {}
        ig_signed_by_section = {}
        for probe_name, ig_vals in ig_results.items():
            ig_by_section[probe_name] = aggregate_by_section(ig_vals, spans)
            # Signed: sum (not abs) over span
            signed_scores = {}
            for name, span in spans.items():
                if span is None:
                    signed_scores[name] = 0.0
                else:
                    s, e = span
                    signed_scores[name] = float(ig_vals[s:e].sum())
            ig_signed_by_section[probe_name] = signed_scores

        results[scenario] = {
            "tokens": tokens,
            "spans": spans,
            "greedy_action_idx": greedy_idx,
            "greedy_action_label": greedy_label,
            "qvals": qvals,
            "saliency": saliency_results,
            "ig": ig_results,
            "ig_by_section": ig_by_section,
            "ig_signed_by_section": ig_signed_by_section,
        }

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\n── Generating figures ────────────────────────────────────────────")
    plot_token_heatmaps(results, args.output_dir)
    plot_component_bars(results, args.output_dir)
    plot_cross_scenario(results, args.output_dir)
    plot_signed_ig(results, args.output_dir)

    # ── Print text summary ─────────────────────────────────────────────────────
    print("\n── Attribution summary (IG, greedy action) ───────────────────────")
    header = f"{'Section':15s}" + "".join(f"  {s:10s}" for s in results)
    print(header)
    print("-" * len(header))
    for name, display in SECTION_LABELS.items():
        row = f"{display:15s}"
        for scenario in results:
            score = results[scenario]["ig_by_section"].get("greedy", {}).get(name, 0.0)
            row += f"  {score:10.4f}"
        print(row)

    print(f"\nDone. Figures written to: {args.output_dir}")


if __name__ == "__main__":
    main()
