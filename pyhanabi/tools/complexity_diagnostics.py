#!/usr/bin/env python3
"""
complexity_diagnostics.py  —  policy complexity diagnostics for R3D2 Hanabi agents.

Measures how "spread out" the agent's attention is across game-state features.
High spread means complex conventions that human partners cannot learn or remember.

Complexity metrics
------------------
  Attribution entropy   — H(p) = −Σ pᵢ log₂ pᵢ  (bits) over the 7 sections.
                          Range: 0 (relies on 1 section) to log₂(7) ≈ 2.81 bits.
                          Target after penalised retraining: H < 1.5 bits.
  Effective features    — 2^H: how many sections the policy "effectively" consults.
                          Baseline ≈ 4–6; target ≤ 3.
  Gini coefficient      — 0 = uniform spread, 1 = all weight on one section.
  Top-2 dominance       — fraction of total attribution in the top-2 sections.
                          Target > 0.70 for human-learnable conventions.

Figures produced
----------------
  diag1_heatmap.png        — attribution heatmap (actions × sections, %)
  diag2_entropy.png        — entropy bars per action type and game phase
  diag3_effective_n.png    — effective feature count grid (scenario × action)
  diag4_concentration.png  — cumulative attribution curves (feature-rank plots)
  diag5_summary.png        — full complexity overview: reliance + entropy grid

Usage (run from pyhanabi/)
--------------------------
    python tools/complexity_diagnostics.py \\
        --weight ../final_r3d2_ckpts/R3D2-2p/a/epoch3000.pthw \\
        --output_dir ../complexity_results
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Import shared infrastructure from attribution.py in the same directory ────
_tools_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _tools_dir)

from attribution import (
    load_model,
    build_fresh_bert,
    build_fresh_heads,
    get_input_embeddings,
    detect_spans,
    compute_ig,
    compute_all_qvalues,
    aggregate_by_section,
    ensure_dir,
    SECTION_LABELS,
    PROBE_ACTIONS,
)

# ══════════════════════════════════════════════════════════════════════════════
# Game scenarios — same three phases as attribution.py plus a "crisis" phase
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS: "OrderedDict[str, OrderedDict]" = OrderedDict([
    ("early_game", OrderedDict([
        ("life_tokens", "Life tokens: 3."),
        ("info_tokens", "Information tokens: 8."),
        ("fireworks",   "Fireworks: R0 Y0 G0 W0 B0."),
        ("own_hand",    "Player 0 (me): ?? ?? ?? ?? ??."),
        ("opp_hand",    "Player 1: R1 Y2 G3 W4 B1."),
        ("discards",    "Discards: none."),
        ("last_action", "Last action: none."),
    ])),
    ("mid_game", OrderedDict([
        ("life_tokens", "Life tokens: 2."),
        ("info_tokens", "Information tokens: 3."),
        ("fireworks",   "Fireworks: R2 Y1 G2 W1 B1."),
        ("own_hand",    "Player 0 (me): ?? ?? ?? ?? ??."),
        ("opp_hand",    "Player 1: R3 Y2 G3 W2 B2."),
        ("discards",    "Discards: R1 Y1 G1 W1."),
        ("last_action", "Last action: player 1 hinted rank 2."),
    ])),
    ("late_game", OrderedDict([
        ("life_tokens", "Life tokens: 1."),
        ("info_tokens", "Information tokens: 1."),
        ("fireworks",   "Fireworks: R4 Y4 G4 W4 B3."),
        ("own_hand",    "Player 0 (me): ?? ?? ?? ?? ??."),
        ("opp_hand",    "Player 1: R5 Y5 G5 W5 B5."),
        ("discards",    "Discards: R1 Y1 G1 W1 B1 R2 Y2 G2."),
        ("last_action", "Last action: player 1 played B3."),
    ])),
    ("crisis", OrderedDict([
        ("life_tokens", "Life tokens: 1."),
        ("info_tokens", "Information tokens: 0."),
        ("fireworks",   "Fireworks: R4 Y4 G4 W4 B4."),
        ("own_hand",    "Player 0 (me): ?? ?? ?? ?? ??."),
        ("opp_hand",    "Player 1: R5 Y5 G5 W5 B5."),
        ("discards",    "Discards: R1 Y1 G1 W1 B1 R2 Y2."),
        ("last_action", "Last action: player 1 discarded R2."),
    ])),
])

# Maximum possible entropy for n_sections equally-weighted sections
MAX_ENTROPY_BITS = float(np.log2(len(SECTION_LABELS)))  # ≈ 2.807 for 7 sections

# Section colours (consistent with attribution.py cross-scenario figure)
SECTION_COLORS = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
    "#f4a261", "#264653", "#a8dadc",
]


# ══════════════════════════════════════════════════════════════════════════════
# Complexity metrics
# ══════════════════════════════════════════════════════════════════════════════

def _to_prob(section_scores: dict) -> np.ndarray:
    """Normalised probability distribution (absolute-value magnitude) over sections."""
    vals = np.array([abs(v) for v in section_scores.values()], dtype=float)
    total = vals.sum()
    if total < 1e-10:
        return np.ones(len(vals)) / len(vals)   # fallback: uniform
    return vals / total


def entropy_bits(section_scores: dict) -> float:
    """Shannon entropy H(p) in bits.  Higher = more spread = more complex."""
    p = _to_prob(section_scores)
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log2(p)))


def effective_features(section_scores: dict) -> float:
    """Hill number 2^H: how many sections the policy effectively consults."""
    return 2.0 ** entropy_bits(section_scores)


def gini_coeff(section_scores: dict) -> float:
    """Gini coefficient: 0 = uniform spread, 1 = all weight on one section."""
    p = np.sort(_to_prob(section_scores))       # ascending
    n = len(p)
    cumsum = np.cumsum(p)
    return float(1.0 - (2.0 / n) * np.sum(cumsum) + 1.0 / n)


def topk_dominance(section_scores: dict, k: int = 2) -> float:
    """Fraction of total attribution captured by the top-k sections."""
    p = _to_prob(section_scores)
    return float(np.sort(p)[::-1][:k].sum())


def all_metrics(section_scores: dict) -> dict:
    return {
        "entropy":  entropy_bits(section_scores),
        "eff_feat": effective_features(section_scores),
        "gini":     gini_coeff(section_scores),
        "top2_dom": topk_dominance(section_scores, k=2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Attribution runner
# ══════════════════════════════════════════════════════════════════════════════

def run_all_attribution(bert, tokenizer, lstm, fc_v, fc_a,
                        act_embeds, action_labels,
                        num_lstm_layer: int, hid_dim: int,
                        ig_steps: int = 50) -> dict:
    """
    Compute IG attribution for every scenario × probe action pair.

    Returns nested dict::

        results[scenario][probe_name] = {
            "section_scores": {section: float, ...},   # absolute IG per section
            "metrics":        {"entropy": ..., "eff_feat": ..., ...},
            "prob_dist":      np.ndarray,               # normalised distribution
        }

    Also stores results[scenario]["greedy_idx"] and results[scenario]["qvals"].
    """
    results = {}
    probe_base = dict(PROBE_ACTIONS)    # {"greedy": None, "D1": 0, ...}

    for scenario, sections in SCENARIOS.items():
        print(f"\n── {scenario} ──────────────────────────────────────────────")
        full_text = " ".join(sections.values())

        _, _tokens, embeds_base = get_input_embeddings(bert, tokenizer, full_text)
        spans = detect_spans(tokenizer, full_text, sections)

        missing = [n for n, s in spans.items() if s is None]
        if missing:
            print(f"  [warn] could not locate sections: {missing}")

        qvals = compute_all_qvalues(
            bert, lstm, fc_v, fc_a, act_embeds, embeds_base, num_lstm_layer, hid_dim
        )
        greedy_idx = int(np.argmax(qvals))
        greedy_label = (action_labels[greedy_idx]
                        if greedy_idx < len(action_labels) else str(greedy_idx))
        print(f"  greedy = {greedy_label} (idx={greedy_idx})  Q={qvals[greedy_idx]:.4f}")

        probe_map = dict(probe_base)
        probe_map["greedy"] = greedy_idx

        scenario_res = {"greedy_idx": greedy_idx, "qvals": qvals}

        for probe_name, probe_idx in probe_map.items():
            if probe_idx is None or probe_idx >= act_embeds.size(0):
                continue

            print(f"  IG [{probe_name:6s}] ...", end=" ", flush=True)
            ig_vals = compute_ig(
                bert, lstm, fc_v, fc_a, act_embeds, embeds_base, probe_idx,
                num_lstm_layer, hid_dim, n_steps=ig_steps,
            )
            section_scores = aggregate_by_section(ig_vals, spans)
            metrics = all_metrics(section_scores)
            print(f"H={metrics['entropy']:.2f}b  eff={metrics['eff_feat']:.1f}f"
                  f"  top2={metrics['top2_dom']*100:.0f}%")

            scenario_res[probe_name] = {
                "section_scores": section_scores,
                "metrics":        metrics,
                "prob_dist":      _to_prob(section_scores),
            }

        results[scenario] = scenario_res

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Attribution heatmap (actions × sections, one panel per scenario)
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(results: dict, probe_names: list, output_dir: str) -> None:
    """
    Heatmap: rows = probe actions, cols = game-state sections.
    Cell value = percentage of total IG magnitude going to that section.
    Uniform rows → spread / complex.  Concentrated rows → focused / simple.
    """
    section_names   = list(SECTION_LABELS.keys())
    section_display = list(SECTION_LABELS.values())
    n_scenarios = len(results)

    fig, axes = plt.subplots(1, n_scenarios,
                             figsize=(6 * n_scenarios, max(3, 0.8 * len(probe_names) + 1.5)))

    if n_scenarios == 1:
        axes = [axes]

    for ax, (scenario, sres) in zip(axes, results.items()):
        valid_probes = [p for p in probe_names if p in sres]
        mat = np.zeros((len(valid_probes), len(section_names)))
        for i, probe in enumerate(valid_probes):
            mat[i] = sres[probe]["prob_dist"] * 100.0     # percentage

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=60, aspect="auto")

        ax.set_xticks(range(len(section_display)))
        ax.set_xticklabels(section_display, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(valid_probes)))
        ax.set_yticklabels(valid_probes, fontsize=9)
        ax.set_title(scenario, fontsize=11, pad=8)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                color = "white" if v > 42 else "black"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7, color=color)

        plt.colorbar(im, ax=ax, label="Attribution %", fraction=0.04, pad=0.02)

    fig.suptitle(
        "Attribution distribution across game-state sections\n"
        "(% of IG magnitude per action — uniform rows = complex, concentrated rows = simple)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "diag1_heatmap.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag1] heatmap → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Entropy bars (one grouped bar chart across all scenarios)
# ══════════════════════════════════════════════════════════════════════════════

def plot_entropy(results: dict, probe_names: list, output_dir: str) -> None:
    """
    Grouped bar chart of Shannon entropy (bits) per probe action, grouped
    by game scenario.  Reference lines at key complexity thresholds.
    The right y-axis shows effective feature count (2^H).
    """
    n_scenarios = len(results)
    n_probes = len(probe_names)
    scenario_names = list(results.keys())

    cmap = plt.cm.Set2
    scenario_colors = [cmap(i / max(n_scenarios - 1, 1)) for i in range(n_scenarios)]

    fig, ax = plt.subplots(figsize=(max(9, 1.8 * n_probes), 5))
    x = np.arange(n_probes)
    width = 0.75 / n_scenarios

    for si, (scenario, sres) in enumerate(results.items()):
        entropies = [
            sres[probe]["metrics"]["entropy"] if probe in sres else 0.0
            for probe in probe_names
        ]
        offset = (si - n_scenarios / 2.0 + 0.5) * width
        ax.bar(x + offset, entropies, width,
               label=scenario, color=scenario_colors[si], alpha=0.85,
               edgecolor="white", linewidth=0.5)

    # Complexity reference lines
    refs = [
        (MAX_ENTROPY_BITS, "#cc0000", "all 7 sections (max complexity)"),
        (np.log2(5),        "#e05000", "5 effective features"),
        (np.log2(4),        "#e07700", "4 effective features"),
        (np.log2(3),        "#cc9900", "3 effective features"),
        (np.log2(2),        "#117733", "2 effective features  ← target"),
    ]
    for val, color, label in refs:
        ax.axhline(val, color=color, linewidth=1.2, linestyle="--", alpha=0.75,
                   label=f"H={val:.2f}b: {label}")

    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, fontsize=10)
    ax.set_ylabel("Entropy (bits)", fontsize=10)
    ax.set_ylim(0, MAX_ENTROPY_BITS + 0.35)
    ax.set_title(
        "Attribution entropy per action type and game phase\n"
        "(higher bars = policy relies on more features simultaneously = harder for humans to learn)",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    # Right axis: effective features (non-linear labels)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    tick_feats = [1, 2, 3, 4, 5, 6, 7]
    tick_pos   = [np.log2(f) for f in tick_feats]
    ax2.set_yticks(tick_pos)
    ax2.set_yticklabels([str(f) for f in tick_feats], fontsize=8, color="dimgray")
    ax2.set_ylabel("Effective features (2^H)", fontsize=9, color="dimgray")

    fig.tight_layout()
    path = os.path.join(output_dir, "diag2_entropy.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag2] entropy bars → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Effective feature count grid (scenario × action)
# ══════════════════════════════════════════════════════════════════════════════

def plot_effective_n(results: dict, probe_names: list, output_dir: str) -> None:
    """
    2-D grid heatmap: scenario (row) × probe action (col).
    Cell = effective feature count (2^H).
    Green = focused (simple), red = spread (complex).
    """
    scenario_names = list(results.keys())
    mat = np.full((len(scenario_names), len(probe_names)), np.nan)

    for si, (scenario, sres) in enumerate(results.items()):
        for pi, probe in enumerate(probe_names):
            if probe in sres:
                mat[si, pi] = sres[probe]["metrics"]["eff_feat"]

    cell_h = max(0.9, 3.5 / len(scenario_names))
    cell_w = max(0.9, 8.0 / len(probe_names))
    fig, ax = plt.subplots(figsize=(cell_w * len(probe_names) + 2,
                                    cell_h * len(scenario_names) + 1.5))

    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=1.0, vmax=len(SECTION_LABELS),
                   aspect="auto")

    ax.set_xticks(range(len(probe_names)))
    ax.set_xticklabels(probe_names, fontsize=11)
    ax.set_yticks(range(len(scenario_names)))
    ax.set_yticklabels(scenario_names, fontsize=11)

    for si in range(mat.shape[0]):
        for pi in range(mat.shape[1]):
            v = mat[si, pi]
            if not np.isnan(v):
                txt_color = "white" if (v > 5.0 or v < 1.5) else "black"
                ax.text(pi, si, f"{v:.1f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=txt_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"Effective features (1 = focused, {len(SECTION_LABELS)} = uniform)",
                   fontsize=9)
    ax.set_title(
        "Effective number of game-state features consulted per decision\n"
        "(green = simple / focused,  red = complex / spread)",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "diag3_effective_n.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag3] effective-feature grid → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Concentration curves (feature-rank plots)
# ══════════════════════════════════════════════════════════════════════════════

def plot_concentration(results: dict, probe_names: list, output_dir: str) -> None:
    """
    For each probe action and game scenario, sort sections by importance
    (descending) and plot cumulative fraction of total attribution.

    Steeper rise = more concentrated = simpler policy.
    An ideal simple policy reaches ≥ 0.80 on top-2 sections.
    """
    n_sections = len(SECTION_LABELS)
    x_ranks = np.arange(1, n_sections + 1)

    action_colors = {
        "greedy": "black",
        "D1":  "steelblue",
        "P1":  "coral",
        "CR":  "seagreen",
        "R1":  "mediumpurple",
    }
    line_styles = ["solid", "dashed", "dotted", "dashdot"]
    scenario_ls = {sc: line_styles[i % len(line_styles)]
                   for i, sc in enumerate(results.keys())}

    fig, axes = plt.subplots(1, len(results),
                             figsize=(5 * len(results), 4.5),
                             sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (scenario, sres) in zip(axes, results.items()):
        for probe in probe_names:
            if probe not in sres:
                continue
            p = sres[probe]["prob_dist"]
            cumul = np.cumsum(np.sort(p)[::-1])
            ax.plot(x_ranks, cumul,
                    marker="o", markersize=4, linewidth=2,
                    label=probe,
                    color=action_colors.get(probe, "gray"))

        # Complexity reference lines
        ax.axhline(0.80, color="#ffaa00", linewidth=1.2, linestyle="--", alpha=0.8,
                   label="80% in top-k")
        ax.axhline(0.60, color="#aaaaaa", linewidth=1.0, linestyle=":", alpha=0.6,
                   label="60% in top-k")

        # Target annotation: ideal simple policy (80% in top-2)
        ax.axvspan(0.5, 2.5, alpha=0.06, color="green", label="target: top-2 dominate")

        ax.set_xticks(x_ranks)
        ax.set_xticklabels([f"top-{i}" for i in x_ranks], rotation=30, fontsize=8)
        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel("Top-k sections included", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Cumulative attribution fraction", fontsize=9)
    fig.suptitle(
        "Attribution concentration curves\n"
        "(steeper = more focused on fewer features = simpler convention)\n"
        "Green band = target zone: 2 sections capture ≥ 80% of attribution",
        fontsize=10, y=1.03,
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "diag4_concentration.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag4] concentration curves → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Summary overview (feature reliance + entropy grid)
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(results: dict, probe_names: list, output_dir: str) -> None:
    """
    Two-panel summary figure.

    Top panel:   mean attribution per section, averaged across all probe
                 actions and game scenarios → "which features dominate overall?"
    Bottom panel: entropy heatmap (scenario × action), annotated with
                 H [bits] and effective feature count [f].
    """
    section_names   = list(SECTION_LABELS.keys())
    section_display = list(SECTION_LABELS.values())
    scenario_names  = list(results.keys())

    # ── Collect mean section importance ─────────────────────────────────────
    all_probs = []
    for sres in results.values():
        for probe in probe_names:
            if probe in sres:
                all_probs.append(sres[probe]["prob_dist"])
    mean_prob = (np.mean(all_probs, axis=0)
                 if all_probs else np.ones(len(section_names)) / len(section_names))

    # ── Collect entropy grid ─────────────────────────────────────────────────
    ent_mat  = np.full((len(scenario_names), len(probe_names)), np.nan)
    eff_mat  = np.full_like(ent_mat, np.nan)
    for si, (scenario, sres) in enumerate(results.items()):
        for pi, probe in enumerate(probe_names):
            if probe in sres:
                ent_mat[si, pi] = sres[probe]["metrics"]["entropy"]
                eff_mat[si, pi] = sres[probe]["metrics"]["eff_feat"]

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.50)

    # ─ Top: mean section reliance bar chart ─────────────────────────────────
    ax_top = fig.add_subplot(gs[0])
    bars = ax_top.bar(range(len(section_names)), mean_prob * 100.0,
                      color=SECTION_COLORS, alpha=0.87, edgecolor="white",
                      linewidth=0.5)
    ax_top.set_xticks(range(len(section_names)))
    ax_top.set_xticklabels(section_display, fontsize=9)
    ax_top.set_ylabel("Mean attribution (%)", fontsize=10)
    ax_top.set_ylim(0, mean_prob.max() * 100 * 1.3)
    ax_top.set_title(
        "Overall feature reliance — which game-state sections are attended to most\n"
        "(averaged over all action types and game phases)",
        fontsize=10,
    )
    ax_top.grid(axis="y", alpha=0.25)

    # Annotate uniform baseline
    uniform_pct = 100.0 / len(section_names)
    ax_top.axhline(uniform_pct, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax_top.text(len(section_names) - 0.5, uniform_pct + 0.3,
                f"uniform ({uniform_pct:.0f}%)", ha="right", fontsize=8, color="gray")

    for bar, val in zip(bars, mean_prob * 100):
        ax_top.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    # ─ Bottom: entropy heatmap ───────────────────────────────────────────────
    ax_bot = fig.add_subplot(gs[1])
    im = ax_bot.imshow(ent_mat, cmap="RdYlGn_r",
                       vmin=0, vmax=MAX_ENTROPY_BITS, aspect="auto")

    ax_bot.set_xticks(range(len(probe_names)))
    ax_bot.set_xticklabels(probe_names, fontsize=10)
    ax_bot.set_yticks(range(len(scenario_names)))
    ax_bot.set_yticklabels(scenario_names, fontsize=10)

    for si in range(len(scenario_names)):
        for pi in range(len(probe_names)):
            h = ent_mat[si, pi]
            if not np.isnan(h):
                eff = eff_mat[si, pi]
                label = f"{h:.2f}b\n({eff:.1f}f)"
                txt_color = "white" if h > 2.0 else "black"
                ax_bot.text(pi, si, label, ha="center", va="center",
                            fontsize=8.5, color=txt_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax_bot, fraction=0.03, pad=0.02)
    cbar.set_label(f"Entropy (bits, max = {MAX_ENTROPY_BITS:.2f})", fontsize=9)
    ax_bot.set_title(
        "Policy complexity per action type and game phase\n"
        "(cell: entropy [bits] / effective features [f] — green = simple, red = complex)",
        fontsize=10,
    )

    path = os.path.join(output_dir, "diag5_summary.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag5] summary overview → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Text summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_complexity_table(results: dict, probe_names: list) -> None:
    print("\n" + "═" * 78)
    print("COMPLEXITY METRICS  (entropy in bits, eff. features = 2^H)")
    print("═" * 78)

    col_w = 9
    header = f"  {'Scenario':12s}  {'Metric':18s}" + "".join(
        f"  {p:>{col_w}s}" for p in probe_names
    )
    print(header)
    print("─" * len(header))

    metrics_spec = [
        ("entropy",  "Entropy (bits)"),
        ("eff_feat", "Eff. features"),
        ("gini",     "Gini coeff."),
        ("top2_dom", "Top-2 dom. %"),
    ]
    for scenario, sres in results.items():
        for mk, mlabel in metrics_spec:
            row = f"  {scenario[:12]:12s}  {mlabel:18s}"
            for probe in probe_names:
                if probe in sres:
                    v = sres[probe]["metrics"][mk]
                    if mk == "top2_dom":
                        row += f"  {v * 100:>{col_w}.1f}%"
                    else:
                        row += f"  {v:>{col_w}.3f} "
                else:
                    row += "  " + " " * (col_w + 1)
            print(row)
        print()

    # Global average entropy
    all_H = [
        sres[probe]["metrics"]["entropy"]
        for sres in results.values()
        for probe in probe_names
        if probe in sres
    ]
    if all_H:
        mean_H  = np.mean(all_H)
        mean_ef = 2.0 ** mean_H
        print(f"  Mean entropy across all actions/phases: "
              f"{mean_H:.3f} bits  "
              f"(effective features: {mean_ef:.2f} / {len(SECTION_LABELS)})")
        pct_of_max = mean_H / MAX_ENTROPY_BITS * 100
        print(f"  Fraction of maximum possible entropy:   {pct_of_max:.1f}%")
        if pct_of_max > 75:
            print("  ⚑  Policy is HIGH-COMPLEXITY — retraining with entropy penalty recommended.")
        elif pct_of_max > 50:
            print("  ⚐  Policy is MODERATE-COMPLEXITY — optional entropy regularisation.")
        else:
            print("  ✓  Policy is LOW-COMPLEXITY — conventions are relatively focused.")


# ══════════════════════════════════════════════════════════════════════════════
# Retraining plan (printed to stdout)
# ══════════════════════════════════════════════════════════════════════════════

RETRAINING_PLAN = """
╔══════════════════════════════════════════════════════════════════════════════╗
║         PLAN: Retraining R3D2 with Attention-Entropy Penalty                ║
╚══════════════════════════════════════════════════════════════════════════════╝

PROBLEM
───────
The agent currently attends to 4–7 game-state features simultaneously for
most decisions (high attribution entropy ≈ 2.4–2.8 bits, effective feature
count ≈ 5–7 out of 7).  This makes conventions unlearnable for human partners:
there is no short rule-of-thumb a person can apply, because the Q-value is
sensitive to everything at once.

GOAL
────
Reduce mean attribution entropy to ≤ 1.5 bits (≤ 3 effective features).
Each action type should concentrate ≥ 70% of its attribution on ≤ 2 sections
so that humans can reason: "the agent discards when hints are low AND discards
are few — I don't need to track anything else."

────────────────────────────────────────────────────────────────────────────────
STEP 1 — Add a non-scripted BERT copy for attention diagnostics during training
────────────────────────────────────────────────────────────────────────────────
File: pyhanabi/q_net.py  (class TextLSTMNet.__init__)

The production call_transformer is torch.jit.trace-d and cannot return
attentions.  Add a plain BertModel shadow kept in sync after every gradient step.

    # In __init__():
    from transformers import BertConfig, BertModel
    cfg_att         = BertConfig.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    cfg_att.num_hidden_layers = self.num_lm_layer
    cfg_att.output_attentions = True
    self.penalty_bert = BertModel(cfg_att)

    # Add sync method (call once per optimiser step in training loop):
    def sync_penalty_bert(self):
        src = {k: v for k, v in self.call_transformer.state_dict().items()}
        self.penalty_bert.load_state_dict(src, strict=False)

────────────────────────────────────────────────────────────────────────────────
STEP 2 — Implement section-level attention entropy loss
────────────────────────────────────────────────────────────────────────────────
File: pyhanabi/q_net.py  (new method on TextLSTMNet)

    def attention_entropy_loss(self, input_ids: torch.Tensor,
                               section_spans: dict) -> torch.Tensor:
        \"\"\"
        Compute mean Shannon entropy of CLS-token attention over game sections.
        Lower entropy = more focused attention = simpler conventions.
        \"\"\"
        out = self.penalty_bert(input_ids)        # uses output_attentions=True
        # out.attentions: list of [batch, num_heads, seq, seq] per layer
        attn  = out.attentions[-1].mean(dim=1)    # mean over heads: [batch, seq, seq]
        cls_a = attn[:, 0, :]                     # CLS row: [batch, seq]

        # Aggregate attention mass per section
        weights = []
        for span in section_spans.values():
            if span is not None:
                s, e = span
                weights.append(cls_a[:, s:e].sum(dim=1, keepdim=True))
        if not weights:
            return input_ids.new_zeros(1, dtype=torch.float)

        S = torch.cat(weights, dim=1)                    # [batch, n_sections]
        p = S / S.sum(dim=1, keepdim=True).clamp(min=1e-8)
        entropy = -(p * (p + 1e-10).log()).sum(dim=1)    # [batch], nats
        return entropy.mean()

────────────────────────────────────────────────────────────────────────────────
STEP 3 — Integrate into the training loss
────────────────────────────────────────────────────────────────────────────────
File: pyhanabi/r2d2.py  (R2D2Agent.loss)

Change signature to accept the new parameters:

    def loss(self, batch, aux_weight, stat,
             entropy_lambda: float = 0.0,
             section_spans: dict = None):

        err, lstm_o, _ = self.td_error(...)
        rl_loss = nn.functional.smooth_l1_loss(err, torch.zeros_like(err),
                                               reduction="none").sum(0)

        loss = rl_loss
        if aux_weight > 0:
            pred1 = self.aux_task(lstm_o, ...)
            loss  = loss + aux_weight * pred1

        if entropy_lambda > 0 and section_spans is not None:
            # Use the first time-step of state observations (shape: [batch, seq])
            obs_ids = batch.obs["priv_s"][0].long()
            ent_loss = self.online_net.attention_entropy_loss(obs_ids, section_spans)
            loss = loss + entropy_lambda * ent_loss
            if stat is not None:
                stat["attn_entropy"].feed(ent_loss.item())

        return loss

Sync the penalty BERT every optimiser step (add to selfplay.py training loop):

    agent.online_net.sync_penalty_bert()

────────────────────────────────────────────────────────────────────────────────
STEP 4 — Curriculum schedule for entropy_lambda
────────────────────────────────────────────────────────────────────────────────
A sudden large penalty will destroy the base policy.
Use a linear ramp — start at zero, increase slowly:

    epoch   0 – 500 :  entropy_lambda = 0.00   (learn base policy undisturbed)
    epoch 500 – 1000:  entropy_lambda = ramp 0.00 → 0.10
    epoch 1000–2000 :  entropy_lambda = 0.10   (light regularisation, monitor score)
    epoch 2000+     :  entropy_lambda = 0.20   (increase only if score holds)

Recommended sweep:  lambda in {0.05, 0.10, 0.20, 0.50}
Never increase lambda if cross-play score drops > 1.0 point vs. baseline.

────────────────────────────────────────────────────────────────────────────────
STEP 5 — Monitoring and evaluation protocol
────────────────────────────────────────────────────────────────────────────────
Run complexity_diagnostics.py on every checkpoint every 250 epochs:

    python tools/complexity_diagnostics.py \\
        --weight ckpts/epoch<N>.pthw \\
        --output_dir complexity_monitor/epoch<N>/

Track three key numbers over training:
  1. Mean attribution entropy  →  should decrease monotonically from ~2.5b to <1.5b
  2. Self-play score           →  should stay within 0.5 pts of unpenalised baseline
  3. Top-2 dominance           →  should increase from ~35% toward >70%

Stopping criteria:
  • Entropy < 1.5 bits AND top-2 dominance > 70%  →  retraining successful
  • Score drops > 1.5 pts below baseline           →  reduce lambda and retrain
  • Entropy plateaus above 2.0 bits after 1000 epochs  →  increase lambda

────────────────────────────────────────────────────────────────────────────────
STEP 6 — Human-interpretability validation
────────────────────────────────────────────────────────────────────────────────
After successful penalised training:
  • Play 50 human-AI games; ask players to write down the 2–3 rules they think
    the agent follows.  A lower-entropy policy should produce consistent, short
    descriptions across players.
  • Run zero-shot cross-play with other agents: simpler attention profiles
    correlate with better generalisation (the agent's decisions depend on fewer
    features that might confuse an unfamiliar partner).

EXPECTED OUTCOMES
─────────────────
  lambda = 0.00 (baseline) :  H ≈ 2.4–2.8 bits,  eff. features ≈ 5–7
  lambda = 0.10 (light)    :  H ≈ 1.8–2.2 bits,  eff. features ≈ 3.5–4.5
  lambda = 0.20 (moderate) :  H ≈ 1.3–1.7 bits,  eff. features ≈ 2.5–3.2
  lambda = 0.50 (strong)   :  H < 1.2 bits,       eff. features < 2.5
                               (risk of score drop — monitor carefully)

QUICK COMMAND REFERENCE
────────────────────────
  # Diagnose a single checkpoint
  python tools/complexity_diagnostics.py \\
      --weight <ckpt>.pthw \\
      --output_dir complexity_results/

  # Quick run (fewer IG steps)
  python tools/complexity_diagnostics.py \\
      --weight <ckpt>.pthw \\
      --output_dir complexity_results/ \\
      --ig_steps 20

  # Compare two checkpoints side-by-side
  for CKPT in baseline.pthw penalised.pthw; do
      python tools/complexity_diagnostics.py \\
          --weight $CKPT \\
          --output_dir complexity_monitor/$(basename $CKPT .pthw)/
  done
"""


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing + main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--weight", required=True,
                   help="Path to .pthw checkpoint (e.g. "
                        "../final_r3d2_ckpts/R3D2-2p/a/epoch3000.pthw)")
    p.add_argument("--output_dir", default="../complexity_results",
                   help="Directory for output figures (default: ../complexity_results)")
    p.add_argument("--device", default="cpu",
                   help="Torch device (default: cpu)")
    p.add_argument("--ig_steps", type=int, default=50,
                   help="Number of IG interpolation steps (reduce to 20 for speed)")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.weight}")
    agent, cfg = load_model(args.weight, args.device)

    hid_dim        = cfg["rnn_hid_dim"]
    num_lstm_layer = cfg["num_lstm_layer"]
    num_lm_layer   = cfg.get("num_lm_layer", 1)

    # ── Load raw state dict ────────────────────────────────────────────────────
    raw_sd  = torch.load(args.weight, map_location="cpu")
    full_sd = OrderedDict(
        (k.replace("_orig_mod.", ""), v) for k, v in raw_sd.items()
    )

    print("Building fresh BertModel and head modules for gradient computation ...")
    bert, tokenizer = build_fresh_bert(full_sd, num_lm_layer=num_lm_layer)
    lstm, fc_v, fc_a = build_fresh_heads(full_sd, hid_dim, num_lstm_layer)

    # ── Action embeddings ──────────────────────────────────────────────────────
    act_tok_path = "action_tokens/2p_action_ids.json"
    if not os.path.exists(act_tok_path):
        raise FileNotFoundError(
            f"{act_tok_path} not found.  Run this script from pyhanabi/."
        )
    with open(act_tok_path) as f:
        act_data = json.load(f)

    act_toks = torch.tensor(act_data["input_ids"])
    with torch.no_grad():
        act_out = bert(input_ids=act_toks)
    act_embeds = act_out.last_hidden_state.mean(dim=1).float()   # [num_actions, 128]

    num_actions = act_embeds.size(0)
    action_labels = (
        [f"D{i+1}" for i in range(5)]
        + [f"P{i+1}" for i in range(5)]
        + [f"C{c}" for c in "RYGWB"]
        + [f"R{r}" for r in "12345"]
        + ["noop"]
    )[:num_actions]

    # ── Run attribution ────────────────────────────────────────────────────────
    print(f"\nRunning IG attribution  ({args.ig_steps} steps)  ...")
    results = run_all_attribution(
        bert, tokenizer, lstm, fc_v, fc_a,
        act_embeds, action_labels,
        num_lstm_layer, hid_dim,
        ig_steps=args.ig_steps,
    )

    # Collect probe names that were actually computed (greedy is scenario-specific,
    # but we always include it in the display)
    probe_names = [p for p in PROBE_ACTIONS if any(p in sres for sres in results.values())]

    # ── Generate figures ───────────────────────────────────────────────────────
    print("\n── Generating diagnostic figures ────────────────────────────────")
    plot_heatmap(results, probe_names, args.output_dir)
    plot_entropy(results, probe_names, args.output_dir)
    plot_effective_n(results, probe_names, args.output_dir)
    plot_concentration(results, probe_names, args.output_dir)
    plot_summary(results, probe_names, args.output_dir)

    # ── Print metrics table ────────────────────────────────────────────────────
    print_complexity_table(results, probe_names)

    # ── Print retraining plan ──────────────────────────────────────────────────
    print(RETRAINING_PLAN)

    print(f"Done.  All figures written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
