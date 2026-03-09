# R3D2 Hanabi Agent — Interpretability Analysis

> Checkpoint analysed: `final_r3d2_ckpts/R3D2-2p/a/epoch3000.pthw`
> Tools: `pyhanabi/tools/interpretability.py`, `attribution.py`, `complexity_diagnostics.py`

---

## Table of Contents

1. [Architecture Primer](#1-architecture-primer)
2. [Phase 1 — Weight & Structural Analysis](#2-phase-1--weight--structural-analysis)
   - 2.1 [Parameter Norms](#21-parameter-norms)
   - 2.2 [LSTM Gate Bias Analysis](#22-lstm-gate-bias-analysis)
   - 2.3 [Value Head vs Advantage Head Geometry](#23-value-head-vs-advantage-head-geometry)
   - 2.4 [BERT Attention Patterns](#24-bert-attention-patterns)
   - 2.5 [Action Embedding Similarity](#25-action-embedding-similarity)
3. [Phase 2 — Attribution Analysis](#3-phase-2--attribution-analysis)
   - 3.1 [Method: Saliency Maps](#31-method-saliency-maps)
   - 3.2 [Method: Integrated Gradients](#32-method-integrated-gradients)
   - 3.3 [Section Detection and Span Mapping](#33-section-detection-and-span-mapping)
   - 3.4 [Results by Game Phase](#34-results-by-game-phase)
   - 3.5 [Signed Attribution: What Pushes Q Up vs Down](#35-signed-attribution-what-pushes-q-up-vs-down)
4. [Phase 3 — Complexity Diagnostics](#4-phase-3--complexity-diagnostics)
   - 4.1 [Complexity Metrics Defined](#41-complexity-metrics-defined)
   - 4.2 [Results: Full Metric Table](#42-results-full-metric-table)
   - 4.3 [Heatmap: Which Features Drive Which Actions](#43-heatmap-which-features-drive-which-actions)
   - 4.4 [Entropy and Effective Feature Count](#44-entropy-and-effective-feature-count)
   - 4.5 [Concentration Curves](#45-concentration-curves)
5. [Cross-Cutting Findings](#5-cross-cutting-findings)
6. [What This Means for Human-AI Collaboration](#6-what-this-means-for-human-ai-collaboration)
7. [Next Steps: Retraining with Entropy Penalty](#7-next-steps-retraining-with-entropy-penalty)
8. [Figure Index](#8-figure-index)

---

## 1. Architecture Primer

R3D2 uses a **TextLSTMNet** (`drrn-lstm`) architecture that processes the game state as a natural-language string rather than a fixed feature vector. The forward pass is:

```
Game state string
        │
        ▼
  TinyBERT encoder          (1 transformer layer, hidden dim = 128)
  cross-encoder/ms-marco-TinyBERT-L-2-v2
        │
        ▼  mean-pool over token dimension
  State embedding           [128-dim]
        │
        ▼
  2-layer LSTM              (hid_dim = 128 × 2 layers)
        │  hidden state h_t
        ▼
  ┌─────────────┐    ┌──────────────────────────────┐
  │  fc_v(h_t)  │    │  fc_a(h_t ⊙ BERT(action_a))  │
  │  [128 → 1]  │    │  [128 → 1]                   │
  └──────┬──────┘    └───────────┬──────────────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
             Q(s, a) = V(s) + A(s, a)
```

There are **21 actions** in the 2-player setting:
- D1–D5: discard cards 0–4 (indices 0–4)
- P1–P5: play cards 0–4 (indices 5–9)
- CR, CY, CG, CW, CB: hint colour to partner (indices 10–14)
- R1–R5: hint rank to partner (indices 15–19)
- noop: index 20

The key property that makes this architecture hard to interpret with standard tools is that `TextLSTMNet` is compiled as a **`torch.jit.ScriptModule`**. Its sub-components (`state_lstm`, `fc_v`, `fc_a`) become `RecursiveScriptModule` objects that cannot be differentiated through using PyTorch's standard autograd. All gradient-based analyses in Phases 2 and 3 required extracting fresh `nn.Module` copies of these weights from the raw state dict.

---

## 2. Phase 1 — Weight & Structural Analysis

**Script:** `pyhanabi/tools/interpretability.py`
**Output directory:** `interpret_results/`

### 2.1 Parameter Norms

**What is measured:** The Frobenius norm (‖W‖_F = √Σᵢⱼ Wᵢⱼ²) of every named parameter tensor. This is the most basic sanity check — it shows whether the network is in a healthy training regime (moderate norms) or pathological (collapsed near zero, or exploded to very large values).

**File:** `interpret_results/weight_norms.txt`

| Parameter | Shape | Norm |
|-----------|-------|------|
| state_lstm.weight_ih_l0 | [512, 128] | 41.52 |
| state_lstm.weight_hh_l0 | [512, 128] | 38.54 |
| state_lstm.bias_ih_l0 | [512] | 2.31 |
| state_lstm.bias_hh_l0 | [512] | 2.21 |
| state_lstm.weight_ih_l1 | [512, 128] | 38.31 |
| state_lstm.weight_hh_l1 | [512, 128] | 32.76 |
| fc_v.weight | [1, 128] | 2.15 |
| fc_v.bias | [1] | 0.13 |
| fc_a.weight | [1, 128] | 2.92 |
| fc_a.bias | [1] | 0.02 |
| word_embeddings.weight | [30522, 128] | 110.00 |
| position_embeddings.weight | [512, 128] | 15.12 |
| encoder.layer.0.intermediate.dense.weight | [512, 128] | 28.62 |
| encoder.layer.0.output.dense.weight | [128, 512] | 29.07 |
| encoder.layer.0.output.LayerNorm.weight | [128] | 20.65 |

**Interpretation:**

- **LSTM weights (‖·‖ ≈ 33–42):** The LSTM input and hidden weights are substantially larger than the output head weights. Large LSTM norms are normal after thousands of training epochs — the LSTM has developed strong, opinionated transformations of the BERT embeddings. Layer 0 is slightly larger than Layer 1 in all metrics (41.5 vs 38.3 for weight_ih), suggesting the first LSTM layer does more heavy lifting in encoding new state information, while the second layer performs a more refined temporal integration.

- **Word embeddings (‖·‖ = 110.00):** The vocabulary embedding matrix has the highest norm in the entire model by a wide margin — 2.6× larger than the next largest weight. This is a strong signal that the word-level representations carry substantial semantic load. Because the embedding matrix has shape [30522, 128], most individual token embedding vectors are small, but collectively they span a very rich subspace. The large norm relative to other layers also suggests the fine-tuning from the pre-trained TinyBERT weights was preserved and built upon substantially.

- **Output heads (fc_v: 2.15, fc_a: 2.92):** Both output heads have norms roughly 10–15× smaller than the LSTM weights. This is expected: the value and advantage heads are linear projections from a well-regularised hidden state, so they do not need large weights to produce useful Q-value estimates. fc_a being slightly larger than fc_v suggests that advantage estimation requires a more directional projection through the hidden space.

- **BERT intermediate/output layers (‖·‖ ≈ 28–29):** The FFN (feed-forward network) inside the transformer has norms similar to the LSTM recurrent matrices, suggesting they operate at comparable signal scales.

---

### 2.2 LSTM Gate Bias Analysis

**What is measured:** An LSTM computes four gated transformations at each timestep:
- **Forget gate** f_t = σ(W_f x_t + U_f h_{t-1} + **b_f**)
- **Input gate** i_t = σ(W_i x_t + U_i h_{t-1} + **b_i**)
- **Gate gate** g_t = tanh(W_g x_t + U_g h_{t-1} + **b_g**)
- **Output gate** o_t = σ(W_o x_t + U_o h_{t-1} + **b_o**)

In PyTorch, the bias vector `bias_ih` has shape [4 × hid_dim] = [512], with the four gate biases concatenated in the order (i, f, g, o). We extract the forget gate slice [128:256] and analyse its mean and distribution.

**Figure:** `interpret_results/lstm_gate_analysis.png`

**Results:**
- **Forget gate bias (b_f):** mean ≈ **−0.52** for Layer 0, **−0.48** for Layer 1
- **Input gate bias (b_i):** mean ≈ **+0.15**
- **Output gate bias (b_o):** mean ≈ **+0.08**

**Interpretation:**

The negative forget gate bias is the most informative structural finding. A forget gate with bias −0.52 means that in the absence of a strong input signal, the default tendency is `σ(−0.52) ≈ 0.37` — the gate is **more closed than open**. The agent therefore selectively keeps information in memory rather than passively accumulating it. This is consistent with a game of Hanabi where most turns do not change the information relevant to a given card's status, so carrying over all state information would introduce noise rather than signal.

In contrast, the positive input gate bias (+0.15 → σ(0.15) ≈ 0.54) means new input is readily accepted when it arrives. This combination — "be selective about what you keep, but accept new information readily" — is the hallmark of an LSTM that has learned to prioritise informative events (a clue being given, a card being played) over the background noise of uninformative turns.

The near-zero output gate bias means the output projection is neutral by default, letting the forget-vs-input gating determine what gets surfaced to the next layer.

---

### 2.3 Value Head vs Advantage Head Geometry

**What is measured:** The cosine similarity between the weight vectors of `fc_v` and `fc_a`:

```
cos(fc_v.weight, fc_a.weight) = (fc_v.weight · fc_a.weight) / (‖fc_v.weight‖ · ‖fc_a.weight‖)
```

Both vectors have shape [128], so this is a dot product in the LSTM hidden state space. A cosine similarity of +1 means both heads read the same directions in hidden space; −1 means they are anti-correlated; 0 means they read completely orthogonal directions.

**Figure:** `interpret_results/fc_v_vs_fc_a.png`

**Result:** cos(fc_v, fc_a) = **−0.0096**

**Interpretation:**

The value and advantage heads are nearly perfectly **orthogonal** in the 128-dimensional hidden state space (cosine ≈ 0). This is a theoretically desirable property of dueling DQN architectures: ideally, `V(s)` and `A(s, a)` should capture genuinely different aspects of the state representation. A near-zero cosine similarity means the two heads are decoding information from different directions in the hidden state — they are not merely rescaled copies of each other.

Concretely: the LSTM hidden state encodes at least two separable kinds of information — one direction that signals "how good is this situation overall" (value) and another that signals "how much better is action a than average" (advantage). This suggests the LSTM has developed a structured latent representation, not a generic blob, which is encouraging for interpretability — the hidden state has decomposable semantic meaning.

---

### 2.4 BERT Attention Patterns

**What is measured:** The single-layer TinyBERT attention matrix. For each of the (default 2) attention heads, we visualise the attention weight matrix `A[h]` of shape [seq_len, seq_len], where `A[h][i, j]` is the attention weight token `i` puts on token `j`.

**Figures:** `interpret_results/bert_attention_maps.png`, `interpret_results/bert_attention_per_head.png`

**Results:**

The analysis revealed two functionally distinct attention heads:

- **Head 0 — Attention Sink Head:** This head directs nearly all attention toward the `[CLS]` and `[SEP]` special tokens regardless of position. The off-diagonal structure is sparse and shows no clear semantic pattern. This is a well-documented phenomenon in BERT-type models — some heads specialise in routing signal to special tokens, effectively serving as a "no-op" or bias-correction mechanism for downstream layers.

- **Head 1 — Semantic Head:** This head shows rich, content-sensitive attention patterns. Tokens corresponding to specific game state values (card identities, counts) attend to related tokens elsewhere in the sequence. There is noticeable attention between the fireworks progress section and the opponent's hand section, and between hint token counts and card rank/colour tokens — which makes functional sense: the agent is connecting "how many hints remain" with "which hints could be given."

**Interpretation:**

The presence of one semantic head and one sink head in a single-layer TinyBERT is consistent with findings in larger BERT models where roughly half the heads in early layers are functional sinks. With only one transformer layer, the agent is processing game state context with a single round of cross-token attention followed by an FFN — this is a fairly shallow semantic computation. The fact that meaningful patterns emerge despite this shallowness suggests that (a) the pre-trained TinyBERT weights provide a strong initialization that the fine-tuning preserved, and (b) the LSTM does substantial additional integration across timesteps to compensate for shallow per-state encoding.

---

### 2.5 Action Embedding Similarity

**What is measured:** For each pair of actions (a, a'), we compute:
1. The **cosine similarity** of their BERT-encoded embeddings: `cos(BERT(action_a), BERT(action_a'))`
2. The **first two PCA components** of the full 21 × 128 action embedding matrix to visualise cluster structure.

**Figures:** `interpret_results/action_embedding_similarity.png`, `interpret_results/action_embedding_pca.png`

**Results:**

- **Within-cluster similarities:**
  - Discard actions (D1–D5): cosine up to **0.9998** — essentially identical embeddings
  - Play actions (P1–P5): cosine up to **0.9997**
  - Colour hint actions (CR–CB): cosine up to **0.9990**
  - Rank hint actions (R1–R5): cosine up to **0.9985**

- **Between-cluster similarities:**
  - Discard vs. Play: cosine ≈ **0.72–0.78**
  - Discard vs. Hint: cosine ≈ **0.41–0.60**
  - Play vs. Hint: cosine ≈ **0.38–0.57**

- **PCA:** The 21 action embeddings form **three clearly separated clusters** in the first two principal components — one cluster for (Discard + Play), one for (Colour hints), and one for (Rank hints + noop). The discard and play clusters overlap considerably in PCA space despite their high within-group similarity.

**Interpretation:**

The near-collapse of within-action-type embeddings (cosine > 0.999) is striking and consequential. It means the BERT encoder treats "discard card 0" and "discard card 4" as semantically indistinguishable — the **which card** information is essentially invisible in the embedding. Similarly, "hint rank 2" and "hint rank 5" produce embeddings with cosine similarity 0.998. This is not entirely surprising because TinyBERT was pre-trained on a generic text corpus where "card 1" and "card 4" have no a priori semantic distinction, and the fine-tuning on Hanabi has apparently not differentiated them.

The practical implication is that the advantage head `fc_a(h_t ⊙ BERT(action_a))` produces almost the same advantage value for all actions within a type — the LSTM hidden state `h_t`, not the action embedding, must carry the information that distinguishes which specific card to play or discard. This shifts interpretive focus: the BERT-encoded action acts more like a "type label" (play vs. discard vs. hint) than a specific action descriptor. The actual card selection policy lives entirely inside the LSTM.

This also implies that the agent's apparent preference for specific cards (e.g., "play card 0 rather than card 4") emerges from subtle gradient interactions in the product `h_t ⊙ BERT(a)` rather than from BERT differentiating the actions. The advantage function is effectively `fc_a(h_t ⊙ type_vec)` where `type_vec` is nearly the same for all actions of the same type.

---

## 3. Phase 2 — Attribution Analysis

**Script:** `pyhanabi/tools/attribution.py`
**Output directory:** `attribution_results/`

### 3.1 Method: Saliency Maps

**Definition:** Given a game state text tokenised into a sequence of embedding vectors e = [e₁, e₂, ..., eₙ], the saliency of token t with respect to Q-value for action a is:

```
Saliency_t = ‖∂Q(s, a) / ∂eₜ‖₂
```

This is the L2 norm of the gradient of the Q-value with respect to the input embedding at position t. It answers: "if we were to make a small perturbation to the embedding at token t, how much would Q change?" Tokens with high saliency are those the model is most sensitive to.

**Properties:**
- Always non-negative (L2 norm)
- Does not indicate direction of influence (does not distinguish between "this token raises Q" and "this token lowers Q")
- Fast to compute: one backward pass per (state, action) pair
- Captures local sensitivity at the input, not global attribution

**Figure:** `attribution_results/fig1_token_heatmaps.png` (top panel of each scenario row)

### 3.2 Method: Integrated Gradients

**Definition:** Integrated Gradients (Sundararajan et al., 2017) computes a signed attribution for each token by integrating gradients along a straight path from a baseline embedding e⁰ to the actual embedding e:

```
IG_t = (eₜ - eₜ⁰) · (1/N) Σₖ₌₀ᴺ⁻¹ [∂Q/∂eₜ]|_{eₜ⁰ + (k/N)(eₜ - eₜ⁰)}
```

We use a zero baseline (e⁰ = 0) and N = 50 interpolation steps. Attribution per token is the **sum over the embedding dimension** (signed scalar). The key properties are:

- **Signed:** positive attribution means this token's presence increases Q; negative means it decreases Q.
- **Complete:** the sum of IG attributions over all tokens exactly equals Q(actual input) − Q(baseline), by the fundamental theorem of calculus.
- **Sensitive to absence, not just perturbation:** by integrating from zero, IG measures "how much does token t contribute to Q relative to a neutral encoding."

For section-level analysis, tokens belonging to each game-state section are identified by span matching (see 3.3), and the **absolute values** of their IG scores are summed:

```
Section_score(s) = Σ_{t ∈ span(s)} |IG_t|
```

Absolute values are used because we want total information influence, not net direction.

**Figure:** `attribution_results/fig1_token_heatmaps.png` (bottom panel), `fig2_component_bars.png`, `fig3_cross_scenario.png`, `fig4_signed_ig.png`

### 3.3 Section Detection and Span Mapping

The game state string is divided into seven labelled sections:

| Key | Content example |
|-----|----------------|
| `life_tokens` | "Life tokens: 2." |
| `info_tokens` | "Information tokens: 3." |
| `fireworks` | "Fireworks: R2 Y1 G2 W1 B1." |
| `own_hand` | "Player 0 (me): ?? ?? ?? ?? ??." |
| `opp_hand` | "Player 1: R3 Y2 G3 W2 B2." |
| `discards` | "Discards: R1 Y1 G1 W1." |
| `last_action` | "Last action: player 1 hinted rank 2." |

Each section is tokenised independently, and its token-ID subsequence is located within the full tokenisation by exact subsequence matching. This assigns a `(start, end)` token span to each section, allowing attribution scores to be aggregated per section rather than per token.

### 3.4 Results by Game Phase

Three game scenarios were analysed. Scores below are absolute IG sums for the greedy action (the action with the highest Q-value in that state). The greedy actions were:
- **Early game:** P1 (play card 0) — Q = highest
- **Mid game:** P1 (play card 0) — Q = highest
- **Late game:** noop — Q = highest (game nearly over, no safe plays)

**Attribution summary (absolute IG, greedy action, not normalised):**

| Section | Early game | Mid game | Late game |
|---------|-----------|---------|----------|
| Life tokens | **2.63** | 2.33 | 0.50 |
| Hint tokens | 0.38 | 0.77 | 0.25 |
| Fireworks | 1.52 | 0.99 | 0.34 |
| Own hand | 1.34 | 0.61 | 0.39 |
| Opp. hand | 0.92 | **2.47** | 0.68 |
| Discards | 1.48 | **3.78** | 0.65 |
| Last action | 0.45 | 0.28 | 0.16 |

**Figure:** `attribution_results/fig2_component_bars.png` (normalised per-action bars), `attribution_results/fig3_cross_scenario.png` (stacked area and line chart across phases)

**Early game analysis:**
Life tokens dominate (2.63), followed by fireworks (1.52), discards (1.48), and own hand (1.34). The agent's greedy choice to play card 0 is primarily influenced by how many lives remain (the risk budget) and the current fireworks state (which cards can be legally played). With 8 hint tokens available and no discards yet, the agent is in a low-information state — it cannot know whether its hand cards are safe to play — but still chooses to play, possibly because early-game risk-taking is optimal for score maximisation. Life tokens being the top feature makes sense: with 3 lives, the agent can afford experimental plays.

**Mid game analysis:**
The pattern shifts dramatically. Discards (3.78) and opponent's hand (2.47) now dominate. With 2 lives remaining and only 3 hint tokens, the agent is in a more constrained regime. The high discard attribution means the agent is closely tracking which cards have already been removed from the deck — this is classic Hanabi strategy, where knowing that all copies of a card are gone (or not) determines whether that card is safe to play. The opponent's hand attribution rise (0.92 → 2.47) reflects that mid-game, the agent can see what its partner holds and factors this into its own Q-values. This is the most information-rich game phase and the attribution scores reflect that — the agent is synthesising multiple information sources simultaneously.

**Late game analysis:**
All attributions collapse. Life tokens drop from 2.33 to 0.50, discards from 3.78 to 0.65. The greedy action is `noop` (do nothing), with Q = 6.60. When the game is nearly won (fireworks at R4 Y4 G4 W4 B3) or the agent cannot safely act, the Q-value landscape flattens — all actions have similar expected returns — so no single feature stands out as decisive. This produces a low-attribution, high-entropy profile (see Phase 3). The small residual attribution on opponent's hand (0.68) and discards (0.65) suggests these are still consulted but not determinative.

**Figure:** `attribution_results/fig3_cross_scenario.png`

The cross-scenario line chart reveals two structural patterns:
1. **Discards follow a V-shape:** low → very high → low. This is consistent with discard information being most useful in the mid-game "crunch" where many cards have been played and knowing which ranks/colours are exhausted is critical.
2. **Life tokens follow a monotone decline:** the agent cares progressively less about remaining lives as the game converges, presumably because with fewer actions left, the risk of losing a life decreases.

---

### 3.5 Signed Attribution: What Pushes Q Up vs Down

**Figure:** `attribution_results/fig4_signed_ig.png`

The signed IG sum per section (without absolute value) reveals directional influence:

| Section | Early game sign | Mid game sign | Late game sign |
|---------|----------------|--------------|----------------|
| Life tokens | **positive** (↑Q) | **positive** | **positive** |
| Hint tokens | mixed | mixed | mixed |
| Fireworks | **positive** | **positive** | positive |
| Own hand | negative (↓Q) | mixed | mixed |
| Opp. hand | **negative** | **positive** | mixed |
| Discards | **positive** | **positive** | mixed |
| Last action | mixed | mixed | mixed |

Key directional findings:
- **Life tokens consistently increase Q:** More lives → higher Q, as expected — the agent values the safety buffer.
- **Own hand decreases Q in early game:** The agent's uncertainty about its own cards (`?? ?? ?? ?? ??`) has a suppressive effect on Q in early game. This is interpretable: unknown cards are a liability, not an asset.
- **Opponent's hand reverses sign mid-game:** In early game, seeing the opponent hold cards that haven't yet been hinted pushes Q down (the agent knows its partner holds unknown-to-partner cards and cannot act on them). In mid-game, the opponent holds playable cards (R3, Y2, G3, W2, B2 against fireworks R2 Y1 G2 W1 B1), which pushes Q up — the partner is a resource.

---

## 4. Phase 3 — Complexity Diagnostics

**Script:** `pyhanabi/tools/complexity_diagnostics.py`
**Output directory:** `complexity_results/`

This phase takes the section-level IG attributions from Phase 2 and computes four scalar complexity metrics designed to answer: "how spread out is the agent's attention across features, and is this complexity problematic?"

### 4.1 Complexity Metrics Defined

**Shannon Entropy (H)**

Given the absolute IG attribution as a probability distribution over sections p = [p₁, ..., p₇] where pᵢ = |IGᵢ| / Σⱼ|IGⱼ|:

```
H(p) = -Σᵢ pᵢ log₂(pᵢ)    [bits]
```

Range: 0 bits (agent consults exactly one section) to log₂(7) ≈ 2.807 bits (agent consults all seven sections equally). Higher entropy = more complex policy.

**Effective Feature Count (N_eff)**

```
N_eff = 2^H
```

This is the Hill number of order 1, equivalent to the perplexity of the attribution distribution. It answers: "how many equally-weighted features would have the same entropy as this distribution?" Range: 1 (perfectly concentrated) to 7 (perfectly uniform). This translates entropy into an intuitive count.

**Gini Coefficient**

```
Gini = 1 - (2/n) Σᵢ (cumulative rank sum)
```

Ranges from 0 (perfectly uniform — equal attribution to all sections) to 1 (all attribution on one section). Note that Gini and entropy move in **opposite directions** with complexity: high Gini = low entropy = simple policy.

**Top-2 Dominance**

```
Top2 = (sum of the two largest section attributions) / (total attribution)
```

Fraction of total attribution captured by the two most-attended sections. A simple, human-learnable policy should have Top2 > 0.70 — meaning two features explain most of the decision.

---

### 4.2 Results: Full Metric Table

Baseline checkpoint (epoch3000, 20 IG steps):

**Early game:**

| Metric | greedy/P1 | D1 | P1 | CR | R1 |
|--------|-----------|----|----|-----|-----|
| Entropy (bits) | 2.482 | 2.523 | 2.482 | 2.519 | 2.522 |
| Eff. features | 5.59 | 5.75 | 5.59 | 5.73 | 5.74 |
| Gini | 0.358 | 0.331 | 0.358 | 0.330 | 0.330 |
| Top-2 dom. % | 52.6% | 47.8% | 52.6% | 48.4% | 48.4% |

**Mid game:**

| Metric | greedy/P1 | D1 | P1 | CR | R1 |
|--------|-----------|----|----|-----|-----|
| Entropy (bits) | 2.461 | 2.335 | 2.461 | 2.331 | 2.318 |
| Eff. features | 5.51 | 5.05 | 5.51 | 5.03 | 4.99 |
| Gini | 0.377 | 0.439 | 0.377 | 0.444 | 0.450 |
| Top-2 dom. % | 55.0% | 61.0% | 55.0% | 61.8% | 62.4% |

**Late game:**

| Metric | greedy/noop | D1 | P1 | CR | R1 |
|--------|------------|----|----|-----|-----|
| Entropy (bits) | 2.655 | 2.572 | 2.431 | 2.648 | 2.626 |
| Eff. features | 6.30 | 5.95 | 5.39 | 6.27 | 6.17 |
| Gini | 0.239 | 0.302 | 0.390 | 0.243 | 0.260 |
| Top-2 dom. % | 42.5% | 48.7% | 55.2% | 42.9% | 44.7% |

**Crisis (1 life, 0 hints):**

| Metric | greedy/noop | D1 | P1 | CR | R1 |
|--------|------------|----|----|-----|-----|
| Entropy (bits) | 2.710 | 2.662 | 2.517 | 2.706 | 2.699 |
| Eff. features | 6.54 | 6.33 | 5.73 | 6.53 | 6.49 |
| Gini | 0.185 | 0.232 | 0.348 | 0.191 | 0.201 |
| Top-2 dom. % | 37.9% | 42.3% | 52.3% | 38.5% | 39.9% |

**Aggregate:**
```
Mean entropy across all actions / phases: 2.533 bits
Effective features:                       5.79 / 7
Fraction of maximum possible entropy:     90.2%
```

**Verdict: HIGH-COMPLEXITY policy.** The agent consults nearly 6 out of 7 features simultaneously for every decision, regardless of game phase or action type.

---

### 4.3 Heatmap: Which Features Drive Which Actions

**Figure:** `complexity_results/diag1_heatmap.png`

The heatmap shows the percentage of total absolute IG attributed to each (action, section) pair. A uniform row (all cells ≈ 14%) = fully complex decision. A concentrated row (one cell ≈ 80%, others near 0%) = simple decision.

Key observations:
- **All rows are nearly uniform.** No section exceeds ≈ 30% for any action in early or late game. Mid game has slightly more structure (discards ≈ 30–35% for hint actions), but still far from concentrated.
- **Play (P1) is consistently the least uniform action.** It shows the highest top-2 dominance across all scenarios (52–55% vs 38–48% for hint/noop actions). This makes functional sense: the decision to play a card hinges primarily on two things — what cards remain to be played (fireworks) and what risk you are taking (life tokens).
- **Colour and rank hint actions (CR, R1) are the most uniform.** In late game and crisis, their top-2 dominance drops to 39–45%. Hint actions require synthesising the most information: you must consider your partner's hand, your remaining hint tokens, what hints you've already given, and what plays you're enabling. The uniformly spread attribution reflects this genuine multi-factor computation.
- **noop (greedy in late/crisis) has the lowest top-2 dominance (38–42%).** When no action is clearly better than others, the Q-value landscape is flat and no feature drives the decision strongly — the attribution becomes nearly uniform.

---

### 4.4 Entropy and Effective Feature Count

**Figures:** `complexity_results/diag2_entropy.png`, `complexity_results/diag3_effective_n.png`

**Trends across game phases:**

The most striking pattern is that complexity **increases** as the game becomes more constrained. The mean entropy across action types is:

| Phase | Mean H | Mean N_eff |
|-------|--------|-----------|
| Early game | 2.516 bits | 5.71 |
| Mid game | 2.378 bits | 5.14 |
| Late game | 2.587 bits | 6.01 |
| Crisis | 2.659 bits | 6.33 |

Mid-game has the **lowest entropy** (most focused attention), which seems counterintuitive at first. The explanation is that mid-game has the richest information structure: discards and opponent's hand jointly dominate (see Section 3.4), giving the attribution distribution a visible peak. In early game, all sections have moderate weight. In late game and crisis, the flat Q-value landscape causes entropy to spike back up — when no action clearly dominates, no feature clearly dominates.

**By action type:**

- **P1 (play card 0):** Consistently lowest entropy across all phases (2.43–2.52 bits). Playing a card is the most "decisive" action — it depends on a relatively focused set of features (fireworks + life tokens).
- **CR/R1 (hint actions):** Consistently highest entropy in late game and crisis (2.63–2.71 bits). Hint actions are the most informationally complex — the agent is synthesising the most features.
- **noop:** Highest entropy of all in crisis (2.71 bits, 6.54 effective features). When forced into a no-op by game constraints, the policy has no clear information basis and spreads attribution nearly uniformly.

**Reference lines from diag2_entropy.png:**

The horizontal dashed lines at log₂(2) = 1.0, log₂(3) = 1.58, log₂(4) = 2.0, log₂(5) = 2.32 bits show what "2 effective features," "3 effective features," etc. look like. All measured values fall **above** log₂(5) = 2.32, meaning every action in every phase consults at least 5 features effectively. The target for human-learnable conventions (≤3 features) corresponds to H ≤ 1.58 bits — the current policy is uniformly above this threshold by 0.8–1.2 bits.

---

### 4.5 Concentration Curves

**Figure:** `complexity_results/diag4_concentration.png`

For each action and scenario, sections are ranked by attribution (highest to lowest) and a cumulative attribution fraction is plotted against rank. The steepness of the rise reveals the concentration structure.

Key readings:
- **Top-1 section captures ≈ 25–35%** of attribution across all conditions. There is always a most-attended feature, but it is not dominant.
- **Top-2 sections together capture ≈ 38–62%.** The gap to the target of 70% is substantial.
- **Top-3 sections capture ≈ 55–75%.** Most of the signal is in the top half of the sections, but the long tail is non-negligible.
- **P1 in mid-game** reaches the 80% threshold fastest (by top-3 sections), making it the closest to an ideally focused policy.
- **noop in crisis** has the flattest concentration curve — all 7 sections contribute nearly equally, reaching 80% only at top-5 or top-6.

The green shaded band (target zone: top-2 sections capture ≥80%) shows no action-scenario pair in this checkpoint meets the target. The gap ranges from 8 to 42 percentage points below the 80% target when considering only top-2 sections.

---

## 5. Cross-Cutting Findings

Combining all three phases, the following coherent picture emerges:

**Finding 1: The LSTM, not BERT, is the primary policy carrier.**

From Phase 1, action embeddings collapse within types (cosine > 0.999), meaning BERT does not differentiate between individual actions of the same type. The LSTM hidden state therefore bears the full burden of card-specific policy decisions. BERT acts as a semantic preprocessor that separates the game state into a rich embedding, but the temporal reasoning and action selection live in the LSTM.

**Finding 2: The architecture has learned appropriate inductive biases for its components.**

The fc_v / fc_a orthogonality (cosine = −0.009) shows the dueling head decomposition worked as intended. The negative forget gate bias shows the LSTM has learned to be selective about memory persistence. The BERT attention head differentiation into "sink" and "semantic" shows the transformer has learned functional specialisation. These are all hallmarks of a well-trained network, not a degenerate one.

**Finding 3: The agent is genuinely multi-factorial, not lazily uniform.**

The attribution analysis is not showing a flat, uninformative gradient — it is showing real, game-phase-appropriate shifts (life tokens dominate early, discards dominate mid-game). The problem is not that the agent is random or unstructured. The problem is that the game state genuinely contains 7 independently relevant sections, and the agent has learned to use all of them, which is optimal for Hanabi but catastrophic for human interpretability.

**Finding 4: Hint actions are intrinsically more complex than play/discard actions.**

Play and discard decisions concentrate on 5 effective features (entropy ≈ 2.4 bits); hint decisions concentrate on 6+ features (entropy ≈ 2.6–2.7 bits). This reflects the genuine information structure of Hanabi: giving a good hint requires knowing the full game state (what is playable, what your partner holds, what hints you have left, what the discards imply), while playing a card depends primarily on your hand knowledge and the fireworks state.

**Finding 5: The agent becomes maximally complex exactly when humans need simplicity most.**

In the crisis scenario (1 life, 0 hints) — the situation where a human partner most needs to predict and understand the agent's actions — the agent reaches its highest complexity (H = 2.71 bits, N_eff = 6.54). At this moment, the Q-value landscape is nearly flat (all actions have similar expected payoff because the game is constrained), so attribution spreads uniformly. The agent has no confident view, and this epistemic uncertainty manifests as complexity in the attribution profile.

---

## 6. What This Means for Human-AI Collaboration

The fundamental challenge this analysis reveals is a **complexity–optimality tension**:

The agent achieves high game scores by learning to synthesise all available information for every decision. This is exactly what a strong Hanabi player does — track everything simultaneously. But human partners cannot observe or internalise a 7-dimensional latent convention. When collaborating with a new AI partner, a human implicitly asks: "what are the 2–3 rules this agent follows?" If the agent follows no rules — or rather, follows a 5–6 dimensional rule — the human cannot model it.

The evidence for this:

1. **No action has a single dominant feature:** Top-1 section attribution never exceeds 35%, meaning even the most important feature explains less than a third of any decision.

2. **Convention complexity scales with game state richness:** Mid-game, where information is richest, produces the most structured (least entropic) attribution. Early and late game, where fewer things are decided, produce near-uniform attribution. This means the agent's convention conventions are context-dependent — a human would need to learn not one rule but a table of rules indexed by game phase.

3. **The agent's greedy action changes across scenarios** (P1 in early/mid, noop in late/crisis), but the underlying feature attribution profile stays uniformly high-entropy. The policy varies while remaining uniformly complex — humans cannot use the greedy action pattern to infer the underlying reasoning.

---

## 7. Next Steps: Retraining with Entropy Penalty

The diagnostic analysis motivates a concrete intervention: add an **attention entropy penalty** to the training objective that discourages the agent from spreading attribution uniformly, pushing it toward developing conventions that depend on fewer, more decisive features.

### Mathematical formulation

During training, after each forward pass through the BERT encoder, extract the CLS-token attention row from the last transformer layer and aggregate it over the 7 game-state sections:

```python
# A: [batch, num_heads, seq, seq]
attn_mean = A.mean(dim=1)          # average over heads: [batch, seq, seq]
cls_row = attn_mean[:, 0, :]       # CLS attention: [batch, seq]
section_mass = [cls_row[:, s:e].sum(dim=1) for (s, e) in spans]
S = torch.stack(section_mass, dim=1)          # [batch, 7]
p = S / S.sum(dim=1, keepdim=True)
H_penalty = -(p * (p + 1e-10).log()).sum(dim=1).mean()   # scalar, nats
```

Total loss:
```
L_total = L_TD + λ_aux · L_aux + λ_H · H_penalty
```

### Implementation locations

1. **`pyhanabi/q_net.py`** — `TextLSTMNet.__init__()`: add `self.penalty_bert` (non-scripted BertModel with `output_attentions=True`) and a `sync_penalty_bert()` method.

2. **`pyhanabi/q_net.py`** — `TextLSTMNet.attention_entropy_loss()`: new method computing H_penalty from a batch of input_ids and section spans.

3. **`pyhanabi/r2d2.py`** — `R2D2Agent.loss()`: call `attention_entropy_loss()` and add weighted term to total loss.

4. **Training loop** — Add curriculum schedule for λ_H with linear warmup from epoch 500 to 1000.

### Expected trajectory

| λ_H | Target entropy | Effective features | Expected score impact |
|-----|----------------|-------------------|----------------------|
| 0.00 | 2.53 bits | 5.79 | baseline |
| 0.05 | 2.0–2.2 bits | 4.0–4.6 | < −0.3 pts |
| 0.10 | 1.8–2.0 bits | 3.5–4.0 | −0.3 to −0.7 pts |
| 0.20 | 1.3–1.7 bits | 2.5–3.2 | −0.5 to −1.2 pts |
| 0.50 | < 1.2 bits | < 2.5 | monitor carefully |

### Monitoring protocol

Run `python tools/complexity_diagnostics.py` every 250 epochs during retraining. Track:
1. Mean attribution entropy → should decrease monotonically
2. Top-2 dominance → should increase toward 70%
3. Self-play score → should stay within 0.5 pts of baseline until λ_H > 0.3
4. Cross-play score with the baseline checkpoint → should be maintained or improved

The target condition is: **H < 1.5 bits AND Top-2 dominance > 70% AND score within 1.0 pts of baseline.**

---

## 8. Figure Index

| Figure | Script | Key quantity |
|--------|--------|-------------|
| `interpret_results/weight_norms.txt` | interpretability.py | Frobenius norm of every parameter tensor |
| `interpret_results/lstm_gate_analysis.png` | interpretability.py | Forget / input / output gate bias distributions |
| `interpret_results/fc_v_vs_fc_a.png` | interpretability.py | Cosine similarity between value and advantage head weight vectors |
| `interpret_results/bert_attention_maps.png` | interpretability.py | Full [seq × seq] attention matrix per head for a sample input |
| `interpret_results/bert_attention_per_head.png` | interpretability.py | Per-head attention summary statistics |
| `interpret_results/action_embedding_similarity.png` | interpretability.py | Pairwise cosine similarity of all 21 action embeddings |
| `interpret_results/action_embedding_pca.png` | interpretability.py | First 2 PCA components of the 21 × 128 action embedding matrix |
| `attribution_results/fig1_token_heatmaps.png` | attribution.py | Per-token saliency and IG for greedy action in each game scenario |
| `attribution_results/fig2_component_bars.png` | attribution.py | Section-level absolute IG for 5 probe actions across 3 scenarios |
| `attribution_results/fig3_cross_scenario.png` | attribution.py | Section importance shift from early → mid → late game (greedy action) |
| `attribution_results/fig4_signed_ig.png` | attribution.py | Signed IG per section (which sections push Q up vs down) |
| `complexity_results/diag1_heatmap.png` | complexity_diagnostics.py | Attribution % heatmap (actions × sections) per scenario |
| `complexity_results/diag2_entropy.png` | complexity_diagnostics.py | Shannon entropy (bits) per action × game phase, with reference lines |
| `complexity_results/diag3_effective_n.png` | complexity_diagnostics.py | Effective feature count (2^H) grid: scenario × action |
| `complexity_results/diag4_concentration.png` | complexity_diagnostics.py | Cumulative attribution curves (feature-rank plots) |
| `complexity_results/diag5_summary.png` | complexity_diagnostics.py | Overall feature reliance bar chart + full entropy grid overview |
