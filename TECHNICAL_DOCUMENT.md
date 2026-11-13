# Technical Report: Spatio-Temporal Possession Detection
**Team Name:** Senators  
**Competition:** More than a Hack 2027 — Phase 1: Ball Possession  
**Organizers:** Metrica Sports, FC Barcelona (Barça Innovation Hub), Mobile World Capital Barcelona  
**Primary Deliverable:** `evaluation/results/Senators.csv` (199,319 frames)  
**Public Code Repository:** https://github.com/anshu8895/senators  
**Public Window Result:** `spellF1 = 0.5161` | `macroF1 = 0.7032` (highest on the public 5-minute sanity-check window)

---

## Benchmark Summary (Public 5-Minute Window)

The public sanity-check ground truth covers 7,500 frames (5 minutes of play, 17 true possessions, mean duration 14.2 s). Evaluated on the official scorer (`evaluation/run_evaluation.py`):

| Submission | `spellF1` (Ranking Key) | `macroF1` (Tie-Breaker) | Possessions | Mean Duration | Trans Latency ($\tau=25$) | Fragmentation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Senators (Our Submission)** | **0.5161** | **0.7032** | 14 | 17.8 s | -9.4 frames | 0.95 |
| `rule_based_pred` (Our Baseline) | 0.4667 | 0.7187 | 19 | 13.4 s | -9.5 frames | 1.24 |
| `baseline_smooth125` | 0.4242 | 0.5812 | 16 | 18.5 s | -7.7 frames | 0.81 |
| `baseline_smooth50` (Contest Floor: 0.2433) | 0.3256 | 0.6207 | 26 | 10.8 s | +4.0 frames | 1.43 |
| `baseline_smooth25` | 0.2868 | 0.6200 | 36 | 7.6 s | +8.2 frames | 1.90 |
| `baseline_raw` | 0.0378 | 0.5972 | 131 | 2.0 s | -3.1 frames | 6.52 |

- **`spellF1` comparison:** $+21.7\%$ relative gain ($1.22\times$) over the strongest public baseline (`baseline_smooth125` at `0.4242`), and $+112\%$ over the published match floor (`0.2433`).
- **`macroF1` comparison:** $+13.3\%$ relative gain over the best baseline macro-F1 (`baseline_smooth50` at `0.6207`).
- **Rule compliance:** 100% of training data was sourced from external open data. Zero frames of the competition match (`BAR-ATH`) were used for training.

---

## 1. Approach

The challenge requires predicting one label per video frame across all 199,319 frames of the tactical video feed (25 fps, ~2 hours 13 minutes) for FC Barcelona (`H`) vs Athletic Club (`A`):
- `H`: Home team (FC Barcelona) in possession, ball alive.
- `A`: Away team (Athletic Club) in possession, ball alive.
- `D`: Dead ball (out of play, stoppage, restart preparation).
- `X`: Abstain / unplayed structural intervals (pre-match, halftime, post-match).

### 1.1 Pipeline Overview

```
[External Metrica Tracking & Events]
 (Sample Games 1 & 2: 286,162 frames)
                  │
                  ▼
[External Label Synthesis (H / A / D)]  ──►  [20-D Feature Extractor with Trajectory Imputation]
                                                                │
                                                                ▼
                                              [PyTorch Transformer Encoder (2 Layers, d=64)]
                                                                │
[BAR-ATH Tactical Tracking (199,319 f)] ──►  [Clean Trajectory Imputation & Feature Extraction]
                                                                │
                                                                ▼
                                              [Inference: P(H), P(A), P(D) Softmax Probabilities]
                                                                │
[Smart Tagging XML + games.py Bounds]   ──►  [Domain Post-Processing (Bilateral Sum, Dead Windows)]
                                                                │
                                                                ▼
                                              [Final Submission: Senators.csv (199,319 rows)]
```

The system operates in three functional stages:
1. **Trajectory Imputation:** Recovering short tracking dropouts via linear extrapolation.
2. **Order-Invariant Feature Extraction:** Compressing 46 anonymous player slots into 20 metric features.
3. **Temporal Attention & Post-Processing:** Modeling sliding 100-frame (4.0 s) windows with a Transformer, smoothed by a bilateral probability window and constrained by structural match bounds.

---

### 1.2 Physics-Based Trajectory Imputation

In the raw competition tracking data, **the ball is untracked (`NaN`) on 43.1% of frames (85,831 out of 199,319 frames)** due to camera occlusion, rapid motion blur, and aerial trajectory.

```
Frame t=100 (Pass Origin) ────► Frames 101..129 (Ball Invisible, NaN) ────► Frame t=130 (Reception)
 Ball at (x0, y0)                 Linear Interpolation across <= 50 frames    Ball at (x1, y1)
                                  Recovers 7,211 frames (8.4% of missing data)
```

- **Short-Gap Linear Interpolation ($\Delta t \le 50\text{ frames} = 2.0\text{ s}$):** Between consecutive sightings within 2.0 seconds, ball coordinates are interpolated linearly along planar axes $x$ and $y$. This recovers **7,211 frames (8.4% of missing ball frames)**, restoring continuous physical inputs during rapid pass exchanges. The remaining ~39.4% of match frames remain untracked.
- **Neutral Stoppage Boundary ($\Delta t > 2.0\text{ s}$):** For gaps exceeding 50 frames, linear interpolation is discontinued. Naive pipelines that fill missing coordinates with `0.0` cause player-to-ball distance to collapse to zero, misleading models into detecting spurious player contact. Our pipeline sets player-to-ball distances to a neutral default of **$50.0\text{ m}$** (unpossessed state) during extended ball absences.

---

### 1.3 Permutation-Invariant 20-Dimensional Spatial Features

The tracking stream contains 46 tracking slots for 22 players. Because slots swap identities and vanish, raw slot arrays cannot be fed directly to a neural network. We compute 20 order-invariant features per frame:

| Index | Feature Description | Formula / Derivation |
| :---: | :--- | :--- |
| `0` | Minimum distance from Home player to ball | $d_{H, \min} = \min_{i \in \text{Home}} \|\mathbf{p}_i - \mathbf{b}\|$ |
| `1` | Minimum distance from Away player to ball | $d_{A, \min} = \min_{j \in \text{Away}} \|\mathbf{p}_j - \mathbf{b}\|$ |
| `2` | Spatial distance advantage delta | $\Delta d = d_{H, \min} - d_{A, \min}$ (negative = Home closer) |
| `3, 4` | 2nd closest player distance to ball | 2nd sorted distance for Home and Away |
| `5, 6` | 3rd closest player distance to ball | 3rd sorted distance for Home and Away |
| `7, 8` | Team mean distance to ball | Average distance of all tracked players per team |
| `9, 10` | Local player density within 5 m | Count of team players with $\|\mathbf{p} - \mathbf{b}\| < 5.0\text{ m}$ |
| `11, 12`| Medium player density within 10 m | Count of team players with $\|\mathbf{p} - \mathbf{b}\| < 10.0\text{ m}$ |
| `13, 14`| Ball pitch position (normalized) | $b_x / 105.0$ and $b_y / 68.0$ (0.5 if untracked) |
| `15, 16`| Team centroid longitudinal position | Mean $x$ coordinate of tracked players / pitch length |
| `17, 18`| Ball field zone indicators | Binary flags: ball in defensive third ($x < 0.33$) or attacking third ($x > 0.67$) |
| `19` | Continuous ball velocity (m/s) | $\Delta \text{dist} \times 25.0$, clipped to $[0.0, 40.0]\text{ m/s}$ |

Sorting within each team roster makes these features strictly invariant to tracker slot permutations.

---

### 1.4 PyTorch Transformer Architecture

The model processes sliding temporal windows of 100 frames ($4.0\text{ s}$ at 25 fps):

```
Input: Tensor (Batch, 100, 20)
  │
  ▼
Linear Projection: 20 -> d_model (64)
  │
  ▼
Sinusoidal Positional Encoding (max_len = 150)
  │
  ▼
Transformer Encoder (2 Layers, Pre-LN, norm_first=True)
  ├─ Multi-Head Attention (4 heads, d_k = 16, dropout = 0.1)
  └─ Feed-Forward Network (Linear 64 -> 256 -> ReLU -> Linear 256 -> 64, dropout = 0.1)
  │
  ▼
Temporal Center Extraction: x[:, window//2, :] (Center Frame at index 50)
  │
  ▼
Classification Head: LayerNorm(64) -> Linear(64, 64) -> GELU -> Dropout(0.1) -> Linear(64, 3)
  │
  ▼
Softmax Output: [P(Home), P(Away), P(Dead)]
```

- **Context Window:** 100 frames ($4.0\text{ s}$).
- **Encoder Layers:** 2 layers, 4 attention heads, $d_{model} = 64$, feed-forward dimension $d_{ff} = 256$ with ReLU activation.
- **Classification Head:** 2-layer MLP with GELU non-linearity and Layer Normalization.
- **Parameter Count:** ~85,000 parameters (compact, preventing overfitting on small tracking corpora).

---

## 2. Decisions

### 2.1 Training Data & External Dataset Methodology

The competition rules state:
> *"No hand labels on the provided game, in any amount and for any purpose — not for the submission, not to train a model and not to validate one... The 5 minutes published here are for checking that your pipeline is correct... not a tuning set."*

In accordance with this rule, **no frame of the competition match (`BAR-ATH`) was used during training.**

- **External Dataset:** Metrica Sports Open Tracking and Event Data ([GitHub Repository](https://github.com/metrica-sports/sample-data)).
- **Matches Used:**
  - `Sample_Game_1`: 145,006 tracking frames, 1,745 event annotations.
  - `Sample_Game_2`: 141,156 tracking frames, 1,935 event annotations.
  - **Total Training Volume:** 286,162 frames ($\approx 190.7$ minutes of professional soccer).
- **External Label Synthesis:** Frame-level labels were synthesized algorithmically from event logs:
  - Sequences of `PASS`, `RECOVERY`, `SHOT`, `CHALLENGE`, `FAULT RECEIVED`, or `BALL LOST` assign continuous possession (`H` or `A`) to the acting team until turnover or stoppage.
  - Events tagged as `SET PIECE`, `BALL OUT`, or `CARD` assign dead-ball state (`D`) spanning 2.0 s pre-event to 3.0 s post-event.
  - External label split: Home (`H`) 32.6%, Away (`A`) 28.6%, Dead Ball (`D`) 38.8%.
- **Optimization:** AdamW optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), Cosine Annealing scheduler, batch size 256, weighted Cross-Entropy loss ($H=1.02, A=1.16, D=0.86$). Early stopping evaluated on an 85/15 chronological split with patience of 8 epochs.
- **Random Seed:** Pinned to `SEED = 42` across Python `random`, `numpy.random`, and `torch.manual_seed` (including DataLoader `torch.Generator`).

---

### 2.2 Assets Used and Dropped

| Asset | Source in Bundle | Decision | Rationale |
| :--- | :--- | :---: | :--- |
| **Tactical Tracking (ATD)** | `data/tactical/BAR-ATH_tactical_atd.txt` | **USED** | Authority 25 fps timeline. Provides metric positions for 46 player slots and ball across 199,319 frames. |
| **ATD Metadata** | `data/tactical/BAR-ATH_tactical_atd.xml` | **USED** | Maps tracking slots to Home (FC Barcelona) and Away (Athletic Club) rosters. |
| **Smart Tagging XML** | `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | **USED** | Provides 638 automated tactical instances. Used solely for set-piece timestamps and goal intervals. |
| **Match Structure (`games.py`)** | `games.py` | **USED** | Authoritative frame boundaries for kickoff anchors, halftime, pre-match, and post-match masks. |
| **Broadcast Video & Tracking** | `data/broadcast/` | **DROPPED** | Unsynchronized with tactical timeline (164k vs 199k frames). Cuts and replays introduce temporal artifacts without providing new physical data. |
| **Tactical Video Pixels (MP4)** | `data/tactical/BAR-ATH_tactical.mp4` | **DROPPED** | Dropped to ensure zero risk of visual memorization or contest rule ambiguity. 2D tracking features provide calibrated metric pitch coordinates without vision inference overhead. |
| **Homography Matrices** | `data/tactical/BAR-ATH_tactical_homography.csv` | **DROPPED** | Calculations operate directly in pitch metric coordinates ($105.3\text{ m} \times 68.0\text{ m}$), bypassing pixel-to-pitch homography transformations. |

---

### 2.3 Post-Processing & Engineering Priors

All post-processing parameters were selected as engineering priors in advance based on standard football sequence conventions, rather than tuned against the public 5-minute slice:

1. **Bilateral Probability Rolling Sum ($w = \pm 90\text{ frames}$):**
   A cumulative rolling sum of class probabilities across window $[t - w, t + w]$ (radius $w = 90$ frames = 3.6 s, total span 181 frames = 7.24 s). This serves as a temporal smoothing filter: it smooths intra-possession passing dynamics and tracking dropouts without blurring major team possession transitions (which typically span 10–25 s).
2. **Micro-Run Absorption Threshold (`min_spell = 50\text{ frames} = 2.0\text{ s}$):**
   Transient rival incursions lasting under 50 frames ($2.0\text{ s}$) that immediately revert to the previous possessor are treated as contested duels within the existing possession spell and are absorbed into the surrounding team label.
3. **Restart Dead-Ball Windows:**
   Stoppage windows applied to Smart Tagging set-piece instances, beginning 2.0 s prior to the logged timestamp to account for whistle latency:
   - *Throw-in:* 3.0 s dead ball.
   - *Goal Kick / Kick-off:* 4.0–4.5 s dead ball.
   - *Corner Kick:* 5.0 s dead ball.
   - *Free Kick:* 5.5 s dead ball.
   - *Goal Celebrations:* The entire interval between goal occurrence and subsequent kickoff restart is assigned to `D`.
4. **Structural Match Zones (Derived dynamically from `games.py`):**
   - Pre-match: frames $0 \dots 11184 \to \text{X}$.
   - Half 1: frames $11185 \dots 90651$.
   - Halftime: frames $90652 \dots 110807 \to \text{X}$.
   - Half 2 Pre-Kickoff: frames $110808 \dots 113741 \to \text{D}$.
   - Half 2 Live Play: frames $113742 \dots 190197$.
   - Post-match: frames $190198 \dots 199318 \to \text{X}$.

---

## 3. Limitations

1. **Persistent Untracked Ball (39.4% of Match):**
   While short-gap interpolation recovers 7,211 frames (8.4% of missing ball observations), approximately 39.4% of all frames remain without a tracked ball. During prolonged aerial flight or camera occlusions, distance features default to a static 50.0 m boundary, which can introduce transition latency on restarts.
2. **Sample Size of Public Validation (17 Possessions):**
   The public ground truth contains only 17 possessions across 5 minutes of play. In a sample of this size, a single boundary disagreement shifts `spellF1` by roughly $\approx 0.06$. As noted in `EVALUATION.md`, bootstrap intervals on this sample are wide, and public performance may not reflect full-match private ranking.
3. **Limited External Training Volume (2 Matches):**
   Training was conducted on 286,162 frames across two open Metrica matches. Differences in tactical style, camera optical zoom, and tracking software calibration between Metrica's open games and the Camp Nou recording represent an unmodeled domain shift.
4. **Pseudo-Label Training Noise:**
   External ground-truth possession labels were synthesized from discrete event logs (passes, recoveries, stoppages) rather than frame-by-frame manual tracking. Event-to-frame boundary heuristics introduce training noise near turnover and stoppage instants.
5. **Goalmouth Congestion:**
   In crowded penalty box situations (8+ players within the 6-yard box), spatial distance differentials compress, occasionally delaying turnover detection by several frames.
6. **Planar 2D Projection:**
   The tracking feed provides 2D metric coordinates on the pitch plane $(x, y)$ but lacks ball altitude ($z$). High aerial clearances appear planar and can cause temporary horizontal displacement before the ball lands.

---

## 4. Reproduction Guide

The codebase is deterministic and reproduces the primary submission `evaluation/results/Senators.csv`.

### Step 1: Environment Setup
Install dependencies with pinned version ranges:
```bash
pip install -r requirements-submission.txt
```
*(Dependencies: `torch>=2.0.0,<=2.5.0`, `numpy>=1.24.0,<=2.3.5`, `pandas>=2.0.0,<=2.3.3`)*

### Step 2: Run Inference (Zero-Retrain Mode)
Using the committed model weights (`ext_transformer_weights.pt`) and normalization parameters (`norm_stats.npz`):
```bash
python3 ext_transformer_detector.py
```
- **Runtime:** Measured ~20 seconds on Apple Silicon (M-series, MPS backend) or ~45 seconds on multi-core CPU.
- **Output:** Writes exactly 199,319 rows to `evaluation/results/Senators.csv`.

### Step 3: (Optional) Retrain from Scratch on External Metrica Data
To retrain the Transformer model deterministically from raw external data:
1. Ensure Metrica sample data is present under `external_data/metrica/data/Sample_Game_1` and `Sample_Game_2` (cloned from `https://github.com/metrica-sports/sample-data`).
2. Run with the `--retrain` flag:
```bash
python3 ext_transformer_detector.py --retrain
```
This retrains the model with `SEED = 42`, overwrites `ext_transformer_weights.pt` and `norm_stats.npz`, and regenerates `evaluation/results/Senators.csv`.

### Step 4: Validate Submission Format
```bash
python3 evaluation/validate_submission.py evaluation/results/Senators.csv
```
Expected output:
```
OK    Senators.csv
      199319 rows, frame 0..199318
      label counts: H 47190 (23.7%), A 32462 (16.3%), D 79205 (39.7%), X 40462 (20.3%)
      reports 227 possessions of mean 14.0 s
Valid.
```

### Step 5: Score on the Public Ground Truth
```bash
python3 evaluation/run_evaluation.py
```
Expected output:
```
rank  submission                macroF1  spellF1   poss     dur  transF1  lat_f  frag
----  ------------------------  -------  -------  -----  ------  -------  -----  ----
   1  Senators                   0.7032   0.5161     14    17.8   0.2564   -9.4  0.95
   2  rule_based_pred            0.7187   0.4667     19    13.4   0.3556   -9.5  1.24
   3  baseline_smooth125         0.5812   0.4242     16    18.5   0.1667   -7.7  0.81
```
