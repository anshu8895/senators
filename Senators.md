# Technical Report: Spatio-Temporal Possession Detection
**Team Name:** Senators  
**Competition:** More than a Hack 2027 — Phase 1: Ball Possession  
**Organizers:** Metrica Sports, FC Barcelona (Barça Innovation Hub), Mobile World Capital Barcelona  
**Primary Deliverable:** `evaluation/results/Senators.csv` (199,319 frames)  
**Public Code Repository:** https://github.com/anshu8895/senators  
**Public Window Result:** `spellF1 = 0.6483` | `macroF1 = 0.8196` (public 5-minute sanity-check window: 17 possessions, so one possession is worth about 0.06 `spellF1`; the scorer's bootstrap puts this window's noise band at about 0.37)

**Which file does what**

| File | Role |
| :--- | :--- |
| `possession_v2.py` | **Entry point.** `--predict` writes the submission; `--train` trains one ensemble member. |
| `ext_v3_s{0,1,2}_weights.pt`, `norm_stats_v3_s*.npz` | The three models of the submitted ensemble. |
| `decoder.py` | Viterbi decoding of the per-frame probabilities into possessions. |
| `tracking_corruption.py`, `metrica_data.py` | Training-data construction from the open Metrica games. |
| `ext_transformer_detector.py` | Shared helpers: loaders, the 20 base features, the transformer, Smart Tagging windows. |
| `ext_v2_weights.pt`, `norm_stats_v2.npz` | The comparison model `Senators_v2` (one model, smoothing instead of the decoder). |
| `metrica_testbed.py` | Held-out check on Metrica Sample Game 3. |

---

## Benchmark Summary

### Public 5-minute window (correctness check only)

The public ground truth covers 7,500 frames (5 minutes of play, 17 true possessions, mean duration 14.2 s). Official scorer (`evaluation/run_evaluation.py`):

| Submission | `spellF1` (ranking key) | `macroF1` (tie-break) | Possessions | Mean duration | Latency (frames, $\tau=25$) | Fragmentation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Senators (final: v3 ensemble + Viterbi decoder)** | **0.6483** | **0.8196** | 12 | 22.4 s | -7.6 | 0.67 |
| `Senators_v2` (one v2 model + smoothing) | 0.5455 | 0.7328 | 16 | 16.2 s | -3.4 | 1.10 |
| `baseline_smooth125` (organiser) | 0.4242 | 0.5812 | 16 | 18.5 s | -7.7 | 0.81 |
| `baseline_smooth50` (organiser) | 0.3256 | 0.6207 | 26 | 10.8 s | +4.0 | 1.43 |
| `baseline_smooth25` (organiser) | 0.2868 | 0.6200 | 36 | 7.6 s | +8.2 | 1.90 |
| `baseline_raw` (organiser) | 0.0378 | 0.5972 | 131 | 2.0 s | -3.1 | 6.52 |

**Read this table as a sanity check, not as evidence of ranking.** With 17 possessions the gap between the two Senators rows is inside the noise band, and `EVALUATION.md` warns against choosing between versions on this window. The published full-match floor is `spellF1 = 0.2433` (`baseline_smooth50`), which this window cannot reproduce. On this window the final output **merges possessions** (12 predicted against 17 true, mean 22.4 s against 14.2 s) and finds only half of the dead-ball frames (precision 0.95, recall 0.52); see Section 3.

### Held-out Sample Game 3 (the comparison the design rests on)

`metrica_testbed.py` scores both models on Metrica Sample Game 3, which was never used for training, with the official metric code. Ground truth comes from Metrica's human event logs, so it is a relative yardstick, not a prediction of the private score. Smart Tagging does not exist for this game, so the dead-ball windows are not applied here. Mean of 3 corruption seeds (one run for `raw`); `spellF1` differences below about 0.04 are noise.

| Condition | `Senators_v2` `spellF1` | **final** `spellF1` | `Senators_v2` `macroF1` | **final** `macroF1` |
| :--- | :---: | :---: | :---: | :---: |
| `raw` (open data as delivered) | 0.563 | **0.594** | 0.825 | **0.841** |
| `indep` (missing ball tells nothing) | 0.480 | **0.508** | 0.793 | **0.810** |
| `bar-ath` (BAR-ATH-like ball gaps on the raw data) | 0.508 | **0.536** | 0.798 | **0.809** |

The final model is ahead by about 0.03 `spellF1` in every condition, which is at the edge of the noise band but has the same sign each time. BAR-ATH lies between `indep` and `raw`. Games 1 and 2 are training data; Game 3 was looked at several times during development (Section 3).

---

## 1. Approach

One label per video frame across all 199,319 frames of the tactical video (25 fps) for FC Barcelona (`H`) vs Athletic Club (`A`): `H` or `A` for the team in possession with the ball alive, `D` for dead ball, and `X` for the unscored intervals (pre-match, half-time, post-match).

### 1.1 Pipeline overview

```
[Metrica open Sample Games 1 & 2: tracking + human event logs]
        │  event logs → frame labels (PASS / RECOVERY / SHOT change possession; stoppages = D)
        │  tracking made as incomplete as BAR-ATH: ball held through dead frames or left as is,
        │  then dropped in runs, players dropped (fresh copies every 3 epochs, both regimes)
        ▼
[28-D features] ─► [Transformer encoder, 4 s window] ─► P(H), P(A), P(D) per frame
                                                          (3 models, seeds 0-2, averaged)
                                                                   │
[BAR-ATH tactical tracking (ATD)] ─► same 28 features ─────────────┘
                                                                   ▼
[Smart Tagging set-piece / goal windows → forced D] ─► [Viterbi decoding per half] ─► labels
                                                                   ▼
        structural zones: X outside the graded halves, D before the 2nd-half kick-off
                                                                   ▼
                                      evaluation/results/Senators.csv (199,319 rows)
```

### 1.2 Why this design

A model trained directly on the open Metrica games learns from data in which the ball is missing almost only while it is dead (on Sample Game 3, 83% of dead-ball frames have no tracked ball against 7% of live frames), so it learns "ball missing means dead ball". BAR-ATH is different: the ball is also missing during live play (41% of graded frames) and is often tracked during stoppages. Every false dead-ball label inside a possession can split it in two, which `spellF1` penalises. The design therefore (1) trains on data where missingness carries no information about the label, (2) gives the model features that do not need the ball, and (3) decodes possessions as whole sequences instead of smoothing frame by frame.

### 1.3 Features (28 per frame, permutation-invariant over the anonymous player slots)

| Index | Feature |
| :---: | :--- |
| 0-2 | Closest Home / Away player distance to the ball, and their difference |
| 3-6 | 2nd and 3rd closest distance, per team |
| 7-8 | Team mean distance to the ball |
| 9-12 | Players within 5 m and within 10 m of the ball, per team |
| 13-14 | Ball position, normalised (0.5 if untracked) |
| 15-16 | Team centroid, longitudinal position |
| 17-18 | Ball in defensive / attacking third flags |
| 19 | Ball speed (m/s, clipped to 40) |
| 20-21 | **Median player speed** of the Home / Away team (finite difference over 0.48 s, speeds above 12 m/s dropped as identity switches, median over players, 2 s moving average) |
| 22 | **Ball tracked** flag (before interpolation) |
| 23 | **Frames since the ball was last tracked**, capped at 100 and divided by 100 |
| 24 | **Ball outside the pitch lines** flag |
| 25 | **Signed distance of the ball to the nearest pitch line** (m, negative outside; 10 if untracked) |
| 26-27 | **Ball displacement over ±0.5 s** (m), and whether both ends were tracked |

(`Senators_v2` uses features 0-23.) Distances use the ball path linearly interpolated across interior gaps of at most 50 frames (2.0 s): on BAR-ATH this fills 7,211 of the 85,831 untracked ball frames, and the remaining 78,620 (39.4% of the match) take a neutral distance of 50.0 m. Player speed (20-21) needs no ball at all, which is why it is the main fallback when the ball is lost.

### 1.4 Model

For each frame the network sees the 100 frames (4.0 s) centred on it.

```
Input (batch, 100, 28) → Linear 28→64 → sinusoidal positional encoding
→ Transformer encoder (2 layers, Pre-LN, 4 heads, feed-forward 256, dropout 0.3)
→ centre frame (index 50) → LayerNorm → Linear 64→64 → GELU → Dropout → Linear 64→3 → softmax
```

106,307 trainable parameters per model. Three models (seeds 0, 1, 2) are averaged at the probability level, so the output does not depend on one seed: training stopped at quite different epochs for the three seeds (best epochs 10, 5 and 10).

### 1.5 Viterbi decoding (`decoder.py`)

The decoder takes the single best path through the three states `H`, `A`, `D` for each half:
- evidence: $\log p_t(\text{state})$ with probabilities clipped at 0.01;
- cost of a switch: 50 nats for `H`↔`A` (a turnover) and 75 nats into or out of `D` (a stoppage), so a flicker is kept only if the evidence outweighs its price;
- Smart Tagging set-piece and goal windows are constraints: inside them only `D` is allowed;
- the first half is decoded over frames 11185-90651 and the second over 113742-190197; frames 110808-113741 (before the second-half kick-off) are `D`, everything else is `X`.

`Senators_v2` instead smooths the probabilities over ±90 frames and removes runs shorter than 50 frames (both constants inherited from the first pipeline).

---

## 2. Decisions

### 2.1 Training data and augmentation

The rules allow hand labels from other games and forbid them on the provided game, so every label comes from other matches.

- **Source:** Metrica Sports open tracking and event data ([GitHub](https://github.com/metrica-sports/sample-data)), Sample Games 1 and 2 (about 286,000 frames; human event annotation produced by Metrica analysts with video tagging software).
- **Frame labels from events:** produced by code (`metrica_data.py`, `ext_transformer_detector.generate_possession_labels_from_events`). Only `PASS`, `RECOVERY` and `SHOT` give possession to the acting team, following `CHALLENGE.md` ("a duel without subsequent control is not a change of possession"); `SET PIECE`, `BALL OUT` and `CARD` events create dead-ball windows; time outside play is `D`.
- **Held out:** the first 85% of each game trains the model; the last 15% is used only for early stopping. **Sample Game 3 is never trained on**; it is the test game.
- **Making the data look like BAR-ATH** (`tracking_corruption.py`). Every training copy is built in one of two regimes, and both appear for both games in every block:
  1. `indep`: the ball is held at its last tracked position through dead-ball frames, then dropped in runs independently of play;
  2. `as-delivered`: the open data's own missing ball plus the same extra gaps.
  In both, players are dropped independently so that about 20.5 of 22 are tracked, the rate `DATA.md` reports. Fresh corrupted copies are generated every 3 epochs, so the model cannot memorise one pattern of gaps.
- **The one BAR-ATH-derived input.** The lengths of the ball-tracked and ball-missing stretches in BAR-ATH's graded halves (only whether the ball field is `NaN`, never a label) set how long the simulated gaps are. The run lengths are cached sorted (`external_data/bar_ath_ball_runs.npz`, not committed), so the cache cannot be read back as the match's visibility timeline. This is an unlabeled statistic of the provided tracking, used for domain adaptation by an algorithm.
- **Recipe:** AdamW (learning rate $3\times10^{-4}$, weight decay $10^{-2}$), cosine schedule over 30 epochs, batch 256, 60,000 random windows per epoch, dropout 0.3, label smoothing 0.1, Gaussian input noise (0.1 in normalised units), class-weighted cross-entropy (H 1.08, A 1.27, D 0.78), early stopping with patience 6 on the validation loss. Seeds 0, 1 and 2 (`torch.manual_seed` and a seeded corruption generator); stopped at epochs 16, 11 and 16, best epochs 10, 5 and 10. Two matches are little data for a transformer, and validation loss starts rising after a few epochs, which is why the recipe regularises hard.

### 2.2 Assets used and dropped

| Asset | Decision | Rationale |
| :--- | :---: | :--- |
| **Tactical tracking (ATD)** `data/tactical/BAR-ATH_tactical_atd.txt` | **USED** | 25 fps authority timeline: player and ball positions. Speeds are recomputed from positions; the tracker's own speed field is not used. |
| **ATD metadata** `..._atd.xml` | **USED** | Maps track slots to Home (Barcelona) and Away. |
| **Smart Tagging XML** | **USED (set pieces and goals only)** | Dead-ball constraints in the decoder. Measured on this file: a kick-off tag starts about 3.4 s before the real kick-off and lasts 5 s; throw-in, goal-kick, corner and free-kick tags last 35 s, and player speed lined up on their start dips and then rises, as around real restarts in Sample Game 3, so such a tag starts about 3.4 s before the restart and its length is mostly the play that follows. The tactical-phase tags (Build Up, Progression, ...) are not used. |
| **`games.py` frame boundaries** | **USED** | Graded halves, second-half kick-off, frame alignment. |
| **Broadcast video and tracking** | **DROPPED** | Not synchronised to the tactical timeline (164k vs 199k frames), with cuts and replays. |
| **Tactical video pixels** | **DROPPED** | No visual model; 2D tracking already gives calibrated pitch coordinates. |
| **Homography matrices** | **DROPPED** | Computation is done in pitch coordinates (105 m × 68 m). |

### 2.3 How the settings were chosen

- **Decoder switch costs (50 nats `H`↔`A`, 75 nats for `D`):** chosen by a grid search over held-out data, the last 15% of Sample Games 1 and 2, using the v2 model (the sweep script is not part of the repository). Making `D` expensive to enter hurt, because it merges stoppages away.
- **Everything else** (features, training recipe, ensemble size) was judged on the validation loss of the held-out tails and on Sample Game 3.
- **Use of the public window.** Diagnosing an earlier model on it showed that it labelled 97% of untracked-ball frames `D` although 39% of those frames were live play; that is what motivated removing the missing-ball shortcut (Section 1.2), so the window shaped the design idea. Every setting of the final pipeline was then chosen on external data as described above. The scoreboard was run after the v2 model and after the final pipeline, and no setting was changed in response to any of its results.
- **Inherited constants.** The Smart Tagging window lengths (corner 5 s, free kick 5.5 s, goal kick 4.5 s, throw-in 3 s, kick-off 4-5 s, each starting 2 s before the tag's start; goals dead from the goal to the kick-off) are `SET_PIECE_DEAD_SECONDS` in `ext_transformer_detector.py`, and the v2 smoothing constants (90 and 50 frames) are its defaults; both were inherited unchanged from the team's first pipeline. That pipeline's 90-frame smoothing window was selected by comparing several window sizes (60 to 125 frames) on the public 5-minute window. Only the comparison model `Senators_v2` uses this smoothing; the final pipeline replaces it with the Viterbi decoder, whose costs were chosen on external data (first bullet of this section).

### 2.4 Possession rules: loose balls, challenges, restarts

1. **Loose balls and passes in flight.** There is no explicit pass-flight or control rule. The Transformer sees 4 s of distance, speed and ball features, and the decoder's switch cost gives temporal inertia: possession stays with the passing team through a flight and a short tracking loss unless evidence for the other team persists.
2. **Challenges and 50/50 duels.** A brief change of team inside a possession is priced by the 50-nat `H`↔`A` cost, so a short rival touch is absorbed; a sustained one becomes a turnover. The training labels follow the `CHALLENGE.md` rule that a duel without control is not a change of possession.
3. **Restarts and stoppages.** Dead ball is `D` and ends a possession (`H D H` is two possessions). Inside Smart Tagging set-piece and goal windows only `D` is allowed; elsewhere stoppages come from the model, which uses ball position near the pitch lines, a ball standing still and slow players.

---

## 3. Limitations

1. **Persistent untracked ball (39.4% of the match after interpolation).** The final output labels 79.4% of untracked-ball frames `D` and 40.6% of graded frames `D`. Some of these are real stoppages with the ball off camera, but the true share is unknown without labels.
2. **Under-segmentation and weak dead-ball recall on the public window.** 12 predicted possessions against 17 true, mean 22.4 s against 14.2 s, 7 of 17 true possessions missed, dead-ball precision 0.95 but recall 0.52 (436 true dead frames labelled `H`). Possessions on both sides of a missed stoppage merge. This is a 17-possession sample and no setting was tuned on it, but the pattern suggests the decoder's `D` cost may be too high for BAR-ATH.
3. **The held-out test is indirect.** Sample Game 3's ground truth is built from event logs, with a different possession length distribution (mean 11.8 s) from the competition's (14.2 s on the public window). Its tracking is made BAR-ATH-like by simulation, and it has no Smart Tagging, so the dead-ball windows are untested on labels. Game 3 was used to judge several design choices (including mixing the two training regimes) after seeing results on it, so its numbers are slightly optimistic.
4. **Limited training data.** Two matches. Validation loss rises after a few epochs and the three seeds stop at different epochs. The domain shift to the Camp Nou recording is not modelled.
5. **Event-derived training labels.** Frame labels come from event logs rather than frame-by-frame annotation, so boundaries near turnovers and stoppages are noisy.
6. **Smart Tagging is automatic.** Tags can be missed or false, and the window lengths are inherited constants.
7. **Planar 2D tracking** has no ball height, so aerial balls look like horizontal movement.
8. **Reproducibility scope.** The output was regenerated byte for byte on the Apple GPU and on the CPU of the machine that produced it. Other hardware or library versions are untested; a one-frame difference at a switch point is possible in principle.

---

## 4. Reproduction Guide

### Step 1: Environment
```bash
pip install -r requirements-submission.txt
```
`torch==2.15.0`, `numpy==2.3.5`, `pandas==2.3.3`, Python 3.13.9. These exact versions were used to regenerate `Senators.csv` byte for byte on 2026-11-15; other Python or library versions are untested. Run every command below with the interpreter these are installed in. The match data goes in `./data` (or `MTAH_DATA_DIR`).

### Step 2: Regenerate the submission (no retraining)
```bash
python3 possession_v2.py --predict
```
Loads the committed weights, averages the three models, decodes, and writes `evaluation/results/Senators.csv`. About 100 s on an Apple GPU (MPS) and 150 s on the CPU (`MTAH_DEVICE=cpu` forces it); the output is identical.

### Step 3: Validate
```bash
python3 evaluation/validate_submission.py evaluation/results/Senators.csv
```
Expected: 199,319 rows, `H 57907 (29.1%)`, `A 36439 (18.3%)`, `D 64511 (32.4%)`, `X 40462 (20.3%)`, 228 possessions of mean 16.6 s.

### Step 4: Score on the public window
```bash
python3 possession_v2.py --predict --tags v2 --decoder smooth --out evaluation/results/Senators_v2.csv
python3 evaluation/run_evaluation.py
```
Reproduces the first table above (the first command writes the comparison model's file).

### Step 5: (Optional) Retrain
Needs the Metrica sample data (`git clone https://github.com/metrica-sports/sample-data external_data/metrica`) and the BAR-ATH tracking (for the ball-gap lengths):
```bash
for s in 0 1 2; do python3 possession_v2.py --train --features v3 --seed $s --tag v3_s$s; done
```
Each run takes 5-12 minutes on Apple silicon and overwrites that member's weights. Training on GPU involves parallel reductions, so a retrained model can differ numerically from the committed one; inference from the committed weights is the reproducible path.

### Step 6: (Optional) Held-out check
```bash
python3 metrica_testbed.py --seeds 3
```
Needs `external_data/metrica`; takes 5-10 minutes. It reproduces the Game 3 table above. Ground truth is Metrica's event logs turned into frame labels by `metrica_data.pseudo_ground_truth` (324 possessions of mean 11.8 s, dead-ball share 32.8%, event frames shifted by one because they are 1-based); Game 3's 22 tracking slots are reassigned after substitutions, so the loader reads the per-segment slot order from the metadata. Scoring uses `evaluation/possession_metrics.py`, the same code as the official scorer.
