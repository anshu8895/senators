"""Senators possession model (final pipeline): v3 features, a 3-model ensemble, Viterbi decoding.

A model trained naively on the Metrica games would learn from data where the ball is missing almost
only while it is dead, and read a missing ball as "dead". BAR-ATH loses the ball during live play
and tracks it during many stoppages. This pipeline therefore differs in these ways:

- features: 20 base ball/player features (ext_transformer_detector), plus team median player speed
  (home, away), a ball-visible flag, frames since the ball was last seen (feature set v2), plus
  four ball cues for a stoppage: outside the pitch lines, distance to the nearest line, movement
  over +-0.5 s, and whether that could be measured (feature set v3);
- training: Sample Games 1 and 2 only, with fresh tracking_corruption every few epochs in two
  regimes (ball held through dead frames and dropped independently of play, or the open data's own
  missing ball plus extra gaps), so the model cannot lean on missingness as a label;
- labels: CHALLENGE.md's rule, only PASS, RECOVERY and SHOT change possession;
- output: three models (seeds 0-2) averaged, then decoder.py's Viterbi decoding with the Smart
  Tagging set-piece and goal windows as dead-ball constraints.

No frame of BAR-ATH and no BAR-ATH label is used for training. Its ball-presence run
lengths (a label-free statistic) set how the ball is dropped in the training copies. Hand labels
come only from other games.

    python3 possession_v2.py --predict      # writes evaluation/results/Senators.csv from the
                                            # committed weights (ext_v3_s{0,1,2}_weights.pt)
    python3 possession_v2.py --train --features v3 --seed 0 --tag v3_s0     # retrain one member

MTAH_DEVICE=cpu forces the CPU (default: the Apple GPU if present).
"""

import os
import sys
import time
import argparse
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext_transformer_detector as det  # noqa: E402
import decoder as dec  # noqa: E402
import metrica_data as md  # noqa: E402
import tracking_corruption as tc  # noqa: E402
from games import GAME, require, source_frames  # noqa: E402

# Viterbi switch costs (nats): H<->A and into/out of D. Chosen on the held-out tails of Sample Games
# 1 and 2 (best of a 4x4 grid), not on Game 3 or BAR-ATH.
LAM_HA, LAM_D = 50.0, 75.0
FINAL_TAGS = ("v3_s0", "v3_s1", "v3_s2")
WINDOW = det.WINDOW_SIZE
HALF = WINDOW // 2
WEIGHTS = os.path.join(HERE, "ext_v2_weights.pt")
NORM = os.path.join(HERE, "norm_stats_v2.npz")


# -- features ------------------------------------------------------------------------------------

def median_speed(px, py, d=6, cap=12.0, smooth=25):
    """Per-frame median speed (m/s) over the tracked players of one team.

    Finite difference over 2*d frames, speeds above cap (identity switches) dropped, median over
    players so a few bad slots cannot move it, then a centred moving average of 2*smooth+1 frames.
    """
    n = len(px)
    v = np.full(px.shape, np.nan, dtype=np.float32)
    v[d:n - d] = np.hypot(px[2 * d:] - px[:n - 2 * d], py[2 * d:] - py[:n - 2 * d]) / (2 * d / 25.0)
    v[v > cap] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(v, axis=1)
    med = np.where(np.isnan(med), np.nanmedian(med), med)
    c = np.concatenate([[0.0], np.cumsum(med)])
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - smooth), np.minimum(n, idx + smooth + 1)
    return ((c[hi] - c[lo]) / (hi - lo)).astype(np.float32)


def extract_features(n, bx, by, home, away):
    """The 20 base features plus 4 that survive a missing ball. Inputs are 0-1 pitch coordinates:
    bx, by arrays with NaN where the ball is not tracked, home and away lists of (x, y) arrays."""
    base = det.extract_features_metrica_clean(n, bx, by, home, away)
    L, W = det.PITCH_LENGTH, det.PITCH_WIDTH
    speeds = [median_speed(np.column_stack([x * L for x, _ in team]),
                           np.column_stack([y * W for _, y in team])) for team in (home, away)]
    seen = ~np.isnan(bx)
    idx = np.arange(n)
    last_seen = np.maximum.accumulate(np.where(seen, idx, -10 ** 6))
    since = np.clip(idx - last_seen, 0, 100) / 100.0
    return np.column_stack([base, speeds[0], speeds[1], seen, since]).astype(np.float32)


def ball_features(n, bx, by):
    """Four ball cues for a stoppage that need no player: ball outside the pitch lines, signed
    distance to the nearest line (negative outside), how far it moved over +-0.5 s, and whether
    that movement could be measured (both ends tracked)."""
    L, W = det.PITCH_LENGTH, det.PITCH_WIDTH
    seen = ~np.isnan(bx)
    x, y = bx * L, by * W
    outside = seen & ((bx < 0) | (bx > 1) | (by < 0) | (by > 1))
    to_line = np.where(seen, np.clip(np.minimum.reduce([x, L - x, y, W - y]), -5.0, 20.0), 10.0)
    k = 12
    moved = np.zeros(n)
    measurable = np.zeros(n)
    d = np.hypot(x[2 * k:] - x[:n - 2 * k], y[2 * k:] - y[:n - 2 * k])
    moved[k:n - k] = np.clip(np.nan_to_num(d), 0.0, 15.0)
    measurable[k:n - k] = ~np.isnan(d)
    return np.column_stack([outside, to_line, moved, measurable]).astype(np.float32)


def extract_features_v3(n, bx, by, home, away):
    """The 24 v2 features plus ball_features(): 28."""
    return np.column_stack([extract_features(n, bx, by, home, away),
                            ball_features(n, bx, by)]).astype(np.float32)


# name -> (extractor, number of features)
FEATURE_SETS = {"v2": (extract_features, 24), "v3": (extract_features_v3, 28)}


def load_competition(raw_path, team_of_track, n):
    """BAR-ATH ATD as the same (bx, by, home, away) inputs extract_features takes."""
    n_slots = len(team_of_track)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    with open(raw_path) as f:
        for line in f:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1
            if not 0 <= fid < n:
                continue
            b = parts[2]
            if b[0] != "N":
                c1 = b.find(",")
                c2 = b.find(",", c1 + 1)
                bx[fid], by[fid] = float(b[:c1]), float(b[c1 + 1:c2])
            for s, tok in enumerate(parts[1].split(";")):
                if tok[0] != "N":
                    c1 = tok.find(",")
                    c2 = tok.find(",", c1 + 1)
                    px[fid, s], py[fid, s] = float(tok[:c1]), float(tok[c1 + 1:c2])
    home = [(px[:, s], py[:, s]) for s in range(n_slots) if team_of_track[s] == 0]
    away = [(px[:, s], py[:, s]) for s in range(n_slots) if team_of_track[s] == 1]
    return bx, by, home, away


# -- training data -------------------------------------------------------------------------------

# How the ball goes missing in a training copy. BAR-ATH sits somewhere between these two and the
# open data cannot say where, so every epoch trains on both.
#   indep:        the ball is held through dead frames, then dropped independently of play
#   as-delivered: the open data's own missing ball (mostly while dead) plus the same extra gaps
REGIMES = ("indep", "as-delivered")


def _frames(arr, lo, hi):
    return [(x[lo:hi], y[lo:hi]) for x, y in arr]


def load_training_games(val_share=0.15):
    """Sample Games 1 and 2 with labels; the first 85% of each trains, the last 15% validates."""
    games = []
    for game in (1, 2):
        n, bx, by, home, away, events = md.load_game(game)
        labels = md.pseudo_ground_truth(events, n)
        labels[labels == "X"] = "D"  # before kickoff, at half time and after the whistle: not in play
        y = np.array([det.LABEL_MAP[v] for v in labels], dtype=np.int64)
        games.append(dict(n=n, bx=bx, by=by, home=home, away=away, y=y,
                          cut=int(n * (1 - val_share))))
    return games


def make_copy(g, lo, hi, regime, rng, extract=extract_features):
    """(features, labels) for frames lo..hi of a game under one missing-ball regime."""
    bx, by = g["bx"][lo:hi], g["by"][lo:hi]
    if regime == "indep":
        bx, by = tc.impute_dead_ball(bx, by, g["y"][lo:hi] == 2)
    cbx, cby, ch, ca = tc.corrupt_tracking(
        bx, by, _frames(g["home"], lo, hi), _frames(g["away"], lo, hi), rng)
    return extract(hi - lo, cbx, cby, ch, ca), g["y"][lo:hi]


# -- model ---------------------------------------------------------------------------------------

def _windows(padded, rows):
    return torch.from_numpy(np.stack([padded[s][i:i + WINDOW] for s, i in rows]))


def _row_table(ys, step=1):
    return np.concatenate([np.column_stack([np.full(len(y), s), np.arange(len(y))])[::step]
                           for s, y in enumerate(ys)])


def train(games, epochs=30, per_epoch=60_000, batch=256, lr=3e-4, patience=6, seed=0,
          dropout=0.3, weight_decay=1e-2, smoothing=0.1, noise=0.1, refresh=3, features="v2"):
    """Fresh corrupted copies every `refresh` epochs, so the model cannot memorise one pattern of
    gaps. Two matches are little data for a transformer: validation loss starts rising after the
    first pass, so it is regularised hard (dropout, weight decay, label smoothing, input noise)."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("  device:", device, " features:", features, " seed:", seed, flush=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    extract, n_features = FEATURE_SETS[features]

    # Validation: both regimes on the held-out tail of each game, fixed for the whole run.
    val_sets = [make_copy(g, g["cut"], g["n"], r, rng, extract) for g in games for r in REGIMES]
    # Normalisation and class weights from one copy of each regime.
    init = [make_copy(g, 0, g["cut"], r, rng, extract) for g in games for r in REGIMES]
    all_feats = np.concatenate([f for f, _ in init])
    mean, std = all_feats.mean(axis=0), all_feats.std(axis=0) + 1e-8
    counts = np.bincount(np.concatenate([y for _, y in init]), minlength=3).astype(np.float32)
    weights = (counts.sum() / (3.0 * counts)).clip(0.5, 10.0)
    print("  class weights: H={:.2f} A={:.2f} D={:.2f}".format(*weights), flush=True)

    def prep(sets):
        return ([np.pad((f - mean) / std, ((HALF, HALF), (0, 0))).astype(np.float32)
                 for f, _ in sets], [y for _, y in sets])

    va_x, va_y = prep(val_sets)
    va_rows = _row_table(va_y, step=4)

    model = det.PossessionTransformer(n_features=n_features, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device),
                                    label_smoothing=smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def labels_of(ys, rows):
        return torch.tensor([ys[s][i] for s, i in rows], dtype=torch.long)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        # Both regimes of both games together: drawing one per epoch made the weights chase
        # whichever regime they saw last (validation loss swung between 0.77 and 1.28).
        if (ep - 1) % refresh == 0:
            tr_x, tr_y = prep([make_copy(g, 0, g["cut"], r, rng, extract)
                                for g in games for r in REGIMES])
            tr_rows = _row_table(tr_y)
        model.train()
        run_loss, run_n = 0.0, 0
        pick = rng.choice(len(tr_rows), size=min(per_epoch, len(tr_rows)), replace=False)
        for b in range(0, len(pick), batch):
            rows = tr_rows[pick[b:b + batch]]
            X, y = _windows(tr_x, rows).to(device), labels_of(tr_y, rows).to(device)
            X = X + noise * torch.randn_like(X)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            run_loss, run_n = run_loss + loss.item() * len(y), run_n + len(y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        loss, total = 0.0, 0
        with torch.no_grad():
            for b in range(0, len(va_rows), 1024):
                rows = va_rows[b:b + 1024]
                X, y = _windows(va_x, rows).to(device), labels_of(va_y, rows).to(device)
                loss += criterion(model(X), y).item() * len(y)
                total += len(y)
        loss /= total
        print("  epoch {:2d}/{}  train_loss {:.4f}  val_loss {:.4f}  ({:.0f}s)".format(
            ep, epochs, run_loss / run_n, loss, time.time() - t0), flush=True)
        if loss < best:
            best, bad = loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("  early stop at epoch", ep)
                break
    model.load_state_dict(best_state)
    return model.cpu(), mean, std


_LOADED = {}


def _paths(tag):
    """Weight and normalisation files of a model. The tag "v2" is the first v2 model's files."""
    if tag == "v2":
        return WEIGHTS, NORM
    return (os.path.join(HERE, "ext_{}_weights.pt".format(tag)),
            os.path.join(HERE, "norm_stats_{}.npz".format(tag)))


def load_model(tag="v2"):
    if tag not in _LOADED:
        device = torch.device(os.environ.get("MTAH_DEVICE")
                              or ("mps" if torch.backends.mps.is_available() else "cpu"))
        weights, norm_path = _paths(tag)
        norm = np.load(norm_path)
        family = str(norm["features"]) if "features" in norm.files else "v2"
        model = det.PossessionTransformer(n_features=FEATURE_SETS[family][1]).to(device)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.eval()
        _LOADED[tag] = dict(model=model, mean=norm["mean"], std=norm["std"], device=device,
                            family=family)
    return _LOADED[tag]


def predict_probs(feats, tag="v2", batch=1024):
    """Softmax probabilities [P(H), P(A), P(D)] for every frame, from its centred 4 s window."""
    m = load_model(tag)
    n = len(feats)
    padded = np.pad((feats - m["mean"]) / m["std"], ((HALF, HALF), (0, 0))).astype(np.float32)
    windows = sliding_window_view(padded, (WINDOW, feats.shape[1]))[:n, 0]
    probs = np.zeros((n, 3), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, n, batch):
            X = torch.from_numpy(np.ascontiguousarray(windows[b:b + batch])).to(m["device"])
            probs[b:b + batch] = torch.softmax(m["model"](X), dim=-1).cpu().numpy()
    return probs


def predict_tracking(n, bx, by, home, away, tags=("v2",)):
    """Probabilities averaged over the models in tags, which must share one feature set."""
    families = {load_model(t)["family"] for t in tags}
    if len(families) != 1:
        raise ValueError("models in one ensemble must share a feature set, got {}".format(families))
    feats = FEATURE_SETS[families.pop()][0](n, bx, by, home, away)
    return np.mean([predict_probs(feats, t) for t in tags], axis=0)


def predict_competition(out_path, tags=("v2",), decoder="viterbi", lam_ha=LAM_HA, lam_d=LAM_D,
                        window=90, min_spell=50):
    """BAR-ATH possession labels from committed weights. decoder="smooth" is the ±90-frame
    smoothing + minimum run (ext_transformer_detector.post_process, used by the v2 model); "viterbi" decodes with decoder.py and takes the Smart Tagging
    set-piece windows as dead-ball constraints."""
    meta_path, raw_path, smart_path = require(["atd_meta", "atd_raw", "smart_tagging"])
    n = source_frames("tactical", GAME)
    root = ET.parse(meta_path).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    team_of_track = np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                             dtype=np.int8)
    bx, by, home, away = load_competition(raw_path, team_of_track, n)
    probs = predict_tracking(n, bx, by, home, away, tags)
    if decoder == "smooth":
        labels = det.post_process(probs, n, smart_tagging_path=smart_path, w=window,
                                  min_spell=min_spell)
    else:
        labels = dec.decode(probs, [(det.HALF1_START, det.HALF1_END),
                                    (det.HALF2_KICKOFF, det.HALF2_END)], lam_ha, lam_d,
                            forced_dead=det.smart_tagging_dead_mask(smart_path, n))
        labels[det.HALF2_START:det.HALF2_KICKOFF] = "D"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(frame, lbl) for frame, lbl in enumerate(labels))
    for lbl in ("H", "A", "D", "X"):
        print("   {}: {:>7,} ({:5.1f}%)".format(lbl, int((labels == lbl).sum()),
                                              100.0 * (labels == lbl).mean()))
    print("written:", out_path)


def main():
    parser = argparse.ArgumentParser(description="Possession model v2/v3")
    parser.add_argument("--train", action="store_true", help="train one model and write its weights")
    parser.add_argument("--features", default="v2", choices=sorted(FEATURE_SETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", help="name of the model to write, e.g. v3_s0 (required to train)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--predict", action="store_true", help="write BAR-ATH labels (needs data)")
    parser.add_argument("--tags", nargs="+", default=list(FINAL_TAGS),
                        help="models to average when predicting (default: the final ensemble)")
    parser.add_argument("--decoder", default="viterbi", choices=("viterbi", "smooth"))
    parser.add_argument("--out", default=os.path.join(HERE, "evaluation", "results", "Senators.csv"))
    args = parser.parse_args()
    if args.predict:
        predict_competition(args.out, tuple(args.tags), args.decoder)
        return
    if not args.train:
        parser.error("nothing to do: pass --train or --predict")
    if not args.tag or args.tag == "v2":
        parser.error("--tag is required to train and may not be v2 (that is the committed v2 model)")
    t0 = time.time()
    print("loading Metrica Sample Games 1 and 2 ...")
    games = load_training_games()
    print("training {} ...".format(args.tag))
    model, mean, std = train(games, epochs=args.epochs, seed=args.seed, features=args.features)
    weights, norm_path = _paths(args.tag)
    torch.save(model.state_dict(), weights)
    np.savez(norm_path, mean=mean, std=std, features=np.array(args.features))
    n_params = sum(p.numel() for p in model.parameters())
    print("saved {} ({:,} parameters) and {}  [{:.0f}s]".format(
        os.path.basename(weights), n_params, os.path.basename(norm_path), time.time() - t0))


if __name__ == "__main__":
    main()
