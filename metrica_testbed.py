"""Held-out testbed for the possession pipeline, on Metrica's open sample games.

Why this exists: the public 5-minute ground truth is 17 possessions, so one possession moves
spellF1 by ~0.06 and tuning on it picks the wrong answer (EVALUATION.md). RULES.md also says it is
not a tuning set. This scores pipeline changes on an independent game instead, with the official
metric code (evaluation/possession_metrics.py).

What it measures, and what it does not:
- Sample Game 3 was never used for training (the model trains on Games 1 and 2). Games 1 and 2 can
  be scored with --game, but those numbers are in-sample.
- Ground truth is *pseudo*: Metrica's human event logs turned into frame labels by the same
  generate_possession_labels_from_events() that makes the training labels. It is a relative
  yardstick for comparing versions, not a prediction of the leaderboard score.
- Sample Game 3 has no Smart Tagging, so anything that depends on it cannot be tested here.
- The "bar-ath" condition makes the tracking as messy as the competition match: the ball goes
  missing in runs drawn from BAR-ATH's own ball-presence run lengths (label-free: it never looks at
  ground truth) and players drop out of the tracking at a rate matching ~20.5 of 22 tracked. The
  dropouts are independent of play, whereas in BAR-ATH they correlate with dead balls, so this is
  the harsher test of a "missing ball means dead ball" shortcut.

Usage (torch needs the anaconda interpreter):
    /opt/anaconda3/bin/python3 metrica_testbed.py                    # game 3, raw + BAR-ATH-like corruption
    /opt/anaconda3/bin/python3 metrica_testbed.py --seeds 5 --window 60

Needs external_data/metrica (git clone https://github.com/metrica-sports/sample-data
external_data/metrica) and, for the corrupted condition, the BAR-ATH tracking file.

To test another pipeline, add a function to PIPELINES with the signature of run_current().
"""

import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "evaluation"))
import possession_metrics as pm  # noqa: E402
import ext_transformer_detector as det  # noqa: E402
from games import GAME, half_video_blocks, require  # noqa: E402

METRICA = os.path.join(HERE, "external_data", "metrica", "data")
GAP_CACHE = os.path.join(HERE, "external_data", "bar_ath_ball_runs.npz")

# Independent player dropout: mean frames tracked / mean frames lost. 400 / (400 + 30) = 93%, the
# 20.5-of-22 tracked rate DATA.md reports for graded frames.
PLAYER_PRESENT_MEAN = 400.0
PLAYER_MISSING_MEAN = 30.0


# -- loading ------------------------------------------------------------------------------------

def _slot_teams(root, home_id):
    """[(first_row, last_row, slot_is_home)] for each tracking segment.

    The 22 tracking slots are re-assigned to different players after substitutions; the metadata
    lists the slot order once per frame segment (DataFormatSpecification).
    """
    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    player_of_channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    segments = []
    for spec in root.iter("DataFormatSpecification"):
        channels = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")][0::2]
        is_home = np.array([team_of_player[player_of_channel[c]] == home_id for c in channels])
        segments.append((int(spec.get("startFrame")) - 1, int(spec.get("endFrame")) - 1, is_home))
    return segments


def load_game3():
    """Sample Game 3 (FIFA/EPTS tracking + JSON events). Team A, the first team listed, is home."""
    gdir = os.path.join(METRICA, "Sample_Game_3")
    root = ET.parse(os.path.join(gdir, "Sample_Game_3_metadata.xml")).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    segments = _slot_teams(root, team_ids[0])
    n_slots = len(segments[0][2])

    with open(os.path.join(gdir, "Sample_Game_3_tracking.txt")) as f:
        lines = f.read().splitlines()
    n = len(lines)
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    for i, line in enumerate(lines):
        _, players, ball = line.split(":")[:3]
        tokens = players.split(";")
        if len(tokens) != n_slots:
            raise ValueError("line {}: {} players, expected {}".format(i + 1, len(tokens), n_slots))
        for s, tok in enumerate(tokens):
            a, b = tok.split(",")[:2]
            px[i, s], py[i, s] = float(a), float(b)
        a, b = ball.split(",")[:2]
        bx[i], by[i] = float(a), float(b)

    # Regroup the slots by team, segment by segment, into 11 home and 11 away columns.
    hx, hy, ax, ay = (np.full((n, 11), np.nan, dtype=np.float32) for _ in range(4))
    for first, last, is_home in segments:
        rows = slice(first, min(last, n - 1) + 1)
        for dst_x, dst_y, pick in ((hx, hy, is_home), (ax, ay, ~is_home)):
            cols = np.flatnonzero(pick)[:11]
            dst_x[rows, :len(cols)] = px[rows][:, cols]
            dst_y[rows, :len(cols)] = py[rows][:, cols]
    home = [(hx[:, s], hy[:, s]) for s in range(11)]
    away = [(ax[:, s], ay[:, s]) for s in range(11)]

    rows = []
    for e in json.load(open(os.path.join(gdir, "Sample_Game_3_events.json")))["data"]:
        start = e["start"]["frame"]
        end = (e.get("end") or {}).get("frame")
        rows.append({
            "Team": "Home" if e["team"]["id"] == team_ids[0] else "Away",
            "Type": e["type"]["name"],
            "Period": e["period"],
            "Start Frame": start,
            "End Frame": start if end is None else end,
        })
    return n, bx, by, home, away, pd.DataFrame(rows)


def load_game12(game):
    """Sample Games 1 and 2 (Metrica CSV format). In-sample: the model trains on these."""
    gdir = os.path.join(METRICA, "Sample_Game_{}".format(game))
    n, bx, by, home, away = det.load_metrica_tracking(gdir)
    events = pd.read_csv(os.path.join(gdir, "Sample_Game_{}_RawEventsData.csv".format(game)))
    return n, bx, by, home, away, events[["Team", "Type", "Period", "Start Frame", "End Frame"]]


def load_game(game):
    return load_game3() if game == 3 else load_game12(game)


# Only events where a team has the ball under control. CHALLENGE, FAULT RECEIVED and BALL LOST are
# duels and turnovers; CHALLENGE.md says a duel without subsequent control is not a change of
# possession, so counting them (as the training labels do) splits real possessions into pieces.
STRICT_TYPES = {"PASS", "RECOVERY", "SHOT"}


def pseudo_ground_truth(events, n, strict=True):
    """Frame labels as an object array of H/A/D/X; X outside each period's event span.

    Event frames are 1-based and tracking rows are 0-based, so every frame is shifted by one.
    strict=False uses the same event types as the training labels.
    """
    events = events.copy()
    for col in ("Start Frame", "End Frame"):
        events[col] = events[col] - 1
    codes = det.generate_possession_labels_from_events(
        events, n, poss_types=STRICT_TYPES if strict else None)
    labels = np.array([det.INV_LABEL[c] for c in codes], dtype=object)

    playing = np.zeros(n, dtype=bool)
    for _, ev in events.groupby("Period"):
        lo = int(ev["Start Frame"].min())
        hi = int(max(ev["End Frame"].max(), ev["Start Frame"].max()))
        playing[max(lo, 0):min(hi, n - 1) + 1] = True
    labels[~playing] = "X"
    return labels


# -- making the tracking as messy as BAR-ATH -----------------------------------------------------

def _runs(mask):
    """Lengths of the consecutive True runs and False runs of a boolean array."""
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [len(mask)]])
    lengths = np.diff(bounds)
    first = bool(mask[0])
    true_runs = lengths[0::2] if first else lengths[1::2]
    false_runs = lengths[1::2] if first else lengths[0::2]
    return true_runs, false_runs


def bar_ath_ball_runs():
    """Run lengths of ball-present and ball-missing stretches in BAR-ATH's graded halves.

    Reads only whether the ball field is NaN, never a label. Cached under external_data/.
    """
    if os.path.exists(GAP_CACHE):
        z = np.load(GAP_CACHE)
        return z["present"], z["missing"]
    (raw_path,) = require(["atd_raw"])
    n = det.source_frames("tactical", GAME)
    present = np.zeros(n, dtype=bool)
    with open(raw_path) as f:
        for line in f:
            fid = int(line[:line.index(":")]) - 1
            if 0 <= fid < n:
                present[fid] = line[line.rindex(":") + 1] != "N"
    pres_runs, miss_runs = [], []
    for _, first, last in half_video_blocks(GAME):
        p, m = _runs(present[first:last + 1])
        pres_runs.append(p)
        miss_runs.append(m)
    pres_runs, miss_runs = np.concatenate(pres_runs), np.concatenate(miss_runs)
    np.savez(GAP_CACHE, present=pres_runs, missing=miss_runs)
    return pres_runs, miss_runs


def _alternating_mask(n, present_runs, missing_runs, rng, missing_share):
    """True where kept. Alternates run lengths drawn from the two empirical distributions.

    Present runs are stretched so that the long-run missing share is missing_share.
    """
    stretch = (1 - missing_share) / missing_share * missing_runs.sum() / present_runs.sum()
    mask = np.ones(n, dtype=bool)
    state = rng.random() < 1 - missing_share
    i = 0
    while i < n:
        length = max(1, int(rng.choice(present_runs if state else missing_runs)
                            * (stretch if state else 1.0)))
        mask[i:i + length] = state
        i += length
        state = not state
    return mask


def corrupt(bx, by, home, away, rng, graded):
    """Copy of the tracking with BAR-ATH-like ball gaps and independent player dropouts.

    The game's own missing ball is kept, and only the shortfall to BAR-ATH's missing rate is added
    (Sample Game 3 already loses the ball on ~32% of graded frames; BAR-ATH on ~41%).
    """
    n = len(bx)
    pres, miss = bar_ath_ball_runs()
    target = miss.sum() / float(pres.sum() + miss.sum())
    own = np.isnan(bx[graded]).mean()
    extra = max(0.0, 1 - (1 - target) / (1 - own))
    keep = _alternating_mask(n, pres, miss, rng, extra) if extra > 0 else np.ones(n, dtype=bool)
    bx, by = np.where(keep, bx, np.nan), np.where(keep, by, np.nan)

    def drop(players):
        out = []
        for x, y in players:
            ok = np.ones(n, dtype=bool)
            i, state = 0, True
            while i < n:
                length = max(1, int(rng.exponential(
                    PLAYER_PRESENT_MEAN if state else PLAYER_MISSING_MEAN)))
                ok[i:i + length] = state
                i += length
                state = not state
            out.append((np.where(ok, x, np.nan), np.where(ok, y, np.nan)))
        return out

    return bx, by, drop(home), drop(away)


# -- pipelines under test ------------------------------------------------------------------------

_MODEL = {}


def _load_model():
    if not _MODEL:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        norm = np.load(os.path.join(HERE, "norm_stats.npz"))
        model = det.PossessionTransformer().to(device)
        model.load_state_dict(torch.load(os.path.join(HERE, "ext_transformer_weights.pt"),
                                         map_location=device))
        _MODEL.update(model=model, mean=norm["mean"], std=norm["std"], device=device)
    return _MODEL


def run_current(n, bx, by, home, away, window=90, min_spell=50):
    """The submitted pipeline, minus the Smart Tagging windows and BAR-ATH structural zones."""
    m = _load_model()
    feats = det.extract_features_metrica_clean(n, bx, by, home, away)
    probs = det.infer_transformer(m["model"], feats, m["mean"], m["std"], m["device"], n)
    return det.post_process(probs, n, None, w=window, min_spell=min_spell, structural=False)


PIPELINES = {"current": run_current}


# -- scoring -------------------------------------------------------------------------------------

def score(gt, pred):
    pred = pred.copy()
    pred[gt == "X"] = "X"  # ungraded frames: nothing to say, and nothing is scored there
    return pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))


def row(name, r, ball_missing):
    return "{:<22}{:>9.4f}{:>9.4f}{:>9}{:>8.1f}{:>9.4f}{:>7.2f}{:>10.1%}".format(
        name, r["macro_f1"], r["spell_f1_mean"],
        "{}/{}".format(r["n_pred_spells"], r["n_gt_spells"]), r["mean_spell_duration_s"],
        r["per_class"]["D"]["f1"], r["fragmentation"], ball_missing)


def main():
    parser = argparse.ArgumentParser(description="Score a pipeline on a Metrica sample game")
    parser.add_argument("--game", type=int, default=3, choices=(1, 2, 3),
                        help="3 is held out; 1 and 2 are in-sample (default: 3)")
    parser.add_argument("--pipeline", default="current", choices=sorted(PIPELINES))
    parser.add_argument("--conditions", nargs="+", default=["raw", "bar-ath"],
                        choices=("raw", "bar-ath"),
                        help="raw = the game's tracking as delivered (default: raw bar-ath)")
    parser.add_argument("--gt", default="strict", choices=("strict", "events"),
                        help="strict: only PASS/RECOVERY/SHOT change possession (default); "
                             "events: the event types the training labels use")
    parser.add_argument("--seeds", type=int, default=3, help="corruption seeds (default: 3)")
    parser.add_argument("--window", type=int, default=90)
    parser.add_argument("--min-spell", type=int, default=50)
    args = parser.parse_args()
    run = PIPELINES[args.pipeline]

    n, bx, by, home, away, events = load_game(args.game)
    gt = pseudo_ground_truth(events, n, strict=args.gt == "strict")
    graded = gt != "X"
    spells = pm.possession_stats(pm.label_codes(gt), pm.valid_runs(pm.label_codes(gt)))
    print("game {}{}: {:,} frames, {:,} graded; pseudo-GT {} possessions of mean {:.1f} s, "
          "D share {:.1%}".format(
              args.game, "" if args.game == 3 else " (IN-SAMPLE)", n, graded.sum(),
              spells["n_spells"], spells["mean_duration_s"],
              (gt[graded] == "D").mean()))
    print("")

    header = "{:<22}{:>9}{:>9}{:>9}{:>8}{:>9}{:>7}{:>10}".format(
        "condition", "macroF1", "spellF1", "poss", "dur", "D-F1", "frag", "ballmiss")
    print(header)
    print("-" * len(header))
    results = {}
    kw = dict(window=args.window, min_spell=args.min_spell)
    if "raw" in args.conditions:
        results["raw"] = [score(gt, run(n, bx, by, home, away, **kw))]
        print(row("raw", results["raw"][0], np.isnan(bx[graded]).mean()))
    if "bar-ath" in args.conditions:
        results["bar-ath"] = []
        for seed in range(args.seeds):
            cbx, cby, chome, caway = corrupt(bx, by, home, away, np.random.default_rng(seed),
                                             graded)
            r = score(gt, run(n, cbx, cby, chome, caway, **kw))
            results["bar-ath"].append(r)
            print(row("bar-ath seed {}".format(seed), r, np.isnan(cbx[graded]).mean()))
        if args.seeds > 1:
            keys = ("macro_f1", "spell_f1_mean")
            mean = {k: np.mean([r[k] for r in results["bar-ath"]]) for k in keys}
            sd = {k: np.std([r[k] for r in results["bar-ath"]]) for k in keys}
            print("{:<22}{:>9.4f}{:>9.4f}   (sd {:.4f} / {:.4f})".format(
                "bar-ath mean", mean["macro_f1"], mean["spell_f1_mean"],
                sd["macro_f1"], sd["spell_f1_mean"]))
    print("\nposs = predicted/true possessions   dur = mean predicted possession (s)   "
          "frag = fragmentation (1.0 is ideal)")
    print("spellF1 differences below ~0.04 and macroF1 below ~0.01 are noise.")


if __name__ == "__main__":
    main()
