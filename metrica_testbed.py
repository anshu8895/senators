"""Held-out testbed for the possession pipeline, on Metrica's open sample games.

Why this exists: the public 5-minute ground truth is 17 possessions, so one possession moves
spellF1 by ~0.06 and tuning on it picks the wrong answer (EVALUATION.md). RULES.md also says it is
not a tuning set. This scores the pipeline on an independent game instead, with the official
metric code (evaluation/possession_metrics.py).

What it measures, and what it does not:
- Sample Game 3 was never used for training (the models train on Games 1 and 2). Games 1 and 2 can
  be scored with --game, but those numbers are in-sample.
- Ground truth is *pseudo*: Metrica's human event logs turned into frame labels (metrica_data.py).
  It is a relative yardstick for comparing versions, not a prediction of the leaderboard score.
- Sample Game 3 has no Smart Tagging, so the Smart Tagging dead-ball windows are not applied here.

Conditions (what the tracking looks like):
- raw: as delivered. The ball is missing almost only while dead, so "missing means dead" works here
  and does NOT work in BAR-ATH. Do not judge a model on this condition alone.
- indep: the ball is held through dead frames, then dropped in BAR-ATH-like runs and players
  dropped, independently of play (tracking_corruption.py). Missingness says nothing about the label.
- bar-ath: the raw tracking plus the same BAR-ATH-like gaps and dropouts. Missingness is still
  correlated with dead balls, as in the raw data. The truth for BAR-ATH lies between indep and raw.

Pipelines:
- v2:    one v2 model (ext_v2_weights.pt) with the +-90-frame smoothing post-processing
- final: the three v3 models averaged, decoded with decoder.py (the submitted model and decoder)

Usage (torch needs the interpreter the pinned requirements are installed in):
    python3 metrica_testbed.py                     # both pipelines, game 3, all conditions
    python3 metrica_testbed.py --seeds 5 --conditions indep
"""

import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "evaluation"))
import possession_metrics as pm  # noqa: E402
import ext_transformer_detector as det  # noqa: E402
import decoder as dec  # noqa: E402
import metrica_data as md  # noqa: E402
import possession_v2 as v2  # noqa: E402
import tracking_corruption as tc  # noqa: E402

# name -> (models averaged, how probabilities become labels)
PIPELINES = {
    "v2": (("v2",), "smooth"),
    "final": (v2.FINAL_TAGS, "viterbi"),
}


def to_labels(method, probs, gt, args):
    """H/A/D labels (X outside the graded frames) without Smart Tagging or BAR-ATH's zones."""
    if method == "smooth":
        return det.post_process(probs, len(probs), None, w=args.window, min_spell=args.min_spell,
                                structural=False)
    return dec.decode(probs, dec.segments_of(gt != "X"), args.lam_ha, args.lam_d)


def score(gt, pred):
    pred = pred.copy()
    pred[gt == "X"] = "X"  # ungraded frames: nothing is scored there
    return pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))


def condition_inputs(name, seed, bx, by, home, away, gt, graded):
    """The tracking as one condition presents it."""
    if name == "raw":
        return bx, by, home, away
    if name == "indep":
        bx, by = tc.impute_dead_ball(bx, by, (gt == "D") | (gt == "X"))
    return tc.corrupt_tracking(bx, by, home, away, np.random.default_rng(seed), graded)


def main():
    parser = argparse.ArgumentParser(description="Score the pipelines on a Metrica sample game")
    parser.add_argument("--game", type=int, default=3, choices=(1, 2, 3),
                        help="3 is held out; 1 and 2 are training data (default: 3)")
    parser.add_argument("--pipelines", nargs="+", default=["v2", "final"], choices=sorted(PIPELINES))
    parser.add_argument("--conditions", nargs="+", default=["raw", "indep", "bar-ath"],
                        choices=("raw", "indep", "bar-ath"))
    parser.add_argument("--seeds", type=int, default=3, help="corruption seeds (default: 3)")
    parser.add_argument("--lam-ha", type=float, default=v2.LAM_HA, help="Viterbi H<->A switch cost")
    parser.add_argument("--lam-d", type=float, default=v2.LAM_D, help="Viterbi cost into/out of D")
    parser.add_argument("--window", type=int, default=90, help="smoothing radius of the v2 pipeline")
    parser.add_argument("--min-spell", type=int, default=50, help="minimum run of the v2 pipeline")
    args = parser.parse_args()

    n, bx, by, home, away, events = md.load_game(args.game)
    gt = md.pseudo_ground_truth(events, n)
    graded = gt != "X"
    codes = pm.label_codes(gt)
    stats = pm.possession_stats(codes, pm.valid_runs(codes))
    print("game {}{}: {:,} frames, {:,} graded; pseudo-GT {} possessions of mean {:.1f} s, "
          "D share {:.1%}\n".format(
              args.game, "" if args.game == 3 else " (TRAINING DATA)", n, graded.sum(),
              stats["n_spells"], stats["mean_duration_s"], (gt[graded] == "D").mean()))

    results, missing = {}, {}
    for cond in args.conditions:
        for seed in range(1 if cond == "raw" else args.seeds):
            cbx, cby, ch, ca = condition_inputs(cond, seed, bx, by, home, away, gt, graded)
            missing.setdefault(cond, []).append(np.isnan(cbx[graded]).mean())
            cache = {}
            for name in args.pipelines:
                tags, method = PIPELINES[name]
                if tags not in cache:
                    cache[tags] = v2.predict_tracking(n, cbx, cby, ch, ca, tags)
                pred = to_labels(method, cache[tags], gt, args)
                results.setdefault((cond, name), []).append(score(gt, pred))

    header = "{:<9}{:<7}{:>9}{:>9}{:>8}{:>10}{:>8}{:>8}{:>8}{:>10}".format(
        "cond", "pipe", "macroF1", "spellF1", "sd", "poss", "dur", "D-F1", "frag", "ballmiss")
    print(header)
    print("-" * len(header))
    for cond in args.conditions:
        for name in args.pipelines:
            rs = results[(cond, name)]
            m = lambda f: float(np.mean([f(r) for r in rs]))  # noqa: E731
            print("{:<9}{:<7}{:>9.4f}{:>9.4f}{:>8.4f}{:>10}{:>8.1f}{:>8.4f}{:>8.2f}{:>10.1%}".format(
                cond, name, m(lambda r: r["macro_f1"]), m(lambda r: r["spell_f1_mean"]),
                float(np.std([r["spell_f1_mean"] for r in rs])),
                "{:.0f}/{}".format(m(lambda r: r["n_pred_spells"]), rs[0]["n_gt_spells"]),
                m(lambda r: r["mean_spell_duration_s"]), m(lambda r: r["per_class"]["D"]["f1"]),
                m(lambda r: r["fragmentation"]), float(np.mean(missing[cond]))))
    print("\nMeans over {} corruption seeds (raw has one run). poss = predicted/true possessions, "
          "dur = mean predicted possession (s),\nfrag = fragmentation (1.0 is ideal), sd = of "
          "spellF1 across seeds. spellF1 differences below ~0.04 and macroF1 below ~0.01 are noise."
          .format(args.seeds))


if __name__ == "__main__":
    main()
