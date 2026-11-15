"""Make open-data tracking as incomplete as the competition match's, for training and testing.

The Metrica sample games lose the ball almost only while it is dead, so a model trained on them
learns "ball missing means dead ball". In BAR-ATH the ball is also missing during live play and is
often tracked while dead, so that shortcut is wrong there. These helpers remove it:

1. impute_dead_ball() holds the last seen ball position through dead-ball frames, so the ball is
   visible while dead;
2. corrupt_tracking() then drops the ball in runs drawn from BAR-ATH's own ball-presence run
   lengths, independently of what is happening in play, and drops players at a rate matching the
   ~20.5 of 22 tracked that DATA.md reports.

Only whether the ball field is NaN is read from BAR-ATH, never a label. The run lengths are cached
sorted, so the cache cannot be read back as the match's visibility timeline.
"""

import os
import numpy as np

from games import GAME, half_video_blocks, require, source_frames

HERE = os.path.dirname(os.path.abspath(__file__))
GAP_CACHE = os.path.join(HERE, "external_data", "bar_ath_ball_runs.npz")

# Independent player dropout: mean frames tracked / mean frames lost. 400 / (400 + 30) = 93%, the
# 20.5-of-22 tracked rate DATA.md reports for graded frames.
PLAYER_PRESENT_MEAN = 400.0
PLAYER_MISSING_MEAN = 30.0


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
    """Sorted run lengths of ball-present and ball-missing stretches in BAR-ATH's graded halves."""
    if os.path.exists(GAP_CACHE):
        z = np.load(GAP_CACHE)
        return np.sort(z["present"]), np.sort(z["missing"])
    (raw_path,) = require(["atd_raw"])
    n = source_frames("tactical", GAME)
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
    pres_runs = np.sort(np.concatenate(pres_runs))
    miss_runs = np.sort(np.concatenate(miss_runs))
    os.makedirs(os.path.dirname(GAP_CACHE), exist_ok=True)
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


def impute_dead_ball(bx, by, dead):
    """Copy of the ball track with the last seen position held through missing dead-ball frames."""
    bx, by = bx.copy(), by.copy()
    last = (np.nan, np.nan)
    for i in range(len(bx)):
        if not np.isnan(bx[i]):
            last = (bx[i], by[i])
        elif dead[i]:
            bx[i], by[i] = last
    return bx, by


def corrupt_tracking(bx, by, home, away, rng, graded=None):
    """Copy of the tracking with BAR-ATH-like ball gaps and independent player dropouts.

    The tracking's own missing ball is kept, and only the shortfall to BAR-ATH's missing rate is
    added, measured over the graded frames (all frames if graded is None).
    """
    n = len(bx)
    pres, miss = bar_ath_ball_runs()
    target = miss.sum() / float(pres.sum() + miss.sum())
    own = np.isnan(bx if graded is None else bx[graded]).mean()
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
