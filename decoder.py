"""Viterbi decoding of per-frame H/A/D probabilities into possessions.

The simpler alternative (ext_transformer_detector.post_process, used by the v2 model) smooths the
probabilities over +-90 frames and then deletes runs shorter than 50 frames. That blurs boundaries
and does not know what a possession is. Here the labels are the single best
path through the three states, where every switch costs something, so a flicker is only kept if the
evidence outweighs its price. A switch between H and A (a turnover) and a switch into or out of D
(a stoppage) are priced separately.
"""

import numpy as np

H, A, D = 0, 1, 2
NAMES = np.array(["H", "A", "D"], dtype=object)


def viterbi(logp, cost):
    """Best state path. logp is (n, 3) log evidence; cost[i][j] is the price of switching i -> j."""
    n = len(logp)
    lp = logp.tolist()
    c = np.asarray(cost, dtype=float).tolist()
    prev = lp[0]
    back = [(0, 0, 0)] * n
    for t in range(1, n):
        cur = lp[t]
        ns = [0.0, 0.0, 0.0]
        nb = [0, 0, 0]
        for j in range(3):
            best, arg = prev[0] - c[0][j], 0
            v = prev[1] - c[1][j]
            if v > best:
                best, arg = v, 1
            v = prev[2] - c[2][j]
            if v > best:
                best, arg = v, 2
            ns[j] = best + cur[j]
            nb[j] = arg
        prev = ns
        back[t] = tuple(nb)
    back = np.array(back, dtype=np.int8)
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(prev))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(probs, segments, lam_ha=150.0, lam_d=75.0, floor=1e-2, forced_dead=None,
           log_prior=None):
    """H/A/D/X labels for every frame; X outside the (first, last) frame segments.

    Each segment (a half) is decoded on its own. forced_dead is an optional boolean array of
    frames that must be D (for example the Smart Tagging set-piece windows); log_prior is an
    optional (n, 3) array added to the log evidence.
    """
    cost = np.array([[0, lam_ha, lam_d], [lam_ha, 0, lam_d], [lam_d, lam_d, 0]], dtype=float)
    logp = np.log(np.clip(probs, floor, 1.0))
    if log_prior is not None:
        logp = logp + log_prior
    if forced_dead is not None:
        logp = logp.copy()
        logp[forced_dead, H] = logp[forced_dead, A] = -1e4
    labels = np.full(len(probs), "X", dtype=object)
    for lo, hi in segments:
        labels[lo:hi + 1] = NAMES[viterbi(logp[lo:hi + 1], cost)]
    return labels


def segments_of(mask):
    """(first, last) frame of each run of True in a boolean array."""
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [len(mask)]])
    return [(int(a), int(b) - 1) for a, b in zip(bounds[:-1], bounds[1:]) if mask[a]]
