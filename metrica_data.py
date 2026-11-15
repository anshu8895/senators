"""Metrica's open sample games as (tracking, pseudo ground truth), for training and testing.

Games 1 and 2 are Metrica CSV exports; Game 3 is FIFA/EPTS (the same format as the competition
ATD). Ground truth here is *pseudo*: Metrica's human event logs turned into frame labels by
ext_transformer_detector.generate_possession_labels_from_events(). It never touches BAR-ATH.
"""

import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

import ext_transformer_detector as det

HERE = os.path.dirname(os.path.abspath(__file__))
METRICA = os.path.join(HERE, "external_data", "metrica", "data")

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
            "Subtype": (e.get("subtypes") or {}).get("name") if isinstance(e.get("subtypes"), dict) else None,
            "Period": e["period"],
            "Start Frame": start,
            "End Frame": start if end is None else end,
        })
    return n, bx, by, home, away, pd.DataFrame(rows)


def load_game12(game):
    """Sample Games 1 and 2 (Metrica CSV format). Training games for the shipped models."""
    gdir = os.path.join(METRICA, "Sample_Game_{}".format(game))
    n, bx, by, home, away = det.load_metrica_tracking(gdir)
    events = pd.read_csv(os.path.join(gdir, "Sample_Game_{}_RawEventsData.csv".format(game)))
    return n, bx, by, home, away, events[["Team", "Type", "Subtype", "Period", "Start Frame", "End Frame"]]


def load_game(game):
    return load_game3() if game == 3 else load_game12(game)


def pseudo_ground_truth(events, n):
    """Frame labels as an object array of H/A/D/X; X outside each period's event span.

    Event frames are 1-based and tracking rows are 0-based, so every frame is shifted by one. Only
    det.POSSESSION_EVENTS change possession (CHALLENGE.md: a duel without control does not).
    """
    events = events.copy()
    for col in ("Start Frame", "End Frame"):
        events[col] = events[col] - 1
    codes = det.generate_possession_labels_from_events(events, n)
    labels = np.array([det.INV_LABEL[c] for c in codes], dtype=object)

    playing = np.zeros(n, dtype=bool)
    for _, ev in events.groupby("Period"):
        lo = int(ev["Start Frame"].min())
        hi = int(max(ev["End Frame"].max(), ev["Start Frame"].max()))
        playing[max(lo, 0):min(hi, n - 1) + 1] = True
    labels[~playing] = "X"
    return labels
