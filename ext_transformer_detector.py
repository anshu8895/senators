"""Shared helpers for the Senators possession pipeline (possession_v2.py is the entry point).

Holds what both kept models use: the Metrica loaders and event-to-label conversion, the 20 base
ball/player features, the transformer, the Smart Tagging dead-ball windows, the v2 smoothing
post-processing, and the match's frame boundaries (from games.py).

Training data is external only (Metrica Sample Games 1 and 2); nothing here fits, tunes or
calibrates on the competition match (BAR-ATH) or its ground truth.

Author: Senators Team
"""

import os
import sys
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET

# ── Determinism ──────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, anchor_video_frame

# ── Constants ────────────────────────────────────────────────────────────────
FPS = 25
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0
N_FEATURES   = 20
WINDOW_SIZE  = 100   # context window (frames) for transformer (4.0s)
LABEL_MAP    = {"H": 0, "A": 1, "D": 2}
INV_LABEL    = {0: "H", 1: "A", 2: "D"}
POSSESSION_EVENTS = {"PASS", "RECOVERY", "SHOT"}  # events after which the acting team has the ball

# Match structural boundaries dynamically derived from official games.py & DATA.md:
_blocks = half_video_blocks(GAME)
HALF1_START    = _blocks[0][1]      # 11185
HALF1_END      = _blocks[0][2]      # 90651
HALFTIME_START = HALF1_END + 1      # 90652
HALF2_START    = _blocks[1][1]      # 110808
HALF2_KICKOFF  = anchor_video_frame(GAMES[GAME]["kickoff_2nd"]) + 1  # 113742
HALF2_END      = _blocks[1][2]      # 190197


# ── PART 1: Metrica Data Loader & Clean Feature Extraction ───────────────────

def load_metrica_tracking(game_dir):
    """Load Metrica Sample Game tracking CSV and return player positions and ball coordinates."""
    home_df = pd.read_csv(os.path.join(game_dir, [f for f in os.listdir(game_dir) if 'Home' in f][0]), skiprows=2)
    away_df = pd.read_csv(os.path.join(game_dir, [f for f in os.listdir(game_dir) if 'Away' in f][0]), skiprows=2)
    
    n_frames = len(home_df)
    ball_col_idx = [i for i, c in enumerate(home_df.columns) if 'Ball' in str(c)]
    if ball_col_idx:
        ball_x_col = home_df.columns[ball_col_idx[0]]
        ball_y_col = home_df.columns[ball_col_idx[0] + 1]
    else:
        ball_x_col = home_df.columns[-2]
        ball_y_col = home_df.columns[-1]
    
    ball_x = home_df[ball_x_col].values.astype(np.float32)
    ball_y = home_df[ball_y_col].values.astype(np.float32)
    
    home_positions = []
    for pname in [c for c in home_df.columns if c.startswith('Player')]:
        idx = home_df.columns.get_loc(pname)
        home_positions.append((
            home_df.iloc[:, idx].values.astype(np.float32),
            home_df.iloc[:, idx + 1].values.astype(np.float32)
        ))
    
    away_positions = []
    for pname in [c for c in away_df.columns if c.startswith('Player')]:
        idx = away_df.columns.get_loc(pname)
        away_positions.append((
            away_df.iloc[:, idx].values.astype(np.float32),
            away_df.iloc[:, idx + 1].values.astype(np.float32)
        ))
        
    return n_frames, ball_x, ball_y, home_positions, away_positions


def generate_possession_labels_from_events(events_path, n_frames, poss_types=None):
    """Convert Metrica event data into per-frame possession labels (H=0, A=1, D=2).

    events_path may also be a DataFrame with the same columns (metrica_data.py builds one from the
    Sample Game 3 JSON events). Only POSSESSION_EVENTS hand possession to the acting team, following
    CHALLENGE.md: a duel without subsequent control is not a change of possession.
    """
    events = events_path if isinstance(events_path, pd.DataFrame) else pd.read_csv(events_path)
    labels = np.full(n_frames, 2, dtype=np.int64)  # Default = D (dead)
    
    dead_types = {"SET PIECE", "BALL OUT", "CARD"}
    poss_types = poss_types or POSSESSION_EVENTS
    events = events.sort_values("Start Frame").reset_index(drop=True)
    
    curr_poss = 2
    last_event_frame = 0
    for _, ev in events.iterrows():
        sf = int(ev["Start Frame"])
        ef = int(ev["End Frame"]) if not pd.isna(ev["End Frame"]) and ev["End Frame"] > 0 else sf
        team = ev["Team"]
        etype = ev["Type"]
        team_code = 0 if team == "Home" else 1
        
        if etype in dead_types:
            dead_start = max(0, sf - 2 * FPS)
            dead_end = min(n_frames, sf + 3 * FPS)
            labels[dead_start:dead_end] = 2
            curr_poss = 2
            last_event_frame = dead_end
        elif etype in poss_types:
            if last_event_frame < sf and curr_poss != 2:
                labels[last_event_frame:sf] = curr_poss
            curr_poss = team_code
            labels[sf:min(n_frames, ef + 1)] = team_code
            last_event_frame = ef + 1
            
    if last_event_frame < n_frames and curr_poss != 2:
        labels[last_event_frame:n_frames] = curr_poss
    return labels


def extract_features_metrica_clean(n_frames, ball_x, ball_y, home_positions, away_positions):
    """Extract N_FEATURES physical features with trajectory interpolation for Metrica data."""
    L, W = PITCH_LENGTH, PITCH_WIDTH
    # Physics-based linear interpolation across missing ball gaps <= 50 frames (2.0s)
    s_bx = pd.Series(ball_x).interpolate(method='linear', limit=50, limit_area='inside').values * L
    s_by = pd.Series(ball_y).interpolate(method='linear', limit=50, limit_area='inside').values * W
    
    px_home = np.column_stack([p[0] * L for p in home_positions])
    py_home = np.column_stack([p[1] * W for p in home_positions])
    px_away = np.column_stack([p[0] * L for p in away_positions])
    py_away = np.column_stack([p[1] * W for p in away_positions])
    
    dx_h = px_home - s_bx[:, None]
    dy_h = py_home - s_by[:, None]
    dist_h = np.hypot(dx_h, dy_h)
    dist_h_filled = np.where(np.isnan(dist_h), 50.0, dist_h)
    
    dx_a = px_away - s_bx[:, None]
    dy_a = py_away - s_by[:, None]
    dist_a = np.hypot(dx_a, dy_a)
    dist_a_filled = np.where(np.isnan(dist_a), 50.0, dist_a)
    
    h_sorted = np.sort(dist_h_filled, axis=1)
    a_sorted = np.sort(dist_a_filled, axis=1)
    
    features = np.zeros((n_frames, 20), dtype=np.float32)
    features[:, 0] = h_sorted[:, 0]
    features[:, 1] = a_sorted[:, 0]
    features[:, 2] = features[:, 0] - features[:, 1]
    features[:, 3] = h_sorted[:, 1] if h_sorted.shape[1] > 1 else 50.0
    features[:, 4] = a_sorted[:, 1] if a_sorted.shape[1] > 1 else 50.0
    features[:, 5] = h_sorted[:, 2] if h_sorted.shape[1] > 2 else 50.0
    features[:, 6] = a_sorted[:, 2] if a_sorted.shape[1] > 2 else 50.0
    
    h_valid = ~np.isnan(dist_h)
    a_valid = ~np.isnan(dist_a)
    with np.errstate(all='ignore'):
        features[:, 7] = np.where(h_valid.any(axis=1), np.nanmean(dist_h, axis=1), 50.0)
        features[:, 8] = np.where(a_valid.any(axis=1), np.nanmean(dist_a, axis=1), 50.0)
        
    features[:, 9] = np.sum((dist_h < 5.0) & h_valid, axis=1)
    features[:, 10] = np.sum((dist_a < 5.0) & a_valid, axis=1)
    features[:, 11] = np.sum((dist_h < 10.0) & h_valid, axis=1)
    features[:, 12] = np.sum((dist_a < 10.0) & a_valid, axis=1)
    
    ball_valid = ~np.isnan(s_bx)
    features[:, 13] = np.where(ball_valid, s_bx / L, 0.5)
    features[:, 14] = np.where(ball_valid, s_by / W, 0.5)
    
    h_px_valid = ~np.isnan(px_home)
    a_px_valid = ~np.isnan(px_away)
    with np.errstate(all='ignore'):
        features[:, 15] = np.where(h_px_valid.any(axis=1), np.nanmean(px_home, axis=1) / L, 0.5)
        features[:, 16] = np.where(a_px_valid.any(axis=1), np.nanmean(px_away, axis=1) / L, 0.5)
        
    features[:, 17] = np.where(ball_valid & (s_bx / L < 0.33), 1.0, 0.0)
    features[:, 18] = np.where(ball_valid & (s_bx / L > 0.67), 1.0, 0.0)
    
    b_dx = np.diff(s_bx, prepend=s_bx[0])
    b_dy = np.diff(s_by, prepend=s_by[0])
    b_spd = np.hypot(b_dx, b_dy) * float(FPS)
    features[:, 19] = np.where(~np.isnan(b_spd), np.clip(b_spd, 0, 40.0), 0.0)
    
    return features


# ── PART 3: PyTorch Transformer Architecture ────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PossessionTransformer(nn.Module):
    def __init__(self, n_features=N_FEATURES, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=WINDOW_SIZE + 50)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        center = x.size(1) // 2
        return self.classifier(x[:, center])


# ── PART 4: Post-Processing ──────────────────────────────────────────────────

# Seconds of dead ball assumed for each set-piece type, from 2 s before the tag's start.
SET_PIECE_DEAD_SECONDS = {
    "corner kick":           5.0,
    "free kick":             5.5,
    "goal kick":             4.5,
    "throw-in":              3.0,
    "kick off (start)":      4.0,
    "kick off (after goal)": 5.0,
}


def smart_tagging_dead_mask(smart_tagging_path, n_frames):
    """Frames inside Smart Tagging set-piece restart windows and goal intervals."""
    root = ET.parse(smart_tagging_path).getroot()
    dead_mask = np.zeros(n_frames, dtype=bool)
    for inst in root.iter("instance"):
        start_s = float(inst.findtext("start"))
        end_s   = float(inst.findtext("end"))
        start_f = max(0, min(n_frames - 1, int(round(start_s * FPS))))
        end_f   = max(0, min(n_frames - 1, int(round(end_s   * FPS))))
        code = (inst.findtext("code") or "").strip().upper()
        tags = {lbl.findtext("group"): lbl.findtext("text") for lbl in inst.findall("label")}
        if code == "SET PIECES":
            sp_type = (tags.get("Type") or "").strip().lower()
            matched_dur = None
            for key, dur in SET_PIECE_DEAD_SECONDS.items():
                if key in sp_type:
                    matched_dur = dur
                    break
            if matched_dur is not None:
                dead_len = int(round(matched_dur * FPS))
                p_start = max(0, start_f - int(2.0 * FPS))
                p_end = min(n_frames - 1, p_start + dead_len)
                dead_mask[p_start : p_end + 1] = True
        elif code == "GOALS":
            dead_mask[start_f : min(n_frames - 1, end_f + 1)] = True
    return dead_mask


def post_process(probs, n_frames, smart_tagging_path=None, w=90, min_spell=50, structural=True):
    """Apply bilateral probability smoothing, structural zones, and Smart Tagging dead ball windows.

    structural=False skips the BAR-ATH-specific X/D zones, for scoring on other games.
    """
    # 1. Bilateral rolling window probability smoothing (radius w = 90 frames = 3.6s)
    cum = np.vstack([np.zeros((1, 3)), np.cumsum(probs, axis=0)])
    idx = np.arange(n_frames)
    lo = np.maximum(0, idx - w)
    hi = np.minimum(n_frames, idx + w + 1)
    w_sum = cum[hi] - cum[lo]
    pred_class = np.argmax(w_sum, axis=1)
    
    labels = np.array([INV_LABEL[p] for p in pred_class])
    
    # 2. Structural match zones (unscored match headers, halftimes, and kickoffs)
    if structural:
        labels[0 : HALF1_START] = "X"
        labels[HALFTIME_START : HALF2_START] = "X"
        labels[HALF2_END + 1 : n_frames] = "X"
        labels[HALF2_START : HALF2_KICKOFF] = "D"

    # 3. Smart Tagging Dead Ball Windows
    if smart_tagging_path and os.path.exists(smart_tagging_path):
        dead_mask = smart_tagging_dead_mask(smart_tagging_path, n_frames)
        for i in range(n_frames):
            if dead_mask[i] and labels[i] in ("H", "A"):
                labels[i] = "D"
                
    # 4. Incursion / micro-duel absorption (< min_spell frames = 2.0s)
    i = 0
    while i < n_frames:
        lbl = labels[i]
        if lbl in ("H", "A"):
            j = i
            while j < n_frames and labels[j] == lbl:
                j += 1
            if j - i < min_spell:
                prev = labels[i-1] if i > 0 else "D"
                nxt = labels[j] if j < n_frames else "D"
                labels[i:j] = prev if prev == nxt else "D"
            i = j
        else:
            i += 1
            
    # Re-apply structural match zones
    if structural:
        labels[0 : HALF1_START] = "X"
        labels[HALFTIME_START : HALF2_START] = "X"
        labels[HALF2_END + 1 : n_frames] = "X"
        labels[HALF2_START : HALF2_KICKOFF] = "D"

    return labels
