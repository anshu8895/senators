"""External Transformer Possession Detector — Team Senators
Competition: More than a Hack 2027 — Phase 1: Ball Possession

Data Locations & Requirements:
- Match constants & boundaries: games.py in repository root
- Competition tracking & video: ./data (or set environment variable MTAH_DATA_DIR)
- External training data (optional for --retrain): external_data/metrica/data/Sample_Game_1 and Sample_Game_2
  (cloned from https://github.com/metrica-sports/sample-data)

Methodology:
- Trained on external open tracking and human event data from Metrica Sports (Sample Games 1 & 2).
- Zero training, fitting, or calibration on the competition match (BAR-ATH) or its ground truth.
- Linear ball trajectory interpolation (limit=50 frames = 2.0s) for interior missing observations;
  untracked frames beyond 50 frames fall back to a neutral 50.0m player-to-ball distance.
- Pre-trained weights committed in ext_transformer_weights.pt (105,795 trainable parameters).
- Note: Running with --retrain will overwrite ext_transformer_weights.pt and norm_stats.npz.
- Output: evaluation/results/Senators.csv (199,319 rows)

Author: Senators Team
"""

import os
import sys
import math
import random
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
from games import GAME, GAMES, require, source_frames, half_video_blocks, anchor_video_frame

# ── Constants ────────────────────────────────────────────────────────────────
FPS = 25
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0
N_FEATURES   = 20
WINDOW_SIZE  = 100   # context window (frames) for transformer (4.0s)
LABEL_MAP    = {"H": 0, "A": 1, "D": 2}
INV_LABEL    = {0: "H", 1: "A", 2: "D"}

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


def generate_possession_labels_from_events(events_path, n_frames):
    """Convert Metrica event data into per-frame possession labels (H=0, A=1, D=2)."""
    events = pd.read_csv(events_path)
    labels = np.full(n_frames, 2, dtype=np.int64)  # Default = D (dead)
    
    dead_types = {"SET PIECE", "BALL OUT", "CARD"}
    poss_types = {"PASS", "RECOVERY", "SHOT", "CHALLENGE", "FAULT RECEIVED", "BALL LOST"}
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


# ── PART 2: Competition Match Clean Feature Extraction ───────────────────────

def extract_features_competition_clean(raw_path, team_of_track, n_frames, pitch):
    """Extract identical 20 physical features with trajectory interpolation for BAR-ATH."""
    L, W = pitch
    n_slots = len(team_of_track)
    
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    
    with open(raw_path) as f:
        for line in f:
            parts = line.split(':')
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1
            if fid < 0 or fid >= n_frames:
                continue
            b = parts[2]
            if b[0] != 'N':
                c1 = b.find(',')
                c2 = b.find(',', c1 + 1)
                bx[fid] = float(b[:c1])
                by[fid] = float(b[c1+1:c2])
            for s_idx, s in enumerate(parts[1].split(';')):
                if s[0] != 'N':
                    gc1 = s.find(',')
                    gc2 = s.find(',', gc1 + 1)
                    px[fid, s_idx] = float(s[:gc1])
                    py[fid, s_idx] = float(s[gc1+1:gc2])
                    
    # Interpolate ball trajectory across short tracking gaps <= 50 frames (2.0s)
    s_bx = pd.Series(bx).interpolate(method='linear', limit=50, limit_area='inside').values * L
    s_by = pd.Series(by).interpolate(method='linear', limit=50, limit_area='inside').values * W
    px = px * L
    py = py * W
    
    dx = px - s_bx[:, None]
    dy = py - s_by[:, None]
    dist = np.hypot(dx, dy)
    dist_filled = np.where(np.isnan(dist), 50.0, dist)
    
    home_mask = (team_of_track == 0)
    away_mask = (team_of_track == 1)
    
    home_dist = np.sort(dist_filled[:, home_mask], axis=1)
    away_dist = np.sort(dist_filled[:, away_mask], axis=1)
    
    features = np.zeros((n_frames, 20), dtype=np.float32)
    features[:, 0] = home_dist[:, 0]
    features[:, 1] = away_dist[:, 0]
    features[:, 2] = features[:, 0] - features[:, 1]
    features[:, 3] = home_dist[:, 1]
    features[:, 4] = away_dist[:, 1]
    features[:, 5] = home_dist[:, 2]
    features[:, 6] = away_dist[:, 2]
    
    h_valid = ~np.isnan(dist[:, home_mask])
    a_valid = ~np.isnan(dist[:, away_mask])
    with np.errstate(all='ignore'):
        features[:, 7] = np.where(h_valid.any(axis=1), np.nanmean(dist[:, home_mask], axis=1), 50.0)
        features[:, 8] = np.where(a_valid.any(axis=1), np.nanmean(dist[:, away_mask], axis=1), 50.0)
        
    features[:, 9] = np.sum((dist[:, home_mask] < 5.0) & h_valid, axis=1)
    features[:, 10] = np.sum((dist[:, away_mask] < 5.0) & a_valid, axis=1)
    features[:, 11] = np.sum((dist[:, home_mask] < 10.0) & h_valid, axis=1)
    features[:, 12] = np.sum((dist[:, away_mask] < 10.0) & a_valid, axis=1)
    
    ball_valid = ~np.isnan(s_bx)
    features[:, 13] = np.where(ball_valid, s_bx / L, 0.5)
    features[:, 14] = np.where(ball_valid, s_by / W, 0.5)
    
    h_px_valid = ~np.isnan(px[:, home_mask])
    a_px_valid = ~np.isnan(px[:, away_mask])
    with np.errstate(all='ignore'):
        features[:, 15] = np.where(h_px_valid.any(axis=1), np.nanmean(px[:, home_mask], axis=1) / L, 0.5)
        features[:, 16] = np.where(a_px_valid.any(axis=1), np.nanmean(px[:, away_mask], axis=1) / L, 0.5)
        
    features[:, 17] = np.where(ball_valid & (s_bx / L < 0.33), 1.0, 0.0)
    features[:, 18] = np.where(ball_valid & (s_bx / L > 0.67), 1.0, 0.0)
    
    b_dx = np.diff(s_bx, prepend=s_bx[0])
    b_dy = np.diff(s_by, prepend=s_by[0])
    b_spd = np.hypot(b_dx, b_dy) * float(FPS)
    features[:, 19] = np.where(~np.isnan(b_spd), np.clip(b_spd, 0, 40.0), 0.0)
    
    return features


# ── PART 3: PyTorch Transformer Architecture ────────────────────────────────

class PossessionDataset(Dataset):
    def __init__(self, features, labels, window=WINDOW_SIZE):
        self.features = features
        self.labels = labels
        self.window = window
        self.n = len(features)
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        half_w = self.window // 2
        start = max(0, idx - half_w)
        end = min(self.n, idx + half_w)
        chunk = self.features[start:end]
        if len(chunk) < self.window:
            pad = np.zeros((self.window - len(chunk), N_FEATURES), dtype=np.float32)
            if start == 0:
                chunk = np.vstack([pad, chunk])
            else:
                chunk = np.vstack([chunk, pad])
        return torch.tensor(chunk, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


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


def train_transformer(features, labels, epochs=35, lr=1e-3, batch_size=256):
    """Train transformer strictly on external Metrica data."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")
    
    feat_mean = features.mean(axis=0)
    feat_std  = features.std(axis=0) + 1e-8
    features_norm = (features - feat_mean) / feat_std
    
    n = len(features_norm)
    split = int(n * 0.85)
    train_ds = PossessionDataset(features_norm[:split], labels[:split], window=WINDOW_SIZE)
    val_ds   = PossessionDataset(features_norm[split:], labels[split:], window=WINDOW_SIZE)
    
    train_gen = torch.Generator().manual_seed(SEED)
    val_gen   = torch.Generator().manual_seed(SEED)
    train_indices = np.arange(0, len(train_ds), 2)
    val_indices   = np.arange(0, len(val_ds), 4)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices, generator=train_gen)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices, generator=val_gen)
    )
    
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    weights = (counts.sum() / (3.0 * counts)).clip(0.5, 10.0)
    print(f"  Class weights: H={weights[0]:.2f}, A={weights[1]:.2f}, D={weights[2]:.2f}")
    
    model = PossessionTransformer().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    best_state = None
    patience = 0
    
    for ep in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        model.eval()
        val_loss, val_tot = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item() * len(y)
                val_tot += len(y)
        vl = val_loss / max(val_tot, 1)
        scheduler.step()
        
        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:2d}/{epochs:2d} -> val_loss: {vl:.4f}")
            
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                print(f"  Early stopping at epoch {ep}")
                break
                
    model.load_state_dict(best_state)
    return model.to(device), feat_mean, feat_std, device


def infer_transformer(model, features, feat_mean, feat_std, device, n_frames):
    """Run trained model on competition features and return class probabilities."""
    features_norm = (features - feat_mean) / feat_std
    dataset = PossessionDataset(features_norm, np.zeros(n_frames, dtype=np.int64), window=WINDOW_SIZE)
    model.eval()
    probs = np.zeros((n_frames, 3), dtype=np.float32)
    batch_size = 512
    with torch.no_grad():
        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            batch_X = torch.stack([dataset[i][0] for i in range(start, end)]).to(device)
            probs[start:end] = torch.softmax(model(batch_X), dim=-1).cpu().numpy()
    return probs


# ── PART 4: Post-Processing ──────────────────────────────────────────────────

def post_process(probs, n_frames, smart_tagging_path=None, w=90, min_spell=50):
    """Apply bilateral probability smoothing, structural zones, and Smart Tagging dead ball windows."""
    # 1. Bilateral rolling window probability smoothing (radius w = 90 frames = 3.6s)
    cum = np.vstack([np.zeros((1, 3)), np.cumsum(probs, axis=0)])
    idx = np.arange(n_frames)
    lo = np.maximum(0, idx - w)
    hi = np.minimum(n_frames, idx + w + 1)
    w_sum = cum[hi] - cum[lo]
    pred_class = np.argmax(w_sum, axis=1)
    
    labels = np.array([INV_LABEL[p] for p in pred_class])
    
    # 2. Structural match zones (unscored match headers, halftimes, and kickoffs)
    labels[0 : HALF1_START] = "X"
    labels[HALFTIME_START : HALF2_START] = "X"
    labels[HALF2_END + 1 : n_frames] = "X"
    labels[HALF2_START : HALF2_KICKOFF] = "D"
    
    # 3. Smart Tagging Dead Ball Windows
    if smart_tagging_path and os.path.exists(smart_tagging_path):
        root = ET.parse(smart_tagging_path).getroot()
        dead_mask = np.zeros(n_frames, dtype=bool)
        SET_PIECE_DEAD_SECONDS = {
            "corner kick":           5.0,
            "free kick":             5.5,
            "goal kick":             4.5,
            "throw-in":              3.0,
            "kick off (start)":      4.0,
            "kick off (after goal)": 5.0,
        }
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
    labels[0 : HALF1_START] = "X"
    labels[HALFTIME_START : HALF2_START] = "X"
    labels[HALF2_END + 1 : n_frames] = "X"
    labels[HALF2_START : HALF2_KICKOFF] = "D"
    
    return labels


# ── PART 5: Main Production Entry Point ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="External Transformer Detector (Senators Team)")
    parser.add_argument("--retrain", action="store_true", help="Force deterministic retraining on Metrica data")
    parser.add_argument("--window", type=int, default=90, help="Smoothing radius (default: 90 frames = 3.6s)")
    parser.add_argument("--out", type=str, default=os.path.join(HERE, "evaluation", "results", "Senators.csv"),
                        help="Output submission path (default: evaluation/results/Senators.csv)")
    args = parser.parse_args()

    print("=" * 70)
    print("🧠 SENATORS EXTERNAL TRANSFORMER — Trajectory Imputation Pipeline")
    print("=" * 70)
    
    weights_path = os.path.join(HERE, "ext_transformer_weights.pt")
    norm_path = os.path.join(HERE, "norm_stats.npz")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")
    
    if os.path.exists(weights_path) and os.path.exists(norm_path) and not args.retrain:
        print("\n[1/3] Found existing trained weights and norm stats. Loading...")
        norm = np.load(norm_path)
        feat_mean = norm["mean"]
        feat_std = norm["std"]
        model = PossessionTransformer().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("  Model loaded successfully!")
    else:
        print("\n[1/3] Training Transformer on clean Metrica Sample Games 1 & 2...")
        feats_list, labels_list = [], []
        for gname in ["Sample_Game_1", "Sample_Game_2"]:
            gdir = os.path.join(HERE, "external_data", "metrica", "data", gname)
            ev_path = os.path.join(gdir, f"{gname}_RawEventsData.csv")
            n_f, bx, by, h_pos, a_pos = load_metrica_tracking(gdir)
            g_labels = generate_possession_labels_from_events(ev_path, n_f)
            g_feats  = extract_features_metrica_clean(n_f, bx, by, h_pos, a_pos)
            feats_list.append(g_feats)
            labels_list.append(g_labels)
            
        ext_features = np.concatenate(feats_list, axis=0)
        ext_labels   = np.concatenate(labels_list, axis=0)
        print(f"  External training set: {len(ext_features):,} frames, 20 clean physical features.")
        
        model, feat_mean, feat_std, device = train_transformer(ext_features, ext_labels, epochs=35)
        torch.save(model.state_dict(), weights_path)
        np.savez(norm_path, mean=feat_mean, std=feat_std)
        print(f"  Model weights saved: {weights_path}")
        print(f"  Norm stats saved:    {norm_path}")
        
    print("\n[2/3] Extracting clean competition features (BAR-ATH)...")
    meta_path, raw_path, smart_path = require(["atd_meta", "atd_raw", "smart_tagging"])
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    team_of_track = np.array([0 if tid == home_id else 1 for tid in slots], dtype=np.int8)
    
    comp_features = extract_features_competition_clean(raw_path, team_of_track, n_frames, pitch)
    print(f"  BAR-ATH clean features: {comp_features.shape}")
    
    print("\n[3/3] Running inference & post-processing...")
    probs = infer_transformer(model, comp_features, feat_mean, feat_std, device, n_frames)
    final_labels = post_process(probs, n_frames, smart_tagging_path=smart_path, w=args.window, min_spell=50)
    
    counts = {lbl: int(np.sum(final_labels == lbl)) for lbl in ("H", "A", "D", "X")}
    print("\n📊 Final Label Distribution:")
    for lbl, count in counts.items():
        print(f"   {lbl}: {count:>7,} ({100.0 * count / n_frames:5.1f}%)")
        
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{frame},{lbl}\n" for frame, lbl in enumerate(final_labels))
    print(f"\n✅ Submission successfully written to: {out_path}")


if __name__ == "__main__":
    main()
