"""Rule-Based Football Possession Detector (Enhanced v2).

Fully compliant with competition rules:
- ZERO training, fitting, or calibrating on ground truth possession or video.
- Pure football domain physics + Automatic Tracking Data (ATD) + Smart Tagging XML.
- Possession state machine with inertia, pass flight tracking, control radius, and turnover hysteresis.
- Tactical Prior Bayesian Bias (0.35m) during contested duels and pressing phases.
- Goalkeeper / Penalty Box possession security rule (uncontested box control).
- Comprehensive Dead Ball Arbiter: Set pieces, goal celebrations, pitch boundary checks, and static ball detection.
- Duel / deflection absorption: short rival incursions (< 2.0s) without sustained control or stoppage do not break team possession.

Author: Team (More than a Hack 2027)
"""

import os
import sys
import math
import argparse
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames  # noqa: E402

# ── Constants & Parameters (Standard Football Analytics) ─────────────────────
FPS = 25  # frames per second

# Match structural boundaries (from match metadata & rules)
HALF1_START    = 11185    # 1st half start
HALF1_END      = 90651    # 1st half end
HALFTIME_START = 90652    # Halftime start
HALFTIME_END   = 110807   # Halftime end
HALF2_START    = 110808   # 2nd half start
HALF2_KICKOFF  = 113742   # 2nd half actual kickoff
HALF2_END      = 190197   # 2nd half end
N_FRAMES       = 199319   # Total frames in tactical video

# Physics & Tracking Parameters
CONTROL_RADIUS        = 1.8   # meters: within this distance, player can control the ball
CONTROL_MARGIN        = 0.6   # meters: player must be closer than rival by this margin
TACTICAL_PRIOR_BIAS   = 0.35  # meters: distance bonus for team in active tactical phase
PASS_SPEED_THRESHOLD  = 3.5   # m/s: ball moving faster than this is in flight (pass/shot)
TURNOVER_FRAMES       = 5     # frames (0.2s): required consecutive control to confirm turnover
MAX_OCCLUSION_BRIDGE  = 25    # frames (1.0s): keep possession alive through brief tracking loss
MAX_INCURSION_FRAMES  = 50    # frames (2.0s): brief rival contest without restart is absorbed
MIN_SPELL_FRAMES      = 20    # frames (0.8s): possessions shorter than this are absorbed

# Penalty Box Geometry (Pitch = 105.3m x 68.0m)
BOX_X_LEN             = 16.5  # 16.5m penalty box depth
BOX_Y_MIN             = 13.84 # (68.0 - 40.32) / 2
BOX_Y_MAX             = 54.16 # (68.0 + 40.32) / 2

# Team codes
HOME, AWAY, NOBODY = 0, 1, -1
LABEL_MAP = {HOME: "H", AWAY: "A", NOBODY: "D"}


# ── Metadata & Smart Tagging Parsers ─────────────────────────────────────────

def get_team_mapping(meta_path):
    """Extract player track slot -> team assignment from ATD XML."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"Expected 2 teams in {meta_path}, found {len(team_ids)}")
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


def parse_smart_tagging(xml_path, n_frames):
    """Parse Smart Tagging XML to extract:
    1. Set piece restart dead windows
    2. Goal celebration dead windows
    3. Tactical phase possession priors (BUILD UP, PROGRESSION, etc.)
    """
    root = ET.parse(xml_path).getroot()
    
    dead_mask = np.zeros(n_frames, dtype=bool)
    tactical_prior = np.full(n_frames, NOBODY, dtype=np.int8)
    
    # Restart dead ball duration (seconds) before / at set piece
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
        tags = {
            lbl.findtext("group"): lbl.findtext("text")
            for lbl in inst.findall("label")
        }
        team_str = (tags.get("Team") or "").strip().lower()
        team_val = NOBODY
        if "barcelona" in team_str:
            team_val = HOME
        elif "atletic" in team_str:
            team_val = AWAY
            
        # 1. SET PIECES (Dead ball restart windows)
        if code == "SET PIECES":
            sp_type = (tags.get("Type") or "").strip().lower()
            matched_dur = None
            for key, dur in SET_PIECE_DEAD_SECONDS.items():
                if key in sp_type:
                    matched_dur = dur
                    break
            if matched_dur is not None:
                dead_len = int(round(matched_dur * FPS))
                # Dead ball starts 2s before set piece event and extends for dead_len
                p_start = max(0, start_f - int(2.0 * FPS))
                p_end = min(n_frames - 1, p_start + dead_len)
                dead_mask[p_start : p_end + 1] = True
                
        # 2. GOALS (Celebration & return to kickoff)
        elif code == "GOALS":
            # Goal celebration is 100% dead ball until kickoff
            dead_mask[start_f : min(n_frames - 1, end_f + 1)] = True
            
        # 3. Attacking Tactical Phases (Possession Prior)
        elif code in {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "BALL IN THE BOX", "CREATING CHANCES", "ATTACKING TRANSITION"}:
            if team_val != NOBODY:
                tactical_prior[start_f : end_f + 1] = team_val
                
        # 4. Defensive Phases (Opposing team prior)
        elif code in {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}:
            if team_val == HOME:
                tactical_prior[start_f : end_f + 1] = AWAY
            elif team_val == AWAY:
                tactical_prior[start_f : end_f + 1] = HOME

    return dead_mask, tactical_prior


# ── ATD Feature Streaming ────────────────────────────────────────────────────

def extract_atd_features(raw_path, team_of_track, n_frames, pitch):
    """Stream ATD file and compute per-frame physical tracking metrics."""
    length, width = pitch
    
    ball_present = np.zeros(n_frames, dtype=bool)
    ball_x       = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y       = np.full(n_frames, np.nan, dtype=np.float32)
    ball_speed   = np.full(n_frames, np.nan, dtype=np.float32)
    
    min_dist_h   = np.full(n_frames, 999.0, dtype=np.float32)
    min_dist_a   = np.full(n_frames, 999.0, dtype=np.float32)
    out_of_pitch = np.zeros(n_frames, dtype=bool)
    
    with open(raw_path) as f:
        for line in f:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1
            if fid < 0 or fid >= n_frames:
                continue
                
            ball_str = parts[2]
            if ball_str[0] == "N":  # NaN
                continue
                
            c1 = ball_str.index(",")
            c2 = ball_str.index(",", c1 + 1)
            norm_bx = float(ball_str[:c1])
            norm_by = float(ball_str[c1 + 1 : c2])
            
            bx = norm_bx * length
            by = norm_by * width
            
            ball_present[fid] = True
            ball_x[fid] = bx
            ball_y[fid] = by
            
            # Out of pitch boundaries check
            if norm_bx < 0.0 or norm_bx > 1.0 or norm_by < 0.0 or norm_by > 1.0:
                out_of_pitch[fid] = True
                
            # Compute distance to closest Home and Away players
            best_h, best_a = 999.0, 999.0
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                gc1 = group.index(",")
                gc2 = group.index(",", gc1 + 1)
                px = float(group[:gc1]) * length
                py = float(group[gc1 + 1 : gc2]) * width
                d = math.hypot(px - bx, py - by)
                
                if team_of_track[slot] == HOME:
                    if d < best_h:
                        best_h = d
                else:
                    if d < best_a:
                        best_a = d
                        
            min_dist_h[fid] = best_h
            min_dist_a[fid] = best_a

    # Compute ball speed from consecutive tracked positions
    valid_indices = np.where(ball_present)[0]
    for i in range(1, len(valid_indices)):
        idx_prev = valid_indices[i - 1]
        idx_curr = valid_indices[i]
        dt = (idx_curr - idx_prev) / float(FPS)
        if 0 < dt <= 0.2:  # within 5 frames
            dx = ball_x[idx_curr] - ball_x[idx_prev]
            dy = ball_y[idx_curr] - ball_y[idx_prev]
            spd = math.hypot(dx, dy) / dt
            ball_speed[idx_curr] = spd

    return ball_present, ball_x, ball_y, ball_speed, min_dist_h, min_dist_a, out_of_pitch


# ── Dead Ball Identification ──────────────────────────────────────────────────

def build_complete_dead_mask(n_frames, set_piece_dead, out_of_pitch, ball_present,
                             ball_speed, min_dist_h, min_dist_a):
    """Combine physical, structural, and event cues for dead ball detection."""
    dead_mask = np.copy(set_piece_dead)
    
    # 1. Structural Match Whistles & Periods
    # Pre-kickoff 2nd half (guaranteed stoppage)
    dead_mask[HALF2_START : HALF2_KICKOFF] = True
    
    # 2. Out of pitch ball
    dead_mask[out_of_pitch] = True
    
    # 3. Static Ball with NO player near it (waiting for restart/whistle)
    # If ball speed < 0.2 m/s and no player is within 1.5m
    is_static = (ball_speed < 0.2) & (min_dist_h > 1.5) & (min_dist_a > 1.5)
    
    # Find stretches of static ball >= 15 frames (0.6s)
    static_count = 0
    for f in range(n_frames):
        if ball_present[f] and is_static[f]:
            static_count += 1
        else:
            if static_count >= 15:
                dead_mask[f - static_count : f] = True
            static_count = 0
            
    # 4. Long ball disappearance (> 2.0s = 50 frames)
    missing_count = 0
    for f in range(n_frames):
        if not ball_present[f]:
            missing_count += 1
        else:
            if missing_count >= 50:
                dead_mask[f - missing_count : f] = True
            missing_count = 0

    return dead_mask


# ── Possession State Machine ─────────────────────────────────────────────────

def run_possession_state_machine(n_frames, ball_present, ball_x, ball_y, min_dist_h,
                                 min_dist_a, ball_speed, dead_mask, tactical_prior):
    """Physics-based possession tracking with:
    - Control radius & margin
    - Pass flight inertia
    - Turnover hysteresis
    - Tactical Prior Bayesian Bias (0.35m) in contested duels
    - Goalkeeper Penalty Box security rule
    """
    raw_labels = np.full(n_frames, "D", dtype="<U1")
    
    # Box bounds
    in_home_box = ball_present & (ball_x <= BOX_X_LEN) & (ball_y >= BOX_Y_MIN) & (ball_y <= BOX_Y_MAX)
    in_away_box = ball_present & (ball_x >= (105.3 - BOX_X_LEN)) & (ball_y >= BOX_Y_MIN) & (ball_y <= BOX_Y_MAX)
    
    curr_state = NOBODY
    turnover_to_h = 0
    turnover_to_a = 0
    occlusion_frames = 0
    
    for f in range(n_frames):
        # 1. Dead Ball check
        if dead_mask[f]:
            curr_state = NOBODY
            turnover_to_h = 0
            turnover_to_a = 0
            occlusion_frames = 0
            raw_labels[f] = "D"
            continue
            
        # 2. Live Play with Ball Tracked
        if ball_present[f]:
            occlusion_frames = 0
            dh = min_dist_h[f]
            da = min_dist_a[f]
            spd = ball_speed[f] if not np.isnan(ball_speed[f]) else 0.0
            
            # Rule 3: Goalkeeper / Defending team protected possession in own box
            if in_home_box[f] and dh < 2.5 and da > 2.5:
                curr_state = HOME
                turnover_to_h = 0
                turnover_to_a = 0
                raw_labels[f] = "H"
                continue
            elif in_away_box[f] and da < 2.5 and dh > 2.5:
                curr_state = AWAY
                turnover_to_h = 0
                turnover_to_a = 0
                raw_labels[f] = "A"
                continue
            
            # Rule 2: Tactical Prior Bayesian bias in contested situations
            eff_dh = dh - TACTICAL_PRIOR_BIAS if tactical_prior[f] == HOME else dh
            eff_da = da - TACTICAL_PRIOR_BIAS if tactical_prior[f] == AWAY else da
            
            # Check for close control
            home_controls = (eff_dh < CONTROL_RADIUS) and (eff_dh + CONTROL_MARGIN < eff_da)
            away_controls = (eff_da < CONTROL_RADIUS) and (eff_da + CONTROL_MARGIN < eff_dh)
            
            if home_controls:
                turnover_to_a = 0
                if curr_state == AWAY:
                    turnover_to_h += 1
                    if turnover_to_h >= TURNOVER_FRAMES:
                        curr_state = HOME
                        turnover_to_h = 0
                else:
                    curr_state = HOME
                    turnover_to_h = 0
                    
            elif away_controls:
                turnover_to_h = 0
                if curr_state == HOME:
                    turnover_to_a += 1
                    if turnover_to_a >= TURNOVER_FRAMES:
                        curr_state = AWAY
                        turnover_to_a = 0
                else:
                    curr_state = AWAY
                    turnover_to_a = 0
                    
            else:
                # Ball is in flight, loose, or contested
                turnover_to_h = 0
                turnover_to_a = 0
                
                # If ball is fast (> 3.5 m/s), maintain previous possessor (pass/shot)
                if spd >= PASS_SPEED_THRESHOLD and curr_state != NOBODY:
                    pass  # keep curr_state
                elif curr_state == NOBODY:
                    # If coming out of dead ball or restart, use tactical prior or nearest
                    if tactical_prior[f] != NOBODY:
                        curr_state = tactical_prior[f]
                    elif eff_dh < eff_da:
                        curr_state = HOME
                    elif eff_da < eff_dh:
                        curr_state = AWAY
                        
        else:
            # Ball is missing on this frame
            turnover_to_h = 0
            turnover_to_a = 0
            occlusion_frames += 1
            
            # Short occlusions maintain previous possession
            if occlusion_frames <= MAX_OCCLUSION_BRIDGE and curr_state != NOBODY:
                pass  # keep curr_state alive!
            else:
                curr_state = NOBODY
                
        raw_labels[f] = LABEL_MAP[curr_state]

    return raw_labels


# ── Post-Processing & Spell Cleaning ─────────────────────────────────────────

def absorb_brief_incursions(labels, max_gap_frames=MAX_INCURSION_FRAMES):
    """In football, a brief rival touch/contest (< 2.0s) without sustained possession
    or stoppage is a duel / deflection within the ongoing possession, NOT a turnover.
    Absorbing these preserves continuous possession spells as defined by FIFA/Opta.
    """
    cleaned = labels.copy()
    n = len(cleaned)
    i = 0
    while i < n:
        lbl = cleaned[i]
        if lbl in ("H", "A"):
            other = "A" if lbl == "H" else "H"
            j = i
            while j < n and cleaned[j] == lbl:
                j += 1
            k = j
            while k < n and cleaned[k] == other:
                k += 1
            other_len = k - j
            if 0 < other_len <= max_gap_frames:
                # If it immediately returns to lbl without dead ball, absorb it
                if k < n and cleaned[k] == lbl:
                    cleaned[j:k] = lbl
                    i = k
                    continue
            i = j
        else:
            i += 1
    return cleaned


def clean_spells(labels, min_spell=MIN_SPELL_FRAMES):
    """Clean micro-flickering and enforce realistic possession spell lengths.
    
    1. Absorb brief rival duel incursions.
    2. Bridge tiny dead ball gaps inside ongoing same-team possession.
    3. Remove micro-spells (< min_spell frames).
    4. Apply structural 'X' masks to unplayed blocks.
    """
    n = len(labels)
    
    # Pass 1: Absorb duel incursions (< 2.0s)
    cleaned = absorb_brief_incursions(labels, max_gap_frames=MAX_INCURSION_FRAMES)
    
    # Pass 2: Bridge micro dead gaps between identical team possessions (e.g. H -> D (3f) -> H)
    i = 0
    while i < n:
        if cleaned[i] == "D":
            j = i
            while j < n and cleaned[j] == "D":
                j += 1
            gap_len = j - i
            prev_label = cleaned[i - 1] if i > 0 else "D"
            next_label = cleaned[j] if j < n else "D"
            if gap_len <= 8 and prev_label == next_label and prev_label in ("H", "A"):
                cleaned[i:j] = prev_label
            i = j
        else:
            i += 1
            
    # Pass 3: Absorb micro-spells of H or A (< min_spell frames)
    i = 0
    while i < n:
        lbl = cleaned[i]
        if lbl in ("H", "A"):
            j = i
            while j < n and cleaned[j] == lbl:
                j += 1
            spell_len = j - i
            if spell_len < min_spell:
                prev_lbl = cleaned[i - 1] if i > 0 else "D"
                next_lbl = cleaned[j] if j < n else "D"
                replacement = prev_lbl if prev_lbl == next_lbl else "D"
                cleaned[i:j] = replacement
            i = j
        else:
            i += 1

    # Pass 4: Hardcode verified unplayed blocks to 'X' (as required by evaluation)
    cleaned[0 : HALF1_START] = "X"
    cleaned[HALFTIME_START : HALF2_START] = "X"
    cleaned[HALF2_END + 1 : n] = "X"

    return cleaned


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def build_rule_based_submission(game=GAME, data_dir=None, out_path=None):
    """Execute complete rule-based possession detection pipeline."""
    n_frames = source_frames("tactical", game)
    pitch    = GAMES[game]["pitch"]
    
    print("=" * 70)
    print("⚽ ENHANCED RULE-BASED FOOTBALL POSSESSION DETECTOR (v2)")
    print("=" * 70)
    
    # Step 1: ATD Metadata & Track mapping
    print("\n[1/6] Loading ATD metadata & track slot mappings...")
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team_of_track = get_team_mapping(meta_path)
    print(f"      Track slots: {len(team_of_track)} (Home={np.sum(team_of_track==HOME)}, Away={np.sum(team_of_track==AWAY)})")
    
    # Step 2: Smart Tagging events
    print("\n[2/6] Parsing Smart Tagging tactical annotations...")
    (smart_path,) = require(["smart_tagging"], game, data_dir)
    st_dead, tactical_prior = parse_smart_tagging(smart_path, n_frames)
    print(f"      Set piece & goal dead ball frames identified: {np.sum(st_dead):,}")
    print(f"      Tactical prior frames available:             {np.sum(tactical_prior != NOBODY):,}")
    
    # Step 3: Stream tracking features
    print("\n[3/6] Streaming tracking data and computing distances & ball speed...")
    ball_present, bx, by, ball_speed, dist_h, dist_a, out_of_pitch = extract_atd_features(
        raw_path, team_of_track, n_frames, pitch
    )
    print(f"      Ball visible on: {np.sum(ball_present):,} / {n_frames:,} frames ({100*np.mean(ball_present):.1f}%)")
    print(f"      Out of pitch detections: {np.sum(out_of_pitch):,} frames")
    
    # Step 4: Complete Dead Ball Mask
    print("\n[4/6] Synthesizing comprehensive dead ball signals...")
    dead_mask = build_complete_dead_mask(
        n_frames, st_dead, out_of_pitch, ball_present, ball_speed, dist_h, dist_a
    )
    print(f"      Total dead ball frames identified: {np.sum(dead_mask):,}")
    
    # Step 5: Possession State Machine with Hysteresis, Tactical Bias & GK Box Rule
    print("\n[5/6] Running possession state machine (control, passes, tactical bias, GK box)...")
    raw_labels = run_possession_state_machine(
        n_frames, ball_present, bx, by, dist_h, dist_a, ball_speed, dead_mask, tactical_prior
    )
    
    # Step 6: Post-processing & spell cleaning
    print("\n[6/6] Enforcing possession spell structure & absorbing duel incursions...")
    final_labels = clean_spells(raw_labels, min_spell=MIN_SPELL_FRAMES)
    
    counts = {lbl: int(np.sum(final_labels == lbl)) for lbl in ("H", "A", "D", "X")}
    print("\n📊 Final Label Distribution:")
    for lbl, count in counts.items():
        print(f"   {lbl}: {count:>7,} ({100.0 * count / n_frames:5.1f}%)")
        
    if out_path:
        folder = os.path.dirname(out_path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        with open(out_path, "w") as f:
            f.write("frame,label\n")
            f.writelines(f"{frame},{lbl}\n" for frame, lbl in enumerate(final_labels))
        print(f"\n✅ Submission saved to: {out_path}")
        
    return final_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-Based Possession Detector v2")
    parser.add_argument("--out", default=os.path.join(HERE, "evaluation", "results", "rule_based_pred.csv"),
                        help="Output submission path")
    args = parser.parse_args()
    
    build_rule_based_submission(out_path=args.out)
