"""Analyze one 9HPT video and export hand kinematics.

Outputs in --out directory:
  tracking.csv              raw detections per frame
  kinematics.csv            smoothed coordinates, path, velocity, acceleration
  metrics.json              summary numbers
  events_candidates.csv     simple event candidates
  annotated.mp4             input video with overlays
  plot_*.png                trajectory/path/velocity/acceleration plots

Example:
  python3 analyze_motion.py \
    --video "../data_rv_26/Data/patient_001/patient_001camP_0 20241121_10_21_17.mp4" \
    --out results/debug_test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None

from Video import VideoReader, VideoWriter
from track_hand import HandTracker, draw_track


# ============================================================
# POMOŽNE FUNKCIJE
# ============================================================

def nine_default_window() -> int:
    return 9


def safe_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.nanmean(values))


def safe_nanmedian(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.nanmedian(values))


def safe_nanpercentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.nanpercentile(values, percentile))


def safe_nanmax(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.nanmax(values))


def positive_odd_window(requested: int, n: int) -> int:
    if n < 3:
        return max(1, n)

    w = int(requested)

    if w < 3:
        w = 3

    if w % 2 == 0:
        w += 1

    if w > n:
        w = n if n % 2 == 1 else n - 1

    return max(3, w)


def get_visible_mask(df: pd.DataFrame) -> pd.Series:
    """Robustly convert visible column to boolean mask."""

    if "visible" not in df.columns:
        return pd.Series(True, index=df.index)

    visible = df["visible"]

    if visible.dtype == object:
        return visible.astype(str).str.lower().isin(["true", "1", "yes", "y"])

    return visible.fillna(False).astype(bool)


def interpolate_and_smooth(series: pd.Series, smooth_window: int, max_gap: int) -> pd.Series:
    """Interpolate short gaps and smooth signal."""

    s = pd.to_numeric(series, errors="coerce")

    if s.notna().sum() == 0:
        return s

    # Interpoliramo samo krajše luknje.
    s = s.interpolate(limit=max_gap, limit_direction="both")

    if s.notna().sum() < 5:
        return s

    valid = s.dropna()
    w = positive_odd_window(smooth_window, len(valid))

    if savgol_filter is not None and w >= 5:
        smoothed = pd.Series(s.copy(), index=s.index)
        smoothed.loc[valid.index] = savgol_filter(
            valid.to_numpy(),
            window_length=w,
            polyorder=2,
            mode="interp",
        )
        return smoothed

    return s.rolling(window=w, center=True, min_periods=1).mean()


# ============================================================
# KINEMATIKA
# ============================================================

def add_kinematics(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    prefix: str,
    fps: float,
    smooth_window: int,
    max_gap: int,
    px_per_mm: Optional[float],
) -> pd.DataFrame:
    """Add smoothed coordinates, cumulative path, speed and acceleration."""

    out = df.copy()

    # Če stolpcev ni, ustvarimo prazne rezultate.
    if x_col not in out.columns or y_col not in out.columns:
        out[f"{prefix}_x_s"] = np.nan
        out[f"{prefix}_y_s"] = np.nan
        out[f"{prefix}_step_px"] = np.nan
        out[f"{prefix}_path_px"] = np.nan
        out[f"{prefix}_vx_px_s"] = np.nan
        out[f"{prefix}_vy_px_s"] = np.nan
        out[f"{prefix}_speed_px_s"] = np.nan
        out[f"{prefix}_ax_px_s2"] = np.nan
        out[f"{prefix}_ay_px_s2"] = np.nan
        out[f"{prefix}_accel_px_s2"] = np.nan
        return out

    x = interpolate_and_smooth(out[x_col], smooth_window, max_gap)
    y = interpolate_and_smooth(out[y_col], smooth_window, max_gap)

    out[f"{prefix}_x_s"] = x
    out[f"{prefix}_y_s"] = y

    arr_x = x.to_numpy(dtype=float)
    arr_y = y.to_numpy(dtype=float)

    finite = np.isfinite(arr_x) & np.isfinite(arr_y)

    # Časovna os
    if "t" in out.columns:
        t = out["t"].to_numpy(dtype=float)
    else:
        dt = 1.0 / fps if fps and fps > 0 else 1.0
        t = np.arange(len(out), dtype=float) * dt

    # ------------------------------------------------------------
    # Korak in kumulativna pot
    # ------------------------------------------------------------

    step = np.full(len(out), np.nan, dtype=float)

    if len(out) > 1:
        dx_step = np.diff(arr_x)
        dy_step = np.diff(arr_y)

        step[1:] = np.sqrt(dx_step * dx_step + dy_step * dy_step)

        bad = ~(finite[1:] & finite[:-1])
        step[1:][bad] = np.nan

    out[f"{prefix}_step_px"] = step
    out[f"{prefix}_path_px"] = pd.Series(step).fillna(0.0).cumsum().to_numpy()

    # ------------------------------------------------------------
    # Hitrost
    # ------------------------------------------------------------

    vx = np.full(len(out), np.nan, dtype=float)
    vy = np.full(len(out), np.nan, dtype=float)

    if finite.sum() >= 3:
        t_valid = t[finite]
        x_valid = arr_x[finite]
        y_valid = arr_y[finite]

        # np.gradient podpira tudi neenakomeren čas, kar je bolj pravilno pri manjkajočih frame-ih.
        vx[finite] = np.gradient(x_valid, t_valid)
        vy[finite] = np.gradient(y_valid, t_valid)

    speed = np.sqrt(vx * vx + vy * vy)

    # ------------------------------------------------------------
    # Pospešek
    # ------------------------------------------------------------

    ax = np.full(len(out), np.nan, dtype=float)
    ay = np.full(len(out), np.nan, dtype=float)

    finite_v = np.isfinite(vx) & np.isfinite(vy)

    if finite_v.sum() >= 3:
        t_valid = t[finite_v]
        vx_valid = vx[finite_v]
        vy_valid = vy[finite_v]

        ax[finite_v] = np.gradient(vx_valid, t_valid)
        ay[finite_v] = np.gradient(vy_valid, t_valid)

    accel = np.sqrt(ax * ax + ay * ay)

    out[f"{prefix}_vx_px_s"] = vx
    out[f"{prefix}_vy_px_s"] = vy
    out[f"{prefix}_speed_px_s"] = speed
    out[f"{prefix}_ax_px_s2"] = ax
    out[f"{prefix}_ay_px_s2"] = ay
    out[f"{prefix}_accel_px_s2"] = accel

    # ------------------------------------------------------------
    # Pretvorba v mm, če je podana kalibracija
    # ------------------------------------------------------------

    if px_per_mm and px_per_mm > 0:
        out[f"{prefix}_step_mm"] = out[f"{prefix}_step_px"] / px_per_mm
        out[f"{prefix}_path_mm"] = out[f"{prefix}_path_px"] / px_per_mm
        out[f"{prefix}_speed_mm_s"] = out[f"{prefix}_speed_px_s"] / px_per_mm
        out[f"{prefix}_accel_mm_s2"] = out[f"{prefix}_accel_px_s2"] / px_per_mm

    return out


# ============================================================
# METRIKE
# ============================================================

def make_detection_metrics(raw: pd.DataFrame, kin: pd.DataFrame) -> Dict[str, object]:
    """Detection quality before and after interpolation/smoothing."""

    total_frames = int(len(raw))

    if total_frames == 0:
        return {
            "total_frames": 0,
            "valid_raw": 0,
            "missing_frames_raw": 0,
            "valid_rate_raw": 0.0,
            "valid_after_cleaning": 0,
            "missing_frames_after_cleaning": 0,
            "valid_rate_after_cleaning": 0.0,
            "interpolated_frames": 0,
        }

    visible_mask = get_visible_mask(raw)

    if "x" in raw.columns and "y" in raw.columns:
        raw_xy_mask = raw["x"].notna() & raw["y"].notna()
    else:
        raw_xy_mask = pd.Series(False, index=raw.index)

    valid_raw_mask = visible_mask & raw_xy_mask
    valid_raw = int(valid_raw_mask.sum())
    missing_raw = int(total_frames - valid_raw)
    valid_rate_raw = valid_raw / total_frames

    if "hand_x_s" in kin.columns and "hand_y_s" in kin.columns:
        valid_clean_mask = kin["hand_x_s"].notna() & kin["hand_y_s"].notna()
    else:
        valid_clean_mask = valid_raw_mask

    valid_after_cleaning = int(valid_clean_mask.sum())
    missing_after_cleaning = int(total_frames - valid_after_cleaning)
    valid_rate_after_cleaning = valid_after_cleaning / total_frames

    interpolated_frames = int((valid_clean_mask & ~valid_raw_mask).sum())

    return {
        "total_frames": int(total_frames),
        "valid_raw": int(valid_raw),
        "missing_frames_raw": int(missing_raw),
        "valid_rate_raw": float(valid_rate_raw),
        "valid_after_cleaning": int(valid_after_cleaning),
        "missing_frames_after_cleaning": int(missing_after_cleaning),
        "valid_rate_after_cleaning": float(valid_rate_after_cleaning),
        "interpolated_frames": int(interpolated_frames),
    }


def make_metrics(df: pd.DataFrame, fps: float, px_per_mm: Optional[float]) -> Dict[str, object]:
    """Summary kinematic metrics."""

    duration_s = float(df["t"].max()) if "t" in df.columns and len(df) else 0.0

    metrics: Dict[str, object] = {
        "fps": float(fps),
        "frames": int(len(df)),
        "duration_s": duration_s,
        "detection_rate": float(get_visible_mask(df).mean()) if len(df) else 0.0,
        "primary_unit": "mm" if px_per_mm and px_per_mm > 0 else "px",
    }

    # ------------------------------------------------------------
    # Vedno izračunamo px metrike, ker jih uporablja summary_results.py
    # ------------------------------------------------------------

    if "hand_path_px" in df.columns:
        values = df["hand_path_px"].to_numpy(dtype=float)
        metrics["total_path_px"] = safe_nanmax(values)
        metrics["hand_total_path_px"] = safe_nanmax(values)

    if "hand_speed_px_s" in df.columns:
        values = df["hand_speed_px_s"].to_numpy(dtype=float)

        metrics["mean_speed_px_s"] = safe_nanmean(values)
        metrics["median_speed_px_s"] = safe_nanmedian(values)
        metrics["p95_speed_px_s"] = safe_nanpercentile(values, 95)
        metrics["max_speed_px_s"] = safe_nanmax(values)

        metrics["hand_mean_speed_px_s"] = metrics["mean_speed_px_s"]
        metrics["hand_median_speed_px_s"] = metrics["median_speed_px_s"]
        metrics["hand_p95_speed_px_s"] = metrics["p95_speed_px_s"]
        metrics["hand_max_speed_px_s"] = metrics["max_speed_px_s"]

    if "hand_accel_px_s2" in df.columns:
        values = df["hand_accel_px_s2"].to_numpy(dtype=float)

        metrics["mean_acceleration_px_s2"] = safe_nanmean(values)
        metrics["median_acceleration_px_s2"] = safe_nanmedian(values)
        metrics["p95_acceleration_px_s2"] = safe_nanpercentile(values, 95)
        metrics["max_acceleration_px_s2"] = safe_nanmax(values)

        metrics["hand_mean_acceleration_px_s2"] = metrics["mean_acceleration_px_s2"]
        metrics["hand_median_acceleration_px_s2"] = metrics["median_acceleration_px_s2"]
        metrics["hand_p95_acceleration_px_s2"] = metrics["p95_acceleration_px_s2"]
        metrics["hand_max_acceleration_px_s2"] = metrics["max_acceleration_px_s2"]

    # ------------------------------------------------------------
    # Primarne metrike: px ali mm, odvisno od kalibracije
    # ------------------------------------------------------------

    suffix = "mm" if px_per_mm and px_per_mm > 0 else "px"

    path_col = f"hand_path_{suffix}"
    speed_col = f"hand_speed_{suffix}_s"
    accel_col = f"hand_accel_{suffix}_s2"

    if path_col in df.columns:
        metrics["hand_total_path"] = safe_nanmax(df[path_col].to_numpy(dtype=float))

    if speed_col in df.columns:
        values = df[speed_col].to_numpy(dtype=float)

        metrics["hand_mean_speed"] = safe_nanmean(values)
        metrics["hand_median_speed"] = safe_nanmedian(values)
        metrics["hand_p95_speed"] = safe_nanpercentile(values, 95)
        metrics["hand_max_speed"] = safe_nanmax(values)

    if accel_col in df.columns:
        values = df[accel_col].to_numpy(dtype=float)

        metrics["hand_mean_acceleration"] = safe_nanmean(values)
        metrics["hand_median_acceleration"] = safe_nanmedian(values)
        metrics["hand_p95_acceleration"] = safe_nanpercentile(values, 95)
        metrics["hand_max_acceleration"] = safe_nanmax(values)

    # ------------------------------------------------------------
    # Če obstajajo mm stolpci, jih dodatno shranimo
    # ------------------------------------------------------------

    if "hand_path_mm" in df.columns:
        metrics["total_path_mm"] = safe_nanmax(df["hand_path_mm"].to_numpy(dtype=float))

    if "hand_speed_mm_s" in df.columns:
        values = df["hand_speed_mm_s"].to_numpy(dtype=float)
        metrics["mean_speed_mm_s"] = safe_nanmean(values)
        metrics["median_speed_mm_s"] = safe_nanmedian(values)
        metrics["p95_speed_mm_s"] = safe_nanpercentile(values, 95)
        metrics["max_speed_mm_s"] = safe_nanmax(values)

    if "hand_accel_mm_s2" in df.columns:
        values = df["hand_accel_mm_s2"].to_numpy(dtype=float)
        metrics["mean_acceleration_mm_s2"] = safe_nanmean(values)
        metrics["median_acceleration_mm_s2"] = safe_nanmedian(values)
        metrics["p95_acceleration_mm_s2"] = safe_nanpercentile(values, 95)
        metrics["max_acceleration_mm_s2"] = safe_nanmax(values)

    # ------------------------------------------------------------
    # Dodatno: palec in kazalec, če obstajata
    # ------------------------------------------------------------

    for part in ["thumb", "index"]:
        pcol = f"{part}_path_{suffix}"
        scol = f"{part}_speed_{suffix}_s"
        acol = f"{part}_accel_{suffix}_s2"

        if pcol in df.columns and df[pcol].notna().any():
            metrics[f"{part}_total_path"] = safe_nanmax(df[pcol].to_numpy(dtype=float))

        if scol in df.columns and df[scol].notna().any():
            values = df[scol].to_numpy(dtype=float)
            metrics[f"{part}_mean_speed"] = safe_nanmean(values)
            metrics[f"{part}_median_speed"] = safe_nanmedian(values)
            metrics[f"{part}_p95_speed"] = safe_nanpercentile(values, 95)
            metrics[f"{part}_max_speed"] = safe_nanmax(values)

        if acol in df.columns and df[acol].notna().any():
            values = df[acol].to_numpy(dtype=float)
            metrics[f"{part}_mean_acceleration"] = safe_nanmean(values)
            metrics[f"{part}_median_acceleration"] = safe_nanmedian(values)
            metrics[f"{part}_p95_acceleration"] = safe_nanpercentile(values, 95)
            metrics[f"{part}_max_acceleration"] = safe_nanmax(values)

    return metrics


# ============================================================
# GRAFI
# ============================================================

def save_plots(df: pd.DataFrame, out_dir: Path, px_per_mm: Optional[float]) -> None:
    unit = "mm" if px_per_mm and px_per_mm > 0 else "px"

    path_col = f"hand_path_{unit}"
    speed_col = f"hand_speed_{unit}_s"
    accel_col = f"hand_accel_{unit}_s2"

    if path_col in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["t"], df[path_col])
        plt.xlabel("t [s]")
        plt.ylabel(f"pot [{unit}]")
        plt.title("Dolžina poti roke skozi čas")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_path.png", dpi=160)
        plt.close()

    if speed_col in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["t"], df[speed_col])
        plt.xlabel("t [s]")
        plt.ylabel(f"hitrost [{unit}/s]")
        plt.title("Hitrost roke skozi čas")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_speed.png", dpi=160)
        plt.close()

    if accel_col in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["t"], df[accel_col])
        plt.xlabel("t [s]")
        plt.ylabel(f"pospešek [{unit}/s²]")
        plt.title("Pospešek roke skozi čas")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_acceleration.png", dpi=160)
        plt.close()

    if "hand_x_s" in df.columns and "hand_y_s" in df.columns:
        plt.figure(figsize=(7, 6))
        plt.plot(df["hand_x_s"], df["hand_y_s"])
        plt.gca().invert_yaxis()
        plt.xlabel("x [px]")
        plt.ylabel("y [px]")
        plt.title("Trajektorija roke v sliki")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_trajectory.png", dpi=160)
        plt.close()


# ============================================================
# PREPROSTI DOGODKI
# ============================================================

def detect_simple_events(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Heuristic event candidates: low-speed moments, thumb-index pinch minima."""

    events: List[Dict[str, object]] = []

    if "hand_speed_px_s" not in df.columns or len(df) < 5:
        return pd.DataFrame(events)

    speed = df["hand_speed_px_s"].to_numpy(dtype=float)
    finite_speed = speed[np.isfinite(speed)]

    if finite_speed.size:
        low_thr = np.nanpercentile(finite_speed, 20)

        for i in range(2, len(df) - 2):
            if not np.isfinite(speed[i]):
                continue

            if speed[i] <= low_thr and speed[i] <= speed[i - 1] and speed[i] <= speed[i + 1]:
                if events and (df.loc[i, "t"] - float(events[-1]["t"])) < 0.25:
                    continue

                events.append({
                    "frame": int(df.loc[i, "frame"]),
                    "t": float(df.loc[i, "t"]),
                    "event": "low_speed_candidate",
                    "value": float(speed[i]),
                })

    if {"thumb_x_s", "thumb_y_s", "index_x_s", "index_y_s"}.issubset(df.columns):
        d = np.sqrt(
            (df["thumb_x_s"] - df["index_x_s"]) ** 2
            + (df["thumb_y_s"] - df["index_y_s"]) ** 2
        ).to_numpy(dtype=float)

        finite_d = d[np.isfinite(d)]

        if finite_d.size:
            pinch_thr = np.nanpercentile(finite_d, 15)
            last_t = -1e9

            for i in range(2, len(df) - 2):
                if not np.isfinite(d[i]):
                    continue

                if (
                    d[i] <= pinch_thr
                    and d[i] <= d[i - 1]
                    and d[i] <= d[i + 1]
                    and df.loc[i, "t"] - last_t >= 0.25
                ):
                    events.append({
                        "frame": int(df.loc[i, "frame"]),
                        "t": float(df.loc[i, "t"]),
                        "event": "pinch_candidate",
                        "value": float(d[i]),
                    })
                    last_t = float(df.loc[i, "t"])

    if events:
        return pd.DataFrame(events).sort_values(["t", "event"])

    return pd.DataFrame(events)


# ============================================================
# GLAVNA ANALIZA VIDEA
# ============================================================

def analyze_video(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(args.video)
    info = reader.info

    tracker = HandTracker(
        use_mediapipe=not args.no_mediapipe,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        fallback_min_area=args.fallback_min_area,
    )

    writer: Optional[VideoWriter] = None

    if not args.no_annotated:
        writer = VideoWriter(out_dir / "annotated.mp4", info.fps, info.width, info.height)

    rows: List[Dict[str, object]] = []
    trail: List[Tuple[float, float]] = []

    print(f"Processing {args.video}")
    print(f"Video: {info.width}x{info.height}, {info.frame_count} frames, {info.fps:.3f} fps")

    for frame_idx, t, frame in reader.frames():
        res = tracker.process(frame, frame_idx, t)
        row = res.to_dict()
        rows.append(row)

        if res.visible and math.isfinite(res.x) and math.isfinite(res.y):
            trail.append((res.x, res.y))

            max_trail_len = int(info.fps * 4) if info.fps and info.fps > 0 else 100

            if len(trail) > max_trail_len:
                trail = trail[-max_trail_len:]

        if writer is not None:
            annotated = draw_track(frame, row, trail=trail)
            writer.write(annotated)

        if frame_idx % 100 == 0 and frame_idx > 0:
            total = info.frame_count if info.frame_count else "?"
            print(f"  {frame_idx}/{total} frames")

    reader.release()
    tracker.close()

    if writer is not None:
        writer.release()

    if not rows:
        raise RuntimeError("No frames were read from the video.")

    # ------------------------------------------------------------
    # Raw tracking
    # ------------------------------------------------------------

    tracking = pd.DataFrame(rows)
    tracking.to_csv(out_dir / "tracking.csv", index=False)

    # ------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------

    kin = tracking.copy()

    kin = add_kinematics(
        kin,
        x_col="x",
        y_col="y",
        prefix="hand",
        fps=info.fps,
        smooth_window=args.smooth_window,
        max_gap=args.max_gap,
        px_per_mm=args.px_per_mm,
    )

    kin = add_kinematics(
        kin,
        x_col="thumb_x",
        y_col="thumb_y",
        prefix="thumb",
        fps=info.fps,
        smooth_window=args.smooth_window,
        max_gap=args.max_gap,
        px_per_mm=args.px_per_mm,
    )

    kin = add_kinematics(
        kin,
        x_col="index_x",
        y_col="index_y",
        prefix="index",
        fps=info.fps,
        smooth_window=args.smooth_window,
        max_gap=args.max_gap,
        px_per_mm=args.px_per_mm,
    )

    kin.to_csv(out_dir / "kinematics.csv", index=False)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    metrics = make_metrics(kin, info.fps, args.px_per_mm)
    metrics.update(make_detection_metrics(tracking, kin))

    metrics["video_path"] = str(args.video)
    metrics["video_name"] = Path(args.video).stem
    metrics["output_dir"] = str(out_dir)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------
    # Events and plots
    # ------------------------------------------------------------

    events = detect_simple_events(kin, info.fps)
    events.to_csv(out_dir / "events_candidates.csv", index=False)

    save_plots(kin, out_dir, args.px_per_mm)

    print("Done.")
    print(f"Raw detection rate: {metrics.get('valid_rate_raw', 0):.1%}")
    print(f"Detection rate after cleaning: {metrics.get('valid_rate_after_cleaning', 0):.1%}")
    print(f"Interpolated frames: {metrics.get('interpolated_frames', 0)}")
    print(f"Mean speed: {metrics.get('mean_speed_px_s', 0):.3f} px/s")
    print(f"P95 speed: {metrics.get('p95_speed_px_s', 0):.3f} px/s")
    print(f"Max speed: {metrics.get('max_speed_px_s', 0):.3f} px/s")
    print(f"Mean acceleration: {metrics.get('mean_acceleration_px_s2', 0):.3f} px/s²")
    print(f"P95 acceleration: {metrics.get('p95_acceleration_px_s2', 0):.3f} px/s²")
    print(f"Max acceleration: {metrics.get('max_acceleration_px_s2', 0):.3f} px/s²")
    print(f"Results: {out_dir.resolve()}")


# ============================================================
# ARGUMENTI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nine-hole peg test hand kinematics analyzer")

    p.add_argument(
        "--video",
        required=True,
        help="Path to input video, e.g. data/test.mp4",
    )

    p.add_argument(
        "--out",
        default="results",
        help="Output directory",
    )

    p.add_argument(
        "--px-per-mm",
        type=float,
        default=None,
        help="Optional calibration scale. If omitted, outputs stay in pixels.",
    )

    p.add_argument(
        "--smooth-window",
        type=int,
        default=nine_default_window(),
        help="Odd smoothing window in frames, default 9",
    )

    p.add_argument(
        "--max-gap",
        type=int,
        default=8,
        help="Interpolate at most this many missing frames",
    )

    p.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.45,
    )

    p.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.45,
    )

    p.add_argument(
        "--fallback-min-area",
        type=int,
        default=1200,
    )

    p.add_argument(
        "--no-mediapipe",
        action="store_true",
        help="Disable MediaPipe and use only OpenCV fallback",
    )

    p.add_argument(
        "--no-annotated",
        action="store_true",
        help="Do not create annotated.mp4",
    )

    return p


if __name__ == "__main__":
    analyze_video(build_arg_parser().parse_args())