"""Analyze one 9HPT video and export hand kinematics.

Outputs in --out directory:
  tracking.csv        raw detections per frame
  kinematics.csv      smoothed coordinates, path, velocity, acceleration
  metrics.json        summary numbers
  annotated.mp4       input video with overlays
  plot_*.png          trajectory/path/velocity/acceleration plots

Example:
  python analyze_motion.py --video data/test.mp4 --out results/test --px-per-mm 5.2
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
except Exception:  # pragma: no cover
    savgol_filter = None

from Video import VideoReader, VideoWriter
from track_hand import HandTracker, draw_track


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


def interpolate_and_smooth(series: pd.Series, smooth_window: int, max_gap: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return s
    s = s.interpolate(limit=max_gap, limit_direction="both")
    if s.notna().sum() < 5:
        return s

    valid = s.dropna()
    w = positive_odd_window(smooth_window, len(valid))
    if savgol_filter is not None and w >= 5:
        smoothed = pd.Series(s.copy(), index=s.index)
        smoothed.loc[valid.index] = savgol_filter(valid.to_numpy(), window_length=w, polyorder=2, mode="interp")
        return smoothed
    return s.rolling(window=w, center=True, min_periods=1).mean()


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
    dt = 1.0 / fps
    x = interpolate_and_smooth(df[x_col], smooth_window, max_gap)
    y = interpolate_and_smooth(df[y_col], smooth_window, max_gap)

    out = df.copy()
    out[f"{prefix}_x_s"] = x
    out[f"{prefix}_y_s"] = y

    arr_x = x.to_numpy(dtype=float)
    arr_y = y.to_numpy(dtype=float)
    finite = np.isfinite(arr_x) & np.isfinite(arr_y)

    step = np.full(len(out), np.nan, dtype=float)
    if len(out) > 1:
        dx_step = np.diff(arr_x)
        dy_step = np.diff(arr_y)
        step[1:] = np.sqrt(dx_step * dx_step + dy_step * dy_step)
        bad = ~(finite[1:] & finite[:-1])
        step[1:][bad] = np.nan
    out[f"{prefix}_step_px"] = step
    out[f"{prefix}_path_px"] = pd.Series(step).fillna(0.0).cumsum().to_numpy()

    vx = np.full(len(out), np.nan, dtype=float)
    vy = np.full(len(out), np.nan, dtype=float)
    if finite.sum() >= 3:
        # np.gradient keeps length and estimates central differences.
        vx[finite] = np.gradient(arr_x[finite], dt)
        vy[finite] = np.gradient(arr_y[finite], dt)
    speed = np.sqrt(vx * vx + vy * vy)

    ax = np.full(len(out), np.nan, dtype=float)
    ay = np.full(len(out), np.nan, dtype=float)
    finite_v = np.isfinite(vx) & np.isfinite(vy)
    if finite_v.sum() >= 3:
        ax[finite_v] = np.gradient(vx[finite_v], dt)
        ay[finite_v] = np.gradient(vy[finite_v], dt)
    accel = np.sqrt(ax * ax + ay * ay)

    out[f"{prefix}_vx_px_s"] = vx
    out[f"{prefix}_vy_px_s"] = vy
    out[f"{prefix}_speed_px_s"] = speed
    out[f"{prefix}_ax_px_s2"] = ax
    out[f"{prefix}_ay_px_s2"] = ay
    out[f"{prefix}_accel_px_s2"] = accel

    if px_per_mm and px_per_mm > 0:
        out[f"{prefix}_step_mm"] = out[f"{prefix}_step_px"] / px_per_mm
        out[f"{prefix}_path_mm"] = out[f"{prefix}_path_px"] / px_per_mm
        out[f"{prefix}_speed_mm_s"] = out[f"{prefix}_speed_px_s"] / px_per_mm
        out[f"{prefix}_accel_mm_s2"] = out[f"{prefix}_accel_px_s2"] / px_per_mm

    return out


def make_metrics(df: pd.DataFrame, fps: float, px_per_mm: Optional[float]) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "fps": float(fps),
        "frames": int(len(df)),
        "duration_s": float(df["t"].max()) if len(df) else 0.0,
        "detection_rate": float(df["visible"].mean()) if len(df) else 0.0,
        "primary_unit": "mm" if px_per_mm and px_per_mm > 0 else "px",
    }
    suffix = "mm" if px_per_mm and px_per_mm > 0 else "px"
    path_col = f"hand_path_{suffix}"
    speed_col = f"hand_speed_{suffix}_s"
    accel_col = f"hand_accel_{suffix}_s2"
    if path_col in df:
        metrics["hand_total_path"] = float(np.nanmax(df[path_col].to_numpy()))
    if speed_col in df:
        metrics["hand_mean_speed"] = float(np.nanmean(df[speed_col].to_numpy()))
        metrics["hand_max_speed"] = float(np.nanmax(df[speed_col].to_numpy()))
    if accel_col in df:
        metrics["hand_mean_acceleration"] = float(np.nanmean(df[accel_col].to_numpy()))
        metrics["hand_max_acceleration"] = float(np.nanmax(df[accel_col].to_numpy()))

    # Optional subproblem: thumb and index if MediaPipe landmarks exist.
    for part in ["thumb", "index"]:
        pcol = f"{part}_path_{suffix}"
        scol = f"{part}_speed_{suffix}_s"
        if pcol in df and df[pcol].notna().any():
            metrics[f"{part}_total_path"] = float(np.nanmax(df[pcol].to_numpy()))
        if scol in df and df[scol].notna().any():
            metrics[f"{part}_mean_speed"] = float(np.nanmean(df[scol].to_numpy()))
            metrics[f"{part}_max_speed"] = float(np.nanmax(df[scol].to_numpy()))
    return metrics


def save_plots(df: pd.DataFrame, out_dir: Path, px_per_mm: Optional[float]) -> None:
    unit = "mm" if px_per_mm and px_per_mm > 0 else "px"
    path_col = f"hand_path_{unit}"
    speed_col = f"hand_speed_{unit}_s"
    accel_col = f"hand_accel_{unit}_s2"

    plt.figure(figsize=(8, 5))
    plt.plot(df["t"], df[path_col])
    plt.xlabel("t [s]")
    plt.ylabel(f"pot [{unit}]")
    plt.title("Dolžina poti roke skozi čas")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_path.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["t"], df[speed_col])
    plt.xlabel("t [s]")
    plt.ylabel(f"hitrost [{unit}/s]")
    plt.title("Hitrost roke skozi čas")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_speed.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["t"], df[accel_col])
    plt.xlabel("t [s]")
    plt.ylabel(f"pospešek [{unit}/s²]")
    plt.title("Pospešek roke skozi čas")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_acceleration.png", dpi=160)
    plt.close()

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


def detect_simple_events(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Heuristic event candidates: low-speed moments, thumb-index pinch minima.

    This is not needed for the basic task, but it gives a starting point for
    grasp/release subproblems. Treat output as candidates for manual checking.
    """
    events: List[Dict[str, object]] = []
    if "hand_speed_px_s" not in df or len(df) < 5:
        return pd.DataFrame(events)

    speed = df["hand_speed_px_s"].to_numpy(dtype=float)
    finite_speed = speed[np.isfinite(speed)]
    if finite_speed.size:
        low_thr = np.nanpercentile(finite_speed, 20)
        for i in range(2, len(df) - 2):
            if not np.isfinite(speed[i]):
                continue
            if speed[i] <= low_thr and speed[i] <= speed[i - 1] and speed[i] <= speed[i + 1]:
                # Keep candidates at least 0.25 s apart.
                if events and (df.loc[i, "t"] - float(events[-1]["t"])) < 0.25:
                    continue
                events.append({"frame": int(df.loc[i, "frame"]), "t": float(df.loc[i, "t"]), "event": "low_speed_candidate", "value": float(speed[i])})

    if {"thumb_x_s", "thumb_y_s", "index_x_s", "index_y_s"}.issubset(df.columns):
        d = np.sqrt((df["thumb_x_s"] - df["index_x_s"]) ** 2 + (df["thumb_y_s"] - df["index_y_s"]) ** 2).to_numpy(dtype=float)
        finite_d = d[np.isfinite(d)]
        if finite_d.size:
            pinch_thr = np.nanpercentile(finite_d, 15)
            last_t = -1e9
            for i in range(2, len(df) - 2):
                if not np.isfinite(d[i]):
                    continue
                if d[i] <= pinch_thr and d[i] <= d[i - 1] and d[i] <= d[i + 1] and df.loc[i, "t"] - last_t >= 0.25:
                    events.append({"frame": int(df.loc[i, "frame"]), "t": float(df.loc[i, "t"]), "event": "pinch_candidate", "value": float(d[i])})
                    last_t = float(df.loc[i, "t"])

    return pd.DataFrame(events).sort_values(["t", "event"]) if events else pd.DataFrame(events)


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
            if len(trail) > int(info.fps * 4):
                trail = trail[-int(info.fps * 4):]
        if writer is not None:
            annotated = draw_track(frame, row, trail=trail)
            writer.write(annotated)

        if frame_idx % 100 == 0 and frame_idx > 0:
            print(f"  {frame_idx}/{info.frame_count if info.frame_count else '?'} frames")

    reader.release()
    tracker.close()
    if writer is not None:
        writer.release()

    if not rows:
        raise RuntimeError("No frames were read from the video.")

    tracking = pd.DataFrame(rows)
    tracking.to_csv(out_dir / "tracking.csv", index=False)

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

    metrics = make_metrics(kin, info.fps, args.px_per_mm)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    events = detect_simple_events(kin, info.fps)
    events.to_csv(out_dir / "events_candidates.csv", index=False)

    save_plots(kin, out_dir, args.px_per_mm)

    print("Done.")
    print(f"Detection rate: {metrics.get('detection_rate', 0):.1%}")
    print(f"Results: {out_dir.resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nine-hole peg test hand kinematics analyzer")
    p.add_argument("--video", required=True, help="Path to input video, e.g. data/test.mp4")
    p.add_argument("--out", default="results", help="Output directory")
    p.add_argument("--px-per-mm", type=float, default=None, help="Optional calibration scale. If omitted, outputs stay in pixels.")
    p.add_argument("--smooth-window", type=int, default= nine_default_window(), help="Odd smoothing window in frames, default 9")
    p.add_argument("--max-gap", type=int, default=8, help="Interpolate at most this many missing frames")
    p.add_argument("--min-detection-confidence", type=float, default=0.45)
    p.add_argument("--min-tracking-confidence", type=float, default=0.45)
    p.add_argument("--fallback-min-area", type=int, default=1200)
    p.add_argument("--no-mediapipe", action="store_true", help="Disable MediaPipe and use only OpenCV fallback")
    p.add_argument("--no-annotated", action="store_true", help="Do not create annotated.mp4")
    return p


def nine_default_window() -> int:
    return 9


if __name__ == "__main__":
    analyze_video(build_arg_parser().parse_args())
