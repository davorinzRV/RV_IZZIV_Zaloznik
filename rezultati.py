import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def rolling_mean(series, window):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def clean_coordinates(df, x_col, y_col, max_step_px, max_gap, smooth_window):
    out = df.copy()

    out["x_clean"] = pd.to_numeric(out[x_col], errors="coerce")
    out["y_clean"] = pd.to_numeric(out[y_col], errors="coerce")

    # odstrani prevelike skoke
    dx = out["x_clean"].diff()
    dy = out["y_clean"].diff()
    step = np.sqrt(dx**2 + dy**2)

    out["step_raw_px"] = step
    jump_mask = step > max_step_px

    out.loc[jump_mask, ["x_clean", "y_clean"]] = np.nan

    valid_raw = out[x_col].notna() & out[y_col].notna()
    valid_after_jump_filter = out["x_clean"].notna() & out["y_clean"].notna()

    # interpolacija kratkih vrzeli
    out["x_interp"] = out["x_clean"].interpolate(
        method="linear",
        limit=max_gap,
        limit_direction="both",
    )
    out["y_interp"] = out["y_clean"].interpolate(
        method="linear",
        limit=max_gap,
        limit_direction="both",
    )

    # glajenje
    out["x_smooth"] = rolling_mean(out["x_interp"], smooth_window)
    out["y_smooth"] = rolling_mean(out["y_interp"], smooth_window)

    valid_after_cleaning = out["x_smooth"].notna() & out["y_smooth"].notna()

    info = {
        "valid_raw": int(valid_raw.sum()),
        "missing_frames_raw": int((~valid_raw).sum()),
        "valid_rate_raw": float(valid_raw.mean()),
        "valid_after_jump_filter": int(valid_after_jump_filter.sum()),
        "valid_after_cleaning": int(valid_after_cleaning.sum()),
        "missing_frames_after_cleaning": int((~valid_after_cleaning).sum()),
        "valid_rate_after_cleaning": float(valid_after_cleaning.mean()),
        "interpolated_frames": int(
            valid_after_cleaning.sum() - valid_after_jump_filter.sum()
        ),
    }

    return out, info


def add_kinematics(df, fps):
    out = df.copy()

    if "frame" in out.columns:
        out["t"] = out["frame"] / fps
    elif "time_sec" in out.columns:
        out["t"] = out["time_sec"]
    elif "t" not in out.columns:
        out["t"] = np.arange(len(out)) / fps

    dx = out["x_smooth"].diff()
    dy = out["y_smooth"].diff()

    step = np.sqrt(dx**2 + dy**2)
    step = step.where(out["x_smooth"].notna() & out["y_smooth"].notna())

    out["step_px"] = step.fillna(0)
    out["path_px"] = out["step_px"].cumsum()

    dt = 1.0 / fps if fps > 0 else 1.0

    out["speed_px_s"] = out["step_px"] / dt
    out["acceleration_px_s2"] = out["speed_px_s"].diff().abs() / dt
    out["acceleration_px_s2"] = out["acceleration_px_s2"].fillna(0)

    return out


def make_metrics(df, fps, input_dir, output_dir, clean_info, max_step_px, smooth_window, max_gap):
    speed = df["speed_px_s"].replace([np.inf, -np.inf], np.nan).dropna()
    acc = df["acceleration_px_s2"].replace([np.inf, -np.inf], np.nan).dropna()

    metrics = {
        "fps": float(fps),
        "frames": int(len(df)),
        "duration_s": float(len(df) / fps) if fps > 0 else float(len(df)),
        "primary_unit": "px",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "max_step_px_filter": float(max_step_px),
        "smooth_window": int(smooth_window),
        "max_gap": int(max_gap),

        "total_path_px": float(df["path_px"].iloc[-1]) if len(df) else 0.0,
        "mean_speed_px_s": float(speed.mean()) if len(speed) else 0.0,
        "median_speed_px_s": float(speed.median()) if len(speed) else 0.0,
        "p95_speed_px_s": float(speed.quantile(0.95)) if len(speed) else 0.0,
        "max_speed_px_s": float(speed.max()) if len(speed) else 0.0,

        "mean_acceleration_px_s2": float(acc.mean()) if len(acc) else 0.0,
        "median_acceleration_px_s2": float(acc.median()) if len(acc) else 0.0,
        "p95_acceleration_px_s2": float(acc.quantile(0.95)) if len(acc) else 0.0,
        "max_acceleration_px_s2": float(acc.max()) if len(acc) else 0.0,
    }

    metrics.update(clean_info)

    return metrics


def save_plots(df, out_dir):
    out_dir = Path(out_dir)

    # trajektorija
    plt.figure(figsize=(7, 6))
    plt.plot(df["x_smooth"], df["y_smooth"])
    plt.gca().invert_yaxis()
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.title("Očiščena trajektorija prijemne točke")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_trajectory_clean.png", dpi=160)
    plt.close()

    # pot
    plt.figure(figsize=(10, 5))
    plt.plot(df["t"], df["path_px"])
    plt.xlabel("t [s]")
    plt.ylabel("pot [px]")
    plt.title("Očiščena dolžina poti skozi čas")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_path_clean.png", dpi=160)
    plt.close()

    # hitrost
    plt.figure(figsize=(10, 5))
    plt.plot(df["t"], df["speed_px_s"])
    plt.xlabel("t [s]")
    plt.ylabel("hitrost [px/s]")
    plt.title("Očiščena hitrost skozi čas")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_speed_clean.png", dpi=160)
    plt.close()

    # pospešek
    plt.figure(figsize=(10, 5))
    plt.plot(df["t"], df["acceleration_px_s2"])
    plt.xlabel("t [s]")
    plt.ylabel("pospešek [px/s²]")
    plt.title("Očiščen pospešek skozi čas")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_acceleration_clean.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Clean tracking output and export kinematics.")
    parser.add_argument("--input", required=True, help="Input result directory, e.g. results/video_v3")
    parser.add_argument("--output", required=True, help="Output clean directory")
    parser.add_argument("--max-step-px", type=float, default=80.0)
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=5)

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input mapa ne obstaja: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Kopiraj annotated.mp4, če obstaja
    if (input_dir / "annotated.mp4").exists():
        shutil.copy2(input_dir / "annotated.mp4", output_dir / "annotated.mp4")

    # Najdi vhodni CSV
    candidates = [
        input_dir / "tracking.csv",
        input_dir / "kinematics.csv",
        input_dir / "hand_tracking.csv",
    ]

    input_csv = None
    for c in candidates:
        if c.exists():
            input_csv = c
            break

    if input_csv is None:
        raise FileNotFoundError(
            f"Ne najdem vhodnega CSV v {input_dir}. Iskal sem tracking.csv, kinematics.csv, hand_tracking.csv."
        )

    df = pd.read_csv(input_csv)

    x_col = find_col(df, ["x", "pinch_x", "hand_x", "x_px"])
    y_col = find_col(df, ["y", "pinch_y", "hand_y", "y_px"])

    if x_col is None or y_col is None:
        raise ValueError(f"Ne najdem x/y stolpcev. Stolpci: {list(df.columns)}")

    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df))

    # fps preberi iz metrics.json, če obstaja
    fps = 25.0
    metrics_in = input_dir / "metrics.json"
    if metrics_in.exists():
        try:
            with open(metrics_in, "r", encoding="utf-8") as f:
                old_metrics = json.load(f)
            fps = float(old_metrics.get("fps", fps))
        except Exception:
            pass

    clean_df, clean_info = clean_coordinates(
        df,
        x_col=x_col,
        y_col=y_col,
        max_step_px=args.max_step_px,
        max_gap=args.max_gap,
        smooth_window=args.smooth_window,
    )

    kin = add_kinematics(clean_df, fps=fps)

    kin.to_csv(output_dir / "kinematics_clean.csv", index=False)

    metrics = make_metrics(
        kin,
        fps=fps,
        input_dir=input_dir,
        output_dir=output_dir,
        clean_info=clean_info,
        max_step_px=args.max_step_px,
        smooth_window=args.smooth_window,
        max_gap=args.max_gap,
    )

    with open(output_dir / "metrics_clean.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_plots(kin, output_dir)

    print("Clean rezultati shranjeni v:", output_dir)
    print("Veljavnost po čiščenju:", round(metrics["valid_rate_after_cleaning"] * 100, 2), "%")


if __name__ == "__main__":
    main()
