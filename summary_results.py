import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
OUT_CSV = RESULTS_DIR / "summary_metrics.csv"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_value(metrics, *keys, default=None):
    for key in keys:
        if key in metrics:
            return metrics[key]
    return default


def safe_values(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def safe_median(values):
    return float(np.nanmedian(values)) if values.size else None


def safe_p95(values):
    return float(np.nanpercentile(values, 95)) if values.size else None


def find_raw_metrics(clean_dir):
    """
    Za:
        results/nekaj_v3_clean

    poišče:
        results/nekaj_v3/metrics.json
    """
    if not clean_dir.name.endswith("_clean"):
        return {}

    raw_name = clean_dir.name.replace("_clean", "")
    raw_metrics_path = clean_dir.parent / raw_name / "metrics.json"

    if raw_metrics_path.exists():
        return read_json(raw_metrics_path)

    return {}


def compute_extra_from_clean_csv(clean_dir):
    """
    Iz kinematics_clean.csv izračuna mediane in 95. percentile.
    """
    csv_path = clean_dir / "kinematics_clean.csv"

    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)

    extra = {}

    if "speed_px_s" in df.columns:
        speed = safe_values(df["speed_px_s"])
        extra["median_speed_px_s"] = safe_median(speed)
        extra["p95_speed_px_s"] = safe_p95(speed)

    if "acceleration_px_s2" in df.columns:
        accel = safe_values(df["acceleration_px_s2"])
        extra["median_acceleration_px_s2"] = safe_median(accel)
        extra["p95_acceleration_px_s2"] = safe_p95(accel)

    return extra


def main():
    rows = []

    metrics_files = sorted(RESULTS_DIR.glob("*_clean/metrics_clean.json"))

    for metrics_path in metrics_files:
        clean_dir = metrics_path.parent

        clean_metrics = read_json(metrics_path)
        raw_metrics = find_raw_metrics(clean_dir)
        extra = compute_extra_from_clean_csv(clean_dir)

        video_name = clean_dir.name.replace("_clean", "")

        row = {
            "video": video_name,

            "frames": get_value(
                clean_metrics,
                "frames",
                default=get_value(raw_metrics, "frames", default=0),
            ),

            "duration_s": get_value(
                clean_metrics,
                "duration_s",
                default=get_value(raw_metrics, "duration_s", default=0),
            ),

            # raw zaznava iz osnovne analize
            "valid_raw": get_value(raw_metrics, "valid_raw", default=None),
            "missing_frames_raw": get_value(raw_metrics, "missing_frames_raw", default=None),
            "valid_rate_raw": get_value(
                raw_metrics,
                "valid_rate_raw",
                default=get_value(raw_metrics, "detection_rate", default=None),
            ),
            "interpolated_frames": get_value(raw_metrics, "interpolated_frames", default=None),

            # clean zaznava
            "valid_rate_after_cleaning": get_value(
                clean_metrics,
                "valid_rate_after_cleaning",
                default=get_value(raw_metrics, "valid_rate_after_cleaning", default=None),
            ),

            # pot iz clean metrik
            "total_path_px": get_value(
                clean_metrics,
                "total_path_px",
                "hand_total_path_px",
                "pinch_total_path_px",
                default=None,
            ),

            # hitrost iz clean metrik + median/p95 iz kinematics_clean.csv
            "mean_speed_px_s": get_value(
                clean_metrics,
                "mean_speed_px_s",
                "hand_mean_speed_px_s",
                "pinch_mean_speed_px_s",
                default=None,
            ),

            "median_speed_px_s": get_value(
                clean_metrics,
                "median_speed_px_s",
                "hand_median_speed_px_s",
                "pinch_median_speed_px_s",
                default=extra.get("median_speed_px_s"),
            ),

            "p95_speed_px_s": get_value(
                clean_metrics,
                "p95_speed_px_s",
                "hand_p95_speed_px_s",
                "pinch_p95_speed_px_s",
                default=extra.get("p95_speed_px_s"),
            ),

            "max_speed_px_s": get_value(
                clean_metrics,
                "max_speed_px_s",
                "hand_max_speed_px_s",
                "pinch_max_speed_px_s",
                default=None,
            ),

            # pospešek iz clean metrik + median/p95 iz kinematics_clean.csv
            "mean_acceleration_px_s2": get_value(
                clean_metrics,
                "mean_acceleration_px_s2",
                "hand_mean_acceleration_px_s2",
                "pinch_mean_acceleration_px_s2",
                default=None,
            ),

            "median_acceleration_px_s2": get_value(
                clean_metrics,
                "median_acceleration_px_s2",
                "hand_median_acceleration_px_s2",
                "pinch_median_acceleration_px_s2",
                default=extra.get("median_acceleration_px_s2"),
            ),

            "p95_acceleration_px_s2": get_value(
                clean_metrics,
                "p95_acceleration_px_s2",
                "hand_p95_acceleration_px_s2",
                "pinch_p95_acceleration_px_s2",
                default=extra.get("p95_acceleration_px_s2"),
            ),

            "max_acceleration_px_s2": get_value(
                clean_metrics,
                "max_acceleration_px_s2",
                "hand_max_acceleration_px_s2",
                "pinch_max_acceleration_px_s2",
                default=None,
            ),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("Ni najdenih clean rezultatov.")
        return

    df = df.sort_values("video").reset_index(drop=True)

    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(df)
    print()
    print("Najdenih rezultatov:", len(df))
    print("Shranjeno v:", OUT_CSV)


if __name__ == "__main__":
    main()