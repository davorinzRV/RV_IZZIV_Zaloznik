from pathlib import Path
import json
import pandas as pd


RESULTS_DIR = Path("results")

rows = []

for metrics_path in sorted(RESULTS_DIR.glob("**/metrics_clean.json")):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    folder_name = metrics_path.parent.name

    rows.append({
        "video": folder_name.replace("_clean", ""),
        "frames": metrics.get("frames"),
        "duration_s": metrics.get("duration_s"),
        "valid_rate_after_cleaning": metrics.get("valid_rate_after_cleaning"),
        "total_path_px": metrics.get("pinch_total_path_px"),
        "mean_speed_px_s": metrics.get("pinch_mean_speed_px_s"),
        "max_speed_px_s": metrics.get("pinch_max_speed_px_s"),
        "mean_acceleration_px_s2": metrics.get("pinch_mean_acceleration_px_s2"),
        "max_acceleration_px_s2": metrics.get("pinch_max_acceleration_px_s2"),
    })

df = pd.DataFrame(rows)

out_csv = RESULTS_DIR / "summary_metrics.csv"
df.to_csv(out_csv, index=False)

print(df)
print()
print(f"Najdenih rezultatov: {len(df)}")
print(f"Shranjeno v: {out_csv}")