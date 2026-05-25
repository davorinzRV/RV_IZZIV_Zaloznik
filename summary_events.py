from pathlib import Path
import pandas as pd

input_csv = Path("event_results/summary_events_candidates.csv")
out_csv = Path("event_results/summary_events_by_video.csv")

df = pd.read_csv(input_csv)

summary = (
    df.groupby("video")
    .agg(
        num_candidates=("event_id", "count"),
        first_event_s=("t_s", "min"),
        last_event_s=("t_s", "max"),
        mean_speed_mm_s=("speed_mm_s", "mean"),
        median_speed_mm_s=("speed_mm_s", "median"),
    )
    .reset_index()
)

summary["first_event_s"] = summary["first_event_s"].round(2)
summary["last_event_s"] = summary["last_event_s"].round(2)
summary["mean_speed_mm_s"] = summary["mean_speed_mm_s"].round(2)
summary["median_speed_mm_s"] = summary["median_speed_mm_s"].round(2)

summary.to_csv(out_csv, index=False)

print(summary)
print()
print("Shranjeno:", out_csv)
