from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
EVENT_OUT_DIR = Path("event_results")

# Kalibracija iz sredinske kamere
PX_PER_MM = 1.356

# Nastavitve detekcije dogodkov
LOW_SPEED_PERCENTILE = 25
MIN_TIME_BETWEEN_EVENTS = 0.35

# Nastavitve grafa
LABEL_EVERY = 5   # na grafu oštevilči samo vsak 5. dogodek


def detect_events(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"frame", "t", "x_smooth", "y_smooth", "speed_px_s"}

    if not required.issubset(df.columns):
        print("Manjkajo stolpci v:", csv_path)
        print("Stolpci:", list(df.columns))
        return pd.DataFrame()

    speed = pd.to_numeric(df["speed_px_s"], errors="coerce").to_numpy(dtype=float)
    time = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)

    finite_speed = speed[np.isfinite(speed)]

    if finite_speed.size == 0:
        return pd.DataFrame()

    # Dogodke iščemo pri najpočasnejših delih gibanja.
    low_speed_threshold = np.nanpercentile(finite_speed, LOW_SPEED_PERCENTILE)

    events = []
    last_event_time = -1e9

    for i in range(2, len(df) - 2):
        if not np.isfinite(speed[i]) or not np.isfinite(time[i]):
            continue

        # Lokalni minimum hitrosti v oknu petih frame-ov
        is_local_minimum = (
            speed[i] <= speed[i - 1]
            and speed[i] <= speed[i + 1]
            and speed[i] <= speed[i - 2]
            and speed[i] <= speed[i + 2]
        )

        is_slow = speed[i] <= low_speed_threshold
        far_enough = (time[i] - last_event_time) >= MIN_TIME_BETWEEN_EVENTS

        if is_local_minimum and is_slow and far_enough:
            x = df.loc[i, "x_smooth"]
            y = df.loc[i, "y_smooth"]

            events.append({
                "event_id": len(events) + 1,
                "frame": int(df.loc[i, "frame"]),
                "t_s": float(time[i]),

                "x_px": float(x) if pd.notna(x) else np.nan,
                "y_px": float(y) if pd.notna(y) else np.nan,

                "x_mm": float(x / PX_PER_MM) if pd.notna(x) else np.nan,
                "y_mm": float(y / PX_PER_MM) if pd.notna(y) else np.nan,

                "speed_px_s": float(speed[i]),
                "speed_mm_s": float(speed[i] / PX_PER_MM),

                "event_type": "low_speed_candidate",
            })

            last_event_time = time[i]

    return pd.DataFrame(events)


def save_event_plot(csv_path: Path, events: pd.DataFrame, video_name: str) -> None:
    df = pd.read_csv(csv_path)

    EVENT_OUT_DIR.mkdir(exist_ok=True)

    out_path = EVENT_OUT_DIR / f"{video_name}_plot_events_clean.png"

    plt.figure(figsize=(18, 7))

    # Glavna krivulja hitrosti
    plt.plot(
        df["t"],
        df["speed_px_s"],
        linewidth=2,
        label="hitrost prijemne točke",
    )

    if not events.empty:
        # Vsi kandidati so prikazani kot točke
        plt.scatter(
            events["t_s"],
            events["speed_px_s"],
            s=70,
            zorder=3,
            label="kandidati za prijem/odlaganje",
        )

        # Zaradi preglednosti oštevilčimo samo vsak LABEL_EVERY dogodek.
        # Vsi dogodki so še vedno zapisani v CSV.
        for idx, (_, row) in enumerate(events.iterrows()):
            event_id = int(row["event_id"])

            if event_id % LABEL_EVERY != 0:
                continue

            x = float(row["t_s"])
            y = float(row["speed_px_s"])

            label = str(event_id)

            # Oznako postavimo nekoliko nad točko
            y_text = y + 160

            # Če je oznaka prenizko, jo dvignemo
            if y_text < 120:
                y_text = 120

            plt.plot(
                [x, x],
                [y, y_text - 10],
                linestyle="--",
                linewidth=0.8,
                alpha=0.45,
            )

            plt.text(
                x,
                y_text,
                label,
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.22",
                    fc="white",
                    ec="gray",
                    alpha=0.9,
                ),
            )

    plt.xlabel("t [s]")
    plt.ylabel("hitrost [px/s]")
    plt.title("Kandidati za prijem/odlaganje na podlagi lokalnih minimumov hitrosti")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    print("Shranjeno:", out_path)


def main():
    EVENT_OUT_DIR.mkdir(exist_ok=True)

    all_events = []

    csv_files = sorted(RESULTS_DIR.glob("*_clean/kinematics_clean.csv"))

    if not csv_files:
        print("Ni najdenih datotek *_clean/kinematics_clean.csv")
        return

    for csv_path in csv_files:
        video_name = csv_path.parent.name.replace("_clean", "")

        print()
        print("Obdelujem:", video_name)

        events = detect_events(csv_path)

        out_csv = EVENT_OUT_DIR / f"{video_name}_events_candidates_clean.csv"
        events.to_csv(out_csv, index=False)

        print("Najdenih kandidatov:", len(events))
        print("Shranjeno:", out_csv)

        save_event_plot(csv_path, events, video_name)

        if not events.empty:
            events_with_video = events.copy()
            events_with_video.insert(0, "video", video_name)
            all_events.append(events_with_video)

    if all_events:
        summary = pd.concat(all_events, ignore_index=True)
    else:
        summary = pd.DataFrame()

    summary_path = EVENT_OUT_DIR / "summary_events_candidates.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("Skupni povzetek:", summary_path)

    if not summary.empty:
        print(
            summary[
                [
                    "video",
                    "event_id",
                    "frame",
                    "t_s",
                    "speed_px_s",
                    "speed_mm_s",
                ]
            ].head(30)
        )


if __name__ == "__main__":
    main()