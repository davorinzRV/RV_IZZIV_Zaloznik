from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Mapa z tracking.csv")
parser.add_argument("--output", required=True, help="Izhodna mapa za očiščene rezultate")
args = parser.parse_args()

INPUT_DIR = Path(args.input)
OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_STEP_PX = 80.0      # največji dovoljen premik med zaporednima frame-oma
SMOOTH_WINDOW = 5       # glajenje koordinat


df = pd.read_csv(INPUT_DIR / "tracking.csv")

# Uporabimo samo vidne zaznave.
df["x_clean"] = df["x"]
df["y_clean"] = df["y"]

df.loc[df["visible"] != 1, ["x_clean", "y_clean"]] = np.nan

# Odstranimo nerealne skoke med zaporednimi frame-i.
dx_raw = df["x_clean"].diff()
dy_raw = df["y_clean"].diff()
step_raw = np.sqrt(dx_raw**2 + dy_raw**2)

bad = step_raw > MAX_STEP_PX
df.loc[bad, ["x_clean", "y_clean"]] = np.nan

# Interpoliramo kratke manjkajoče dele.
df["x_clean"] = df["x_clean"].interpolate(limit=5, limit_direction="both")
df["y_clean"] = df["y_clean"].interpolate(limit=5, limit_direction="both")

# Gladimo koordinate.
df["x_smooth"] = (
    df["x_clean"]
    .rolling(SMOOTH_WINDOW, center=True, min_periods=1)
    .median()
)

df["y_smooth"] = (
    df["y_clean"]
    .rolling(SMOOTH_WINDOW, center=True, min_periods=1)
    .median()
)

# Izračun kinematike.
t = df["t"].to_numpy()
x = df["x_smooth"].to_numpy()
y = df["y_smooth"].to_numpy()

valid = np.isfinite(x) & np.isfinite(y)

dx = np.diff(x, prepend=x[0])
dy = np.diff(y, prepend=y[0])
dt = np.diff(t, prepend=t[0])

dt[dt <= 0] = np.nan

step = np.sqrt(dx**2 + dy**2)

# Če so še vedno NaN vrednosti, jih ne štejemo v pot.
step[~np.isfinite(step)] = 0.0

speed = step / dt
speed[0] = 0.0
speed[~np.isfinite(speed)] = 0.0

acceleration = np.diff(speed, prepend=speed[0]) / dt
acceleration[0] = 0.0
acceleration[~np.isfinite(acceleration)] = 0.0

path = np.cumsum(step)

df["step_px"] = step
df["path_px"] = path
df["speed_px_s"] = speed
df["acceleration_px_s2"] = acceleration

# Shranimo očiščeno tabelo.
df.to_csv(OUTPUT_DIR / "kinematics_clean.csv", index=False)

metrics = {
    "frames": int(len(df)),
    "duration_s": float(df["t"].max()),
    "primary_unit": "px",
    "input_dir": str(INPUT_DIR),
    "output_dir": str(OUTPUT_DIR),
    "max_step_px_filter": MAX_STEP_PX,
    "smooth_window": SMOOTH_WINDOW,
    "valid_rate_after_cleaning": float(valid.mean()),
    "pinch_total_path_px": float(np.nanmax(path)),
    "pinch_mean_speed_px_s": float(np.nanmean(speed)),
    "pinch_max_speed_px_s": float(np.nanmax(speed)),
    "pinch_mean_acceleration_px_s2": float(np.nanmean(np.abs(acceleration))),
    "pinch_max_acceleration_px_s2": float(np.nanmax(np.abs(acceleration))),
}

with open(OUTPUT_DIR / "metrics_clean.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_plot(filename, title, ylabel, values):
    plt.figure(figsize=(12, 6))
    plt.plot(t, values)
    plt.xlabel("t [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


save_plot(
    "plot_path_clean.png",
    "Očiščena dolžina poti prijemne točke skozi čas",
    "pot [px]",
    path,
)

save_plot(
    "plot_speed_clean.png",
    "Očiščena hitrost prijemne točke skozi čas",
    "hitrost [px/s]",
    speed,
)

save_plot(
    "plot_acceleration_clean.png",
    "Očiščen pospešek prijemne točke skozi čas",
    "pospešek [px/s²]",
    np.abs(acceleration),
)

plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.gca().invert_yaxis()
plt.xlabel("x [px]")
plt.ylabel("y [px]")
plt.title("Očiščena trajektorija prijemne točke v sliki")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot_trajectory_clean.png", dpi=150)
plt.close()

print(f"Shranjeno v: {OUTPUT_DIR}")
print(json.dumps(metrics, indent=2, ensure_ascii=False))