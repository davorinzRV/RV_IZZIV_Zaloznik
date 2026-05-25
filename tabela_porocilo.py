from pathlib import Path
import pandas as pd

input_csv = Path("results/summary_metrics.csv")
out_csv = Path("summary_table_for_report.csv")
out_md = Path("summary_table_for_report.md")

df = pd.read_csv(input_csv)

# Izberemo samo stolpce, ki so uporabni za poročilo
cols = [
    "video",
    "frames",
    "duration_s",
    "valid_rate_raw",
    "valid_rate_after_cleaning",
    "interpolated_frames",
    "total_path_px",
    "mean_speed_px_s",
    "median_speed_px_s",
    "p95_speed_px_s",
    "max_speed_px_s",
    "mean_acceleration_px_s2",
    "p95_acceleration_px_s2",
    "max_acceleration_px_s2",
]

df = df[cols].copy()

# Lepša imena stolpcev za poročilo
df = df.rename(columns={
    "video": "Video",
    "frames": "Št. slik",
    "duration_s": "Trajanje [s]",
    "valid_rate_raw": "Zaznava pred čiščenjem [%]",
    "valid_rate_after_cleaning": "Zaznava po čiščenju [%]",
    "interpolated_frames": "Interpolirane slike",
    "total_path_px": "Pot [px]",
    "mean_speed_px_s": "Povp. hitrost [px/s]",
    "median_speed_px_s": "Mediana hitrosti [px/s]",
    "p95_speed_px_s": "P95 hitrost [px/s]",
    "max_speed_px_s": "Maks. hitrost [px/s]",
    "mean_acceleration_px_s2": "Povp. pospešek [px/s²]",
    "p95_acceleration_px_s2": "P95 pospešek [px/s²]",
    "max_acceleration_px_s2": "Maks. pospešek [px/s²]",
})

# Pretvorba deležev v procente
df["Zaznava pred čiščenjem [%]"] = df["Zaznava pred čiščenjem [%]"] * 100
df["Zaznava po čiščenju [%]"] = df["Zaznava po čiščenju [%]"] * 100

# Za poročilo zaokrožimo vrednosti
round_cols = {
    "Trajanje [s]": 2,
    "Zaznava pred čiščenjem [%]": 2,
    "Zaznava po čiščenju [%]": 2,
    "Pot [px]": 1,
    "Povp. hitrost [px/s]": 1,
    "Mediana hitrosti [px/s]": 1,
    "P95 hitrost [px/s]": 1,
    "Maks. hitrost [px/s]": 1,
    "Povp. pospešek [px/s²]": 1,
    "P95 pospešek [px/s²]": 1,
    "Maks. pospešek [px/s²]": 1,
}

df = df.round(round_cols)

# Shranimo kot CSV in Markdown tabelo
df.to_csv(out_csv, index=False)
out_md.write_text(df.to_markdown(index=False), encoding="utf-8")

print("Tabela za poročilo:")
print(df)
print()
print("Shranjeno:")
print(out_csv)
print(out_md)