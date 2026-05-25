from pathlib import Path
import pandas as pd

input_csv = Path("tabela_porocilo.csv")
output_csv = Path("tabela_porocilo_mm.csv")

PX_PER_MM = 1.356  # cam_mid_resized2

df = pd.read_csv(input_csv)

df["Pot [mm]"] = df["Pot [px]"] / PX_PER_MM

df["Povp. hitrost [mm/s]"] = df["Povp. hitrost [px/s]"] / PX_PER_MM
df["Mediana hitrosti [mm/s]"] = df["Mediana hitrosti [px/s]"] / PX_PER_MM
df["P95 hitrost [mm/s]"] = df["P95 hitrost [px/s]"] / PX_PER_MM
df["Maks. hitrost [mm/s]"] = df["Maks. hitrost [px/s]"] / PX_PER_MM

df["Povp. pospešek [mm/s²]"] = df["Povp. pospešek [px/s²]"] / PX_PER_MM
df["P95 pospešek [mm/s²]"] = df["P95 pospešek [px/s²]"] / PX_PER_MM
df["Maks. pospešek [mm/s²]"] = df["Maks. pospešek [px/s²]"] / PX_PER_MM

cols = [
    "Pot [mm]",
    "Povp. hitrost [mm/s]",
    "Mediana hitrosti [mm/s]",
    "P95 hitrost [mm/s]",
    "Maks. hitrost [mm/s]",
    "Povp. pospešek [mm/s²]",
    "P95 pospešek [mm/s²]",
    "Maks. pospešek [mm/s²]",
]

df[cols] = df[cols].round(1)

df.to_csv(output_csv, index=False)

print(df)
print()
print("Shranjeno:", output_csv)
print("Uporabljen faktor:", PX_PER_MM, "px/mm")
