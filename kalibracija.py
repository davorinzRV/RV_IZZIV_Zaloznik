import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Mapa s kalibracijskimi slikami
root = Path("porocilo_slike/kalibracija")

# Velikost enega kvadrata šahovnice v mm
SQUARE_SIZE_MM = 20.0

# Poskusimo obe možnosti:
# Če info.md pomeni 9x6 notranjih vogalov -> (9, 6)
# Če info.md pomeni 9x6 polj -> notranjih vogalov je (8, 5)
PATTERN_CANDIDATES = [(9, 6), (8, 5)]

image_files = sorted(
    list(root.rglob("*.jpg")) +
    list(root.rglob("*.jpeg")) +
    list(root.rglob("*.png"))
)

if not image_files:
    print("Ni najdenih kalibracijskih slik.")
    raise SystemExit(1)

rows = []

for img_path in image_files:
    img = cv2.imread(str(img_path))

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found = False

    for pattern_size in PATTERN_CANDIDATES:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if not ret:
            continue

        found = True

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        nx, ny = pattern_size
        pts = corners.reshape(ny, nx, 2)

        distances_px = []

        # horizontalne razdalje med sosednjimi vogali
        for y in range(ny):
            for x in range(nx - 1):
                d = np.linalg.norm(pts[y, x + 1] - pts[y, x])
                distances_px.append(d)

        # vertikalne razdalje med sosednjimi vogali
        for y in range(ny - 1):
            for x in range(nx):
                d = np.linalg.norm(pts[y + 1, x] - pts[y, x])
                distances_px.append(d)

        distances_px = np.array(distances_px, dtype=float)

        mean_square_px = float(np.mean(distances_px))
        median_square_px = float(np.median(distances_px))
        px_per_mm_mean = mean_square_px / SQUARE_SIZE_MM
        px_per_mm_median = median_square_px / SQUARE_SIZE_MM

        rows.append({
            "folder": img_path.parent.name,
            "image": img_path.name,
            "pattern": f"{nx}x{ny}",
            "mean_square_px": mean_square_px,
            "median_square_px": median_square_px,
            "px_per_mm_mean": px_per_mm_mean,
            "px_per_mm_median": px_per_mm_median,
        })

        # shrani sliko z najdenimi vogali
        out_img = img.copy()
        cv2.drawChessboardCorners(out_img, pattern_size, corners, ret)

        out_dir = Path("porocilo_slike/kalibracija_detected")
        out_dir.mkdir(exist_ok=True)

        out_path = out_dir / f"{img_path.parent.name}_{img_path.stem}_detected.jpg"
        cv2.imwrite(str(out_path), out_img)

        break

    if not found:
        print("Ni najdene šahovnice:", img_path)

df = pd.DataFrame(rows)

if df.empty:
    print("Na nobeni sliki nisem našel šahovnice.")
    raise SystemExit(1)

df.to_csv("calibration_px_per_mm_all.csv", index=False)

summary = df.groupby("folder").agg({
    "px_per_mm_mean": ["mean", "std", "count"],
    "px_per_mm_median": ["mean", "std"]
}).reset_index()

summary.columns = [
    "folder",
    "px_per_mm_mean",
    "px_per_mm_std",
    "num_images",
    "px_per_mm_median_mean",
    "px_per_mm_median_std",
]

summary.to_csv("calibration_px_per_mm_summary.csv", index=False)

print("\nVse meritve:")
print(df)

print("\nPovzetek po kamerah/mapah:")
print(summary)

print("\nShranjeno:")
print("calibration_px_per_mm_all.csv")
print("calibration_px_per_mm_summary.csv")
print("porocilo_slike/kalibracija_detected/")
