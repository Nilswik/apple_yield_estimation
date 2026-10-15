"""Äpplen i enskilda OAK-D-bilder (färgbild + djupbild som matchar den pixel för pixel).

Varje bild analyseras för sig med samma delar som pipeline_3d.py: detektor, avstånd, 3 m-gräns, hur synligt
äpplet är och storlek. Bilderna överlappar, så antalen per bild får inte läggas ihop till ett träd.

Indata: en mapp med <namn>_metadata.json, färgbild och djupbild (PNG, mm) per bild, som OAK-D exporterar dem.

Användning (från mappen Main):
    python oak_d_snapshots.py <mapp med bilderna>      # resultat i ../Results/oak_d
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import pipeline_3d as pipe3d
import visualize
from visualize import put_text

WEIGHTS = "models/apple_run2b.pt"  # run 2b: bäst på OAK-D (F1 0,73 mot 0,58 för run 1)
IMGSZ = 1280
CONF = 0.55
MIN_VISIBLE = 0.55                 # tydligt synligt: minst 55 % av ett helt äpples bredd syns
MEASURE_CONF = 0.6                 # för att mätas: hög confidence, rund, inte avskuren, inget framför

ORANGE = (0, 165, 255)


def plural(n, one, many):
    return one if n == 1 else many


def analyse_capture(model, meta, folder, calibration):
    """Detektera och mät äpplena i en bild. Returnerar (färgbild, djupbild, tabell med en rad per äpple)."""
    img = cv2.imread(str(folder / meta["rgb_file"]))
    H, W = img.shape[:2]
    fx = float(meta["camera_intrinsics"][0][0])

    depth = cv2.imread(str(folder / meta["depth_file"]), cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth[(depth <= 0) | (depth > pipe3d.DEPTH_READ_MM)] = np.nan
    filled = pipe3d.fill_holes(depth)

    result = model.predict(img, classes=[pipe3d.APPLE_CLASS], conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    keep = pipe3d.suppress_nested(boxes, scores)

    rows = []
    for (x1, y1, x2, y2), conf in zip(boxes[keep], scores[keep]):
        w, h = x2 - x1, y2 - y1
        size = (w + h) / 2
        z, source, occluded = pipe3d.box_depth(depth, filled, x1, y1, x2, y2)

        # 1. hur synligt: rutans kortaste sida / väntad bredd för ett medelstort äpple på avståndet z
        # ? ett delvis dolt äpple ger en smal ruta, även om detektorn är säker
        visible = min(w, h) / (fx * calibration["median_raw_diameter_mm"] / z) if np.isfinite(z) else np.nan
        # 2. storlek i mm, samma formel och kalibrering som pipeline_3d
        # ! storleksskalan är anpassad på Orbbec-kameran; den kan behöva kontrolleras för OAK-D
        diameter = size * z / (fx - size / 2) * calibration["scale"] if source == "apple" else np.nan
        at_edge = bool(x1 < 3 or y1 < 3 or x2 > W - 3 or y2 > H - 3)
        far = bool(np.isfinite(z) and z > pipe3d.MAX_DEPTH_MM)

        # 3. mäts bara om äpplet är rent: eget djup, nära, helt i bild, säkert, runt, inte skymt
        # ? pipeline_3d följer äpplet över flera bildrutor; här finns bara en bild, så kraven gäller den
        measured = bool(
            source == "apple" and not far and not at_edge and conf >= MEASURE_CONF
            and max(w, h) / max(min(w, h), 1) <= pipe3d.VIS_MAX_ASPECT
            and np.isfinite(occluded) and occluded <= pipe3d.VIS_MAX_OCCLUDED
            and np.isfinite(visible) and visible >= 0.8
        )

        rows.append(dict(
            x1=x1, y1=y1, x2=x2, y2=y2, conf=float(conf), size_px=size,
            z_mm=z, depth_source=source, occluded=occluded, visible_share=visible,
            far=far, at_edge=at_edge, measured=measured,
            diameter_mm=diameter if measured else np.nan,
            mass_g=pipe3d.sphere_mass_g(diameter) if measured else np.nan,
        ))

    return img, filled, pd.DataFrame(rows)


def summarize_capture(name, meta, detections):
    """En rad per bild: äpplen inom avståndsgränsen, hur många som syns tydligt och hur många som mättes."""
    # ! bilderna överlappar: antalen per bild får inte läggas ihop till ett träd
    if len(detections):
        near = detections[~detections.far]
        clear = near[~(near.visible_share < MIN_VISIBLE)]
        measured = near[near.measured]
    else:
        near = clear = measured = detections

    return dict(
        capture=name,
        time=meta["captured_at_local"][11:19],
        apples=len(near),
        clearly_visible=len(clear),
        too_far=int(detections.far.sum()) if len(detections) else 0,
        measured=len(measured),
        median_diameter_mm=round(float(measured.diameter_mm.median()), 1) if len(measured) else None,
    )


def draw(img, filled, detections, title):
    """Bilden med rutorna bredvid djupbilden i färg."""
    vis = img.copy()
    for box in detections.itertuples():
        if box.far:
            visualize.draw_box(vis, box, visualize.RED)
            continue

        hidden = np.isfinite(box.visible_share) and box.visible_share < MIN_VISIBLE
        visualize.draw_box(vis, box, ORANGE if hidden else visualize.GREEN, 2)
        if box.measured:
            visualize.draw_box(vis, box, visualize.CYAN, pad=2)
            put_text(vis, f"{box.diameter_mm:.0f} mm", (int(box.x1), int(box.y1) - 3), 0.38, visualize.CYAN)

    out = np.hstack([vis, visualize.colorize_depth(filled, img.shape[1], img.shape[0], pipe3d.MAX_DEPTH_MM)])
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    put_text(out, title, (6, 17), 0.48)
    put_text(out, "grön tydligt synligt | orange delvis dolt | cyan uppmätt | röd för långt bort   |   "
                  "djup: rött nära, blått 3 m, grått för långt", (6, out.shape[0] - 8), 0.38)
    return out


def main():
    ap = argparse.ArgumentParser(description="Äpplen i enskilda OAK-D-bilder.")
    ap.add_argument("folder", type=Path, help="mapp med OAK-D-bilderna")
    ap.add_argument("--out", type=Path, default=Path("../Results/oak_d"), help="mapp för resultaten")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((pipe3d.CALIBRATION_DIR / "calibration.json").read_text())
    model = YOLO(WEIGHTS)

    summaries, all_detections, thumbs = [], [], []
    for meta_path in sorted(args.folder.glob("*_metadata.json")):
        meta = json.loads(meta_path.read_text())
        name = meta_path.name.replace("_metadata.json", "")

        img, filled, detections = analyse_capture(model, meta, args.folder, calibration)
        s = summarize_capture(name, meta, detections)
        summaries.append(s)
        all_detections.append(detections.assign(capture=name))

        title = (f"{name} {s['time']}: {s['apples']} {plural(s['apples'], 'äpple', 'äpplen')}, "
                 f"{s['clearly_visible']} tydligt {plural(s['clearly_visible'], 'synligt', 'synliga')}, "
                 f"{s['measured']} {plural(s['measured'], 'uppmätt', 'uppmätta')}")
        print(title)

        out = draw(img, filled, detections, title)
        cv2.imwrite(str(args.out / f"{name}_annotated.png"), out)
        thumbs.append(cv2.resize(out[:, : img.shape[1]], (480, 300)))

    if not summaries:
        print(f"Inga bilder (*_metadata.json) hittades i {args.folder}")
        return

    pd.DataFrame(summaries).to_csv(args.out / "summary.csv", index=False)
    pd.concat(all_detections, ignore_index=True).round(3).to_csv(args.out / "apples.csv", index=False)
    cv2.imwrite(str(args.out / "overview.png"), visualize.tile_grid(thumbs, 4))


if __name__ == "__main__":
    main()
