"""Äpplen i enskilda OAK-D-bilder (RGB + djupbild som matchar RGB-bilden pixel för pixel, ingen video).

Varje bild analyseras för sig med samma delar som pipeline_3d.py: fine-tuned detektor, avstånd till varje äpple från
djupbilden, 3 m-gräns för bakgrunden, kontroll av hur synligt äpplet är, och storlek på helt synliga äpplen.
Ingen räkning över en passage: bilderna tas med några sekunders mellanrum från olika platser och överlappar, så
antalen per bild får inte läggas ihop till en summa för trädet.

Indata: en mapp med, per bild, <namn>_metadata.json (camera_intrinsics, rgb_file, depth_file, captured_at_local),
RGB-bilden och en djupbild (PNG, mm) som matchar RGB-bilden pixel för pixel (OAK-D-exportens format).

Användning (från mappen Main):
    python oak_d_snapshots.py <mapp med bilderna>              # resultat i ../Results/oak_d
    python oak_d_snapshots.py <mapp> --out ../mina_resultat
Standard: detektorn från run 2b (tränad på Orbbec + OAK-D), bildstorlek 1280, confidence 0,55.
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
from visualize import num, put_text

WEIGHTS = "models/apple_run2b.pt"  # run 2b: bäst på OAK-D (test-F1 0,73 mot 0,58 för run 1)
IMGSZ = 1280         # bästa bildstorleken för run 2b på OAK-D-bilderna (640x400)
CONF = 0.55          # bästa confidence för räkning med run 2b på OAK-D
MIN_VISIBLE = 0.55   # samma regel som kontrollen "tydligt synligt" som diskuterats för pipeline_3d
MEASURE_CONF = 0.6   # ersättning för "helt synligt" med en enda vy: hög confidence, rund, inte avskuren, inget framför


def plural(n, one, many):
    """Rätt form på svenska: plural(1, "äpple", "äpplen") -> "äpple"."""
    return one if n == 1 else many


def analyse_capture(model, rgb_path, depth_path, meta, calibration, imgsz=IMGSZ, conf=CONF):
    """Detektera, placera och (där det går) mät äpplena i en bild.

    Parametrar:
        model: YOLO-modell.
        rgb_path, depth_path: bildens färgbild och djupbild (PNG, mm, matchar färgbilden pixel för pixel).
        meta: bildens metadata.json (camera_intrinsics, ...).
        calibration: storlekskalibreringen från pipeline_3d (scale, median_raw_diameter_mm).
        imgsz, conf: detektorns bildstorlek och confidence.

    Returnerar:
        img (färgbilden), filled (djup med luckor fyllda) och detections-tabellen: en rad per äpple med avstånd,
        visible_share (rutans bredd / bredden av ett helt äpple på det avståndet), far, measured, diameter_mm och mass_g.
    """
    img = cv2.imread(str(rgb_path))
    H, W = img.shape[:2]
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth[(depth <= 0) | (depth > pipe3d.DEPTH_READ_MM)] = np.nan
    filled = pipe3d.fill_holes(depth)
    fx = float(meta["camera_intrinsics"][0][0])
    result = model.predict(img, classes=[pipe3d.APPLE_CLASS], conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes, scores = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
    keep = pipe3d.suppress_nested(boxes, scores)
    rows = []
    for (x1, y1, x2, y2), box_conf in zip(boxes[keep], scores[keep]):
        w, h = x2 - x1, y2 - y1
        size = (w + h) / 2
        z, source, occluded = pipe3d.box_depth(depth, filled, x1, y1, x2, y2)
        # hur stor del av ett helt äpples bredd som syns: rutans kortaste sida / väntad bredd för ett medeläpple på avståndet z
        visible = min(w, h) / (fx * calibration["median_raw_diameter_mm"] / z) if np.isfinite(z) else np.nan
        diameter = size * z / (fx - size / 2) * calibration["scale"] if source == "apple" else np.nan
        at_edge = bool(x1 < 3 or y1 < 3 or x2 > W - 3 or y2 > H - 3)
        far = bool(np.isfinite(z) and z > pipe3d.MAX_DEPTH_MM)
        measured = bool(source == "apple" and not far and not at_edge and box_conf >= MEASURE_CONF
                        and max(w, h) / max(min(w, h), 1) <= pipe3d.VIS_MAX_ASPECT
                        and np.isfinite(occluded) and occluded <= pipe3d.VIS_MAX_OCCLUDED
                        and np.isfinite(visible) and visible >= 0.8)
        rows.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, conf=float(box_conf), size_px=size, z_mm=z, depth_source=source,
                         occluded=occluded, visible_share=visible, far=far, at_edge=at_edge, measured=measured,
                         diameter_mm=diameter if measured else np.nan,
                         mass_g=pipe3d.sphere_mass_g(diameter) if measured else np.nan))
    return img, filled, pd.DataFrame(rows)


def draw(img, filled, detections, title):
    """Bilden med sina rutor (grön tydligt synligt, orange delvis dolt, cyan uppmätt, röd för långt bort) bredvid
    djupbilden i färg, med en rubrikrad."""
    vis = img.copy()
    for box in detections.itertuples():
        p1, p2 = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
        if box.far:
            cv2.rectangle(vis, p1, p2, (0, 0, 255), 1)
            continue
        hidden = np.isfinite(box.visible_share) and box.visible_share < MIN_VISIBLE
        cv2.rectangle(vis, p1, p2, (0, 165, 255) if hidden else (0, 255, 0), 2)
        if box.measured:
            cv2.rectangle(vis, (p1[0] - 2, p1[1] - 2), (p2[0] + 2, p2[1] + 2), (255, 255, 0), 1)
            put_text(vis, f"{box.diameter_mm:.0f} mm", (p1[0], p1[1] - 3), 0.38, (255, 255, 0))
    depth_vis = visualize.colorize_depth(filled, img.shape[1], img.shape[0], pipe3d.MAX_DEPTH_MM)
    out = np.hstack([vis, depth_vis])
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    put_text(out, title, (6, 17), 0.48)
    put_text(out, "grön tydligt synligt | orange delvis dolt | cyan uppmätt | röd för långt bort   |   djup: rött nära, blått 3 m, grått för långt",
             (6, out.shape[0] - 8), 0.38)
    return out


def summarize_capture(name, meta, img, detections):
    """En sammanfattningsrad per bild: äpplen (inom avståndsgränsen), tydligt synliga, uppmätta, ljusstyrka."""
    if len(detections):
        near = detections[~detections.far]
        clear = near[~(near.visible_share < MIN_VISIBLE)]
        measured = near[near.measured]
    else:
        near = clear = measured = detections
    brightness = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    return dict(capture=name, time=meta["captured_at_local"][11:19], apples=len(near), clearly_visible=len(clear),
                partly_hidden=len(near) - len(clear), too_far=int(detections.far.sum()) if len(detections) else 0,
                measured=len(measured),
                median_diameter_mm=round(float(measured.diameter_mm.median()), 1) if len(measured) else None,
                mean_mass_g=round(float(measured.mass_g.mean()), 1) if len(measured) else None,
                median_apple_distance_m=round(float(near.z_mm.median()) / 1000, 2) if len(near) else None,
                brightness=round(brightness), dark=brightness < 60)


def main():
    ap = argparse.ArgumentParser(description="Äpplen i enskilda OAK-D-bilder.")
    ap.add_argument("folder", type=Path, help="mapp med *_metadata.json, RGB-bilder och djupbilder")
    ap.add_argument("--out", type=Path, default=Path("../Results/oak_d"), help="mapp för resultaten")
    ap.add_argument("--weights", default=WEIGHTS, help="detektor (standard: run 2b)")
    ap.add_argument("--imgsz", type=int, default=IMGSZ, help=f"detektorns bildstorlek (standard {IMGSZ})")
    ap.add_argument("--conf", type=float, default=CONF, help=f"confidence för räkning (standard {CONF})")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((pipe3d.CALIBRATION_DIR / "calibration.json").read_text())
    model = YOLO(args.weights)
    summaries, all_detections, thumbs = [], [], []
    for meta_path in sorted(args.folder.glob("*_metadata.json")):
        meta = json.loads(meta_path.read_text())
        name = meta_path.name.replace("_metadata.json", "")
        img, filled, detections = analyse_capture(model, args.folder / meta["rgb_file"], args.folder / meta["depth_file"],
                                                  meta, calibration, args.imgsz, args.conf)
        s = summarize_capture(name, meta, img, detections)
        summaries.append(s)
        all_detections.append(detections.assign(capture=name))
        title = (f"{name} {s['time']}: {s['apples']} {plural(s['apples'], 'äpple', 'äpplen')} ({s['clearly_visible']} tydligt "
                 f"{plural(s['clearly_visible'], 'synligt', 'synliga')}), {s['too_far']} för långt bort, "
                 f"{s['measured']} {plural(s['measured'], 'uppmätt', 'uppmätta')}"
                 + (f", median {s['median_diameter_mm']:.0f} mm" if s["measured"] else "")
                 + ("   [mycket mörk bild]" if s["dark"] else ""))
        out = draw(img, filled, detections, title)
        cv2.imwrite(str(args.out / f"{name}_annotated.png"), out)
        thumbs.append(cv2.resize(out[:, : img.shape[1]], (480, 300)))
        print(title)
    if not summaries:
        print(f"Inga bilder (*_metadata.json) hittades i {args.folder}")
        return
    pd.DataFrame(summaries).to_csv(args.out / "summary.csv", index=False)
    all_detections = pd.concat(all_detections, ignore_index=True)
    all_detections.round(3).to_csv(args.out / "apples.csv", index=False)
    cv2.imwrite(str(args.out / "overview.png"), visualize.tile_grid(thumbs, 4))
    measured = all_detections[all_detections.measured] if len(all_detections) else all_detections
    median_mm = f"{measured.diameter_mm.median():.0f}" if len(measured) else "-"
    mean_g = f"{measured.mass_g.mean():.0f}" if len(measured) else "-"
    print(f"\n{len(summaries)} bilder | äpplen per bild, median {num(pd.DataFrame(summaries).apples.median(), 0)} | "
          f"{len(measured)} uppmätta äpplen: median {median_mm} mm, medel {mean_g} g")


if __name__ == "__main__":
    main()
