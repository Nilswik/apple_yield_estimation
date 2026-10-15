"""Apples in single OAK-D snapshots (rgb + aligned depth per capture, no video).

Each capture is analysed on its own with the same parts as pipeline_3d.py: fine-tuned detector, distance per apple
from the aligned depth, 3 m background limit, clearly-visible check, and size of fully visible apples.
There is no pass counting: the captures are seconds apart from different spots and overlap, so the per-image
counts must not be added up into a tree total.

Input: a folder with, per capture, <name>_metadata.json (camera_intrinsics, rgb_file, depth_file, captured_at_local),
the RGB image and a depth PNG in mm aligned to the RGB image (the format of the OAK-D export).

Usage (from the Main folder):
    python oak_d_snapshots.py <folder with the captures>              # results in ../OAK_D_RES
    python oak_d_snapshots.py <folder> --out ../my_results
Defaults: run 2b detector (trained on Orbbec + OAK-D), image size 1280, confidence 0.55.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import pipeline as P2
import pipeline_3d as P3

WEIGHTS = "runs/detect/runs/apple_ft_run2b/weights/last.pt"  # run 2b: best on OAK-D (test F1 0.73 vs 0.58 for run 1)
IMGSZ = 1280         # best size for run 2b on the 640x400 OAK-D captures
CONF = 0.55          # best counting confidence for run 2b on OAK-D
MIN_VISIBLE = 0.55   # same rule as the clearly-visible check discussed for pipeline_3d
MEASURE_CONF = 0.6   # single-view stand-in for "fully visible": confident, round, not cut off, nothing in front


def analyse_capture(model, rgb_path, depth_path, meta, cal, imgsz=IMGSZ, conf=CONF):
    img = cv2.imread(str(rgb_path))
    H, W = img.shape[:2]
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth[(depth <= 0) | (depth > P3.DEPTH_READ_MM)] = np.nan
    filled = P3.fill_holes(depth)
    fx = float(meta["camera_intrinsics"][0][0])
    res = model.predict(img, classes=[47], conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes, scores = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
    keep = P2.suppress_nested(boxes, scores)
    rows = []
    for (x1, y1, x2, y2), c in zip(boxes[keep], scores[keep]):
        w, h = x2 - x1, y2 - y1
        size = (w + h) / 2
        z, source, occluded = P3.box_depth(depth, filled, x1, y1, x2, y2)
        visible = min(w, h) / (fx * cal["median_raw_diameter_mm"] / z) if np.isfinite(z) else np.nan
        diameter = size * z / (fx - size / 2) * cal["scale"] if source == "apple" else np.nan
        at_edge = bool(x1 < 3 or y1 < 3 or x2 > W - 3 or y2 > H - 3)
        far = bool(np.isfinite(z) and z > P3.MAX_DEPTH_MM)
        measured = bool(source == "apple" and not far and not at_edge and c >= MEASURE_CONF
                        and max(w, h) / max(min(w, h), 1) <= P3.VIS_MAX_ASPECT
                        and np.isfinite(occluded) and occluded <= P3.VIS_MAX_OCCLUDED
                        and np.isfinite(visible) and visible >= 0.8)
        rows.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, conf=float(c), size_px=size, z_mm=z, depth_source=source,
                         occluded=occluded, visible_share=visible, far=far, at_edge=at_edge, measured=measured,
                         diameter_mm=diameter if measured else np.nan,
                         mass_g=P3.DENSITY * (np.pi / 6) * (diameter / 10) ** 3 if measured else np.nan))
    return img, filled, pd.DataFrame(rows)


def draw(img, filled, det, title):
    vis = img.copy()
    for b in det.itertuples():
        p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        if b.far:
            cv2.rectangle(vis, p1, p2, (0, 0, 255), 1)
            continue
        hidden = np.isfinite(b.visible_share) and b.visible_share < MIN_VISIBLE
        cv2.rectangle(vis, p1, p2, (0, 165, 255) if hidden else (0, 255, 0), 2)
        if b.measured:
            cv2.rectangle(vis, (p1[0] - 2, p1[1] - 2), (p2[0] + 2, p2[1] + 2), (255, 255, 0), 1)
            P2.put_text(vis, f"{b.diameter_mm:.0f}mm", (p1[0], p1[1] - 3), 0.38, (255, 255, 0))
    depth_vis = P3.colorize_depth(filled, img.shape[1], img.shape[0])
    out = np.hstack([vis, depth_vis])
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    P2.put_text(out, title, (6, 17), 0.48)
    P2.put_text(out, "green clearly visible | orange partly hidden | cyan measured | red too far   |   depth: red near, blue 3 m, gray too far",
                (6, out.shape[0] - 8), 0.38)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=Path("../OAK_D_RES"))
    ap.add_argument("--weights", default=WEIGHTS, help="detector (default: run 2b)")
    ap.add_argument("--imgsz", type=int, default=IMGSZ, help=f"detector image size (default {IMGSZ})")
    ap.add_argument("--conf", type=float, default=CONF, help=f"counting confidence (default {CONF})")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cal = json.loads(Path("3d_annotated/calibration/calibration.json").read_text())
    model = YOLO(args.weights)
    summary, all_det, thumbs = [], [], []
    for meta_path in sorted(args.folder.glob("*_metadata.json")):
        meta = json.loads(meta_path.read_text())
        name = meta_path.name.replace("_metadata.json", "")
        img, filled, det = analyse_capture(model, args.folder / meta["rgb_file"], args.folder / meta["depth_file"], meta, cal, args.imgsz, args.conf)
        near = det[~det.far] if len(det) else det
        clear = near[~(near.visible_share < MIN_VISIBLE)] if len(near) else near
        m = near[near.measured] if len(near) else near
        brightness = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
        s = dict(capture=name, time=meta["captured_at_local"][11:19], apples=len(near), clearly_visible=len(clear),
                 partly_hidden=len(near) - len(clear), too_far=int(det.far.sum()) if len(det) else 0, measured=len(m),
                 median_diameter_mm=round(float(m.diameter_mm.median()), 1) if len(m) else None,
                 mean_mass_g=round(float(m.mass_g.mean()), 1) if len(m) else None,
                 median_apple_distance_m=round(float(near.z_mm.median()) / 1000, 2) if len(near) else None,
                 brightness=round(brightness), dark=brightness < 60)
        summary.append(s)
        all_det.append(det.assign(capture=name))
        title = (f"{name} {s['time']}: {s['apples']} apples ({s['clearly_visible']} clearly visible), {s['too_far']} too far, "
                 f"{s['measured']} measured" + (f", median {s['median_diameter_mm']:.0f} mm" if s["measured"] else "")
                 + ("   [very dark image]" if s["dark"] else ""))
        out = draw(img, filled, det, title)
        cv2.imwrite(str(args.out / f"{name}_annotated.png"), out)
        thumbs.append(cv2.resize(out[:, : img.shape[1]], (480, 300)))
        print(title)
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    pd.concat(all_det, ignore_index=True).round(3).to_csv(args.out / "apples.csv", index=False)
    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    cv2.imwrite(str(args.out / "overview.png"), np.vstack([np.hstack(thumbs[i:i + 4]) for i in range(0, len(thumbs), 4)]))
    allm = pd.concat(all_det)
    allm = allm[allm.measured]
    print(f"\n{len(summary)} captures | apples per capture median {pd.DataFrame(summary).apples.median():.0f} | "
          f"{len(allm)} measured apples: median {allm.diameter_mm.median():.0f} mm, mean {allm.mass_g.mean():.0f} g")


if __name__ == "__main__":
    main()
