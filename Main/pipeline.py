"""Apple yield pipeline: detect -> measure camera motion -> count by strip sampling -> size with depth -> yield.

Usage (from the Main folder):
    python pipeline.py ../Archive/test_images ../Archive/Tree2 --weights runs/detect/runs/apple_ft/weights/last.pt --conf-keep 0.45
    python pipeline.py ../Archive/Tree2 --mean-weight 94 --weight-source assumed     # fixed mean weight instead of image sizes

Writes results/<recording>/ with annotated.mp4 (ends with the final summary), final.png, counts.csv,
observations.csv and summary.json, and prints the summary in the terminal.

How the count works (no identity matching, so a flickering detection cannot create extra apples):
  1. Detect apples in every frame.
  2. Measure the camera motion between frames from the apples themselves (the shift most apple pairs agree on).
  3. Every K-th frame counts only the apples in the strip of the scene it is responsible for. The strips tile the
     scene (boundaries halfway between neighbouring sampled frames), so each apple is counted about once.
  4. The count is the average over several spacings K and start frames; their spread gives the plausible range.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Reference apple (hand measured 7.0 x 6.5 x 6.0 cm, 109 g) -> effective sphere density
REF_DIAMETER_MM, REF_MASS_G = 65.0, 109.0
DENSITY = REF_MASS_G / ((np.pi / 6) * (REF_DIAMETER_MM / 10) ** 3)

CONF_LOW = 0.08     # detections kept for the camera-motion vote and the video
CONF_KEEP = 0.15    # an apple counts if its detection reaches this confidence (0.15 pretrained, 0.45 fine-tuned)
Z_RANGE_MM = (500, 3500)
COUNT_ERROR = 0.10  # per-frame count error: 3% and 9% on the two held-out marked frames (fine-tuned model), rounded up
IMAGE_WEIGHT_ERROR = 0.30  # image-based weights ran 25-35% light against the 109 g reference apple
SPACINGS = (3, 5, 8)       # every K-th frame; the counts for these spacings and up to 4 start frames each are averaged
HOLD_SECONDS = 4
FPS = 15

MAX_SHIFT = 100     # px: largest camera motion between two frames
SHIFT_BIN = 4       # px: resolution of the shift vote
NESTED_IOMIN = 0.5  # a box with more than this share of its area inside a stronger box is a duplicate on the same apple


def align_depth_to_color(depth_mm, meta, z_min=300, z_max=6000):
    di, ci, ex = meta["depth_intrinsics"], meta["color_intrinsics"], meta["depth_to_color_extrinsics"]
    R = np.array(ex["rotation"], float).reshape(3, 3)
    t = np.array(ex["translation_mm"], float)
    v, u = np.nonzero((depth_mm > z_min) & (depth_mm < z_max))
    z = depth_mm[v, u].astype(np.float32)
    pts = np.stack([(u - di["cx"]) * z / di["fx"], (v - di["cy"]) * z / di["fy"], z], axis=1) @ R.T + t
    uc = np.round(pts[:, 0] * ci["fx"] / pts[:, 2] + ci["cx"]).astype(int)
    vc = np.round(pts[:, 1] * ci["fy"] / pts[:, 2] + ci["cy"]).astype(int)
    ok = (uc >= 0) & (uc < ci["width"]) & (vc >= 0) & (vc < ci["height"])
    out = np.full((ci["height"], ci["width"]), np.inf, np.float32)
    np.minimum.at(out, (vc[ok], uc[ok]), pts[ok, 2])
    out[np.isinf(out)] = np.nan
    return out


def suppress_nested(boxes, scores, iomin=NESTED_IOMIN):
    """Drop a box when most of it lies inside a more confident box. The detector often stacks 2-5 boxes on one apple,
    and ordinary NMS keeps them when their sizes differ. Real neighbouring apples overlap far less than this."""
    keep = []
    for i in np.argsort(-scores):
        x1, y1, x2, y2 = boxes[i]
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        dup = False
        for j in keep:
            a1, b1, a2, b2 = boxes[j]
            inter = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
            if inter / min(area, max((a2 - a1) * (b2 - b1), 1e-6)) > iomin:
                dup = True
                break
        if not dup:
            keep.append(i)
    return np.array(keep, dtype=int)


def detect_frames(rec, meta, model, imgsz, frames):
    """Run the detector on every frame. One row per detection, with distance and size."""
    fx = meta["color_intrinsics"]["fx"]
    rows = []
    for i, f in enumerate(frames):
        color = cv2.imread(str(f))
        H, W = color.shape[:2]
        depth = cv2.imread(str(rec / "depth" / (f.stem + ".png")), cv2.IMREAD_UNCHANGED)
        depth_c = align_depth_to_color(depth, meta)
        res = model.predict(color, classes=[47], conf=CONF_LOW, imgsz=imgsz, verbose=False)[0]
        boxes, scores = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
        keep = suppress_nested(boxes, scores)
        for (x1, y1, x2, y2), c in zip(boxes[keep], scores[keep]):
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            core = depth_c[int(cy - h / 4): int(cy + h / 4) + 1, int(cx - w / 4): int(cx + w / 4) + 1]
            zv = core[np.isfinite(core)]
            z = float(np.median(zv)) if len(zv) >= 4 else np.nan
            rows.append(dict(
                frame=i, conf=float(c), x1=x1, y1=y1, x2=x2, y2=y2, cx=cx, cy=cy, size=(w + h) / 2, z_mm=z,
                aspect=max(w, h) / max(min(w, h), 1),
                at_edge=bool(x1 < 5 or y1 < 5 or x2 > W - 5 or y2 > H - 5),
                diameter_mm=(w + h) / 2 * z / fx,
            ))
    return pd.DataFrame(rows)


def vote_shift(P, Q, prior):
    """Camera motion between two frames: the displacement most apple pairs agree on. None if there is no clear winner."""
    if len(P) < 4 or len(Q) < 4:
        return None
    D = (Q[None, :, :] - P[:, None, :]).reshape(-1, 2)
    if prior is not None:
        D = D[np.hypot(*(D - prior).T) < 40]  # motion is smooth: stay near the previous estimate
    D = D[np.hypot(*D.T) < MAX_SHIFT]
    if len(D) < 5:
        return None
    edges = np.arange(-MAX_SHIFT, MAX_SHIFT + SHIFT_BIN, SHIFT_BIN)
    H, _, _ = np.histogram2d(D[:, 0], D[:, 1], bins=[edges, edges])
    H = cv2.GaussianBlur(H.astype(np.float32), (0, 0), 1.0) * (2 * np.pi)
    ix, iy = np.unravel_index(np.argmax(H), H.shape)
    if H[ix, iy] < 5:
        return None
    peak = np.array([edges[ix] + SHIFT_BIN / 2, edges[iy] + SHIFT_BIN / 2])
    return D[np.hypot(*(D - peak).T) < 2 * SHIFT_BIN].mean(axis=0)


def camera_positions(det, n_frames):
    """Cumulative camera shift per frame (scene position = image position + shift)."""
    good = {f: g[g.conf >= CONF_KEEP][["cx", "cy"]].to_numpy() for f, g in det.groupby("frame")}
    empty = np.zeros((0, 2))
    shift, last, lost = [np.zeros(2)], None, 0
    for i in range(1, n_frames):
        d = vote_shift(good.get(i - 1, empty), good.get(i, empty), last)
        if d is None:
            d = last if last is not None else np.zeros(2)  # no clear vote: keep moving the same way
            lost += 1
        else:
            last = d
        shift.append(shift[-1] - d)
    return np.array(shift), lost


def strip_sample(det, shift, W, k, offset):
    """Detections counted by every k-th frame (starting at `offset`), each only inside its own strip of the scene.

    A detection counts in the sampled frame whose centre is nearest to it in the scene, so the strips tile the scene
    whichever way the camera moves. (Before 2026-10-05 the strips assumed the camera moved one way only; for the
    other direction most strips came out empty and the count was wrong.)"""
    frames = np.arange(offset, len(shift), k)
    d = det[(det.conf >= CONF_KEEP) & det.frame.isin(frames)].copy()
    d["scene_x"] = d.cx + shift[d.frame.to_numpy(), 0]
    centers = W / 2 + shift[frames, 0]
    own = np.searchsorted(frames, d.frame.to_numpy())
    nearest = np.abs(d.scene_x.to_numpy()[:, None] - centers[None, :]).argmin(axis=1)
    return d[nearest == own]


def count_apples(det, shift, W):
    """Average count over several spacings and start frames, plus the spread."""
    table = [dict(k=k, offset=off, count=len(strip_sample(det, shift, W, k, off))) for k in SPACINGS for off in range(min(k, 4))]
    t = pd.DataFrame(table)
    return t, float(t["count"].mean()), int(t["count"].min()), int(t["count"].max())


def summarize(rec, n_frames, det, shift, lost, W, mean_weight_g, weight_source="manual value", weight_error=0.0):
    table, mean_n, lo_n, hi_n = count_apples(det, shift, W)
    typical = table.iloc[(table["count"] - mean_n).abs().argmin()]  # the sampling variant closest to the average: shown in the video
    counted = strip_sample(det, shift, W, int(typical.k), int(typical.offset))
    good = det[(det.conf >= CONF_KEEP) & (det.aspect < 1.3) & ~det.at_edge & det.z_mm.between(*Z_RANGE_MM)]
    image_mean = float(((np.pi / 6) * (good.diameter_mm / 10) ** 3 * DENSITY).mean())

    if mean_weight_g:
        mean_g, w_err, source = mean_weight_g, weight_error, weight_source
    else:
        mean_g, w_err, source = image_mean, IMAGE_WEIGHT_ERROR, "image sizes"
    n = int(round(mean_n))
    count_low, count_high = int(round(lo_n * (1 - COUNT_ERROR))), int(round(hi_n * (1 + COUNT_ERROR)))
    summary = dict(
        recording=rec.name, frames=n_frames, camera_pan_px=float(abs(shift[-1, 0] - shift[0, 0])),
        frames_without_clear_camera_motion=int(lost), apples_counted=n, count_low=count_low, count_high=count_high,
        count_by_spacing={f"every_{k}th_frame": float(g["count"].mean()) for k, g in table.groupby("k")},
        median_diameter_mm=float(good.diameter_mm.median()), mean_weight_g=mean_g, weight_source=source,
        weight_error=w_err, estimate_kg=n * mean_g / 1000,
        kg_low=count_low * mean_g * (1 - w_err) / 1000, kg_high=count_high * mean_g * (1 + w_err) / 1000,
        estimate_kg_from_image_sizes=n * image_mean / 1000, conf_keep=CONF_KEEP, count_error_assumed=COUNT_ERROR,
        note=f"plausible range, not a statistical interval: lowest and highest count over {len(table)} strip-sampling variants, "
             f"widened by the per-frame counting error ({COUNT_ERROR:.0%}); weight +/-{w_err:.0%}",
    )
    return summary, table, counted


def put_text(img, text, org, scale, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def summary_panel(img, s, name):
    """Darken the image and write the final result on it."""
    H, W = img.shape[:2]
    out = (img * 0.35).astype(np.uint8)
    cv2.rectangle(out, (W // 2 - 420, H // 2 - 190), (W // 2 + 420, H // 2 + 190), (20, 20, 20), -1)
    cv2.rectangle(out, (W // 2 - 420, H // 2 - 190), (W // 2 + 420, H // 2 + 190), (200, 200, 200), 2)
    x, y = W // 2 - 380, H // 2 - 120
    put_text(out, name, (x, y), 0.8, (200, 200, 200))
    put_text(out, f"{s['apples_counted']} apples counted", (x, y + 70), 1.6, (120, 255, 120), 3)
    put_text(out, f"plausible range {s['count_low']} - {s['count_high']}", (x, y + 105), 0.8, (200, 200, 200))
    put_text(out, f"Estimated yield: {s['estimate_kg']:.1f} kg", (x, y + 160), 1.3, (255, 255, 255), 2)
    put_text(out, f"plausible range {s['kg_low']:.1f} - {s['kg_high']:.1f} kg", (x, y + 195), 0.8, (200, 200, 200))
    put_text(out, f"mean apple {s['mean_weight_g']:.0f} g ({s['weight_source']}, +/-{s['weight_error']:.0%})", (x, y + 235), 0.8, (200, 200, 200))
    put_text(out, "Visible apples only, one side of the tree", (x, y + 275), 0.65, (150, 150, 150))
    return out


def render(frames, det, counted, summary, out_dir, name):
    """Draw every confident detection (gray), highlight the ones counted in this frame (green), end with the summary."""
    first = cv2.imread(str(frames[0]))
    H, W = first.shape[:2]
    video = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    shown = {f: g for f, g in det[det.conf >= CONF_KEEP].groupby("frame")}
    running, vis = 0, first
    for i, f in enumerate(frames):
        vis = cv2.imread(str(f))
        c = counted[counted.frame == i]
        running += len(c)
        for r in shown.get(i, pd.DataFrame()).itertuples():
            cv2.rectangle(vis, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (150, 150, 150), 1)
        for r in c.itertuples():
            cv2.rectangle(vis, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 255, 0), 2)
        cv2.rectangle(vis, (0, 0), (W, 34), (0, 0, 0), -1)
        put_text(vis, f"frame {i}   counted so far: {running}   (green = counted in this frame, gray = counted in another)", (10, 24), 0.6)
        video.write(vis)
    final = summary_panel(vis, summary, name)
    cv2.imwrite(str(out_dir / "final.png"), final)
    for _ in range(HOLD_SECONDS * FPS):
        video.write(final)
    video.release()


def print_summary(s):
    bar = "=" * 56
    print(bar)
    print(f" {s['recording']}")
    print(f"   Apples counted : {s['apples_counted']}   (plausible range {s['count_low']} - {s['count_high']})")
    print(f"   Estimated yield: {s['estimate_kg']:.1f} kg   (plausible range {s['kg_low']:.1f} - {s['kg_high']:.1f} kg)")
    print(f"   Mean apple     : {s['mean_weight_g']:.0f} g ({s['weight_source']})")
    print(f"   Median diameter: {s['median_diameter_mm']:.0f} mm   | camera pan {s['camera_pan_px']:.0f} px")
    print("   Count by spacing: " + ", ".join(f"every {k.split('_')[1]}: {v:.0f}" for k, v in s["count_by_spacing"].items()))
    print(bar)


def process(rec, out_root, model, imgsz, mean_weight_g=None, weight_source="manual value", weight_error=0.0):
    meta = json.loads((rec / "meta.json").read_text())
    out = out_root / rec.name
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted((rec / "color").glob("*.jpg"))
    W = cv2.imread(str(frames[0])).shape[1]

    det = detect_frames(rec, meta, model, imgsz, frames)
    shift, lost = camera_positions(det, len(frames))
    summary, table, counted = summarize(rec, len(frames), det, shift, lost, W, mean_weight_g, weight_source, weight_error)
    det.assign(scene_x=det.cx + shift[det.frame.to_numpy(), 0]).to_csv(out / "observations.csv", index=False)
    table.to_csv(out / "counts.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    render(frames, det, counted, summary, out, rec.name)
    return summary


def main():
    global CONF_LOW, CONF_KEEP, COUNT_ERROR
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+", type=Path)
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--mean-weight", type=float, default=None, help="mean apple weight in grams (overrides the image-based weight)")
    ap.add_argument("--weight-source", default="manual value", help='label shown with the weight, e.g. "assumed" or "weighed sample, 10 apples"')
    ap.add_argument("--weight-error", type=float, default=0.20, help="relative uncertainty of --mean-weight (0.20 = +/-20%%)")
    ap.add_argument("--conf-low", type=float, default=None, help=f"detections kept for the video and camera-motion vote (default {CONF_LOW})")
    ap.add_argument("--conf-keep", type=float, default=None, help=f"an apple counts at this confidence (default {CONF_KEEP}; use 0.45 for the fine-tuned model)")
    ap.add_argument("--count-error", type=float, default=None, help=f"per-frame counting error used for the range (default {COUNT_ERROR})")
    args = ap.parse_args()

    CONF_LOW = args.conf_low if args.conf_low is not None else CONF_LOW
    CONF_KEEP = args.conf_keep if args.conf_keep is not None else CONF_KEEP
    COUNT_ERROR = args.count_error if args.count_error is not None else COUNT_ERROR
    model = YOLO(args.weights)
    for rec in args.recordings:
        print_summary(process(rec, args.out, model, args.imgsz, args.mean_weight, args.weight_source, args.weight_error))


if __name__ == "__main__":
    main()
