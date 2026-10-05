"""3D apple yield pipeline: the 2D pipeline (pipeline.py) plus the depth data.

Depth is used for four things:
  1. Distance limit: detections farther than MAX_DEPTH_MM (3.0 m) are background (dishes, buckets, rakes, the next
     tree) and are not counted. Detections without a depth reading are kept.
  2. Depth-aware counting: when the camera travels sideways, near apples cross the image faster than far ones.
     Measured on these recordings: apple motion = 1 + beta x (tree distance / apple distance - 1), beta about 0.8.
     Each apple's strip is scaled by its own distance, so near and far apples are each counted about once.
     Measured: almost no double counts (0-1 per pass). Inside the hand-marked views the count matches the hand count
     within 0-7%. (A +-2 frame memory, WINDOW, is available but off: it counted 16% more than people do.)
  3. Size: apples that stay fully visible over several frames (confident, round, steady size, nothing in front of
     them in depth) are measured in mm from their box size and distance.
  4. Calibration: one hand-weighed reference apple (109 g, 7.0 x 6.5 x 6.0 cm, average size on tree 1) fixes the size
     scale. The scale is fitted on the tree-1 recordings that are not used for testing (--calibrate) and then applied
     to every recording. Yield = apple count x mean weight of that recording's fully visible apples.

Usage (from the Main folder):
    python pipeline_3d.py --calibrate      # once: fits the size scale -> 3d_annotated/calibration/
    python pipeline_3d.py ../Archive/test_images ../Archive/Tree2 ../Archive/1dceb275-6e92-4368-9df5-1526972df094

Writes 3d_annotated/<recording>/: annotated_3d.mp4 (ends with the summary), final.png, measured_apples.png,
apples_measured.csv, counts.csv, detections.csv and summary.json. Add --reuse to skip detection when
detections.csv already exists (only the drawing and the numbers are redone).
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

import pipeline as P2  # shared 2D parts: duplicate-box suppression, camera-motion vote, depth reprojection

WEIGHTS = "runs/detect/runs/apple_ft/weights/last.pt"
IMGSZ = 1920
CONF_LOW = 0.2       # detections used to follow an apple between frames
CONF_KEEP = 0.45     # detections that count (fine-tuned model)
MAX_DEPTH_MM = 3000  # farther than this is background
DEPTH_READ_MM = 10000  # depth beyond this is not trusted at all
COUNT_ERROR = 0.10   # per-frame count error on the held-out marked frames (7% with the distance limit), rounded up
SPACINGS = (3, 5, 8)  # every K-th frame; 4 start frames each
WINDOW = 0            # frames before/after a sampled frame that also report its apples (0 = off). 2 rescued apples the
                      # detector blinked on, but inside the hand-marked views it counted 16% more than the hand count
                      # (it adds half-hidden apples people do not count), while one frame matched within 0-7%.
WINDOW_MIN_FRAMES = 2  # an apple counts when seen in at least this many of those frames
FPS = 15
HOLD_SECONDS = 5
OUT = Path("3d_annotated")
ARCHIVE = Path("../Archive")

# Reference apple, hand measured on tree 1: 7.0 x 6.5 x 6.0 cm, 109 g, "average size of the apples on the tree"
REF_DIAMETER_MM = 65.0
REF_MASS_G = 109.0
DENSITY = REF_MASS_G / ((np.pi / 6) * (REF_DIAMETER_MM / 10) ** 3)  # g/cm3 for a sphere of the mean diameter
# Tree-1 recordings used to fit the size scale. test_images (tree 1) and Tree2 + 1dceb275 (both tree 2) are tests.
CALIBRATION_RECORDINGS = ["1aa65ece-d32f-4254-bcd3-80b26d2db7b9", "341f04ca-054a-4e52-ae96-139378821ea8",
                          "5fa2faba-e6d8-410e-803e-a52e8a0d9089", "849d204a-348a-4494-b9c9-a9e0d3903359",
                          "Annotation_tetst", "d4d77803-a829-42b5-8eb2-a4c86f31e99a"]
TREE_OF = {"test_images": "tree 1", "Tree2": "tree 2", "1dceb275-6e92-4368-9df5-1526972df094": "tree 2"}

# "Fully visible": followed over several frames, confident, round, steady size, nothing in front of it in depth
VIS_WINDOW = 6           # frames before and after the counted frame
VIS_MIN_VIEWS = 7
VIS_MIN_CONF = 0.6
VIS_MAX_ASPECT = 1.2
VIS_MAX_SIZE_CV = 0.10   # box size may vary at most 10% between views (a leaf moving over it changes the size)
VIS_MIN_SIZE_PX = 16
VIS_MIN_DEPTH_VIEWS = 4
VIS_MAX_DEPTH_SPREAD = 0.06  # distance may vary at most 6% between views
OCCLUDER_MM = 60         # depth pixels this much in front of the apple's surface belong to something in front of it
VIS_MAX_OCCLUDED = 0.15

MARKED_FRAMES = [("test_images", 15, "annotations/test_images_000015_points.csv"),
                 ("Tree2", 11, "annotations/Tree2_000011_points.csv"),
                 ("1aa65ece-d32f-4254-bcd3-80b26d2db7b9", 23, "annotations/1aa65e_000023_points.csv")]


# ---------------------------------------------------------------- depth

def fill_holes(depth, k=5):
    """Close the small gaps the reprojection leaves (a grid of empty pixels): each gap gets the nearest surface."""
    valid = np.isfinite(depth)
    big = np.where(valid, depth, 1e9).astype(np.float32)
    nearest = cv2.erode(big, np.ones((k, k), np.uint8))
    out = depth.copy()
    gap = ~valid & (nearest < 1e8)
    out[gap] = nearest[gap]
    return out


def load_depth(rec, frame_path, meta):
    raw = cv2.imread(str(rec / "depth" / (frame_path.stem + ".png")), cv2.IMREAD_UNCHANGED)
    aligned = P2.align_depth_to_color(raw, meta, z_max=DEPTH_READ_MM)
    return aligned, fill_holes(aligned)


def box_depth(aligned, filled, x1, y1, x2, y2):
    """Distance to the apple (median over the box centre), how much of the apple is covered by something in front of
    it, and a fallback distance from the surroundings when the apple itself has no depth."""
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    core = filled[int(cy - h / 4): int(cy + h / 4) + 1, int(cx - w / 4): int(cx + w / 4) + 1]
    zc = core[np.isfinite(core)]
    if len(zc) >= 4:
        z = float(np.median(zc))
        yy, xx = np.mgrid[int(y1): int(y2) + 1, int(x1): int(x2) + 1]
        inside = ((xx - cx) / max(w / 2, 1)) ** 2 + ((yy - cy) / max(h / 2, 1)) ** 2 <= 1
        box = aligned[int(y1): int(y2) + 1, int(x1): int(x2) + 1]
        zb = box[inside[: box.shape[0], : box.shape[1]] & np.isfinite(box)]
        occluded = float((zb < z - OCCLUDER_MM).mean()) if len(zb) >= 4 else np.nan
        return z, "apple", occluded
    around = filled[max(0, int(cy - 1.5 * h)): int(cy + 1.5 * h) + 1, max(0, int(cx - 1.5 * w)): int(cx + 1.5 * w) + 1]
    za = around[np.isfinite(around)]
    return (float(np.median(za)), "around", np.nan) if len(za) >= 4 else (np.nan, "none", np.nan)


# ---------------------------------------------------------------- detection and motion

def detect(rec, meta, model, frames):
    """Run the detector on every frame. One row per detection, with distance, occlusion and raw size in mm."""
    fx = meta["color_intrinsics"]["fx"]
    rows = []
    for i, f in enumerate(frames):
        color = cv2.imread(str(f))
        H, W = color.shape[:2]
        aligned, filled = load_depth(rec, f, meta)
        res = model.predict(color, classes=[47], conf=CONF_LOW, imgsz=IMGSZ, verbose=False)[0]
        boxes, scores = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
        keep = P2.suppress_nested(boxes, scores)
        for (x1, y1, x2, y2), c in zip(boxes[keep], scores[keep]):
            w, h = x2 - x1, y2 - y1
            size = (w + h) / 2
            z, source, occluded = box_depth(aligned, filled, x1, y1, x2, y2)
            # front surface distance + radius = centre distance:  D = size * (z + D/2) / fx
            diameter = size * z / (fx - size / 2) if source == "apple" else np.nan
            rows.append(dict(frame=i, conf=float(c), x1=x1, y1=y1, x2=x2, y2=y2, cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                             size=size, aspect=max(w, h) / max(min(w, h), 1),
                             at_edge=bool(x1 < 5 or y1 < 5 or x2 > W - 5 or y2 > H - 5),
                             z_mm=z, z_source=source, occluded=occluded, diameter_raw_mm=diameter,
                             far=bool(np.isfinite(z) and z > MAX_DEPTH_MM)))
    return pd.DataFrame(rows)


def depth_motion(det, shift):
    """How much of the camera motion is sideways travel (near apples move faster) rather than turning.
    Fits  apple motion / dominant motion = 1 + beta x (z_ref / z - 1);  beta 1 = pure travel, 0 = pure turning."""
    good = det[(det.conf >= CONF_KEEP) & (det.z_source == "apple") & ~det.far]
    z_ref = float(good.z_mm.median())
    by_frame = {f: g for f, g in good.groupby("frame")}
    xs, ys = [], []
    for f in range(len(shift) - 1):
        a, b = by_frame.get(f), by_frame.get(f + 1)
        d = shift[f] - shift[f + 1]  # dominant image motion from f to f+1
        if a is None or b is None or abs(d[0]) < 4:
            continue
        bx, by = b.cx.to_numpy(), b.cy.to_numpy()
        for r in a.itertuples():
            dist = np.hypot(bx - (r.cx + d[0]), by - (r.cy + d[1]))
            j = int(dist.argmin())
            if dist[j] < 0.6 * r.size:
                xs.append(z_ref / r.z_mm - 1)
                ys.append((bx[j] - r.cx) / d[0] - 1)
    xs, ys = np.array(xs), np.array(ys)
    ok = np.abs(ys) < 1.5
    if ok.sum() < 30 or np.sum(xs[ok] ** 2) == 0:
        return z_ref, 0.0, int(ok.sum())
    beta = float(np.sum(xs[ok] * ys[ok]) / np.sum(xs[ok] ** 2))
    return z_ref, float(np.clip(beta, 0, 1.2)), int(ok.sum())


def motion_factor(det, z_ref, beta):
    """Per detection: how fast it crosses the image relative to the dominant motion (1 when its distance is unknown)."""
    m = 1 + beta * (z_ref / det.z_mm - 1)
    m = m.where((det.z_source == "apple") & ~det.far, 1.0)
    return m.clip(0.4, 2.5).fillna(1.0)


# ---------------------------------------------------------------- counting

def strip_count(det, shift, W, k, offset, use_depth=True, use_limit=True, window=0):
    """Apples counted by every k-th frame: each apple counts only in the sampled frame whose centre is nearest to it in
    the scene. With use_depth, 'nearest' is measured in the apple's own depth layer (its motion factor m).

    window = 0: only the sampled frame itself is looked at (one chance per apple).
    window > 0 (short-term object permanence): the frames up to `window` before and after a sampled frame also report
    the apples in its strip, moved to where they would be in the sampled frame. Detections that land on the same spot
    are one apple, and an apple counts when it is seen in at least WINDOW_MIN_FRAMES of those frames. This rescues
    apples the detector missed in the sampled frame and drops one-frame flicker. Returns one row per counted apple
    (its view closest in time to the sampled frame)."""
    frames = np.arange(offset, len(shift), k)
    d = det[det.conf >= CONF_KEEP]
    if use_limit:
        d = d[~d.far]
    fr = d.frame.to_numpy().astype(int)
    m = d.m.to_numpy() if use_depth else np.ones(len(d))
    # which sampled frame's strip each detection belongs to (seen from its own frame)
    dist = np.abs((d.cx.to_numpy() - W / 2)[:, None] + m[:, None] * (shift[fr, 0][:, None] - shift[frames, 0][None, :]))
    strip = dist.argmin(axis=1)
    if not window:
        return d[frames[strip] == fr]
    near = np.abs(fr - frames[strip]) <= window
    d, m, fr, strip = d[near], m[near], fr[near], strip[near]
    target = frames[strip]
    px = d.cx.to_numpy() + m * (shift[fr, 0] - shift[target, 0])
    py = d.cy.to_numpy() + m * (shift[fr, 1] - shift[target, 1])
    size, conf = d["size"].to_numpy(), d.conf.to_numpy()
    reps = []
    for j in np.unique(strip):
        idx = np.flatnonzero(strip == j)
        idx = idx[np.argsort(-conf[idx])]
        clusters = []  # [anchor index, set of frames, member indices]
        for i in idx:
            for c in clusters:
                a = c[0]
                if fr[i] not in c[1] and np.hypot(px[i] - px[a], py[i] - py[a]) < 0.6 * max(size[i], size[a]):
                    c[1].add(fr[i])
                    c[2].append(i)
                    break
            else:
                clusters.append([i, {fr[i]}, [i]])
        for _, seen, members in clusters:
            if len(seen) >= WINDOW_MIN_FRAMES:
                reps.append((min(members, key=lambda i: (abs(fr[i] - frames[j]), -conf[i])), frames[j] not in seen))
    reps.sort()
    # rescued = no confident detection in the sampled frame itself: the one-frame count would have missed it
    return d.iloc[[i for i, _ in reps]].assign(rescued=[r for _, r in reps])


def count_variants(det, shift, W):
    rows = []
    for name, use_depth, use_limit, window in (("2d", False, False, 0), ("2d_limit", False, True, 0),
                                               ("3d_single_frame", True, True, 0), ("3d", True, True, WINDOW)):
        for k in SPACINGS:
            for off in range(min(k, 4)):
                rows.append(dict(method=name, k=k, offset=off,
                                 count=len(strip_count(det, shift, W, k, off, use_depth, use_limit, window))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- fully visible apples

def follow(start, by_frame, shift, n_frames):
    """Views of one counted apple in the frames around it, following its own (depth-scaled) motion."""
    views = [start]
    for step in (1, -1):
        cur, f, misses = start, int(start.frame), 0
        while misses < 2 and abs(f + step - start.frame) <= VIS_WINDOW and 0 <= f + step < n_frames:
            f += step
            px = cur.cx + cur.m * (shift[int(cur.frame), 0] - shift[f, 0])
            py = cur.cy + cur.m * (shift[int(cur.frame), 1] - shift[f, 1])
            g = by_frame.get(f)
            hit = None
            if g is not None:
                dist = np.hypot(g.cx.to_numpy() - px, g.cy.to_numpy() - py)
                j = int(dist.argmin())
                cand = g.iloc[j]
                if dist[j] < max(5.0, 0.5 * cur["size"]) and 0.7 < cand["size"] / cur["size"] < 1.4:
                    hit = cand
            if hit is None:
                misses += 1
            else:
                views.append(hit)
                cur, misses = hit, 0
    return pd.DataFrame(views)


def measure_visible(counted, det, shift):
    """For every counted apple: follow it, judge whether it is fully visible, and measure it (raw mm, before calibration)."""
    pool = det[(det.conf >= CONF_LOW) & ~det.far]
    by_frame = {f: g for f, g in pool.groupby("frame")}
    n_frames = len(shift)
    rows, all_views = [], []
    for idx, start in counted.iterrows():
        v = follow(start, by_frame, shift, n_frames)
        zv = v[v.z_source == "apple"]
        occ = v.occluded.dropna()
        row = dict(apple=idx, frame=int(start.frame), views=len(v), min_conf=float(v.conf.min()),
                   max_aspect=float(v.aspect.max()), size_px=float(v["size"].median()),
                   size_cv=float(v["size"].std() / v["size"].mean()) if len(v) > 1 else np.nan,
                   at_edge=bool(v.at_edge.any()), depth_views=len(zv),
                   z_mm=float(zv.z_mm.median()) if len(zv) else np.nan,
                   depth_spread=float(zv.z_mm.std() / zv.z_mm.mean()) if len(zv) > 1 else np.nan,
                   occluded=float(occ.median()) if len(occ) else np.nan,
                   diameter_raw_mm=float(zv.diameter_raw_mm.median()) if len(zv) else np.nan)
        row["fully_visible"] = bool(
            row["views"] >= VIS_MIN_VIEWS and row["min_conf"] >= VIS_MIN_CONF and row["max_aspect"] <= VIS_MAX_ASPECT
            and row["size_cv"] <= VIS_MAX_SIZE_CV and not row["at_edge"] and row["size_px"] >= VIS_MIN_SIZE_PX
            and row["depth_views"] >= VIS_MIN_DEPTH_VIEWS and row["depth_spread"] <= VIS_MAX_DEPTH_SPREAD
            and row["occluded"] <= VIS_MAX_OCCLUDED)
        rows.append(row)
        all_views.append(v.assign(apple=idx))
    return pd.DataFrame(rows), pd.concat(all_views) if all_views else pd.DataFrame()


# ---------------------------------------------------------------- one recording

def analyse(rec, model, out, reuse=False):
    """Detection, camera motion, counts and fully visible apples for one recording (no calibration needed)."""
    meta = json.loads((rec / "meta.json").read_text())
    frames = sorted((rec / "color").glob("*.jpg"))
    W = cv2.imread(str(frames[0])).shape[1]
    cache = out / "detections.csv"
    if reuse and cache.exists():
        det = pd.read_csv(cache)
    else:
        det = detect(rec, meta, model, frames)
    P2.CONF_KEEP = CONF_KEEP
    shift, lost = P2.camera_positions(det[~det.far], len(frames))
    z_ref, beta, n_moves = depth_motion(det, shift)
    det["m"] = motion_factor(det, z_ref, beta)
    out.mkdir(parents=True, exist_ok=True)
    det.to_csv(cache, index=False)

    table = count_variants(det, shift, W)
    t3 = table[table.method == "3d"]
    typical = t3.iloc[(t3["count"] - t3["count"].mean()).abs().argmin()]  # the sampling variant shown in the video
    counted = strip_count(det, shift, W, int(typical.k), int(typical.offset), window=WINDOW)
    apples, views = measure_visible(counted, det, shift)
    return dict(rec=rec, meta=meta, frames=frames, W=W, det=det, shift=shift, lost=lost, z_ref=z_ref, beta=beta,
                n_moves=n_moves, table=table, counted=counted, apples=apples, views=views)


def crop_sheet(r, apples, label, path, cols=8, tile=120):
    """One tile per measured apple, from its most confident view."""
    tiles = []
    for a in apples.itertuples():
        v = r["views"][(r["views"].apple == a.apple) & (r["views"].z_source == "apple")]
        if not len(v):
            continue
        best = v.loc[v.conf.idxmax()]
        img = cv2.imread(str(r["frames"][int(best.frame)]))
        half = int(max(24, 1.4 * best["size"]))
        pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REFLECT)
        cx, cy = int(best.cx) + half, int(best.cy) + half
        t = cv2.resize(pad[cy - half: cy + half, cx - half: cx + half], (tile, tile), interpolation=cv2.INTER_CUBIC)
        s = tile / (2 * half)
        cv2.rectangle(t, (int(tile / 2 - best["size"] / 2 * s), int(tile / 2 - best["size"] / 2 * s)),
                      (int(tile / 2 + best["size"] / 2 * s), int(tile / 2 + best["size"] / 2 * s)), (255, 255, 0), 1)
        P2.put_text(t, label(a), (3, tile - 6), 0.4)
        tiles.append(t)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def rescued_sheet(r, path, cols=8, tile=110):
    """Apples counted only thanks to the +-WINDOW frame memory (the detector missed them in the sampled frame)."""
    tiles = []
    for _, b in r["counted"][r["counted"].rescued].iterrows():
        img = cv2.imread(str(r["frames"][int(b.frame)]))
        half = int(max(24, 1.6 * b["size"]))
        pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REFLECT)
        cx, cy = int(b.cx) + half, int(b.cy) + half
        t = cv2.resize(pad[cy - half: cy + half, cx - half: cx + half], (tile, tile), interpolation=cv2.INTER_CUBIC)
        s = tile / (2 * half)
        cv2.rectangle(t, (int(tile / 2 - b["size"] / 2 * s), int(tile / 2 - b["size"] / 2 * s)),
                      (int(tile / 2 + b["size"] / 2 * s), int(tile / 2 + b["size"] / 2 * s)), (0, 255, 0), 1)
        P2.put_text(t, f"f{int(b.frame)} c{b.conf:.2f}", (3, tile - 6), 0.38)
        tiles.append(t)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


# ---------------------------------------------------------------- calibration

def calibrate(model, reuse=False):
    out = OUT / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    measured, per_rec = [], []
    for name in CALIBRATION_RECORDINGS:
        r = analyse(ARCHIVE / name, model, out / name, reuse)
        v = r["apples"][r["apples"].fully_visible].assign(recording=name)
        measured.append(v)
        crop_sheet(r, v, lambda a: f"{a.diameter_raw_mm:.0f} mm {a.z_mm / 1000:.1f} m", out / name / "fully_visible_apples.png")
        per_rec.append(dict(recording=name, counted=len(r["counted"]), fully_visible=len(v),
                            median_raw_diameter_mm=float(v.diameter_raw_mm.median()) if len(v) else None))
        print(f"  {name[:12]:12s} counted {len(r['counted']):3d}  fully visible {len(v):3d}  "
              f"median raw diameter {v.diameter_raw_mm.median():.1f} mm  beta {r['beta']:.2f}")
    m = pd.concat(measured, ignore_index=True)
    d = m.diameter_raw_mm.to_numpy()
    scale = REF_DIAMETER_MM / float(np.median(d))
    rng = np.random.default_rng(0)
    boot = REF_DIAMETER_MM / np.array([np.median(rng.choice(d, len(d))) for _ in range(2000)])
    cal = dict(scale=scale, scale_low=float(np.percentile(boot, 5)), scale_high=float(np.percentile(boot, 95)),
               scale_bootstrap=boot.round(5).tolist(), apples=len(d), median_raw_diameter_mm=float(np.median(d)),
               reference=dict(diameter_mm=REF_DIAMETER_MM, mass_g=REF_MASS_G, axes_cm=[7.0, 6.5, 6.0],
                              note="hand-weighed apple, described as average size on tree 1"),
               density_g_cm3=DENSITY, per_recording=per_rec, recordings=CALIBRATION_RECORDINGS,
               note="scale = reference diameter / median measured diameter of the fully visible tree-1 apples. "
                    "The same physical apples appear in several recordings, so the bootstrap range is optimistic.")
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    m.to_csv(out / "fully_visible_apples.csv", index=False)
    depth_check_plot(m, scale, out / "depth_check.png")
    print(f"Size scale {scale:.3f} (90% {cal['scale_low']:.3f}-{cal['scale_high']:.3f}) from {len(d)} fully visible "
          f"tree-1 apples; median measured {np.median(d):.1f} mm -> reference {REF_DIAMETER_MM:.0f} mm")
    return cal


def depth_check_plot(m, scale, path):
    """Apple size in pixels shrinks with distance; in mm (pixels x distance) it should not."""
    z = m.z_mm.to_numpy() / 1000
    surface, ink, ink2, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3", "#2a78d6"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=surface)
    for a in ax:
        a.set_facecolor(surface)
        a.spines[["top", "right"]].set_visible(False)
        a.spines[["left", "bottom"]].set_color(grid)
        a.grid(axis="y", color=grid, linewidth=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors=ink2, length=0)
        a.set_xlabel("distance to apple (m)", color=ink2)
    ax[0].scatter(z, m.size_px, s=30, color=series, edgecolor=surface, linewidth=1.2)
    ax[0].set_ylabel("apple size in the image (px)", color=ink2)
    ax[0].set_ylim(0, float(m.size_px.max()) * 1.15)
    ax[0].set_title("In pixels: far apples look smaller", loc="left", color=ink, fontsize=11)
    dmm = m.diameter_raw_mm * scale
    ax[1].axhline(REF_DIAMETER_MM, color=ink2, linewidth=1)
    ax[1].scatter(z, dmm, s=30, color=series, edgecolor=surface, linewidth=1.2, zorder=3)
    ax[1].text(1.01, REF_DIAMETER_MM, "reference\napple\n65 mm", color=ink2, fontsize=9, va="center",
               transform=ax[1].get_yaxis_transform())
    ax[1].set_ylabel("apple diameter from depth (mm)", color=ink2)
    ax[1].set_ylim(0, max(100, float(dmm.max()) + 10))
    ax[1].set_title("In mm, using depth: size no longer depends on distance", loc="left", color=ink, fontsize=11)
    fig.suptitle(f"Fully visible tree-1 apples used for the calibration (n = {len(m)})", x=0.01, ha="left", color=ink, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------- results for a test recording

def summarize(r, cal):
    table, apples = r["table"], r["apples"]
    t = {k: g["count"] for k, g in table.groupby("method")}
    n = int(round(t["3d"].mean()))
    count_low = int(round(t["3d"].min() * (1 - COUNT_ERROR)))
    count_high = int(round(t["3d"].max() * (1 + COUNT_ERROR)))

    k = cal["scale"]
    vis = apples[apples.fully_visible].copy()
    vis["diameter_mm"] = vis.diameter_raw_mm * k
    vis["mass_g"] = DENSITY * (np.pi / 6) * (vis.diameter_mm / 10) ** 3
    if len(vis) >= 5:
        masses, source = vis.mass_g.to_numpy(), f"measured with depth on {len(vis)} fully visible apples"
    else:  # too few clean apples in this recording: fall back on the calibration apples (tree 1)
        cal_m = pd.read_csv(OUT / "calibration" / "fully_visible_apples.csv")
        masses = (DENSITY * (np.pi / 6) * (cal_m.diameter_raw_mm * k / 10) ** 3).to_numpy()
        source = f"only {len(vis)} fully visible apples here; tree-1 calibration apples used"
    rng = np.random.default_rng(1)
    kb = np.array(cal["scale_bootstrap"])
    boot = np.array([rng.choice(masses, len(masses)).mean() * (rng.choice(kb) / k) ** 3 for _ in range(4000)])
    mean_g = float(masses.mean())
    g_low, g_high = (float(x) for x in np.percentile(boot, [5, 95]))

    det = r["det"]
    conf_det = det[det.conf >= CONF_KEEP]
    summary = dict(
        recording=r["rec"].name, tree=TREE_OF.get(r["rec"].name, "?"), frames=len(r["frames"]),
        apples_counted=n, count_low=count_low, count_high=count_high,
        count_2d_no_limit=float(t["2d"].mean()), count_2d_with_limit=float(t["2d_limit"].mean()),
        count_3d_single_frame=float(t["3d_single_frame"].mean()), count_3d=float(t["3d"].mean()),
        background_removed=float(t["2d"].mean() - t["2d_limit"].mean()),
        detections_beyond_limit=int(conf_det.far.sum()), detections_without_depth=int((conf_det.z_source == "none").sum()),
        max_depth_m=MAX_DEPTH_MM / 1000, tree_distance_m=r["z_ref"] / 1000, sideways_travel_share_beta=r["beta"],
        fully_visible_apples=int(len(vis)), median_diameter_mm=float(vis.diameter_mm.median()) if len(vis) else None,
        mean_weight_g=mean_g, weight_low_g=g_low, weight_high_g=g_high, weight_source=source,
        estimate_kg=n * mean_g / 1000, kg_low=count_low * g_low / 1000, kg_high=count_high * g_high / 1000,
        size_scale=k, size_scale_range=[cal["scale_low"], cal["scale_high"]],
        camera_pan_px=float(abs(r["shift"][-1, 0] - r["shift"][0, 0])),
        note=f"plausible range, not a statistical interval: count = lowest/highest of {len(t['3d'])} strip-sampling variants "
             f"+/-{COUNT_ERROR:.0%} counting error; weight = 90% bootstrap over the fully visible apples and the size scale. "
             "Size scale from one reference apple (109 g) assumed to be average for tree 1.",
    )
    return summary, vis


def summary_panel(img, s):
    H, W = img.shape[:2]
    out = (img * 0.3).astype(np.uint8)
    x0, y0, x1, y1 = W // 2 - 450, H // 2 - 235, W // 2 + 450, H // 2 + 235
    cv2.rectangle(out, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.rectangle(out, (x0, y0), (x1, y1), (200, 200, 200), 2)
    x, y = x0 + 40, y0 + 50
    gray, light = (190, 190, 190), (150, 150, 150)
    P2.put_text(out, f"{s['recording']}  ({s['tree']})  - 3D", (x, y), 0.75, gray)
    P2.put_text(out, f"{s['apples_counted']} apples counted", (x, y + 65), 1.5, (120, 255, 120), 3)
    P2.put_text(out, f"plausible range {s['count_low']} - {s['count_high']}  (average of 12 samplings)", (x, y + 100), 0.75, gray)
    P2.put_text(out, f"Estimated yield: {s['estimate_kg']:.1f} kg", (x, y + 155), 1.25, (255, 255, 255), 2)
    P2.put_text(out, f"plausible range {s['kg_low']:.1f} - {s['kg_high']:.1f} kg", (x, y + 190), 0.75, gray)
    P2.put_text(out, f"mean apple {s['mean_weight_g']:.0f} g ({s['weight_low_g']:.0f}-{s['weight_high_g']:.0f}), "
                     f"median {s['median_diameter_mm'] or 0:.0f} mm", (x, y + 235), 0.7, gray)
    P2.put_text(out, s["weight_source"], (x, y + 265), 0.6, gray)
    P2.put_text(out, "size scale from the 109 g reference apple (tree 1)", (x, y + 293), 0.6, gray)
    P2.put_text(out, f"depth limit {s['max_depth_m']:.1f} m: {s['background_removed']:.0f} background objects not counted",
                (x, y + 321), 0.6, gray)
    P2.put_text(out, "Visible apples only, one side of the tree", (x, y + 360), 0.6, light)
    return out


def colorize_depth(filled, w, h):
    z = cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST)
    norm = np.clip((np.nan_to_num(z, nan=0) - 500) / (MAX_DEPTH_MM - 500), 0, 1)
    img = cv2.applyColorMap((255 - norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # near = red, far = blue
    img[np.isnan(z)] = (0, 0, 0)
    img[np.nan_to_num(z, nan=0) > MAX_DEPTH_MM] = (80, 80, 80)
    return img


def render(r, s, vis, out):
    frames, det, counted, views = r["frames"], r["det"], r["counted"], r["views"]
    H, W = cv2.imread(str(frames[0])).shape[:2]
    video = cv2.VideoWriter(str(out / "annotated_3d.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    shown = {f: g for f, g in det[det.conf >= CONF_KEEP].groupby("frame")}
    counted_by = {f: g for f, g in counted.groupby("frame")}
    vis_d = vis.set_index("apple").diameter_mm.to_dict()
    measured_by = {f: g for f, g in views[views.apple.isin(vis_d)].groupby("frame")} if len(views) else {}
    iw, ih = 320, 180
    running, img = 0, None
    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        _, filled = load_depth(r["rec"], f, r["meta"])
        for b in shown.get(i, pd.DataFrame()).itertuples():
            p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
            if b.far:
                cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
                P2.put_text(img, f"{b.z_mm / 1000:.1f} m", (p1[0], p1[1] - 4), 0.45, (120, 120, 255))
            else:
                cv2.rectangle(img, p1, p2, (170, 170, 170), 1)
        c = counted_by.get(i, pd.DataFrame())
        running += len(c)
        for b in c.itertuples():
            cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 255, 0), 2)
        for b in measured_by.get(i, pd.DataFrame()).itertuples():
            cv2.rectangle(img, (int(b.x1) - 2, int(b.y1) - 2), (int(b.x2) + 2, int(b.y2) + 2), (255, 255, 0), 2)
            P2.put_text(img, f"{vis_d[b.apple]:.0f} mm", (int(b.x1), int(b.y1) - 6), 0.5, (255, 255, 0))
        n_far = int(shown[i].far.sum()) if i in shown else 0
        cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
        P2.put_text(img, f"frame {i}   counted so far (this sampling): {running}   depth limit {MAX_DEPTH_MM / 1000:.1f} m"
                         f"   beyond limit in this frame: {n_far}", (10, 24), 0.6)
        # legend
        lx, ly = 10, H - 70
        cv2.rectangle(img, (lx - 6, ly - 20), (lx + 330, H - 8), (0, 0, 0), -1)
        for j, (col, txt) in enumerate((((0, 255, 0), "counted in this frame"), ((255, 255, 0), "fully visible: measured in mm"),
                                        ((0, 0, 255), "too far (background): not counted"))):
            cv2.rectangle(img, (lx, ly - 10 + 20 * j), (lx + 14, ly + 2 + 20 * j), col, 2)
            P2.put_text(img, txt, (lx + 24, ly + 2 + 20 * j), 0.45)
        # depth inset
        inset = colorize_depth(filled, iw, ih)
        x0, y0 = W - iw - 10, H - ih - 10
        img[y0:y0 + ih, x0:x0 + iw] = inset
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih), (255, 255, 255), 1)
        P2.put_text(img, f"depth: red 0.5 m - blue {MAX_DEPTH_MM / 1000:.0f} m, gray = too far", (x0, y0 - 6), 0.42)
        video.write(img)
    final = summary_panel(img, s)
    cv2.imwrite(str(out / "final.png"), final)
    for _ in range(HOLD_SECONDS * FPS):
        video.write(final)
    video.release()


def render_audit(r, out, hold_seconds=2):
    """Slow video for checking the count by eye: every counted apple gets a number on the frame where it is counted,
    and that frame is held still. audit_apples.csv lists the numbers."""
    frames, counted = r["frames"], r["counted"].sort_values(["frame", "cx"])
    H, W = cv2.imread(str(frames[0])).shape[:2]
    video = cv2.VideoWriter(str(out / "audit_numbered.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    numbers, n = [], 0
    by = {f: g for f, g in counted.groupby("frame")}
    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        g = by.get(i)
        if g is None:
            cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
            P2.put_text(img, f"frame {i}   apples counted so far: {n}", (10, 24), 0.65)
            video.write(img)
            continue
        for _, b in g.iterrows():
            n += 1
            numbers.append(dict(number=n, frame=i, x=round(b.cx), y=round(b.cy), size_px=round(b["size"], 1),
                                conf=round(b.conf, 2), distance_m=round(b.z_mm / 1000, 2) if np.isfinite(b.z_mm) else None))
            cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 255, 0), 2)
            P2.put_text(img, str(n), (int(b.x2) + 2, int(b.y1) + 10), 0.5, (0, 255, 0))
        cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
        P2.put_text(img, f"frame {i}   apples counted so far: {n}   (new numbers in this frame: {n - len(g) + 1}-{n})", (10, 24), 0.65)
        for _ in range(hold_seconds * FPS):
            video.write(img)
    video.release()
    pd.DataFrame(numbers).to_csv(out / "audit_apples.csv", index=False)


def print_summary(s):
    bar = "=" * 64
    print(bar)
    print(f" {s['recording']}  ({s['tree']})")
    print(f"   Apples counted : {s['apples_counted']}   (plausible range {s['count_low']} - {s['count_high']})")
    print(f"      2D, no limit {s['count_2d_no_limit']:.0f} | 2D + 3 m limit {s['count_2d_with_limit']:.0f} | "
          f"3D one frame {s['count_3d_single_frame']:.0f} | 3D +-{WINDOW} frames {s['count_3d']:.0f}")
    print(f"   Mean apple     : {s['mean_weight_g']:.0f} g ({s['weight_low_g']:.0f}-{s['weight_high_g']:.0f}), "
          f"median {s['median_diameter_mm'] or 0:.0f} mm, {s['weight_source']}")
    print(f"   Estimated yield: {s['estimate_kg']:.1f} kg   (plausible range {s['kg_low']:.1f} - {s['kg_high']:.1f} kg)")
    print(f"   Tree at {s['tree_distance_m']:.2f} m, sideways travel share {s['sideways_travel_share_beta']:.2f}, "
          f"{s['detections_beyond_limit']} detections beyond the limit")
    print(bar)


def process(rec, model, cal, reuse=False):
    out = OUT / rec.name
    r = analyse(rec, model, out, reuse)
    s, vis = summarize(r, cal)
    r["table"].to_csv(out / "counts.csv", index=False)
    r["apples"].merge(vis[["apple", "diameter_mm", "mass_g"]], on="apple", how="left").to_csv(out / "apples_measured.csv", index=False)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    crop_sheet(r, vis, lambda a: f"{a.diameter_mm:.0f} mm {a.mass_g:.0f} g", out / "measured_apples.png")
    if WINDOW:
        rescued_sheet(r, out / "rescued_apples.png")
    render_audit(r, out)
    render(r, s, vis, out)
    return s


# ---------------------------------------------------------------- check on the hand-marked frames

def evaluate_marked(model):
    """Recall and precision on the hand-marked frames, with and without the depth limit."""
    rows = []
    for name, f, csv in MARKED_FRAMES:
        rec = ARCHIVE / name
        meta = json.loads((rec / "meta.json").read_text())
        fp = rec / "color" / f"{f:06d}.jpg"
        det = detect(rec, meta, model, [fp])
        det = det[det.conf >= CONF_KEEP].reset_index(drop=True)
        pts = pd.read_csv(csv)[["x", "y"]].to_numpy()
        for limit in (False, True):
            d = det[~det.far] if limit else det
            hit_b, hit_p = set(), set()
            for i, b in d.iterrows():
                inside = [j for j, (px, py) in enumerate(pts) if j not in hit_p and b.x1 - 8 <= px <= b.x2 + 8 and b.y1 - 8 <= py <= b.y2 + 8]
                if inside:
                    j = min(inside, key=lambda j: (pts[j][0] - b.cx) ** 2 + (pts[j][1] - b.cy) ** 2)
                    hit_b.add(i)
                    hit_p.add(j)
            rows.append(dict(frame=f"{name[:12]}:{f}", depth_limit=f"{MAX_DEPTH_MM / 1000:.1f} m" if limit else "none",
                             marked=len(pts), detections=len(d), recall=len(hit_p) / len(pts),
                             precision_lower_bound=len(hit_b) / max(len(d), 1), count_error=len(d) / len(pts) - 1))
    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "marked_frames_eval.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="*", type=Path)
    ap.add_argument("--calibrate", action="store_true", help="fit the size scale on the tree-1 calibration recordings")
    ap.add_argument("--evaluate", action="store_true", help="score the detector on the hand-marked frames")
    ap.add_argument("--reuse", action="store_true", help="reuse detections.csv from an earlier run")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    model = YOLO(WEIGHTS)
    cal_path = OUT / "calibration" / "calibration.json"
    cal = calibrate(model, args.reuse) if args.calibrate or not cal_path.exists() else json.loads(cal_path.read_text())
    if args.evaluate:
        print(evaluate_marked(model).to_string(index=False))
    rows = []
    for rec in args.recordings:
        s = process(rec, model, cal, args.reuse)
        print_summary(s)
        rows.append(s)
    if rows:
        cols = ["recording", "tree", "count_2d_no_limit", "count_2d_with_limit", "count_3d_single_frame", "count_3d", "count_low", "count_high",
                "fully_visible_apples", "median_diameter_mm", "mean_weight_g", "estimate_kg", "kg_low", "kg_high"]
        pd.DataFrame(rows)[cols].round(1).to_csv(OUT / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
