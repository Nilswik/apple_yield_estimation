# Apple yield POC (Optifruit): project context

Student proof of concept: count apples and estimate yield per tree from RGB-D video. The stakeholder is non-technical and happy with rough estimates; he liked the earlier `Workshop.ipynb`. Keep things simple and POC-sized. Don't push heavy data collection or 3D reconstruction.

## Layout (all paths relative to this folder, `Äpple/`)
- `Main/pipeline_3d.py`: **the current system** (2D pipeline + depth). Usage in the file header. Run from `Main/`. Output in `Main/3d_annotated/` (has its own README with results).
- `Main/pipeline.py`: the 2D pipeline; `pipeline_3d.py` imports its shared parts (box suppression, camera-motion vote, depth reprojection, strip sampling).
- `Main/train.py`: fine-tune YOLO11m on `Main/dataset/` (class 47 = COCO apple). `--data`, `--epochs`, `--name`.
- `Main/eval_models.py`, `Main/eval_curve.py`: score detectors against hand-marked dots.
- `Main/annotations/*_points.csv` + `marked_screenshots/`: the user's hand-marked apples (3 frames), used as ground truth.
- `Main/annotation_batch/images/`: frames with X-AnyLabeling `.json` labels. 7 are corrected and used for training; the other 8 are pre-labelled but not yet corrected. **`1dceb2_*` frames are tree 2**: don't train on them while Tree2 is used as a test.
- `Main/runs/detect/runs/apple_ft/weights/last.pt`: the fine-tuned detector (7 tree-1 frames). `ft3`, `ft5` are the learning-curve runs.
- `Main/results_finetuned/`, `Main/results/`: older 2D outputs, made **before the strip-sampling bug fix** (see below). Kept for history.
- `Main/STAKEHOLDER_SUMMARY.md`: one-page summary already sent to the stakeholder (with the 2D videos). Its counts and kg are outdated.
- `Main/Model_lab.ipynb`: early baselines (red-color mask, pretrained YOLO). `capture.py`: RealSense recorder (older, from before the Orbbec data).
- `Archive/`: recordings, all tagged "test1" in meta.json but actually two trees:
  - **Tree 1 (red apples):** 1aa65ece, 341f04ca, 5fa2faba, 849d204a, Annotation_tetst (=6d73f59e), d4d77803 (used for training frames and for the size calibration), and `test_images` (=dd626255, held out).
  - **Tree 2 (pale apples, white house):** `Tree2` (=8e91c104) and `1dceb275` (second pass of the same tree). Both held out from training.
  - `fdc3cc89` is a lab scene with no apples; ignore it.

## Data format
Orbbec Gemini 2 recordings: `color/NNNNNN.jpg` (1280x720), `depth/NNNNNN.png` (uint16 mm, 1280x800, **not aligned** to color), `meta.json` with intrinsics and extrinsics (color fx about 689). The pipelines reproject depth onto color themselves; `pipeline_3d.py` also fills the reprojection gaps. Windows note: `cv2.imread` fails on paths with the letter Ä when given absolute paths; use relative paths.

## How the system works (pipeline_3d.py)
1. Detect apples (fine-tuned YOLO11m, imgsz 1920, conf 0.45 counts, 0.2 for following) in every frame; drop stacked duplicate boxes.
2. Depth per apple (median over the box centre; neighbourhood fallback). **Distance limit 3.0 m**: farther = background, not counted. No-depth detections are kept.
3. Camera motion from a vote on apple displacements. Motion is mostly sideways travel (apple motion = 1 + beta x (z_ref/z - 1), beta 0.6-0.85; 0.18 for 1dceb275), so each apple's strip is scaled by its distance.
4. Count by strip sampling: each apple counts in the one sampled frame whose centre is nearest in the scene (K in 3, 5, 8; 12 variants averaged). Double counts about 1 per pass (also seen in the audit video); camera motion agrees with optical flow within 2%. Inside hand-marked views the count matches the hand count (70/68, 48/45, 74-78/76). Tried and dropped: whole-pass position memory and per-apple optical-flow tracking (both fragment, 3-6x too many); a +-2 frame memory (`WINDOW`, off) counted 16% more than people do.
5. Fully visible apples (>= 7 views, conf >= 0.6, round, size CV <= 10%, steady depth, nothing in front in depth) are measured in mm.
6. Size scale 1.153 fitted on 92 fully visible apples from the six tree-1 recordings so the median matches the reference apple (65 mm, 109 g). Mass = 0.758 g/cm3 x sphere volume.
7. Yield = count x mean mass of that recording's fully visible apples. Ranges are plausible ranges, not statistical intervals.
Commands: `python pipeline_3d.py --calibrate` (once), then `python pipeline_3d.py <recordings...> [--evaluate] [--reuse]`.

**Bug fixed 2026-10-05:** `strip_sample` assumed the camera always moved one way; both test recordings move the other way, so earlier 2D counts (86 / 55 apples sent to the stakeholder) were wrong. Corrected 2D: 101 / 51.

## Current results (2026-10-05, 3D)
- test_images (tree 1): 100 apples (83-122), mean 116 g (27 apples measured, median 67 mm), 11.6 kg (8.7-15.8).
- Tree2 (tree 2): 50 apples (40-61), 141 g (only 6 apples measured), 7.1 kg (4.4-10.6).
- 1dceb275 (tree 2, second pass): 43 apples (33-52), 127 g (8 apples), 5.5 kg (3.5-8.0).
- The user expects 70-80 on test_images: the extra are partly hidden apples ("slivers"), which people do not count. A rule "count only apples with >= 55% of a whole apple's width visible, judged with depth (min box side / (fx x 56.4 mm / distance))" gives 81 / 46 / 39 and matches the hand-marked dots equally well (F1 0.80, precision 0.79 -> 0.82). Not implemented yet; for yield, partly hidden apples are still real apples, so report both numbers if it is added.
- Detector on held-out marked frames: fine-tuned m F1 0.855 (pretrained m 0.80, x 0.84). Distance limit: precision +1.5-2.7 points, at most one marked apple lost per frame. Depth-aware counting changes counts by at most 3.

## Known caveats (state these honestly)
- Whole-pass counts are **not validated** by a hand count. Per-frame detection is (about 7% count error on held-out marked frames).
- The size calibration rests on **one** weighed apple (7.0 x 6.5 x 6.0 cm, 109 g), assumed to be from tree 1 and of average size. Raw box+depth sizes run about 13% small; the scale fixes that.
- Tree-2 weights come from only 6-8 apples. The detector was fine-tuned on tree 1 only. Some leaves are still counted; small shaded apples are missed.
- Failed approaches (good for the report's "unsuccessful iteration"): red-color thresholds, Hough circles, brightening/CLAHE preprocessing, BoT-SORT and cluster-based identity matching (double-counted, 2-3x too many), second-stage crop classifier (did not help).

## Next steps / pending
- Tell the stakeholder the corrected numbers (the 2D summary he got used the buggy count and an assumed 94 g).
- Optional: add the 55% "clearly visible" rule (see Current results) as a second count in the summary and videos.
- **New data coming from the stakeholder: Luxonis (OAK) camera.** Totally new hardware, a good generalisation test. Needs: intrinsics (fx, fy, cx, cy), color resolution, whether depth is aligned to color, depth unit, format/fps. Plan: write a converter into the `color/`, `depth/`, `meta.json` layout, run **untouched** first, mark one frame with dots for a hand count, then judge. The size scale will need re-checking for that camera.
- Hand-count one full pass (two-frame method: dot frame 0 and the new part of the last frame), weigh about 10 apples (ideally the measured ones), label a few frames from other trees.
- Camera setup idea for real use: higher, near-level mount instead of low and tilted at the sky; A/B comparison planned.

## Tools and environment
- Python venv at `.venv/` (torch + CUDA on an RTX 4070, ultralytics, opencv, scipy, matplotlib). Model files `yolo11m.pt`, `yolo11x.pt` in this folder.
- Annotation: X-AnyLabeling 4.1.0 in its own venv under `Downloads/X-AnyLabeling-main/X-AnyLabeling-main`; start with `.\.venv\Scripts\xanylabeling.exe`. It reads/writes the `.json` next to each image.
- Training writes weights under `Main/runs/detect/runs/<name>/weights/`.
