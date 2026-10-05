# 3D results (depth added)

Made with `Main/pipeline_3d.py` on 2026-10-05. The 2D results in `Main/results_finetuned/` are kept for comparison; they were made before the counting bug fix described in `CLAUDE.md`.

## What depth is used for

1. **Distance limit (3.0 m).** Detections farther away are background (dishes, buckets, rakes, the garden behind the tree) and are not counted. They show as red boxes with their distance in the videos.
2. **Size of fully visible apples.** Apples seen clearly for at least 7 frames, confident, round, with a steady size and nothing in front of them in depth, are measured in mm from box size and distance (cyan boxes with mm labels in the videos).
3. **Calibration with the weighed apple.** The measured sizes run about 13% small. One factor (1.15) fixes that so the median fully visible apple on tree 1 matches the 109 g / 65 mm reference apple. It was fitted on six tree-1 recordings that are not used as tests (`calibration/`).
4. **Depth-aware counting.** Near apples cross the image faster than far ones when the camera moves sideways, so each apple's counting strip is scaled by its distance. Each apple is counted in one sampled frame.

Yield = apples counted x mean weight of that recording's fully visible apples.

## Results

| Recording | Tree | Apples counted (range) | Fully visible apples measured | Mean apple | Yield (range) |
|---|---|---|---|---|---|
| test_images | 1 | 100 (83 - 122) | 27 | 116 g, median 67 mm | 11.6 kg (8.7 - 15.8) |
| Tree2 | 2 | 50 (40 - 61) | 6 | 141 g, median 70 mm | 7.1 kg (4.4 - 10.6) |
| 1dceb275 (second pass of tree 2) | 2 | 43 (33 - 52) | 8 | 127 g, median 66 mm | 5.5 kg (3.5 - 8.0) |

## What was checked

- `calibration/depth_check.png`: in pixels apples shrink with distance; in mm (with depth) the size stays flat. This is what the depth data adds.
- test_images (tree 1, not used in calibration) gives a median of 67 mm and 116 g against the 65 mm / 109 g reference: the measurement is consistent across recordings.
- Inside the view of each hand-marked frame, the count matches the hand count: 70 vs 68, 48 vs 45, 74-78 vs 76.
- Double counting is rare: about 1 apple per pass, confirmed in the audit video. The camera-motion estimate agrees with optical flow within 2%.
- The count includes partly hidden apples ("slivers") that a person does not count. Counting only apples with at least 55% of a whole apple's width visible (judged with depth) gives 81 / 46 / 39 and matches the hand-marked frames equally well. Not switched on yet.
- `marked_frames_eval.csv`: the distance limit raises precision on the hand-marked frames by 1.5 - 2.7 points and costs at most one marked apple per frame.

Tried and dropped: a memory of where apples were seen over the whole pass, and optical-flow tracking of each apple. Both split apples into pieces (3-6 times too many). Remembering apples over +-2 frames worked technically but counted 16% more than people do.

## Limits

- The pass counts are not checked against a full hand count of a pass.
- Everything depends on one weighed apple, assumed to come from tree 1 and to be of average size.
- Tree 2 means come from only 6 - 8 fully visible apples, hence the wide ranges.
- The ranges are plausible ranges, not statistical intervals.

## Files per recording

`annotated_3d.mp4` (ends with the summary), `audit_numbered.mp4` + `audit_apples.csv` (every counted apple numbered, frame held still, for checking by eye), `final.png`, `measured_apples.png` (the apples used for the weight), `apples_measured.csv`, `counts.csv` (all counting variants), `detections.csv`, `summary.json`. `comparison.csv` collects the numbers for all three recordings.
