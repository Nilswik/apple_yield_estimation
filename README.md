# Apple yield estimation (RGB-D)

Counts apples in Orbbec Gemini 2 recordings, measures their size with depth and estimates yield.
Project background and results are in `CLAUDE.md`.

## Setup

1. Clone the repo into a folder whose path has only plain letters (for example `C:\apple`). OpenCV cannot open images on paths with letters like Ä.
2. Put the `Archive` folder in the repo root, next to `Main`:
   ```
   apple/
     Archive/        <- the recordings (not in the repo)
     Main/
     requirements.txt
   ```
3. Install Python 3.12 and the packages:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```
   Skip the `--index-url` line if there is no NVIDIA GPU (it runs on the CPU, just slower).

## Run

From the `Main` folder:
```
cd Main
python pipeline_3d.py ../Archive/test_images ../Archive/Tree2 ../Archive/1dceb275-6e92-4368-9df5-1526972df094
```

This writes `Results/pipeline_3d/<recording>/` with `annotated_3d.mp4`, `audit_numbered.mp4`, `final.png` and the numbers (`summary.json`, `comparison.csv`), and prints the summary (in Swedish). Expected:

| Recording | Apples | Yield |
|---|---|---|
| test_images | 100 | 11.6 kg |
| Tree2 | 50 | 7.1 kg |
| 1dceb275 | 43 | 5.5 kg |

The repo contains everything the run needs, all in `Main/`: the fine-tuned detector (`models/apple_run1.pt`), the size calibration (`calibration/`) and the hand-marked apples for `--evaluate` (`annotations/`). Do not run `--calibrate` unless you want to redo the calibration; it overwrites `calibration/calibration.json`.

## Folders

- `Main/`: only what the working version needs (`pipeline_3d.py` computes, `visualize.py` draws, `oak_d_snapshots.py` for OAK-D).
- `Results/`: all outputs (`pipeline_3d/`, `oak_d/`, plus older runs). Not in the repo, except a README.
- `Development/` (local, not in the repo): training data and scripts, all training runs, evaluation scripts, the old 2D pipeline and notes.
- `HOW_IT_WORKS.md`: step-by-step explanation of the method.

## OAK-D snapshots (Luxonis)

For single OAK-D captures: a folder with, per capture, `<name>_metadata.json`, the RGB image and the depth PNG (mm, aligned to RGB), as exported by the OAK-D. The folder can be anywhere (avoid letters like Ä in its path).

```
cd Main
python oak_d_snapshots.py C:/path/to/the/oak-d/folder
```

This writes `Results/oak_d/`: one annotated image per capture (with its depth map), `overview.png`, `summary.csv` and `apples.csv`. It uses the run 2b detector (`Main/models/apple_run2b.pt`), trained on both cameras. Counts are per capture; the captures overlap, so they must not be added up into a tree total.
