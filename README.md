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

This writes `Main/3d_annotated/<recording>/` with `annotated_3d.mp4`, `audit_numbered.mp4`, `final.png` and the numbers (`summary.json`, `comparison.csv`), and prints the summary. Expected:

| Recording | Apples | Yield |
|---|---|---|
| test_images | 100 | 11.6 kg |
| Tree2 | 50 | 7.1 kg |
| 1dceb275 | 43 | 5.5 kg |

The repo contains the fine-tuned detector (`Main/runs/detect/runs/apple_ft/weights/last.pt`) and the size calibration (`Main/3d_annotated/calibration/calibration.json`), so nothing else is needed. Do not run `--calibrate` unless you want to redo the calibration; it overwrites `calibration.json`.
