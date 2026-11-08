# Project 26: Two-Stage (Faster R-CNN) vs One-Stage (YOLOv8) Detectors

DS307 Computer Vision, Assignment 2, IFHE University, Hyderabad.

This project benchmarks pretrained COCO detectors on the same COCO val2017 images under the same conditions:
**Faster R-CNN ResNet-50-FPN v2** and **Faster R-CNN MobileNetV3-Large-FPN** (two-stage, torchvision), and
**YOLOv8n / YOLOv8s / YOLOv8m** (one-stage, Ultralytics). It also includes a live side-by-side demo.

## The 5 commands

You only need [uv](https://docs.astral.sh/uv/getting-started/installation/). Run everything from this folder.

| Command | What it does |
|---|---|
| `uv run bench` | Benchmark: downloads data + weights, runs all models, writes `results/` |
| `uv run report` | Diagrams + `CV_Assignment2_Report.docx` (and a `.pdf` copy) |
| `uv run slides` | `CV_Assignment2_Slides.pptx` |
| `uv run build` | `report` + `slides` in one go |
| `uv run demo` | Live demo in the browser (http://localhost:8501) |

The first `uv run` creates `.venv/` and installs the exact versions from `uv.lock` automatically
(or do it explicitly with `uv sync`).

Typical order: `uv run bench`, then `uv run build`, then `uv run demo` for the presentation.

## Benchmark options

| Hardware | Images | Models |
|---|---|---|
| CUDA GPU | 500 | all 5 |
| CPU only | 100 | 4 (YOLOv8m skipped) |

```bash
uv run bench --n 200                       # change the number of images
uv run bench --include-largest             # on CPU, also run YOLOv8m
uv run bench --models YOLOv8n YOLOv8s      # only some models
uv run bench --charts-only                 # redraw charts from results/results.json
```

Re-running the benchmark overwrites `results/`. Afterwards run `uv run build`: the report and slides
read every number from `results/results.json`.

**GPU:** `pyproject.toml` installs the small CPU build of PyTorch. On a machine with an NVIDIA GPU,
delete the `[tool.uv.sources]` and `[[tool.uv.index]]` sections at the bottom of `pyproject.toml`
and run `uv sync` to get the CUDA build.

## Demo

```bash
uv run demo
```

In the sidebar, pick the two-stage model, the one-stage model, the score threshold and the input:
upload image, webcam snapshot or live webcam. On a CPU, use **FasterRCNN-MobileNetV3-FPN** vs
**YOLOv8n** for the live webcam so the frames keep up.
Stop it with Ctrl+C.

Backup without a browser (writes a side-by-side image and prints the timings):

```bash
uv run demo --image data/val2017/000000000139.jpg --out demo_out.jpg
```

## Files

| File | What it does |
|---|---|
| `tasks.py` | The 5 short commands (`uv run bench` etc.), registered in `pyproject.toml` |
| `common.py` | Model registry, loading, one `detect()` function for both libraries, timing, drawing |
| `benchmark.py` | Data download, warm-up + timing loop, COCO evaluation, charts, sample images, `results.json` |
| `figures.py` | Architecture diagram and flowchart (no data) |
| `demo.py` | Streamlit demo plus a command-line fallback |
| `build_report.py` | Builds the Word report from `results.json` |
| `build_slides.py` | Builds the PowerPoint deck from `results.json` |
| `data/` | Downloaded COCO images and `instances_val2017.json` |
| `weights/` | Downloaded model weights |
| `pyproject.toml`, `uv.lock` | Dependencies and their exact versions (managed by uv) |
| `results/` | `results.json`, charts, `sample1-4.jpg`, diagrams |

## Notes

* Latency = preprocessing + forward pass + NMS for one image at batch size 1, after 10 warm-up images.
  On GPU, `torch.cuda.synchronize()` runs before and after each timed call. Reading the image from disk is not timed.
* Accuracy = official COCO bbox evaluation (pycocotools), score threshold 0.001, at most 100 boxes per image.
* Only `annotations/instances_val2017.json` is extracted from the 241 MB COCO annotation zip, using HTTP
  range requests, so the zip is never stored on disk.
