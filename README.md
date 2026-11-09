# Project 26: Two-Stage (Faster R-CNN) vs One-Stage (YOLOv8) Detectors

> **DS307 Computer Vision — Assignment 2**  
> Faculty of Science & Technology, IFHE University, Hyderabad

An empirical benchmark, interactive intelligence dashboard, and neural pipeline simulator evaluating **Faster R-CNN** (two-stage, torchvision) against **YOLOv8** (one-stage, Ultralytics) under identical hardware conditions on the standard COCO val2017 dataset.

---

## 🎯 Key Benchmark Findings (COCO val2017, CPU)

| Family | Model | Parameters | Weight | Latency (Mean) | Frame Rate | COCO mAP | mAP Small (<32² px) | mAP Medium | mAP Large (>96² px) |
|---|---|---|---|---|---|---|---|---|---|
| **Two-Stage** | Faster R-CNN ResNet-50-FPN v2 | 43.7M | 175.2 MB | 4,411 ms | 0.23 FPS | **56.0%** | **40.1%** | **60.1%** | 74.1% |
| **Two-Stage** | Faster R-CNN MobileNetV3-FPN | 19.4M | 77.8 MB | 735 ms | 1.36 FPS | 39.1% | 20.5% | 45.0% | 61.8% |
| **One-Stage** | YOLOv8s | 11.2M | 22.6 MB | 133 ms | 7.52 FPS | **53.8%** | 36.6% | 56.8% | **74.4%** |
| **One-Stage** | YOLOv8n | **3.2M** | **6.5 MB** | **53 ms** | **18.69 FPS** | 45.3% | 29.0% | 47.0% | 69.9% |

### Core Architectural Conclusions:
1. **⚡ Speed Advantage:** YOLOv8n delivers an **~82.5x speedup** over Faster R-CNN ResNet-50 on CPU while using **93% fewer parameters** and **96% less storage**.
2. **🔍 Small Object Superiority:** Faster R-CNN ResNet-50 maintains an **11.1 percentage point lead in `mAP_small` (40.1% vs 29.0%)** due to:
   - Sub-pixel bilinear sampling in **RoIAlign** (eliminating coordinate quantization defects).
   - Higher native input resolution (800px short edge vs 640px letterbox).
   - Dedicated second-stage MLP box coordinate refinement.
3. **🎯 The Pareto Sweet Spot:** **YOLOv8s** delivers **53.8% mAP** (within 2.2% of Faster R-CNN) while achieving real-time responsiveness (**7.5+ FPS** on standard CPU).

---

## 🚀 Quickstart: The 5 Core Commands

Managed entirely with [uv](https://docs.astral.sh/uv/getting-started/installation/). Run from the project root:

```bash
uv run bench      # 1. Runs the benchmark on COCO val2017, calculates pycocotools mAP, writes results/
uv run report     # 2. Programmatically builds CV_Assignment2_Report.docx (all numbers read from results.json)
uv run slides     # 3. Builds CV_Assignment2_Slides.pptx (10 slides with speaker notes)
uv run build      # 4. Runs report + slides in one command
uv run demo       # 5. Launches the 7-Tab Vision Intelligence Studio (http://localhost:8501)
```

The first `uv run` automatically creates `.venv/` and synchronizes the exact environment from `uv.lock`.

---

## 🔬 Interactive Vision Intelligence Studio Features

The web application (`uv run demo`) features **7 dedicated interactive modules**:

1. **🔬 Live Detection Studio:**
   - 1-click sample presets (Crowded Street, Small Objects, Indoor Dining, Sports) or custom image/webcam upload.
   - Side-by-side detection cards with live speedup multipliers (`⚡ 82.5x FASTER`).
   - Interactive Plotly class distribution chart and raw coordinate inspector.

2. **🏁 The Live Model Race & Neural Simulator:**
   - **The Live Race:** Pits Faster R-CNN against YOLOv8 in real time. Watch YOLO finish in ~50ms while Faster R-CNN visibly chugs through its sequential proposal loop on screen!
   - **Animated Neural Visualizer:** Real laser scanline feature extraction, FPN heatmap bloom, RPN candidate proposal swarms, and RoIAlign zoom.
   - **Latency Waterfall:** Shows that **96% of Faster R-CNN's latency** is spent sequentially evaluating 1,000 RoIs in the second-stage MLP.

3. **🎯 Visual Consensus & Diff Map:**
   - Automated pairwise IoU bipartite matching.
   - Bounding boxes classified into:
     - 🟢 **Agreed (Green):** Detected by both models with exact IoU overlap.
     - 🔵 **Faster R-CNN Exclusive (Blue):** Small/occluded objects caught only by two-stage detection.
     - 🟠 **YOLO Exclusive (Orange):** Objects caught only by YOLO.
   - **Sub-Pixel Object Inspector:** Zoom into any detected object with sub-pixel magnification, COCO scale classification, and mathematical IoU formula calculations.
   - **Sensitivity Sweep:** Graphs bounding box retention across confidence thresholds ($0.1$ to $0.9$).

4. **📊 Interactive Benchmark & Analytics Studio:**
   - Dynamic Pareto frontier scatter plot ($mAP$ vs Latency).
   - Precision breakdown by object scale (Small, Medium, Large).
   - Throughput (FPS) comparison.

5. **🚀 Edge Hardware Deployment Advisor:**
   - Automated latency budget feasibility diagnosis across **Raspberry Pi 4/5**, **NVIDIA Jetson Orin Nano**, **Apple iPhone 15/16 (CoreML)**, **Laptop CPU**, and **RTX 4090**.
   - Target frame rate selector (1 to 120 FPS) with instant status (`✅ FEASIBLE` vs `❌ BOTTLENECK`).

6. **🧠 Architecture & Methodology Deep Dive:**
   - Two-Stage (RPN + RoIAlign) vs One-Stage (Anchor-Free Decoupled Head) pipeline comparisons.
   - Technical specification matrix and structural flowcharts.

7. **💬 CV Intelligence Assistant:**
   - Domain-specific chatbot grounded in the project's COCO benchmark metrics.
   - 1-click **"Summary for Professor"** prompt generating an academic defense summary.

---

## 📁 Repository Structure

| File / Folder | Purpose |
|---|---|
| `demo.py` | 7-tab Streamlit dashboard and CLI fallback (`python demo.py --image img.jpg`) |
| `common.py` | Unified model registry, device synchronization, and `timed_detect()` interface |
| `benchmark.py` | HTTP range request data fetcher, warm-up loop, pycocotools evaluator, and chart generator |
| `build_report.py` | Generates `CV_Assignment2_Report.docx` with dynamic numbers from `results.json` |
| `build_slides.py` | Generates `CV_Assignment2_Slides.pptx` (10 widescreen slides with speaker notes) |
| `figures.py` | Renders architectural block diagram (`fig_architectures.png`) and methodology flowchart |
| `tasks.py` | Entry points for short uv commands (`bench`, `report`, `slides`, `build`, `demo`) |
| `results/` | `results.json`, Matplotlib charts, sample annotations, and diagrams |
| `pyproject.toml` | Dependencies and scripts configuration |
| `uv.lock` | Cryptographically pinned dependency lockfile |

---

## 💡 Benchmark Engineering Highlights

* **HTTP Range Requests (`HttpRangeFile`):** Fetches the single 25 MB `instances_val2017.json` file directly from inside the 241 MB COCO annotation archive using HTTP byte-range seeks without downloading or storing the full archive.
* **Fair Latency Measurement:** Batch size 1, excluding disk I/O, with 10 warm-up images and GPU synchronization (`torch.cuda.synchronize()`).
* **Official Evaluation:** Uses standard COCO metric `pycocotools.cocoeval` across 10 IoU thresholds from 0.50 to 0.95.
