"""
benchmark.py - Two-stage (Faster R-CNN) vs one-stage (YOLOv8) detectors on a COCO val2017 subset.

Steps
  1. Download (and cache) the first N COCO val2017 images + instances_val2017.json
  2. For each model: warm up, time every image (batch size 1), collect detections
  3. Score detections with the official COCO evaluator (pycocotools)
  4. Save results/results.json, the charts and 4 side-by-side example images

Usage
  uv run bench                        # N=500 on GPU, N=100 and no YOLOv8m on CPU
  uv run bench --n 200 --models YOLOv8n YOLOv8s
  uv run bench --charts-only          # redraw charts from results/results.json
"""
import argparse
import contextlib
import gc
import io
import json
import os
import platform
import sys
import urllib.request
import zipfile
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

import common
from common import MODELS, load_model, timed_detect

ROOT = common.ROOT
DATA = os.path.join(ROOT, "data")
IMG_DIR = os.path.join(DATA, "val2017")
ANN_FILE = os.path.join(DATA, "instances_val2017.json")
RES_DIR = os.path.join(ROOT, "results")
ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{}"
CPU_SKIP = ["YOLOv8m"]   # on CPU we skip the largest YOLO model to keep the run short


# ----------------------------------------------------------------------------- data download
class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable 'file' backed by HTTP Range requests.

    zipfile only needs seek/read, so this lets us pull the single 25 MB
    instances_val2017.json out of the 241 MB annotations zip without downloading all of it.
    """

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = [offset, self.pos + offset, self.size + offset][whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data


def get_annotations():
    if not os.path.exists(ANN_FILE):
        os.makedirs(DATA, exist_ok=True)
        print("Fetching instances_val2017.json from the COCO annotations zip (range requests)...")
        with zipfile.ZipFile(HttpRangeFile(ANN_ZIP_URL)) as z:
            with z.open("annotations/instances_val2017.json") as src, open(ANN_FILE + ".part", "wb") as dst:
                while chunk := src.read(4 << 20):
                    dst.write(chunk)
        os.replace(ANN_FILE + ".part", ANN_FILE)
    with contextlib.redirect_stdout(io.StringIO()):          # silence pycocotools prints
        from pycocotools.coco import COCO
        return COCO(ANN_FILE)


def get_images(coco, n):
    """First n image ids (sorted) - downloaded once, then read from data/val2017/."""
    os.makedirs(IMG_DIR, exist_ok=True)
    ids = sorted(coco.getImgIds())[:n]
    for img_id in tqdm(ids, desc="images", leave=False):
        fn = coco.imgs[img_id]["file_name"]
        path = os.path.join(IMG_DIR, fn)
        if not os.path.exists(path):
            urllib.request.urlretrieve(IMG_URL.format(fn), path + ".part")
            os.replace(path + ".part", path)
    return ids


# ----------------------------------------------------------------------------- evaluation
def coco_eval(coco, img_ids, dets):
    """Official COCO bbox metrics for our subset of images."""
    from pycocotools.cocoeval import COCOeval
    keys = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large"]
    if not dets:
        return {k: 0.0 for k in keys + ["AR100"]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco.loadRes(dets)
        ev = COCOeval(coco, coco_dt, "bbox")
        ev.params.imgIds = img_ids                 # only score the images we actually ran
        ev.evaluate(); ev.accumulate(); ev.summarize()
    out = {k: float(ev.stats[i]) for i, k in enumerate(keys)}
    out["AR100"] = float(ev.stats[8])
    return out


def run_model(name, device, coco, img_ids, warmup=10):
    info = load_model(name, device)
    paths = [os.path.join(IMG_DIR, coco.imgs[i]["file_name"]) for i in img_ids]

    # Warm-up: the first calls are slow (memory allocation, cuDNN autotuning, lazy init) -
    # they would distort the average latency, so they are run but not timed.
    for p in paths[:warmup]:
        timed_detect(info, cv2.imread(p))

    dets, lat, raw = [], [], {}
    for img_id, p in tqdm(list(zip(img_ids, paths)), desc=name):
        img = cv2.imread(p)                                   # disk I/O is NOT timed
        (boxes, scores, cids, names), ms = timed_detect(info, img)
        lat.append(ms)
        raw[img_id] = (boxes, scores, names)
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, cids):
            # COCO format wants [x, y, width, height]
            dets.append({"image_id": img_id, "category_id": int(c),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "score": float(s)})

    lat = np.array(lat)
    res = {"family": info["family"], "params_M": info["params"] / 1e6,
           "weight_file_MB": info["weight_mb"], "latency_ms_mean": float(lat.mean()),
           "latency_ms_std": float(lat.std()), "latency_ms_median": float(np.median(lat)),
           "fps": float(1000.0 / lat.mean()), "num_detections": len(dets)}
    res.update(coco_eval(coco, img_ids, dets))
    del info; gc.collect()
    return res, raw


# ----------------------------------------------------------------------------- qualitative samples
def pick_sample_images(coco, img_ids):
    """4 deterministic picks: most crowded, most small objects, and two ordinary scenes."""
    def anns(i):
        return coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
    n_obj = {i: len(anns(i)) for i in img_ids}
    n_small = {i: sum(a["area"] < 32 ** 2 for a in anns(i)) for i in img_ids}
    crowded = max(img_ids, key=lambda i: n_obj[i])
    small = max((i for i in img_ids if i != crowded), key=lambda i: n_small[i])
    rest = sorted((i for i in img_ids if i not in (crowded, small) and 3 <= n_obj[i] <= 8),
                  key=lambda i: n_obj[i])
    others = [rest[len(rest) // 3], rest[2 * len(rest) // 3]] if len(rest) >= 2 else img_ids[:2]
    return [("crowded scene", crowded), ("small objects", small),
            ("typical scene", others[0]), ("typical scene", others[1])], n_obj, n_small


def save_samples(coco, img_ids, raw_left, raw_right, left_name, right_name):
    picks, n_obj, n_small = pick_sample_images(coco, img_ids)
    meta = []
    for k, (why, img_id) in enumerate(picks, 1):
        img = cv2.imread(os.path.join(IMG_DIR, coco.imgs[img_id]["file_name"]))
        panels = []
        for name, raw, fam in ((left_name, raw_left, "two-stage"), (right_name, raw_right, "one-stage")):
            b, s, n = raw[img_id]
            kept = int((s >= 0.5).sum())
            panels.append(common.draw(img, b, s, n, 0.5, common.FAMILY_BGR[fam],
                                      f"{name}  ({kept} boxes, score>=0.5)"))
        gap = np.full((img.shape[0], 8, 3), 255, np.uint8)
        out = os.path.join(RES_DIR, f"sample{k}.jpg")
        cv2.imwrite(out, np.hstack([panels[0], gap, panels[1]]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        meta.append({"file": f"sample{k}.jpg", "image_id": img_id, "reason": why,
                     "gt_objects": n_obj[img_id], "gt_small_objects": n_small[img_id],
                     "left_boxes": int((raw_left[img_id][1] >= 0.5).sum()),
                     "right_boxes": int((raw_right[img_id][1] >= 0.5).sum())})
    return meta


# ----------------------------------------------------------------------------- charts
BLUE, ORANGE = "#2a78d6", "#eb6834"                        # two-stage, one-stage
SIZE_COLORS = ["#c4c4bf", "#8a8a85", "#4b4b47"]             # small, medium, large (neutral ramp: blue/orange mean family)
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e3"


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=30)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _family_legend(ax):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Two-stage (Faster R-CNN)"),
                       Patch(color=ORANGE, label="One-stage (YOLOv8)")],
              frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              borderaxespad=0.2)


def make_charts(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    models = results["models"]
    names = list(models)
    colors = [BLUE if models[m]["family"] == "two-stage" else ORANGE for m in names]
    short = [m.replace("FasterRCNN-", "FRCNN-\n").replace("-FPN-v2", "-FPN v2").replace("-FPN", "-FPN")
             for m in names]
    n_img, dev = results["meta"]["num_images"], results["meta"]["device_name"]
    sub = f"COCO val2017, first {n_img} images, {dev}"

    def bar_chart(key, title, ylabel, fname, fmt):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        vals = [models[m][key] for m in names]
        bars = ax.bar(short, vals, color=colors, width=0.6, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=9, color=INK)
        _style(ax, f"{title}\n{sub}", ylabel)
        ax.set_ylim(0, max(vals) * 1.18)
        _family_legend(ax)
        fig.tight_layout(); fig.savefig(os.path.join(RES_DIR, fname), dpi=160); plt.close(fig)

    bar_chart("mAP", "COCO mAP@[.5:.95] (higher is better)", "mAP", "chart_map.png", "{:.3f}")
    bar_chart("fps", "Throughput in frames per second (higher is better)", "FPS", "chart_fps.png", "{:.2f}")
    bar_chart("latency_ms_mean", "Mean latency per image, batch size 1 (lower is better)",
              "milliseconds", "chart_latency.png", "{:.0f}")

    # speed vs accuracy scatter
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m, c in zip(names, colors):
        r = models[m]
        ax.scatter(r["fps"], r["mAP"], s=40 + 6 * r["params_M"], color=c,
                   edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(m, (r["fps"], r["mAP"]), xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("FPS (log scale) - faster to the right", color=MUTED)
    _style(ax, f"Speed vs accuracy (marker area ~ parameter count)\n{sub}", "mAP@[.5:.95]")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    _family_legend(ax)
    xs = [models[m]["fps"] for m in names]
    ax.set_xlim(min(xs) / 1.6, max(xs) * 2.2)
    ys = [models[m]["mAP"] for m in names]
    ax.set_ylim(min(ys) - 0.05, max(ys) + 0.06)
    fig.tight_layout(); fig.savefig(os.path.join(RES_DIR, "chart_speed_vs_accuracy.png"), dpi=160)
    plt.close(fig)

    # per-object-size grouped bars
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(names)); w = 0.26
    for k, (key, lab) in enumerate([("mAP_small", "small (< 32² px)"),
                                    ("mAP_medium", "medium"), ("mAP_large", "large (> 96² px)")]):
        vals = [models[m][key] for m in names]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=SIZE_COLORS[k], label=lab,
                      edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, short)
    _style(ax, f"mAP by object size\n{sub}", "mAP@[.5:.95]")
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    ax.set_ylim(0, max(models[m]["mAP_large"] for m in names) * 1.25)
    fig.tight_layout(); fig.savefig(os.path.join(RES_DIR, "chart_map_by_size.png"), dpi=160)
    plt.close(fig)
    print("charts written to", RES_DIR)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="number of images (default 500 GPU / 100 CPU)")
    ap.add_argument("--models", nargs="+", default=None, choices=list(MODELS))
    ap.add_argument("--include-largest", action="store_true", help="on CPU, also run YOLOv8m")
    ap.add_argument("--charts-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)

    if args.charts_only:
        with open(os.path.join(RES_DIR, "results.json")) as f:
            make_charts(json.load(f))
        return

    device = common.get_device()
    on_cpu = device.type == "cpu"
    n = args.n or (100 if on_cpu else 500)
    models = args.models or [m for m in MODELS if not (on_cpu and not args.include_largest and m in CPU_SKIP)]
    notes = []
    if on_cpu:
        notes.append(f"No CUDA GPU found: ran on CPU with N={n} images"
                     + ("" if "YOLOv8m" in models else " and skipped YOLOv8m (largest YOLO model)") + ".")
    for msg in notes:
        print("NOTE:", msg)

    coco = get_annotations()
    # sanity check: YOLO index -> COCO id mapping equals the sorted category ids of the annotation file
    assert sorted(coco.getCatIds()) == common.COCO_IDS, "COCO category id list mismatch"
    img_ids = get_images(coco, n)

    results = {"meta": {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": device.type,
        "device_name": torch.cuda.get_device_name(0) if not on_cpu else f"CPU ({platform.processor() or platform.machine()}, {os.cpu_count()} threads)",
        "torch_threads": torch.get_num_threads(),
        "num_images": len(img_ids), "num_gt_boxes": len(coco.getAnnIds(imgIds=img_ids, iscrowd=False)),
        "dataset": "COCO val2017 (first N image ids, sorted)",
        "batch_size": 1, "warmup_images": 10, "eval_score_threshold": 0.001, "max_detections": 100,
        "input_size": {"torchvision": "short side 800 (max 1333)", "yolo": "640 letterbox"},
        "timing": "end-to-end per image: preprocessing + forward + post-processing (NMS); image decoding excluded",
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "torchvision": __import__("torchvision").__version__,
                     "ultralytics": __import__("ultralytics").__version__,
                     "pycocotools": __import__("importlib.metadata").metadata.version("pycocotools"), "opencv": cv2.__version__, "numpy": np.__version__},
        "notes": notes}, "models": {}}

    raws = {}
    for name in models:
        res, raw = run_model(name, device, coco, img_ids)
        results["models"][name] = res
        if name in ("FasterRCNN-R50-FPN-v2", "YOLOv8s"):
            raws[name] = raw
        print(f"{name:28s} mAP={res['mAP']:.3f}  mAP50={res['mAP50']:.3f}  "
              f"lat={res['latency_ms_mean']:.1f}ms  FPS={res['fps']:.2f}")
        with open(os.path.join(RES_DIR, "results.json"), "w") as f:   # save after every model
            json.dump(results, f, indent=2)

    if len(raws) == 2:
        results["samples"] = save_samples(coco, img_ids, raws["FasterRCNN-R50-FPN-v2"], raws["YOLOv8s"],
                                          "Faster R-CNN R50-FPN v2", "YOLOv8s")
    with open(os.path.join(RES_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    make_charts(results)


if __name__ == "__main__":
    main()
