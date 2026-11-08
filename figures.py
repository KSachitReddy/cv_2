"""
figures.py - draws the explanatory diagrams used in the report and slides (no data involved):
  results/fig_flowchart.png      benchmarking methodology flowchart
  results/fig_architectures.png  Faster R-CNN (two-stage) vs YOLOv8 (one-stage) pipelines
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8a85", "#222222"
FILL = {BLUE: "#e3eefb", ORANGE: "#fde9e0", GREY: "#f1f1ef"}


def box(ax, x, y, w, h, text, edge=GREY, fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL[edge], ec=edge, lw=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))


def flowchart():
    fig, ax = plt.subplots(figsize=(7, 8.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    steps = [
        ("Download & cache first N COCO val2017\nimages + instances_val2017.json", GREY),
        ("Detect hardware (CUDA GPU or CPU)\nchoose N and model list", GREY),
        ("Load pretrained model\n(Faster R-CNN via torchvision / YOLOv8 via Ultralytics)", GREY),
        ("Warm-up: run 10 images, not timed", GREY),
        ("For each image (batch size 1):\nsync -> start timer -> detect -> sync -> stop timer", GREY),
        ("Convert boxes to COCO format\n(YOLO class index -> COCO category id)", GREY),
        ("pycocotools COCOeval (bbox):\nmAP, mAP50, mAP75, mAP S/M/L", GREY),
        ("results.json -> charts, samples,\nreport and slides", GREY),
    ]
    ys = [11.2 - i * 1.45 for i in range(len(steps))]
    for (t, c), y in zip(steps, ys):
        box(ax, 5, y, 7.4, 1.0, t, c)
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, 5, a - 0.5, 5, b + 0.5)
    # loop arrow "next model"
    ax.annotate("", (8.75, ys[2]), (8.75, ys[6]),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4, connectionstyle="arc3,rad=0"))
    ax.plot([8.7, 8.75], [ys[6], ys[6]], color=BLUE, lw=1.4)
    ax.text(9.0, (ys[2] + ys[6]) / 2, "repeat for\nevery model", rotation=90, va="center",
            ha="center", fontsize=8.5, color=BLUE)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_flowchart.png"), dpi=170); plt.close(fig)


def architectures():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 22); ax.set_ylim(0, 10.4); ax.axis("off")
    # two-stage row
    ax.text(0.2, 9.9, "Two-stage: Faster R-CNN (ResNet-50 / MobileNetV3 + FPN)", fontsize=12,
            color=BLUE, weight="bold", va="center")
    row = [(1.6, "Image\n(resized,\nshort side 800)"), (4.9, "Backbone\nResNet-50 or\nMobileNetV3"),
           (8.2, "FPN\nmulti-scale\nfeature maps"), (11.5, "Stage 1: RPN\nanchors -> objectness\n+ box deltas, NMS"),
           (14.8, "RoIAlign\n~1000 proposals ->\nfixed 7x7 features"), (18.1, "Stage 2: box head\nclass scores +\nbox refinement"),
           (21.0, "NMS\nper class")]
    y = 7.6
    for i, (x, t) in enumerate(row):
        box(ax, x, y, 2.9 if i < 6 else 1.7, 2.0, t, BLUE, 8.5)
        if i:
            px = row[i - 1][0]
            arrow(ax, px + (1.45 if i - 1 < 6 else 0.85), y, x - (1.45 if i < 6 else 0.85), y, BLUE)
    ax.annotate("proposals", (14.8, 6.55), (11.5, 6.55), fontsize=8, color=BLUE, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1, connectionstyle="arc3,rad=0.35"))
    # one-stage row
    ax.text(0.2, 4.5, "One-stage: YOLOv8 (anchor-free, single forward pass)", fontsize=12,
            color=ORANGE, weight="bold", va="center")
    row2 = [(1.6, "Image\n(letterbox\n640x640)"), (5.4, "Backbone\nCSPDarknet\n(C2f blocks)"),
            (9.4, "Neck\nPAN-FPN\n3 scales (P3-P5)"), (13.6, "Decoupled head\nclass branch | box branch\nevery grid cell, anchor-free"),
            (17.7, "Decode boxes\n(DFL distances\nto 4 sides)"), (21.0, "NMS")]
    y = 2.3
    for i, (x, t) in enumerate(row2):
        w = 3.3 if 0 < i < 5 else (2.9 if i == 0 else 1.7)
        box(ax, x, y, w, 2.0, t, ORANGE, 8.5)
        if i:
            pw = 3.3 if 0 < i - 1 < 5 else 2.9
            arrow(ax, row2[i - 1][0] + pw / 2, y, x - w / 2, y, ORANGE)
    ax.text(11, 0.45, "No region proposals and no per-region second pass: every location predicts "
            "class + box at once, which is why it is faster.", ha="center", fontsize=9, color=INK, style="italic")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_architectures.png"), dpi=170); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    flowchart(); architectures()
    print("diagrams written to", OUT)


if __name__ == "__main__":
    main()
