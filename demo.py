"""
demo.py - Ultimate Computer Vision Intelligence Studio, Neural Simulator & Production Suite
Comparing Two-Stage (Faster R-CNN) vs One-Stage (YOLOv8) Detectors

Features:
- Camera Device Switcher (DirectShow Laptop HD Webcam vs Phone / External Cam)
- Live Model Race Simulator (Real-time physical speed battle with stage progression)
- Animated Neural Pipeline Simulator (Laser scanline, FPN heatmap bloom, RPN swarm, RoIAlign zoom)
- SAHI (Slicing Aided Hyper Inference) for small-object detection enhancement
- Environmental Stress-Testing / Adversarial Robustness Suite (Rain, Fog, Night Mode, Motion Blur)
- Official COCO Ground Truth Overlay & Real-Time Error Matrix (TP / FP / FN, Precision, Recall, F1)
- Video File Processor (Upload MP4/AVI/MOV and render side-by-side video detection)
- Visual Consensus & Detection Diff Map (Pairwise IoU bipartite matching with agreed vs exclusive detections)
- Sub-Pixel Object-Level Inspector (Enlarged crop inspection with IoU overlap and COCO scale classification)
- Interactive Benchmark Analytics Studio with dynamic Plotly charts from results.json
- Edge Hardware Deployment & Production Feasibility Advisor (Jetson, Raspberry Pi, iPhone, RTX 4090)
- Architecture & Methodology Deep Dive (RPN vs Anchor-Free, Pareto frontier)
- 1-Click Export Suite (Download side-by-side composite, consensus map, and JSON metrics)
- Interactive CV Intelligence Chat Assistant with rich domain knowledge and quick inquiries
"""
import argparse
import io
import json
import os
import sys
import tempfile
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

import common

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")
DATA_DIR = os.path.join(ROOT, "data")
ANN_FILE = os.path.join(DATA_DIR, "instances_val2017.json")

TWO_STAGE = ["FasterRCNN-R50-FPN-v2", "FasterRCNN-MobileNetV3-FPN"]
ONE_STAGE = ["YOLOv8s", "YOLOv8n", "YOLOv8m"]
DEMO_SCORE = 0.25

SAMPLE_PRESETS = [
    {
        "id": "sample1",
        "name": "Sample 1: Crowded Street",
        "file": os.path.join(RESULTS_DIR, "sample1.jpg"),
        "desc": "Dense urban street with many overlapping pedestrians & vehicles (29 GT objects).",
        "coco_id": 9590,
        "icon": "🚶‍♂️"
    },
    {
        "id": "sample2",
        "name": "Sample 2: Small Objects",
        "file": os.path.join(RESULTS_DIR, "sample2.jpg"),
        "desc": "Challenging scene with distant and tiny objects (16 GT objects, 15 small).",
        "coco_id": 7511,
        "icon": "🔍"
    },
    {
        "id": "sample3",
        "name": "Sample 3: Indoor Dining",
        "file": os.path.join(RESULTS_DIR, "sample3.jpg"),
        "desc": "Typical indoor dining scene with utensils, table, and chairs (4 GT objects).",
        "coco_id": 7088,
        "icon": "🍽️"
    },
    {
        "id": "sample4",
        "name": "Sample 4: Sports & Action",
        "file": os.path.join(RESULTS_DIR, "sample4.jpg"),
        "desc": "Active sports movement with players and equipment (6 GT objects).",
        "coco_id": 8211,
        "icon": "🏃"
    }
]


def load_warm(name):
    """Load a model and run it once on a blank image so the first shown timing is realistic.
    Includes robust fallback in case of network/download errors.
    """
    try:
        info = common.load_model(name, common.get_device(), score_thresh=DEMO_SCORE)
        common.detect(info, np.zeros((480, 640, 3), np.uint8))
        return info
    except Exception as e:
        # Fallback to guaranteed local model
        fallback = "YOLOv8s" if name in ONE_STAGE else "FasterRCNN-MobileNetV3-FPN"
        try:
            info = common.load_model(fallback, common.get_device(), score_thresh=DEMO_SCORE)
            common.detect(info, np.zeros((480, 640, 3), np.uint8))
            info["warning"] = f"⚠️ Could not load requested model '{name}' ({e}). Automatically using offline fallback '{fallback}'."
            return info
        except Exception:
            raise e


def capture_laptop_camera_frame(cam_idx=1):
    """Capture a high-definition frame directly from the chosen camera index via DirectShow."""
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def apply_environmental_stress(img, mode='rain', intensity=0.5):
    """Simulate severe real-world environmental degradation on input image."""
    h, w = img.shape[:2]
    out = img.copy()

    if mode == "🌧️ Synthetic Rain":
        num_drops = int(1200 * intensity)
        for _ in range(num_drops):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = int(18 * intensity) + 5
            cv2.line(out, (x, y), (x + int(length * 0.35), y + length), (220, 220, 220), 1)

    elif mode == "🌫️ Atmospheric Fog / Haze":
        haze = np.full_like(img, 215)
        alpha = min(0.85, 0.65 * intensity + 0.1)
        out = cv2.addWeighted(img, 1.0 - alpha, haze, alpha, 0)

    elif mode == "🌙 Low-Light / Night Mode":
        factor = max(0.08, 1.0 - 0.85 * intensity)
        dark = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        noise = np.random.normal(0, 18 * intensity, dark.shape).astype(np.int16)
        out = np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif mode == "🌪️ Motion Blur / Camera Shake":
        k = max(3, int(29 * intensity))
        if k % 2 == 0:
            k += 1
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1.0 / k
        out = cv2.filter2D(img, -1, kernel)

    return out


def run_sahi_inference(model_info, image_bgr, score_thresh=0.25, iou_thresh=0.5, grid=(2, 2), overlap=0.2):
    """Slicing Aided Hyper Inference (SAHI) for fine-scale small object detection."""
    h, w = image_bgr.shape[:2]
    gh, gw = grid

    tile_w = int(w / (gw - (gw - 1) * overlap))
    tile_h = int(h / (gh - (gh - 1) * overlap))
    step_x = int(tile_w * (1 - overlap))
    step_y = int(tile_h * (1 - overlap))

    all_boxes, all_scores, all_names = [], [], []

    # 1. Global image pass
    (b_g, s_g, _, n_g), _ = common.timed_detect(model_info, image_bgr)
    keep_g = s_g >= score_thresh
    all_boxes.extend(b_g[keep_g])
    all_scores.extend(s_g[keep_g])
    all_names.extend([n_g[i] for i in range(len(s_g)) if keep_g[i]])

    # 2. Local sliced tiles
    y = 0
    while y < h:
        x = 0
        y_end = min(h, y + tile_h)
        while x < w:
            x_end = min(w, x + tile_w)
            tile = image_bgr[y:y_end, x:x_end]
            if tile.shape[0] > 60 and tile.shape[1] > 60:
                (b_t, s_t, _, n_t), _ = common.timed_detect(model_info, tile)
                keep_t = s_t >= score_thresh
                for b, s, n in zip(b_t[keep_t], s_t[keep_t], [n_t[i] for i in range(len(s_t)) if keep_t[i]]):
                    all_boxes.append([b[0] + x, b[1] + y, b[2] + x, b[3] + y])
                    all_scores.append(s)
                    all_names.append(n)
            x += step_x
            if x_end >= w:
                break
        y += step_y
        if y_end >= h:
            break

    if len(all_boxes) == 0:
        return np.empty((0, 4)), np.empty(0), []

    all_b = np.array(all_boxes, dtype=np.float32)
    all_s = np.array(all_scores, dtype=np.float32)

    indices = cv2.dnn.NMSBoxes(
        [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in all_b],
        all_s.tolist(),
        score_thresh,
        iou_thresh
    )
    if len(indices) > 0:
        idxs = indices.flatten()
        return all_b[idxs], all_s[idxs], [all_names[i] for i in idxs]
    return np.empty((0, 4)), np.empty(0), []


def run_pair_detailed(left, right, img, thr, use_sahi=False):
    """Run both detectors on one BGR image.
    Supports SAHI slicing for small-object enhancement.
    """
    outs, detections_list, raw_results, stats = [], [], [], []
    for info in (left, right):
        if use_sahi and info["family"] == "one-stage":
            t0 = time.perf_counter()
            boxes, scores, names = run_sahi_inference(info, img, score_thresh=thr)
            ms = (time.perf_counter() - t0) * 1000.0
            keep = scores >= thr
            filtered_boxes, filtered_scores, filtered_names = boxes, scores, names
        else:
            (boxes, scores, coco_ids, names), ms = common.timed_detect(info, img)
            keep = scores >= thr
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep]
            filtered_names = [names[i] for i in range(len(scores)) if keep[i]]

        annotated = common.draw(img, filtered_boxes, filtered_scores, filtered_names, thr, common.FAMILY_BGR[info["family"]])
        outs.append(annotated)
        raw_results.append((boxes, scores, names, filtered_boxes, filtered_scores, filtered_names))

        det_records = []
        for b, s, n in zip(filtered_boxes, filtered_scores, filtered_names):
            det_records.append({
                "model": info["name"] + (" (SAHI)" if (use_sahi and info["family"] == "one-stage") else ""),
                "family": info["family"],
                "class": n,
                "confidence": float(s),
                "box": [round(coord, 1) for coord in b.tolist()]
            })
        detections_list.append(det_records)

        stats.append({
            "name": info["name"] + (" (SAHI)" if (use_sahi and info["family"] == "one-stage") else ""),
            "family": info["family"],
            "ms": ms,
            "fps": 1000.0 / ms if ms > 0 else 0,
            "boxes": int(len(filtered_boxes)),
            "classes_counter": Counter(filtered_names),
            "params_M": info.get("params", 0) / 1e6,
            "weight_mb": info.get("weight_mb", 0)
        })

    return outs, detections_list, raw_results, stats


def compute_iou_matches(boxes1, boxes2, iou_thresh=0.5):
    """Pairwise IoU bipartite matching between two sets of bounding boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return [], list(range(len(boxes1))), list(range(len(boxes2)))

    b1 = np.array(boxes1)
    b2 = np.array(boxes2)

    x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / np.maximum(union, 1e-8)

    matches = []
    unmatched1 = set(range(len(boxes1)))
    unmatched2 = set(range(len(boxes2)))

    while True:
        if len(unmatched1) == 0 or len(unmatched2) == 0:
            break
        max_idx = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[max_idx] < iou_thresh:
            break
        i, j = int(max_idx[0]), int(max_idx[1])
        matches.append((i, j, float(iou[i, j])))
        unmatched1.discard(i)
        unmatched2.discard(j)
        iou[i, :] = -1
        iou[:, j] = -1

    return matches, sorted(list(unmatched1)), sorted(list(unmatched2))


def draw_consensus_map(img, b1, s1, n1, b2, s2, n2, matches, u1, u2):
    """Draw a visual consensus comparison overlay."""
    canvas = img.copy()

    for i, j, iou in matches:
        box = b1[i]
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
        tag = f"Agreed: {n1[i]} (IoU {iou:.2f})"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 160, 0), -1)
        cv2.putText(canvas, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    for i in u1:
        x1, y1, x2, y2 = [int(v) for v in b1[i]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (214, 120, 42), 2)
        tag = f"FRCNN only: {n1[i]} {s1[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), (214, 120, 42), -1)
        cv2.putText(canvas, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    for j in u2:
        x1, y1, x2, y2 = [int(v) for v in b2[j]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (52, 104, 235), 2)
        tag = f"YOLO only: {n2[j]} {s2[j]:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), (52, 104, 235), -1)
        cv2.putText(canvas, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def load_coco_ground_truth(coco_id):
    """Load official COCO ground truth bounding boxes and class names for a given image id."""
    if not os.path.exists(ANN_FILE):
        return None, None
    try:
        from pycocotools.coco import COCO
        coco = COCO(ANN_FILE)
        ann_ids = coco.getAnnIds(imgIds=coco_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        gt_names = []
        for a in anns:
            x, y, w, h = a["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_names.append(coco.cats[a["category_id"]]["name"])
        return np.array(gt_boxes, dtype=np.float32), gt_names
    except Exception:
        return None, None


def evaluate_against_gt(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresh=0.5):
    """Evaluate predicted boxes against human ground truth to find TP, FP, FN, Precision, Recall, F1."""
    if gt_boxes is None or len(gt_boxes) == 0:
        return {"tp": 0, "fp": len(pred_boxes), "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "matched_gt": []}
    if len(pred_boxes) == 0:
        return {"tp": 0, "fp": 0, "fn": len(gt_boxes), "precision": 0.0, "recall": 0.0, "f1": 0.0, "matched_gt": []}

    matched_gt = set()
    tp, fp = 0, 0

    for pb, pc in zip(pred_boxes, pred_classes):
        best_iou = 0.0
        best_gt_idx = -1
        for idx, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
            if pc == gc and idx not in matched_gt:
                x1 = max(pb[0], gb[0])
                y1 = max(pb[1], gb[1])
                x2 = min(pb[2], gb[2])
                y2 = min(pb[3], gb[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (pb[2] - pb[0]) * (pb[3] - pb[1])
                area2 = (gb[2] - gb[0]) * (gb[3] - gb[1])
                union = area1 + area2 - inter
                iou = inter / max(union, 1e-8)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "matched_gt": list(matched_gt)}


def draw_gt_overlay(img, gt_boxes, gt_names):
    """Draw ground truth boxes in gold."""
    canvas = img.copy()
    if gt_boxes is None:
        return canvas
    for (x1, y1, x2, y2), name in zip(gt_boxes, gt_names):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(canvas, p1, p2, (0, 215, 255), 2)  # Gold
        tag = f"GT: {name}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(canvas, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), (0, 180, 220), -1)
        cv2.putText(canvas, tag, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def load_benchmark_data():
    """Load saved benchmark results if available."""
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ----------------------------------------------------------------------------- Simulation Helpers
def extract_rpn_proposals(model_info, image_bgr, max_props=100):
    """Extract real RPN proposals directly from torchvision Faster R-CNN."""
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(model_info["device"])
        model = model_info["model"]
        images, _ = model.transform(tensor)
        features = model.backbone(images.tensors)
        proposals, _ = model.rpn(images, features)

        orig_h, orig_w = image_bgr.shape[:2]
        trans_h, trans_w = images.image_sizes[0]
        scale_x = orig_w / trans_w
        scale_y = orig_h / trans_h

        props = proposals[0][:max_props].detach().cpu().numpy().copy()
        props[:, [0, 2]] *= scale_x
        props[:, [1, 3]] *= scale_y
        return props, features
    except Exception:
        return np.empty((0, 4)), None


def extract_feature_heatmap(features, orig_shape):
    """Generate a heatmap from the first feature pyramid layer."""
    try:
        if features is None:
            return None
        f_key = '0' if '0' in features else list(features.keys())[0]
        f_map = features[f_key][0].detach().cpu().numpy()
        act = np.mean(f_map, axis=0)
        act = (act - act.min()) / (act.max() - act.min() + 1e-8)
        act_uint8 = np.uint8(255 * act)
        heatmap = cv2.applyColorMap(cv2.resize(act_uint8, (orig_shape[1], orig_shape[0])), cv2.COLORMAP_JET)
        return heatmap
    except Exception:
        return None


def draw_rpn_visualization(image_bgr, proposals, num_to_draw=50):
    """Draw RPN candidate proposals on the image with distinctive styling."""
    canvas = image_bgr.copy()
    num = min(len(proposals), num_to_draw)
    for i in range(num):
        x1, y1, x2, y2 = proposals[i].astype(int)
        color = (0, 255, 255) if i < 15 else (255, 200, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
    return canvas


def draw_roialign_demo(image_bgr, box):
    """Generate a close-up visualization of RoIAlign with a 7x7 bilinear sampling grid."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_bgr.shape[1], x2), min(image_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        crop = np.zeros((280, 280, 3), dtype=np.uint8)
    else:
        crop = image_bgr[y1:y2, x1:x2].copy()

    crop_resized = cv2.resize(crop, (280, 280), interpolation=cv2.INTER_NEAREST)
    cell = 280 / 7.0

    for i in range(8):
        pos = int(i * cell)
        cv2.line(crop_resized, (pos, 0), (pos, 280), (220, 220, 220), 1)
        cv2.line(crop_resized, (0, pos), (280, pos), (220, 220, 220), 1)

    for r in range(7):
        for c in range(7):
            cx = (c + 0.5) * cell
            cy = (r + 0.5) * cell
            offset = cell * 0.25
            for ox in (-offset, offset):
                for oy in (-offset, offset):
                    cv2.circle(crop_resized, (int(cx + ox), int(cy + oy)), 2, (0, 0, 255), -1)

    return crop_resized


def draw_yolo_grid_demo(image_bgr, stride=32, active_boxes=None):
    """Draw the multi-scale anchor-free feature grid on the image."""
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    grid_color = (180, 180, 180)
    for x in range(0, w, stride):
        cv2.line(canvas, (x, 0), (x, h), grid_color, 1)
    for y in range(0, h, stride):
        cv2.line(canvas, (0, y), (w, y), grid_color, 1)

    if active_boxes is not None and len(active_boxes) > 0:
        for b in active_boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            gx = (cx // stride) * stride
            gy = (cy // stride) * stride
            cv2.rectangle(canvas, (gx, gy), (gx + stride, gy + stride), (0, 140, 255), -1)
            cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.arrowedLine(canvas, (cx, cy), (x1, cy), (0, 255, 255), 1, tipLength=0.2)
            cv2.arrowedLine(canvas, (cx, cy), (x2, cy), (0, 255, 255), 1, tipLength=0.2)
            cv2.arrowedLine(canvas, (cx, cy), (cx, y1), (0, 255, 255), 1, tipLength=0.2)
            cv2.arrowedLine(canvas, (cx, cy), (cx, y2), (0, 255, 255), 1, tipLength=0.2)

    return canvas


# ----------------------------------------------------------------------------- Streamlit UI
def streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="R-CNN vs YOLO Vision Intelligence Studio",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #64748B;
            font-size: 1.05rem;
            margin-bottom: 1.2rem;
        }
        .badge-two-stage {
            background: #DBEAFE;
            color: #1E40AF;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        .badge-one-stage {
            background: #FEF3C7;
            color: #92400E;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        .head-to-head-banner {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            color: #F8FAFC;
            border-radius: 14px;
            padding: 1.2rem 1.6rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .step-box {
            background: #F8FAFC;
            border-left: 4px solid #3B82F6;
            padding: 12px 18px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

    dev = common.get_device()
    bench_data = load_benchmark_data()

    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown('<div class="main-title">🎯 Object Detection Intelligence Studio</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Two-Stage (Faster R-CNN) vs One-Stage (YOLOv8) Detector Benchmarking, SAHI Slicing & Production Advisor</div>', unsafe_allow_html=True)
    with col_t2:
        device_badge = f"🖥️ Hardware: **{dev.type.upper()}**"
        if dev.type == "cuda":
            device_badge = f"🚀 Hardware: **CUDA GPU** ({torch.cuda.get_device_name(0)})"
        st.info(device_badge)

    load = st.cache_resource(show_spinner="Loading model weights...")(load_warm)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Detector Configuration")
        left_name = st.selectbox("Two-Stage Model (High Precision)", TWO_STAGE, index=0)
        right_name = st.selectbox("One-Stage Model (High Speed)", ONE_STAGE, index=1)

        thr = st.slider("Detection Confidence Threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
        iou_match_thr = st.slider("Consensus IoU Matching Threshold", min_value=0.30, max_value=0.90, value=0.50, step=0.05)

        st.markdown("---")
        st.subheader("⚡ Advanced Extensions")
        use_sahi = st.checkbox("🔍 Enable SAHI Slicing (2x2 Grid)", value=False, help="Slices image into 2x2 tiles with overlap to boost YOLO small-object accuracy.")

        st.markdown("---")
        st.subheader("🌪️ Adversarial Stress Testing")
        enable_stress = st.checkbox("Enable Environmental Stress", value=False)
        stress_mode = "None"
        stress_intensity = 0.5
        if enable_stress:
            stress_mode = st.radio("Stress Mode:", ["🌧️ Synthetic Rain", "🌫️ Atmospheric Fog / Haze", "🌙 Low-Light / Night Mode", "🌪️ Motion Blur / Camera Shake"])
            stress_intensity = st.slider("Corruption Intensity", 0.1, 1.0, 0.5, 0.05)

        st.markdown("---")
        st.caption("DS307 Computer Vision • Assignment 2 • Project 26")

    # Main Tabs
    tab_playground, tab_sim, tab_consensus, tab_video, tab_benchmark, tab_deploy, tab_architecture, tab_chat = st.tabs([
        "🔬 Live Detection Studio",
        "🏁 Neural Simulator & Race",
        "🎯 Ground Truth & Consensus",
        "🎥 Video File Processor",
        "📊 Benchmark & Analytics",
        "🚀 Edge Deployment Advisor",
        "🧠 Architecture & Methodology",
        "💬 CV Intelligence Assistant"
    ])

    img = None
    selected_desc = ""
    active_coco_id = None

    # =========================================================================
    # TAB 1: LIVE DETECTION STUDIO
    # =========================================================================
    with tab_playground:
        st.subheader("Interactive Detection Playground")
        st.write("Select an instant sample preset, upload an image, or capture directly from your **Laptop Integrated HD Webcam** or phone camera.")

        input_mode = st.radio(
            "Choose Input Source:",
            [
                "✨ Instant Presets (Recommended)",
                "💻 Laptop / Phone Camera Snapshot",
                "🎥 Live Camera Stream",
                "📁 Upload Custom Image"
            ],
            horizontal=True
        )

        if input_mode == "✨ Instant Presets (Recommended)":
            p_cols = st.columns(4)
            chosen_preset_id = st.session_state.get("selected_preset", "sample1")

            for idx, p in enumerate(SAMPLE_PRESETS):
                with p_cols[idx]:
                    is_active = (chosen_preset_id == p["id"])
                    btn_label = f"{p['icon']} {p['name'].split(':')[0]}"
                    if is_active:
                        btn_label = f"👉 {btn_label}"
                    if st.button(btn_label, key=f"btn_{p['id']}", use_container_width=True):
                        st.session_state["selected_preset"] = p["id"]
                        chosen_preset_id = p["id"]
                    st.caption(p["desc"])

            for p in SAMPLE_PRESETS:
                if p["id"] == chosen_preset_id:
                    active_coco_id = p["coco_id"]
                    if os.path.exists(p["file"]):
                        img = cv2.imread(p["file"])
                        selected_desc = f"**{p['name']}**: {p['desc']}"
                    else:
                        img = np.zeros((480, 640, 3), np.uint8)

        elif input_mode == "💻 Laptop / Phone Camera Snapshot":
            cam_col1, cam_col2 = st.columns([2, 1])
            with cam_col1:
                cam_choice = st.selectbox(
                    "Select Active Camera Hardware:",
                    [
                        "💻 Camera 1: Laptop Built-in HD Webcam (1280x720 DirectShow)",
                        "📱 Camera 0: Phone / Primary Camera (DirectShow)"
                    ],
                    index=0
                )
                selected_cam_idx = 1 if "Camera 1" in cam_choice else 0
            with cam_col2:
                st.write("")
                st.write("")
                take_snap = st.button("📸 Capture Frame from Camera", type="primary", use_container_width=True)

            if take_snap:
                with st.spinner(f"Accessing Camera {selected_cam_idx}..."):
                    captured_frame = capture_laptop_camera_frame(selected_cam_idx)
                    if captured_frame is not None:
                        st.session_state["cam_snapshot_img"] = captured_frame
                        st.success(f"Frame captured successfully from Camera {selected_cam_idx} ({captured_frame.shape[1]}x{captured_frame.shape[0]} px)!")
                    else:
                        st.error(f"Could not open Camera {selected_cam_idx}. Make sure no other application is using it.")

            if "cam_snapshot_img" in st.session_state:
                img = st.session_state["cam_snapshot_img"]
                selected_desc = f"Live Snapshot from Camera {selected_cam_idx} ({img.shape[1]}x{img.shape[0]} px)"

        elif input_mode == "🎥 Live Camera Stream":
            st.info("Live streaming runs continuous inference frame-by-frame.")
            cam_str_choice = st.selectbox("Select Camera for Stream:", ["💻 Camera 1: Laptop Built-in HD Webcam", "📱 Camera 0: Phone / Primary Camera"])
            str_cam_idx = 1 if "Camera 1" in cam_str_choice else 0
            max_frames = st.slider("Max Frames to Stream", 10, 300, 60, 10)

            if st.button("▶️ Start Live Camera Stream", type="primary"):
                cap = cv2.VideoCapture(str_cam_idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(str_cam_idx)
                if not cap.isOpened():
                    st.error("Failed to connect to selected camera.")
                else:
                    left, right = load(left_name), load(right_name)
                    st.caption("Running stream... Click 'Stop' in top right to cancel.")
                    stream_cols = st.columns(2)
                    holders = [c.empty() for c in stream_cols]
                    text_holders = [c.empty() for c in stream_cols]
                    totals = [[], []]
                    for _ in range(max_frames):
                        ok, frame = cap.read()
                        if not ok:
                            break
                        outs, _, _, stats = run_pair_detailed(left, right, frame, thr, use_sahi=use_sahi)
                        for k in range(2):
                            totals[k].append(stats[k]["ms"])
                            holders[k].image(cv2.cvtColor(outs[k], cv2.COLOR_BGR2RGB), use_container_width=True)
                            avg_ms = np.mean(totals[k])
                            text_holders[k].markdown(
                                f"**{stats[k]['name']}** | {stats[k]['ms']:.0f} ms | "
                                f"Avg: **{1000.0/avg_ms:.1f} FPS** | Boxes: {stats[k]['boxes']}"
                            )
                    cap.release()
                    st.success("Webcam stream completed.")

        elif input_mode == "📁 Upload Custom Image":
            up = st.file_uploader("Upload an image (JPG, PNG, WebP):", type=["jpg", "jpeg", "png", "webp", "bmp"])
            if up:
                img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
                selected_desc = f"Custom upload: `{up.name}` ({img.shape[1]}x{img.shape[0]} px)"

        # Apply stress if enabled
        if img is not None and enable_stress:
            img = apply_environmental_stress(img, stress_mode, stress_intensity)
            selected_desc += f" • [Stress: {stress_mode} @ {stress_intensity*100:.0f}%]"

        # Process and show detection for static inputs
        if img is not None:
            left, right = load(left_name), load(right_name)
            if "warning" in left:
                st.warning(left["warning"])
            if "warning" in right:
                st.warning(right["warning"])
            outs, det_records, raw_results, stats = run_pair_detailed(left, right, img, thr, use_sahi=use_sahi)

            st.session_state["active_sim_image"] = img
            st.session_state["last_detection_stats"] = stats
            st.session_state["last_detection_records"] = det_records
            st.session_state["last_raw_results"] = raw_results
            st.session_state["last_outs"] = outs
            st.session_state["active_coco_id"] = active_coco_id

            ms_left = stats[0]["ms"]
            ms_right = stats[1]["ms"]
            speedup = ms_left / ms_right if ms_right > 0 else 1.0

            sahi_badge = " • <span style='color:#38BDF8; font-weight:600;'>[SAHI Slicing Enabled]</span>" if use_sahi else ""
            banner_html = f"""
            <div class="head-to-head-banner">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
                    <div>
                        <span style="font-size: 1.25rem; font-weight: 700;">⚔️ Head-to-Head Comparison</span>{sahi_badge}<br>
                        <span style="color: #94A3B8; font-size: 0.95rem;">{selected_desc}</span>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; padding: 8px 16px; border-radius: 8px;">
                        <span style="color: #34D399; font-weight: 700; font-size: 1.15rem;">⚡ {stats[1]['name']} is {speedup:.1f}x FASTER</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(banner_html, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<span class="badge-two-stage">TWO-STAGE</span> **{stats[0]["name"]}**', unsafe_allow_html=True)
                st.image(cv2.cvtColor(outs[0], cv2.COLOR_BGR2RGB), use_container_width=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latency", f"{stats[0]['ms']:.0f} ms")
                m2.metric("Frame Rate", f"{stats[0]['fps']:.1f} FPS")
                m3.metric("Detections", f"{stats[0]['boxes']} boxes")
                m4.metric("Parameters", f"{stats[0]['params_M']:.1f}M")

            with c2:
                st.markdown(f'<span class="badge-one-stage">ONE-STAGE</span> **{stats[1]["name"]}**', unsafe_allow_html=True)
                st.image(cv2.cvtColor(outs[1], cv2.COLOR_BGR2RGB), use_container_width=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latency", f"{stats[1]['ms']:.0f} ms")
                m2.metric("Frame Rate", f"{stats[1]['fps']:.1f} FPS")
                m3.metric("Detections", f"{stats[1]['boxes']} boxes")
                m4.metric("Parameters", f"{stats[1]['params_M']:.1f}M")

            st.markdown("---")
            st.subheader("🔍 Detection Breakdown & Category Distribution")

            counts_left = stats[0]["classes_counter"]
            counts_right = stats[1]["classes_counter"]
            all_classes = sorted(list(set(list(counts_left.keys()) + list(counts_right.keys()))))

            if all_classes:
                chart_df = pd.DataFrame([
                    {"Category": c, "Count": counts_left.get(c, 0), "Model": f"{stats[0]['name']}"}
                    for c in all_classes
                ] + [
                    {"Category": c, "Count": counts_right.get(c, 0), "Model": f"{stats[1]['name']}"}
                    for c in all_classes
                ])

                fig_classes = px.bar(
                    chart_df,
                    x="Category",
                    y="Count",
                    color="Model",
                    barmode="group",
                    title="Detected Object Counts by Category",
                    color_discrete_map={
                        f"{stats[0]['name']}": "#2563EB",
                        f"{stats[1]['name']}": "#F59E0B"
                    }
                )
                fig_classes.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_classes, use_container_width=True)

            st.markdown("---")
            st.subheader("💾 1-Click Export Suite")
            exp1, exp2 = st.columns(2)
            with exp1:
                gap = np.full((img.shape[0], 10, 3), 255, dtype=np.uint8)
                composite = np.hstack([outs[0], gap, outs[1]])
                _, comp_jpg = cv2.imencode(".jpg", composite)
                st.download_button(
                    "📥 Download Side-by-Side Annotated Image (JPG)",
                    data=comp_jpg.tobytes(),
                    file_name="side_by_side_detection.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            with exp2:
                report_dict = {
                    "models": [stats[0]["name"], stats[1]["name"]],
                    "threshold": thr,
                    "sahi_enabled": use_sahi,
                    "stress_mode": stress_mode,
                    "stats": stats,
                    "detections": det_records[0] + det_records[1]
                }
                st.download_button(
                    "📄 Download Detection Metrics (JSON)",
                    data=json.dumps(report_dict, indent=2),
                    file_name="detection_report.json",
                    mime="application/json",
                    use_container_width=True
                )

    # =========================================================================
    # TAB 2: NEURAL PIPELINE SIMULATOR & LIVE INFERENCE RACE
    # =========================================================================
    with tab_sim:
        st.subheader("🎮 Live Neural Pipeline Simulator & Model Race")
        st.write("Experience the physical speed and internal neural operations of Two-Stage vs One-Stage architectures through live animations and a side-by-side inference race.")

        sim_img = st.session_state.get("active_sim_image")
        if sim_img is None:
            if os.path.exists(SAMPLE_PRESETS[0]["file"]):
                sim_img = cv2.imread(SAMPLE_PRESETS[0]["file"])
            else:
                sim_img = np.zeros((480, 640, 3), np.uint8)

        st.markdown("### 🏁 1. The Live Model Race (Physical Speed Battle)")
        if st.button("🏁 Start Live Inference Race!", type="primary", use_container_width=True):
            r_col1, r_col2 = st.columns(2)
            left_model = load(left_name)
            right_model = load(right_name)

            with r_col1:
                st.markdown(f"**🔵 {left_name} (Two-Stage)**")
                p_left = st.progress(0, text="Waiting at start line...")
                t_left = st.empty()
                img_left = st.empty()
            with r_col2:
                st.markdown(f"**🟠 {right_name} (One-Stage)**")
                p_right = st.progress(0, text="Waiting at start line...")
                t_right = st.empty()
                img_right = st.empty()

            t0_yolo = time.perf_counter()
            p_right.progress(20, text="Unified dense regression pass (P3-P5)...")
            time.sleep(0.04)
            p_right.progress(70, text="Anchor-free decoupled head...")
            (b_r, s_r, _, n_r), ms_r = common.timed_detect(right_model, sim_img)
            p_right.progress(100, text=f"🏆 FINISHED in {ms_r:.0f} ms ({1000/ms_r:.1f} FPS)!")
            t_right.success(f"⚡ **YOLO Finished in {ms_r:.0f} ms!**")
            out_yolo = common.draw(sim_img, b_r, s_r, n_r, thr, common.FAMILY_BGR["one-stage"])
            img_right.image(cv2.cvtColor(out_yolo, cv2.COLOR_BGR2RGB), use_container_width=True)

            p_left.progress(10, text="Stage 1: Resizing & FPN Feature Maps...")
            time.sleep(0.08)
            p_left.progress(25, text="Stage 1: RPN sliding anchor boxes (proposing 1,000 RoIs)...")
            time.sleep(0.12)

            for roi_pct in [40, 55, 70, 85]:
                p_left.progress(roi_pct, text=f"Stage 2: Sequential RoIAlign & MLP Head evaluation (RoI {int(roi_pct*10)}/1000)...")
                time.sleep(0.15)

            (b_l, s_l, _, n_l), ms_l = common.timed_detect(left_model, sim_img)
            p_left.progress(100, text=f"DONE in {ms_l:.0f} ms ({1000/ms_l:.1f} FPS).")
            t_left.info(f"🐢 **Faster R-CNN Finished in {ms_l:.0f} ms**")
            out_rcnn = common.draw(sim_img, b_l, s_l, n_l, thr, common.FAMILY_BGR["two-stage"])
            img_left.image(cv2.cvtColor(out_rcnn, cv2.COLOR_BGR2RGB), use_container_width=True)

            speed_diff = ms_l / ms_r if ms_r > 0 else 1.0
            st.success(f"🎉 **Race Verdict:** {right_name} crossed the finish line **{speed_diff:.1f}x faster** than {left_name}!")

        st.markdown("---")
        st.markdown("### 🎬 2. Animated Layer-by-Layer Neural Pipeline")

        sim_col1, sim_col2 = st.columns([1, 1])
        with sim_col1:
            chosen_pipeline = st.radio("Choose Neural Pipeline to Animate:", ["🔵 Faster R-CNN (Two-Stage)", "🟠 YOLOv8 (One-Stage)"], horizontal=True)
        with sim_col2:
            anim_speed = st.select_slider("Animation Playback Speed:", options=["Slow (Educational)", "Normal", "Fast"], value="Normal")

        delay = 0.12 if anim_speed == "Slow (Educational)" else (0.07 if anim_speed == "Normal" else 0.03)

        if st.button("▶️ Play Live Neural Animation", use_container_width=True):
            anim_placeholder = st.empty()
            caption_placeholder = st.empty()
            h, w = sim_img.shape[:2]

            if "Faster R-CNN" in chosen_pipeline:
                caption_placeholder.info("Stage 1.1: Backbone Convolutional Scanline Feature Extraction...")
                for y in range(0, h, max(20, h // 8)):
                    f = sim_img.copy()
                    cv2.line(f, (0, y), (w, y), (0, 255, 255), 3)
                    anim_placeholder.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), use_container_width=True)
                    time.sleep(delay)

                caption_placeholder.info("Stage 1.2: Feature Pyramid Network (FPN) Multi-scale activations...")
                sim_model = load(left_name)
                _, features = extract_rpn_proposals(sim_model, sim_img, max_props=10)
                heatmap = extract_feature_heatmap(features, sim_img.shape)
                if heatmap is not None:
                    for alpha in [0.2, 0.4, 0.6]:
                        blended = cv2.addWeighted(sim_img, 1.0 - alpha, heatmap, alpha, 0)
                        anim_placeholder.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), use_container_width=True)
                        time.sleep(delay)

                caption_placeholder.info("Stage 1.3: Region Proposal Network (RPN) anchor evaluation & candidate proposal swarm...")
                props, _ = extract_rpn_proposals(sim_model, sim_img, max_props=80)
                for k in [15, 35, 60, 80]:
                    p_canvas = draw_rpn_visualization(sim_img, props, num_to_draw=k)
                    anim_placeholder.image(cv2.cvtColor(p_canvas, cv2.COLOR_BGR2RGB), use_container_width=True)
                    time.sleep(delay)

                caption_placeholder.info("Stage 2: RoIAlign Bilinear Extraction & Box Head Refinement...")
                (b, s, _, n), _ = common.timed_detect(sim_model, sim_img)
                final_canvas = common.draw(sim_img, b, s, n, thr, common.FAMILY_BGR["two-stage"])
                anim_placeholder.image(cv2.cvtColor(final_canvas, cv2.COLOR_BGR2RGB), use_container_width=True)
                caption_placeholder.success("✅ Faster R-CNN Pipeline Complete: High-precision localization achieved.")

            else:
                caption_placeholder.info("Stage 1: Letterbox 640x640 Fixed Geometry Preprocessing...")
                r = min(640 / h, 640 / w)
                nh, nw = int(h * r), int(w * r)
                resized = cv2.resize(sim_img, (nw, nh))
                letterbox = np.full((640, 640, 3), 114, dtype=np.uint8)
                dx, dy = (640 - nw) // 2, (640 - nh) // 2
                letterbox[dy:dy+nh, dx:dx+nw] = resized
                anim_placeholder.image(cv2.cvtColor(letterbox, cv2.COLOR_BGR2RGB), use_container_width=True)
                time.sleep(delay * 2)

                for stride in [32, 16, 8]:
                    caption_placeholder.info(f"Stage 2: Multi-scale Feature Pyramid Grid Stride {stride} (Stride 8 = small, 16 = med, 32 = large)...")
                    grid_canvas = draw_yolo_grid_demo(sim_img, stride=stride)
                    anim_placeholder.image(cv2.cvtColor(grid_canvas, cv2.COLOR_BGR2RGB), use_container_width=True)
                    time.sleep(delay)

                caption_placeholder.info("Stage 3: Concurrent Single-Pass Anchor-Free Center Point Regression...")
                sim_yolo = load(right_name)
                (b, s, _, n), _ = common.timed_detect(sim_yolo, sim_img)
                keep = s >= thr
                active_boxes = b[keep]
                active_canvas = draw_yolo_grid_demo(sim_img, stride=32, active_boxes=active_boxes)
                anim_placeholder.image(cv2.cvtColor(active_canvas, cv2.COLOR_BGR2RGB), use_container_width=True)
                time.sleep(delay * 2)

                caption_placeholder.info("Stage 4: Fast Vectorized NMS in < 3 ms...")
                final_yolo = common.draw(sim_img, b, s, n, thr, common.FAMILY_BGR["one-stage"])
                anim_placeholder.image(cv2.cvtColor(final_yolo, cv2.COLOR_BGR2RGB), use_container_width=True)
                caption_placeholder.success("✅ YOLOv8 Pipeline Complete: Real-time throughput in one single pass.")

    # =========================================================================
    # TAB 3: GROUND TRUTH & VISUAL CONSENSUS
    # =========================================================================
    with tab_consensus:
        st.subheader("🎯 Ground Truth Evaluation, Error Matrix & Consensus Diff")
        st.write("Compare predictions against **Official COCO Human Ground Truth** (TP / FP / FN, Precision, Recall, F1) and inspect model agreement.")

        curr_img = st.session_state.get("active_sim_image")
        curr_raw = st.session_state.get("last_raw_results")
        curr_coco_id = st.session_state.get("active_coco_id")

        if curr_img is not None and curr_raw is not None:
            b1, s1, n1 = curr_raw[0][3], curr_raw[0][4], curr_raw[0][5]
            b2, s2, n2 = curr_raw[1][3], curr_raw[1][4], curr_raw[1][5]

            # 1. Ground Truth Section if available
            gt_boxes, gt_names = load_coco_ground_truth(curr_coco_id) if curr_coco_id else (None, None)
            if gt_boxes is not None:
                st.markdown("#### 🏆 Official Ground Truth Evaluation (COCO val2017)")
                eval1 = evaluate_against_gt(b1, n1, gt_boxes, gt_names, iou_thresh=0.50)
                eval2 = evaluate_against_gt(b2, n2, gt_boxes, gt_names, iou_thresh=0.50)

                gt_cols = st.columns(4)
                gt_cols[0].metric("Ground Truth Objects", f"{len(gt_boxes)} objects")
                gt_cols[1].metric(f"{left_name.split('-')[0]} F1-Score", f"{eval1['f1']*100:.1f}%", f"P: {eval1['precision']*100:.0f}%, R: {eval1['recall']*100:.0f}%")
                gt_cols[2].metric(f"{right_name} F1-Score", f"{eval2['f1']*100:.1f}%", f"P: {eval2['precision']*100:.0f}%, R: {eval2['recall']*100:.0f}%")
                gt_cols[3].metric("TP / FP / FN (FRCNN vs YOLO)", f"TP: {eval1['tp']} vs {eval2['tp']}", f"Misses: {eval1['fn']} vs {eval2['fn']}")

                show_gt = st.checkbox("Show Gold Ground Truth Boxes on Canvas", value=True)
                gt_canvas = draw_gt_overlay(curr_img, gt_boxes, gt_names) if show_gt else curr_img
                st.image(cv2.cvtColor(gt_canvas, cv2.COLOR_BGR2RGB), caption=f"Ground Truth Annotations ({len(gt_boxes)} human-verified labels)", use_container_width=True)
                st.markdown("---")

            # 2. Consensus Diff Map
            st.markdown("#### ⚔️ Consensus Diff Map (Faster R-CNN vs YOLOv8)")
            matches, u1, u2 = compute_iou_matches(b1, b2, iou_thresh=iou_match_thr)
            total_unique = len(matches) + len(u1) + len(u2)
            consensus_rate = (len(matches) / total_unique * 100) if total_unique > 0 else 0
            mean_iou = np.mean([m[2] for m in matches]) if len(matches) > 0 else 0.0

            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Agreed Detections", f"{len(matches)} objects", f"{consensus_rate:.1f}% consensus")
            cm2.metric("Mean Agreement IoU", f"{mean_iou:.2f}", "Overlap precision")
            cm3.metric(f"Exclusive to {left_name.split('-')[0]}", f"{len(u1)} objects", "Missed by YOLO")
            cm4.metric(f"Exclusive to {right_name}", f"{len(u2)} objects", "Missed by FRCNN")

            consensus_canvas = draw_consensus_map(curr_img, b1, s1, n1, b2, s2, n2, matches, u1, u2)
            st.image(cv2.cvtColor(consensus_canvas, cv2.COLOR_BGR2RGB),
                     caption="Consensus Overlay: Green = Agreed | Blue = Faster R-CNN only | Orange = YOLO only",
                     use_container_width=True)

            # Sub-Pixel Object Zoom
            st.markdown("---")
            st.subheader("🔍 Sub-Pixel Object Crop Inspector")
            object_options = []
            for idx, (i, j, iou_score) in enumerate(matches):
                object_options.append(f"🟢 Agreed Object #{idx+1}: {n1[i]} (IoU {iou_score:.2f})")
            for idx, i in enumerate(u1):
                object_options.append(f"🔵 Faster R-CNN Only #{idx+1}: {n1[i]} ({s1[i]*100:.0f}%)")
            for idx, j in enumerate(u2):
                object_options.append(f"🟠 YOLO Only #{idx+1}: {n2[j]} ({s2[j]*100:.0f}%)")

            if object_options:
                chosen_obj = st.selectbox("Choose Detected Object to Zoom In:", object_options)
                t_box = None
                t_meta = {}
                if chosen_obj.startswith("🟢"):
                    match_idx = int(chosen_obj.split("#")[1].split(":")[0]) - 1
                    i, j, iou_val = matches[match_idx]
                    t_box = b1[i]
                    t_meta = {"name": n1[i], "frcnn_score": s1[i], "yolo_score": s2[j], "iou": iou_val, "status": "Agreed by Both"}
                elif chosen_obj.startswith("🔵"):
                    u_idx = int(chosen_obj.split("#")[1].split(":")[0]) - 1
                    i = u1[u_idx]
                    t_box = b1[i]
                    t_meta = {"name": n1[i], "frcnn_score": s1[i], "yolo_score": 0.0, "iou": 0.0, "status": "Faster R-CNN Exclusive"}
                else:
                    u_idx = int(chosen_obj.split("#")[1].split(":")[0]) - 1
                    j = u2[u_idx]
                    t_box = b2[j]
                    t_meta = {"name": n2[j], "frcnn_score": 0.0, "yolo_score": s2[j], "iou": 0.0, "status": "YOLO Exclusive"}

                if t_box is not None:
                    pad = 20
                    x1, y1, x2, y2 = [int(v) for v in t_box]
                    h_im, w_im = curr_img.shape[:2]
                    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                    cx2, cy2 = min(w_im, x2 + pad), min(h_im, y2 + pad)
                    obj_crop = curr_img[cy1:cy2, cx1:cx2].copy()

                    area_px = (x2 - x1) * (y2 - y1)
                    scale_cat = "Small (<32² px)" if area_px < 32**2 else ("Medium (32²-96² px)" if area_px < 96**2 else "Large (>96² px)")

                    z_col1, z_col2 = st.columns([1, 1])
                    with z_col1:
                        st.image(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB), caption=f"Magnified Crop: {t_meta['name']}", use_container_width=True)
                    with z_col2:
                        st.markdown(f"**Category:** `{t_meta['name']}`")
                        st.markdown(f"**Consensus Status:** `{t_meta['status']}`")
                        st.markdown(f"**COCO Object Scale:** **{scale_cat}** ({area_px:.0f} px²)")
                        if t_meta['iou'] > 0:
                            st.markdown(f"**Bounding Box IoU Overlap:** **{t_meta['iou']:.3f}**")
                            st.latex(r"\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = " + f"{t_meta['iou']:.2f}")
                        st.markdown(f"• **Faster R-CNN Confidence:** `{t_meta['frcnn_score']*100:.1f}%`")
                        st.markdown(f"• **YOLOv8 Confidence:** `{t_meta['yolo_score']*100:.1f}%`")
        else:
            st.info("Run detection in the 'Live Detection Studio' tab first.")

    # =========================================================================
    # TAB 4: VIDEO FILE PROCESSOR
    # =========================================================================
    with tab_video:
        st.subheader("🎥 Video File Side-by-Side Batch Processor")
        st.write("Upload an MP4, AVI, or MOV video clip (dashcam, drone footage, sports) to process and render a side-by-side detection video.")

        video_file = st.file_uploader("Upload Video File:", type=["mp4", "avi", "mov", "mkv"])
        frame_skip = st.slider("Process Every Nth Frame (Speed Optimization):", 1, 5, 2)
        max_vid_frames = st.slider("Max Frames to Process:", 10, 200, 50, 10)

        if video_file and st.button("▶️ Process Video Side-by-Side", type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            st.write(f"Source video: {total_frames} frames, {fps_src:.1f} FPS.")
            prog_bar = st.progress(0, text="Processing frames...")
            vid_col = st.empty()

            left_m, right_m = load(left_name), load(right_name)
            frame_idx = 0
            processed = 0

            while cap.isOpened() and processed < max_vid_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    outs, _, _, _ = run_pair_detailed(left_m, right_m, frame, thr)
                    gap = np.full((frame.shape[0], 8, 3), 255, dtype=np.uint8)
                    comp_frame = np.hstack([outs[0], gap, outs[1]])
                    vid_col.image(cv2.cvtColor(comp_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    processed += 1
                    prog_bar.progress(processed / max_vid_frames, text=f"Processed frame {processed}/{max_vid_frames}...")
                frame_idx += 1

            cap.release()
            try:
                os.unlink(tfile.name)
            except Exception:
                pass
            st.success(f"Video processing finished ({processed} frames evaluated).")

    # =========================================================================
    # TAB 5: INTERACTIVE BENCHMARK & ANALYTICS STUDIO
    # =========================================================================
    with tab_benchmark:
        st.subheader("📊 Comprehensive COCO val2017 Benchmark")
        st.write("Quantitative analysis evaluating speed, mean Average Precision (mAP), and model footprint under identical hardware conditions.")

        if bench_data and "models" in bench_data:
            models_dict = bench_data["models"]
            meta = bench_data.get("meta", {})

            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Evaluated Images", meta.get("num_images", 100))
            b_col2.metric("Ground Truth Boxes", meta.get("num_gt_boxes", 709))
            b_col3.metric("Batch Size", meta.get("batch_size", 1))
            b_col4.metric("Benchmark Device", meta.get("device_name", "CPU"))

            bench_rows = []
            for m_name, m_stats in models_dict.items():
                bench_rows.append({
                    "Model": m_name,
                    "Family": m_stats.get("family", "Unknown"),
                    "mAP [0.50:0.95]": m_stats.get("mAP", 0) * 100,
                    "mAP50": m_stats.get("mAP50", 0) * 100,
                    "mAP75": m_stats.get("mAP75", 0) * 100,
                    "mAP_Small": m_stats.get("mAP_small", 0) * 100,
                    "mAP_Medium": m_stats.get("mAP_medium", 0) * 100,
                    "mAP_Large": m_stats.get("mAP_large", 0) * 100,
                    "Latency (ms)": m_stats.get("latency_ms_mean", 0),
                    "Latency Std (ms)": m_stats.get("latency_ms_std", 0),
                    "FPS": m_stats.get("fps", 0),
                    "Params (M)": m_stats.get("params_M", 0),
                    "Weight (MB)": m_stats.get("weight_file_MB", 0)
                })
            df_bench = pd.DataFrame(bench_rows)

            st.markdown("#### 1. Speed vs Accuracy Trade-off (Pareto Frontier)")
            fig_pareto = px.scatter(
                df_bench,
                x="Latency (ms)",
                y="mAP [0.50:0.95]",
                color="Family",
                size="Weight (MB)",
                hover_name="Model",
                text="Model",
                title="mAP vs Inference Latency (Bubble size = Model Weight MB)",
                color_discrete_map={"two-stage": "#2563EB", "one-stage": "#F59E0B"},
                labels={"mAP [0.50:0.95]": "mAP (%)", "Latency (ms)": "Latency (ms, log scale)"},
                log_x=True
            )
            fig_pareto.update_traces(textposition='top right', marker=dict(opacity=0.85, line=dict(width=1, color='DarkSlateGrey')))
            fig_pareto.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pareto, use_container_width=True)

            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                st.markdown("#### 2. Accuracy Breakdown by Object Scale")
                scale_df = df_bench.melt(
                    id_vars=["Model", "Family"],
                    value_vars=["mAP_Small", "mAP_Medium", "mAP_Large"],
                    var_name="Object Scale",
                    value_name="mAP (%)"
                )
                scale_df["Object Scale"] = scale_df["Object Scale"].map({
                    "mAP_Small": "Small (<32² px)",
                    "mAP_Medium": "Medium (32²-96² px)",
                    "mAP_Large": "Large (>96² px)"
                })
                fig_scale = px.bar(
                    scale_df,
                    x="Model",
                    y="mAP (%)",
                    color="Object Scale",
                    barmode="group",
                    title="Detection Accuracy across Object Scales",
                    color_discrete_sequence=["#93C5FD", "#3B82F6", "#1D4ED8"]
                )
                fig_scale.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_scale, use_container_width=True)

            with col_ch2:
                st.markdown("#### 3. Inference Throughput & Latency")
                fig_fps = px.bar(
                    df_bench,
                    x="Model",
                    y="FPS",
                    color="Family",
                    text="FPS",
                    title="Inference Speed (Frames Per Second - Higher is Better)",
                    color_discrete_map={"two-stage": "#2563EB", "one-stage": "#F59E0B"}
                )
                fig_fps.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig_fps.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_fps, use_container_width=True)

            st.markdown("#### 📋 Complete Benchmark Summary Table")
            st.dataframe(
                df_bench.style.format({
                    "mAP [0.50:0.95]": "{:.1f}%",
                    "mAP50": "{:.1f}%",
                    "mAP75": "{:.1f}%",
                    "mAP_Small": "{:.1f}%",
                    "mAP_Medium": "{:.1f}%",
                    "mAP_Large": "{:.1f}%",
                    "Latency (ms)": "{:.1f}",
                    "Latency Std (ms)": "{:.1f}",
                    "FPS": "{:.2f}",
                    "Params (M)": "{:.2f}",
                    "Weight (MB)": "{:.1f}"
                }),
                use_container_width=True
            )
        else:
            st.warning("Benchmark results file not found.")

    # =========================================================================
    # TAB 6: EDGE HARDWARE DEPLOYMENT ADVISOR
    # =========================================================================
    with tab_deploy:
        st.subheader("🚀 Edge Hardware Deployment & Production Feasibility Advisor")
        st.write("Select your deployment platform and real-world latency requirements to get an automated architectural feasibility diagnosis.")

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            hw_choice = st.selectbox(
                "Target Deployment Hardware:",
                [
                    "🤖 Raspberry Pi 4 / 5 (Broadcom Quad-Core ARM CPU)",
                    "🚗 NVIDIA Jetson Orin Nano (40 TOPS Edge AI)",
                    "📱 Apple iPhone 15/16 (Apple Neural Engine / CoreML)",
                    "💻 Intel Core i7 / i9 Laptop CPU (x86_64 8-threads)",
                    "⚡ NVIDIA RTX 4090 / Cloud GPU (TensorRT FP16)"
                ]
            )
        with d_col2:
            target_fps = st.select_slider("Target Frame Rate Requirement:", options=[1, 5, 15, 30, 60, 120], value=30)

        budget_ms = 1000.0 / target_fps
        st.write(f"⏱️ **Target Latency Budget:** $\\le$ **{budget_ms:.1f} ms per frame** ({target_fps} FPS)")

        hw_profiles = {
            "Raspberry Pi 4 / 5": {
                "YOLOv8n": {"ms": 36.0, "fps": 27.8, "ram": "150 MB", "opt": "NCNN / ONNX INT8"},
                "YOLOv8s": {"ms": 95.0, "fps": 10.5, "ram": "320 MB", "opt": "ONNX INT8"},
                "Faster R-CNN MobileNetV3": {"ms": 680.0, "fps": 1.47, "ram": "650 MB", "opt": "PyTorch CPU"},
                "Faster R-CNN R50": {"ms": 5200.0, "fps": 0.19, "ram": "1.4 GB", "opt": "PyTorch CPU"}
            },
            "NVIDIA Jetson Orin Nano": {
                "YOLOv8n": {"ms": 4.2, "fps": 238.0, "ram": "220 MB", "opt": "TensorRT FP16"},
                "YOLOv8s": {"ms": 9.5, "fps": 105.0, "ram": "410 MB", "opt": "TensorRT FP16"},
                "Faster R-CNN MobileNetV3": {"ms": 18.0, "fps": 55.5, "ram": "550 MB", "opt": "TensorRT FP16"},
                "Faster R-CNN R50": {"ms": 36.0, "fps": 27.8, "ram": "980 MB", "opt": "TensorRT FP16"}
            },
            "Apple iPhone 15/16": {
                "YOLOv8n": {"ms": 3.8, "fps": 263.0, "ram": "180 MB", "opt": "CoreML ANE INT8"},
                "YOLOv8s": {"ms": 7.4, "fps": 135.0, "ram": "290 MB", "opt": "CoreML ANE FP16"},
                "Faster R-CNN MobileNetV3": {"ms": 16.5, "fps": 60.6, "ram": "420 MB", "opt": "CoreML"},
                "Faster R-CNN R50": {"ms": 65.0, "fps": 15.4, "ram": "890 MB", "opt": "CoreML"}
            },
            "Intel Core i7 / i9 Laptop CPU": {
                "YOLOv8n": {"ms": 24.0, "fps": 41.7, "ram": "310 MB", "opt": "OpenVINO FP16"},
                "YOLOv8s": {"ms": 58.0, "fps": 17.2, "ram": "520 MB", "opt": "OpenVINO FP16"},
                "Faster R-CNN MobileNetV3": {"ms": 280.0, "fps": 3.57, "ram": "780 MB", "opt": "PyTorch CPU"},
                "Faster R-CNN R50": {"ms": 1400.0, "fps": 0.71, "ram": "1.8 GB", "opt": "PyTorch CPU"}
            },
            "NVIDIA RTX 4090 / Cloud GPU": {
                "YOLOv8n": {"ms": 1.1, "fps": 909.0, "ram": "500 MB VRAM", "opt": "TensorRT FP16"},
                "YOLOv8s": {"ms": 2.3, "fps": 434.0, "ram": "850 MB VRAM", "opt": "TensorRT FP16"},
                "Faster R-CNN MobileNetV3": {"ms": 6.5, "fps": 153.8, "ram": "1.2 GB VRAM", "opt": "TensorRT FP16"},
                "Faster R-CNN R50": {"ms": 14.5, "fps": 69.0, "ram": "2.2 GB VRAM", "opt": "TensorRT FP16"}
            }
        }

        chosen_key = [k for k in hw_profiles.keys() if k.split()[0] in hw_choice][0]
        specs = hw_profiles[chosen_key]

        dep_rows = []
        for m, inf in specs.items():
            feasible = inf["ms"] <= budget_ms
            status_text = "✅ FEASIBLE" if feasible else f"❌ BOTTLENECK (Need {inf['ms']:.1f} ms)"
            dep_rows.append({
                "Model": m,
                "Latency (Est.)": f"{inf['ms']:.1f} ms",
                "Max FPS": f"{inf['fps']:.1f} FPS",
                "Status": status_text,
                "Memory Footprint": inf["ram"],
                "Recommended Acceleration": inf["opt"]
            })
        st.table(pd.DataFrame(dep_rows))

    # =========================================================================
    # TAB 7: ARCHITECTURE & METHODOLOGY DEEP DIVE
    # =========================================================================
    with tab_architecture:
        st.subheader("🧠 Architectural Deep Dive: Two-Stage vs One-Stage Detectors")
        arch_col1, arch_col2 = st.columns(2)
        with arch_col1:
            st.markdown("""
            ### 🔵 Two-Stage Detectors (e.g. Faster R-CNN)
            **Core Concept:** Divide detection into proposal generation followed by refinement.

            1. **Backbone & FPN:** Feature extraction using ResNet-50 / MobileNetV3 + Feature Pyramid Network.
            2. **Stage 1 (Region Proposal Network - RPN):** Lightweight convolutional network slides across pyramid features to propose ~1000 candidate regions of interest (RoIs) with binary objectness scores.
            3. **RoIAlign:** Preserves precise spatial alignment without coordinate quantization by bilinearly interpolating feature maps.
            4. **Stage 2 (Box Head):** Two-headed fully connected network refines bounding coordinates and predicts multi-class probabilities for each RoI.
            """)

        with arch_col2:
            st.markdown("""
            ### 🟠 One-Stage Detectors (e.g. YOLOv8)
            **Core Concept:** Frame object detection as a single unified end-to-end dense regression task.

            1. **Backbone & Neck:** Modified CSPDarknet backbone + Path Aggregation Network (PANet / C2f modules).
            2. **Anchor-Free Decoupled Head:** Separates classification and bounding box regression into parallel branches, directly predicting distance from grid cell centers (Distribution Focal Loss).
            3. **Task-Aligned Assigner:** Dynamically selects optimal positive anchors during training based on joint classification confidence and IoU overlap.
            4. **Unified Forward Pass:** Evaluates all spatial positions simultaneously in a single pass without intermediate proposal extraction.
            """)

        st.markdown("---")
        comp_data = {
            "Dimension": [
                "Pipeline Architecture",
                "Proposal Generation",
                "Spatial Feature Alignment",
                "Head Design",
                "Typical Inference Speed (CPU)",
                "Small Object Detection",
                "Edge / Mobile Suitability",
                "Model Weight Footprint"
            ],
            "Faster R-CNN (Two-Stage)": [
                "Two-phase: Proposals -> Refinement",
                "Explicit RPN with Anchor Boxes",
                "RoIAlign (Bilinear Interpolation)",
                "Fully connected MLP Heads",
                "0.2 - 1.4 FPS (Slow)",
                "⭐⭐⭐⭐⭐ Superior (40.1% mAP_s)",
                "Poor (Heavy compute requirement)",
                "77 - 175 MB"
            ],
            "YOLOv8 (One-Stage)": [
                "Single unified dense regression pass",
                "None (Anchor-free direct prediction)",
                "Strided feature pyramid grids",
                "Decoupled classification & regression heads",
                "7.5 - 22.0 FPS (Fast)",
                "⭐⭐⭐ Good (29.0% - 36.5% mAP_s)",
                "⭐⭐⭐⭐⭐ Excellent (Edge ready)",
                "6.5 - 22.5 MB"
            ]
        }
        st.table(pd.DataFrame(comp_data))

    # =========================================================================
    # TAB 8: COMPUTER VISION INTELLIGENCE CHAT ASSISTANT
    # =========================================================================
    with tab_chat:
        st.subheader("💬 Computer Vision Intelligence Assistant")
        st.write("Ask questions about detector architectures, trade-offs, benchmark results, or real-world deployment recommendations.")

        st.markdown("**Quick Inquiries:**")
        q1, q2, q3, q4 = st.columns(4)
        prompt_to_submit = None

        if q1.button("⚡ Why is YOLOv8 so fast?"):
            prompt_to_submit = "Why is YOLOv8 so much faster than Faster R-CNN?"
        if q2.button("🔍 Small object comparison?"):
            prompt_to_submit = "How do they compare on small objects?"
        if q3.button("📱 Best model for drone/edge?"):
            prompt_to_submit = "Which model is best for mobile or drone deployment?"
        if q4.button("🎓 Summary for Professor"):
            prompt_to_submit = "Generate an executive summary for my Computer Vision professor."

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I am your Computer Vision Benchmark Assistant. I can help analyze the trade-offs between Faster R-CNN and YOLOv8, explain their architectures, or break down the results on COCO val2017. What would you like to know?"
                }
            ]

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask a question about Two-Stage vs One-Stage detectors...")
        if prompt_to_submit:
            user_input = prompt_to_submit

        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            resp = generate_assistant_response(user_input, bench_data, st.session_state.get("last_detection_stats"))
            st.session_state.chat_messages.append({"role": "assistant", "content": resp})
            with st.chat_message("assistant"):
                st.markdown(resp)


def generate_assistant_response(query, bench_data, live_stats=None):
    """Knowledgeable CV assistant answering based on project context and benchmark numbers."""
    q = query.lower()

    if "professor" in q or "executive summary" in q or "summary for" in q:
        return (
            "### 🎓 Executive Summary: Assignment 2 (Project 26)\n\n"
            "**Core Objective:** Evaluate the speed vs accuracy trade-off between modern Two-Stage (Faster R-CNN) and One-Stage (YOLOv8) detectors on COCO val2017 under identical hardware.\n\n"
            "**Key Quantitative Findings:**\n"
            "1. **Throughput Advantage:** YOLOv8n achieves **18.7+ FPS** (~53 ms) on CPU compared to **0.23 FPS** (~4,411 ms) for Faster R-CNN R50 — an **~82.5x speedup** with **93% fewer parameters**.\n"
            "2. **Small Object Precision:** Faster R-CNN ResNet-50 leads on small objects (**40.1% vs 29.0% mAP_s**), proving the value of RoIAlign and higher native resolution (800px vs 640px).\n"
            "3. **Pareto Optimal Model:** **YOLOv8s** delivers **53.8% mAP** (within 2.2% of Faster R-CNN) while maintaining **7.52 FPS** on CPU, making it the most balanced general-purpose detector.\n\n"
            "**Architectural Verdict:** The classic rule 'two-stage is accurate, one-stage is fast' is partially evolving. Modern anchor-free one-stage detectors have closed the overall mAP gap to under 3%, but two-stage architectures still retain an edge for dense sub-pixel localization."
        )

    if "small" in q or "tiny" in q or "sahi" in q:
        return (
            "### 🔍 Small Object Performance Comparison & SAHI Solution\n\n"
            "According to the COCO val2017 benchmark from this project:\n"
            "- **Faster R-CNN ResNet-50-FPN v2:** Achieves **40.1% mAP_small**.\n"
            "- **YOLOv8s:** Achieves **36.6% mAP_small**.\n"
            "- **YOLOv8n:** Achieves **29.0% mAP_small**.\n\n"
            "**Why Two-Stage dominates small objects:**\n"
            "1. **RoIAlign:** Faster R-CNN uses RoIAlign to sample sub-pixel features with bilinear interpolation without coordinate quantization.\n"
            "2. **Feature Resolution:** Faster R-CNN processes input images resized to 800px on the short side (vs 640px letterbox for YOLO), providing higher spatial pixel density for small items.\n"
            "3. **SAHI Slicing:** In industry, **SAHI (Slicing Aided Hyper Inference)** slices the image into 2x2 overlapping crops, running YOLO on each at full resolution to close the small-object gap!"
        )

    if "fast" in q or "speed" in q or "latency" in q or "fps" in q or "why is yolo" in q:
        return (
            "### ⚡ Why YOLOv8 is Dramatically Faster\n\n"
            "In our CPU benchmark, **YOLOv8n runs at ~18.7–22.0 FPS (~45–53 ms)** compared to **Faster R-CNN R50 at ~0.23 FPS (~4,411 ms)** — an incredible **~80x to 90x speedup**!\n\n"
            "The architectural reasons include:\n"
            "1. **Single-Pass Dense Regression:** YOLOv8 predicts class probabilities and bounding box offsets across all grid cells simultaneously in one unified network pass.\n"
            "2. **No RPN & No RoI Crop Bottleneck:** Faster R-CNN must generate ~1000 region proposals, filter them with NMS, crop features using RoIAlign for each candidate, and pass each crop through a second MLP head.\n"
            "3. **Lightweight Parameters:** YOLOv8n has only **3.16M parameters (6.5 MB)**, whereas Faster R-CNN R50 has **43.7M parameters (175 MB)** — a ~14x reduction in model weight!"
        )

    if "mobile" in q or "drone" in q or "edge" in q or "deploy" in q or "embedded" in q or "raspberry" in q:
        return (
            "### 📱 Deployment Recommendation for Edge & Mobile Devices\n\n"
            "**Recommended Choice: YOLOv8n or YOLOv8s**\n\n"
            "**Key Reasons:**\n"
            "1. **Throughput:** On constrained CPU/NPU hardware, YOLOv8n achieves real-time frame rates (18–30+ FPS) suitable for video streams and drones.\n"
            "2. **Memory Footprint:** At only **6.5 MB**, YOLOv8n easily fits in L3 cache or embedded flash memory, whereas Faster R-CNN requires 175 MB.\n"
            "3. **Export Compatibility:** Ultralytics provides 1-click export to ONNX, TensorRT, CoreML, TFLite, and OpenVINO with INT8 quantization."
        )

    return (
        f"### 💡 Computer Vision Benchmark Insights\n\n"
        f"You asked: *\"{query}\"*\n\n"
        "Here are key findings from our benchmark project comparing Two-Stage vs One-Stage detectors:\n"
        "- **Speed Winner:** YOLOv8n achieves **18.7+ FPS** on CPU (~53 ms), making it suitable for live video.\n"
        "- **Accuracy Winner:** Faster R-CNN R50-FPN v2 achieves **56.0% mAP** on COCO val2017, leading significantly on small objects (40.1% vs 29.0%).\n"
        "- **Best Balance:** YOLOv8s delivers **53.8% mAP** (nearly matching Faster R-CNN) while running at **7.5 FPS** (~133 ms), making it an outstanding general-purpose detector.\n\n"
        "Feel free to explore the new **Ground Truth Error Matrix**, **SAHI Slicing**, and **Video File Processor** tabs!"
    )


# ----------------------------------------------------------------------------- CLI fallback
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--left", default=TWO_STAGE[0], choices=TWO_STAGE)
    ap.add_argument("--right", default=ONE_STAGE[0], choices=ONE_STAGE)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="demo_out.jpg")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    left, right = load_warm(a.left), load_warm(a.right)
    outs, _, _, stats = run_pair_detailed(left, right, img, a.thr)
    for info, s in zip((left, right), stats):
        print(f"{info['name']:28s} {s['ms']:8.1f} ms  {s['fps']:6.2f} FPS  {s['boxes']} boxes")
    gap = np.full((img.shape[0], 8, 3), 255, np.uint8)
    cv2.imwrite(a.out, np.hstack([outs[0], gap, outs[1]]))
    print("saved", a.out)


if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as in_streamlit
    except ImportError:
        in_streamlit = lambda: False
    if in_streamlit():
        streamlit_app()
    else:
        cli()
