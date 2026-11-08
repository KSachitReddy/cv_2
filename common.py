"""
common.py - code shared by benchmark.py and demo.py

It hides the differences between the two families of detectors behind ONE function:

    detect(model_info, image_bgr) -> boxes [N,4] (x1,y1,x2,y2), scores [N], coco_ids [N], names [N]

so that the benchmark and the demo treat Faster R-CNN (torchvision) and YOLOv8 (Ultralytics)
in exactly the same way.
"""
import os
import time

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models import detection as tvdet

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(ROOT, "weights")          # YOLO .pt files are stored here
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "weights", "torch"))  # torchvision weights go here too

# The 80 COCO category ids in sorted order. YOLO predicts class index 0..79;
# index i corresponds to COCO_IDS[i]. (benchmark.py re-checks this against the annotation file.)
COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
            78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

# Every model we compare. "family" is what the project is about: two-stage vs one-stage.
MODELS = {
    "FasterRCNN-R50-FPN-v2": dict(family="two-stage", kind="torchvision",
                                  builder=tvdet.fasterrcnn_resnet50_fpn_v2,
                                  weights=tvdet.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
    "FasterRCNN-MobileNetV3-FPN": dict(family="two-stage", kind="torchvision",
                                       builder=tvdet.fasterrcnn_mobilenet_v3_large_fpn,
                                       weights=tvdet.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1),
    "YOLOv8n": dict(family="one-stage", kind="yolo", file="yolov8n.pt"),
    "YOLOv8s": dict(family="one-stage", kind="yolo", file="yolov8s.pt"),
    "YOLOv8m": dict(family="one-stage", kind="yolo", file="yolov8m.pt"),
}


def get_device():
    """Use the GPU if PyTorch can see one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device):
    """GPU work is asynchronous: wait for it to finish so that timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_model(name, device, score_thresh=0.001, max_det=100):
    """Load a pretrained COCO model and return a small dict describing it."""
    cfg = MODELS[name]
    info = dict(name=name, family=cfg["family"], kind=cfg["kind"], device=device,
                score_thresh=score_thresh, max_det=max_det)
    if cfg["kind"] == "torchvision":
        # box_score_thresh: keep low-confidence boxes too (needed for a fair mAP computation)
        model = cfg["builder"](weights=cfg["weights"], box_score_thresh=score_thresh,
                               box_detections_per_img=max_det)
        model.eval().to(device)
        info["model"] = model
        info["names"] = cfg["weights"].meta["categories"]       # index == COCO category id
        info["params"] = sum(p.numel() for p in model.parameters())
        fname = os.path.basename(cfg["weights"].url)
        info["weight_file"] = os.path.join(torch.hub.get_dir(), "checkpoints", fname)
    else:
        from ultralytics import YOLO
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        path = os.path.join(WEIGHTS_DIR, cfg["file"])
        model = YOLO(path)                                        # downloads on first use
        model.to(device)
        info["model"] = model
        info["names"] = model.names                               # {0: 'person', 1: 'bicycle', ...}
        info["params"] = sum(p.numel() for p in model.model.parameters())
        info["weight_file"] = path
    info["weight_mb"] = os.path.getsize(info["weight_file"]) / 1e6
    return info


@torch.no_grad()
def detect(info, image_bgr):
    """Run one image (OpenCV BGR uint8) through the model.

    The timed region of the benchmark wraps this whole call, so latency = preprocessing +
    network + post-processing (NMS etc.) - i.e. the real end-to-end cost for one image.
    """
    if info["kind"] == "torchvision":
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(info["device"])
        out = info["model"]([tensor])[0]          # model resizes internally (short side 800)
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        coco_ids = out["labels"].cpu().numpy().astype(int)
        names = [info["names"][c] for c in coco_ids]
    else:
        res = info["model"].predict(image_bgr, conf=info["score_thresh"], iou=0.7, imgsz=640,
                                    max_det=info["max_det"], device=info["device"], verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        coco_ids = np.array([COCO_IDS[c] for c in cls], dtype=int)   # YOLO index -> COCO id
        names = [info["names"][c] for c in cls]
    return boxes, scores, coco_ids, names


def timed_detect(info, image_bgr):
    """detect() plus wall-clock latency in milliseconds (synchronised on GPU)."""
    sync(info["device"])
    t0 = time.perf_counter()
    result = detect(info, image_bgr)
    sync(info["device"])
    return result, (time.perf_counter() - t0) * 1000.0


def draw(image_bgr, boxes, scores, names, min_score=0.5, color=(214, 120, 42), title=None):
    """Draw boxes with 'label score' tags. Returns a new image."""
    img = image_bgr.copy()
    for (x1, y1, x2, y2), s, n in zip(boxes, scores, names):
        if s < min_score:
            continue
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        tag = f"{n} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(img, tag, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if title:
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (30, 30, 30), -1)
        cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# Box colours (BGR): blue for two-stage, orange for one-stage - same as in the charts.
FAMILY_BGR = {"two-stage": (214, 120, 42), "one-stage": (52, 104, 235)}
