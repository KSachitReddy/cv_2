"""
demo.py - live side-by-side demo: Faster R-CNN (two-stage) vs YOLOv8 (one-stage)

  Web app (recommended for the presentation):
      uv run demo
  Fallback without a browser (writes a side-by-side image):
      uv run demo --image data/val2017/000000000139.jpg --out demo_out.jpg

Sources in the web app: uploaded image, webcam snapshot, or a live webcam stream.
"""
import argparse
import os
import sys

import cv2
import numpy as np

import common

TWO_STAGE = ["FasterRCNN-R50-FPN-v2", "FasterRCNN-MobileNetV3-FPN"]
ONE_STAGE = ["YOLOv8s", "YOLOv8n", "YOLOv8m"]
DEMO_SCORE = 0.25          # models keep boxes >= 0.25; the slider filters further for display


def load_warm(name):
    """Load a model and run it once on a blank image so the first shown timing is realistic."""
    info = common.load_model(name, common.get_device(), score_thresh=DEMO_SCORE)
    common.detect(info, np.zeros((480, 640, 3), np.uint8))
    return info


def run_pair(left, right, img, thr):
    """Run both detectors on one BGR image. Returns two annotated images and their stats."""
    outs, stats = [], []
    for info in (left, right):
        (b, s, _, n), ms = common.timed_detect(info, img)
        keep = s >= thr
        outs.append(common.draw(img, b, s, n, thr, common.FAMILY_BGR[info["family"]]))
        stats.append(dict(ms=ms, fps=1000.0 / ms, boxes=int(keep.sum())))
    return outs, stats


# ----------------------------------------------------------------------------- Streamlit UI
def streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="R-CNN vs YOLO", layout="wide")
    st.title("Two-stage vs one-stage detection: Faster R-CNN vs YOLOv8")
    dev = common.get_device()
    st.caption(f"Running on **{dev.type.upper()}**. Pretrained COCO weights, batch size 1. "
               "Time = preprocessing + network + NMS for one image.")

    load = st.cache_resource(show_spinner="Loading model...")(load_warm)

    with st.sidebar:
        st.header("Settings")
        left_name = st.selectbox("Two-stage model", TWO_STAGE)
        right_name = st.selectbox("One-stage model", ONE_STAGE)
        thr = st.slider("Score threshold", DEMO_SCORE, 0.95, 0.5, 0.05)
        source = st.radio("Input", ["Upload image", "Webcam snapshot", "Live webcam"])
        if dev.type == "cpu":
            st.info("CPU: for the live webcam choose the MobileNetV3 Faster R-CNN to keep it responsive.")

    left, right = load(left_name), load(right_name)

    def show(outs, stats):
        cols = st.columns(2)
        for col, info, img, s in zip(cols, (left, right), outs, stats):
            with col:
                st.subheader(f"{info['name']} ({info['family']})")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")
                a, b, c = st.columns(3)
                a.metric("Inference", f"{s['ms']:.0f} ms")
                b.metric("FPS", f"{s['fps']:.1f}")
                c.metric("Boxes", s["boxes"])

    img = None
    if source == "Upload image":
        up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp"])
        if up:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    elif source == "Webcam snapshot":
        snap = st.camera_input("Take a picture")
        if snap:
            img = cv2.imdecode(np.frombuffer(snap.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    if img is not None:
        show(*run_pair(left, right, img, thr))
        return

    if source == "Live webcam":
        cap_src = st.number_input("Camera index", 0, 5, 0)
        max_frames = st.slider("Max frames to process", 10, 500, 100, 10)
        if not st.button("Start"):
            return
        cap = cv2.VideoCapture(cap_src)
        if not cap.isOpened():
            st.error("Could not open the webcam.")
            return
        st.caption("Press 'Stop' in the top-right corner of Streamlit to end early.")
        cols = st.columns(2)
        holders = [c.empty() for c in cols]
        texts = [c.empty() for c in cols]
        totals = [[], []]
        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            outs, stats = run_pair(left, right, frame, thr)
            for k in range(2):
                totals[k].append(stats[k]["ms"])
                holders[k].image(cv2.cvtColor(outs[k], cv2.COLOR_BGR2RGB), width="stretch")
                avg = np.mean(totals[k])
                texts[k].markdown(f"**{(left, right)[k]['name']}** - this frame {stats[k]['ms']:.0f} ms, "
                                  f"average {avg:.0f} ms (**{1000 / avg:.1f} FPS**), {stats[k]['boxes']} boxes")
        cap.release()
        st.success(f"Processed {len(totals[0])} frames.")


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
    outs, stats = run_pair(left, right, img, a.thr)
    for info, s in zip((left, right), stats):
        print(f"{info['name']:28s} {s['ms']:8.1f} ms  {s['fps']:6.2f} FPS  {s['boxes']} boxes")
    gap = np.full((img.shape[0], 8, 3), 255, np.uint8)
    cv2.imwrite(a.out, np.hstack([outs[0], gap, outs[1]]))
    print("saved", a.out)


if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as in_streamlit
    except ImportError:
        in_streamlit = lambda: False  # noqa: E731
    if in_streamlit():          # (a plain if-statement: Streamlit would print the value of a bare expression)
        streamlit_app()
    else:
        cli()
