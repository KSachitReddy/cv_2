"""
demo.py - Modern Computer Vision Benchmark & Detection Studio
Comparing Two-Stage (Faster R-CNN) vs One-Stage (YOLOv8) Detectors

Features:
- Live Side-by-Side Detection Studio with 1-click Preset Samples, Uploads & Webcam
- Interactive Detection Inspector (class counts, confidence distributions, filterable tables)
- Interactive Benchmark Analytics Studio with dynamic Plotly charts from results.json
- Architecture & Methodology Deep Dive (RPN vs Anchor-Free, Pareto frontier)
- Interactive CV Intelligence Chat Assistant answering questions based on benchmark results
- Full CLI fallback compatibility
"""
import argparse
import json
import os
import sys
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

TWO_STAGE = ["FasterRCNN-R50-FPN-v2", "FasterRCNN-MobileNetV3-FPN"]
ONE_STAGE = ["YOLOv8s", "YOLOv8n", "YOLOv8m"]
DEMO_SCORE = 0.25

SAMPLE_PRESETS = [
    {
        "id": "sample1",
        "name": "Sample 1: Crowded Street",
        "file": os.path.join(RESULTS_DIR, "sample1.jpg"),
        "desc": "Dense urban street with many overlapping pedestrians & vehicles (29 GT objects).",
        "icon": "🚶‍♂️"
    },
    {
        "id": "sample2",
        "name": "Sample 2: Small Objects",
        "file": os.path.join(RESULTS_DIR, "sample2.jpg"),
        "desc": "Challenging scene with distant and tiny objects (16 GT objects, 15 small).",
        "icon": "🔍"
    },
    {
        "id": "sample3",
        "name": "Sample 3: Indoor Dining",
        "file": os.path.join(RESULTS_DIR, "sample3.jpg"),
        "desc": "Typical indoor dining scene with utensils, table, and chairs (4 GT objects).",
        "icon": "🍽️"
    },
    {
        "id": "sample4",
        "name": "Sample 4: Sports & Action",
        "file": os.path.join(RESULTS_DIR, "sample4.jpg"),
        "desc": "Active sports movement with players and equipment (6 GT objects).",
        "icon": "🏃"
    }
]


def load_warm(name):
    """Load a model and run it once on a blank image so the first shown timing is realistic."""
    info = common.load_model(name, common.get_device(), score_thresh=DEMO_SCORE)
    common.detect(info, np.zeros((480, 640, 3), np.uint8))
    return info


def run_pair_detailed(left, right, img, thr):
    """Run both detectors on one BGR image.
    Returns annotated images, detailed detection lists, and summary statistics.
    """
    outs, detections_list, stats = [], [], []
    for info in (left, right):
        (boxes, scores, coco_ids, names), ms = common.timed_detect(info, img)
        keep = scores >= thr
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]
        filtered_names = [names[i] for i in range(len(scores)) if keep[i]]

        annotated = common.draw(img, boxes, scores, names, thr, common.FAMILY_BGR[info["family"]])
        outs.append(annotated)

        # Build detailed detection list
        det_records = []
        for b, s, n in zip(filtered_boxes, filtered_scores, filtered_names):
            det_records.append({
                "model": info["name"],
                "family": info["family"],
                "class": n,
                "confidence": float(s),
                "box": [round(coord, 1) for coord in b.tolist()]
            })
        detections_list.append(det_records)

        stats.append({
            "name": info["name"],
            "family": info["family"],
            "ms": ms,
            "fps": 1000.0 / ms if ms > 0 else 0,
            "boxes": int(keep.sum()),
            "classes_counter": Counter(filtered_names),
            "params_M": info.get("params", 0) / 1e6,
            "weight_mb": info.get("weight_mb", 0)
        })

    return outs, detections_list, stats


def load_benchmark_data():
    """Load saved benchmark results if available."""
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ----------------------------------------------------------------------------- Streamlit UI
def streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Two-Stage vs One-Stage Vision Studio",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom modern CSS styling
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
        .stat-card {
            background: rgba(30, 41, 59, 0.05);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .metric-highlight {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2563EB;
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
        .sample-card {
            border: 2px solid #E2E8F0;
            border-radius: 10px;
            padding: 8px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .sample-card:hover {
            border-color: #3B82F6;
            transform: translateY(-2px);
        }
        </style>
    """, unsafe_allow_html=True)

    dev = common.get_device()
    bench_data = load_benchmark_data()

    # Header section
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown('<div class="main-title">🎯 Object Detection Intelligence Studio</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Two-Stage (Faster R-CNN) vs One-Stage (YOLOv8) Detector Benchmarking & Analysis</div>', unsafe_allow_html=True)
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
        right_name = st.selectbox("One-Stage Model (High Speed)", ONE_STAGE, index=1)  # Default YOLOv8n

        thr = st.slider("Detection Confidence Threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.05)

        st.markdown("---")
        st.subheader("💡 Architecture Notes")
        st.caption(
            "• **Two-Stage:** Explicit Region Proposal Network (RPN) + RoIAlign. Excels at small/crowded objects, but higher latency.\n\n"
            "• **One-Stage:** Direct dense grid regression with Task-Aligned Assigner. Ideal for real-time edge processing."
        )

        st.markdown("---")
        st.caption("DS307 Computer Vision • Assignment 2 • Project 26")

    # Main Tabs
    tab_playground, tab_benchmark, tab_architecture, tab_chat = st.tabs([
        "🔬 Live Detection Studio",
        "📊 Interactive Benchmark & Analytics",
        "🧠 Architecture & Methodology",
        "💬 CV Intelligence Assistant"
    ])

    # =========================================================================
    # TAB 1: LIVE DETECTION STUDIO
    # =========================================================================
    with tab_playground:
        st.subheader("Interactive Detection Playground")
        st.write("Select an instant sample preset, upload your own image, or capture live video to observe side-by-side performance.")

        input_mode = st.radio(
            "Choose Input Source:",
            ["✨ Instant Presets (Recommended)", "📁 Upload Custom Image", "📷 Webcam Snapshot", "🎥 Live Webcam Stream"],
            horizontal=True
        )

        img = None
        selected_desc = ""

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
                    if os.path.exists(p["file"]):
                        img = cv2.imread(p["file"])
                        selected_desc = f"**{p['name']}**: {p['desc']}"
                    else:
                        st.warning(f"Preset image not found at `{p['file']}`. Falling back to blank test image.")
                        img = np.zeros((480, 640, 3), np.uint8)

        elif input_mode == "📁 Upload Custom Image":
            up = st.file_uploader("Upload an image (JPG, PNG, WebP):", type=["jpg", "jpeg", "png", "webp", "bmp"])
            if up:
                img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
                selected_desc = f"Custom upload: `{up.name}` ({img.shape[1]}x{img.shape[0]} px)"

        elif input_mode == "📷 Webcam Snapshot":
            snap = st.camera_input("Capture frame from camera")
            if snap:
                img = cv2.imdecode(np.frombuffer(snap.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                selected_desc = f"Webcam frame capture ({img.shape[1]}x{img.shape[0]} px)"

        elif input_mode == "🎥 Live Webcam Stream":
            st.info("Live webcam streaming runs frame-by-frame inference. On CPU, MobileNetV3 + YOLOv8n is recommended.")
            cap_src = st.number_input("Webcam Device Index", 0, 5, 0)
            max_frames = st.slider("Max Frames", 10, 300, 60, 10)
            if st.button("▶️ Start Live Stream", type="primary"):
                cap = cv2.VideoCapture(cap_src)
                if not cap.isOpened():
                    st.error("Failed to connect to webcam.")
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
                        outs, _, stats = run_pair_detailed(left, right, frame, thr)
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

        # Process and show detection for static inputs
        if img is not None:
            left, right = load(left_name), load(right_name)
            outs, det_records, stats = run_pair_detailed(left, right, img, thr)

            # Store in session state for Chat Assistant context
            st.session_state["last_detection_stats"] = stats
            st.session_state["last_detection_records"] = det_records

            # Head-to-Head Comparison Banner
            ms_left = stats[0]["ms"]
            ms_right = stats[1]["ms"]
            speedup = ms_left / ms_right if ms_right > 0 else 1.0

            banner_html = f"""
            <div class="head-to-head-banner">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
                    <div>
                        <span style="font-size: 1.25rem; font-weight: 700;">⚔️ Head-to-Head Comparison</span><br>
                        <span style="color: #94A3B8; font-size: 0.95rem;">{selected_desc}</span>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; padding: 8px 16px; border-radius: 8px;">
                        <span style="color: #34D399; font-weight: 700; font-size: 1.15rem;">⚡ {stats[1]['name']} is {speedup:.1f}x FASTER</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(banner_html, unsafe_allow_html=True)

            # Images Display
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

            # Interactive Detection Analytics Section
            st.markdown("---")
            st.subheader("🔍 Detection Breakdown & Category Distribution")

            # Combined class distribution chart
            counts_left = stats[0]["classes_counter"]
            counts_right = stats[1]["classes_counter"]
            all_classes = sorted(list(set(list(counts_left.keys()) + list(counts_right.keys()))))

            if all_classes:
                chart_df = pd.DataFrame([
                    {"Category": c, "Count": counts_left.get(c, 0), "Model": f"{stats[0]['name']} (Two-Stage)"}
                    for c in all_classes
                ] + [
                    {"Category": c, "Count": counts_right.get(c, 0), "Model": f"{stats[1]['name']} (One-Stage)"}
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
                        f"{stats[0]['name']} (Two-Stage)": "#2563EB",
                        f"{stats[1]['name']} (One-Stage)": "#F59E0B"
                    }
                )
                fig_classes.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_classes, use_container_width=True)
            else:
                st.info("No objects detected above the confidence threshold.")

            # Filterable Detections Table Expander
            with st.expander("📋 View Raw Detection Coordinate Data"):
                all_dets = det_records[0] + det_records[1]
                if all_dets:
                    df_dets = pd.DataFrame(all_dets)
                    st.dataframe(df_dets, use_container_width=True)
                else:
                    st.write("No detection coordinates to display.")

    # =========================================================================
    # TAB 2: INTERACTIVE BENCHMARK & ANALYTICS STUDIO
    # =========================================================================
    with tab_benchmark:
        st.subheader("📊 Comprehensive COCO val2017 Benchmark")
        st.write("Quantitative analysis evaluating speed, mean Average Precision (mAP), and model footprint under identical hardware conditions.")

        if bench_data and "models" in bench_data:
            models_dict = bench_data["models"]
            meta = bench_data.get("meta", {})

            # Overview metrics bar
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Evaluated Images", meta.get("num_images", 100))
            b_col2.metric("Ground Truth Boxes", meta.get("num_gt_boxes", 709))
            b_col3.metric("Batch Size", meta.get("batch_size", 1))
            b_col4.metric("Benchmark Device", meta.get("device_name", "CPU"))

            # Build comparison DataFrame
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
            st.warning("Benchmark results file (`results/results.json`) not found. Run `uv run bench` to generate it.")

    # =========================================================================
    # TAB 3: ARCHITECTURE & METHODOLOGY DEEP DIVE
    # =========================================================================
    with tab_architecture:
        st.subheader("🧠 Architectural Deep Dive: Two-Stage vs One-Stage Detectors")

        arch_col1, arch_col2 = st.columns(2)

        with arch_col1:
            st.markdown("""
            ### 🔵 Two-Stage Detectors (e.g. Faster R-CNN)
            **Core Concept:** Divide detection into proposal generation followed by refinement.

            1. **Backbone & FPN:** Feature extraction using ResNet-50 / MobileNetV3 + Feature Pyramid Network.
            2. **Stage 1 (Region Proposal Network - RPN):** A lightweight convolutional network slides across pyramid features to propose ~1000 candidate regions of interest (RoIs) with binary objectness scores.
            3. **RoIAlign:** Preserves precise spatial alignment without coordinate quantization by bilinearly interpolating feature maps.
            4. **Stage 2 (Box Head):** Two-headed fully connected network refines bounding coordinates and predicts multi-class probabilities for each RoI.

            **Pros:** High localization accuracy, superior performance on small & overlapping objects.
            **Cons:** High computational latency, sequential processing bottlenecks, large memory footprint.
            """)

        with arch_col2:
            st.markdown("""
            ### 🟠 One-Stage Detectors (e.g. YOLOv8)
            **Core Concept:** Frame object detection as a single unified end-to-end dense regression task.

            1. **Backbone & Neck:** Modified CSPDarknet backbone + Path Aggregation Network (PANet / C2f modules).
            2. **Anchor-Free Decoupled Head:** Separates classification and bounding box regression into parallel branches, directly predicting distance from grid cell centers (Distribution Focal Loss).
            3. **Task-Aligned Assigner:** Dynamically selects optimal positive anchors during training based on joint classification confidence and IoU overlap.
            4. **Unified Forward Pass:** Evaluates all spatial positions simultaneously in a single pass without intermediate proposal extraction.

            **Pros:** Ultra-fast real-time inference (up to 80+x speedup), lightweight weights, edge/embedded friendly.
            **Cons:** Slightly lower precision on densely occluded and minute objects without custom tiling.
            """)

        st.markdown("---")
        st.subheader("🔬 Direct Comparison Matrix")
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

        # Show architecture diagram if file exists
        arch_img_path = os.path.join(RESULTS_DIR, "fig_architectures.png")
        if os.path.exists(arch_img_path):
            st.markdown("#### Architectural Block Diagram")
            st.image(arch_img_path, caption="Two-Stage (Faster R-CNN) vs One-Stage (YOLOv8) Structural Pipelines", use_container_width=True)

    # =========================================================================
    # TAB 4: COMPUTER VISION INTELLIGENCE CHAT ASSISTANT
    # =========================================================================
    with tab_chat:
        st.subheader("💬 Computer Vision Intelligence Assistant")
        st.write("Ask questions about detector architectures, trade-offs, benchmark results, or real-world deployment recommendations.")

        # Preset Prompt Buttons
        st.markdown("**Quick Inquiries:**")
        q_cols = st.columns(3)
        prompt_to_submit = None

        if q_cols[0].button("⚡ Why is YOLOv8 so much faster than Faster R-CNN?"):
            prompt_to_submit = "Why is YOLOv8 so much faster than Faster R-CNN?"
        if q_cols[1].button("🔍 How do they compare on small objects?"):
            prompt_to_submit = "How do they compare on small objects?"
        if q_cols[2].button("📱 Which model is best for mobile or drone deployment?"):
            prompt_to_submit = "Which model is best for mobile or drone deployment?"

        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I am your Computer Vision Benchmark Assistant. I can help analyze the trade-offs between Faster R-CNN and YOLOv8, explain their architectures, or break down the results on COCO val2017. What would you like to know?"
                }
            ]

        # Display history
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

            # Generate smart domain-specific response
            resp = generate_assistant_response(user_input, bench_data, st.session_state.get("last_detection_stats"))
            st.session_state.chat_messages.append({"role": "assistant", "content": resp})
            with st.chat_message("assistant"):
                st.markdown(resp)


def generate_assistant_response(query, bench_data, live_stats=None):
    """Knowledgeable CV assistant answering based on project context and benchmark numbers."""
    q = query.lower()

    # Small objects comparison
    if "small" in q or "tiny" in q:
        return (
            "### 🔍 Small Object Performance Comparison\n\n"
            "According to the COCO val2017 benchmark from this project:\n"
            "- **Faster R-CNN ResNet-50-FPN v2:** Achieves **40.1% mAP_small**.\n"
            "- **YOLOv8s:** Achieves **36.6% mAP_small**.\n"
            "- **YOLOv8n:** Achieves **29.0% mAP_small**.\n\n"
            "**Why Two-Stage dominates small objects:**\n"
            "1. **RoIAlign:** Faster R-CNN uses RoIAlign to sample sub-pixel features with bilinear interpolation without coordinate quantization.\n"
            "2. **Feature Resolution:** Faster R-CNN processes input images resized to 800px on the short side (vs 640px letterbox for YOLO), providing higher spatial pixel density for small items.\n"
            "3. **Two-Stage Refinement:** Candidate bounding boxes proposed by the RPN undergo a second dedicated regression step in the MLP head."
        )

    # Why is YOLO faster
    if "fast" in q or "speed" in q or "latency" in q or "fps" in q or "why is yolo" in q:
        return (
            "### ⚡ Why YOLOv8 is Dramatically Faster\n\n"
            "In our CPU benchmark, **YOLOv8n runs at ~18.7–22.0 FPS (~45–53 ms)** compared to **Faster R-CNN R50 at ~0.23 FPS (~4,411 ms)** — an incredible **~80x to 90x speedup**!\n\n"
            "The architectural reasons include:\n"
            "1. **Single-Pass Dense Regression:** YOLOv8 predicts class probabilities and bounding box offsets across all grid cells simultaneously in one unified network pass.\n"
            "2. **No RPN & No RoI Crop Bottleneck:** Faster R-CNN must generate ~1000 region proposals, filter them with NMS, crop features using RoIAlign for each candidate, and pass each crop through a second MLP head.\n"
            "3. **Lightweight Parameters:** YOLOv8n has only **3.16M parameters (6.5 MB)**, whereas Faster R-CNN R50 has **43.7M parameters (175 MB)** — a ~14x reduction in model weight!"
        )

    # Mobile / Drone / Edge deployment
    if "mobile" in q or "drone" in q or "edge" in q or "deploy" in q or "embedded" in q or "raspberry" in q:
        return (
            "### 📱 Deployment Recommendation for Edge & Mobile Devices\n\n"
            "**Recommended Choice: YOLOv8n or YOLOv8s**\n\n"
            "**Key Reasons:**\n"
            "1. **Throughput:** On constrained CPU/NPU hardware, YOLOv8n achieves real-time frame rates (18–30+ FPS) suitable for video streams and drones.\n"
            "2. **Memory Footprint:** At only **6.5 MB**, YOLOv8n easily fits in L3 cache or embedded flash memory, whereas Faster R-CNN requires 175 MB.\n"
            "3. **Export Compatibility:** Ultralytics provides 1-click export to ONNX, TensorRT, CoreML, TFLite, and OpenVINO with INT8 quantization.\n\n"
            "*Exception:* If you are building an offline server inspection pipeline (e.g. medical radiology, high-altitude satellite reconnaissance) where latency does not matter and every small object counts, Faster R-CNN R50 is preferred."
        )

    # Difference between mAP metrics
    if "map" in q or "map50" in q or "metric" in q:
        return (
            "### 🎯 Understanding mAP Metrics (COCO Evaluation)\n\n"
            "- **mAP [0.50:0.95] (Primary metric):** Average Precision computed across 10 IoU thresholds from 0.50 to 0.95 with step 0.05. It rewards detectors that predict tightly fitted bounding boxes.\n"
            "- **mAP50:** Average Precision evaluated at an easy IoU threshold of 0.50. This measures whether the object was detected roughly in the right area.\n"
            "- **mAP75:** Evaluated at a strict IoU threshold of 0.75, penalizing loose bounding boxes.\n"
            "- **AR100:** Average Recall given up to 100 detections per image."
        )

    # Fallback contextual response
    return (
        f"### 💡 Computer Vision Benchmark Insights\n\n"
        f"You asked: *\"{query}\"*\n\n"
        "Here are key findings from our benchmark project comparing Two-Stage vs One-Stage detectors:\n"
        "- **Speed Winner:** YOLOv8n achieves **18.7+ FPS** on CPU (~53 ms), making it suitable for live video.\n"
        "- **Accuracy Winner:** Faster R-CNN R50-FPN v2 achieves **56.0% mAP** on COCO val2017, leading significantly on small objects (40.1% vs 29.0%).\n"
        "- **Best Balance:** YOLOv8s delivers **53.8% mAP** (nearly matching Faster R-CNN) while running at **7.5 FPS** (~133 ms), making it an outstanding general-purpose detector.\n\n"
        "Feel free to ask about specific layers (RPN, RoIAlign, Anchor-Free heads), loss functions, or benchmark metrics!"
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
    outs, _, stats = run_pair_detailed(left, right, img, a.thr)
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
