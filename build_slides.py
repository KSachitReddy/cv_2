"""
build_slides.py - builds CV_Assignment2_Slides.pptx (10 slides) from results/results.json.

All numbers come from results.json. Each slide has speaker notes so that every team member
knows what to say. Usage:  uv run slides
"""
import json
import os

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "CV_Assignment2_Slides.pptx")

NAVY = RGBColor(0x14, 0x1E, 0x33)        # dark slides
INK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x5F, 0x63, 0x68)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE = RGBColor(0x2A, 0x78, 0xD6)        # two-stage (same as charts)
ORANGE = RGBColor(0xEB, 0x68, 0x34)      # one-stage
TINT = RGBColor(0xF2, 0xF5, 0xF9)        # card background
HEAD, BODY = "Cambria", "Calibri"
W, H = Inches(13.333), Inches(7.5)

def main():
    R = json.load(open(os.path.join(RES, "results.json")))
    M, META = R["models"], R["meta"]
    A, B = "FasterRCNN-R50-FPN-v2", "YOLOv8s"


    def f3(x):
        return f"{x:.3f}"


    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    BLANK = prs.slide_layouts[6]


    def text(slide, x, y, w, h, content, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, name=None):
        """content: a string, or a list of strings (one paragraph each). '• ' prefix -> bullet."""
        tb = slide.shapes.add_textbox(x, y, w, h)
        if name:
            tb.name = name
        tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.05)
        for k, line in enumerate([content] if isinstance(content, str) else content):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            bullet = line.startswith("• ")
            run = p.add_run(); run.text = ("•  " + line[2:]) if bullet else line
            run.font.size = Pt(size); run.font.color.rgb = color; run.font.bold = bold; run.font.name = font
            p.alignment = align; p.space_after = Pt(8 if bullet else 4)
        return tb


    def title(slide, t, dark=False):
        text(slide, Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.9), t, 34, WHITE if dark else INK,
             True, HEAD, name="Title")


    def card(slide, x, y, w, h, fill=TINT):
        s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        s.adjustments[0] = 0.06
        s.fill.solid(); s.fill.fore_color.rgb = fill; s.line.fill.background(); s.shadow.inherit = False
        return s


    def stat(slide, x, y, w, value, label, color):
        card(slide, x, y, w, Inches(1.55))
        text(slide, x + Inches(0.2), y + Inches(0.12), w - Inches(0.4), Inches(0.8), value, 36, color, True, HEAD)
        text(slide, x + Inches(0.2), y + Inches(0.9), w - Inches(0.4), Inches(0.6), label, 13, MUTED)


    def picture(slide, path, x, y, w=None, h=None):
        return slide.shapes.add_picture(path, x, y, w, h)


    def new(dark=False):
        s = prs.slides.add_slide(BLANK)
        if dark:
            bg = s.background.fill; bg.solid(); bg.fore_color.rgb = NAVY
        return s


    def footer(slide, n, dark=False):
        text(slide, Inches(11.9), Inches(7.0), Inches(1.0), Inches(0.35), str(n), 11,
             WHITE if dark else MUTED, align=PP_ALIGN.RIGHT)


    has_pair = A in M and B in M
    speedup = M[B]["fps"] / M[A]["fps"] if has_pair else None
    n_img, dev = META["num_images"], META["device_name"]

    # 1 ---------------------------------------------------------------- title
    s = new(dark=True)
    text(s, Inches(0.8), Inches(1.5), Inches(11.7), Inches(0.5), "DS307 Computer Vision  ·  Assignment 2  ·  Project 26",
         16, RGBColor(0xB8, 0xC4, 0xD6))
    text(s, Inches(0.8), Inches(2.1), Inches(11.7), Inches(1.9),
         "Two-Stage vs One-Stage Object Detection: Faster R-CNN vs YOLO for Speed and Accuracy", 40, WHITE, True, HEAD)
    text(s, Inches(0.8), Inches(4.5), Inches(11.7), Inches(1.4),
         ["{{NAME_1}} · {{NAME_2}} · {{NAME_3}} · {{NAME_4}}",
          "Faculty: {{FACULTY_NAME}}  ·  Faculty of Science & Technology, IFHE University, Hyderabad  ·  Nov 2026"],
         16, RGBColor(0xDD, 0xE3, 0xEC))
    s.notes_slide.notes_text_frame.text = (
        "Introduce the team and the question: is the classic rule 'two-stage = accurate, one-stage = fast' still true "
        "for current models? We test it with real measurements.")

    # 2 ---------------------------------------------------------------- problem
    s = new(); title(s, "How much accuracy does speed cost?")
    text(s, Inches(0.6), Inches(1.45), Inches(6.2), Inches(5.3), [
        "• Two-stage (R-CNN family): propose regions first, then classify and refine each region",
        "• One-stage (YOLO family): predict all boxes and classes in one pass over the image",
        "• Classic belief: two-stage is more accurate (especially on small objects), one-stage is faster",
        f"• Our test: {len(M)} pretrained COCO models, same {n_img} COCO val2017 images, batch size 1, same evaluation",
        "• Measured: latency, FPS, parameters, weight size, COCO mAP (overall and by object size)",
    ], 17)
    p1 = picture(s, os.path.join(RES, "sample1.jpg"), Inches(7.1), Inches(1.5), w=Inches(5.7))
    picture(s, os.path.join(RES, "sample2.jpg"), Inches(7.1), Inches(1.5) + p1.height + Inches(0.25), w=Inches(5.7))
    text(s, Inches(7.1), Inches(6.35), Inches(5.7), Inches(0.5),
         "COCO val2017: Faster R-CNN (left, blue) vs YOLOv8s (right, orange)", 12, MUTED)
    footer(s, 2)
    s.notes_slide.notes_text_frame.text = (
        "Explain the two families in one sentence each. Then state exactly what we measured and that the conditions "
        "were identical for every model, which makes the comparison fair.")

    # 3 ---------------------------------------------------------------- architectures
    s = new(); title(s, "Two architectures, one goal")
    picture(s, os.path.join(RES, "fig_architectures.png"), Inches(0.6), Inches(1.35), w=Inches(12.1))
    footer(s, 3)
    s.notes_slide.notes_text_frame.text = (
        "Top: Faster R-CNN. Backbone + FPN give multi-scale features. The RPN puts anchors at every position, scores "
        "objectness and regresses boxes, keeps about 1000 proposals after NMS. RoIAlign crops a 7x7 feature for each "
        "proposal and the box head classifies and refines it. Bottom: YOLOv8. CSPDarknet backbone, PAN-FPN neck, and an "
        "anchor-free decoupled head that predicts class and box at every grid cell in one pass. The per-proposal second "
        "stage is the main reason Faster R-CNN is slower.")

    # 4 ---------------------------------------------------------------- methodology
    s = new(); title(s, "Fair benchmarking methodology")
    picture(s, os.path.join(RES, "fig_flowchart.png"), Inches(0.6), Inches(1.3), h=Inches(5.9))
    items = [
        ("Same data", f"First {n_img} COCO val2017 images, {META['num_gt_boxes']} ground-truth boxes"),
        ("Fair timing", f"{META['warmup_images']} warm-up images; batch 1; CUDA sync on GPU; disk I/O excluded"),
        ("Fair accuracy", f"pycocotools COCOeval, score threshold {META['eval_score_threshold']}, max {META['max_detections']} boxes"),
        ("Hardware", dev),
    ]
    for k, (h, b) in enumerate(items):
        y = Inches(1.45 + k * 1.4)
        card(s, Inches(6.4), y, Inches(6.3), Inches(1.2))
        text(s, Inches(6.6), y + Inches(0.1), Inches(5.9), Inches(0.45), h, 18, BLUE, True, HEAD)
        text(s, Inches(6.6), y + Inches(0.55), Inches(5.9), Inches(0.6), b, 14, INK)
    footer(s, 4)
    s.notes_slide.notes_text_frame.text = (
        "Warm-up removes one-time costs such as memory allocation. On a GPU, kernels are asynchronous, so we "
        "synchronise before stopping the timer. Low score threshold 0.001 keeps the full precision-recall curve for "
        "mAP. YOLO class indices 0-79 are mapped to COCO category ids. " + " ".join(META["notes"]))

    # 5 ---------------------------------------------------------------- results table
    s = new(); title(s, "Results: speed and accuracy")
    cols = ["Model", "Family", "Params (M)", "Latency (ms)", "FPS", "mAP", "mAP50", "mAP small"]
    rows = [[m, x["family"], f"{x['params_M']:.1f}", f"{x['latency_ms_mean']:.0f} ± {x['latency_ms_std']:.0f}",
             f"{x['fps']:.2f}", f3(x["mAP"]), f3(x["mAP50"]), f3(x["mAP_small"])] for m, x in M.items()]
    tbl = s.shapes.add_table(len(rows) + 1, len(cols), Inches(0.6), Inches(1.5), Inches(12.1),
                             Inches(0.55 * (len(rows) + 1))).table
    widths = [3.4, 1.4, 1.3, 1.7, 1.0, 1.1, 1.1, 1.1]
    for c, w_ in enumerate(widths):
        tbl.columns[c].width = Inches(w_)
    best_map = max(M, key=lambda m: M[m]["mAP"]); best_fps = max(M, key=lambda m: M[m]["fps"])
    for r_ in range(len(rows) + 1):
        for c in range(len(cols)):
            cell = tbl.cell(r_, c)
            val = cols[c] if r_ == 0 else rows[r_ - 1][c]
            cell.text = ""
            run = cell.text_frame.paragraphs[0].add_run(); run.text = val
            run.font.size = Pt(15); run.font.name = BODY
            cell.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            cell.fill.solid()
            if r_ == 0:
                cell.fill.fore_color.rgb = NAVY; run.font.color.rgb = WHITE; run.font.bold = True
            else:
                m = list(M)[r_ - 1]
                cell.fill.fore_color.rgb = WHITE if r_ % 2 else TINT
                run.font.color.rgb = INK
                if c == 1:
                    run.font.color.rgb = BLUE if M[m]["family"] == "two-stage" else ORANGE; run.font.bold = True
                if (c == 5 and m == best_map) or (c == 4 and m == best_fps):
                    run.font.bold = True
    text(s, Inches(0.6), Inches(1.6 + 0.55 * (len(rows) + 1)), Inches(12.1), Inches(0.8),
         f"COCO val2017, first {n_img} images, {dev}. mAP = COCO mAP@[.5:.95]. Bold = best in column.", 12, MUTED)
    bal = max(M, key=lambda m: M[m]["mAP"] * (M[m]["fps"] >= 5))   # most accurate model that is at least 5 FPS
    cy = Inches(1.6 + 0.55 * (len(rows) + 1) + 0.75)
    for k, (val, lab, col) in enumerate([
            (f3(M[best_map]["mAP"]), f"highest mAP: {best_map}", BLUE if M[best_map]["family"] == "two-stage" else ORANGE),
            (f"{M[best_fps]['fps']:.1f} FPS", f"fastest: {best_fps}", BLUE if M[best_fps]["family"] == "two-stage" else ORANGE),
            (f3(M[bal]["mAP"]), f"best balance: {bal}, most accurate model at 5+ FPS ({M[bal]['fps']:.1f} FPS)",
             BLUE if M[bal]["family"] == "two-stage" else ORANGE)]):
        stat(s, Inches(0.6 + k * 4.1), cy, Inches(3.9), val, lab, col)
    footer(s, 5)
    s.notes_slide.notes_text_frame.text = (
        f"Read the table from top to bottom. Best mAP: {best_map} ({f3(M[best_map]['mAP'])}). Fastest: {best_fps} "
        f"({M[best_fps]['fps']:.1f} FPS). Latency is mean ± standard deviation over {n_img} images.")

    # 6 ---------------------------------------------------------------- trade-off
    s = new(); title(s, "The speed-accuracy trade-off")
    picture(s, os.path.join(RES, "chart_speed_vs_accuracy.png"), Inches(0.6), Inches(1.35), w=Inches(7.9))
    if has_pair:
        d = M[B]["mAP"] - M[A]["mAP"]
        stat(s, Inches(8.9), Inches(1.5), Inches(3.8), f"{speedup:.1f}x", "YOLOv8s throughput vs Faster R-CNN R50-FPN v2", ORANGE)
        stat(s, Inches(8.9), Inches(3.25), Inches(3.8), f"{d * 100:+.1f} pts",
             f"mAP difference, YOLOv8s minus Faster R-CNN ({f3(M[B]['mAP'])} vs {f3(M[A]['mAP'])})", ORANGE if d > 0 else BLUE)
        stat(s, Inches(8.9), Inches(5.0), Inches(3.8), f"{M[A]['params_M'] / M[B]['params_M']:.1f}x",
             "fewer parameters in YOLOv8s", ORANGE)
    footer(s, 6)
    pareto = [m for m in M if not any(M[o]["fps"] > M[m]["fps"] and M[o]["mAP"] > M[m]["mAP"] for o in M)]
    s.notes_slide.notes_text_frame.text = (
        f"Up and to the right is better. Pareto-optimal models (nothing is both faster and more accurate): "
        f"{', '.join(pareto)}. One-stage is faster because there is no per-proposal second stage and the input is "
        "640 instead of 800 pixels.")

    # 7 ---------------------------------------------------------------- small objects (native chart)
    s = new(); title(s, "Object size matters most")
    cd = CategoryChartData()
    cd.categories = [m.replace("FasterRCNN-", "FRCNN ").replace("-FPN-v2", "").replace("-FPN", "") for m in M]
    for key, lab in (("mAP_small", "Small"), ("mAP_medium", "Medium"), ("mAP_large", "Large")):
        cd.add_series(lab, [round(M[m][key], 3) for m in M])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.35), Inches(8.2), Inches(5.6), cd)
    ch = gf.chart
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(13)
    for ser, col in zip(ch.series, ("C4C4BF", "8A8A85", "4B4B47")):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = RGBColor.from_string(col)
    plot = ch.plots[0]; plot.gap_width = 60; plot.overlap = -5
    plot.has_data_labels = True
    plot.data_labels.font.size = Pt(10); plot.data_labels.number_format = "0.00"
    plot.data_labels.number_format_is_linked = False; plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    ch.category_axis.tick_labels.font.size = Pt(12)
    ch.value_axis.tick_labels.font.size = Pt(11)
    ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE6, 0xE6, 0xE3)
    ch.value_axis.format.line.fill.background()
    best_small = max(M, key=lambda m: M[m]["mAP_small"])
    pts = [f"• Small objects (< 32x32 px) are hardest for every model",
           f"• Best small-object mAP: {best_small} ({f3(M[best_small]['mAP_small'])})"]
    if has_pair:
        a_s, b_s = M[A]["mAP_small"], M[B]["mAP_small"]
        pts.append(f"• Faster R-CNN R50 {'beats' if a_s > b_s else 'does not beat'} YOLOv8s on small objects "
                   f"({f3(a_s)} vs {f3(b_s)})")
    pts.append("• Two-stage helps through 800-px input, a stride-4 FPN level and per-region refinement, but backbone "
               "and training recipe matter just as much")
    card(s, Inches(9.1), Inches(1.5), Inches(3.65), Inches(5.3))
    text(s, Inches(9.3), Inches(1.7), Inches(3.3), Inches(5.0), pts, 15)
    footer(s, 7)
    s.notes_slide.notes_text_frame.text = (
        f"Caution: only {n_img} images, so per-size numbers have high variance. Explain why small objects are hard: "
        "few pixels, features vanish after down-sampling, and a one-pixel box error is a large IoU error.")

    # 8 ---------------------------------------------------------------- qualitative
    s = new(); title(s, "What the detectors see")
    samples = R.get("samples", [])
    for k, smp in enumerate(samples[1:3] if len(samples) >= 3 else samples[:2]):
        y = Inches(1.35 + k * 2.95)
        pic = picture(s, os.path.join(RES, smp["file"]), Inches(0.6), y, h=Inches(2.75))
        text(s, Inches(0.6) + pic.width + Inches(0.3), y + Inches(0.3), Inches(12.7) - pic.width - Inches(0.9), Inches(2.3),
             [smp["reason"].capitalize(), f"{smp['gt_objects']} ground-truth objects ({smp['gt_small_objects']} small)",
              f"Faster R-CNN: {smp['left_boxes']} boxes", f"YOLOv8s: {smp['right_boxes']} boxes", "score ≥ 0.5"],
             15, INK)
    footer(s, 8)
    s.notes_slide.notes_text_frame.text = (
        "Blue boxes: Faster R-CNN (left). Orange boxes: YOLOv8s (right). Point out missed small objects and "
        "merged boxes in crowded areas. More examples are in the report (results/sample1-4.jpg).")

    # 9 ---------------------------------------------------------------- live demo
    s = new(); title(s, "Live demo")
    demo_img = os.path.join(RES, "demo_screenshot.png")
    if not os.path.exists(demo_img):
        demo_img = os.path.join(RES, "sample3.jpg")
    pic = picture(s, demo_img, Inches(0.6), Inches(1.4), w=Inches(7.6))
    card(s, Inches(8.6), Inches(1.4), Inches(4.1), Inches(5.4))
    text(s, Inches(8.8), Inches(1.6), Inches(3.8), Inches(0.5), "uv run demo", 16, INK, True, "Courier New")
    text(s, Inches(8.8), Inches(2.25), Inches(3.8), Inches(4.5), [
        "• Pick a two-stage and a one-stage model",
        "• Input: uploaded image, webcam snapshot or live webcam",
        "• Both models run on the same frame",
        "• Shows boxes, labels, inference time, FPS and box count for each model",
    ], 15)
    footer(s, 9)
    s.notes_slide.notes_text_frame.text = (
        "Switch to the browser. 1) Upload a photo (e.g. a crowded street) and show both outputs and timings. 2) Take a "
        "webcam snapshot. 3) Live webcam with MobileNet Faster R-CNN vs YOLOv8n to show the FPS gap. Backup if "
        "the browser fails: uv run demo --image data/val2017/<file>.jpg")

    # 10 --------------------------------------------------------------- conclusion
    s = new(dark=True); title(s, "Conclusion", dark=True)
    concl = []
    if has_pair:
        concl.append(f"• YOLOv8s: {speedup:.1f}x faster than Faster R-CNN R50-FPN v2, mAP {f3(M[B]['mAP'])} vs {f3(M[A]['mAP'])}")
    concl += [f"• Most accurate: {best_map}  ·  fastest: {best_fps}",
              ("• \"Two-stage = accurate, one-stage = fast\" still holds in direction, but the accuracy margin "
               f"({abs(M[A]['mAP'] - M[B]['mAP']) * 100:.1f} mAP points) is small next to the {speedup:.0f}x speed gap"
               if has_pair and M[A]["mAP"] > M[B]["mAP"] else
               "• \"Two-stage = accurate, one-stage = fast\" is only partly true: the one-stage model matches or beats "
               "the two-stage one on accuracy here"),
              f"• Limitations: {n_img}-image subset, pretrained only, timings depend on hardware, no TensorRT/ONNX",
              "• Next: full val2017 on GPU, ONNX/TensorRT, equal input sizes, fine-tuning on one custom dataset"]
    text(s, Inches(0.8), Inches(1.6), Inches(11.7), Inches(4.8), concl, 19, WHITE)
    text(s, Inches(0.8), Inches(6.4), Inches(11.7), Inches(0.5), "Thank you. Questions?", 20,
         RGBColor(0xB8, 0xC4, 0xD6), True, HEAD)
    footer(s, 10, dark=True)
    s.notes_slide.notes_text_frame.text = "Summarise in three sentences, then invite questions."

    prs.save(OUT)
    print("written", OUT)


if __name__ == "__main__":
    main()
