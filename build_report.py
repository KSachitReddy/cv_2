"""
build_report.py - builds CV_Assignment2_Report.docx from results/results.json

Every number in the report is read from results/results.json (produced by benchmark.py).
Comparative sentences ("X is faster than Y") are generated from those numbers, so the text
always matches the data. Code snippets are copied from the real source files with `inspect`.

The table of contents uses the sample-report format (S. No. | Section | Page). Page numbers are
found by converting the .docx to PDF with LibreOffice and searching each page for the headings
(two passes). If LibreOffice is not installed, the page column is left blank.

Usage:  uv run report
"""
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import benchmark
import common

ROOT = common.ROOT
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "CV_Assignment2_Report.docx")
TITLE = "Comparative Analysis of Two-Stage (R-CNN) vs. One-Stage (YOLO) Detectors for Speed and Accuracy"

# Details the students must fill in (left as visible placeholders in the document).
STUDENTS = [("{{NAME_1}}", "{{REGD_NO_1}}"), ("{{NAME_2}}", "{{REGD_NO_2}}"),
            ("{{NAME_3}}", "{{REGD_NO_3}}"), ("{{NAME_4}}", "{{REGD_NO_4}}")]
PROGRAM = "{{PROGRAM}}"            # e.g. B.Tech CSE, III Year, Semester V, Section E/F/G
FACULTY = "{{FACULTY_NAME}}"
GITHUB = "{{GITHUB_LINK}}"

# Published full-val2017 box mAP of the YOLOv8 weights, from the Ultralytics docs (ref. [6]).
# Used only in the "comparison with published results" table, clearly labelled as such.
# The torchvision values are read from the weights' own metadata at run time.
YOLO_PUBLISHED_MAP = {"YOLOv8n": 37.3, "YOLOv8s": 44.9, "YOLOv8m": 50.2}


# ============================================================================ formatting helpers
def set_cell_bg(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def set_widths(table, widths_cm):
    """Fixed column widths: set the table grid and every cell (LibreOffice reads the grid)."""
    table.autofit = False
    grid = table._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths_cm):
        gc.set(qn("w:w"), str(int(Cm(w).twips)))
    for row in table.rows:
        for cell, w in zip(row.cells, widths_cm):
            cell.width = Cm(w)


def add_field(run, instr):
    """Insert a Word field such as PAGE into a run."""
    for tag, text in (("begin", None), ("instr", instr), ("separate", None), ("text", "1"), ("end", None)):
        if tag == "instr":
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text
        elif tag == "text":
            el = OxmlElement("w:t"); el.text = text
        else:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag)
        run._r.append(el)


class Report:
    def __init__(self):
        self.doc = Document()
        self.fig_no = 0
        self.tab_no = 0
        self.headings = []          # (number, title, level) for the table of contents
        self._setup_styles()

    def _setup_styles(self):
        d = self.doc
        sec = d.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        sec.left_margin = sec.right_margin = Cm(2.54)
        sec.top_margin = sec.bottom_margin = Cm(2.54)
        st = d.styles["Normal"]
        st.font.name = "Times New Roman"; st.font.size = Pt(12)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        st.paragraph_format.line_spacing = 1.5
        st.paragraph_format.space_after = Pt(4)
        for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Title", 20)):
            s = d.styles[name]
            s.font.name = "Times New Roman"; s.font.size = Pt(size); s.font.bold = True
            s.font.color.rgb = RGBColor(0, 0, 0)
            rpr = s.element.get_or_add_rPr()
            rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts"); rpr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(a), "Times New Roman")
            s.paragraph_format.space_before = Pt(12 if name != "Heading 2" else 8)
            s.paragraph_format.space_after = Pt(6)
            s.paragraph_format.line_spacing = 1.15
        # footer page number, centred
        p = sec.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        add_field(p.add_run(), "PAGE")

    # ---- content primitives
    def para(self, text="", bold=False, italic=False, align=None, size=None, space_after=None):
        p = self.doc.add_paragraph()
        self._runs(p, text, bold, italic, size)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if align is None else align
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        return p

    @staticmethod
    def _runs(p, text, bold=False, italic=False, size=None):
        """Supports **bold** fragments inside text."""
        for part in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*)", text):
            if not part:
                continue
            is_b, is_i = part.startswith("**"), (part.startswith("*") and not part.startswith("**"))
            r = p.add_run(part.strip("*") if (is_b or is_i) else part)
            r.bold = bold or is_b; r.italic = italic or is_i
            if size:
                r.font.size = Pt(size)

    def bullets(self, items):
        for it in items:
            p = self.doc.add_paragraph(style="List Bullet")
            self._runs(p, it)
            p.paragraph_format.space_after = Pt(2)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def numbered(self, items):
        for k, it in enumerate(items, 1):
            p = self.doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.9); p.paragraph_format.first_line_indent = Cm(-0.6)
            p.paragraph_format.space_after = Pt(2)
            self._runs(p, f"{k}.  {it}")

    def h1(self, number, title, new_page=True):
        text = f"{number}. {title}" if number else title
        h = self.doc.add_heading(text, level=1)
        h.paragraph_format.page_break_before = new_page
        self.headings.append((number, title, 1))

    def h2(self, number, title):
        self.doc.add_heading(f"{number} {title}", level=2)
        self.headings.append((number, title, 2))

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def figure(self, path, caption, width_cm=15.5):
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(path, width=Cm(width_cm))
        self.fig_no += 1
        c = self.para(f"Figure {self.fig_no}: {caption}", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=10)
        c.paragraph_format.line_spacing = 1.15
        return self.fig_no

    def table(self, header, rows, caption, widths=None, font=9.5):
        self.tab_no += 1
        c = self.para(f"Table {self.tab_no}: {caption}", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=10)
        c.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for r_i, row in enumerate([header] + rows):
            for c_i, val in enumerate(row):
                cell = t.cell(r_i, c_i)
                cell.text = ""
                p = cell.paragraphs[0]
                p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(0)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_i else WD_ALIGN_PARAGRAPH.LEFT
                self._runs(p, str(val), bold=(r_i == 0), size=font)
                if r_i == 0:
                    set_cell_bg(cell, "D9E2F3")
        if widths:
            set_widths(t, widths)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(0)
        return self.tab_no

    def code(self, src):
        t = self.doc.add_table(rows=1, cols=1); t.style = "Table Grid"
        cell = t.cell(0, 0); set_cell_bg(cell, "F4F4F2")
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(0)
        lines = src.rstrip("\n").split("\n")
        for k, line in enumerate(lines):
            r = p.add_run(line); r.font.name = "Courier New"; r.font.size = Pt(8.5)
            r._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Courier New")
            if k < len(lines) - 1:
                r.add_break()
        self.doc.add_paragraph().paragraph_format.space_after = Pt(0)


# ============================================================================ number helpers
def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:.1f}"


def load_results():
    with open(os.path.join(RES, "results.json")) as f:
        return json.load(f)


def facts(R):
    """Derive every comparative statement used in the text from results.json."""
    M = R["models"]
    two = [m for m in M if M[m]["family"] == "two-stage"]
    one = [m for m in M if M[m]["family"] == "one-stage"]
    F = {"M": M, "two": two, "one": one}
    F["fastest"] = max(M, key=lambda m: M[m]["fps"])
    F["slowest"] = min(M, key=lambda m: M[m]["fps"])
    F["best"] = max(M, key=lambda m: M[m]["mAP"])
    F["worst"] = min(M, key=lambda m: M[m]["mAP"])
    F["best_small"] = max(M, key=lambda m: M[m]["mAP_small"])
    F["best_two"] = max(two, key=lambda m: M[m]["mAP"]) if two else None
    F["best_one"] = max(one, key=lambda m: M[m]["mAP"]) if one else None
    F["fast_one"] = max(one, key=lambda m: M[m]["fps"]) if one else None
    F["fast_two"] = max(two, key=lambda m: M[m]["fps"]) if two else None
    # Pareto front: models for which no other model is both faster and more accurate
    F["pareto"] = [m for m in M if not any(M[o]["fps"] > M[m]["fps"] and M[o]["mAP"] > M[m]["mAP"] for o in M)]
    return F


METRIC_NAMES = {"mAP": "mAP", "mAP50": "mAP50", "mAP75": "mAP75", "mAP_small": "small-object mAP",
                "mAP_medium": "medium-object mAP", "mAP_large": "large-object mAP"}


def pair_verdict(a, b, tie=0.005):
    """Which of the two models leads on each accuracy metric (differences below `tie` count as a tie)."""
    a_wins = [METRIC_NAMES[k] for k in METRIC_NAMES if a[k] - b[k] > tie]
    b_wins = [METRIC_NAMES[k] for k in METRIC_NAMES if b[k] - a[k] > tie]
    ties = [METRIC_NAMES[k] for k in METRIC_NAMES if abs(a[k] - b[k]) <= tie]
    return a_wins, b_wins, ties


def ratio(a, b):
    return a / b if b else float("nan")


# ============================================================================ the report
def build(toc_pages=None):
    R = load_results(); F = facts(R); M = F["M"]; meta = R["meta"]
    n_img, dev = meta["num_images"], meta["device_name"]
    on_cpu = meta["device"] == "cpu"
    r = Report(); d = r.doc
    A, B = "FasterRCNN-R50-FPN-v2", "YOLOv8s"         # the headline pair
    has_pair = A in M and B in M

    # ------------------------------------------------------------------ cover page
    logo_top = os.path.join(ROOT, "assets", "logo_ifhe.png"); logo_bottom = os.path.join(ROOT, "assets", "logo_icfaitech.png")
    if os.path.exists(logo_top):
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(logo_top, width=Cm(13))
    for t, sz, b, gap in (("A", 12, False, 6), ("Computer Vision Course Assignment Project Report", 13, False, 6),
                          ("On", 12, False, 20), (TITLE, 16, True, 26), ("Submitted by:", 12, True, 6)):
        r.para(t, bold=b, align=WD_ALIGN_PARAGRAPH.CENTER, size=sz, space_after=gap)
    for name, regd in STUDENTS:
        r.para(f"{name}  ({regd})", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
    r.para(PROGRAM, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=22)
    r.para(f"Faculty Name: {FACULTY}", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
    if os.path.exists(logo_bottom):
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(logo_bottom, width=Cm(6))
    for t in ("Faculty of Science & Technology", "IFHE University, Hyderabad", "Nov 2026"):
        r.para(t, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)

    # ------------------------------------------------------------------ abstract
    r.page_break()
    r.para("ABSTRACT", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=16, space_after=12)
    abstract = (
        "Object detectors fall into two broad families. Two-stage detectors such as Faster R-CNN first "
        "propose candidate regions and then classify and refine each region, while one-stage detectors such "
        "as YOLO predict classes and boxes for the whole image in a single forward pass. This project measures "
        "the practical speed-accuracy trade-off between the two families under identical conditions. Five "
        "pretrained COCO models were considered: Faster R-CNN with ResNet-50-FPN (v2) and MobileNetV3-Large-FPN "
        "backbones (torchvision), and YOLOv8 n/s/m (Ultralytics). "
        f"We benchmarked {len(M)} of them ({', '.join(M)}) on the first {n_img} images of COCO val2017 "
        f"on {dev}, measuring per-image end-to-end latency at batch size 1 after a warm-up, frames per second, "
        "parameter count and weight size, and accuracy with the official COCO evaluation "
        "(mAP@[.5:.95], mAP50, mAP75 and mAP for small, medium and large objects). ")
    if has_pair:
        a, b = M[A], M[B]
        abstract += (
            f"YOLOv8s ran {ratio(b['fps'], a['fps']):.1f}x faster than Faster R-CNN R50-FPN v2 "
            f"({b['fps']:.2f} vs {a['fps']:.2f} FPS) with an mAP of {f3(b['mAP'])} against {f3(a['mAP'])}. ")
    abstract += (
        f"The most accurate model in our run was {F['best']} (mAP {f3(M[F['best']]['mAP'])}) and the fastest was "
        f"{F['fastest']} ({M[F['fastest']]['fps']:.1f} FPS). Small objects were the hardest category for every "
        f"model; the best small-object mAP was {f3(M[F['best_small']]['mAP_small'])} ({F['best_small']}). "
        "We also built a Streamlit application that runs both detectors side by side on images, video and a "
        "webcam and reports their inference time live. The results show that a modern one-stage detector "
        "now offers a better speed-accuracy balance for real-time use, while the two-stage design remains a "
        "strong, well-understood baseline whose region-based second stage is costly on resource-limited hardware.")
    r.para(abstract)
    r.para("**Keywords:** Object detection, Faster R-CNN, YOLOv8, Region Proposal Network, "
           "mean Average Precision (mAP), COCO dataset, Inference latency, Speed-accuracy trade-off")

    # ------------------------------------------------------------------ table of contents (sample format)
    r.page_break()
    r.para("TABLE OF CONTENTS", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=16, space_after=12)
    toc_anchor = d.add_paragraph()          # the TOC table is inserted here at the end

    # ------------------------------------------------------------------ 1. introduction
    r.h1("1", "Introduction")
    r.h2("1.1", "Background of the Project")
    r.para("Object detection asks a model to find every object of interest in an image, draw a bounding box "
           "around it and name its class. It underpins autonomous driving, video surveillance, robotics, retail "
           "analytics and medical imaging. Since 2014 deep convolutional networks have dominated the task, and they "
           "have evolved along two lines. The R-CNN family [1], [5] (R-CNN, Fast R-CNN, Faster R-CNN) is "
           "**two-stage**: a region proposal step selects a few hundred to a few thousand candidate boxes and a second "
           "network classifies and refines each one. The YOLO family [2] (You Only Look Once) is **one-stage**: it "
           "treats detection as a single regression problem and predicts all boxes and classes in one pass over the "
           "image. The common belief is that two-stage detectors are more accurate, especially on small objects, "
           "while one-stage detectors are faster. Both families have changed a lot since their first papers, so the "
           "belief needs to be checked with current models.")
    r.h2("1.2", "Problem Definition")
    r.para("Given a fixed set of pretrained detectors, a fixed set of test images and fixed hardware, we want to "
           "quantify how much accuracy is gained or lost and how much speed is gained or lost when moving from a "
           "two-stage to a one-stage detector. The comparison must be fair: the same images, batch size 1 (as in a "
           "live video stream), the same evaluation code, the same score threshold, and timing that excludes "
           "one-time start-up costs. The question we answer is: **for a real-time application on our hardware, which "
           "detector family gives the better trade-off, and where does each one fail?**")
    r.h2("1.3", "Objectives")
    r.bullets([
        "Benchmark two-stage detectors (Faster R-CNN ResNet-50-FPN v2 and MobileNetV3-Large-FPN) and one-stage "
        "detectors (YOLOv8n, YOLOv8s, YOLOv8m) with pretrained COCO weights and no retraining.",
        "Measure speed: mean and standard deviation of per-image latency, FPS, parameter count and weight size.",
        "Measure accuracy with the official COCO protocol: mAP@[.5:.95], mAP50, mAP75 and mAP by object size.",
        "Analyse the speed-accuracy trade-off and small-object behaviour, supported by charts and visual examples.",
        "Build a live demo that runs both detectors side by side on images, video and a webcam.",
    ])

    # ------------------------------------------------------------------ 2. case design
    r.h1("2", "Case Design / Algorithm Implementation")
    r.h2("2.1", "Dataset")
    r.para(f"We use **COCO 2017 validation** [3] (5,000 images, 80 object classes). Because the experiments run "
           f"on {'a CPU' if on_cpu else 'a single GPU'}, we evaluate on the first **{n_img} images** (sorted by image "
           f"id), which contain **{meta['num_gt_boxes']} ground-truth boxes**. Only the image files we need are "
           "downloaded, and only instances_val2017.json is extracted from the annotation archive (with HTTP range "
           "requests, so the 241 MB zip is never stored on disk). All models were trained on COCO train2017, so "
           "val2017 is unseen data for every model.")
    r.h2("2.2", "Tools, Languages and Frameworks")
    v = meta["versions"]
    r.table(["Component", "Used for", "Version"], [
        ["Python", "All code", v["python"]],
        ["PyTorch", "Tensor computation, inference", v["torch"]],
        ["torchvision", "Faster R-CNN models and weights", v["torchvision"]],
        ["Ultralytics", "YOLOv8 models and weights", v["ultralytics"]],
        ["pycocotools", "Official COCO mAP evaluation", v["pycocotools"]],
        ["OpenCV, NumPy", "Image I/O, drawing, arrays", f"{v['opencv']}, {v['numpy']}"],
        ["Matplotlib", "Charts and diagrams", "-"],
        ["Streamlit", "Live demo web application", "-"],
        ["python-docx, python-pptx", "Generating this report and the slides", "-"],
    ], "Software stack (versions recorded by benchmark.py)", widths=[4.2, 7.5, 3.5])

    r.h2("2.3", "Two-Stage Detector: Faster R-CNN")
    r.para("Faster R-CNN [1] has four parts (Figure 1, top row):")
    r.bullets([
        "**Backbone:** a CNN (ResNet-50 or MobileNetV3) pretrained on ImageNet that turns the image into feature maps.",
        "**Feature Pyramid Network (FPN) [4]:** combines deep, semantically strong but coarse features with shallow, "
        "high-resolution features through a top-down path with lateral connections. Each pyramid level handles a "
        "range of object sizes, which helps with small objects.",
        "**Region Proposal Network (RPN), stage 1:** a small convolutional network slides over every pyramid level. "
        "At each position it places several **anchors** (reference boxes of different sizes and aspect ratios) and "
        "predicts an objectness score and four box offsets for each anchor. After NMS the top proposals "
        "(about 1,000 at test time) are kept.",
        "**RoI head, stage 2:** RoIAlign (the improved form of RoI pooling [5]) cuts a fixed 7x7 feature grid out of "
        "the feature map for every proposal using bilinear interpolation. Fully connected layers then predict "
        "the class (81 outputs including background) and a class-specific box refinement. Per-class NMS gives the "
        "final detections.",
    ])
    r.para("We use torchvision's **v2** weights for the ResNet-50 model, which were trained with a modern, longer "
           "recipe [8] and reach a higher mAP than the original 2015 configuration.")
    r.para("The second stage is run once **per proposal**, so its cost grows with the number of proposals. "
           "torchvision also resizes images so the short side is 800 pixels, which helps accuracy but costs time.")
    r.h2("2.4", "One-Stage Detector: YOLOv8")
    r.para("YOLO [2] removes the proposal stage. YOLOv8 [6] works as follows (Figure 1, bottom row):")
    r.bullets([
        "**Backbone:** CSPDarknet with C2f blocks (cross-stage partial connections) to extract features cheaply.",
        "**Neck:** a PAN-FPN that fuses features top-down and bottom-up, producing three scales (strides 8, 16, 32).",
        "**Decoupled, anchor-free head:** every grid cell on every scale directly predicts class probabilities in one "
        "branch and box geometry in another branch, as distances from the cell centre to the four box sides "
        "(Distribution Focal Loss). There are no hand-designed anchors.",
        "**Post-processing:** a confidence threshold and NMS. The whole network runs once per image at 640x640.",
    ])
    r.para("Because everything is a single dense convolutional pass with no per-region computation, YOLO is well "
           "suited to real-time video. The n/s/m variants share the design and differ only in depth and width.")
    r.figure(os.path.join(RES, "fig_architectures.png"),
             "Pipelines of the two detector families compared in this project")

    r.h2("2.5", "Evaluation Metrics")
    r.bullets([
        "**IoU (Intersection over Union):** area of overlap between predicted and ground-truth box divided by the "
        "area of their union. A detection is a true positive if IoU is at least a threshold and the class is right.",
        "**Precision** = TP / (TP + FP), the share of detections that are correct; **Recall** = TP / (TP + FN), the "
        "share of objects that were found.",
        "**AP (Average Precision):** the area under the precision-recall curve of one class, which is obtained by "
        "sweeping the score threshold. COCO uses 101-point interpolation [9].",
        "**mAP:** AP averaged over the 80 classes. **mAP@[.5:.95]** (the main COCO metric) also averages over ten IoU "
        "thresholds from 0.50 to 0.95, which rewards precise localisation. **mAP50** and **mAP75** use a single IoU "
        "threshold. **mAP small / medium / large** restrict evaluation to objects with area below 32x32 pixels, "
        "between 32x32 and 96x96, and above 96x96.",
        "**Latency** is the wall-clock time for one image (preprocessing + network + NMS); **FPS = 1000 / mean "
        "latency in ms**.",
    ])
    r.h2("2.6", "Benchmarking Methodology")
    r.bullets([
        f"Batch size 1 and the default input size of each model (torchvision: {meta['input_size']['torchvision']}; "
        f"YOLO: {meta['input_size']['yolo']}).",
        f"{meta['warmup_images']} warm-up images per model, not timed (first calls include memory allocation and "
        "lazy initialisation).",
        "torch.cuda.synchronize() before and after each timed call on GPU, because CUDA kernels run asynchronously.",
        f"Image decoding from disk is excluded from the timing; {meta['timing'].split(':')[0]} {meta['timing'].split(':')[1].strip()}.",
        f"Score threshold {meta['eval_score_threshold']} and at most {meta['max_detections']} detections per image "
        "for evaluation, so that the full precision-recall curve can be computed.",
        "YOLO class indices 0-79 are mapped to the sorted COCO category ids (1-90 with gaps); torchvision already "
        "outputs COCO ids. benchmark.py asserts that this mapping matches the annotation file.",
    ] + ([f"**Hardware note:** {nt}" for nt in meta["notes"]]))
    r.figure(os.path.join(RES, "fig_flowchart.png"), "Flowchart of the benchmarking procedure (benchmark.py)",
             width_cm=11)

    r.h2("2.7", "Key Code Snippets")
    r.para("A single function hides the differences between the two libraries, so both families go through "
           "exactly the same timing and evaluation code (common.py):")
    src = inspect.getsource(common.detect)
    src = "\n".join(l for l in src.split("\n") if not l.strip().startswith(('"""', "The timed", "so latency", "network +")))
    r.code(src)
    r.para("Timing with synchronisation (common.py) and conversion to COCO format plus evaluation (benchmark.py):")
    r.code(inspect.getsource(common.timed_detect).replace('    """detect() plus wall-clock latency in milliseconds (synchronised on GPU)."""\n', ""))
    ev = inspect.getsource(benchmark.coco_eval)
    ev = ev[ev.index("    with contextlib"):]
    r.code("# COCO wants [x, y, width, height]; one dict per detected box\n"
           "dets.append({\"image_id\": img_id, \"category_id\": int(c),\n"
           "             \"bbox\": [x1, y1, x2 - x1, y2 - y1], \"score\": float(s)})\n\n" + ev.rstrip())
    r.h2("2.8", "Modules and Live Demo Application")
    r.table(["File", "Purpose"], [
        ["common.py", "Model registry, loading, unified detect(), timing, drawing"],
        ["benchmark.py", "Data download, timing loop, COCO evaluation, charts, sample images, results.json"],
        ["figures.py", "Architecture diagram and methodology flowchart"],
        ["demo.py", "Streamlit side-by-side demo (image, upload, webcam, video) and CLI fallback"],
        ["build_report.py / build_slides.py", "Generate this report and the presentation from results.json"],
    ], "Project modules", widths=[5, 10.5])
    r.para("The demo (uv run demo) lets the user choose a two-stage and a one-stage model, an input source "
           "(uploaded image, webcam snapshot or live webcam) and a score threshold. "
           "Both models run on the same frame and the app shows the annotated outputs next to each other with the "
           "inference time, FPS and number of boxes of each model. Models are cached and warmed up once when "
           "first selected so the demo responds quickly.")

    # ------------------------------------------------------------------ 3. results
    r.h1("3", "Results and Discussion")
    r.h2("3.1", "Quantitative Results")
    r.para(f"All results below were measured on **{dev}** with **{n_img} COCO val2017 images** "
           f"(run date {meta['date'][:10]}).")
    rows = []
    for m, x in M.items():
        rows.append([m, x["family"], f"{x['params_M']:.1f}", f"{x['weight_file_MB']:.1f}",
                     f"{x['latency_ms_mean']:.1f} ± {x['latency_ms_std']:.1f}", f"{x['fps']:.2f}",
                     f3(x["mAP"]), f3(x["mAP50"]), f3(x["mAP75"])])
    t_main = r.table(["Model", "Family", "Params (M)", "Weights (MB)", "Latency (ms)", "FPS",
                      "mAP", "mAP50", "mAP75"], rows,
                     "Speed and accuracy of all models (mAP = COCO mAP@[.5:.95])",
                     widths=[3.6, 1.6, 1.3, 1.4, 2.2, 1.0, 1.1, 1.3, 1.3], font=8.5)
    t_size = r.table(["Model", "mAP small", "mAP medium", "mAP large", "AR@100"],
                     [[m, f3(x["mAP_small"]), f3(x["mAP_medium"]), f3(x["mAP_large"]), f3(x["AR100"])]
                      for m, x in M.items()], "Accuracy by object size and average recall",
                     widths=[5.2, 2.6, 2.6, 2.6, 2.6])
    r.para("Note on weight size: the Ultralytics .pt files store weights in 16-bit floating point while the "
           "torchvision files are 32-bit, so file size is not proportional to parameter count across libraries.",
           size=10)
    r.figure(os.path.join(RES, "chart_map.png"), "COCO mAP@[.5:.95] per model", 13.5)
    r.figure(os.path.join(RES, "chart_fps.png"), "Throughput (FPS) per model", 13.5)
    r.figure(os.path.join(RES, "chart_latency.png"), "Mean per-image latency per model", 13.5)
    r.figure(os.path.join(RES, "chart_speed_vs_accuracy.png"), "Speed vs accuracy trade-off", 13.5)
    r.figure(os.path.join(RES, "chart_map_by_size.png"), "mAP by object size", 13.5)

    r.h2("3.2", "Speed-Accuracy Trade-off")
    txt = []
    lat_sorted = sorted(M, key=lambda m: M[m]["latency_ms_mean"])
    txt.append(f"From fastest to slowest the models rank: {', '.join(lat_sorted)}. The fastest model, "
               f"{F['fastest']}, runs at {M[F['fastest']]['fps']:.1f} FPS, which is "
               f"{ratio(M[F['fastest']]['fps'], M[F['slowest']]['fps']):.0f}x the throughput of the slowest, "
               f"{F['slowest']} ({M[F['slowest']]['fps']:.2f} FPS).")
    if has_pair:
        a, b = M[A], M[B]
        diff = b["mAP"] - a["mAP"]
        txt.append(f"Comparing the main pair, YOLOv8s is {ratio(b['fps'], a['fps']):.1f}x faster than Faster R-CNN "
                   f"R50-FPN v2 and its mAP is {abs(diff) * 100:.1f} points {'higher' if diff > 0 else 'lower'} "
                   f"({f3(b['mAP'])} vs {f3(a['mAP'])}). It also uses {ratio(a['params_M'], b['params_M']):.1f}x fewer "
                   "parameters.")
    if F["fast_two"] and F["best_one"]:
        mt, yo = M[F["fast_two"]], M[F["best_one"]]
        txt.append(f"Even the lighter two-stage model, {F['fast_two']}, reaches only {mt['fps']:.2f} FPS "
                   f"({mt['latency_ms_mean']:.0f} ms per image) with mAP {f3(mt['mAP'])}, while {F['best_one']} "
                   f"reaches {yo['fps']:.2f} FPS with mAP {f3(yo['mAP'])}.")
    pf = F["pareto"]
    txt.append(f"The Pareto-optimal models (no other model is both faster and more accurate) are: {', '.join(pf)}. "
               + ("All of them are one-stage models, which means that on this hardware a YOLOv8 variant is the better "
                  "choice at every speed requirement." if all(M[m]["family"] == "one-stage" for m in pf) else
                  "The front contains both families, so the right choice depends on whether speed or accuracy "
                  "matters more for the application."))
    for t in txt:
        r.para(t)
    r.para("**Why is the one-stage detector faster?** Faster R-CNN runs its region head separately for each of "
           "about 1,000 proposals and processes larger (800-pixel) inputs, whereas YOLOv8 makes all predictions with "
           "one dense pass over a 640-pixel input. On a CPU, which has little parallelism compared with a GPU, the "
           "extra per-region work and larger input are especially expensive.")

    r.h2("3.3", "Small-Object Performance")
    small_rank = sorted(M, key=lambda m: -M[m]["mAP_small"])
    s_txt = (f"Every model scores far lower on small objects than on large ones (for example {F['best']}: "
             f"{f3(M[F['best']]['mAP_small'])} small vs {f3(M[F['best']]['mAP_large'])} large). The ranking on small "
             f"objects is " + ", ".join(f"{m} ({f3(M[m]['mAP_small'])})" for m in small_rank) + ". ")
    if has_pair:
        a_s, b_s = M[A]["mAP_small"], M[B]["mAP_small"]
        if a_s > b_s:
            s_txt += (f"Here the classical expectation holds: Faster R-CNN R50-FPN v2 beats YOLOv8s on small objects "
                      f"({f3(a_s)} vs {f3(b_s)}). Its larger 800-pixel input keeps more pixels per small object, the FPN "
                      "gives it a high-resolution P2 level (stride 4, versus stride 8 as the finest YOLOv8 level), and "
                      "the second stage re-examines each candidate region with RoIAlign.")
        else:
            s_txt += (f"In our run YOLOv8s matches or beats Faster R-CNN R50-FPN v2 on small objects ({f3(b_s)} vs "
                      f"{f3(a_s)}), contrary to the classical expectation. YOLOv8 benefits from many years of training "
                      "improvements (mosaic augmentation, better losses and label assignment), which narrows the "
                      "historical gap, although Faster R-CNN still has the finer P2 feature level and larger input.")
    r.para(s_txt)
    if F["two"]:
        mob = "FasterRCNN-MobileNetV3-FPN"
        if mob in M:
            r.para(f"The MobileNetV3 version of Faster R-CNN shows how much the backbone matters: its small-object "
                   f"mAP is only {f3(M[mob]['mAP_small'])}. It uses the same 800-pixel input, but its FPN is built only on the "
                   "last two MobileNetV3 stages (stride 16 and 32 plus one pooled level), so it has no high-resolution "
                   "feature level for small objects, and its backbone is much weaker. Being two-stage by itself does "
                   "not guarantee good small-object accuracy.")
    r.para(f"Caution: with only {n_img} images the small-object category contains few instances of many classes, "
           "so these per-size numbers have a large variance and differences of a few points are not significant.")

    r.h2("3.4", "Qualitative Results")
    samples = R.get("samples", [])
    for s in samples:
        r.figure(os.path.join(RES, s["file"]),
                 f"{s['reason'].capitalize()} (COCO image {s['image_id']}, {s['gt_objects']} ground-truth objects, "
                 f"{s['gt_small_objects']} small). Left: Faster R-CNN R50-FPN v2, {s['left_boxes']} boxes; "
                 f"right: YOLOv8s, {s['right_boxes']} boxes (score >= 0.5)", 15.5)
    if samples:
        cr = samples[0]
        r.para(f"In the crowded scene ({cr['gt_objects']} annotated objects), Faster R-CNN produced "
               f"{cr['left_boxes']} boxes above 0.5 confidence and YOLOv8s produced {cr['right_boxes']}. "
               "Both detectors find the large, clearly visible objects reliably; the differences appear on small, "
               "overlapping and partially occluded objects, where boxes are missed or merged by NMS.")
        sm = next((x for x in samples if x["reason"] == "small objects"), None)
        if sm:
            more = "Faster R-CNN" if sm["left_boxes"] > sm["right_boxes"] else "YOLOv8s"
            r.para(f"The small-object image ({sm['gt_small_objects']} of {sm['gt_objects']} objects are small) shows the "
                   f"largest gap: {sm['left_boxes']} boxes from Faster R-CNN against {sm['right_boxes']} from YOLOv8s. "
                   f"{more} recovers more of the small, distant objects, which "
                   + ("agrees" if (sm["left_boxes"] > sm["right_boxes"]) == (M[A]["mAP_small"] > M[B]["mAP_small"])
                      else "does not fully agree")
                   + f" with the small-object mAP in Table {t_size}. Two cautions apply to these pictures: the two models calibrate their confidence scores "
                   "differently (Faster R-CNN tends to give higher scores), so one fixed display threshold of 0.5 is "
                   "not equally strict for both; and a box count above the number of annotated objects means some "
                   "boxes are duplicates, false positives or objects the annotators did not label.")

    r.h2("3.5", "Comparison with Published Results")
    pub_rows = []
    for m, x in M.items():
        if common.MODELS[m]["kind"] == "torchvision":
            meta_w = common.MODELS[m]["weights"].meta
            pub = meta_w.get("_metrics", {}).get("COCO-val2017", {}).get("box_map")
            src = "[7] torchvision weights metadata"
        else:
            pub = YOLO_PUBLISHED_MAP.get(m); src = "[6] Ultralytics docs"
        pub_rows.append([m, f"{pub:.1f}" if pub is not None else "-", pct(x["mAP"]), src])
    r.table(["Model", "Published mAP (5,000 images)", f"Our mAP ({n_img} images)", "Source"], pub_rows,
            "Our mAP (x100) next to the published full-val2017 figures",
            widths=[4.8, 3.4, 3.0, 4.4])
    r.para("Our values differ from the published ones because we evaluate a small subset; the published numbers use "
           "all 5,000 validation images. The ranking of the models is the useful part of the comparison. Published "
           "speeds are measured on GPUs with TensorRT or similar runtimes and are not comparable to our "
           f"{'CPU ' if on_cpu else ''}timings.")

    r.h2("3.6", "Challenges Faced and Solutions")
    r.bullets([
        "**Different libraries, different outputs:** torchvision returns COCO category ids (1-90), YOLO returns "
        "class indices (0-79). We map indices to sorted COCO ids and assert the mapping against the annotation file.",
        "**Fair timing:** first inferences are slow and GPU calls are asynchronous. We added 10 un-timed warm-up "
        "images and synchronisation around every timed call, and excluded image loading from the timed region.",
        "**Fair accuracy:** detectors normally drop low-confidence boxes, which truncates the precision-recall curve. "
        "We set the score threshold to 0.001 and the maximum to 100 detections for both libraries.",
        "**Limited disk space and no GPU:** instead of downloading the 241 MB annotation archive, only the one JSON "
        "file is extracted with HTTP range requests, and on CPU the run uses 100 images and skips the largest "
        "YOLO model by default.",
        "**Reproducibility:** every number in this report is read from results.json, which the benchmark writes, "
        "so the report can be regenerated after any re-run.",
    ])

    # ------------------------------------------------------------------ 4. conclusion
    r.h1("4", "Conclusion")
    r.h2("4.1", "Summary of What Was Achieved")
    summ = (f"We built a reproducible benchmark that compares two-stage and one-stage detectors under identical "
            f"conditions and evaluated {len(M)} pretrained models on {n_img} COCO val2017 images on {dev}. ")
    if has_pair:
        a, b = M[A], M[B]
        summ += (f"The headline result is that YOLOv8s delivered {ratio(b['fps'], a['fps']):.1f}x the throughput of "
                 f"Faster R-CNN R50-FPN v2 with {'higher' if b['mAP'] > a['mAP'] else 'lower'} overall mAP "
                 f"({f3(b['mAP'])} vs {f3(a['mAP'])}). ")
    summ += (f"The best overall accuracy came from {F['best']} and the best small-object accuracy from "
             f"{F['best_small']}. ")
    if has_pair:
        a_w, b_w, ties = pair_verdict(M[A], M[B])
        if len(a_w) > len(b_w):
            summ += ("The classical rule \"two-stage is accurate, one-stage is fast\" still holds in direction: "
                     f"Faster R-CNN R50-FPN v2 leads YOLOv8s on {', '.join(a_w)}"
                     + (f", while YOLOv8s leads on {', '.join(b_w)}" if b_w else "")
                     + (f" and they are level on {', '.join(ties)}" if ties else "")
                     + f". However, the accuracy margin ({abs(M[A]['mAP'] - M[B]['mAP']) * 100:.1f} mAP points) is "
                     f"small compared with the {ratio(M[B]['fps'], M[A]['fps']):.0f}x speed difference, so for "
                     "real-time use the one-stage detector is the better choice, and the two-stage detector is worth "
                     "its cost only when accuracy, especially on small objects, matters more than speed. ")
        else:
            summ += ("The classical rule \"two-stage is accurate, one-stage is fast\" is only partly true for these "
                     f"models: YOLOv8s is faster and leads on {', '.join(b_w) or 'no metric'}, while Faster R-CNN "
                     f"leads on {', '.join(a_w) or 'no metric'}. ")
    summ += "We also delivered a live side-by-side demo application."
    r.para(summ)
    r.h2("4.2", "Limitations")
    r.bullets([
        f"Only {n_img} of the 5,000 COCO val2017 images were used, so mAP values have noticeable sampling variance.",
        "Only pretrained weights were evaluated; no model was trained or fine-tuned on a common recipe, so "
        "differences partly reflect training recipes (YOLOv8 is far newer than the original Faster R-CNN).",
        f"Timings are hardware-dependent: they were measured on {dev}"
        + (", where memory was shared with other applications" if on_cpu else "") + ", and will differ on other "
        "machines, especially on GPUs where the speed gap usually shrinks in absolute terms.",
        "No inference optimisation (TensorRT, ONNX Runtime, half precision, batching or quantisation) was used.",
        "Input resolutions differ (800 vs 640 pixels) because we used each model's default configuration.",
    ] + (["YOLOv8m was skipped in the CPU run to keep the run time short."] if "YOLOv8m" not in M else []))
    r.h2("4.3", "Possible Improvements")
    r.bullets([
        "Evaluate on all 5,000 validation images on a GPU and repeat timings several times to give confidence intervals.",
        "Export models to ONNX/TensorRT and use FP16 or INT8 to measure deployment-level speed.",
        "Compare at equal input resolution, and add other designs (RetinaNet, DETR / RT-DETR, YOLO newer versions).",
        "Fine-tune both families on the same custom dataset with the same budget for a training-controlled comparison.",
        "Measure energy and memory use, which matter for edge devices.",
    ])

    # ------------------------------------------------------------------ references
    r.h1("", "References")
    r.numbered([
        "S. Ren, K. He, R. Girshick, and J. Sun, \"Faster R-CNN: Towards real-time object detection with region "
        "proposal networks,\" in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 28, 2015, pp. 91-99.",
        "J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, \"You only look once: Unified, real-time object "
        "detection,\" in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, 2016, "
        "pp. 779-788.",
        "T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, \"Microsoft "
        "COCO: Common objects in context,\" in *Proc. European Conf. Computer Vision (ECCV)*, Zurich, Switzerland, "
        "2014, pp. 740-755.",
        "T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie, \"Feature pyramid networks for "
        "object detection,\" in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Honolulu, HI, "
        "USA, 2017, pp. 2117-2125.",
        "R. Girshick, \"Fast R-CNN,\" in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, Santiago, Chile, 2015, "
        "pp. 1440-1448.",
        "G. Jocher, A. Chaurasia, and J. Qiu, \"Ultralytics YOLOv8,\" Ultralytics, 2023. [Online]. Available: "
        "https://docs.ultralytics.com/models/yolov8/ [Accessed: Nov. 8, 2026].",
        "PyTorch Contributors, \"Models and pre-trained weights - Object detection (torchvision),\" PyTorch.org. "
        "[Online]. Available: https://docs.pytorch.org/vision/stable/models.html [Accessed: Nov. 8, 2026].",
        "Y. Li, S. Xie, X. Chen, P. Dollar, K. He, and R. Girshick, \"Benchmarking detection transfer learning with "
        "vision transformers,\" arXiv:2111.11429, 2021. (Training recipe behind torchvision's Faster R-CNN v2 weights.)",
        "COCO Consortium, \"COCO detection evaluation,\" cocodataset.org. [Online]. Available: "
        "https://cocodataset.org/#detection-eval [Accessed: Nov. 8, 2026].",
    ])

    # ------------------------------------------------------------------ appendix
    r.h1("", "Appendix")
    r.h2("A.", "Source Code")
    r.para(f"GitHub repository: {GITHUB}")
    r.h2("B.", "How to Run")
    r.para("The project is managed with uv (https://docs.astral.sh/uv/). uv reads pyproject.toml and uv.lock, "
           "creates the virtual environment and installs the exact package versions on first use.")
    r.code("uv sync          # one-time setup: environment + packages\n"
           "uv run bench     # benchmark: downloads data + weights, writes results/\n"
           "uv run build     # diagrams + this report + the slides\n"
           "uv run demo      # live demo in the browser")
    r.h2("C.", "Project Structure")
    r.code("cv26_rcnn_vs_yolo/\n"
           "├── tasks.py           the short commands (uv run bench / report / slides / build / demo)\n"
           "├── common.py          model loading, unified detect(), timing, drawing\n"
           "├── benchmark.py       COCO subset benchmark -> results/results.json + charts\n"
           "├── figures.py         diagrams for report and slides\n"
           "├── demo.py            Streamlit live demo (+ CLI fallback)\n"
           "├── build_report.py    -> CV_Assignment2_Report.docx\n"
           "├── build_slides.py    -> CV_Assignment2_Slides.pptx\n"
           "├── pyproject.toml / uv.lock   dependencies (uv)\n"
           "├── README.md\n"
           "├── data/              COCO images + instances_val2017.json (downloaded)\n"
           "├── weights/           model weights (downloaded)\n"
           "└── results/           results.json, charts, sample images, diagrams")
    r.h2("D.", "Experiment Environment")
    r.para(f"Device: {dev}; PyTorch threads: {meta['torch_threads']}; run date: {meta['date']}; "
           f"versions: " + ", ".join(f"{k} {v_}" for k, v_ in meta["versions"].items()) + ".", size=10)

    # ------------------------------------------------------------------ fill the TOC table
    top = [(n, t, lvl) for n, t, lvl in r.headings]
    tbl = d.add_table(rows=1 + len(top), cols=3); tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c, (txt, w) in enumerate((("S. No.", 1.8), ("Section Title", 11.5), ("Page No.", 2.2))):
        cell = tbl.cell(0, c); cell.text = ""; cell.width = Cm(w)
        run = cell.paragraphs[0].add_run(txt); run.bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_bg(cell, "D9E2F3")
    for i, (n, t, lvl) in enumerate(top, 1):
        vals = [n, t, str(toc_pages.get(i - 1, "")) if toc_pages else ""]
        for c, val in enumerate(vals):
            cell = tbl.cell(i, c); cell.text = ""
            p = cell.paragraphs[0]; p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val); run.bold = (lvl == 1)
            if c == 1 and lvl == 2:
                p.paragraph_format.left_indent = Cm(0.6)
            if c != 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.width = Cm((1.8, 11.5, 2.2)[c])
    set_widths(tbl, (1.8, 11.5, 2.2))
    toc_anchor._p.addnext(tbl._tbl)          # move the table to the TOC page
    d.save(OUT)
    return r.headings


# ============================================================================ page numbers for the TOC
def find_pages(headings):
    """Convert the docx to PDF and return {heading index: page number}."""
    if not shutil.which("soffice") or not shutil.which("pdftotext"):
        print("LibreOffice/pdftotext not found: TOC page numbers left blank")
        return None
    tmp = tempfile.mkdtemp()
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, OUT],
                   check=True, capture_output=True, timeout=180)
    pdf = os.path.join(tmp, os.path.basename(OUT).replace(".docx", ".pdf"))
    text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout
    pages = text.split("\f")
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
    found, start = {}, 3                      # headings start after the TOC page (page 3)
    for i, (n, t, _) in enumerate(headings):
        key = norm(f"{n}. {t}" if n and "." not in n else (f"{n} {t}" if n else t))
        for p in range(start, len(pages)):
            if any(norm(line).startswith(key) for line in pages[p].split("\n")):
                found[i] = p + 1; start = p
                break
    shutil.copy(pdf, OUT.replace(".docx", ".pdf"))   # handy preview / upload copy
    return found


def main():
    import figures
    figures.main()                            # the report needs the diagrams, so (re)draw them first
    hs = build()
    pages = find_pages(hs)
    if pages is not None:
        build(pages)
        pages2 = find_pages(hs)               # re-check after the numbers were inserted
        if pages2 != pages:
            build(pages2); find_pages(hs)
        missing = [hs[i][1] for i in range(len(hs)) if i not in (pages2 or {})]
        if missing:
            print("could not locate page for:", missing)
    print("written", OUT)


if __name__ == "__main__":
    main()
