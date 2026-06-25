"""
Professional PPT Generator — Autonomous Vehicle & Object Detection System
Design: Clean minimal, no visible boxes, dark navy + gold palette
Run: python generate_ppt.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
import sys

# ── Color Palette ────────────────────────────────────────────────────────────
NAVY      = RGBColor(0x0D, 0x1B, 0x3E)
NAVY_MID  = RGBColor(0x14, 0x26, 0x54)
BLUE      = RGBColor(0x25, 0x63, 0xEB)
BLUE_LT   = RGBColor(0x93, 0xC5, 0xFD)
GOLD      = RGBColor(0xF5, 0x9E, 0x0B)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
OFF_WHITE = RGBColor(0xF8, 0xFA, 0xFF)
LGRAY     = RGBColor(0xE5, 0xE7, 0xEB)
MGRAY     = RGBColor(0x6B, 0x72, 0x80)
DGRAY     = RGBColor(0x1F, 0x2A, 0x3C)
RED       = RGBColor(0xEF, 0x44, 0x44)
GREEN     = RGBColor(0x10, 0xB9, 0x81)
ORANGE    = RGBColor(0xF9, 0x73, 0x16)
TEAL      = RGBColor(0x06, 0xB6, 0xD4)

W = 13.333
H = 7.5

prs = Presentation()
prs.slide_width  = Inches(W)
prs.slide_height = Inches(H)
blank = prs.slide_layouts[6]   # completely blank layout


# ── Helpers ───────────────────────────────────────────────────────────────────

def rect(slide, l, t, w, h, fill, alpha=None):
    """Add a borderless filled rectangle."""
    from pptx.util import Inches
    shp = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = fill
    shp.line.width = 0
    return shp


def txt(slide, text, l, t, w, h, size, bold=False, color=WHITE,
        align=PP_ALIGN.LEFT, italic=False, wrap=True, name='Calibri Light'):
    """Add a clean textbox (no visible border)."""
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf  = box.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size    = Pt(size)
    run.font.bold    = bold
    run.font.italic  = italic
    run.font.color.rgb = color
    run.font.name    = name
    box.line.color.rgb = color
    box.line.width   = 0
    return box


def mtxt(slide, lines, l, t, w, h, size, bold=False, color=WHITE,
         align=PP_ALIGN.LEFT, spacing=1.15, name='Calibri Light'):
    """Multi-line textbox with line spacing."""
    from pptx.util import Pt as PT
    from pptx.oxml.ns import qn
    from lxml import etree
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf  = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        # line spacing
        pPr = p._pPr
        if pPr is None:
            pPr = p._p.get_or_add_pPr()
        lnSpc = etree.SubElement(pPr, qn('a:lnSpc'))
        spcPct = etree.SubElement(lnSpc, qn('a:spcPct'))
        spcPct.set('val', str(int(spacing * 100000)))
        run = p.add_run()
        if isinstance(line, dict):
            run.text           = line['text']
            run.font.size      = Pt(line.get('size', size))
            run.font.bold      = line.get('bold', bold)
            run.font.color.rgb = line.get('color', color)
            run.font.name      = line.get('name', name)
        else:
            run.text           = line
            run.font.size      = Pt(size)
            run.font.bold      = bold
            run.font.color.rgb = color
            run.font.name      = name
    box.line.color.rgb = color
    box.line.width     = 0
    return box


def line(slide, x1, y1, x2, y2, color, width_pt=1.5):
    """Draw a line (as thin rectangle)."""
    from pptx.util import Inches, Pt
    import math
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    # horizontal line only (simplification for this use case)
    w_inch = abs(x2 - x1)
    h_inch = 0.01  # ~0.7pt
    shp = slide.shapes.add_shape(1,
        Inches(min(x1,x2)), Inches(y1),
        Inches(w_inch), Emu(int(width_pt * 12700)))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.color.rgb = color
    shp.line.width = 0
    return shp


def slide_num(slide, n, total=14):
    """Bottom-right slide number."""
    txt(slide, f'{n} / {total}', 12.1, 7.15, 1.1, 0.25,
        size=8, color=MGRAY, align=PP_ALIGN.RIGHT, name='Calibri')


def dark_header(slide, section_tag, title, subtitle=None):
    """Standard dark-slide header: gold tag, white title, blue subtitle."""
    txt(slide, section_tag, 0.5, 0.35, 5, 0.3, size=9, bold=True, color=GOLD,
        name='Calibri', align=PP_ALIGN.LEFT)
    txt(slide, title, 0.5, 0.65, 11.5, 0.9, size=32, bold=True, color=WHITE,
        name='Calibri Light', align=PP_ALIGN.LEFT)
    if subtitle:
        txt(slide, subtitle, 0.5, 1.5, 11, 0.4, size=14, color=BLUE_LT,
            name='Calibri Light')
    line(slide, 0.5, 1.45, 4.5, 1.45, GOLD, width_pt=2)


def light_header(slide, section_tag, title, subtitle=None):
    """Standard light-slide header."""
    txt(slide, section_tag, 0.5, 0.3, 5, 0.3, size=9, bold=True, color=BLUE,
        name='Calibri', align=PP_ALIGN.LEFT)
    txt(slide, title, 0.5, 0.55, 11.5, 0.85, size=30, bold=True, color=NAVY,
        name='Calibri Light', align=PP_ALIGN.LEFT)
    if subtitle:
        txt(slide, subtitle, 0.5, 1.35, 11, 0.35, size=13, color=MGRAY,
            name='Calibri Light')
    line(slide, 0.5, 1.32, 4.0, 1.32, BLUE, width_pt=2)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, NAVY)                        # full dark background
rect(s, 0, 0, 0.06, H, GOLD)                     # left gold stripe
rect(s, 0, H-0.06, W, 0.06, GOLD)                # bottom gold line

txt(s, 'IBM INTERNSHIP  ·  RESEARCH PROJECT  ·  2024',
    0.5, 0.5, 12, 0.35, size=9, bold=True, color=GOLD,
    name='Calibri', align=PP_ALIGN.LEFT)

txt(s, 'Autonomous Vehicle &\nObject Detection System',
    0.5, 1.15, 12, 2.2, size=42, bold=True, color=WHITE,
    name='Calibri Light', align=PP_ALIGN.LEFT)

line(s, 0.5, 3.45, 6.5, 3.45, GOLD, width_pt=2.5)

txt(s, 'A Unified Real-Time Perception Pipeline',
    0.5, 3.6, 11, 0.5, size=18, color=BLUE_LT,
    name='Calibri Light', align=PP_ALIGN.LEFT)

mtxt(s, [
    {'text': 'Object Detection  ·  3D BEV Perception  ·  Lane Detection  ·  Depth Estimation  ·  Collision Avoidance',
     'size': 12, 'color': LGRAY, 'name': 'Calibri'},
], 0.5, 4.2, 12, 0.4, size=12, color=LGRAY, name='Calibri')

txt(s, 'Powered by YOLOv11  ·  BEVFormer  ·  BEVFusion  ·  Ultra-Fast Lane  ·  Depth Anything V2  ·  ByteTrack',
    0.5, 4.65, 12, 0.35, size=11, color=MGRAY, name='Calibri Light')

rect(s, 0.5, 6.3, 1.5, 0.06, BLUE)
txt(s, 'Datasets: KITTI  ·  nuScenes', 0.5, 6.45, 5, 0.35, size=10,
    color=LGRAY, name='Calibri')
txt(s, 'Training: HPC Cluster  ·  SLURM  ·  PyTorch DDP', 6.5, 6.45, 6, 0.35,
    size=10, color=LGRAY, name='Calibri', align=PP_ALIGN.RIGHT)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — AGENDA
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, OFF_WHITE)
rect(s, 0, 0, W, 0.06, NAVY)
rect(s, 0, 0, 0.06, H, NAVY)

light_header(s, 'OVERVIEW', 'Agenda')

items = [
    ('01', 'Problem Statement',         'What is the core challenge in autonomous driving perception?'),
    ('02', 'Literature Review',         '15 high-impact papers: object detection, 3D BEV, lane, tracking'),
    ('03', 'Research Gaps',             'Critical gaps identified from surveyed literature'),
    ('04', 'Proposed Architecture',     'Our unified 5-task real-time perception pipeline'),
    ('05', 'Tech Stack & Datasets',     'Models, frameworks, KITTI & nuScenes'),
    ('06', 'HPC Training Strategy',     'Distributed training, SLURM, PyTorch DDP'),
    ('07', 'Contributions & Impact',    'How our work advances the state of the art'),
]

for i, (num, title, desc) in enumerate(items):
    y = 1.75 + i * 0.76
    rect(s, 0.5, y, 0.55, 0.55, NAVY)
    txt(s, num, 0.5, y+0.08, 0.55, 0.4, size=12, bold=True, color=WHITE,
        align=PP_ALIGN.CENTER, name='Calibri')
    txt(s, title, 1.2, y+0.02, 5.5, 0.3, size=13, bold=True, color=NAVY,
        name='Calibri', align=PP_ALIGN.LEFT)
    txt(s, desc, 1.2, y+0.28, 11.5, 0.28, size=10, color=MGRAY,
        name='Calibri Light', align=PP_ALIGN.LEFT)

slide_num(s, 2)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — PROBLEM STATEMENT
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, 0, 0.06, H, GOLD)
rect(s, 0, H-0.06, W, 0.06, GOLD)

dark_header(s, 'PROBLEM STATEMENT', 'What Problem Are We Solving?')

# Left column — the challenge
txt(s, 'THE CHALLENGE', 0.5, 1.8, 6, 0.3, size=9, bold=True, color=GOLD, name='Calibri')
problems = [
    '—  Vehicles, pedestrians, cyclists must be detected in real-time\n     across all weather conditions and lighting scenarios',
    '—  Road lanes must be identified to keep the vehicle in lane',
    '—  Distance to surrounding objects must be estimated from a\n     single camera (no LiDAR dependency)',
    '—  Collision risk must be predicted before it happens',
    '—  All of this must run simultaneously, in one system, in real-time',
]
for i, p in enumerate(problems):
    txt(s, p, 0.5, 2.15 + i*0.88, 5.9, 0.75, size=11.5, color=WHITE,
        name='Calibri Light', align=PP_ALIGN.LEFT)

# Right column — the gap
rect(s, 6.9, 1.7, 5.9, 5.35, NAVY_MID)
txt(s, 'THE GAP IN EXISTING WORK', 7.1, 1.8, 5.5, 0.3, size=9, bold=True,
    color=TEAL, name='Calibri')
line(s, 6.95, 2.18, 12.6, 2.18, TEAL, width_pt=1)

gaps = [
    ('❶', 'Papers solve each task in isolation'),
    ('❷', 'No unified real-time pipeline exists'),
    ('❸', 'Depth & tracking never combined\n    for collision estimation'),
    ('❹', 'Open-vocab detection never tested\n    on driving benchmarks'),
    ('❺', 'No joint evaluation metric for ADAS'),
]
for i, (n, g) in enumerate(gaps):
    txt(s, n, 7.1, 2.35 + i*0.88, 0.4, 0.4, size=14, color=GOLD, bold=True, name='Calibri')
    txt(s, g, 7.6, 2.35 + i*0.88, 5.2, 0.75, size=11.5, color=WHITE,
        name='Calibri Light', align=PP_ALIGN.LEFT)

slide_num(s, 3)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — LITERATURE REVIEW OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, OFF_WHITE)
rect(s, 0, 0, W, 0.06, NAVY)
rect(s, 0, 0, 0.06, H, NAVY)

light_header(s, 'LITERATURE REVIEW', 'Survey of 15 High-Impact Papers',
             subtitle='Spanning CVPR · ECCV · ICCV · NeurIPS · ICRA · IEEE TPAMI · CoRL  (2018–2024)')

categories = [
    (BLUE,   'Object\nDetection',     'YOLOv11, YOLOv10\nRT-DETR\nGrounding DINO',         '5 papers'),
    (NAVY,   '3D / BEV\nPerception',  'BEVFormer, BEVFusion\nSparse4D v3, StreamPETR\nDETR3D', '5 papers'),
    (GOLD,   'LiDAR\nEncoding',       'PointPillars\nVoxelNet',                              '2 papers'),
    (GREEN,  'Lane &\nDepth',         'CLRNet\nUFLD v2\nDepth Anything V2',                  '3 papers'),
    (TEAL,   'Multi-Object\nTracking','ByteTrack',                                            '1 paper'),
]

for i, (col, cat, models, count) in enumerate(categories):
    x = 0.45 + i * 2.58
    rect(s, x, 1.9, 2.38, 4.1, col)
    txt(s, count, x+0.15, 1.98, 2.1, 0.3, size=9, bold=True,
        color=RGBColor(0xFF,0xFF,0xFF), name='Calibri', align=PP_ALIGN.LEFT)
    txt(s, cat, x+0.15, 2.28, 2.1, 0.8, size=15, bold=True, color=WHITE,
        name='Calibri Light', align=PP_ALIGN.LEFT)
    line(s, x+0.15, 3.12, x+2.1, 3.12, WHITE, width_pt=1)
    txt(s, models, x+0.15, 3.22, 2.1, 1.9, size=10.5, color=WHITE,
        name='Calibri Light', align=PP_ALIGN.LEFT)

txt(s, 'Key Venues',
    0.5, 6.25, 3, 0.3, size=9, bold=True, color=NAVY, name='Calibri')
txt(s, 'CVPR  ·  ECCV  ·  ICCV  ·  NeurIPS  ·  ICRA  ·  IEEE TPAMI  ·  CoRL  ·  arXiv',
    0.5, 6.55, 12, 0.3, size=11, color=MGRAY, name='Calibri Light')

slide_num(s, 4)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — LITERATURE TABLE (Papers 1-8)
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, OFF_WHITE)
rect(s, 0, 0, W, 0.06, NAVY)
rect(s, 0, 0, 0.06, H, NAVY)

light_header(s, 'LITERATURE REVIEW', 'Papers 01–08: Detection, BEV & Tracking')

# Table header
rect(s, 0.35, 1.65, 12.6, 0.38, NAVY)
cols = [('  #', 0.35, 0.25), ('Authors / Title', 0.62, 4.2),
        ('Venue', 4.84, 1.2), ('Model', 6.06, 1.7),
        ('Results', 7.78, 2.5), ('Limitation', 10.3, 2.65)]
for label, x, w in cols:
    txt(s, label, x+0.05, 1.68, w, 0.32, size=9, bold=True,
        color=WHITE, name='Calibri', align=PP_ALIGN.LEFT)

papers8 = [
    ('01', 'Li et al.',    'BEVFormer',        'ECCV 2022',     'BEVFormer',     '56.9% NDS · 46.0% mAP\n(nuScenes)',       'Camera-only; degrades\nin low light'),
    ('02', 'Liu et al.',   'BEVFusion',        'ICRA 2023',     'BEVFusion',     '72.9% NDS · 70.2% mAP\n(nuScenes)',       'Needs LiDAR + Camera;\ncalibration sensitive'),
    ('03', 'Zhang et al.', 'ByteTrack',        'ECCV 2022',     'ByteTrack',     '80.3 MOTA · 77.3 IDF1\n(MOT17)',          'No ReID; struggles in\nextreme crowds'),
    ('04', 'Lin et al.',   'Sparse4D v3',      'arXiv 2023',    'Sparse4D v3',   '56.1% NDS · 49.0% AMOTA\n(nuScenes)',     'Complex pipeline;\nhigh training cost'),
    ('05', 'Zhao et al.',  'RT-DETR',          'CVPR 2024',     'RT-DETR',       '53.1% AP @ 108 FPS\n(COCO)',              'Higher memory than\nYOLO during training'),
    ('06', 'Wang et al.',  'YOLOv10',          'NeurIPS 2024',  'YOLOv10',       '46.3% AP · 25% fewer\nparams vs YOLOv9', 'NMS-free adds\noptimization complexity'),
    ('07', 'Ultralytics',  'YOLOv11',          '2024',          'YOLOv11',       '54.7% mAP (x) · 22%\nfewer params',      'No peer-reviewed\npublication'),
    ('08', 'Wang et al.',  'StreamPETR',       'ICCV 2023',     'StreamPETR',    '67.6% NDS · 65.3%\nAMOTA (nuScenes)',    'Temporal dependency\nincreases latency'),
]

row_colors = [WHITE, OFF_WHITE]
for i, (num, auth, title, venue, model, results, limit) in enumerate(papers8):
    y = 2.1 + i * 0.63
    rect(s, 0.35, y, 12.6, 0.62, row_colors[i % 2])
    txt(s, num,     0.4,  y+0.1, 0.22, 0.45, size=9,    bold=True,  color=BLUE,  name='Calibri')
    txt(s, auth,    0.63, y+0.02, 4.15, 0.25, size=8.5,  bold=True,  color=NAVY,  name='Calibri')
    txt(s, title,   0.63, y+0.27, 4.15, 0.3,  size=8,    bold=False, color=MGRAY, name='Calibri Light')
    txt(s, venue,   4.84, y+0.1,  1.18, 0.45, size=8,    bold=False, color=DGRAY, name='Calibri')
    txt(s, model,   6.06, y+0.1,  1.68, 0.45, size=8,    bold=True,  color=BLUE,  name='Calibri')
    txt(s, results, 7.78, y+0.02, 2.45, 0.55, size=7.5,  bold=False, color=GREEN, name='Calibri')
    txt(s, limit,   10.3, y+0.02, 2.6,  0.55, size=7.5,  bold=False, color=RED,   name='Calibri Light')

slide_num(s, 5)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — LITERATURE TABLE (Papers 9-15)
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, OFF_WHITE)
rect(s, 0, 0, W, 0.06, NAVY)
rect(s, 0, 0, 0.06, H, NAVY)

light_header(s, 'LITERATURE REVIEW', 'Papers 09–15: Lane, Depth, LiDAR & Open-Vocabulary')

rect(s, 0.35, 1.65, 12.6, 0.38, NAVY)
for label, x, w in cols:
    txt(s, label, x+0.05, 1.68, w, 0.32, size=9, bold=True,
        color=WHITE, name='Calibri', align=PP_ALIGN.LEFT)

papers7 = [
    ('09', 'Zheng et al.',  'CLRNet',             'CVPR 2022',      'CLRNet',             '80.47% F1 (CULane)\n97.89% (TuSimple)',     'Larger model; struggles\nwith extreme curves'),
    ('10', 'Qin et al.',    'UFLDv2',             'IEEE TPAMI 2022','UFLDv2',             '96.4% Acc · 300+ FPS\n(TuSimple)',           'Fixed grid; misses\nextreme-angle lanes'),
    ('11', 'Yang et al.',   'Depth Anything V2',  'NeurIPS 2024',   'Depth Anything V2',  'SOTA NYUv2 & KITTI\n10× faster vs diffusion','Night / tunnel\nperformance drops'),
    ('12', 'Lang et al.',   'PointPillars',       'CVPR 2019',      'PointPillars',       '82.58% AP Car\n(KITTI Mod.) @ 62 Hz',       'Loses fine-grained\nvertical spatial info'),
    ('13', 'Zhou & Tuzel',  'VoxelNet',           'CVPR 2018',      'VoxelNet',           'SOTA KITTI (2018)\nAll 3 classes',           'Memory-intensive;\nnot real-time (2018)'),
    ('14', 'Wang et al.',   'DETR3D',             'CoRL 2022',      'DETR3D',             '42.5% NDS · 34.6% mAP\n(nuScenes val)',      'No temporal modeling;\nsingle frame only'),
    ('15', 'Liu et al.',    'Grounding DINO',     'ECCV 2024',      'Grounding DINO',     '52.5% AP zero-shot\n(COCO)',                  'Never tested on\ndriving benchmarks'),
]

for i, (num, auth, title, venue, model, results, limit) in enumerate(papers7):
    y = 2.1 + i * 0.72
    rect(s, 0.35, y, 12.6, 0.71, row_colors[i % 2])
    txt(s, num,     0.4,  y+0.12, 0.22, 0.45, size=9,   bold=True,  color=BLUE,  name='Calibri')
    txt(s, auth,    0.63, y+0.04, 4.15, 0.25, size=8.5, bold=True,  color=NAVY,  name='Calibri')
    txt(s, title,   0.63, y+0.3,  4.15, 0.35, size=8,   bold=False, color=MGRAY, name='Calibri Light')
    txt(s, venue,   4.84, y+0.12, 1.18, 0.45, size=8,   bold=False, color=DGRAY, name='Calibri')
    txt(s, model,   6.06, y+0.12, 1.68, 0.45, size=8,   bold=True,  color=BLUE,  name='Calibri')
    txt(s, results, 7.78, y+0.04, 2.45, 0.6,  size=7.5, bold=False, color=GREEN, name='Calibri')
    txt(s, limit,   10.3, y+0.04, 2.6,  0.6,  size=7.5, bold=False, color=RED,   name='Calibri Light')

slide_num(s, 6)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — RESEARCH GAPS
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, 0, 0.06, H, GOLD)
rect(s, 0, H-0.06, W, 0.06, GOLD)

dark_header(s, 'RESEARCH GAPS', 'What the Literature Leaves Unsolved',
            subtitle='Critical gaps identified across 15 surveyed papers')

gaps = [
    (RED,    'G1', 'CRITICAL', 'No Unified Pipeline',
     'All 15 papers solve tasks in isolation. No open-source system\ncombines detection + lane + depth + tracking + collision in real-time.'),
    (RED,    'G2', 'CRITICAL', '16-Point Camera vs LiDAR Gap',
     'BEVFormer (camera) scores 56.9 NDS vs BEVFusion (LiDAR+cam) at 72.9 NDS.\nThis gap has persisted since 2022 with no clear solution.'),
    (RED,    'G3', 'CRITICAL', 'Depth + Tracking Never Fused for TTC',
     'Depth Anything V2 and ByteTrack are studied separately.\nNo paper combines them for Time-to-Collision collision warning.'),
    (ORANGE, 'G4', 'MODERATE', 'Grounding DINO Untested on Driving Data',
     'Paper 15 evaluates only on COCO/LVIS. Zero-shot performance\non KITTI and nuScenes has never been measured.'),
    (ORANGE, 'G5', 'MODERATE', 'No Joint ADAS Evaluation Metric',
     'COCO, CULane, MOT17 are all separate benchmarks.\nNo paper proposes a unified driving perception score.'),
]

for i, (col, tag, severity, title, desc) in enumerate(gaps):
    y = 1.85 + i * 0.98
    rect(s, 0.5, y, 0.06, 0.75, col)
    rect(s, 0.7, y, 0.9,  0.3,  col)
    txt(s, severity, 0.72, y+0.04, 0.88, 0.24, size=7.5, bold=True,
        color=WHITE, name='Calibri', align=PP_ALIGN.CENTER)
    txt(s, tag,  1.75, y+0.02, 0.5,  0.28, size=11, bold=True, color=col, name='Calibri')
    txt(s, title, 2.3, y+0.02, 4.5, 0.3, size=12, bold=True, color=WHITE, name='Calibri')
    txt(s, desc, 1.75, y+0.38, 11, 0.52, size=10, color=LGRAY, name='Calibri Light')

slide_num(s, 7)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — PROPOSED ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, OFF_WHITE)
rect(s, 0, 0, W, 0.06, NAVY)
rect(s, 0, 0, 0.06, H, NAVY)

light_header(s, 'PROPOSED SOLUTION', 'Unified Real-Time Perception Pipeline',
             subtitle='Five SOTA models connected into one inference graph — camera-first, HPC-trained')

# Pipeline flow
stages = [
    (NAVY,  '📷',  'INPUT\nSOURCE',      'Camera\nVideo / Image'),
    (BLUE,  '🎯',  'YOLOv11',            'Object Detection\n2D Boxes + Classes'),
    (GOLD,  '🛣️',  'Ultra-Fast\nLane v2','Lane Detection\nF1 > 96%'),
    (GREEN, '📏',  'Depth\nAnything V2', 'Monocular Depth\nMetric Distance'),
    (TEAL,  '🔄',  'ByteTrack',          'Multi-Object\nTracking + Velocity'),
    (RED,   '⚠️',  'TTC\nCalculator',    'Collision Risk\nTime-to-Collision'),
]

for i, (col, icon, name_, desc) in enumerate(stages):
    x = 0.42 + i * 2.15
    rect(s, x, 1.85, 1.95, 2.0, col)
    txt(s, icon,  x+0.1, 1.92, 1.75, 0.45, size=18, color=WHITE,
        align=PP_ALIGN.CENTER, name='Segoe UI Emoji')
    txt(s, name_, x+0.1, 2.42, 1.75, 0.7, size=11, bold=True, color=WHITE,
        align=PP_ALIGN.CENTER, name='Calibri')
    line(s, x+0.25, 3.14, x+1.7, 3.14, WHITE, width_pt=1)
    txt(s, desc,  x+0.1, 3.2, 1.75, 0.62, size=9, color=WHITE,
        align=PP_ALIGN.CENTER, name='Calibri Light')
    if i < 5:
        txt(s, '→', x+2.0, 2.6, 0.2, 0.4, size=20, color=MGRAY,
            align=PP_ALIGN.CENTER, name='Calibri')

# Output box
rect(s, 0.42, 4.2, 12.45, 1.2, NAVY)
txt(s, 'OUTPUT  —  Single annotated frame: Bounding Boxes  ·  Track IDs  ·  Lane Overlay  ·  Depth Heatmap  ·  ⚠ Collision Warning',
    0.6, 4.42, 12.1, 0.4, size=12.5, bold=True, color=WHITE,
    align=PP_ALIGN.CENTER, name='Calibri')
txt(s, 'Export: ONNX  ·  TensorRT (3–5× faster)  ·  CoreML  ·  OpenVINO',
    0.6, 4.85, 12.1, 0.4, size=10, color=LGRAY,
    align=PP_ALIGN.CENTER, name='Calibri Light')

txt(s, 'Training: BEVFormer (camera-only 3D) and BEVFusion (LiDAR+Camera 3D) run in parallel track on HPC',
    0.5, 5.6, 12.3, 0.35, size=10, color=MGRAY,
    align=PP_ALIGN.CENTER, name='Calibri Light')

slide_num(s, 8)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — TECH STACK
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, 0, 0.06, H, GOLD)
rect(s, 0, H-0.06, W, 0.06, GOLD)

dark_header(s, 'TECH STACK', 'Models, Frameworks & Tools')

categories_tech = [
    (BLUE,  'Detection',
     ['YOLOv11 (2D, real-time)', 'YOLOv10 (NMS-free)', 'RT-DETR (transformer)', 'Grounding DINO (open-vocab)']),
    (GREEN, '3D / BEV',
     ['BEVFormer (camera-only)', 'BEVFusion (LiDAR+cam)', 'Sparse4D v3 (temporal)', 'StreamPETR (online)']),
    (GOLD,  'Lane & Depth',
     ['Ultra-Fast Lane v2', 'CLRNet', 'Depth Anything V2', 'Metric depth (absolute)']),
    (TEAL,  'Tracking',
     ['ByteTrack (primary)', 'BotSort (ReID-enhanced)', 'Simple IoU Tracker (baseline)']),
    (ORANGE,'Frameworks',
     ['PyTorch + TorchVision', 'Ultralytics Ecosystem', 'MMDetection3D', 'Albumentations']),
    (RED,   'Infra / Deploy',
     ['HPC + SLURM + DDP', 'Mixed Precision (AMP)', 'ONNX / TensorRT', 'TensorBoard / WandB']),
]

for i, (col, cat, items_list) in enumerate(categories_tech):
    row, col_i = divmod(i, 3)
    x = 0.45 + col_i * 4.28
    y = 1.8  + row   * 2.55
    rect(s, x, y, 0.06, 2.1, col)
    txt(s, cat, x+0.2, y+0.08, 3.9, 0.35, size=12, bold=True, color=col, name='Calibri')
    line(s, x+0.2, y+0.46, x+3.95, y+0.46, MGRAY, width_pt=0.8)
    for j, item in enumerate(items_list):
        txt(s, f'·  {item}', x+0.2, y+0.55+j*0.38, 3.9, 0.35,
            size=10.5, color=LGRAY, name='Calibri Light')

slide_num(s, 9)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — DATASETS
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, OFF_WHITE)
rect(s, 0, 0, W, 0.06, NAVY)
rect(s, 0, 0, 0.06, H, NAVY)

light_header(s, 'DATASETS', 'Benchmarks Used for Training & Evaluation')

datasets = [
    (NAVY, 'KITTI', '2D & 3D Object Detection',
     [('Classes', '8 — Car, Van, Truck, Pedestrian,\nCyclist, Tram, Misc'),
      ('Samples', '7,481 training / 7,518 testing frames'),
      ('Sensors', 'Stereo camera + 64-beam Velodyne LiDAR'),
      ('Metric',  'AP at 0.5/0.7 IoU (Easy / Moderate / Hard)'),
      ('Source',  'cvlibs.net/datasets/kitti')]),
    (BLUE, 'nuScenes', 'Multi-Camera 3D Detection & Tracking',
     [('Classes', '10 — Car, Truck, Bus, Pedestrian,\nCyclist, Motorcycle + more'),
      ('Samples', '700 train / 150 val scenes (28k frames)'),
      ('Sensors', '6 cameras + 1 LiDAR + 5 radars, 360°'),
      ('Metric',  'NDS (nuScenes Detection Score), mAP, AMOTA'),
      ('Source',  'nuscenes.org')]),
    (GREEN, 'TuSimple\n& CULane', 'Lane Detection Benchmarks',
     [('TuSimple', '3,626 train / 2,782 test clips,\n96.4% UFLD Acc'),
      ('CULane',   '88k train / 34k test frames,\nChallenging urban scenes'),
      ('Metric',   'F1@50 IoU (CULane), Accuracy (TuSimple)'),
      ('Source',   'github.com/TuSimple/tusimple-benchmark')]),
]

for i, (col, name_, subtitle_, stats) in enumerate(datasets):
    x = 0.42 + i * 4.3
    rect(s, x, 1.75, 0.07, 4.65, col)
    txt(s, name_,     x+0.2, 1.78, 3.95, 0.55, size=18, bold=True, color=col, name='Calibri Light')
    txt(s, subtitle_, x+0.2, 2.3,  3.95, 0.35, size=10, color=MGRAY, name='Calibri Light')
    line(s, x+0.2, 2.7, x+4.0, 2.7, LGRAY, width_pt=1)
    for j, (label, val) in enumerate(stats):
        txt(s, label, x+0.2, 2.82+j*0.72, 1.2,  0.3, size=9,  bold=True, color=NAVY,  name='Calibri')
        txt(s, val,   x+1.45, 2.82+j*0.72, 2.7, 0.55, size=9.5, color=DGRAY, name='Calibri Light')

slide_num(s, 10)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — HPC TRAINING
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, 0, 0.06, H, GOLD)
rect(s, 0, H-0.06, W, 0.06, GOLD)

dark_header(s, 'TRAINING STRATEGY', 'High Performance Computing — Distributed Training')

# Left: DDP explanation
txt(s, 'PyTorch Distributed Data Parallel  (DDP)', 0.5, 1.85, 6, 0.35,
    size=13, bold=True, color=GOLD, name='Calibri')

ddp_pts = [
    '·  Each GPU holds a complete model copy',
    '·  Forward + backward run independently per GPU',
    '·  Gradients synchronized via NCCL all-reduce',
    '·  Effective batch = batch_per_gpu × num_gpus',
    '·  LR scaled linearly with effective batch size',
    '·  Mixed precision (AMP) gives ~2× speedup',
]
for i, p in enumerate(ddp_pts):
    txt(s, p, 0.5, 2.3+i*0.52, 6.0, 0.45, size=11, color=WHITE, name='Calibri Light')

# Right: SLURM job config
rect(s, 7.0, 1.75, 5.85, 5.1, NAVY_MID)
txt(s, 'SLURM Job Configuration', 7.2, 1.82, 5.5, 0.35, size=11, bold=True,
    color=TEAL, name='Calibri')
line(s, 7.05, 2.22, 12.6, 2.22, TEAL, width_pt=1)

slurm_lines = [
    ('#SBATCH --nodes=4',           GOLD),
    ('#SBATCH --gres=gpu:8',        GOLD),
    ('#SBATCH --time=48:00:00',     GOLD),
    ('',                            WHITE),
    ('torchrun \\',                 GREEN),
    ('  --nproc_per_node=8 \\',     GREEN),
    ('  --nnodes=4 \\',             GREEN),
    ('  --rdzv_backend=c10d \\',    GREEN),
    ('  training/train_ddp.py \\',  WHITE),
    ('  --config configs/yolo11.yaml', WHITE),
]
for i, (line_txt, col) in enumerate(slurm_lines):
    txt(s, line_txt, 7.2, 2.35+i*0.43, 5.5, 0.4, size=9.5,
        color=col, name='Courier New')

slide_num(s, 11)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — OUR CONTRIBUTIONS
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, OFF_WHITE)
rect(s, 0, 0, W, 0.06, NAVY)
rect(s, 0, 0, 0.06, H, NAVY)

light_header(s, 'CONTRIBUTIONS', 'What Our Work Contributes to the Field')

contribs = [
    (GREEN,  '✓', 'Unified 5-Task Pipeline',
     'First open-source system combining object detection + lane detection + depth estimation\n+ multi-object tracking + collision warning into one real-time HPC-trainable pipeline'),
    (BLUE,   '✓', 'Novel TTC Estimation (Closes Gap G3)',
     'First integration of ByteTrack per-object velocity with Depth Anything V2 metric depth\nfor real-time Time-to-Collision estimation from a single monocular camera'),
    (GOLD,   '✓', 'Open-Vocab on Driving Data (Closes Gap G4)',
     'First evaluation of Grounding DINO on KITTI and nuScenes — establishing a zero-shot\nbaseline against fine-tuned YOLOv11 on driving benchmarks'),
    (TEAL,   '✓', 'Camera vs Fusion Comparison (Closes Gap G2)',
     'Controlled comparison of BEVFormer (camera-only) vs BEVFusion (LiDAR+camera)\nunder identical training settings to quantify the sensor fusion benefit'),
    (ORANGE, '✓', 'HPC-Ready Open Codebase',
     'Full pipeline with SLURM scripts, DDP training, ONNX/TensorRT export and\nMMDetection3D integration — reproducible on any HPC cluster'),
]

for i, (col, tick, title, desc) in enumerate(contribs):
    y = 1.75 + i * 1.0
    rect(s, 0.42, y+0.05, 0.06, 0.75, col)
    txt(s, tick,  0.65, y+0.1,  0.35, 0.4, size=16, color=col, bold=True, name='Calibri')
    txt(s, title, 1.1,  y+0.08, 11.5, 0.35, size=12.5, bold=True, color=NAVY, name='Calibri')
    txt(s, desc,  1.1,  y+0.42, 11.5, 0.52, size=10.5, color=MGRAY, name='Calibri Light')

slide_num(s, 12)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — EXPECTED RESULTS
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, 0, 0.06, H, GOLD)
rect(s, 0, H-0.06, W, 0.06, GOLD)

dark_header(s, 'EXPECTED RESULTS', 'Performance Targets & Evaluation Plan')

metrics = [
    ('YOLOv11\n(KITTI)',        'mAP@0.5',  '≥ 85%',   BLUE,   '2D Object Detection'),
    ('YOLOv11\n(nuScenes)',     'mAP@0.5',  '≥ 60%',   BLUE,   '10-class vehicle detection'),
    ('BEVFormer\n(nuScenes)',   'NDS',       '≥ 50%',   GOLD,   'Camera-only 3D detection'),
    ('BEVFusion\n(nuScenes)',   'NDS',       '≥ 70%',   GOLD,   'LiDAR+Camera fusion'),
    ('UFLDv2\n(TuSimple)',      'Accuracy',  '≥ 96%',   GREEN,  'Lane detection'),
    ('TTC System\n(combined)',  'Latency',   '< 50 ms', TEAL,   'Real-time on RTX 3090'),
]

for i, (model, metric, target, col, label) in enumerate(metrics):
    row, c = divmod(i, 3)
    x = 0.5  + c * 4.25
    y = 1.85 + row * 2.55
    rect(s, x, y, 3.95, 2.1, NAVY_MID)
    rect(s, x, y, 3.95, 0.06, col)
    txt(s, model,  x+0.15, y+0.18, 3.7, 0.7,  size=13, bold=True,  color=WHITE, name='Calibri Light', align=PP_ALIGN.CENTER)
    txt(s, metric, x+0.15, y+0.85, 3.7, 0.3,  size=9,  bold=False, color=MGRAY, name='Calibri', align=PP_ALIGN.CENTER)
    txt(s, target, x+0.15, y+1.12, 3.7, 0.5,  size=22, bold=True,  color=col,   name='Calibri Light', align=PP_ALIGN.CENTER)
    txt(s, label,  x+0.15, y+1.65, 3.7, 0.35, size=8,  bold=False, color=LGRAY, name='Calibri', align=PP_ALIGN.CENTER)

slide_num(s, 13)


# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — THANK YOU
# ════════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, 0, 0.06, H, GOLD)
rect(s, W-0.06, 0, 0.06, H, GOLD)
rect(s, 0, H-0.06, W, 0.06, GOLD)
rect(s, 0, 0, W, 0.06, GOLD)

txt(s, 'Thank You', 0.5, 1.6, 12.3, 1.6, size=54, bold=True, color=WHITE,
    name='Calibri Light', align=PP_ALIGN.CENTER)

line(s, 3.5, 3.3, 9.8, 3.3, GOLD, width_pt=2)

txt(s, 'Autonomous Vehicle & Object Detection System',
    0.5, 3.5, 12.3, 0.5, size=15, color=BLUE_LT,
    name='Calibri Light', align=PP_ALIGN.CENTER)

refs = [
    'BEVFormer (ECCV 2022)  ·  BEVFusion (ICRA 2023)  ·  Sparse4D v3 (arXiv 2023)  ·  ByteTrack (ECCV 2022)',
    'CLRNet (CVPR 2022)  ·  UFLDv2 (TPAMI 2022)  ·  Depth Anything V2 (NeurIPS 2024)',
    'RT-DETR (CVPR 2024)  ·  YOLOv10 (NeurIPS 2024)  ·  YOLOv11 (Ultralytics 2024)',
    'PointPillars (CVPR 2019)  ·  VoxelNet (CVPR 2018)  ·  DETR3D (CoRL 2022)  ·  Grounding DINO (ECCV 2024)  ·  StreamPETR (ICCV 2023)',
]
for i, ref in enumerate(refs):
    txt(s, ref, 0.5, 4.3+i*0.42, 12.3, 0.38, size=8.5, color=MGRAY,
        name='Calibri Light', align=PP_ALIGN.CENTER)

txt(s, 'Datasets: KITTI  ·  nuScenes  ·  TuSimple  ·  CULane',
    0.5, 6.15, 12.3, 0.3, size=9, color=LGRAY,
    name='Calibri', align=PP_ALIGN.CENTER)

txt(s, 'Training Infrastructure: HPC Cluster  ·  SLURM  ·  PyTorch DDP  ·  Mixed Precision AMP',
    0.5, 6.5, 12.3, 0.3, size=9, color=LGRAY,
    name='Calibri', align=PP_ALIGN.CENTER)

slide_num(s, 14)


# ── Save ─────────────────────────────────────────────────────────────────────
OUT = 'Autonomous_Vehicle_Detection_Presentation.pptx'
prs.save(OUT)
print(f'[OK] Saved: {OUT}')
print(f'  Slides: 14')
print(f'  Design: Dark navy + gold + blue palette, no bordered boxes')
