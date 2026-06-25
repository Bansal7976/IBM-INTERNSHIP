"""
Fix Autonomous_Vehicle_Detection_Presentation.pptx:
  - Remove full-slide black background rectangles
  - Switch all text from white → dark navy
  - Recolour thin accent bars to navy
  - Set slide background to white
  - Remove emoji characters from text
  - Fix any text-box alignment issues
"""
import sys, io, copy, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree
import copy

SRC = "Autonomous_Vehicle_Detection_Presentation.pptx"
DST = "Autonomous_Vehicle_Detection_Presentation_Fixed.pptx"

EMU = 914400
SLIDE_W = 13.333 * EMU
SLIDE_H = 7.5 * EMU

# Colour map
C_NAVY    = RGBColor(0x1C, 0x35, 0x57)  # main text / titles
C_BLUE    = RGBColor(0x25, 0x63, 0xEB)  # accent bars + bold labels
C_DGRAY   = RGBColor(0x1E, 0x29, 0x3B)  # body text
C_MGRAY   = RGBColor(0x47, 0x55, 0x69)  # sub-text
C_LGRAY   = RGBColor(0x94, 0xA3, 0xB8)  # page numbers / hints
C_WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
C_XGRAY   = RGBColor(0xF1, 0xF4, 0xF8)  # table alt-row fill
C_DIVIDER = RGBColor(0xCB, 0xD5, 0xE1)  # thin rule

# Emoji pattern to strip
EMOJI_RE = re.compile(
    u"[\U00010000-\U0010ffff"
    u"\U0001F300-\U0001F9FF"
    u"\U00002600-\U000027BF"
    u"\U0000FE00-\U0000FE0F"
    u"❶❷❸❹❺✓📷🎯🛣️📏🔄⚠️→·]",
    flags=re.UNICODE
)

def hex_to_rgb(hexstr):
    h = hexstr.strip('#')
    return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

def shape_covers_slide(shp):
    """True if this rectangle is a full-slide background cover."""
    try:
        l = shp.left; t = shp.top
        w = shp.width; h = shp.height
        return (abs(l) < 20000 and abs(t) < 20000 and
                abs(w - SLIDE_W) < 20000 and abs(h - SLIDE_H) < 20000)
    except:
        return False

def is_accent_bar(shp):
    """Thin vertical or horizontal decorative line (w<=0.10" or h<=0.10")."""
    try:
        w = shp.width / EMU
        h = shp.height / EMU
        return (w <= 0.12 or h <= 0.12)
    except:
        return False

def recolour_run(run, new_color):
    run.font.color.rgb = new_color

def choose_text_color(run_text, para_text, shp_name, slide_idx):
    """Pick the right dark colour for a text run."""
    txt = run_text.strip()
    # page numbers → light gray
    if re.match(r'^\d+ / \d+$', txt):
        return C_LGRAY
    # section labels (all-caps short strings)
    if txt.isupper() and len(txt) <= 30:
        return C_NAVY
    # G1..G5, 01-15 numbering
    if re.match(r'^G\d$', txt) or re.match(r'^0?\d{1,2}$', txt) or re.match(r'^\d\d$', txt):
        return C_BLUE
    # CRITICAL / MODERATE severity tags
    if txt in ('CRITICAL', 'MODERATE', 'HIGH'):
        return C_BLUE
    # code-looking text
    if txt.startswith('#SBATCH') or txt.startswith('torchrun') or txt.startswith('--') or txt.startswith('training/'):
        return C_NAVY
    return C_DGRAY

def strip_emojis(text):
    return EMOJI_RE.sub('', text)

def fix_shape_fill(shp, slide_idx):
    """
    - Full-slide bg rect → make white
    - Thin accent bar → make navy
    - Other filled rects → light gray fill for visual structure
    """
    if shape_covers_slide(shp):
        shp.fill.solid()
        shp.fill.fore_color.rgb = C_WHITE
        # remove line border
        shp.line.color.rgb = C_WHITE
        return

    if is_accent_bar(shp):
        try:
            shp.fill.solid()
            shp.fill.fore_color.rgb = C_BLUE
        except:
            pass
        return

    # Other rectangles — check if they are column/row containers
    try:
        w = shp.width / EMU
        h = shp.height / EMU
        # Large panel rectangles (column separators, row containers)
        if h > 3.0 and w < 3.0:
            # These are column side-bar separators → make divider-colour thin bars
            shp.fill.solid()
            shp.fill.fore_color.rgb = C_BLUE
            return
        if h < 0.08:
            # Horizontal rules
            shp.fill.solid()
            shp.fill.fore_color.rgb = C_DIVIDER
            try: shp.line.color.rgb = C_DIVIDER
            except: pass
            return
        # Row highlight rectangles in tables / lit review tables
        if w > 10.0 and h < 0.8:
            fill = shp.fill
            # Alt-row tint → very light gray
            fill.solid()
            fill.fore_color.rgb = C_XGRAY
            try: shp.line.color.rgb = C_DIVIDER
            except: pass
            return
        # Panel / container rectangles → white with border
        shp.fill.solid()
        shp.fill.fore_color.rgb = C_XGRAY
        try: shp.line.color.rgb = C_DIVIDER
        except: pass
    except:
        pass

prs = Presentation(SRC)

# Set slide background to white for all slides
for slide in prs.slides:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = C_WHITE

slide_idx = 0
for slide in prs.slides:
    slide_idx += 1

    shapes_to_process = list(slide.shapes)
    for shp in shapes_to_process:
        # --- Recolour shape fills ---
        try:
            if shp.shape_type == 1:  # AUTO_SHAPE
                fix_shape_fill(shp, slide_idx)
        except:
            pass

        # --- Recolour text ---
        if shp.has_text_frame:
            for para in shp.text_frame.paragraphs:
                para_text = "".join(r.text for r in para.runs)
                for run in para.runs:
                    # Strip emojis
                    run.text = strip_emojis(run.text)
                    # Pick colour
                    col = choose_text_color(run.text, para_text, shp.name, slide_idx)
                    recolour_run(run, col)

    # --- Special per-slide fixes ---

    # Slide 1 (Title): Make main title navy, subtitle mgray
    if slide_idx == 1:
        for shp in slide.shapes:
            if not shp.has_text_frame: continue
            for para in shp.text_frame.paragraphs:
                full = "".join(r.text for r in para.runs)
                for run in para.runs:
                    if 'IBM INTERNSHIP' in full or 'RESEARCH PROJECT' in full:
                        run.font.color.rgb = C_BLUE
                        run.font.size = Pt(11)
                    elif 'Autonomous Vehicle' in full or 'Object Detection System' in full:
                        run.font.color.rgb = C_NAVY
                        try:
                            shp.text_frame.paragraphs[0].font.bold = True
                        except: pass
                    elif 'Unified Real-Time' in full:
                        run.font.color.rgb = C_MGRAY
                    elif 'Powered by' in full:
                        run.font.color.rgb = C_MGRAY
                    elif 'Datasets:' in full or 'Training:' in full:
                        run.font.color.rgb = C_LGRAY

    # Slide 2 (Agenda): section numbers → blue
    if slide_idx == 2:
        for shp in slide.shapes:
            if not shp.has_text_frame: continue
            for para in shp.text_frame.paragraphs:
                full = "".join(r.text for r in para.runs)
                for run in para.runs:
                    if re.match(r'^0[1-9]$', full.strip()):
                        run.font.color.rgb = C_BLUE
                        run.font.bold = True
                    elif full.strip() in ('Agenda',):
                        run.font.color.rgb = C_NAVY
                        run.font.size = Pt(28)
                        run.font.bold = True
                    elif full.strip() == 'OVERVIEW':
                        run.font.color.rgb = C_BLUE
                        run.font.size = Pt(11)
                        run.font.bold = True

    # Slides 5 & 6 (Lit Review tables): header bg change
    if slide_idx in (5, 6):
        for shp in slide.shapes:
            if not shp.has_text_frame: continue
            for para in shp.text_frame.paragraphs:
                full = "".join(r.text for r in para.runs).strip()
                for run in para.runs:
                    if full in ('#', 'Authors / Title', 'Venue', 'Model', 'Results', 'Limitation'):
                        run.font.color.rgb = C_WHITE
                        run.font.bold = True

    # Slide 8 (Pipeline): arrow text → navy
    if slide_idx == 8:
        for shp in slide.shapes:
            if not shp.has_text_frame: continue
            for para in shp.text_frame.paragraphs:
                full = "".join(r.text for r in para.runs).strip()
                for run in para.runs:
                    run.text = strip_emojis(run.text)
                    if full in ('→', ''):
                        run.font.color.rgb = C_MGRAY
                    elif full in ('OUTPUT  —  Single annotated frame: Bounding Boxes  ·  Track IDs  ·  Lane Overlay',
                                  'Export: ONNX  ·  TensorRT (3–5× faster)  ·  CoreML  ·  OpenVINO'):
                        run.font.color.rgb = C_NAVY
                    else:
                        run.font.color.rgb = C_DGRAY

    # Slide 11 (Training): code text → navy monospace
    if slide_idx == 11:
        for shp in slide.shapes:
            if not shp.has_text_frame: continue
            for para in shp.text_frame.paragraphs:
                full = "".join(r.text for r in para.runs).strip()
                for run in para.runs:
                    if full.startswith('#SBATCH') or full.startswith('torchrun') or \
                       full.startswith('--') or full.startswith('training/'):
                        run.font.color.rgb = C_NAVY
                        run.font.name = 'Consolas'
                        run.font.size = Pt(12)

    # Slide 12 (Contributions): checkmark → blue
    if slide_idx == 12:
        for shp in slide.shapes:
            if not shp.has_text_frame: continue
            for para in shp.text_frame.paragraphs:
                full = "".join(r.text for r in para.runs).strip()
                for run in para.runs:
                    stripped = strip_emojis(run.text).strip()
                    if not stripped:
                        run.text = ''

    # Slide 14 (Thank You): big title navy
    if slide_idx == 14:
        for shp in slide.shapes:
            if not shp.has_text_frame: continue
            for para in shp.text_frame.paragraphs:
                full = "".join(r.text for r in para.runs).strip()
                for run in para.runs:
                    if 'Thank You' in full:
                        run.font.color.rgb = C_NAVY
                        run.font.size = Pt(48)
                        run.font.bold = True
                    elif 'Autonomous Vehicle' in full and 'Object Detection' in full:
                        run.font.color.rgb = C_BLUE
                        run.font.bold = True
                    elif re.search(r'BEVFormer|BEVFusion|CLRNet|YOLOv|RT-DETR|Sparse4D|ByteTrack|PointPillars|VoxelNet|DETR3D|Grounding', full):
                        run.font.color.rgb = C_MGRAY
                        run.font.size = Pt(11)
                    elif 'Datasets:' in full or 'Training' in full:
                        run.font.color.rgb = C_LGRAY
                        run.font.size = Pt(11)

# --- Fix table header rows on slides 5 & 6: make header row navy bg ---
# We need to colour the row rectangles that serve as header backgrounds
for i, slide in enumerate(prs.slides):
    slide_idx = i + 1
    if slide_idx not in (5, 6): continue
    for shp in slide.shapes:
        if shp.shape_type == 1:
            try:
                # Header row rectangle: y near 1.65", h ~ 0.38"
                y_in = shp.top / EMU
                h_in = shp.height / EMU
                w_in = shp.width / EMU
                if abs(y_in - 1.65) < 0.1 and abs(h_in - 0.38) < 0.05 and w_in > 10:
                    shp.fill.solid()
                    shp.fill.fore_color.rgb = C_NAVY
                    try: shp.line.color.rgb = C_NAVY
                    except: pass
            except: pass

prs.save(DST)
print(f"Saved: {DST}")
