const pptxgen = require("pptxgenjs");

// ── Palette: BLACK & WHITE ONLY ───────────────────────────────────────────
const BLACK  = "111111";   // titles, strong text
const DARK   = "374151";   // body text
const MID    = "6B7280";   // sub-labels, page numbers
const RULE   = "D1D5DB";   // thin divider lines
const WHITE  = "FFFFFF";   // backgrounds

const HF = "Calibri";  // single font throughout

// ── Layout ────────────────────────────────────────────────────────────────
const W = 13.333, H = 7.5;
const ML = 0.65, MR = 0.65, CW = W - ML - MR;
const TY  = 0.45;   // title top
const TH  = 0.72;   // title box height
const RY  = TY + TH + 0.06;   // rule y
const CY  = RY + 0.28;        // content start y

let pres;

// ── Helpers ───────────────────────────────────────────────────────────────
function addTitle(s, text, sub) {
  s.addText(text, {
    x: ML, y: TY, w: CW, h: TH,
    fontFace: HF, fontSize: 28, bold: true, color: BLACK,
    align: "left", valign: "bottom", margin: 0
  });
  s.addShape(pres.shapes.LINE, {
    x: ML, y: RY, w: CW, h: 0, line: { color: BLACK, width: 1 }
  });
  if (sub) {
    s.addText(sub, {
      x: ML, y: RY + 0.06, w: CW, h: 0.34,
      fontFace: HF, fontSize: 12, color: MID, margin: 0
    });
  }
}

function pn(s, n) {
  s.addText(`${n} / 8`, {
    x: W - MR - 0.6, y: H - 0.4, w: 0.6, h: 0.28,
    fontFace: HF, fontSize: 10, color: MID, align: "right", margin: 0
  });
}

function row(s, x, y, w, label, value, isLast) {
  s.addText(label, {
    x, y: y + 0.04, w: 1.5, h: 0.38,
    fontFace: HF, fontSize: 12, bold: true, color: BLACK, margin: 0, valign: "middle"
  });
  s.addText(value, {
    x: x + 1.6, y: y + 0.04, w: w - 1.6, h: 0.38,
    fontFace: HF, fontSize: 12, color: DARK, margin: 0, valign: "middle"
  });
  if (!isLast) {
    s.addShape(pres.shapes.LINE, {
      x, y: y + 0.44, w, h: 0, line: { color: RULE, width: 0.5 }
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────
async function main() {
  pres = new pptxgen();
  pres.defineLayout({ name: "WIDE", width: W, height: H });
  pres.layout = "WIDE";
  pres.author  = "Vishal Bansal & Krish Manwani";
  pres.title   = "Autonomous Vehicle & Object Detection System";

  // ── SLIDE 1 — TITLE ────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };

    s.addShape(pres.shapes.LINE, {
      x: ML, y: 1.6, w: 0, h: 3.2, line: { color: BLACK, width: 1.5 }
    });

    s.addText("IBM Internship  ·  Research Project  ·  2024", {
      x: ML + 0.26, y: 1.6, w: CW - 0.26, h: 0.4,
      fontFace: HF, fontSize: 12, color: MID, margin: 0
    });
    s.addText("Autonomous Vehicle &\nObject Detection System", {
      x: ML + 0.26, y: 2.1, w: 10, h: 1.85,
      fontFace: HF, fontSize: 40, bold: true, color: BLACK,
      margin: 0, lineSpacingMultiple: 1.05
    });
    s.addText(
      "A unified real-time perception pipeline: Object Detection · 3D BEV · Lane Detection · Depth Estimation",
      { x: ML + 0.26, y: 4.08, w: 9.5, h: 0.52,
        fontFace: HF, fontSize: 14, color: DARK, margin: 0 }
    );
    s.addText("Powered by  YOLOv11 · BEVFormer · BEVFusion · Ultra-Fast Lane · Depth Anything V2", {
      x: ML + 0.26, y: 4.72, w: 9.5, h: 0.38,
      fontFace: HF, fontSize: 12, color: MID, margin: 0
    });

    s.addShape(pres.shapes.LINE, {
      x: ML + 0.26, y: 5.5, w: 5, h: 0, line: { color: RULE, width: 0.8 }
    });
    s.addText("Vishal Bansal  |  Krish Manwani", {
      x: ML + 0.26, y: 5.68, w: 8, h: 0.4,
      fontFace: HF, fontSize: 15, bold: true, color: BLACK, margin: 0
    });
    s.addText("Datasets: KITTI · nuScenes    Training: HPC Cluster · SLURM · PyTorch DDP", {
      x: ML + 0.26, y: 6.12, w: 10, h: 0.34,
      fontFace: HF, fontSize: 11, color: MID, margin: 0
    });
  }

  // ── SLIDE 2 — PROBLEM STATEMENT & GAPS ─────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    addTitle(s, "Problem Statement & Research Gaps");

    const colW = (CW - 0.4) / 2;
    const lx = ML, rx = ML + colW + 0.4;

    // Left: The Challenge
    s.addText("The Challenge", {
      x: lx, y: CY + 0.1, w: colW, h: 0.38,
      fontFace: HF, fontSize: 14, bold: true, color: BLACK, margin: 0
    });
    const challenges = [
      "Detect vehicles, pedestrians, cyclists in real-time across all conditions",
      "Identify road lanes to maintain lane discipline",
      "Estimate distance to obstacles from a single camera (monocular depth)",
      "Predict collision risk before it occurs",
      "All tasks must run simultaneously in one system",
    ];
    challenges.forEach((c, i) => {
      const y = CY + 0.6 + i * 0.76;
      s.addText(`${i + 1}`, {
        x: lx, y: y + 0.04, w: 0.3, h: 0.34,
        fontFace: HF, fontSize: 13, bold: true, color: BLACK, margin: 0
      });
      s.addText(c, {
        x: lx + 0.36, y: y + 0.04, w: colW - 0.36, h: 0.6,
        fontFace: HF, fontSize: 12.5, color: DARK, margin: 0, lineSpacingMultiple: 1.2
      });
      if (i < challenges.length - 1) {
        s.addShape(pres.shapes.LINE, {
          x: lx, y: y + 0.68, w: colW, h: 0, line: { color: RULE, width: 0.5 }
        });
      }
    });

    // vertical divider
    s.addShape(pres.shapes.LINE, {
      x: lx + colW + 0.2, y: CY, w: 0, h: 5.5, line: { color: RULE, width: 0.8 }
    });

    // Right: Research Gaps
    s.addText("Gaps in Existing Literature", {
      x: rx, y: CY + 0.1, w: colW, h: 0.38,
      fontFace: HF, fontSize: 14, bold: true, color: BLACK, margin: 0
    });
    const gaps = [
      ["G1", "No unified pipeline — all 15 papers solve tasks in isolation"],
      ["G2", "16-point NDS gap: camera-only (56.9) vs LiDAR+camera (72.9)"],
      ["G3", "Depth + tracking never combined for Time-to-Collision estimation"],
      ["G4", "Grounding DINO never benchmarked on driving datasets"],
      ["G5", "No joint ADAS evaluation metric across tasks"],
    ];
    gaps.forEach((g, i) => {
      const y = CY + 0.6 + i * 0.88;
      s.addText(g[0], {
        x: rx, y: y + 0.04, w: 0.36, h: 0.34,
        fontFace: HF, fontSize: 13, bold: true, color: BLACK, margin: 0
      });
      s.addText(g[1], {
        x: rx + 0.42, y: y + 0.04, w: colW - 0.42, h: 0.72,
        fontFace: HF, fontSize: 12.5, color: DARK, margin: 0, lineSpacingMultiple: 1.2
      });
      if (i < gaps.length - 1) {
        s.addShape(pres.shapes.LINE, {
          x: rx, y: y + 0.8, w: colW, h: 0, line: { color: RULE, width: 0.5 }
        });
      }
    });
    pn(s, 2);
  }

  // ── SLIDE 3 — LITERATURE REVIEW ─────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    addTitle(s, "Literature Review", "15 papers  ·  CVPR · ECCV · ICCV · NeurIPS · ICRA · IEEE TPAMI · CoRL  (2018–2024)");

    // Table header
    const hY = CY + 0.22;
    const cols = [
      { label: "#",            x: ML,         w: 0.42 },
      { label: "Authors / Paper", x: ML+0.46,  w: 3.6  },
      { label: "Venue",        x: ML+4.1,     w: 1.55 },
      { label: "Model",        x: ML+5.7,     w: 1.7  },
      { label: "Key Result",   x: ML+7.46,    w: 2.65 },
      { label: "Limitation",   x: ML+10.16,   w: 2.52 },
    ];

    // header bg
    s.addShape(pres.shapes.RECTANGLE, {
      x: ML, y: hY, w: CW, h: 0.36, fill: { color: BLACK }
    });
    cols.forEach(c => {
      s.addText(c.label, {
        x: c.x + 0.06, y: hY, w: c.w, h: 0.36,
        fontFace: HF, fontSize: 11.5, bold: true, color: WHITE,
        valign: "middle", margin: 0
      });
    });

    const papers = [
      ["01","Li et al.","BEVFormer","ECCV 2022","56.9 NDS · 46.0 mAP (nuScenes)","Camera-only; low-light degradation"],
      ["02","Liu et al.","BEVFusion","ICRA 2023","72.9 NDS · 70.2 mAP (nuScenes)","Needs LiDAR + calibration"],
      ["03","Zhang et al.","ByteTrack","ECCV 2022","80.3 MOTA · 77.3 IDF1 (MOT17)","No ReID; dense crowd issues"],
      ["04","Lin et al.","Sparse4D v3","arXiv 2023","56.1 NDS · 49.0 AMOTA","High training cost"],
      ["05","Wang et al.","StreamPETR","ICCV 2023","67.6 NDS · 65.3 AMOTA","Latency from temporal dep."],
      ["06","Zhao et al.","RT-DETR","CVPR 2024","53.1 AP @ 108 FPS (COCO)","Higher memory than YOLO"],
      ["07","Wang et al.","YOLOv10","NeurIPS 2024","46.3 AP · 25% fewer params","NMS-free optimization complex"],
      ["08","Ultralytics","YOLOv11","2024","54.7 mAP(x) · 22% fewer params","No peer-reviewed publication"],
      ["09","Zheng et al.","CLRNet","CVPR 2022","80.47 F1 (CULane)","Struggles with extreme curves"],
      ["10","Qin et al.","UFLDv2","TPAMI 2022","96.4 Acc · 300+ FPS (TuSimple)","Fixed grid; extreme angles"],
      ["11","Yang et al.","Depth Anything V2","NeurIPS 2024","SOTA NYUv2 & KITTI · 10× faster","Night / tunnel drops"],
      ["12","Lang et al.","PointPillars","CVPR 2019","82.58 AP Car @ 62 Hz (KITTI)","Loses vertical spatial info"],
      ["13","Zhou & Tuzel","VoxelNet","CVPR 2018","SOTA KITTI (2018)","Memory-intensive; not real-time"],
      ["14","Wang et al.","DETR3D","CoRL 2022","42.5 NDS · 34.6 mAP","No temporal modeling"],
      ["15","Liu et al.","Grounding DINO","ECCV 2024","52.5 AP zero-shot (COCO)","Never tested on driving data"],
    ];

    const rowH = 0.34;
    papers.forEach((p, i) => {
      const ry = hY + 0.36 + i * rowH;
      const fill = i % 2 === 1 ? "F9FAFB" : WHITE;
      s.addShape(pres.shapes.RECTANGLE, {
        x: ML, y: ry, w: CW, h: rowH, fill: { color: fill }, line: { color: RULE, width: 0.3 }
      });
      const vals = p;
      [[cols[0], vals[0]], [cols[1], `${vals[1]} · ${vals[2]}`],
       [cols[2], vals[3]], [cols[3], vals[2]],
       [cols[4], vals[4]], [cols[5], vals[5]]].forEach(([c, v], ci) => {
        s.addText(v, {
          x: c.x + 0.06, y: ry, w: c.w, h: rowH,
          fontFace: HF, fontSize: ci === 0 ? 11 : 10.5,
          color: ci === 0 ? BLACK : DARK,
          bold: ci === 0, valign: "middle", margin: 0
        });
      });
    });
    pn(s, 3);
  }

  // ── SLIDE 4 — PROPOSED ARCHITECTURE ─────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    addTitle(s, "Proposed Architecture", "Unified real-time perception pipeline — camera-first, HPC-trained");

    // Pipeline boxes (horizontal)
    const boxes = [
      ["INPUT\nCamera / Video", ""],
      ["YOLOv11", "Object Detection\n2D Boxes + Classes"],
      ["Ultra-Fast\nLane v2", "Lane Detection\nF1 > 96%"],
      ["Depth\nAnything V2", "Monocular Depth\nMetric Distance"],
      ["ByteTrack", "Multi-Object\nTracking + Velocity"],
      ["TTC\nCalculator", "Collision Risk\nTime-to-Collision"],
    ];

    const bw = (CW - 5 * 0.18) / 6, bh = 1.5;
    const sy = CY + 0.4;

    boxes.forEach((b, i) => {
      const bx = ML + i * (bw + 0.18);
      // box outline only - no fill color, just black border
      s.addShape(pres.shapes.RECTANGLE, {
        x: bx, y: sy, w: bw, h: bh,
        fill: { color: i === 0 || i === 5 ? "F3F4F6" : WHITE },
        line: { color: BLACK, width: 0.8 }
      });
      s.addText(b[0], {
        x: bx, y: sy + 0.12, w: bw, h: 0.72,
        fontFace: HF, fontSize: 12, bold: true, color: BLACK,
        align: "center", valign: "middle", margin: 0, lineSpacingMultiple: 1.1
      });
      if (b[1]) {
        s.addShape(pres.shapes.LINE, {
          x: bx + 0.15, y: sy + 0.88, w: bw - 0.3, h: 0,
          line: { color: RULE, width: 0.5 }
        });
        s.addText(b[1], {
          x: bx + 0.06, y: sy + 0.96, w: bw - 0.12, h: 0.48,
          fontFace: HF, fontSize: 10.5, color: DARK,
          align: "center", valign: "middle", margin: 0, lineSpacingMultiple: 1.1
        });
      }
      if (i < boxes.length - 1) {
        s.addShape(pres.shapes.LINE, {
          x: bx + bw, y: sy + bh / 2, w: 0.18, h: 0,
          line: { color: BLACK, width: 1 }
        });
      }
    });

    // Output bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: ML, y: sy + bh + 0.35, w: CW, h: 0.62,
      fill: { color: "F3F4F6" }, line: { color: BLACK, width: 0.6 }
    });
    s.addText(
      "OUTPUT  —  Annotated frame: Bounding Boxes · Track IDs · Lane Overlay · Depth Map · TTC Value\n" +
      "Export: ONNX  ·  TensorRT (3–5× faster)  ·  CoreML  ·  OpenVINO",
      { x: ML + 0.2, y: sy + bh + 0.36, w: CW - 0.4, h: 0.6,
        fontFace: HF, fontSize: 12, color: DARK, valign: "middle", margin: 0, lineSpacingMultiple: 1.25 }
    );

    // Note on 3D branch
    s.addText(
      "Parallel 3D Branch: BEVFormer (camera-only, NDS 56.9) and BEVFusion (LiDAR+camera, NDS 72.9) trained on HPC cluster via SLURM",
      { x: ML, y: sy + bh + 1.2, w: CW, h: 0.44,
        fontFace: HF, fontSize: 12, color: MID, margin: 0 }
    );
    pn(s, 4);
  }

  // ── SLIDE 5 — TECH STACK & DATASETS ─────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    addTitle(s, "Tech Stack & Datasets");

    const lw = CW * 0.55, rw = CW * 0.42, rx = ML + lw + 0.12;

    // Left: Tech stack in clean rows
    const sections = [
      ["Detection", "YOLOv11  ·  YOLOv10  ·  RT-DETR  ·  Grounding DINO"],
      ["3D / BEV", "BEVFormer  ·  BEVFusion  ·  Sparse4D v3  ·  StreamPETR"],
      ["Lane & Depth", "Ultra-Fast Lane v2  ·  CLRNet  ·  Depth Anything V2"],
      ["Tracking", "ByteTrack  ·  BotSort  ·  Simple IoU Tracker"],
      ["Frameworks", "PyTorch  ·  MMDetection3D  ·  Ultralytics  ·  Albumentations"],
      ["Deploy", "ONNX  ·  TensorRT  ·  CoreML  ·  OpenVINO  ·  TensorBoard"],
    ];
    s.addText("Technology Stack", {
      x: ML, y: CY + 0.12, w: lw, h: 0.38,
      fontFace: HF, fontSize: 14, bold: true, color: BLACK, margin: 0
    });
    sections.forEach((sec, i) => {
      const y = CY + 0.62 + i * 0.74;
      s.addText(sec[0], {
        x: ML, y: y + 0.04, w: 1.55, h: 0.36,
        fontFace: HF, fontSize: 12, bold: true, color: BLACK, valign: "middle", margin: 0
      });
      s.addText(sec[1], {
        x: ML + 1.65, y: y + 0.04, w: lw - 1.65, h: 0.36,
        fontFace: HF, fontSize: 12, color: DARK, valign: "middle", margin: 0
      });
      if (i < sections.length - 1) {
        s.addShape(pres.shapes.LINE, {
          x: ML, y: y + 0.66, w: lw, h: 0, line: { color: RULE, width: 0.5 }
        });
      }
    });

    // Divider
    s.addShape(pres.shapes.LINE, {
      x: rx - 0.06, y: CY, w: 0, h: 5.5, line: { color: RULE, width: 0.8 }
    });

    // Right: Datasets
    s.addText("Datasets", {
      x: rx, y: CY + 0.12, w: rw, h: 0.38,
      fontFace: HF, fontSize: 14, bold: true, color: BLACK, margin: 0
    });
    const datasets = [
      ["KITTI", "2D + 3D detection", "7,481 frames · 8 classes · stereo + LiDAR"],
      ["nuScenes", "3D BEV detection", "28k frames · 10 classes · 6 cams + LiDAR"],
      ["TuSimple", "Lane detection", "3,626 train clips · 96.4% UFLD accuracy"],
      ["CULane", "Lane detection", "88k train frames · urban & highway scenes"],
    ];
    datasets.forEach((d, i) => {
      const y = CY + 0.62 + i * 1.18;
      s.addText(d[0], {
        x: rx, y: y + 0.04, w: rw, h: 0.36,
        fontFace: HF, fontSize: 13, bold: true, color: BLACK, margin: 0
      });
      s.addText(d[1], {
        x: rx, y: y + 0.38, w: rw, h: 0.3,
        fontFace: HF, fontSize: 11, color: MID, margin: 0
      });
      s.addText(d[2], {
        x: rx, y: y + 0.65, w: rw, h: 0.38,
        fontFace: HF, fontSize: 12, color: DARK, margin: 0
      });
      if (i < datasets.length - 1) {
        s.addShape(pres.shapes.LINE, {
          x: rx, y: y + 1.1, w: rw, h: 0, line: { color: RULE, width: 0.5 }
        });
      }
    });
    pn(s, 5);
  }

  // ── SLIDE 6 — TRAINING STRATEGY + EXPECTED RESULTS ───────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    addTitle(s, "Training Strategy & Expected Results");

    const lw = CW * 0.52, rw = CW * 0.45, rx = ML + lw + 0.1;

    // Left: HPC Training
    s.addText("HPC Distributed Training  (SLURM + PyTorch DDP)", {
      x: ML, y: CY + 0.12, w: lw, h: 0.38,
      fontFace: HF, fontSize: 14, bold: true, color: BLACK, margin: 0
    });
    const training = [
      ["Nodes / GPUs", "4 nodes × 8 GPUs = 32 GPUs total"],
      ["Framework", "torchrun + NCCL all-reduce"],
      ["Effective batch", "batch_per_gpu × 32"],
      ["Precision", "Mixed precision AMP  (~2× speedup)"],
      ["LR scaling", "Linear warmup with batch size"],
      ["Monitoring", "TensorBoard  ·  WandB"],
    ];
    training.forEach((t, i) => {
      const y = CY + 0.65 + i * 0.65;
      s.addText(t[0], {
        x: ML, y: y + 0.04, w: 1.85, h: 0.36,
        fontFace: HF, fontSize: 12, bold: true, color: BLACK, valign: "middle", margin: 0
      });
      s.addText(t[1], {
        x: ML + 1.95, y: y + 0.04, w: lw - 1.95, h: 0.36,
        fontFace: HF, fontSize: 12, color: DARK, valign: "middle", margin: 0
      });
      if (i < training.length - 1) {
        s.addShape(pres.shapes.LINE, {
          x: ML, y: y + 0.58, w: lw, h: 0, line: { color: RULE, width: 0.5 }
        });
      }
    });

    // Divider
    s.addShape(pres.shapes.LINE, {
      x: rx - 0.04, y: CY, w: 0, h: 5.5, line: { color: RULE, width: 0.8 }
    });

    // Right: Expected Results
    s.addText("Performance Targets", {
      x: rx, y: CY + 0.12, w: rw, h: 0.38,
      fontFace: HF, fontSize: 14, bold: true, color: BLACK, margin: 0
    });
    const targets = [
      ["YOLOv11 (KITTI)", "mAP@0.5  ≥ 85%"],
      ["YOLOv11 (nuScenes)", "mAP@0.5  ≥ 60%"],
      ["BEVFormer", "NDS  ≥ 50%"],
      ["BEVFusion", "NDS  ≥ 70%"],
      ["UFLDv2 (TuSimple)", "Accuracy  ≥ 96%"],
      ["TTC System", "Latency  < 50 ms (RTX 3090)"],
    ];
    targets.forEach((t, i) => {
      const y = CY + 0.65 + i * 0.78;
      s.addText(t[0], {
        x: rx, y: y + 0.04, w: rw, h: 0.36,
        fontFace: HF, fontSize: 12, bold: true, color: BLACK, valign: "middle", margin: 0
      });
      s.addText(t[1], {
        x: rx, y: y + 0.38, w: rw, h: 0.3,
        fontFace: HF, fontSize: 13, color: DARK, margin: 0
      });
      if (i < targets.length - 1) {
        s.addShape(pres.shapes.LINE, {
          x: rx, y: y + 0.7, w: rw, h: 0, line: { color: RULE, width: 0.5 }
        });
      }
    });
    pn(s, 6);
  }

  // ── SLIDE 7 — CONTRIBUTIONS ──────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    addTitle(s, "Contributions", "What our work contributes to the field");

    const contribs = [
      ["Unified 5-Task Pipeline",
       "First open-source system combining object detection + lane detection + depth estimation + tracking + TTC in one inference graph."],
      ["Novel TTC Estimation  (Closes G3)",
       "First integration of ByteTrack per-object velocity with Depth Anything V2 metric depth for real-time Time-to-Collision prediction."],
      ["Open-Vocabulary on Driving Data  (Closes G4)",
       "First benchmark evaluation of Grounding DINO on KITTI and nuScenes — establishing a zero-shot baseline for autonomous driving."],
      ["Camera vs Fusion Study  (Closes G2)",
       "Controlled comparison of BEVFormer (camera-only) vs BEVFusion (LiDAR+camera) under identical HPC training conditions."],
      ["HPC-Ready Codebase  (Closes G1)",
       "Full pipeline with SLURM scripts, DDP training, ONNX/TensorRT export and MMDetection3D integration — open source."],
    ];

    contribs.forEach((c, i) => {
      const y = CY + 0.18 + i * 1.04;
      s.addText(c[0], {
        x: ML, y: y + 0.04, w: 4.2, h: 0.38,
        fontFace: HF, fontSize: 13.5, bold: true, color: BLACK, valign: "middle", margin: 0
      });
      s.addText(c[1], {
        x: ML + 4.35, y: y + 0.04, w: CW - 4.35, h: 0.82,
        fontFace: HF, fontSize: 12.5, color: DARK, valign: "middle", margin: 0, lineSpacingMultiple: 1.2
      });
      if (i < contribs.length - 1) {
        s.addShape(pres.shapes.LINE, {
          x: ML, y: y + 0.94, w: CW, h: 0, line: { color: RULE, width: 0.5 }
        });
      }
    });
    pn(s, 7);
  }

  // ── SLIDE 8 — THANK YOU ──────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };

    s.addShape(pres.shapes.LINE, {
      x: ML, y: 1.8, w: 0, h: 3.0, line: { color: BLACK, width: 1.5 }
    });

    s.addText("Thank You", {
      x: ML + 0.26, y: 1.85, w: 9, h: 1.25,
      fontFace: HF, fontSize: 48, bold: true, color: BLACK, margin: 0
    });
    s.addText("Questions & Discussion", {
      x: ML + 0.26, y: 3.2, w: 9, h: 0.52,
      fontFace: HF, fontSize: 20, color: DARK, bold: false, margin: 0
    });
    s.addShape(pres.shapes.LINE, {
      x: ML + 0.26, y: 4.0, w: 5.0, h: 0, line: { color: RULE, width: 0.8 }
    });
    s.addText("Vishal Bansal  |  Krish Manwani", {
      x: ML + 0.26, y: 4.2, w: 8, h: 0.44,
      fontFace: HF, fontSize: 16, bold: true, color: BLACK, margin: 0
    });
    s.addText("IBM Internship Project  ·  2024", {
      x: ML + 0.26, y: 4.65, w: 8, h: 0.34,
      fontFace: HF, fontSize: 13, color: MID, margin: 0
    });
    s.addText(
      "Papers: BEVFormer (ECCV'22) · BEVFusion (ICRA'23) · Sparse4D v3 · StreamPETR (ICCV'23) · RT-DETR (CVPR'24) · " +
      "YOLOv10/11 · CLRNet (CVPR'22) · UFLDv2 (TPAMI'22) · Depth Anything V2 (NeurIPS'24) · " +
      "PointPillars (CVPR'19) · VoxelNet (CVPR'18) · DETR3D (CoRL'22) · Grounding DINO (ECCV'24) · ByteTrack (ECCV'22)",
      { x: ML + 0.26, y: 5.5, w: CW - 0.26, h: 0.9,
        fontFace: HF, fontSize: 10, color: MID, margin: 0, lineSpacingMultiple: 1.3 }
    );
    s.addText("Datasets: KITTI · nuScenes · TuSimple · CULane", {
      x: ML + 0.26, y: H - 0.55, w: CW, h: 0.3,
      fontFace: HF, fontSize: 10, color: MID, margin: 0
    });
  }

  await pres.writeFile({ fileName: "Autonomous_Vehicle_Detection_Final.pptx" });
  console.log("done — 8 slides");
}

main().catch(e => { console.error(e); process.exit(1); });
