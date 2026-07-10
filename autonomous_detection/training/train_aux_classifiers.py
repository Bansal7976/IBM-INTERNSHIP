"""Train the two auxiliary classifiers (Jobs C + D). Each trains in <1h on 1 GPU.

  1. Traffic light state (red/yellow/green/off)  <- LISA crops
  2. Lane marking type (solid/dashed/double)     <- BDD100K lane crops

Usage:
    # Build crop datasets first:
    python training/train_aux_classifiers.py prepare-lisa --lisa data/lisa_tl
    python training/train_aux_classifiers.py prepare-lanes --bdd data/bdd100k

    # Then train:
    python training/train_aux_classifiers.py train-tl
    python training/train_aux_classifiers.py train-lane
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

TL_CLASSES = ["red", "yellow", "green", "off"]
LANE_CLASSES = ["solid", "dashed", "double_solid", "double_yellow"]

# LISA annotation tag -> our class
LISA_TAG_MAP = {
    "stop": "red", "stopLeft": "red",
    "warning": "yellow", "warningLeft": "yellow",
    "go": "green", "goLeft": "green", "goForward": "green",
}


# --------------------------------------------------------------------------
# Dataset preparation
# --------------------------------------------------------------------------

def prepare_lisa(lisa_root: str, out_dir: str = "data/tl_crops"):
    """Extract traffic-light crops + state labels from LISA CSV annotations."""
    root, out = Path(lisa_root), Path(out_dir)
    for c in TL_CLASSES:
        (out / c).mkdir(parents=True, exist_ok=True)

    n = 0
    for csv_path in root.rglob("frameAnnotationsBOX.csv"):
        day_dir = csv_path.parent
        with open(csv_path) as f:
            for row in csv.DictReader(f, delimiter=";"):
                cls = LISA_TAG_MAP.get(row.get("Annotation tag", ""))
                if cls is None:
                    continue
                img_rel = Path(row["Filename"]).name
                # frames live in <clip>/frames/ next to the csv
                candidates = list(day_dir.rglob(img_rel))
                if not candidates:
                    continue
                img = cv2.imread(str(candidates[0]))
                if img is None:
                    continue
                x1 = int(row["Upper left corner X"]); y1 = int(row["Upper left corner Y"])
                x2 = int(row["Lower right corner X"]); y2 = int(row["Lower right corner Y"])
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0 or min(crop.shape[:2]) < 8:
                    continue
                cv2.imwrite(str(out / cls / f"{n:06d}.jpg"), crop)
                n += 1
    print(f"LISA crops written: {n} -> {out}")


def prepare_lane_crops(bdd_root: str, out_dir: str = "data/lane_crops",
                       patch: int = 32, per_image: int = 8):
    """Sample 32x32 patches along BDD100K lane polylines, labeled by lane style."""
    root, out = Path(bdd_root), Path(out_dir)
    for c in LANE_CLASSES:
        (out / c).mkdir(parents=True, exist_ok=True)

    labels_json = root / "labels" / "lane" / "polygons" / "lane_train.json"
    if not labels_json.exists():
        # alternate BDD100K layout
        matches = list(root.rglob("lane_train.json")) or list(root.rglob("*lane*train*.json"))
        if not matches:
            raise SystemExit(f"Lane labels not found under {root} — download "
                             "'Lane Marking' labels from bdd-data.berkeley.edu")
        labels_json = matches[0]

    with open(labels_json) as f:
        frames = json.load(f)

    half, n = patch // 2, 0
    for frame in frames[:20000]:  # 20K images is plenty for a patch classifier
        img_path = root / "images" / "100k" / "train" / frame["name"]
        if not img_path.exists():
            continue
        img = None
        for lab in frame.get("labels", []):
            attrs = lab.get("attributes", {})
            style = attrs.get("laneStyle", "")           # "solid" | "dashed"
            ltype = attrs.get("laneType", "")            # single/double white/yellow
            if style == "solid" and "double" in ltype:
                cls = "double_yellow" if "yellow" in ltype else "double_solid"
            elif style in ("solid", "dashed"):
                cls = style
            else:
                continue
            poly = lab.get("poly2d")
            if not poly:
                continue
            if img is None:
                img = cv2.imread(str(img_path))
                if img is None:
                    break
            pts = np.asarray(poly[0]["vertices"])
            idx = np.linspace(0, len(pts) - 1, per_image).astype(int)
            for x, y in pts[idx]:
                x, y = int(x), int(y)
                if half <= x < img.shape[1] - half and half <= y < img.shape[0] - half:
                    crop = img[y - half:y + half, x - half:x + half]
                    cv2.imwrite(str(out / cls / f"{n:07d}.jpg"), crop)
                    n += 1
    print(f"Lane patches written: {n} -> {out}")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

class CropFolderDataset(Dataset):
    def __init__(self, root: Path, classes: list[str], size: int, train: bool):
        self.samples = []
        for i, c in enumerate(classes):
            files = sorted((root / c).glob("*.jpg"))
            cut = int(len(files) * 0.9)
            files = files[:cut] if train else files[cut:]
            self.samples += [(f, i) for f in files]
        aug = [transforms.ColorJitter(0.3, 0.3, 0.3),
               transforms.RandomHorizontalFlip()] if train else []
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=True),
            *aug,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return self.tf(img), label


def train_classifier(crop_dir: str, classes: list[str], out_name: str,
                     size: int = 64, epochs: int = 15, batch: int = 256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(crop_dir)
    train_ds = CropFolderDataset(root, classes, size, train=True)
    val_ds = CropFolderDataset(root, classes, size, train=False)
    print(f"Train {len(train_ds)} / Val {len(val_ds)} crops, classes={classes}")

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=batch, num_workers=4)

    net = resnet18(weights="IMAGENET1K_V1")
    net.fc = nn.Linear(512, len(classes))
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    WEIGHTS_DIR.mkdir(exist_ok=True)
    for ep in range(epochs):
        net.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(net(x), y).backward()
            opt.step()
        sched.step()

        net.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = net(x.to(device)).argmax(1).cpu()
                correct += int((pred == y).sum())
                total += len(y)
        acc = correct / max(total, 1)
        print(f"epoch {ep + 1}/{epochs}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), WEIGHTS_DIR / out_name)

    print(f"Best val accuracy: {best_acc:.4f} -> weights/{out_name}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-lisa"); p.add_argument("--lisa", default="data/lisa_tl")
    p = sub.add_parser("prepare-lanes"); p.add_argument("--bdd", default="data/bdd100k")
    sub.add_parser("train-tl")
    sub.add_parser("train-lane")
    args = ap.parse_args()

    if args.cmd == "prepare-lisa":
        prepare_lisa(args.lisa)
    elif args.cmd == "prepare-lanes":
        prepare_lane_crops(args.bdd)
    elif args.cmd == "train-tl":
        train_classifier("data/tl_crops", TL_CLASSES, "tl_state.pt")
    elif args.cmd == "train-lane":
        train_classifier("data/lane_crops", LANE_CLASSES, "lane_type.pt")


if __name__ == "__main__":
    main()
