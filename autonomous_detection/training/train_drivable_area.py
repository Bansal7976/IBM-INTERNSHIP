"""Train DrivableNet (models/drivable_area.py) — binary drivable-surface
segmentation. Fixes lane detection failing on roads with no usable painted
markings (common on Indian roads; see models/drivable_area.py docstring for
the full rationale and citations).

DATA FORMAT THIS SCRIPT EXPECTS
---------------------------------
    <data_root>/images/*.jpg              — RGB frames
    <data_root>/masks/<same_stem>.png     — single-channel masks:
                                             nonzero = drivable, 0 = not drivable

IDD Segmentation ships label PNGs with its OWN multi-class IDs (41 classes
across a 4-level hierarchy), not a ready-made binary drivable mask — so
there's a one-time `prepare` step to threshold IDD's labels into the binary
format above. That step needs IDD's own class-ID table to know which pixel
values count as "drivable" (road / drivable-fallback), which is why this
script does NOT hardcode a guessed ID number: get the authoritative
class->ID mapping for your exact IDD release from the AutoNUE devkit
(github.com/AutoNUE/public-code, `helpers/anue_labels.py`) — it ships
alongside the dataset download and is the source of truth, not this script.

Usage:
    # One-time: turn IDD's multi-class label PNGs into binary drivable masks.
    # Look up the correct ids for "road" / "drivable fallback" in
    # helpers/anue_labels.py from the AutoNUE devkit for YOUR IDD release
    # (id_level you choose determines which ids to pass here — this project
    # doesn't guess at that number so it can't silently mislabel your data).
    python training/train_drivable_area.py prepare \
        --idd_seg data/IDD_Segmentation \
        --drivable_ids 0,1 \
        --out data/drivable_binary

    # Train:
    python training/train_drivable_area.py train \
        --data data/drivable_binary --epochs 40 --batch 32
    # -> weights/drivable_area.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Running this file directly puts training/ on sys.path, not the project
# root, which breaks the `from models...` import below.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.drivable_area import DrivableNet  # noqa: E402  (needs sys.path above)


# --------------------------------------------------------------------------
# One-time: IDD multi-class labels -> binary drivable masks
# --------------------------------------------------------------------------

def prepare_binary_masks(idd_seg_root: Path, drivable_ids: set[int], out_root: Path):
    """Threshold IDD Segmentation's label PNGs into binary drivable masks.

    IDD Segmentation's usual layout (AutoNUE devkit convention):
        <idd_seg_root>/leftImg8bit/{train,val}/<seq>/<frame>_leftImg8bit.png
        <idd_seg_root>/gtFine/{train,val}/<seq>/<frame>_gtFine_labelids.png
    This walks whatever *_labelids.png / *_label.png files it can find so it
    tolerates minor layout differences between IDD release versions.
    """
    label_files = sorted(idd_seg_root.rglob("*label*.png"))
    if not label_files:
        raise SystemExit(f"No *label*.png files found under {idd_seg_root} — "
                          "check the IDD Segmentation extraction layout.")

    out_images = out_root / "images"
    out_masks = out_root / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    n = 0
    for label_path in tqdm(label_files, desc="IDD labels -> binary masks"):
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label is None:
            continue
        binary = np.isin(label, list(drivable_ids)).astype(np.uint8) * 255

        # Find the matching RGB frame: swap gtFine/*_gtFine_labelids -> leftImg8bit/*_leftImg8bit
        img_path = Path(str(label_path)
                         .replace("gtFine", "leftImg8bit")
                         .replace("_label.png", "_leftImg8bit.png")
                         .replace("_labelids.png", "_leftImg8bit.png"))
        if not img_path.exists():
            continue

        stem = f"idd_{n:06d}"
        cv2.imwrite(str(out_images / f"{stem}.jpg"), cv2.imread(str(img_path)))
        cv2.imwrite(str(out_masks / f"{stem}.png"), binary)
        n += 1

    print(f"Wrote {n} image/mask pairs -> {out_root}")


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class DrivableAreaDataset(Dataset):
    def __init__(self, root: Path, size: tuple = (512, 256), augment: bool = False):
        self.images = sorted((root / "images").glob("*.jpg"))
        self.masks_dir = root / "masks"
        self.size = size          # (w, h)
        self.augment = augment
        if not self.images:
            raise SystemExit(f"No images found under {root / 'images'}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            # Corrupted pair — return a black/empty sample rather than crash
            # a multi-hour training run over one bad file.
            w, h = self.size
            return (torch.zeros(3, h, w), torch.zeros(1, h, w))

        w, h = self.size
        img = cv2.resize(img, (w, h))
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()

        img_t = torch.from_numpy(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(data_root: Path, epochs: int, batch: int, lr: float, val_frac: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = DrivableAreaDataset(data_root, augment=True)
    n_val = max(1, int(len(full) * val_frac))
    n_train = len(full) - n_val
    train_set, val_set = torch.utils.data.random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                               num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=2)

    model = DrivableNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_iou = 0.0
    WEIGHTS_DIR.mkdir(exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            logits = nn.functional.interpolate(
                logits, size=masks.shape[2:], mode="bilinear", align_corners=False)
            loss = loss_fn(logits, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        sched.step()

        model.eval()
        inter, union = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                logits = nn.functional.interpolate(
                    logits, size=masks.shape[2:], mode="bilinear", align_corners=False)
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter += (pred * masks).sum().item()
                union += ((pred + masks) > 0).float().sum().item()
        iou = inter / max(union, 1e-6)

        print(f"epoch {epoch}: train_loss={train_loss / len(train_loader):.4f}  "
              f"val_iou={iou:.4f}")
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), WEIGHTS_DIR / "drivable_area.pth")
            print(f"  -> saved new best (IoU={iou:.4f})")

    print(f"\nDone. Best val IoU: {best_iou:.4f}")
    print(f"Weights: {WEIGHTS_DIR / 'drivable_area.pth'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="IDD labels -> binary drivable masks")
    p_prep.add_argument("--idd_seg", required=True, type=Path)
    p_prep.add_argument("--drivable_ids", required=True,
                        help="Comma-separated pixel IDs meaning 'drivable' in "
                             "YOUR IDD release — look these up in the AutoNUE "
                             "devkit's helpers/anue_labels.py, don't guess.")
    p_prep.add_argument("--out", required=True, type=Path)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True, type=Path)
    p_train.add_argument("--epochs", type=int, default=40)
    p_train.add_argument("--batch", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--val_frac", type=float, default=0.1)

    args = ap.parse_args()
    if args.cmd == "prepare":
        ids = {int(x) for x in args.drivable_ids.split(",")}
        prepare_binary_masks(args.idd_seg, ids, args.out)
    elif args.cmd == "train":
        train(args.data, args.epochs, args.batch, args.lr, args.val_frac)


if __name__ == "__main__":
    main()
