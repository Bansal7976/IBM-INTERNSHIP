"""
Distributed Data Parallel (DDP) training script.
Use this for multi-GPU training on HPC clusters.

Launch commands:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 training/train_ddp.py --config configs/yolov11/yolov11_kitti.yaml

    # Multi-node (see slurm/ for SLURM scripts)
    torchrun --nproc_per_node=8 --nnodes=4 \
             --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
             training/train_ddp.py --config configs/yolov11/yolov11_kitti.yaml

Key DDP concepts:
    - Each GPU runs a complete copy of the model
    - Forward + backward runs independently on each GPU
    - Gradients are synchronized (all-reduced) before optimizer step
    - Batch size is effective_batch = batch_per_gpu * num_gpus
    - Learning rate scaling: lr = base_lr * (effective_batch / 256)
"""

import os
import argparse
import yaml
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', default=None)
    return p.parse_args()


def setup_ddp():
    """Initialize the process group from environment variables set by torchrun."""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def cleanup_ddp():
    dist.destroy_process_group()


def scale_lr(base_lr: float, batch_size: int, world_size: int, base_batch: int = 256) -> float:
    """Linear LR scaling rule: lr = base_lr * (batch_size * world_size / base_batch)."""
    return base_lr * (batch_size * world_size) / base_batch


def train_ddp(args):
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # For Ultralytics, DDP is handled internally when device='0,1,2,3'
    # This script demonstrates manual DDP for custom PyTorch models
    # For YOLO, just pass device='0,1,2,3' to model.train()

    if cfg.get('framework', 'ultralytics') == 'ultralytics':
        if is_main:
            print("Ultralytics handles DDP internally.")
            print("Running with Ultralytics DDP...")
            from ultralytics import YOLO
            train_cfg = cfg['training']

            model = YOLO(cfg['model'].get('weights', 'yolo11m.pt'))
            model.train(
                data=cfg['dataset']['yaml'],
                epochs=train_cfg.get('epochs', 50),
                imgsz=cfg['dataset'].get('img_size', 640),
                batch=train_cfg.get('batch_size', 16),
                device=','.join(str(i) for i in range(world_size)),
                workers=train_cfg.get('workers', 8),
                amp=train_cfg.get('amp', True),
            )
        cleanup_ddp()
        return

    # ── Custom PyTorch DDP training loop ──────────────────────────────────────
    # Import your custom model here
    # from models.custom_model import MyModel

    device = torch.device(f'cuda:{local_rank}')

    # model = MyModel(cfg)
    # model = model.to(device)
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_dataset = _build_dataset(cfg, 'train')
    val_dataset   = _build_dataset(cfg, 'val')

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    batch_size = cfg['training'].get('batch_size', 16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=cfg['training'].get('workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )

    base_lr = cfg['training'].get('lr0', 0.01)
    effective_lr = scale_lr(base_lr, batch_size, world_size)
    if is_main:
        print(f"World size: {world_size}, effective batch: {batch_size * world_size}")
        print(f"Scaled LR: {base_lr} → {effective_lr:.6f}")

    # optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])
    # scaler = torch.cuda.amp.GradScaler(enabled=cfg['training'].get('amp', True))

    epochs = cfg['training'].get('epochs', 50)
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)   # ensure different shuffles per epoch

        # ── train one epoch ──────────────────────────────────────────────────
        # model.train()
        # for batch in train_loader:
        #     images = batch['images'].to(device, non_blocking=True)
        #     ...
        #     with torch.cuda.amp.autocast():
        #         loss = model(images, ...)
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        #     optimizer.zero_grad()

        if is_main and epoch % 5 == 0:
            # Save checkpoint only from rank 0
            pass  # torch.save(model.module.state_dict(), f'checkpoint_epoch{epoch}.pth')

    cleanup_ddp()


def _build_dataset(cfg: dict, split: str):
    """Build dataset from config."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.datasets import build_dataset
    return build_dataset(cfg['dataset'], split)


if __name__ == '__main__':
    args = parse_args()
    train_ddp(args)
