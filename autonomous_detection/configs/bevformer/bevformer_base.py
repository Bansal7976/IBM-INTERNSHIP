# BEVFormer-Base configuration for nuScenes 3D detection
# Framework: MMDetection3D
#
# This mirrors the official BEVFormer config structure.
# Source: github.com/fundamentalvision/BEVFormer/projects/configs/bevformer/bevformer_base.py
#
# Performance: 51.7 NDS, 41.6 mAP on nuScenes val
# Training: 8 GPUs × 24 epochs ≈ 36h on A100s
#
# IMPORTANT: Use the official BEVFormer repo config directly.
# This file is a reference / documentation copy.

# Key hyperparameters you may want to tune:
_base_ = [
    # These would reference official MMDet3D base configs
    # '../_base_/datasets/nus-3d.py',
    # '../_base_/default_runtime.py',
]

# ─── Key Architecture Parameters ─────────────────────────────────────────────

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]   # detection range (meters)
bev_h_ = 200        # BEV grid height (resolution: 51.2*2/200 = 0.512m per cell)
bev_w_ = 200        # BEV grid width
num_query = 900     # number of object queries

# Image backbone: ResNet-101 with DCN
img_backbone = dict(
    type='ResNet',
    depth=101,
    num_stages=4,
    out_indices=(1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=False),
    norm_eval=True,
    style='caffe',
    dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
    stage_with_dcn=(False, False, True, True),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='ckpts/resnet101_msra-6aea974b.pth'
    )
)

img_neck = dict(
    type='FPN',
    in_channels=[512, 1024, 2048],
    out_channels=256,
    start_level=0,
    add_extra_convs='on_output',
    num_outs=4,
    relu_before_extra_convs=True,
)

# BEVFormer transformer
transformer = dict(
    type='PerceptionTransformer',
    rotate_prev_bev=True,
    use_shift=True,
    use_can_bus=True,
    embed_dims=256,
    encoder=dict(
        type='BEVFormerEncoder',
        num_layers=6,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        transformerlayers=dict(
            type='BEVFormerLayer',
            attn_cfgs=[
                dict(
                    type='TemporalSelfAttention',   # temporal attention
                    embed_dims=256,
                    num_levels=1,
                ),
                dict(
                    type='SpatialCrossAttention',   # camera → BEV attention
                    pc_range=point_cloud_range,
                    deformable_attention=dict(
                        type='MSDeformableAttention3D',
                        embed_dims=256,
                        num_points=8,
                        num_levels=4,
                    ),
                    embed_dims=256,
                )
            ],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
        ),
    ),
    decoder=dict(
        type='DetectionTransformerDecoder',
        num_layers=6,
        return_intermediate=True,
        transformerlayers=dict(
            type='DetrTransformerDecoderLayer',
            attn_cfgs=[
                dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                dict(
                    type='CustomMSDeformableAttention',
                    embed_dims=256,
                    num_levels=1,
                ),
            ],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
        ),
    ),
)

# Training schedule
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),    # slower LR for pretrained backbone
        }
    ),
    weight_decay=0.01,
)

runner = dict(type='EpochBasedRunner', max_epochs=24)

# ─── Learning Concepts from BEVFormer ─────────────────────────────────────────
"""
BEVFormer Key Ideas (great for learning):

1. BEV (Bird's Eye View) Space:
   - 3D detection in top-down grid (200×200 cells covering 51.2m radius)
   - Avoids depth estimation errors of monocular 3D detection

2. Spatial Cross-Attention:
   - For each BEV grid cell, project reference points to each camera image
   - Apply deformable attention to gather features from image space
   - Effectively "lifts" 2D features into 3D/BEV space

3. Temporal Self-Attention:
   - Previous frame's BEV features are aligned using ego-motion
   - Current BEV queries attend to historical BEV (provides temporal context)
   - Critical for detecting occluded objects from multiple viewpoints

4. CAN Bus:
   - Vehicle speed + steering used as positional embedding
   - Helps with temporal alignment between frames

5. Why this beats monocular 3D detectors:
   - Uses all cameras simultaneously (360° coverage)
   - Temporal aggregation catches objects unseen in current frame
   - BEV space makes scale estimation more straightforward
"""
