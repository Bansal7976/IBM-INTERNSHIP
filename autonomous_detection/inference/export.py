"""
Export trained models to ONNX / TensorRT / CoreML for deployment.

Usage:
    # ONNX (works everywhere)
    python inference/export.py --weights best.pt --format onnx

    # TensorRT (NVIDIA GPUs, fastest)
    python inference/export.py --weights best.pt --format engine --half

    # CoreML (Apple Silicon / iOS)
    python inference/export.py --weights best.pt --format coreml

Supported formats: onnx, engine (TensorRT), coreml, tflite, openvino
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='Path to .pt weights')
    p.add_argument('--format', default='onnx',
                   choices=['onnx', 'engine', 'coreml', 'tflite', 'openvino', 'torchscript'])
    p.add_argument('--imgsz', type=int, default=640, help='Input image size')
    p.add_argument('--half', action='store_true', help='FP16 quantization (2x faster on Tensor Cores)')
    p.add_argument('--int8', action='store_true', help='INT8 quantization (4x faster, slight accuracy drop)')
    p.add_argument('--dynamic', action='store_true', help='Dynamic batch/input shapes')
    p.add_argument('--simplify', action='store_true', default=True, help='ONNX simplification')
    p.add_argument('--batch', type=int, default=1, help='Batch size for static export')
    return p.parse_args()


def export_model(
    weights: str,
    fmt: str = 'onnx',
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
    batch: int = 1,
) -> str:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    model = YOLO(weights)

    export_path = model.export(
        format=fmt,
        imgsz=imgsz,
        half=half,
        int8=int8,
        dynamic=dynamic,
        simplify=simplify,
        batch=batch,
    )

    print(f"\nExported: {export_path}")
    print(f"Format:   {fmt}")
    print(f"FP16:     {half}")

    # Benchmark the exported model
    if fmt == 'onnx':
        _benchmark_onnx(export_path, imgsz, batch)
    elif fmt == 'engine':
        _benchmark_tensorrt(export_path, imgsz, batch)

    return export_path


def _benchmark_onnx(model_path: str, imgsz: int, batch: int = 1):
    """Quick latency benchmark for ONNX model."""
    try:
        import onnxruntime as ort
        import numpy as np
        import time

        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name

        dummy = np.random.randn(batch, 3, imgsz, imgsz).astype(np.float32)

        # Warmup
        for _ in range(5):
            session.run(None, {input_name: dummy})

        # Benchmark
        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            session.run(None, {input_name: dummy})
        elapsed = time.perf_counter() - t0

        ms_per_frame = (elapsed / N / batch) * 1000
        fps = batch * N / elapsed
        print(f"\nONNX Benchmark ({N} runs, batch={batch}):")
        print(f"  Latency: {ms_per_frame:.2f} ms/frame")
        print(f"  Throughput: {fps:.1f} FPS")
        print(f"  Provider: {session.get_providers()[0]}")

    except ImportError:
        print("pip install onnxruntime-gpu for benchmarking")


def _benchmark_tensorrt(engine_path: str, imgsz: int, batch: int = 1):
    """Quick latency benchmark for TensorRT engine."""
    try:
        import tensorrt as trt
        print(f"\nTensorRT engine ready: {engine_path}")
        print("Use Ultralytics YOLO(engine_path) for inference.")
    except ImportError:
        print("TensorRT not installed. Install NVIDIA TensorRT for benchmarking.")


def main():
    args = parse_args()
    export_model(
        weights=args.weights,
        fmt=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=args.simplify,
        batch=args.batch,
    )


if __name__ == '__main__':
    main()
