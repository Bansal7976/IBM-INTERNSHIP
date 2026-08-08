"""mmcv 1.x → 2.x compatibility shim for CLRNet.

CLRNet was written against mmcv 1.x.  mmcv 2.x removed several top-level
symbols (jit, load, dump, parallel, runner).  This module patches them back
as no-ops or thin wrappers so CLRNet can be imported for inference without
downgrading mmcv.

Import this module BEFORE any clrnet.* import.
"""
import sys
import types

import torch
import mmcv as _mmcv


# ── 1. mmcv.jit ─────────────────────────────────────────────────────────────
# Was an optional torch.jit decorator in mmcv 1.x; removed in 2.x.
# For inference we just want a no-op decorator.
if not hasattr(_mmcv, "jit"):
    def _jit(func=None, **kw):
        return func if func is not None else (lambda f: f)
    _mmcv.jit = _jit

if not hasattr(_mmcv, "is_jit_tracing"):
    _mmcv.is_jit_tracing = lambda: False


# ── 2. mmcv.load / mmcv.dump ────────────────────────────────────────────────
# Used by CLRNet for JSON/YAML/pkl I/O.
if not hasattr(_mmcv, "load"):
    def _load(filename, **kw):
        from pathlib import Path
        import json, pickle
        p = Path(filename)
        if p.suffix == ".json":
            return json.loads(p.read_text())
        if p.suffix in (".pkl", ".pickle"):
            return pickle.loads(p.read_bytes())
        try:
            import yaml
            return yaml.safe_load(p.read_text())
        except ImportError:
            return {}
    _mmcv.load = _load

if not hasattr(_mmcv, "dump"):
    def _dump(obj, filename, **kw):
        from pathlib import Path
        import json, pickle
        p = Path(filename)
        if p.suffix == ".json":
            p.write_text(json.dumps(obj, indent=2))
        else:
            p.write_bytes(pickle.dumps(obj))
    _mmcv.dump = _dump


# ── 3. mmcv.runner ──────────────────────────────────────────────────────────
# load_checkpoint is the critical piece for loading CLRNet weights.
if not hasattr(_mmcv, "runner"):
    _runner = types.ModuleType("mmcv.runner")

    def _load_checkpoint(model, filename, map_location=None, strict=False, **kw):
        ckpt = torch.load(filename, map_location=map_location or "cpu")
        state = ckpt.get("state_dict", ckpt.get("net", ckpt))
        model.load_state_dict(state, strict=strict)
        return ckpt

    _runner.load_checkpoint = _load_checkpoint
    _runner.BaseRunner = object
    _runner.EpochBasedRunner = object
    _runner.build_optimizer = lambda cfg, model: None

    # fp16 decorators — CLRNet's heads/backbones decorate forward() with
    # @auto_fp16() / @force_fp32(). Next error after the jit fix on the HPC
    # cluster was exactly: "cannot import name 'auto_fp16' from 'mmcv.runner'".
    #
    # Identity decorators are the correct behavior here, not a shortcut: in
    # mmcv 1.x auto_fp16 checks `self.fp16_enabled` and, when it's unset or
    # False, calls the undecorated function directly. We run inference in
    # fp32 and never set fp16_enabled, so the 1.x decorator would have been a
    # passthrough anyway.
    def _fp16_decorator(*d_args, **d_kwargs):
        # supports @auto_fp16, @auto_fp16(), and @auto_fp16(apply_to=(...))
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return lambda func: func

    _runner.auto_fp16 = _fp16_decorator
    _runner.force_fp32 = _fp16_decorator
    _runner.wrap_fp16_model = lambda model: model
    _runner.get_dist_info = lambda: (0, 1)

    sys.modules["mmcv.runner"] = _runner
    _mmcv.runner = _runner


# ── 4. mmcv.parallel ────────────────────────────────────────────────────────
# MMDataParallel is used in CLRNet's main.py (training).  For inference via
# our wrapper we only need the class to exist so imports don't fail.
if not hasattr(_mmcv, "parallel"):
    import torch.nn as _nn
    _parallel = types.ModuleType("mmcv.parallel")

    class MMDataParallel(_nn.DataParallel):
        """Minimal MMDataParallel drop-in for CLRNet imports."""
        pass

    _parallel.MMDataParallel = MMDataParallel
    _parallel.scatter = lambda inputs, target_gpus, dim=0: (inputs, {})
    _parallel.collate = lambda batch, samples_per_gpu=1: batch
    sys.modules["mmcv.parallel"] = _parallel
    _mmcv.parallel = _parallel
