"""
onnx_utils.py - ONNX Runtime session and exporter helpers shared by the
forest pipeline and the DNN trainers.

ONNX Runtime sizes its default intra-op thread pool from the machine's
physical cores and pins every worker to a core. Inside a restricted cpuset
(batch slots, cgroups, taskset: e.g. 94 of 384 CPUs on the training node) the
pinning fails for cores outside the set and ORT prints one
"pthread_setaffinity_np failed ... Specify the number of threads explicitly
so the affinity is not set" error per worker (144 lines per session in the
v2 training logs). Sizing the pools explicitly disables the pinning, so every
session in the training code is created through make_session().

The DNN trainers export with the TorchScript-based exporter on purpose: it
targets the opset the CMSSW ONNX Runtime integration was validated with
(pixel_features.ONNX_OPSET). torch >= 2.9 switches the default to the
torch.export-based exporter (opset >= 18, requires onnxscript), so the choice
is pinned with dynamo=False and only that deprecation notice is silenced.
"""

import os
import warnings

import numpy as np


def available_cpus():
    """CPUs this process may run on (cpuset-aware)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def make_session(model, intra_threads=None):
    """ONNX Runtime CPU session with explicitly sized thread pools.

    intra_threads=None uses min(8, available CPUs); pass 1 for single-thread
    benchmarks (the configuration CMSSW runs inference with)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads or min(8, available_cpus())
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(model, sess_options=so, providers=["CPUExecutionProvider"])


_LEGACY_EXPORT_NOTICE = "You are using the legacy TorchScript-based ONNX export"


def export_torch_onnx(module, example, path, opset, **kwargs):
    """torch.onnx.export pinned to the TorchScript exporter (see module doc)."""
    import torch

    with warnings.catch_warnings():
        # the pinned exporter's own deprecation notices (the call-site one and
        # the ones raised inside torch.onnx while it runs)
        warnings.filterwarnings("ignore", message=_LEGACY_EXPORT_NOTICE, category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.onnx")
        torch.onnx.export(module, example, path, dynamo=False, opset_version=opset, **kwargs)


def verify_torch_onnx(module, path, input_dim, n=4096, tol=1e-5, seed=0):
    """Compare an exported DNN with its torch module on random inputs.
    Returns the max |diff|; raises if it exceeds tol."""
    import torch

    x = np.random.default_rng(seed).standard_normal((n, input_dim)).astype(np.float32)
    sess = make_session(path)
    out_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0].ravel()
    module.eval()
    with torch.no_grad():
        out_torch = module(torch.from_numpy(x)).numpy().ravel()
    max_diff = float(np.abs(out_onnx - out_torch).max())
    if max_diff >= tol:
        raise AssertionError(f"ONNX/torch mismatch for {path}: max |diff|={max_diff:.2e} >= {tol}")
    return max_diff
