"""
One place to open onnxruntime sessions, so a small server can tune them with environment variables.

DM_ORT_THREADS=1   on a host with a fraction of a CPU (Render free: 0.1 CPU) stops onnxruntime from starting one
                   thread per visible core, which only competes for the same small CPU share.
DM_ORT_ARENA=0     turns off onnxruntime's memory pool, which otherwise keeps the peak allocation forever.
                   Measured with all three checks loaded: peak 483 -> 230 MB (Render free has 512 MB).
Unset = onnxruntime's defaults (what the training/evaluation scripts use).
"""

import os


def ort_session(path, threads=None):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    threads = threads or int(os.environ.get("DM_ORT_THREADS", "0"))
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    if os.environ.get("DM_ORT_ARENA") == "0":
        opts.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
