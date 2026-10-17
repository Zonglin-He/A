"""Strict local weight loading through the extracted TA-STVG compatibility shim.

Only the tested construction/loading functions are retained in _tastvg_load;
there are no imports of historical experiment runners in the runtime package.
"""

import numpy as np
import torch

from .config import ROOT, EXPERT_SNAPSHOT, EXPERT_SHA256
from .tensors import sha256


def load(config):
    from ._tastvg_load import load_model_on_device
    from .observations import SpatialExpert, QuerySubjectParser

    checkpoint = ROOT / config.checkpoint
    expert_weights = ROOT / EXPERT_SNAPSHOT / "model.safetensors"
    for path, expected in ((checkpoint, config.checkpoint_sha256), (expert_weights, EXPERT_SHA256)):
        if sha256(path) != expected:
            raise RuntimeError(f"Pinned model weights changed: {path}")
    torch.set_num_threads(4)
    torch.manual_seed(20260910)
    np.random.seed(20260910)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runtime = ROOT / "artifacts/tastvg_runtime" / ("hc-stvg2" if config.source_dataset == "hcstvg2" else "vidstg")
    model, _, _ = load_model_on_device(config.source_dataset, runtime, checkpoint=checkpoint,
                                       device="cuda", source_dataset=config.source_dataset)
    model.eval().requires_grad_(False)
    expert = SpatialExpert(ROOT / EXPERT_SNAPSHOT)
    parser = QuerySubjectParser(ROOT / ".cache/stanza")
    return model, expert, parser
