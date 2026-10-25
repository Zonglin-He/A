"""Pinned algorithm imports only. This module never constructs a model/optimizer."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = ROOT / "outputs/post_suite_tta_v1"


def _checked_source(method, relative):
    lock = json.loads((Path(__file__).parent / "sources.lock.json").read_text())
    record = next(x for x in lock["methods"] if x["name"] == method)
    inventory = PRIVATE / record["private_inventory"]
    if hashlib.sha256(inventory.read_bytes()).hexdigest() != record["inventory_sha256"]:
        raise RuntimeError("Source inventory binding changed")
    path = PRIVATE / "generic_sources" / method / relative
    expected = next(x["sha256"] for x in json.loads(inventory.read_text())["files"]
                    if x["path"] == str(path.relative_to(PRIVATE)))
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Upstream source bytes changed")
    return path


def _module(method, relative, name):
    path = _checked_source(method, relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_algorithm(method):
    """Import audited pinned code; call no adaptation, input or model function."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Preparation import checks require CUDA_VISIBLE_DEVICES='' explicitly")
    if method == "Tent":
        return _module(method, "tent.py", "_prepared_official_tent")
    if method == "SAR":
        module = _module(method, "sar.py", "_prepared_official_sar")
        module.sam_module = _module(method, "sam.py", "_prepared_official_sam")
        return module
    if method == "EATA":
        return _module(method, "eata.py", "_prepared_official_eata")
    if method == "CoTTA":
        # Upstream has an absolute sibling import. Restore the previous binding.
        previous = sys.modules.get("my_transforms")
        transforms = _module(method, "imagenet/my_transforms.py", "_prepared_cotta_transforms")
        try:
            sys.modules["my_transforms"] = transforms
            return _module(method, "imagenet/cotta.py", "_prepared_official_cotta")
        finally:
            if previous is None:
                sys.modules.pop("my_transforms", None)
            else:
                sys.modules["my_transforms"] = previous
    if method == "MEMO":
        # The upstream test script parses arguments/loads data at import time.
        # Preserve and compile ONLY its unchanged marginal-entropy definition.
        import numpy as np
        import torch
        path = _checked_source(method, "imagenet-exps/test_calls/test_adapt.py")
        source = path.read_text()
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "marginal_entropy")
        fragment = ast.get_source_segment(source, fn)
        module = types.ModuleType("_prepared_official_memo_objective")
        module.__dict__.update(torch=torch, np=np)
        exec(compile(fragment, str(path), "exec"), module.__dict__)
        module.preparation_scope = "Unchanged official objective only; real augmentation and STVG model bridge still required"
        return module
    raise ValueError("Unsupported prepared method")


def import_check():
    import torch
    entry = {"Tent": "Tent", "SAR": "SAR", "EATA": "EATA", "MEMO": "marginal_entropy", "CoTTA": "CoTTA"}
    result = {}
    for method, symbol in entry.items():
        module = load_algorithm(method)
        if not callable(getattr(module, symbol)):
            raise RuntimeError("Missing upstream algorithm entry")
        result[method] = {"entry": symbol, "import_passed": True,
                          "model_or_optimizer_constructed": False, "algorithm_called": False}
    if torch.cuda.is_initialized():
        raise RuntimeError("Preparation unexpectedly initialized CUDA")
    return {"methods": result, "torch_version": torch.__version__,
            "CUDA_initialized": False, "new_models": 0, "optimizer_steps": 0,
            "new_Source_or_expert_or_GT": 0}


if __name__ == "__main__":
    print(json.dumps(import_check(), indent=2))
