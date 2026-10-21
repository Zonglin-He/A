"""CPU synthetic interface checks, not CG checkpoint/GPU or task evaluation."""
import ast
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from stvg import cgstvg as cg


class ForbiddenTargets:
    def __getitem__(self, _key):
        raise AssertionError("Ground-truth target was read")

    def __iter__(self):
        raise AssertionError("Ground-truth targets were iterated")


class Layer(nn.Module):
    def forward(self, query_tgt, **_kwargs):
        return query_tgt + .1, None


def synthetic_decoder(dim=256):
    """Run the actual official PosDecoder.forward, replacing only learned ops.

    Real checkpoint/model construction is deliberately not disguised by this
    fixture. Actual learned layers and RGB capture remain separate GPU gates.
    """
    path = cg.VENDOR / "models/grounding_model/query_decoder.py"
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "PosDecoder")
    functions = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in {"forward", "gt_info"}]
    namespace = dict(torch=torch, Tensor=Tensor, Optional=Optional,
                     gen_sineembed_for_position=lambda b: b.repeat(1, 1, dim // 4))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    Synthetic = type("SyntheticCG", (nn.Module,),
                     {"forward": namespace["forward"], "gt_info": namespace["gt_info"]})
    d = Synthetic()
    d.layers = nn.ModuleList([Layer()])
    d.ref_point_head = nn.Identity()
    d.query_scale = nn.Identity()
    d.bbox_embed = nn.Linear(dim, 4)
    d.conf = nn.Linear(dim, 1)
    d.conf2 = nn.Linear(dim, 1)
    d.norm = nn.Identity()
    d.d_model = dim
    d.return_intermediate = True
    d.return_weights = False
    d.theta_s = .5
    d.theta_t = .5
    d.generate_roi_feature = lambda feature, *_args: feature.mean(0)
    d.get_context_index_by_our = lambda conf, *_args: torch.arange(len(conf))
    d.generate_context = lambda roi2, roi3, index: torch.cat((roi2[index], roi3[index]))
    return d.eval().requires_grad_(False)


def arguments(n=4, dim=256, target=None):
    return dict(query_tgt=torch.zeros(n, 1, dim), pred_boxes=torch.ones(n, 1, 4) * .5,
                query_time=torch.zeros(n, 1, dim), query_mask=torch.zeros(1, n, dtype=torch.bool),
                encoded_feature=torch.ones(2, n, dim), encoded_pos=torch.zeros(2, n, dim),
                encoded_mask=torch.zeros(n, 2, dtype=torch.bool), feature_size=(1, 1, None),
                targets=target, iteration_rate=-1, time_info=torch.ones(n))


def test_source_domain_and_hash_binding():
    assert cg.source_for("hc2") == "vidstg"
    assert cg.source_for("vidstg") == "hc2"
    assert cg.CHECKPOINTS["vidstg"][0] == "vidstg.pth"
    with pytest.raises(ValueError):
        cg.source_for("hc1")


def test_real_inference_code_does_not_read_synthetic_targets():
    torch.manual_seed(7)
    d = synthetic_decoder()
    labels = [dict(eval=True, boxs=torch.zeros(4, 4), durations=[4], actioness=torch.ones(4))]
    reference = d(**arguments(target=labels))
    original = type(d).forward
    with pytest.raises(AssertionError, match="Ground-truth"):
        d(**arguments(target=ForbiddenTargets()))
    with cg.inference_only(d) as audit:
        actual = d(**arguments(target=ForbiddenTargets()))
        assert audit["only_dead_gt_read_removed"]
        assert all(torch.equal(a, b) for a, b in zip(actual, reference))
        with pytest.raises(RuntimeError, match="restricted"):
            d(**{**arguments(), "iteration_rate": 0})
        d.train()
        with pytest.raises(RuntimeError, match="restricted"):
            d(**arguments())
        d.eval()
    assert "forward" not in d.__dict__
    assert type(d).forward is original
    # The original method still performs its original read after scope exit.
    with pytest.raises(AssertionError, match="Ground-truth"):
        d(**arguments(target=ForbiddenTargets()))


def test_cg_return_shape_and_replay_gradient():
    torch.manual_seed(9)
    d = synthetic_decoder()
    kw = [arguments(), arguments()]
    with cg.inference_only(d):
        raw = [cg.select_boxes(d(**x)) for x in kw]
    source = torch.stack([raw[p % 2][p // 2] for p in range(8)])
    E = source.double().clone()
    E[:, 0] += .02
    model = nn.Module()
    model.ground_decoder = nn.Module()
    model.ground_decoder.decoder = d
    replay = cg.SpatialReplay(model, kw, source, E, list(range(8)), device="cpu")
    assert replay.state.shape == (8, 256)
    zero = replay.values()
    assert torch.equal(zero["boxes"], E)
    assert torch.equal(zero["native"], source)
    value = replay.values()["boxes"].sum()
    gradient = torch.autograd.grad(value, replay.state)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    with torch.no_grad():
        replay.state.add_(.03 * gradient.sign())
    assert not torch.equal(replay.values()["boxes"], E)
    assert all(not p.requires_grad and p.grad is None for p in replay.decoder.parameters())
    with pytest.raises(ValueError):
        cg.select_boxes(torch.zeros(6, 1, 4, 4))


def test_strong_uses_cg_six_layers_and_same_physical_support():
    from stvg.temporal import probabilities, mass_table
    records = [dict(frame_ids=[0, 2, 4, 6]), dict(frame_ids=[1, 3, 5, 7])]
    torch.manual_seed(10)
    native = [torch.randn(6, 1, 4, 2), torch.randn(6, 1, 4, 2)]
    hires = [torch.randn(6, 1, 4, 2), torch.randn(6, 1, 4, 2)]
    actual = cg.strong_distribution(records, native, hires)
    prior = np.mean([mass_table(probabilities(records, [x[k] for x in native]))[1]
                     for k in range(6)], axis=0)
    high = mass_table(probabilities(records, [x[-1] for x in hires]))[1]
    np.testing.assert_array_equal(actual["posterior"], .5 * prior + .5 * high)
    assert actual["frame_ids"] == list(range(8))
    with pytest.raises(ValueError):
        cg.strong_distribution(records, [x[:5] for x in native], hires)


def test_official_imports_refuse_foreign_models(monkeypatch):
    import types
    module = types.ModuleType("models")
    module.__file__ = "/unrelated/TA-STVG/models/__init__.py"
    monkeypatch.setitem(sys.modules, "models", module)
    with pytest.raises(RuntimeError, match="fresh CG worker"):
        cg.official_imports()


def test_actual_cg_pipeline_accepts_explicit_precision_and_residency_patch():
    path = cg.VENDOR / "models/pipeline.py"
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "CGSTVG")
    forward = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "forward")
    namespace = {}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"), namespace)
    compiled, audit = cg.frozen_prefix_forward(namespace["forward"])
    assert callable(compiled)
    assert audit["frozen_prefix"] == "float16_autocast"
    assert audit["decoder_and_heads"] == "float32"
    assert audit["visual_tensor_host_staging"]
    # This checks the storage wrapper's exact source anchors on the CG Swin,
    # without constructing the model or executing a GPU operation.
    from stvg._attention import storage_forward
    path = cg.VENDOR / "models/vidswin/video_swin_transformer.py"
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "WindowAttention3D")
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "forward")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    assert callable(storage_forward(ns["forward"], ns))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["2d", "3d"])
def test_real_roi_host_coordinates_preserve_output_and_feature_gradient(dtype, kind):
    path = cg.VENDOR / "models/grounding_model/query_decoder.py"
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "PosDecoder")
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "generate_roi_feature")
    # Use the actual official coordinate conversion, too.
    utilpath = cg.VENDOR / "models/net_utils.py"
    utiltree = ast.parse(utilpath.read_text())
    boxfn = next(x for x in utiltree.body if isinstance(x, ast.FunctionDef) and x.name == "box_cxcywh_to_xyxy")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[boxfn], type_ignores=[]), str(utilpath), "exec"), namespace)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    RealROI = type("RealROI", (nn.Module,), {"generate_roi_feature": namespace["generate_roi_feature"]})
    decoder = RealROI().eval()
    original = decoder.generate_roi_feature.__func__
    torch.manual_seed(81)
    feature = torch.randn(33, 6, 4, dtype=dtype, requires_grad=True)
    boxes = torch.tensor([[.5,.5,.8,.8], [0.,0.,.05,.1], [1.2,.5,.2,.3],
                          [.4,-.1,.2,.2], [.32,.49,.01,.01], [.8,.8,1.4,1.4]], dtype=dtype)
    expected = decoder.generate_roi_feature(feature, (3, 5, None), boxes, kind)
    expected_grad = torch.autograd.grad(expected.square().sum(), feature)[0]
    with cg.roi_coordinates_on_host(decoder) as audit:
        assert audit["applied"] and audit["ROI_clone_slice_mean_unchanged"]
        actual = decoder.generate_roi_feature(feature, (3, 5, None), boxes, kind)
        actual_grad = torch.autograd.grad(actual.square().sum(), feature)[0]
    assert decoder.generate_roi_feature.__func__ is original
    assert "generate_roi_feature" not in decoder.__dict__
    assert torch.equal(actual, expected)
    assert torch.equal(actual_grad, expected_grad)


def test_synthetic_roi_function_is_not_force_patched():
    d = synthetic_decoder()
    original = d.generate_roi_feature
    with cg.roi_coordinates_on_host(d) as audit:
        assert not audit["applied"]
        assert d.generate_roi_feature is original


def test_cg_config_rebinds_only_backbone_metadata_and_checkpoint():
    import json
    for dataset in ["hc2", "vidstg"]:
        original = json.loads((cg.ROOT / "configs/legacy_opd.json").read_text())["datasets"][dataset]
        actual = cg.Predictor(dataset).config
        changed = {k for k in set(original) | set(actual) if original.get(k) != actual.get(k)}
        assert changed == {"backbone", "source_checkpoint", "temporal", "E_calibration_provenance"}
        assert actual["E_calibration"] == original["E_calibration"]
        assert actual["temporal"] == "Strong CG-STVG only, original MBR"
        assert "not recalibrated" in actual["E_calibration_provenance"]
