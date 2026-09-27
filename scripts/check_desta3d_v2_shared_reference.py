"""CPU/mock contract tests for DESTA-3D v2 shared-reference PTD decoding.

Run with CUDA disabled, e.g.:
    CUDA_VISIBLE_DEVICES='' .venv-ptd-audit/bin/python -B \
      scripts/check_desta3d_v2_shared_reference.py

No checkpoint, media, labels, or GT files are opened.
"""
from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

PTD_SRC = ROOT / "external/ParallelTubeDecoding/src"
sys.path.insert(0, str(PTD_SRC))

from vg_tta.desta3d_v2_shared_reference import (
    SharedPrefixError,
    SharedReferenceTime,
    decode_shared_reference_two_pass,
    validate_shared_reference_time,
)


def _token_ids():
    ordered_time = (10, 11, 12)
    coord_id_to_value = {10000 + i: i for i in range(1001)}
    return {
        "eos": 7,
        "text_mask": 8,
        "null": 9,
        "box_start": 30,
        "box_end": 31,
        "ref_start": 1,
        "ref_end": 2,
        "time_start": 3,
        "time_end": 4,
        "newline": 6,
        "time_tokens": set(ordered_time),
        "ordered_time_tokens": ordered_time,
        "time_token_indices": {token: i for i, token in enumerate(ordered_time)},
        "coord_id_to_value": coord_id_to_value,
    }


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        if torch.is_tensor(token_ids):
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        return " ".join(f"<{int(token)}>" for token in token_ids)

    def convert_tokens_to_ids(self, token):
        if token.startswith("<") and token.endswith(">"):
            value = token[1:-1]
            if value.isdecimal() and 0 <= int(value) <= 1000:
                return 10000 + int(value)
        return None


class FakeMerger(nn.Module):
    def forward(self, tokens):
        return tokens


class FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.merger = FakeMerger()


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = FakeVisual()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeCore()
        self.config = SimpleNamespace(video_token_id=999)
        self.prefills = []
        self.cache_objects = []

    def forward(self, input_ids, attention_mask=None, video_grid_thw=None, **kwargs):
        nframes = int(video_grid_thw[0, 0])
        tokens = torch.ones((nframes, 4), dtype=torch.float32)
        updated = self.model.visual.merger(tokens)
        self.prefills.append({
            "input_ids": input_ids.detach().cpu().clone(),
            "attention_mask": attention_mask.detach().cpu().clone(),
            "mm_token_type_ids": kwargs.get("mm_token_type_ids").detach().cpu().clone(),
            "merger_tokens": updated.detach().cpu().clone(),
        })
        cache = object()
        self.cache_objects.append(cache)
        return SimpleNamespace(past_key_values=cache)


class FakeAdapter:
    def __init__(self, spatial_delta=2.0, event_delta=1.0):
        self.spatial_delta = float(spatial_delta)
        self.event_delta = float(event_delta)

    def __call__(self, visual, query_tokens, *, query_mask, frame_times, gate_override=None):
        return {
            "updated_tokens_event": visual + self.event_delta,
            "updated_tokens_spatial": visual + self.spatial_delta,
        }


class FakePTD(ModuleType):
    def __init__(self, *, malformed_event=False):
        super().__init__("model.ptd_generation")
        self.ids = _token_ids()
        self.malformed_event = malformed_event
        self.generate_calls = 0
        self.event_probe_count = 0
        self.probes = []
        self.configure_ptd_model = lambda model, block_size=6: None
        self.build_ptd_token_ids = lambda tokenizer, *, max_time_tokens: dict(self.ids)
        self.resolve_ptd_attn_implementation = lambda value: value
        self.get_token_id = lambda tokenizer, token, required=True: tokenizer.convert_tokens_to_ids(token)
        self._parse_semantic_block = self.parse_semantic_block
        self._parse_temporal_block = self.parse_temporal_block
        self._run_cached_ptd_probe = self.run_cached_probe
        self._run_language_model = self.run_language_model
        self._validate_box_blocks = self.validate_box_blocks

    def run_language_model(
        self, model, input_ids, past_key_values, *, attention_mask,
        position_offsets, logits_to_keep, ptd_attention_plan=None,
    ):
        # Mock the logits already produced by the spatial box probe. The
        # coordinate slice encodes sequence position and token value so the
        # test can catch wrong block offsets or token-ID ordering.
        logits = torch.zeros((1, logits_to_keep, 11001), dtype=torch.float32)
        values = torch.arange(1001, dtype=torch.float32)
        for position in range(logits_to_keep):
            logits[0, position, 10000:11001] = position * 2000 + values
        return SimpleNamespace(past_key_values=past_key_values), logits

    def parse_semantic_block(self, block, token_ids, *, first_block, block_size):
        values = [int(x) for x in block]
        if len(values) != block_size:
            raise RuntimeError("wrong semantic block size")
        if first_block and values[0] != token_ids["ref_start"]:
            raise RuntimeError("missing reference start")
        if token_ids["ref_end"] in values:
            idx = values.index(token_ids["ref_end"])
            return values[:idx + 1], True
        if token_ids["null"] in values:
            raise RuntimeError("null before reference end")
        return values, False

    def parse_temporal_block(self, block, token_ids, *, block_size):
        values = [int(x) for x in block]
        if len(values) != block_size or values[:1] != [token_ids["time_start"]]:
            raise RuntimeError("invalid temporal start")
        if values[1] not in token_ids["time_tokens"] or values[2] not in token_ids["time_tokens"]:
            raise RuntimeError("invalid time token")
        if values[3] != token_ids["time_end"]:
            raise RuntimeError("invalid temporal end")
        start = token_ids["time_token_indices"][values[1]]
        end = token_ids["time_token_indices"][values[2]]
        if end < start:
            raise RuntimeError("reversed interval")
        return values[:4], list(token_ids["ordered_time_tokens"][start:end + 1])

    def validate_box_blocks(self, blocks, token_ids, *, expected_blocks, block_size):
        if tuple(blocks.shape) != (expected_blocks, block_size):
            raise RuntimeError("wrong box block tensor shape")
        out = []
        for row in blocks.detach().cpu().tolist():
            row = [int(x) for x in row]
            if row[0] != token_ids["box_start"] or row[5] != token_ids["box_end"]:
                raise RuntimeError("bad box delimiters")
            if any(x not in token_ids["coord_id_to_value"] for x in row[1:5]):
                raise RuntimeError("bad coordinate token")
            out.append(row)
        return out

    def run_cached_probe(
        self, model, generated, token_ids, past_key_values, *, query_token_ids,
        probe_position_starts, context_limits, block_size, temperature, top_p,
        top_k, ptd_attn_implementation,
    ):
        q = torch.as_tensor(query_token_ids).detach().cpu().reshape(-1)
        self.probes.append({
            "generated": generated.detach().cpu().clone(),
            "cache": past_key_values,
            "query": q.clone(),
            "starts": torch.as_tensor(probe_position_starts).detach().cpu().clone(),
            "contexts": torch.as_tensor(context_limits).detach().cpu().clone(),
        })
        if q.numel() > 1:
            probe_ids = torch.full(
                (1, q.numel() * 6), token_ids["null"], dtype=generated.dtype
            )
            self._run_language_model(
                model,
                probe_ids,
                past_key_values,
                attention_mask=None,
                position_offsets=torch.arange(probe_ids.shape[1]),
                logits_to_keep=probe_ids.shape[1],
                ptd_attention_plan=None,
            )
            c = token_ids["coord_id_to_value"]
            value_to_id = {value: token for token, value in c.items()}
            block = [token_ids["box_start"], value_to_id[100], value_to_id[120],
                     value_to_id[800], value_to_id[880], token_ids["box_end"]]
            return torch.tensor([block] * q.numel(), dtype=torch.long), past_key_values
        self.event_probe_count += 1
        if self.event_probe_count == 1:
            row = [token_ids["ref_start"], 20, token_ids["ref_end"], token_ids["null"], token_ids["null"], token_ids["null"]]
        elif self.event_probe_count == 2:
            row = [token_ids["time_start"], 11, 12, token_ids["time_end"], token_ids["null"], token_ids["null"]]
        else:
            raise AssertionError("unexpected event pass PTD probe")
        return torch.tensor([row], dtype=torch.long), past_key_values

    def generate_ptd(
        self, model, tokenizer, inputs, *, max_new_tokens, max_time_tokens,
        temperature, generation_format, ptd_attn_implementation,
    ):
        self.generate_calls += 1
        assert generation_format == "temporal_localization"
        assert max_time_tokens == 3
        if self.malformed_event:
            model(**inputs)
            return torch.tensor([[50, 51]], dtype=inputs["input_ids"].dtype), SimpleNamespace(stopped=False)
        outputs = model(**inputs)
        past = outputs.past_key_values
        generated = inputs["input_ids"]
        q = torch.tensor([self.ids["ref_end"]], dtype=generated.dtype)
        raw_sem, past = self._run_cached_ptd_probe(
            model, generated, self.ids, past, query_token_ids=q,
            probe_position_starts=torch.tensor([generated.shape[1] - 1]),
            context_limits=torch.tensor([generated.shape[1]]), block_size=6,
            temperature=0., top_p=None, top_k=None, ptd_attn_implementation="sdpa",
        )
        semantic, complete = self._parse_semantic_block(
            raw_sem[0].tolist(), self.ids, first_block=True, block_size=6
        )
        assert complete
        q = torch.tensor([self.ids["ref_end"]], dtype=generated.dtype)
        raw_time, past = self._run_cached_ptd_probe(
            model, generated, self.ids, past, query_token_ids=q,
            probe_position_starts=torch.tensor([generated.shape[1] - 1]),
            context_limits=torch.tensor([generated.shape[1]]), block_size=6,
            temperature=0., top_p=None, top_k=None, ptd_attn_implementation="sdpa",
        )
        temporal, _anchors = self._parse_temporal_block(raw_time[0].tolist(), self.ids, block_size=6)
        completion = semantic + [self.ids["newline"]] + temporal + [self.ids["eos"]]
        return torch.tensor([completion], dtype=generated.dtype), SimpleNamespace(stopped=True)


def _fake_inputs():
    return {
        "input_ids": torch.tensor([[100, 101, 999, 102]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 0, 1, 0]], dtype=torch.long),
        "video_grid_thw": torch.tensor([[3, 2, 2]], dtype=torch.long),
        "pixel_values_videos": torch.zeros((3, 3, 8, 8)),
    }


def _fake_fields():
    visual = torch.ones((1, 3, 1, 1, 4))
    return {
        "visual_grid": visual,
        "query_tokens": torch.ones((1, 1, 4)),
        "query_mask": torch.ones((1, 1), dtype=torch.bool),
        "frame_times": torch.tensor([[0., 1., 2.]]),
    }


class SharedReferenceTests(unittest.TestCase):
    def _run_fake_decode(self, *, spatial_delta=2.0, malformed_event=False):
        fake_ptd = FakePTD(malformed_event=malformed_event)
        fake_model_package = ModuleType("model")
        fake_model_package.__path__ = []
        fake_model_package.ptd_generation = fake_ptd
        fake_model = FakeModel()
        processor = SimpleNamespace(tokenizer=FakeTokenizer())
        with patch.dict(sys.modules, {"model": fake_model_package, "model.ptd_generation": fake_ptd}):
            result = decode_shared_reference_two_pass(
                fake_model,
                processor,
                _fake_inputs(),
                FakeAdapter(spatial_delta=spatial_delta),
                _fake_fields(),
            )
        return result, fake_model, fake_ptd

    def test_shared_actual_ids_fresh_spatial_kv_and_anchor_queries(self):
        result, model, ptd = self._run_fake_decode(spatial_delta=2.)
        self.assertTrue(result["format_ok"])
        self.assertFalse(result["GT_used"])
        self.assertEqual(ptd.generate_calls, 1, "spatial pass must not decode semantic/time again")
        self.assertEqual(len(model.prefills), 2, "event and spatial require separate prefills")
        self.assertEqual(len(ptd.probes), 3, "2 event probes plus 1 parallel box probe")
        event_cache_refs = [p["cache"] for p in ptd.probes[:2]]
        spatial_probe = ptd.probes[2]
        self.assertTrue(all(spatial_probe["cache"] is not cache for cache in event_cache_refs))

        ids = ptd.ids
        expected_suffix = [1, 20, 2, ids["newline"], 3, 11, 12, 4, ids["newline"]]
        spatial_prefix = model.prefills[1]["input_ids"][0].tolist()
        self.assertEqual(spatial_prefix[-len(expected_suffix):], expected_suffix)
        self.assertEqual(spatial_probe["generated"][0].tolist(), spatial_prefix)
        self.assertEqual(spatial_probe["query"].tolist(), [11, 12])
        prefix_len = len(spatial_prefix)
        self.assertEqual(spatial_probe["starts"].tolist(), [prefix_len, prefix_len + 8])
        self.assertEqual(spatial_probe["contexts"].tolist(), [prefix_len, prefix_len])
        self.assertEqual(result["spatial"]["positions"], [1, 2])
        self.assertEqual(tuple(result["spatial"]["boxes"].shape), (2, 4))
        self.assertEqual(tuple(result["spatial"]["logits"].shape), (2, 4, 1001))
        self.assertEqual(len(result["spatial"]["coordinate_token_ids"]), 1001)
        expected_logits = torch.empty((2, 4, 1001), dtype=torch.float32)
        coord_values = torch.arange(1001, dtype=torch.float32)
        for box_index in range(2):
            for coordinate_index in range(4):
                probe_position = box_index * 6 + coordinate_index + 1
                expected_logits[box_index, coordinate_index] = probe_position * 2000 + coord_values
        torch.testing.assert_close(result["spatial"]["logits"], expected_logits)
        self.assertEqual(result["spatial"]["shared_reference_token_ids"], [1, 20, 2])
        self.assertEqual(result["spatial"]["shared_time_token_ids"], [3, 11, 12, 4])
        # The appended non-multimodal response IDs carry the official zero type.
        self.assertEqual(model.prefills[1]["input_ids"].shape, model.prefills[1]["attention_mask"].shape)
        self.assertEqual(model.prefills[1]["mm_token_type_ids"][0, -len(expected_suffix):].tolist(), [0] * len(expected_suffix))

    def test_spatial_branch_change_cannot_rewrite_event_decision(self):
        first, _, ptd_a = self._run_fake_decode(spatial_delta=2.)
        second, _, ptd_b = self._run_fake_decode(spatial_delta=9.)
        self.assertEqual(first["event"]["completion_token_ids"], second["event"]["completion_token_ids"])
        self.assertEqual(first["event"]["semantic_token_ids"], second["event"]["semantic_token_ids"])
        self.assertEqual(first["event"]["temporal"], second["event"]["temporal"])
        self.assertEqual(ptd_a.generate_calls, ptd_b.generate_calls, 1)

    def test_bad_event_prefix_is_retained_as_failure_without_spatial_fallback(self):
        result, model, ptd = self._run_fake_decode(malformed_event=True)
        self.assertFalse(result["format_ok"])
        self.assertIsNone(result["spatial"])
        self.assertIn("did not stop successfully", result["event"]["failure"])
        self.assertEqual(len(model.prefills), 1)
        self.assertEqual(ptd.generate_calls, 1)
        self.assertEqual(len(ptd.probes), 0)

    def test_explicit_validator_rejects_reversed_or_changed_time_tokens(self):
        good, _model, ptd = self._run_fake_decode()
        shared = good["event"]["shared_reference_time"]
        wrong = replace(shared, interval=(2, 1))
        with self.assertRaises(SharedPrefixError):
            validate_shared_reference_time(wrong, ptd.ids, ptd)
        wrong_tokens = replace(shared, time_token_ids=(3, 12, 11, 4))
        with self.assertRaises(SharedPrefixError):
            validate_shared_reference_time(wrong_tokens, ptd.ids, ptd)

    def test_official_ptd_parsers_accept_and_preserve_shared_token_ids_cpu(self):
        # Exercise the pinned official parser functions themselves on CPU, not
        # just the module-local mock parsers used for orchestration tests.
        sys.modules.pop("model", None)
        sys.modules.pop("model.ptd_generation", None)
        pg = importlib.import_module("model.ptd_generation")
        ids = _token_ids()
        shared = SharedReferenceTime(
            reference_token_ids=(1, 20, 2),
            time_token_ids=(3, 11, 12, 4),
            time_anchor_ids=(11, 12),
            interval=(1, 2),
            event_completion_ids=(1, 20, 2, 6, 3, 11, 12, 4, 7),
        )
        checked = validate_shared_reference_time(shared, ids, pg)
        self.assertIs(checked, shared)
        self.assertEqual(checked.spatial_prefix_ids(ids["newline"]), (1, 20, 2, 6, 3, 11, 12, 4, 6))


if __name__ == "__main__":
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contract test unexpectedly initialized CUDA")
    unittest.main(verbosity=2)
