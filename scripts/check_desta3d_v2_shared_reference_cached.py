"""CPU/mock contract check for the official incremental-cache revision.

Run with CUDA disabled:
    CUDA_VISIBLE_DEVICES='' .venv-ptd-audit/bin/python -B \\
      scripts/check_desta3d_v2_shared_reference_cached.py

No checkpoint, media, source/target labels, or GT files are opened.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/ParallelTubeDecoding/src"))

import torch

from scripts.check_desta3d_v2_shared_reference import (
    FakeAdapter,
    FakeModel,
    FakeTokenizer,
    FakePTD,
    _fake_fields,
    _fake_inputs,
)
from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass


class CachedFakePTD(FakePTD):
    def __init__(self, *, malformed_event=False, alternative_spatial_reference=True):
        super().__init__(malformed_event=malformed_event)
        self.alternative_spatial_reference = alternative_spatial_reference
        self.generator_formats = []
        self.in_spatial_generation = False

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
                model, probe_ids, past_key_values, attention_mask=None,
                position_offsets=torch.arange(probe_ids.shape[1]),
                logits_to_keep=probe_ids.shape[1], ptd_attention_plan=None,
            )
            c = token_ids["coord_id_to_value"]
            value_to_id = {value: token for token, value in c.items()}
            block = [token_ids["box_start"], value_to_id[100], value_to_id[120],
                     value_to_id[800], value_to_id[880], token_ids["box_end"]]
            return torch.tensor([block] * q.numel(), dtype=torch.long), past_key_values
        self.event_probe_count += 1
        if self.event_probe_count == 1:
            value_to_id = {value: token for token, value in token_ids["coord_id_to_value"].items()}
            second = 21 if self.alternative_spatial_reference and self.in_spatial_generation else 20
            row = [token_ids["ref_start"], second, token_ids["ref_end"], token_ids["null"], token_ids["null"], token_ids["null"]]
        elif self.event_probe_count == 2:
            row = [token_ids["time_start"], 11, 12, token_ids["time_end"], token_ids["null"], token_ids["null"]]
        else:
            raise AssertionError("unexpected cached PTD probe order")
        return torch.tensor([row], dtype=torch.long), past_key_values

    def generate_ptd(
        self, model, tokenizer, inputs, *, max_new_tokens, max_time_tokens,
        temperature, generation_format, ptd_attn_implementation,
    ):
        self.generator_formats.append(generation_format)
        if generation_format == "temporal_localization":
            return super().generate_ptd(
                model, tokenizer, inputs, max_new_tokens=max_new_tokens,
                max_time_tokens=max_time_tokens, temperature=temperature,
                generation_format=generation_format,
                ptd_attn_implementation=ptd_attn_implementation,
            )
        if generation_format != "spatio_temporal_grounding":
            raise AssertionError(f"unexpected PTD format: {generation_format}")
        self.generate_calls += 1
        self.in_spatial_generation = True
        self.event_probe_count = 0
        nframes = int(inputs["video_grid_thw"][0, 0])
        token_ids = self.ids
        cache = model(**inputs).past_key_values
        original = inputs["input_ids"].detach().clone()
        generated = original
        prefix_len = int(generated.shape[1])
        q = generated[0, -1:].clone()
        raw_sem, cache = self._run_cached_ptd_probe(
            model, generated, token_ids, cache,
            query_token_ids=q,
            probe_position_starts=torch.tensor([prefix_len - 1]),
            context_limits=torch.tensor([prefix_len - 1]), block_size=6,
            temperature=0., top_p=None, top_k=None,
            ptd_attn_implementation=ptd_attn_implementation,
        )
        semantic, done = self._parse_semantic_block(
            raw_sem[0].tolist(), token_ids, first_block=True, block_size=6
        )
        assert done
        generated = torch.cat((generated, torch.tensor([[*semantic, token_ids["newline"]]])), dim=1)
        prefix_len = int(generated.shape[1])
        raw_time, cache = self._run_cached_ptd_probe(
            model, generated, token_ids, cache,
            query_token_ids=torch.tensor([token_ids["ref_end"]]),
            probe_position_starts=torch.tensor([prefix_len - 1]),
            context_limits=torch.tensor([prefix_len]), block_size=6,
            temperature=0., top_p=None, top_k=None,
            ptd_attn_implementation=ptd_attn_implementation,
        )
        temporal, anchors = self._parse_temporal_block(
            raw_time[0].tolist(), token_ids, block_size=6
        )
        generated = torch.cat((generated, torch.tensor([[*temporal, token_ids["newline"]]])), dim=1)
        prefix_len = int(generated.shape[1])
        starts = prefix_len + 8 * torch.arange(len(anchors))
        contexts = torch.full((len(anchors),), prefix_len, dtype=torch.long)
        raw_boxes, cache = self._run_cached_ptd_probe(
            model, generated, token_ids, cache,
            query_token_ids=torch.tensor(anchors),
            probe_position_starts=starts, context_limits=contexts,
            block_size=6, temperature=0., top_p=None, top_k=None,
            ptd_attn_implementation=ptd_attn_implementation,
        )
        boxes = self._validate_box_blocks(raw_boxes, token_ids, expected_blocks=len(anchors), block_size=6)
        rows = []
        for i, (anchor, block) in enumerate(zip(anchors, boxes)):
            rows.extend([anchor, *block, token_ids["eos"] if i == len(anchors)-1 else token_ids["newline"]])
        generated = torch.cat((generated, torch.tensor([rows])), dim=1)
        return generated[:, original.shape[1]:], SimpleNamespace(stopped=True)


class CachedSharedReferenceTests(unittest.TestCase):
    def run_decode(self):
        ptd = CachedFakePTD()
        model_package = type(sys)("model")
        model_package.__path__ = []
        model_package.ptd_generation = ptd
        model = FakeModel()
        processor = SimpleNamespace(tokenizer=FakeTokenizer())
        with patch.dict(sys.modules, {"model": model_package, "model.ptd_generation": ptd}):
            result = decode_shared_reference_two_pass(
                model, processor, _fake_inputs(), FakeAdapter(), _fake_fields()
            )
        return result, model, ptd

    def test_official_incremental_schedule_forces_event_ids_and_captures_box_logits(self):
        result, model, ptd = self.run_decode()
        self.assertTrue(result["format_ok"])
        self.assertEqual(ptd.generator_formats, ["temporal_localization", "spatio_temporal_grounding"])
        self.assertEqual(len(model.prefills), 2)
        self.assertTrue(torch.equal(model.prefills[0]["input_ids"], model.prefills[1]["input_ids"]))
        # Two event probes then semantic, temporal, and box probes in the fresh spatial cache.
        self.assertEqual(len(ptd.probes), 5)
        event_caches = {id(row["cache"]) for row in ptd.probes[:2]}
        spatial_caches = {id(row["cache"]) for row in ptd.probes[2:]}
        self.assertTrue(event_caches.isdisjoint(spatial_caches))
        box_probe = ptd.probes[-1]
        self.assertEqual(box_probe["query"].tolist(), [11, 12])
        self.assertEqual(box_probe["contexts"].tolist(), [13, 13])
        self.assertEqual(box_probe["starts"].tolist(), [13, 21])
        self.assertEqual(result["spatial"]["shared_reference_token_ids"], [1, 20, 2])
        self.assertEqual(result["spatial"]["shared_time_token_ids"], [3, 11, 12, 4])
        self.assertEqual(result["spatial"]["positions"], [1, 2])
        self.assertEqual(tuple(result["spatial"]["logits"].shape), (2, 4, 1001))

    def test_bad_event_prefix_does_not_start_spatial_pass(self):
        ptd = CachedFakePTD(malformed_event=True)
        model_package = type(sys)("model")
        model_package.__path__ = []
        model_package.ptd_generation = ptd
        model = FakeModel()
        processor = SimpleNamespace(tokenizer=FakeTokenizer())
        with patch.dict(sys.modules, {"model": model_package, "model.ptd_generation": ptd}):
            result = decode_shared_reference_two_pass(
                model, processor, _fake_inputs(), FakeAdapter(), _fake_fields()
            )
        self.assertFalse(result["format_ok"])
        self.assertIsNone(result["spatial"])
        self.assertEqual(ptd.generate_calls, 1)
        self.assertEqual(len(model.prefills), 1)


if __name__ == "__main__":
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contract test unexpectedly initialized CUDA")
    unittest.main(verbosity=2)
