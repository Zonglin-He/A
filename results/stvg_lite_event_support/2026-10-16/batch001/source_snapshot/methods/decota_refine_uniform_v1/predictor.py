"""Uniform anchors + absolute boxes; native temporal parameter TTA unchanged."""
import time
import torch
from methods.decota_refine_v1.predictor import DeCoTARefinePredictor
from methods.decota_refine_tuned_v1.predictor import validate_config
from methods.decota_refine_uniform_v1.api import select_positions, reconstruct, coverage_radius
from vg_tta.tg_spatial_tta_v1 import visual_query
from vg_tta.foreground_runtime import state_digest


class UniformDeCoTARefinePredictor:
    def __init__(self, student, expert, parser, *, backbone, config,
                 selector='uniform', reconstruction='absolute'):
        if selector not in ('uniform', 'minimax'):
            raise ValueError(selector)
        if reconstruction not in ('absolute', 'minimum_energy', 'residual'):
            raise ValueError(reconstruction)
        self.config = validate_config(config)
        self.selector = selector; self.reconstruction = reconstruction
        self.expert = expert; self.parser = parser
        self.native = DeCoTARefinePredictor(student, expert, parser, backbone=backbone,
            temporal_config=dict(lr=config['lr'], steps=config['steps'], gamma=0.))

    def predict(self, frames, frame_ids, metadata, *, subject=None,
                parsed_visual_query=None, audit_state=True):
        started = time.perf_counter()
        parsed = parsed_visual_query if parsed_visual_query is not None else visual_query(self.parser, metadata['caption'])
        # Reuse the sealed temporal route with its spatial expert branch off.
        r = self.native.predict(frames, frame_ids, metadata, subject=subject,
            parsed_visual_query={**parsed, 'phrase': ''}, audit_state=audit_state)
        assert not r['expert'] and not r['pseudo']
        runtime = r['runtime']; cfg = self.config
        pos = select_positions(frame_ids, runtime['indices'], cfg['keyframes'], self.selector)
        before = state_digest(self.expert.model) if audit_state else None
        probes, pseudo = [], []
        torch.cuda.synchronize();tick = time.perf_counter()
        if parsed['phrase']:
            for i in pos:
                z = self.expert(frames[i], parsed['phrase'], parsed['entity'], cfg)
                z.update(position=i, frame_id=frame_ids[i]);probes.append(z)
                if z['accepted']:
                    pseudo.append({k:z[k] for k in ['position','frame_id','box','score','margin']})
        torch.cuda.synchronize();expert_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        boxes, audit = reconstruct(runtime['boxes'], pseudo, frame_ids, self.reconstruction)
        if audit_state:
            assert state_digest(self.expert.model)==before
        r.update(boxes=boxes, keyframes=pos, expert=probes, pseudo=pseudo, spatial_audit=audit,
            config=dict(cfg), selector=self.selector, reconstruction=self.reconstruction,
            selected_coverage_radius=coverage_radius(frame_ids, runtime['indices'], pos),
            timing=dict(expert_seconds=expert_seconds, refinement_seconds=time.perf_counter()-tick,
                wall_seconds=time.perf_counter()-started, peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9))
        return r
