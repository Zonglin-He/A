"""Source-frozen P7-A WHEN plus the original joint OPD spatial update.

The temporal readout never changes the Native observation support or the
spatial objective. Every query still resets QR and Adam; only fractional LN
history persists. Production registration is an explicit separate action.
"""
import hashlib
import json
from pathlib import Path
import time
import torch
from vg_tta.decota_spatial_opd_tunable_v1 import fit
from vg_tta.spatial_online_state_v1 import arrival
from methods.decota_final_simplified_v1.config import MethodConfig
from methods.decota_final_simplified_v1.objectives import prediction, project
from methods.decota_final_simplified_v1.observations import observations


class P7AOPDPredictor:
    def __init__(self, student, expert, *, dataset, config=None):
        if dataset not in ('hc2', 'vidstg'):
            raise ValueError(dataset)
        self.student = student.eval().requires_grad_(False)
        self.expert = expert
        self.expert.model.eval().requires_grad_(False)
        self.dataset = dataset
        self.previous = None
        if config is None:
            old = Path(__file__).resolve().parents[1]/'decota_spatial_opd_v1/configs.json'
            config = json.loads(old.read_text())['datasets'][dataset]['config']
        self.config = dict(config)
        self.temporal_config = MethodConfig.for_direction(
            'vid_to_hc1' if dataset == 'hc2' else 'hc2_to_vid')

    def reset_stream(self):
        self.previous = None

    def predict(self, frames, frame_ids, metadata, *, parses, key=None):
        from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
        key = key or hashlib.sha256(json.dumps(
            [metadata['caption'], frame_ids], ensure_ascii=False).encode()).hexdigest()
        row = dict(input=metadata, frame_ids=list(frame_ids), parses=parses, key=key)
        started = time.perf_counter()
        batch, records, base = frozen_forward(self.student, frames, row)
        native = prediction(base.zero['logits'], base.zero['boxes'], records, frame_ids)
        # Sparse evidence remains inside the original Native interval I0.
        evidence = observations(self.expert, parses, frames, frame_ids, native['indices'], audit=True)
        initial = arrival(base.initial, self.previous, 'O-split')
        result = fit(base, initial, evidence, frame_ids, key, 'on_policy', config=self.config)
        alpha = self.config['writeback']
        self.previous = {n: (torch.zeros_like(v.cpu()) if n == 'spatial.query_residual'
                            else v.cpu()+(result['state'][n]-v.cpu())*alpha)
                         for n, v in initial.items()}
        temporal_started = time.perf_counter()
        frozen_time = project(base.zero['logits'], base.zero['actions'], records, self.temporal_config)
        iv = [r['interval'] for r in frozen_time['offsets']]
        interval = [min(x[0] for x in iv), max(x[1] for x in iv)]
        indices = [frame_ids.index(interval[0]), frame_ids.index(interval[1]-1)]
        return dict(boxes=result['final'], indices=indices, physical_interval=interval,
                    frame_ids=list(frame_ids), spatial_audit=result, expert=evidence,
                    Native_interval=native['physical_interval'], Native_observation_indices=native['indices'],
                    projection=frozen_time, temporal_source_frozen=True, GT_used=False,
                    active_parameters=1792, compute=dict(
                        projection_seconds=time.perf_counter()-temporal_started,
                        complete_predict_seconds=time.perf_counter()-started))
