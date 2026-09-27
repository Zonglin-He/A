"""Configurable sparse refinement, reusing the sealed temporal implementation."""
import math
import time
from pathlib import Path
import json
import torch
from methods.decota_refine_v1.predictor import DeCoTARefinePredictor
from methods.decota_refine_v1.api import refine
from vg_tta.tg_spatial_tta_v1 import keyframes,visual_query
from vg_tta.foreground_runtime import state_digest


def validate_config(config):
    required={'lr','steps','keyframes','phrase_threshold','distinct_margin','nms_iou'}
    if set(config)!=required:raise ValueError('Expected exactly six tunable coordinates')
    c=dict(config)
    if not math.isfinite(c['lr']) or c['lr']<=0:raise ValueError('lr must be positive and finite')
    for name,minimum in [('steps',1),('keyframes',0)]:
        if isinstance(c[name],bool) or not isinstance(c[name],int) or c[name]<minimum:raise ValueError(name)
    for name in ['phrase_threshold','distinct_margin','nms_iou']:
        if not math.isfinite(c[name]) or not 0<=c[name]<=1:raise ValueError(name)
    return c


class TunedDeCoTARefinePredictor:
    def __init__(self,student,expert,parser,*,backbone,config):
        self.config=validate_config(config);self.expert=expert;self.parser=parser
        self.native=DeCoTARefinePredictor(student,expert,parser,backbone=backbone,
            temporal_config=dict(lr=config['lr'],steps=config['steps'],gamma=0.))

    def predict(self,frames,frame_ids,metadata,*,subject=None,parsed_visual_query=None,audit_state=True):
        started=time.perf_counter()
        parsed=parsed_visual_query if parsed_visual_query is not None else visual_query(self.parser,metadata['caption'])
        # Request only the unchanged native/temporal route of the sealed
        # predictor. Its old fixed-eight-frame expert branch is explicitly off.
        r=self.native.predict(frames,frame_ids,metadata,subject=subject,
            parsed_visual_query={**parsed,'phrase':''},audit_state=audit_state)
        assert not r['expert'] and not r['pseudo']
        runtime=r['runtime'];cfg=self.config
        pos=keyframes(frame_ids,runtime['indices'],runtime['native_indices'],runtime['evidence'],cfg['keyframes'])
        before=state_digest(self.expert.model) if audit_state else None
        probes=[];pseudo=[];torch.cuda.synchronize();tick=time.perf_counter()
        if parsed['phrase']:
            for i in pos:
                z=self.expert(frames[i],parsed['phrase'],parsed['entity'],cfg)
                z.update(position=i,frame_id=frame_ids[i]);probes.append(z)
                if z['accepted']:pseudo.append({k:z[k] for k in ['position','frame_id','box','score','margin']})
        torch.cuda.synchronize();expert_seconds=time.perf_counter()-tick;tick=time.perf_counter()
        boxes,audit=refine(runtime['boxes'],pseudo,frame_ids)
        if audit_state:assert state_digest(self.expert.model)==before
        r.update(boxes=boxes,keyframes=pos,expert=probes,pseudo=pseudo,spatial_audit=audit,config=dict(cfg),
            timing=dict(expert_seconds=expert_seconds,refinement_seconds=time.perf_counter()-tick,
                wall_seconds=time.perf_counter()-started,peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9))
        return r


def load_selected(backbone,group,selection_path=None):
    root=Path(__file__).resolve().parents[2]
    path=Path(selection_path) if selection_path else root/'artifacts/decota_refine_tuning_v1/selection_barrier.json'
    from scripts.decota_matrix_common_v1 import sha
    p=json.loads(path.read_text());r=p['configs'][backbone][group]
    if sha(r['path'])!=r['sha256']:raise RuntimeError('Selected configuration changed after sealing')
    return validate_config(r['config'])
