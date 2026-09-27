"""One frozen spatial expert, zero spatial backward, unchanged temporal TTA."""
import time
import numpy as np
import torch
from methods.decota_v1 import decode
from methods.decota_s_v1.api import native_episode
from methods.decota_refine_v1.api import refine
from vg_tta.decota_tastvg_episode_v1 import make_batch
from vg_tta.tg_spatial_tta_v1 import keyframes,visual_query
from vg_tta.foreground_runtime import state_digest
from vg_tta.metrics import interval_from_logits


def tube_inclusion(z):
    # Native single-view legal start/end posterior, inclusive frame support.
    z=z.detach().cpu().float();n=len(z)
    score=z[:,0].log_softmax(0)[:,None]+z[:,1].log_softmax(0)[None,:]
    legal=torch.triu(torch.ones(n,n,dtype=torch.bool),diagonal=1)
    joint=torch.zeros_like(score);joint[legal]=torch.softmax(score[legal],0)
    return (joint.sum(1).cumsum(0)-torch.cat([torch.zeros(1),joint.sum(0).cumsum(0)[:-1]])).clamp(0,1).tolist()


class DeCoTARefinePredictor:
    def __init__(self,student,expert,parser,*,backbone,temporal_config):
        if backbone not in ['tastvg','tubedetr']:raise ValueError(backbone)
        self.student=student.eval().requires_grad_(False);self.expert=expert
        self.expert.model.eval().requires_grad_(False);self.parser=parser
        self.backbone=backbone;self.temporal_config=dict(temporal_config);self.snapshot=None
        if backbone=='tubedetr':
            from scripts.evaluate_fullspan_scale_shift_external_v1 import _snapshot_full
            self.snapshot=_snapshot_full(student)

    def predict(self,frames,frame_ids,metadata,*,subject=None,parsed_visual_query=None,audit_state=True):
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        before=state_digest(self.student) if audit_state else None
        expert_before=state_digest(self.expert.model) if audit_state else None
        parsed=parsed_visual_query if parsed_visual_query is not None else visual_query(self.parser,metadata['caption'])
        if self.backbone=='tastvg':
            if subject is None:subject=self.parser(metadata['caption'])['subject']
            batch=make_batch(frames,frame_ids,metadata,subject,self.student)
            runtime=native_episode(self.student,batch,self.temporal_config)
            # No spatial solver is called; caches are released immediately.
            del runtime['caches'],batch
        else:
            from vg_tta.unanchored_shift_predictor_v1 import predict_episode
            r=predict_episode(self.student,frames,frame_ids,metadata['caption'],str(metadata['index']),
                selected_config=self.temporal_config,source_snapshot=self.snapshot,
                baselines=False,check_controls=True,device='cuda:0')
            n=r['native_predictions'];z=n['frozen']['pred_sted'][0].cpu()
            native=list(interval_from_logits(n['frozen']['pred_sted']))
            extent=interval_from_logits(n['ours']['pred_sted'])
            result=decode(z,extent,frame_ids,native_indices=native)
            runtime=dict(boxes=n['frozen']['pred_boxes'].float().cpu(),native_indices=native,
                indices=list(result['indices']),frame_ids=list(frame_ids),evidence=tube_inclusion(z),
                temporal_audit=r['audits'],timing=r['timing'])
            del r
        positions=keyframes(frame_ids,runtime['indices'],runtime['native_indices'],runtime['evidence'],8)
        probes=[];pseudo=[];torch.cuda.synchronize();tick=time.perf_counter()
        if parsed['phrase']:
            for pos in positions:
                pr=self.expert(frames[pos],parsed['phrase'],parsed['entity'])
                pr.update(position=pos,frame_id=frame_ids[pos]);probes.append(pr)
                if pr['accepted']:pseudo.append({k:pr[k] for k in ['position','frame_id','box','score','margin']})
        torch.cuda.synchronize();expert_seconds=time.perf_counter()-tick
        tick=time.perf_counter();boxes,spatial_audit=refine(runtime['boxes'],pseudo,frame_ids)
        refinement_seconds=time.perf_counter()-tick
        if audit_state:
            assert state_digest(self.student)==before and state_digest(self.expert.model)==expert_before
        assert all(not p.requires_grad and p.grad is None for p in self.student.parameters())
        return dict(boxes=boxes,indices=runtime['indices'],frame_ids=list(frame_ids),runtime=runtime,
            keyframes=positions,expert=probes,pseudo=pseudo,spatial_audit=spatial_audit,
            spatial_backward_calls=0,spatial_optimizer_steps=0,GT_used=False,
            state_reset_checked=audit_state,timing=dict(expert_seconds=expert_seconds,refinement_seconds=refinement_seconds,
                wall_seconds=time.perf_counter()-start,peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9))
