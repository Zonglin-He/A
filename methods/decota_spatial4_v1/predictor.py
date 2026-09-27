"""Exact four-original-frame A4; no unrequested weak-view expert calls."""
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from methods.decota_spatial10_v1.predictor import Spatial10Predictor,load_config,sha256

ROOT=Path(__file__).resolve().parents[2]
METHOD=Path(__file__).resolve().parent

def config(direction):
    c=load_config(direction)
    c.update(json.loads((METHOD/'configs.json').read_text())['overrides'])
    return c

def observations(expert,parses,frames,ids,interval):
    from methods.decota_refine_uniform_v1.api import uniform_positions
    from scripts.run_decota_spatial_extension_v1 import ContextView
    from vg_tta.decota_s2_system_v1 import probe_from_detection
    from vg_tta.parametric_observation_v1 import tensor_hash
    context,s1=parses['context'],parses['old']
    eligible=context['eligible'] and len(expert.processor.tokenizer(context['context'])['input_ids'])<=256
    positions=uniform_positions(ids,interval,4);obs={};anchors=[]
    for pos in positions:
        if not eligible and not s1['phrase']:continue
        rgb=frames[pos];inp={}
        def hook(m,args,kw):
            for k in ('pixel_values','pixel_mask','input_ids','attention_mask'):
                if k in kw:inp[k]=dict(sha256=tensor_hash(kw[k]),shape=list(kw[k].shape),dtype=str(kw[k].dtype))
        h=expert.model.register_forward_pre_hook(hook,with_kwargs=True)
        torch.cuda.synchronize();tick=time.perf_counter()
        try:
            with torch.no_grad():
                d=ContextView(expert,context)(rgb,context['context'],context['entity']) if eligible else expert(rgb,s1['phrase'],s1['entity'])
        finally:h.remove()
        torch.cuda.synchronize()
        if eligible:z=probe_from_detection(d,pos,ids[pos])
        else:z=dict(position=pos,frame_id=ids[pos],accepted=d['accepted'],reason=d['reason'],margin=d['margin'],
            boxes=torch.tensor([d['box']]) if d['box'] else torch.empty(0,4),target_scores=[d['score']] if d['box'] else [],candidate_ids=[0] if d['box'] else [])
        receipt=dict(position=pos,frame_id=ids[pos],view='original',scale=1.,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            inputs=inp,seconds=time.perf_counter()-tick,forward_seconds=d['seconds'],text=d['text'],
            target_span=context['span'] if eligible else None,context_active=eligible)
        obs[('original',pos)]=dict(probe=z,receipt=receipt)
        if z['accepted']:
            j=int(np.argmax(z['target_scores']))
            anchors.append(dict(position=pos,frame_id=ids[pos],box=torch.as_tensor(z['boxes'][j]).tolist(),score=float(z['target_scores'][j]),weight=1.))
    return dict(observations=obs,positions4=positions,anchors={'single4':anchors},new_DINO=len(obs),actual_observation_positions=sorted(p for _,p in obs))

class Spatial4Predictor(Spatial10Predictor):
    def __init__(self,model,expert,parser,direction):
        super().__init__(model,expert,parser,direction)
        self._config=config(direction)
        manifest=METHOD/'WORKING_METHOD.json'
        if manifest.exists():
            for f,h in json.loads(manifest.read_text())['code_pins'].items():
                if sha256(ROOT/f)!=h:raise RuntimeError(f'Locked A4 dependency changed: {f}')

    def _predict(self,frames,ids,metadata):
        from scripts.audit_parametric_reinsertion_v1 import full_prediction
        from vg_tta.decota_spatial_extension_v1 import parse_context
        from vg_tta.decota_tastvg_episode_v1 import fitted_merge,make_batch
        from vg_tta.foreground_runtime import state_digest
        from vg_tta.parametric_observation_v1 import ObservationReplay
        from vg_tta.simplification_partial_v1 import fit
        from vg_tta.shared_state_v1 import capture_shared
        from vg_tta.tg_spatial_tta_v1 import visual_query
        md,ed=state_digest(self.model),state_digest(self.expert.model)
        parses=dict(subject=self.parser(metadata['caption'])['subject'],context=parse_context(self.parser,metadata['caption']),old=visual_query(self.parser,metadata['caption']))
        batch=make_batch(frames,ids,metadata,parses['subject'],self.model)
        base16,_,records,_,_,views=capture_shared(self.model,batch)
        it=ObservationReplay(self.model,views,len(ids),'spatial')
        with torch.no_grad():zero=it.values()
        native=list(fitted_merge(zero['logits'],records,ids))
        ex=observations(self.expert,parses,frames,ids,native);c=self._config
        assert {n:p.numel() for n,p in it.named}==c['trainable_parameters']
        result=fit(it,records,ids,anchors=copy.deepcopy(ex['anchors']['single4']),planned=4,kappa=None,gamma=0.,lr=c['lr'],steps=c['steps'])
        assert all(torch.equal(a.cpu(),b) for a,b in zip(zero['logits'],result['final']['logits']))
        live=full_prediction(self.model,frames,ids,metadata,parses['subject'],result['state'],result['final'])
        assert live['indices']==native
        assert state_digest(self.model)==md and state_digest(self.expert.model)==ed
        return dict(method='decota_spatial4_v1',direction=self.direction,boxes=live['boxes'],indices=native,frame_ids=ids,
            I_seed=native,I_out=native,temporal_module='native_fp32_no_update',temporal_parameter_updates=False,
            adaptation=result,parameter_updates=result['state_delta']>0,expert=ex,parses=parses,
            native_boxes=zero['boxes'].cpu(),native_logits=[z.cpu() for z in zero['logits']],
            native_fp16_indices=list(base16['predicted_indices']),full_forward_audit=live['audit'],
            source_restored=True,expert_frozen=True,GT_online=False,student_output_only=True,coverage_coefficient=0,interpolation_output=False)
