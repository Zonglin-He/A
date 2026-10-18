"""Matched original32 content-only conditional-DINO validation, no stage1 fit."""
from pathlib import Path
import os,sys,time,hashlib,collections,gc,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_dino_content_match32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_content_match32'
PRE=ROOT/'artifacts/stvg_dino_conditional_delta32_v2'


def verify(live=False):
    p=read(D/'PLAN_LOCK.json')
    if live:assert time.time()<p['deadline_epoch']
    for f,h in read(D/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f
    from main_method.runtime import resource_guard
    resource_guard();return p


def prepare():
    old=read(PRE/'PLAN_LOCK.json');assert read(PRE/'FINAL_COMPLETION.json')['count']==32
    own=['vg_tta/stvg_dino_content_match32_v1.py','scripts/run_stvg_dino_content_match32_v1.py','scripts/score_stvg_dino_content_match32_v1.py','scripts/audit_stvg_dino_content_match32_v1.py','scripts/report_stvg_dino_content_match32_v1.py','protocols/stvg_dino_content_match32_v1.md']
    deps=['vg_tta/stvg_dino_conditional_opd32_v1.py','vg_tta/stvg_dino_conditional_delta32_v2.py','vg_tta/stvg_error_state_correction_v1.py','methods/decota_final_simplified_v1/observations.py','methods/CURRENT_METHOD.json',
          '.conda/tubedetr/lib/python3.11/site-packages/transformers/models/grounding_dino/modeling_grounding_dino.py',
          '.cache/grounding_dino_http_v1/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/config.json']
    for r in old['records']:deps.extend([r['prior_Source'],r['prior_prediction'],str((PRE/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))])
    p=dict(records=old['records'],scale=old['scale'],seed=old['seed'],count=32,per_domain=16,threshold=.35,
        references='exact original repaired fixed24 actions, Source trajectories and FP32 anchor boxes',
        content='nearest initial native proposal versus first argmax native final mean sigmoid target-token score',
        mask='identical geometry-nearest exclusion for both conditions; only appended content changes',
        feedback_gate='both domains: more accepted anchors AND positive known-anchor gated IoU correction AND better gated correction than nearest',
        query_unit=True,old_target_exposure=True,no_stage1_fit=True,no_parameter_search=True,
        weights_frozen=True,GT_after_all32_sealed=True,created_epoch=time.time(),deadline_epoch=time.time()+2400,GPU_seconds_cap=900,
        conditional_OPD=dict(only_after_gate=True,rounds=3,samples_per_round=8,fixed_samples=24,
            update='mu+mean(y-a)',same_E_scale_seed_threshold=True,GPU_seconds_cap=900,wall_seconds_cap=2400))
    put(D/'PLAN_LOCK.json',p);put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in own+deps},plan_sha256=sha(D/'PLAN_LOCK.json')))
    put(D/'STATUS.json',dict(status='prepared',count=0,total=32,GT_read=False),True)


def measure():
    import numpy as np,torch
    from scripts.run_decota_paper_main_v1 import gpu,read_row,frames_for
    from methods.decota_final_simplified_v1.observations import SpatialExpert,target_tokens
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_dino_content_match32_v1 import ContentDecoder,matched_targets
    p=verify(True);lease=gpu();started=time.perf_counter();cache=collections.OrderedDict();files=[]
    try:
        expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);mh=state_hash(expert.model.state_dict())
        for ix,r in enumerate(p['records']):
            verify(True);tick=time.perf_counter();s=load(ROOT/r['prior_Source']);e=load(ROOT/r['prior_prediction']);old=load(PRE/'predictions'/(r['alias']+'.pt'))
            source=np.asarray(e['outputs']['Source_no_correction'],np.float64);positions=e['planned_positions'];active=old['active'];caches=[];quals=[]
            controls=[];actions=np.empty((0,4));matched={m:dict(targets=actions.copy(),details=[])for m in ['nearest','target_high']};maxerr=0.;scoreerr=0.;anchorchecks=0
            if active:
                row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,cache);assert list(ids)==list(s['frame_ids'])
                for pos in positions:
                    rc=s['expert']['observations'][('original',pos)]['receipt'];assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==rc['rgb_sha256']
                    text=rc['text'];inp=expert.processor(images=frames[pos],text=text,return_tensors='pt').to('cuda');offsets=expert.processor.tokenizer(text,return_offsets_mapping=True)['offset_mapping']
                    if rc['context_active']:ent=target_tokens(text,offsets,rc['target_span'])
                    else:
                        entity=row['parses']['old']['entity'];a=text.find(entity);assert a>=0;ent=target_tokens(text,offsets,[a,a+len(entity)])
                    c=ContentDecoder(expert.model,inp,ent);caches.append(c)
                    if ix<2:quals.append(c.qualify_content())
                step=old['fits']['Fixed_sample_distill']['rounds'][0];actions=step['actions'].copy()
                matched,controls=matched_targets(caches,source,actions,np.array(p['scale'][r['dataset']]),positions)
                for a,b in zip(matched['nearest']['details'],step['details']):
                    assert a['legal']==b['legal'] and a['accepted_anchors']==b['accepted_anchors']
                    for pos,v in a['anchors'].items():
                        ov=b['anchors'][pos];assert v['slot']==ov['slot'] and v['accepted']==ov['accepted']
                        maxerr=max(maxerr,float(abs(v['box']-ov['box']).max()));scoreerr=max(scoreerr,abs(v['score']-ov['score']));anchorchecks+=1
                assert maxerr<1e-5 and scoreerr<1e-5,(maxerr,scoreerr)
                assert np.max(abs(matched['nearest']['targets']-step['targets']))<1e-5
            assert state_hash(expert.model.state_dict())==mh
            cost=dict(case_seconds=time.perf_counter()-tick,encoding_frames=len(caches),encoding_native_forward_seconds=sum(c.encoding_full_forward_seconds for c in caches),
                conditional_decoder_calls=sum(c.decoder_calls for c in caches),conditional_decoder_seconds=sum(c.decoder_seconds for c in caches))
            payload=dict(alias=r['alias'],dataset=r['dataset'],active=active,actions=actions,matched=matched,controls=controls,planned_positions=positions,frame_ids=s['frame_ids'],
                interval=e['interval'],GT_read=False,pretrained_weights_frozen_exact=True,qualifications=quals,compute=cost,
                nearest_prior_checks=dict(anchors=anchorchecks,max_box_error=maxerr,max_score_error=scoreerr))
            files.append(save(D/'predictions'/(r['alias']+'.pt'),payload));put(D/'STATUS.json',dict(status='running',count=ix+1,total=32,GT_read=False),True)
            print(r['alias'],ix+1,round(time.perf_counter()-started,3),flush=True)
            if ix==1:put(D/'PILOT2_TIMING.json',dict(status='pass',count=2,one_per_domain=True,seconds=time.perf_counter()-started,qualifications=quals))
            del caches,payload;gc.collect();torch.cuda.empty_cache();assert time.perf_counter()-started<p['GPU_seconds_cap']
        verify(True);put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,all_outputs_before_GT=True,GPU_seconds=time.perf_counter()-started,pretrained_state_sha256=mh))
    finally:lease.close()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','measure']);args=a.parse_args()
    try:globals()[args.stage]()
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',error=str(e),type=type(e).__name__,partials_preserved=True));raise
