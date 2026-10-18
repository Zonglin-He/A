"""Prelocked32, four original anchors, max3 student-conditioned distill rounds."""
from pathlib import Path
import os,sys,time,collections,gc,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_dino_conditional_opd32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_conditional32'
ARMS=['Source_MBR','E_MBR','One_correction','Fixed_sample_distill','On_policy_distill']


def verify(closed=False):
    p=read(D/'PLAN_LOCK.json')
    if not closed:assert time.time()<p['deadline_epoch']
    for f,h in read(D/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f
    from main_method.runtime import resource_guard
    resource_guard();return p


def prepare():
    p=read(ROOT/'artifacts/stvg_query_e_opd64_v1/PLAN_LOCK.json');records=[]
    for ds in ['hc2','vidstg']:records.extend(sorted([r for r in p['records']if r['dataset']==ds],key=lambda r:r['alias'])[:16])
    # First two actual timing cases are one/domain without changing the cohort.
    records=records[:1]+records[16:17]+records[1:16]+records[17:]
    own=['vg_tta/stvg_dino_conditional_opd32_v1.py','scripts/run_stvg_dino_conditional_opd32_v1.py','protocols/stvg_dino_conditional_opd32_v1.md']
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256
    assert sha(ROOT/EXPERT_SNAPSHOT/'model.safetensors')==EXPERT_SHA256
    inputs={}
    for r in records:
        for f in ['prior_Source','prior_prediction']:
            load(ROOT/r[f]);inputs[r[f]]=sha(ROOT/r[f])
    c=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')
    scales={ds:c['source_domains']['vidstg'if ds=='hc2'else'hc2']['scale']for ds in ['hc2','vidstg']}
    put(D/'PLAN_LOCK.json',dict(status='locked_before_model_and_new_GT',records=records,count=32,per_domain=16,
        selection='lexicographically first16 aliases/domain of prior fixed64; no scores/GT/quality filter',
        seed=20261018,rounds=3,samples_per_round=8,total_samples=24,scale=scales,distribution='unit-covariance Gaussian over source-normalized shared4 correction, mean initialized E/scale',
        distill='exact minimizer of average forward KL from unit Gaussians at teacher-corrected actions; detached targets; no optimizer/LR search',
        teacher_target='mean across anchors with original target-token score>=.35 of corrected box minus original Source in(cx,cy,logw,logh), divided by source scale',
        invalid='any cxcywh coordinate outside open(0,1) at any anchor => entire action teacher no-op; no clamp or rejection resampling',
        no_E_support='all new arms exact E no-op, no teacher encoding; retained in denominator',
        content='nearest initial native proposal by IoU, deterministic first argmax; its original content embedding',
        attention='900 original queries unchanged context; each external replaces matched native slot in its own attention support; external-external isolated',
        one_correction='24 identical E references logically; deduplicate identical decoder work; exact shared4 correction from conditional outputs',
        fixed='24 initial-distribution candidates once; one exact Gaussian mean distillation',
        dynamic='same24 epsilon draws split8x3; teacher recomputed around updated mean, exact distill each round',
        teacher_budget_per_arm='24 logical reference queries per encoded anchor, <=4 anchors; unique decoder work separately reported',
        native_replay_tolerance=0.,cloned_reference_box_tolerance=1e-4,
        time='same frozen final physical distribution MBR interval in all arms; no temporal update',pretrained_weights_frozen=True,
        GT='only after all32 outputs sealed; old exposed development cohort, no new target selection',
        GPU_seconds_cap=900,wall_seconds_cap=2400,created_epoch=time.time(),deadline_epoch=time.time()+2400,
        stop='interface invalid => stop; if one-correction mean below E in either domain, stop interface/no LR rescue; dynamic must beat E and fixed both domains to merit expansion',
        CURRENT_unchanged=True,second_backbone=False,new_training=False))
    hf=ROOT/'.conda/tubedetr/lib/python3.11/site-packages/transformers/models/grounding_dino/modeling_grounding_dino.py'
    pins={**inputs,**{f:sha(ROOT/f)for f in own},str(hf.relative_to(ROOT)):sha(hf),str((ROOT/EXPERT_SNAPSHOT/'model.safetensors').relative_to(ROOT)):EXPERT_SHA256,
          'methods/CURRENT_METHOD.json':sha(ROOT/'methods/CURRENT_METHOD.json')}
    put(D/'RUNTIME_LOCK.json',dict(pins=pins,plan_sha256=sha(D/'PLAN_LOCK.json')))
    put(D/'STATUS.json',dict(status='prepared',count=0,total=32,GT_read=False),True)


def measure():
    import numpy as np,torch
    from scripts.run_decota_paper_main_v1 import gpu,read_row,frames_for
    from methods.decota_final_simplified_v1.observations import SpatialExpert,target_tokens
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_error_state_correction_v1 import encode,decode
    from vg_tta.stvg_dino_conditional_opd32_v1 import ConditionalDecoder,conditional_targets,distill
    p=verify();lease=gpu();start=time.perf_counter();cache=collections.OrderedDict();files=[]
    try:
        expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);mh=state_hash(expert.model.state_dict())
        for index,r in enumerate(p['records']):
            verify();tick=time.perf_counter();s=load(ROOT/r['prior_Source']);e=load(ROOT/r['prior_prediction']);b=np.asarray(e['outputs']['Source_no_correction'],np.float64)
            row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,cache);assert list(ids)==list(s['frame_ids'])
            positions=e['planned_positions'];assert positions==s['expert']['positions4'];active=e['estimate']['support_frames']>0
            mu=np.asarray(e['estimate']['delta'])/np.asarray(p['scale'][r['dataset']]);scale=np.asarray(p['scale'][r['dataset']])
            outputs={'Source_MBR':b,'E_MBR':np.asarray(e['outputs']['E_DINO_support'])};fits={};caches=[];quals=[]
            if active:
                for pos in positions:
                    obs=s['expert']['observations'][('original',pos)];rc=obs['receipt'];assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==rc['rgb_sha256']
                    text=rc['text'];inp=expert.processor(images=frames[pos],text=text,return_tensors='pt').to('cuda')
                    offsets=expert.processor.tokenizer(text,return_offsets_mapping=True)['offset_mapping']
                    if rc['context_active']:ent=target_tokens(text,offsets,rc['target_span'])
                    else:
                        entity=row['parses']['old']['entity'];ent=[j for j,(lo,hi)in enumerate(offsets)if hi>lo and entity in text[lo:hi]]
                        if not ent:
                            a=text.find(entity);assert a>=0;ent=target_tokens(text,offsets,[a,a+len(entity)])
                    decoder=ConditionalDecoder(expert.model,inp,ent);caches.append(decoder)
                    if index<2:quals.append(decoder.qualify())
                rng=np.random.default_rng(p['seed']+int(hashlib.sha256(r['alias'].encode()).hexdigest()[:8],16));eps=rng.standard_normal((24,4))
                for arm in ARMS[2:]:
                    m=mu.copy();rounds=[];at=time.perf_counter()
                    chunks=[np.repeat(mu[None],24,axis=0)]if arm=='One_correction'else[mu+eps]if arm=='Fixed_sample_distill'else list(np.split(eps,3))
                    for j,a in enumerate(chunks):
                        actions=m+a if arm=='On_policy_distill'else a
                        targets,detail=conditional_targets(caches,b,actions,scale,positions)
                        m,loss=distill(m,targets);rounds.append(dict(round=j,actions=actions,targets=targets,details=detail,**loss))
                    outputs[arm]=decode(encode(b)+m*scale)
                    fits[arm]=dict(rounds=rounds,mean=m,seconds=time.perf_counter()-at,logical_teacher_queries=24*len(positions))
            else:
                for a in ARMS[2:]:outputs[a]=outputs['E_MBR'].copy();fits[a]=dict(rounds=[],mean=mu.copy(),seconds=0.,logical_teacher_queries=0)
            assert state_hash(expert.model.state_dict())==mh
            compute=dict(case_seconds=time.perf_counter()-tick,full_image_text_encodings=len(caches),
                encoding_full_forward_seconds=sum(c.encoding_full_forward_seconds for c in caches),
                conditional_decoder_seconds=sum(c.decoder_seconds for c in caches),conditional_decoder_calls=sum(c.decoder_calls for c in caches),
                logical_candidate_queries=sum(c.logical_queries for c in caches),unique_candidate_queries=sum(c.unique_queries for c in caches),
                new_Source_calls=0,new_DINO_image_frames=len(caches),GPU_seconds_so_far=time.perf_counter()-start)
            out=dict(alias=r['alias'],dataset=r['dataset'],frame_ids=ids,planned_positions=positions,interval=e['interval'],outputs=outputs,fits=fits,
                active=active,qualifications=quals,compute=compute,GT_read=False,pretrained_weights_frozen_exact=True)
            files.append(save(D/'predictions'/(r['alias']+'.pt'),out));put(D/'STATUS.json',dict(status='running',stage='conditional_decoder_and_distill',count=index+1,total=32,GT_read=False),True)
            if index==1:put(D/'PILOT2_TIMING.json',dict(status='pass',count=2,one_per_domain=True,seconds=time.perf_counter()-start,qualification=quals,original_budget_not_reset=True))
            print(r['alias'],index+1,round(time.perf_counter()-start,3),flush=True)
            del caches,out,frames;gc.collect();torch.cuda.empty_cache()
            assert time.perf_counter()-start<p['GPU_seconds_cap']
        verify();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all_outputs_before_GT=True,GT_read=False,
            GPU_seconds=time.perf_counter()-start,pretrained_state_sha256=mh))
    finally:lease.close()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','measure']);arg=a.parse_args()
    try:globals()[arg.stage]()
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',error=str(e),type=type(e).__name__,partials_preserved=True));raise
