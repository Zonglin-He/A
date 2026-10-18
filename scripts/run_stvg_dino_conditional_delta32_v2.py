"""Prelocked32, four original anchors, max3 student-conditioned distill rounds."""
from pathlib import Path
import os,sys,time,collections,gc,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_dino_conditional_delta32_v2'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_conditional_delta32'
ARMS=['Source_MBR','E_MBR','One_correction','Fixed_sample_distill','On_policy_distill']


def verify(closed=False):
    p=read(D/'PLAN_LOCK.json')
    if not closed:assert time.time()<p['deadline_epoch']
    for f,h in read(D/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f
    from main_method.runtime import resource_guard
    resource_guard();return p


def prepare():
    from vg_tta.stvg_dino_conditional_delta32_v2 import contract
    oldD=ROOT/'artifacts/stvg_dino_conditional_opd32_v1'
    old=read(oldD/'PLAN_LOCK.json');oldrt=read(oldD/'RUNTIME_LOCK.json');props=contract()
    import torch
    assert not torch.cuda.is_initialized()
    props['CUDA_initialized']=False;put(D/'CPU_PROPERTY_COMPLETION.json',props)
    inputs={}
    for r in old['records']:
        for name in ['prior_Source','prior_prediction']:
            load(ROOT/r[name]);inputs[r[name]]=sha(ROOT/r[name])
        f=oldD/'predictions'/(r['alias']+'.pt');load(f);inputs[str(f.relative_to(ROOT))]=sha(f)
    own=['vg_tta/stvg_dino_conditional_delta32_v2.py','scripts/run_stvg_dino_conditional_delta32_v2.py',
         'scripts/score_stvg_dino_conditional_delta32_v2.py','scripts/audit_stvg_dino_conditional_delta32_v2.py',
         'scripts/report_stvg_dino_conditional_delta32_v2.py','protocols/stvg_dino_conditional_delta32_v2.md']
    locked={k:v for k,v in old.items()if k not in ['status','created_epoch','deadline_epoch','stop']}
    locked.update(status='single_update_fix_locked_before_new_model_outputs_GT',
        distill='mu_new=mu+mean(teacher_action-sampled_action). Same population update at fixed mean for zero-mean Gaussian, different finite-sample estimator; not old empirical KL optimum.',
        one_correction='exact old one-correction output and trace reused; no new teacher or fit for this invariant baseline',
        fixed='same24 initial actions, fresh conditional feedback, one correction-difference mean update',
        dynamic='same24 epsilon split8x3, current corrected mean each round, fresh conditional feedback at changed paths',
        old_plan_SHA256=sha(oldD/'PLAN_LOCK.json'),old_feedback_and_predictions_preserved=True,
        old_target_GT_exposure_acknowledged=True,new_GT_after_all32_predictions=True,
        teacher_noop_means_zero_distribution_contribution=True,
        stop='Feedback remains scarce or accepted feedback remains harmful => stop interface. Dynamic must beat E and fixed to motivate more work; no automatic expansion or parameter change.',
        created_epoch=time.time(),deadline_epoch=time.time()+2400)
    put(D/'PLAN_LOCK.json',locked)
    pins={**oldrt['pins'],**inputs,**{f:sha(ROOT/f)for f in own},
          'artifacts/stvg_dino_conditional_opd32_v1/PLAN_LOCK.json':sha(oldD/'PLAN_LOCK.json')}
    put(D/'RUNTIME_LOCK.json',dict(pins=pins,plan_sha256=sha(D/'PLAN_LOCK.json')))
    put(D/'STATUS.json',dict(status='prepared',count=0,total=32,GT_read=False),True)


def measure():
    import numpy as np,torch
    from scripts.run_decota_paper_main_v1 import gpu,read_row,frames_for
    from methods.decota_final_simplified_v1.observations import SpatialExpert,target_tokens
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_error_state_correction_v1 import encode,decode
    from vg_tta.stvg_dino_conditional_delta32_v2 import ConditionalDecoder,conditional_targets,distill
    p=verify();lease=gpu();start=time.perf_counter();cache=collections.OrderedDict();files=[]
    try:
        expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);mh=state_hash(expert.model.state_dict())
        for index,r in enumerate(p['records']):
            verify();tick=time.perf_counter();s=load(ROOT/r['prior_Source']);e=load(ROOT/r['prior_prediction']);b=np.asarray(e['outputs']['Source_no_correction'],np.float64)
            row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,cache);assert list(ids)==list(s['frame_ids'])
            positions=e['planned_positions'];assert positions==s['expert']['positions4'];active=e['estimate']['support_frames']>0
            mu=np.asarray(e['estimate']['delta'])/np.asarray(p['scale'][r['dataset']]);scale=np.asarray(p['scale'][r['dataset']])
            outputs={'Source_MBR':b,'E_MBR':np.asarray(e['outputs']['E_DINO_support'])};fits={};caches=[];quals=[]
            old=load(ROOT/'artifacts/stvg_dino_conditional_opd32_v1/predictions'/(r['alias']+'.pt'))
            outputs['One_correction']=old['outputs']['One_correction'].copy();fits['One_correction']={**old['fits']['One_correction'],'reused_from_v1':True}
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
                for arm in ARMS[3:]:
                    m=mu.copy();rounds=[];at=time.perf_counter()
                    chunks=[np.repeat(mu[None],24,axis=0)]if arm=='One_correction'else[mu+eps]if arm=='Fixed_sample_distill'else list(np.split(eps,3))
                    for j,a in enumerate(chunks):
                        actions=m+a if arm=='On_policy_distill'else a
                        targets,detail=conditional_targets(caches,b,actions,scale,positions)
                        if arm=='Fixed_sample_distill' or j==0:
                            prior_step=old['fits'][arm]['rounds'][j]
                            assert np.array_equal(actions,prior_step['actions'])and np.array_equal(targets,prior_step['targets']),'initial unchanged teacher path must be exact'
                        m,loss=distill(m,targets,actions);rounds.append(dict(round=j,actions=actions,targets=targets,details=detail,**loss))
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
