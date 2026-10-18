"""Original32 paired attention exclusion, actual one-step update, layer traces."""
from pathlib import Path
import sys,os,time,hashlib,collections,gc,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_dino_mask_align32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_mask_align32'
PRE=ROOT/'artifacts/stvg_dino_content_match32_v1'
ARMS=['Source_MBR','E_MBR','A_unaligned_once','B_aligned_once']


def verify(live=False):
    p=read(D/'PLAN_LOCK.json')
    if live:assert time.time()<p['deadline_epoch']
    for f,h in read(D/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f
    from main_method.runtime import resource_guard
    resource_guard();return p


def prepare():
    from vg_tta.stvg_dino_mask_align32_v1 import contract
    old=read(PRE/'PLAN_LOCK.json');props=contract();put(D/'CPU_PROPERTY_COMPLETION.json',props)
    files=['vg_tta/stvg_dino_mask_align32_v1.py','scripts/run_stvg_dino_mask_align32_v1.py','scripts/score_stvg_dino_mask_align32_v1.py','scripts/audit_stvg_dino_mask_align32_v1.py','scripts/report_stvg_dino_mask_align32_v1.py','protocols/stvg_dino_mask_align32_v1.md',
        'vg_tta/stvg_dino_content_match32_v1.py','vg_tta/stvg_dino_conditional_opd32_v1.py','methods/decota_final_simplified_v1/observations.py','methods/CURRENT_METHOD.json',
        '.conda/tubedetr/lib/python3.11/site-packages/transformers/models/grounding_dino/modeling_grounding_dino.py']
    for r in old['records']:files.extend([r['prior_Source'],r['prior_prediction'],str((PRE/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))])
    put(D/'PLAN_LOCK.json',dict(records=old['records'],scale=old['scale'],seed=old['seed'],count=32,per_domain=16,threshold=.35,
        changed_factor='only high-content query native-attention exclusion: geometry-nearest A versus content-source B',
        fixed_reference_and_content=True,native_initial_and_six_layers_saved=True,one_update_denominator=24,MBR_unchanged=True,
        continued_dynamic='only if B coverage>A and B once vIoU>E in both domains; same32 3x8, each sum/24',
        no_target_tuning=True,no_new_GT_before_all32_seal=True,old_target_development_exposure=True,
        created_epoch=time.time(),deadline_epoch=time.time()+2400,GPU_seconds_cap=900))
    put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in files},plan_sha256=sha(D/'PLAN_LOCK.json')))
    put(D/'STATUS.json',dict(status='prepared',count=0,total=32,GT_read=False),True)


def run(dynamic=False):
    import numpy as np,torch
    from scripts.run_decota_paper_main_v1 import gpu,read_row,frames_for
    from methods.decota_final_simplified_v1.observations import SpatialExpert,target_tokens
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_error_state_correction_v1 import encode,decode
    from vg_tta.stvg_dino_mask_align32_v1 import MaskDecoder,feedback,update
    p=verify(True);outdir=D/'dynamic'if dynamic else D
    if dynamic:
        assert read(P/'DECISION.json')['pass_for_dynamic'];assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    oldgpu=read(D/'GLOBAL_PREDICTION_BARRIER.json')['GPU_seconds']if dynamic else 0.
    lease=gpu();tick=time.perf_counter();framecache=collections.OrderedDict();files=[]
    try:
        expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);mh=state_hash(expert.model.state_dict())
        for ix,r in enumerate(p['records']):
            verify(True);start=time.perf_counter();s=load(ROOT/r['prior_Source']);e=load(ROOT/r['prior_prediction']);old=load(PRE/'predictions'/(r['alias']+'.pt'));source=np.asarray(e['outputs']['Source_no_correction'],float);E=np.asarray(e['outputs']['E_DINO_support'],float)
            scale=np.array(p['scale'][r['dataset']]);mu=np.asarray(e['estimate']['delta'])/scale;positions=e['planned_positions'];active=old['active'];caches=[];native={};quals=[];traces={};fits={};matched={};actions=np.empty((0,4));priorchecks=0;maxerr=0.
            outputs=dict(Source_MBR=source,E_MBR=E)
            if active:
                row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,framecache);assert list(ids)==list(s['frame_ids'])
                for pos in positions:
                    rc=s['expert']['observations'][('original',pos)]['receipt'];assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==rc['rgb_sha256']
                    text=rc['text'];inp=expert.processor(images=frames[pos],text=text,return_tensors='pt').to('cuda');offsets=expert.processor.tokenizer(text,return_offsets_mapping=True)['offset_mapping']
                    if rc['context_active']:ent=target_tokens(text,offsets,rc['target_span'])
                    else:
                        entity=row['parses']['old']['entity'];lo=text.find(entity);assert lo>=0;ent=target_tokens(text,offsets,[lo,lo+len(entity)])
                    c=MaskDecoder(expert.model,inp,ent);caches.append(c);native[pos]=c.native_trace()
                    if ix<2:quals.append(c.qualify_mask())
                if not dynamic:
                    actions=old['actions'].copy();matched,traces=feedback(caches,source,actions,scale,positions)
                    for a,b in zip(matched['A_unaligned']['details'],old['matched']['target_high']['details']):
                        assert a['legal']==b['legal']and a['accepted_anchors']==b['accepted_anchors']
                        for pos,v in a['anchors'].items():
                            prev=b['anchors'][pos];assert v['slot']==prev['slot']and v['mask_slot']==prev['mask_slot']and v['accepted']==prev['accepted']
                            maxerr=max(maxerr,float(abs(v['box']-prev['box']).max()),abs(v['score']-prev['score']));priorchecks+=1
                    assert maxerr<1e-5,maxerr
                    for m,f in matched.items():
                        new,info=update(mu,f['targets'],actions);fits[m]=info;outputs[m+'_once']=E.copy()if info['all_identity_exact_noop']else decode(encode(source)+new*scale)
                else:
                    rng=np.random.default_rng(p['seed']+int(hashlib.sha256(r['alias'].encode()).hexdigest()[:8],16));eps=rng.standard_normal((24,4));assert np.array_equal(mu+eps,old['actions']);new=mu.copy();rounds=[]
                    for j,chunk in enumerate(np.split(eps,3)):
                        a=new+chunk;res,tr=feedback(caches,source,a,scale,positions,paired=False);f=res['B_aligned'];new,info=update(new,f['targets'],a,24)
                        rounds.append(dict(round=j,actions=a,targets=f['targets'],details=f['details'],traces=tr,**info))
                    outputs['B_aligned_dynamic']=E.copy()if all(x['all_identity_exact_noop']for x in rounds)else decode(encode(source)+new*scale);fits['B_aligned_dynamic']=dict(rounds=rounds,mean_after=new)
            else:
                for name in (['B_aligned_dynamic']if dynamic else ['A_unaligned_once','B_aligned_once']):outputs[name]=E.copy()
            assert state_hash(expert.model.state_dict())==mh
            cost=dict(case_seconds=time.perf_counter()-start,encoding_frames=len(caches),encoding_native_forward_seconds=sum(c.encoding_full_forward_seconds for c in caches),decoder_calls=sum(c.decoder_calls for c in caches),decoder_seconds=sum(c.decoder_seconds for c in caches))
            obj=dict(alias=r['alias'],dataset=r['dataset'],frame_ids=s['frame_ids'],planned_positions=positions,interval=e['interval'],active=active,outputs=outputs,actions=actions,matched=matched,traces=traces,native=native,fits=fits,
                initial_mean=mu,scale=scale,qualifications=quals,compute=cost,GT_read=False,pretrained_weights_frozen_exact=True,prior_A_checks=priorchecks,prior_A_max_error=maxerr)
            files.append(save(outdir/'predictions'/(r['alias']+'.pt'),obj));put(outdir/'STATUS.json',dict(status='running',count=ix+1,total=32,GT_read=False),True)
            if ix==1:put(outdir/'PILOT2_TIMING.json',dict(status='pass',count=2,one_per_domain=True,seconds=time.perf_counter()-tick,qualification=quals))
            print(r['alias'],ix+1,round(time.perf_counter()-tick,3),flush=True)
            del caches,obj;gc.collect();torch.cuda.empty_cache();assert oldgpu+time.perf_counter()-tick<p['GPU_seconds_cap']
        put(outdir/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all_outputs_before_GT=True,GT_read=False,GPU_seconds=time.perf_counter()-tick,pretrained_state_sha256=mh))
    finally:lease.close()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','measure','dynamic']);args=a.parse_args()
    try:prepare()if args.stage=='prepare'else run(args.stage=='dynamic')
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',error=str(e),type=type(e).__name__,partials_preserved=True));raise
