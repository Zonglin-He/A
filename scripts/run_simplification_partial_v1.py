"""F36 bounded single GPU worker. A/T contain no label or evaluation imports."""
import argparse
import copy
import fcntl
import gc
import hashlib
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
from scripts.run_parametric_observation_v1 import cpu_tree,rank
from scripts.run_spatial10_components_v1 import (parent,observations,capture_timed,existing)
from scripts.prepare_simplification_partial_v1 import OUT,PARENT

OWN=['scripts/run_simplification_partial_v1.py','vg_tta/simplification_partial_v1.py']


def plan():
    p=read(OUT/'LOCK.json')
    for f,h in p['protected_pins'].items(): assert sha(ROOT/f)==h, ('changed_parent',f)
    return p


class Budget:
    def __init__(self,p):
        self.caps=p['caps']; self.file=OUT/'COUNTERS.json'
        self.counts=read(self.file) if self.file.exists() else {k:0 for k in self.caps}
    def charge(self,k,n):
        assert self.counts[k]+n<=self.caps[k],('budget_exceeded',k,self.counts)
        self.counts[k]+=n;status(self.file,self.counts)
    def reserve(self,**needs):
        assert all(self.counts[k]+v<=self.caps[k] for k,v in needs.items()),('insufficient_complete_block_budget',needs)


def dest(stage,key): return OUT/stage/(key.replace(':','_')+'.pt')


def record(p,x):
    save(p,cpu_tree(x));write(p.with_suffix('.json'),dict(key=x['key'],sha256=sha(p),completed=time.time(),
            code_pins={f:sha(ROOT/f) for f in OWN}))


def canonical(z):
    z=copy.deepcopy(z)
    z['final']['logits']=z['final']['logits'][:2]
    z['state']={n:p for n,p in z['state'].items() if n.startswith('spatial.')}
    return z


def parents(r):
    old,path,h=parent(r)
    assert old['fits']['S_PRIVATE10']['restore_exact']
    return old,dict(path=path,sha256=h),canonical(old['fits']['S_PRIVATE10'])


def verify_original(zero,qa,old,records,ids):
    import torch
    from vg_tta.decota_tastvg_episode_v1 import fitted_merge
    assert qa['rgb_sha256']==old['qa'][0]['rgb_sha256']
    assert qa['input_tensor_sha256']==old['qa'][0]['input_tensor_sha256']
    assert torch.equal(zero['boxes'].cpu(),old['native_boxes'])
    assert list(fitted_merge(zero['logits'],records,ids))==old['native_indices']
    assert all(torch.equal(a.cpu(),b) for a,b in zip(zero['logits'],old['fits']['S_PRIVATE10']['final']['logits'][:2]))


def space_one(model,expert,r,c,budget,smoke=False,repair=False):
    import torch
    from vg_tta.parametric_observation_v1 import ObservationReplay
    from vg_tta.simplification_partial_v1 import fit
    from vg_tta.decota_tastvg_episode_v1 import fitted_merge
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    from methods.decota_refine_uniform_v1.api import reconstruct
    budget.reserve(fits=7 if smoke else 5,backward=60,new_DINO=16)
    frames,base,records,views,qa,cost=capture_timed(model,r)
    ids=r['input']['frame_ids'];old,pinfo,a0=parents(r)
    it=ObservationReplay(model,views,len(ids),'spatial')
    with torch.no_grad(): zero=it.values()
    verify_original(zero,qa,old,records,ids)
    native=list(fitted_merge(zero['logits'],records,ids))
    # Original-eight dev observations live in F34 observe; exact receipt checks remain mandatory.
    if 'development' in r['roles']:
        oldex=parent(r,'observe')[0]['expert']
    else: oldex=old['expert']
    ex=observations(expert,r,frames,ids,native,oldex,True,budget)
    assert ex['anchors']['weak']==a0['anchors']
    kp=None if c=='hcstvg1_test' else 2.
    lr=.01 if c=='hcstvg1_test' else .1
    fits={'A0':a0};reuse={'A0':pinfo};audits={}
    audits['parent']=full_prediction(model,frames,ids,r['input'],r['subject'],a0['state'],a0['final'])['audit']
    f35path=PARENT/'spatial'/(r['key'].replace(':','_')+'.pt')
    if existing(f35path):
        f35=load(f35path)
        if 'D_gamma0' in f35['fits']:
            fits['A1']=f35['fits']['D_gamma0']
            assert fits['A1']['anchors']==ex['anchors']['weak']
            reuse['A1']=dict(path=str(f35path),sha256=sha(f35path),arm='D_gamma0')
    ones=copy.deepcopy(ex['anchors']['weak'])
    for a in ones:a['weight']=1.
    specs=[('A1',ex['anchors']['weak'],4,kp),('A2',ones,4,kp),('A3',ones,4,None),
           ('A4',ex['anchors']['single4'],4,None),('A5',ex['anchors']['single8'],8,None)]
    for name,anchors,K,kappa in specs:
        if name in fits:continue
        if name=='A3' and kp is None:
            fits[name]=fits['A2'];reuse[name]=dict(alias='A2',reason='HC rho already raw');continue
        z=fit(it,records,ids,anchors=anchors,planned=K,kappa=kappa,lr=lr,steps=10,charge=budget.charge)
        assert all(torch.equal(a,b.cpu()) for a,b in zip(z['final']['logits'],zero['logits']))
        assert z['final']['indices']==native
        audits[name]=full_prediction(model,frames,ids,r['input'],r['subject'],z['state'],z['final'])['audit']
        fits[name]=z
        print('F36 A',r['key'],name,'best',z['best_step'],'seconds',round(z['seconds'],2),flush=True)
    if smoke:
        for name,kwargs in [('steps0',dict(steps=0,lr=lr)),('lr0',dict(steps=1,lr=0.))]:
            z=fit(it,records,ids,anchors=ex['anchors']['single4'],planned=4,charge=budget.charge,**kwargs)
            assert z['state_delta']==0 and torch.equal(z['final']['boxes'],zero['boxes'].cpu())
            audits[name]=full_prediction(model,frames,ids,r['input'],r['subject'],z['state'],z['final'])['audit']
            record(dest('noops',r['key']+'_space_'+name),dict(key=r['key'],fit=z,audit=audits[name]))
    controls={'Frozen':dict(boxes=zero['boxes'].cpu(),indices=native)}
    for name,z in fits.items():
        aa=z['anchors']
        for mode in ('direct','absolute'):
            bb,a=reconstruct(zero['boxes'].cpu(),aa,ids,mode)
            controls[name+'_'+mode]=dict(boxes=bb,indices=native,audit=a)
    result=dict(key=r['key'],cohort=c,role=r['f36_role'],source=r['source'],group=r['group'],
        native_boxes=zero['boxes'].cpu(),native_indices=native,native_logits=[z.cpu() for z in zero['logits']],
        records=records,frame_ids=ids,qa=qa,cost=cost,expert=ex,fits=fits,reuse=reuse,
        controls=controls,audits=audits,GT_online=False,parent=pinfo)
    record(dest('space',r['key']),result)


def crop_prediction(model,frames,ids,metadata,subject,budget):
    """One complete two-offset frozen prediction; FP32 suffix exactly as parent."""
    import torch
    from vg_tta.closure_replay_v1 import floating32
    from vg_tta.shared_state_v1 import capture_shared
    from vg_tta.decota_tastvg_episode_v1 import make_batch
    from vg_tta.parametric_observation_v1 import tensor_hash
    contexts=[]
    def norm_before(mod,args):
        ctx=torch.autocast('cuda',enabled=False);ctx.__enter__();contexts.append(ctx)
        return floating32(args)
    def ground_before(mod,args,kw):return floating32(args),floating32(kw)
    def model_after(mod,args,out):contexts.pop().__exit__(None,None,None)
    hooks=[model.ground_encoder.encoder.norm.register_forward_pre_hook(norm_before),
           model.ground_decoder.register_forward_pre_hook(ground_before,with_kwargs=True),
           model.register_forward_hook(model_after)]
    budget.charge('crop_teacher',1);tick=time.perf_counter()
    try:
        batch=make_batch(frames,ids,metadata,subject,model)
        base,_,records,_,_,views=capture_shared(model,batch)
        indices=list(base['predicted_indices'])
        qa=dict(rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
            input_sha256=tensor_hash(batch['videos'].tensors),caption=metadata['caption'],subject=subject,
            frame_ids=ids,placeholder_zero=not bool(batch['targets'][0]['actioness'].any()))
        assert qa['placeholder_zero'] and len(views)==2
        torch.cuda.synchronize()
        return dict(indices=indices,interval=[ids[indices[0]],ids[indices[1]]+1],
                    logits=[v['output']['pred_sted'].cpu() for v in views],records=records,qa=qa,
                    seconds=time.perf_counter()-tick,complete_predictions=1,offset_forwards=2,
                    suffix_precision='FP32',GT_online=False)
    finally:
        for h in hooks:h.remove()
        while contexts:contexts.pop().__exit__(None,None,None)


def temporal_one(model,r,c,budget,phase,smoke=False):
    import numpy as np
    import torch
    from vg_tta.parametric_observation_v1 import ObservationReplay,posterior_decode
    from vg_tta.simplification_partial_v1 import (FullInputHeadReplay,crop_windows,observation,
        make_supervision,fit,convex_output)
    from vg_tta.time_space_repair_v1 import legal_logp
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    budget.reserve(fits=11 if smoke else 9,backward=47 if smoke else 45,crop_teacher=4)
    frames,base,records,views,qa,cost=capture_timed(model,r)
    ids=r['input']['frame_ids'];old,pinfo,a0=parents(r)
    replay=ObservationReplay(model,views,len(ids),'head');it=FullInputHeadReplay(replay)
    with torch.no_grad():zero=it.values()
    verify_original(zero,qa,old,records,ids)
    native=old['native_indices'];physical=[ids[native[0]],ids[native[1]]+1]
    crop_path=dest('teachers',r['key'])
    if existing(crop_path):
        tc=load(crop_path);assert tc['qa']==qa and tc['native_interval']==physical
    else:
        windows=crop_windows(ids,native,[r['input']['start_frame'],r['input']['end_frame']])
        preds=[]
        for w in windows['windows']:
            pos=w['positions'];rgb=np.ascontiguousarray(frames[pos]);ci=w['frame_ids']
            correct=crop_prediction(model,rgb,ci,r['input'],r['subject'],budget)
            wrongmeta={**r['input'],'caption':r['wrong_query']['caption']}
            wrong=crop_prediction(model,rgb,ci,wrongmeta,r['wrong_query']['subject'],budget)
            assert correct['qa']['rgb_sha256']==wrong['qa']['rgb_sha256']
            assert correct['qa']['input_sha256']==wrong['qa']['input_sha256']
            preds.append(dict(window=w,correct=correct,wrong=wrong,donor=r['wrong_query']))
        tc=dict(key=r['key'],windows=windows,predictions=preds,qa=qa,native_interval=physical,
            GT_online=False,unique_crops=len(preds),planned_crops=2,student_full_input=True)
        record(crop_path,tc)
    obs={k:[] for k in ('partial','point','nativeclip','wrong')}
    for t in tc['predictions']:
        w=t['window'];left,right=w['bounds']
        obs['partial'].append(observation(w,t['correct']['interval']))
        obs['point'].append(observation(w,t['correct']['interval'],'point'))
        obs['wrong'].append(observation(w,t['wrong']['interval']))
        obs['nativeclip'].append(observation(w,[max(left,physical[0]),min(right,physical[1])]))
    teachers={n:make_supervision(records,oo) for n,oo in obs.items()}
    if phase=='development':
        specs=[(f'partial_b{beta:g}_lr{lr:g}','partial',beta,lr) for beta in (.1,1.,10.) for lr in (1e-4,1e-3,1e-2)]
    else:
        sel=read(OUT/'TEMPORAL_SELECTION.json')['directions'][c]
        names=('point','nativeclip','wrong') if phase=='controls_dev' else ('partial','point','nativeclip','wrong')
        specs=[(n,n,sel['beta'],sel['lr']) for n in names]
    fits={};fullsystems={};audits={}
    for name,kind,beta,lr in specs:
        z=fit(it,records,ids,teacher=teachers[kind],beta=beta,lr=lr,steps=5,charge=budget.charge)
        assert torch.equal(z['final']['boxes'],zero['boxes'].cpu())
        combo={**a0['state'],**z['state']};expected={**z['final'],'boxes':a0['final']['boxes']}
        full=full_prediction(model,frames,ids,r['input'],r['subject'],combo,expected)
        fits[name]=z;fullsystems[name]=full;audits[name]=full['audit']
        print('F36 T',r['key'],name,'best',z['best_step'],'delta',round(z['state_delta'],5),flush=True)
    if smoke:
        for name,steps,lr in [('steps0',0,.001),('lr0',1,0.)]:
            z=fit(it,records,ids,teacher=teachers['partial'],beta=1.,lr=lr,steps=steps,charge=budget.charge)
            assert z['state_delta']==0 and all(torch.equal(a,b.cpu()) for a,b in zip(z['final']['logits'],zero['logits']))
            a=full_prediction(model,frames,ids,r['input'],r['subject'],z['state'],z['final'])['audit']
            record(dest('noops',r['key']+'_time_'+name),dict(key=r['key'],fit=z,audit=a))
    outputs={}
    refs=[legal_logp(z)[0] for z in zero['logits']]
    for beta in sorted({s[2] for s in specs}):
        oo=convex_output(refs,teachers['partial'],beta)
        oo['indices']=posterior_decode(oo['q'],records,ids)
        outputs[f'b{beta:g}']=oo
    record(dest('temporal_'+phase,r['key']),dict(key=r['key'],cohort=c,role=r['f36_role'],source=r['source'],group=r['group'],
        frame_ids=ids,records=records,qa=qa,cost=cost,teacher_path=str(crop_path),teacher_sha256=sha(crop_path),
        teachers=teachers,fits=fits,fullsystems=fullsystems,outputs=outputs,audits=audits,
        native_indices=native,native_boxes=zero['boxes'].cpu(),spatial_parent=a0['final'],parent=pinfo,
        full_input_head_factorization_exact=True,GT_online=False))


def runtime(stage,role,cohort=None,limit=0):
    import torch
    from scripts.run_closure_v1 import model_for
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    p=plan();budget=Budget(p);n=0;expert=None
    if role=='evaluation':
        assert (OUT/'TEMPORAL_SELECTION.json').exists() and (OUT/'SPATIAL_SELECTION.json').exists(), 'evaluation before selection lock'
    if stage=='space':
        cfg=read(ROOT/'artifacts/decota_refine_v1/lock.json')
        assert sha(Path(cfg['expert_snapshot'])/'model.safetensors')==cfg['expert_sha256']
        expert=SpatialExpert(cfg['expert_snapshot']);ed=state_digest(expert.model)
    for c,rr in p['rows'].items():
        if cohort and cohort!=c:continue
        d='space' if stage=='space' else 'temporal_'+role
        relevant='development' if role=='controls_dev' else role
        pending=[r for r in rr if r['f36_role']==relevant and not existing(dest(d,r['key']))]
        pending.sort(key=lambda r:rank(r['group']))
        if not pending or (limit and n>=limit):continue
        tick=time.perf_counter();model=model_for(c);md=state_digest(model)
        write(OUT/'loads'/f'{stage}_{c}_{time.time_ns()}.json',dict(seconds=time.perf_counter()-tick,stage=stage,cohort=c))
        for r in pending:
            if limit and n>=limit:break
            smoke=role=='development' and not list((OUT/'noops').glob(c+'*_'+('space' if stage=='space' else 'time')+'_lr0.pt'))
            if stage=='space':space_one(model,expert,r,c,budget,smoke)
            else:temporal_one(model,r,c,budget,role,smoke)
            assert state_digest(model)==md
            if expert:assert state_digest(expert.model)==ed
            n+=1;status(OUT/'STATUS.json',dict(stage=stage,role=role,last=r['key'],completed_invocation=n,
                  counts=budget.counts,time=time.time()))
        del model;gc.collect();torch.cuda.empty_cache()
    write(OUT/'invocations'/f'{stage}_{role}_{time.time_ns()}.json',dict(stage=stage,role=role,completed=n,counts=budget.counts))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['space','temporal'])
    ap.add_argument('--role',choices=['development','controls_dev','evaluation'],default='development')
    ap.add_argument('--cohort');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    from scripts.run_decota_refine_v1 import configure
    configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tick=time.time();failure=None
    try:runtime(a.stage,a.role,a.cohort,a.limit)
    except BaseException as e:failure=repr(e);raise
    finally:
        write(OUT/'leases'/f'{a.stage}_{time.time_ns()}.json',dict(stage=a.stage,role=a.role,seconds=time.time()-tick,failure=failure))
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':main()
