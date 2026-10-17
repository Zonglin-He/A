"""Finite R1 four-anchor extension. Reuse old slot0 exactly; no old runtime mutation."""
from pathlib import Path
import ast,copy,inspect,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_semantic_preference_4anchor32_v1'
PRIOR=ROOT/'artifacts/stvg_semantic_preference_r1_32_v1'
OLD=ROOT/'artifacts/stvg_qwen35_geometry32_v1'

def verify():
    r=read(BASE/'EXECUTION_RUNTIME.json')
    for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
    if (BASE/'ACTIVE_EXECUTION.json').exists():assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch'],'four-anchor compute deadline'
    return r

def prepare():
    assert not BASE.exists();BASE.mkdir()
    for n in ['ROSTER_LOCK.json','INPUT_JOB_LOCK.json','INPUT_BARRIER.json']:(BASE/n).write_bytes((PRIOR/n).read_bytes())
    jobs=read(BASE/'INPUT_JOB_LOCK.json')['records'];assert len(jobs)==128 and all(len(j['candidates'])==4 for j in jobs)
    direct=read(ROOT/'artifacts/stvg_r1_direct_reliable32_v1/INPUT_JOB_LOCK.json')['records']
    assert len(direct)==128
    dm={j['job_id']:j for j in direct}
    for j in jobs:
        d=dm[j['job_id']]
        for k in ['case_id','slot','position','physical_frame_id','source_case_sha256','query_text','image','context']:assert j[k]==d[k],(j['job_id'],k)
    for cid in {j['case_id'] for j in jobs}:
        q=[j for j in jobs if j['case_id']==cid];assert sorted(j['slot'] for j in q)==[0,1,2,3] and len({j['position'] for j in q})==4
    reused=[]
    for f in read(PRIOR/'teachers/R1/QUALIFICATION_COMPLETION.json')['files']:
        p=ROOT/f['path'];assert sha(p)==f['sha256'];d=read(p);assert next(j for j in jobs if j['job_id']==d['job_id'])['slot']==0
        target=BASE/'teachers/R1'/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes());reused.append(dict(path=str(target.relative_to(ROOT)),sha256=sha(target),bytes=target.stat().st_size))
    assert len(reused)==32
    write(BASE/'REUSE_LOCK.json',dict(status='locked',count=32,files=reused,input_job_sha256=sha(BASE/'INPUT_JOB_LOCK.json'),prior_input_sha256=sha(PRIOR/'INPUT_JOB_LOCK.json'),reused_prefill_calls=64,GT_read=False))
    write(BASE/'QUALIFICATION_PLAN.json',dict(status='locked',count=128,probe_jobs=[j['job_id'] for j in jobs],maximum_new_R1_prefill_calls=192,total_actual_prefill_calls=256,GT_read=False))
    write(BASE/'DIRECT_INPUT_ALIGNMENT.json',dict(status='pass',count=32,planned_anchors=128,exact_four_anchor_positions_and_pixels_context_Source_query=True,Direct_input_sha256=sha(ROOT/'artifacts/stvg_r1_direct_reliable32_v1/INPUT_JOB_LOCK.json'),preference_input_sha256=sha(BASE/'INPUT_JOB_LOCK.json'),Teacher_calls_not_equal='two order likelihoods per anchor vs one Direct generation',remaining_differences=['fixed_existing_candidate_support vs continuous Direct boxes','fixed_support_distillation vs resampled32 Gaussian OPD'],GT_read=False))
    write(BASE/'R1_PRIORITY_AUTHORIZATION002.json',dict(status='explicitly_authorized',scope='full32 four anchors align Direct; matched GEOM/PREF/D2; no search',GT_read=False))
    write(BASE/'STATUS.json',dict(status='prepared',stage='CPU_contract',count=32,GT_read=False))

def collect():
    from scripts import stvg_semantic_preference_interface_v2 as old
    verify();reuse=read(BASE/'REUSE_LOCK.json');assert reuse['input_job_sha256']==reuse['prior_input_sha256']
    for f in reuse['files']:assert sha(ROOT/f['path'])==f['sha256']
    source=inspect.getsource(old.collect)
    needle="probe=read(BASE/'QUALIFICATION_PLAN.json')['probe_jobs'];outputs=[];calls=0"
    assert source.count(needle)==1
    source=source.replace(needle,"probe=[j for j in read(BASE/'QUALIFICATION_PLAN.json')['probe_jobs'] if not (BASE/'teachers/R1'/(j+'.json')).exists()];outputs=list(reuse_rows);calls=0")
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(128),n) if type(n.value)is int and n.value==8 else n
    source=ast.unparse(ast.fix_missing_locations(Count().visit(ast.parse(source))))
    ns=dict(old.__dict__);ns.update(BASE=BASE,verify=verify,reuse_rows=reuse['files']);exec(compile(source,__file__,'exec'),ns);ns['collect']('R1')
    d=read(BASE/'teachers/R1/QUALIFICATION_COMPLETION.json');assert d['count']==128 and d['actual_prefill_calls']==192 and len(d['files'])==128
    jobs={j['job_id']:j for j in read(BASE/'INPUT_JOB_LOCK.json')['records']};seen=set()
    for f in d['files']:
        assert sha(ROOT/f['path'])==f['sha256'];r=read(ROOT/f['path']);assert r['job_id'] not in seen;seen.add(r['job_id']);assert len(r['evidence'])==2
        n=len(jobs[r['job_id']]['candidates'])
        for e in r['evidence']:
            p=old.categorical_from_logprobs(e['raw_label_logprobs']);assert p==e['conditional_label_posterior'] and old.restore_candidate_order(p,e['display_order'])==e['candidate_order_posterior']
        avg=[sum(e['candidate_order_posterior'][i] for e in r['evidence'])/2 for i in range(n+1)];assert avg==r['permutation_averaged_preference']
    assert seen==set(jobs)
    write(BASE/'R1_ROOT_READBACK.json',dict(status='pass',count=32,anchors=128,new_prefill_calls=192,reused_prefill_calls=64,total_actual_prefill_calls=256,GT_read=False))

def setup():
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    verify();assert read(BASE/'R1_ROOT_READBACK.json')['status']=='pass'
    jobs=read(BASE/'INPUT_JOB_LOCK.json')['records'];rr=read(BASE/'ROSTER_LOCK.json');assert rr['count']==32
    supports=[];refs=[];diagnostics=[]
    for rec in rr['records']:
        cid=rec['probe_alias'];case,_=load_case(rec);active=[]
        for j in sorted([j for j in jobs if j['case_id']==cid],key=lambda j:j['position']):
            f=BASE/'teachers/R1'/(j['job_id']+'.json');d=read(f);p=d['permutation_averaged_preference'];boxes=[]
            for c in j['candidates']:
                x,y,u,v=c['box_xyxy'];boxes.append([(x+u)/2,(y+v)/2,u-x,v-y])
            chosen=max(range(5),key=p.__getitem__);native=case['zero']['boxes'][j['position']]
            reason='admitted'
            if chosen==0:reason='Teacher_abstention'
            elif not all(all(0<v<1 and math.isfinite(v) for v in b) for b in boxes):reason='fixed_candidate_not_in_open_logit_chart_anchor_rejected'
            elif not all(0<float(v)<1 and math.isfinite(float(v)) for v in native):reason='Source_anchor_not_in_open_logit_chart_anchor_rejected'
            a=dict(job_id=j['job_id'],slot=j['slot'],position=j['position'],physical_frame_id=j['physical_frame_id'],boxes_cxcywh=boxes,posterior=p,admission_reason=reason,teacher_sha256=sha(f))
            diagnostics.append(dict(case_id=cid,dataset=rec['dataset'],**a))
            if reason=='admitted':
                a['reference']=dict(position=j['position'],physical_frame_id=j['physical_frame_id'],box_cxcywh=boxes[chosen-1],weight=1.,candidate_id='semantic_preference_argmax',teacher_output_sha256=sha(f),GT_read=False);active.append(a)
        aa=[a['reference'] for a in active];status='admitted' if aa else 'no_admitted_anchor_Source_noop'
        supports.append(dict(case_id=cid,dataset=rec['dataset'],active=active,common_reference=aa,admission_reason=status,GT_read=False));refs.append(dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=case['source_case_sha256'],anchors=aa,admission_reason=status,GT_read=False));del case
    for n in ['SOURCE_CAPTURE_BARRIER.json','TEMPORAL_LOCK.json','BASELINE_BARRIER.json']:(BASE/n).write_bytes((OLD/n).read_bytes())
    write(BASE/'SUPPORT_LOCK.json',dict(status='locked',count=32,records=supports,anchor_decisions=diagnostics,GT_read=False));write(BASE/'D1_REFERENCE_LOCK.json',dict(status='locked',count=32,records=[dict(case_id=r['case_id'],anchors=r['anchors']) for r in refs],GT_read=False))
    for arm in ['GEOM','PREF']:
        a=BASE/'arms'/arm;t=a/'trials/G32';a.mkdir(parents=True)
        for n in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json']:(a/n).write_bytes((BASE/n).read_bytes())
        write(a/'SCOPE_CONFIG.json',dict(recipe='lite_fullstep',label='four_anchor_fixed_support_R1_'+arm,GT_read=False));write(a/'teacher/G32/ROOT_READBACK.json',dict(status='pass',count=32,GT_read=False))
        write(t/'REFERENCE_LOCK.json',dict(status='locked',count=32,records=refs,GT_read=False));(t/'TEMPORAL_LOCK.json').write_bytes((BASE/'TEMPORAL_LOCK.json').read_bytes())
        d=copy.deepcopy(read(ROOT/'artifacts/stvg_semantic_preference_student32_v1/arms'/arm/'trials/G32/DESIGN_LOCK.json'));d.update(scope='four_anchor_fixed_support_'+arm,cohort='same exposed32 all four prelocked slots');write(t/'DESIGN_LOCK.json',d)
    write(BASE/'STUDENT_INPUT_COMPLETION.json',dict(status='pass',count=32,planned_anchors=128,admitted_anchors=sum(len(s['active']) for s in supports),no_fit_sources=sum(not s['active'] for s in supports),same_reference_admission_support_GEOM_PREF_D2=True,GT_read=False))

def policy(mean,evidence,key,step,*,config,support,arm):
    import torch
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
    active=support['active'];boxes=mean.new_tensor([a['boxes_cxcywh'] for a in active]);p=mean.new_tensor([a['posterior'] for a in active]);assert boxes.shape==(len(mean),4,4) and ((boxes>0)&(boxes<1)).all()
    samples=torch.logit(boxes).detach();r=aligned_iou(boxes,evidence[:,None,:]).detach();w=(p[:,1:] if arm=='PREF' else (1-p[:,0,None])*torch.softmax(r/config['tau'],-1)).detach()
    return dict(mean=mean.detach().clone(),samples=samples,rewards=r,used_rewards=(p[:,1:] if arm=='PREF' else r).detach(),weights=w,permutations=None,informative=bool((w!=w[:,:1]).any()),seed=None,sample_origin='fixed_existing_candidate_atoms_not_Gaussian_draws',feedback_arm=arm)

def rollout_audit(row,evidence,cfg,positions,key,step,support,arm):
    import numpy as np
    from scripts import audit_stvg_spatial_lite_fit002 as audit
    a=lambda x:np.asarray(audit._arr(x),dtype=float)
    active=support['active'];assert positions==[r['position'] for r in active]
    roll=row['rollout'];mean=a(roll['mean']);samples=a(roll['samples']);w=a(roll['weights']);boxes=np.asarray([r['boxes_cxcywh'] for r in active],float);p=np.asarray([r['posterior'] for r in active],float)
    assert np.max(abs(samples-np.log(boxes/(1-boxes))))<3e-6
    reward=audit._iou(boxes,evidence[:,None,:]);q=np.exp(reward/cfg['tau']-(reward/cfg['tau']).max(-1,keepdims=True));q=q/q.sum(-1,keepdims=True)*(1-p[:,0,None]);target=p[:,1:] if arm=='PREF' else q
    qe=float(np.max(abs(w-target)));assert qe<3e-6
    logp=-np.square(samples-mean[:,None,:]).sum(-1)/(2*cfg['sigma']**2)-2*math.log(2*math.pi*cfg['sigma']**2);loss=float(-(w*logp).sum(-1).mean());le=abs(loss-row['loss']);assert le<.001+3e-5*abs(loss)
    informative=bool(np.any(w!=w[:,:1]));ge=0.
    if informative:
        expected=(w[:,:,None]*(mean[:,None,:]-samples)).sum(1)/(cfg['sigma']**2*len(positions));ge=float(np.max(abs(a(row['mean_gradient'])-expected)));assert ge<.001+3e-5*float(np.max(abs(expected)))
    assert roll['informative']==informative and roll['seed'] is None
    return dict(mean=mean,samples=samples,weights=w,evidence=evidence,expected_loss=loss,informative=informative,loss_error=le,mean_gradient_error=ge,reward_error=float(np.max(abs(a(roll['rewards'])-reward))),q_error=qe)

def study():
    from scripts import stvg_semantic_preference_student32_v1 as s
    s.BASE=BASE;s.PREF=BASE;s.verify=verify;s.policy=policy;s.rollout_audit=rollout_audit
    return s

def run(stage,arm=None):
    s=study()
    if stage=='root':
        from scripts import audit_stvg_spatial_lite_fit002 as audit
        from scripts.audit_stvg_fixed_support_adam002 import adam_audit
        audit._adam_audit=adam_audit
    if stage=='lock':
        r=read(BASE/'EXECUTION_RUNTIME.json')
        for aa in ['GEOM','PREF']:
            a=BASE/'arms'/aa;t=a/'trials/G32';pins=dict(r['pins']);paths=[BASE/'SUPPORT_LOCK.json',BASE/'R1_ROOT_READBACK.json',BASE/'teachers/R1/QUALIFICATION_COMPLETION.json',a/'ROSTER_LOCK.json',a/'SOURCE_CAPTURE_BARRIER.json',a/'SCOPE_CONFIG.json',a/'teacher/G32/ROOT_READBACK.json',t/'REFERENCE_LOCK.json',t/'TEMPORAL_LOCK.json',t/'DESIGN_LOCK.json'];pins.update({str(p.relative_to(ROOT)):sha(p) for p in paths});write(t/'RUNTIME.json',dict(status='locked',trial_id='G32',pins=pins,protected={},GT_read=False))
        return
    return getattr(s,stage)(arm) if arm else getattr(s,stage)()
if __name__=='__main__':
    k=sys.argv[1]
    if k in ['prepare','collect','setup']:globals()[k]()
    else:run(k,sys.argv[2] if len(sys.argv)>2 else None)
