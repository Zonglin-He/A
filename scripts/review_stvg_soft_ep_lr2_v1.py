#!/usr/bin/env python3
"""Independent no-GT CPU readback and prediction seal for the LR-x2 trial."""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1/lr2'
LR1=ROOT/'artifacts/stvg_soft_ep_opd_v1/mu025'
PARENT=ROOT/'artifacts/stvg_evidence_constrained_v1/round2_prompt009'
ARMS=('F','T','S_M0','TS_M0','S_M1','TS_M1','S_M2','TS_M2','S_M0matched','TS_M0matched')

def need(ok,msg):
    if not ok: raise RuntimeError(msg)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def load_pt(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)
def write_once(p,obj):
    p=Path(p); need(not p.exists(),f'refusing to overwrite {p.name}'); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp'); need(not tmp.exists(),f'stale temporary path {tmp.name}')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); os.replace(tmp,p)
def xyxy(x):
    x=np.asarray(x.detach().cpu().double().numpy() if hasattr(x,'detach') else x,dtype=np.float64)
    return np.concatenate((x[:,:2]-x[:,2:]/2,x[:,:2]+x[:,2:]/2),axis=1)
def same(a,b):
    a,b=np.asarray(a),np.asarray(b); return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)

def _verify_runtime():
    rt=read(BASE/'RUNTIME.json'); need(rt.get('status')=='locked' and rt.get('GT_read') is False,'runtime not locked/GT-free')
    need(isinstance(rt.get('pins'),dict) and isinstance(rt.get('protected'),dict),'runtime pins malformed')
    for rel,d in {**rt['pins'],**rt['protected']}.items(): need(sha(ROOT/rel)==d,'runtime pin mismatch: '+rel)
    return rt

def main():
    import torch
    from scripts.stvg_reference_recovery_common_v1 import load_case
    from scripts.run_stvg_table1_student004 import stable_fit_key
    from scripts.stvg_reference_recovery_fit_v1 import locked_config
    from scripts import stvg_target_teacher_student_v1 as core
    rt=_verify_runtime(); design=read(BASE/'DESIGN_LOCK.json'); rosterdoc=read(BASE/'ROSTER_LOCK.json'); refdoc=read(BASE/'REFERENCE_LOCK.json')
    need(design.get('status')=='locked' and design.get('GT_read') is False and design['response_settings']['mu']==0.25,'design lock invalid')
    need(rosterdoc.get('status')=='locked' and rosterdoc.get('GT_read') is False and rosterdoc.get('count')==32,'roster lock invalid')
    need(refdoc.get('status')=='locked' and refdoc.get('GT_read') is False and refdoc.get('count')==32,'reference lock invalid')
    roster=rosterdoc['records']; refs={r['case_id']:r for r in refdoc['records']}; aliases=[r['probe_alias'] for r in roster]
    need(len(aliases)==32 and len(set(aliases))==32 and set(aliases)==set(refs),'membership mismatch')
    need(sum(len(refs[c]['anchors']) for c in aliases)==49,'locked support count mismatch')
    for path,name in [(BASE/'STUDENT_GLOBAL_BARRIER.json','student'),(LR1/'STUDENT_GLOBAL_BARRIER.json','mu025 student'),(PARENT/'STUDENT_GLOBAL_BARRIER.json','M0 parent')]:
        d=read(path); need(d.get('status')=='complete' and d.get('count')==32 and d.get('GT_read') is False,name+' barrier invalid')
    student_bar=read(BASE/'STUDENT_GLOBAL_BARRIER.json'); files={x['path']:x for x in student_bar['files']}
    expected={str((BASE/'student'/f'{c}.pt').relative_to(ROOT)) for c in aliases}
    need(set(files)==expected and len(files)==32,'student file set mismatch')
    parent_bar=read(PARENT/'STUDENT_GLOBAL_BARRIER.json'); pfiles={x['path']:x for x in parent_bar['files']}
    lr1_bar=read(LR1/'STUDENT_GLOBAL_BARRIER.json'); lfiles={x['path']:x for x in lr1_bar['files']}
    checks=0; fit_counts={m:0 for m in ('M1','M2','M0matched')}; fit_steps={m:0 for m in fit_counts}; source_captures=0; empty=0
    solver_status_counts={}; max_solver_residual=0.0
    for rec in roster:
        cid=rec['probe_alias']; ref=refs[cid]; case,sourcepath=load_case(rec)
        need(case.get('GT_read') is False and case.get('source_case_sha256')==ref['source_case_sha256'],'source binding mismatch '+cid)
        ppath=PARENT/'student'/(cid+'.pt'); lpath=LR1/'student'/(cid+'.pt'); path=BASE/'student'/(cid+'.pt')
        prel=str(ppath.relative_to(ROOT)); lrel=str(lpath.relative_to(ROOT)); rel=str(path.relative_to(ROOT))
        need(prel in pfiles and sha(ppath)==pfiles[prel]['sha256'],'M0 parent file mismatch '+cid)
        need(lrel in lfiles and sha(lpath)==lfiles[lrel]['sha256'],'LR1 comparison file mismatch '+cid)
        need(rel in files and sha(path)==files[rel]['sha256'],'LR2 output file mismatch '+cid)
        receipt=read(path.with_suffix('.receipt.json')); need(receipt.get('status')=='complete' and receipt.get('GT_read') is False and receipt.get('sha256')==sha(path),'student receipt mismatch '+cid)
        out=load_pt(path); parent=load_pt(ppath); lr1=load_pt(lpath)
        need(out.get('schema')=='stvg-soft-ep-lr2-case-v1' and out.get('GT_read') is False and out.get('case_id')==cid and out.get('dataset')==rec['dataset'],'payload identity mismatch '+cid)
        need(out.get('runtime_sha256')==sha(BASE/'RUNTIME.json') and out.get('source_case_sha256')==case['source_case_sha256'] and out.get('source_file_sha256')==sha(sourcepath) and out.get('source_restored') is True,'source custody mismatch '+cid)
        need(out.get('source_model_state_sha256')==case['source_model_state_sha256'] and out.get('pixel_sha256')==case['pixel_sha256'],'source model/pixel mismatch '+cid)
        need(out.get('reference_records')==ref and list(map(int,out['frame_ids']))==list(map(int,case['frame_ids'])),'reference/frame binding mismatch '+cid)
        need(out.get('parent_student_sha256')==pfiles[prel]['sha256'] and out.get('lr1_trial_student_sha256')==lfiles[lrel]['sha256'] and out.get('lr1_trial_runtime_sha256')==sha(LR1/'RUNTIME.json'),'trial source binding mismatch '+cid)
        need(set(out['arms'])==set(ARMS),'arm set mismatch '+cid)
        need(set(out['fits'])=={'M1','M2','M0matched'} and set(out['readouts'])==set(out['fits']),'fit/readout keys mismatch '+cid)
        for dst,src in [('F','F'),('T','T'),('S_M0','S_new_crop'),('TS_M0','TS_new_crop')]:
            need(same(out['arms'][dst]['boxes'],parent['arms'][src]['boxes']) and list(out['arms'][dst]['interval'])==list(parent['arms'][src]['interval']),'exact M0/control reuse mismatch '+cid+'/'+dst)
        for dst in ('F','T','S_M0','TS_M0'):
            need(same(out['arms'][dst]['boxes'],lr1['arms'][dst]['boxes']) and list(out['arms'][dst]['interval'])==list(lr1['arms'][dst]['interval']),'mu025 baseline mismatch '+cid+'/'+dst)
        anchors=ref['anchors']; positions=[int(a['position']) for a in anchors]
        if not anchors:
            empty+=1
            for name in ('M1','M2','M0matched'):
                need(out['fits'][name] is None and out['readouts'][name] is None,'empty-reference fit/readout is not null '+cid+'/'+name)
                need(same(out['arms']['S_'+name]['boxes'],out['arms']['F']['boxes']) and same(out['arms']['TS_'+name]['boxes'],out['arms']['T']['boxes']),'empty-reference fallback mismatch '+cid+'/'+name)
            checks+=1; continue
        source_captures+=1
        key=stable_fit_key(rec['dataset'],rec['query_key_sha256']); cfg=locked_config(rec['dataset'],'current5'); cfg['lr']*=2.0
        need(cfg==design['configs'][rec['dataset']],'LR2 config differs from design '+cid)
        fitsha=sha(BASE/'REFERENCE_LOCK.json')
        for oldname in ('M1','M2'):
            oldfit=lr1['fits'].get(oldname)
            if oldfit is not None:
                oldcfg=locked_config(rec['dataset'],'current5')
                need(oldfit.get('config')==oldcfg and oldfit.get('GT_read') is False,'saved LR1 fit/config mismatch '+cid+'/'+oldname)
        for name in ('M1','M2'):
            f=out['fits'][name]; o=out['readouts'][name]
            need(f.get('GT_read') is False and f.get('mode')==name and f.get('config')==cfg and f.get('settings')==design['response_settings'],'soft fit contract mismatch '+cid+'/'+name)
            need(f.get('seed_key')==key and f.get('active_parameters')==1792 and f.get('observation_positions')==positions and len(f.get('path',[]))==5,'soft fit key/steps mismatch '+cid+'/'+name)
            need(core._tree_equal(f.get('initial_state'),case['source_light_initial_state']),'soft fit did not start from source-light state '+cid+'/'+name)
            need(int(f['optimizer_steps'])==sum(bool(x.get('accepted')) for x in f['path']),'soft accepted-step count mismatch '+cid+'/'+name)
            need(torch.equal(o['boxes'].detach().cpu(),f['final_cached_decoder_boxes'].detach().cpu()),'soft full/cached readout mismatch '+cid+'/'+name)
            want=xyxy(o['boxes']); need(np.array_equal(out['arms']['S_'+name]['boxes'],want) and np.array_equal(out['arms']['TS_'+name]['boxes'],want),'soft arm/readout mismatch '+cid+'/'+name)
            need(list(out['arms']['S_'+name]['interval'])==list(out['arms']['F']['interval']) and list(out['arms']['TS_'+name]['interval'])==list(out['arms']['T']['interval']),'soft temporal control mismatch '+cid+'/'+name)
            if name=='M2':
                for row in f['path']:
                    s=row.get('solver')
                    if s:
                        status=s.get('status'); residual=float(s['solver_residual_rel'])
                        need(status in ('converged','max_iter','near_zero_response_identity') and math.isfinite(residual),'M2 solver status/residual invalid '+cid)
                        need(0<=int(s.get('iterations',-1))<=16 and residual>=0.0,'M2 solver iteration/residual range invalid '+cid)
                        if status=='converged': need(residual<=1e-3+1e-6,'M2 converged residual exceeds configured tolerance '+cid)
                        if status=='near_zero_response_identity': need(float(s['lambda'])==0.0,'M2 identity fallback lambda is nonzero '+cid)
                        solver_status_counts[status]=solver_status_counts.get(status,0)+1; max_solver_residual=max(max_solver_residual,residual)
            fit_counts[name]+=1; fit_steps[name]+=int(f['optimizer_steps'])
        f=out['fits']['M0matched']; o=out['readouts']['M0matched']
        need(f.get('schema')=='stvg-instance-loss-weighted-fit-v1' and f.get('GT_read') is False and f.get('reference_role')=='teacher','M0matched fit identity mismatch '+cid)
        need(f.get('algorithm')=='validated_loss_weighted_iou' and f.get('validation')=='weighted_reference_IoU_non_decrease_1e-7_then_stop' and f.get('reference_sha256')==fitsha,'M0matched validation/source mismatch '+cid)
        need(f.get('config')==cfg and f.get('seed_key')==key and f.get('active_parameters')==1792 and f.get('observation_positions')==positions and len(f.get('path',[]))==5,'M0matched config/key/steps mismatch '+cid)
        need(core._tree_equal(f.get('initial_state'),case['source_light_initial_state']),'M0matched did not start from source-light state '+cid)
        need(int(f['optimizer_steps'])+int(f['rejected_optimizer_steps'])==int(f['proposed_optimizer_steps']),'M0matched optimizer counts inconsistent '+cid)
        need(torch.equal(o['boxes'].detach().cpu(),f['final_cached_decoder_boxes'].detach().cpu()),'M0matched full/cached readout mismatch '+cid)
        want=xyxy(o['boxes']); need(np.array_equal(out['arms']['S_M0matched']['boxes'],want) and np.array_equal(out['arms']['TS_M0matched']['boxes'],want),'M0matched arm/readout mismatch '+cid)
        need(list(out['arms']['S_M0matched']['interval'])==list(out['arms']['F']['interval']) and list(out['arms']['TS_M0matched']['interval'])==list(out['arms']['T']['interval']),'M0matched temporal control mismatch '+cid)
        fit_counts['M0matched']+=1; fit_steps['M0matched']+=int(f['optimizer_steps']); checks+=1
    need(source_captures==23 and empty==9 and fit_counts=={'M1':23,'M2':23,'M0matched':23},'actual fit/capture cardinality mismatch')
    need(student_bar.get('fit_invocations')==69 and student_bar.get('source_captures')==23 and student_bar.get('full_readout_calls')==69,'worker barrier counts mismatch')
    result=dict(schema='stvg-soft-ep-lr2-independent-readback-v1',status='pass',count=32,nonempty_reference_sources=source_captures,
        empty_reference_sources=empty,reference_anchor_count=49,fit_invocations=sum(fit_counts.values()),fit_count_by_method=fit_counts,
        optimizer_steps_by_method=fit_steps,full_readout_calls=69,checks=checks,M2_solver_status_counts=solver_status_counts,
        M2_max_solver_residual_rel=max_solver_residual,GT_read=False,CPU_only=True,model_loaded=False,GPU_used=False,
        scope='source/reference/input binding, exact controls, LR-x2 fits, M0matched validated-OPD accounting, saved/full readout equality; no GT scoring',
        bindings={'runtime_sha256':sha(BASE/'RUNTIME.json'),'design_sha256':sha(BASE/'DESIGN_LOCK.json'),'roster_sha256':sha(BASE/'ROSTER_LOCK.json'),
            'reference_lock_sha256':sha(BASE/'REFERENCE_LOCK.json'),'student_barrier_sha256':sha(BASE/'STUDENT_GLOBAL_BARRIER.json'),
            'lr1_student_barrier_sha256':sha(LR1/'STUDENT_GLOBAL_BARRIER.json'),'lr1_global_prediction_barrier_sha256':sha(LR1/'GLOBAL_PREDICTION_BARRIER.json'),
            'lr1_score_rows_sha256':sha(LR1/'POSTSEAL_SCORE_ROWS.json'),'parent_student_barrier_sha256':sha(PARENT/'STUDENT_GLOBAL_BARRIER.json')},time=time.time())
    write_once(BASE/'ROOT_READBACK.json',result)
    pins=[]
    for x in student_bar['files']:
        pins.append(dict(path=x['path'],sha256=x['sha256'],bytes=x.get('bytes')))
        rp=(ROOT/x['path']).with_suffix('.receipt.json'); pins.append(dict(path=str(rp.relative_to(ROOT)),sha256=sha(rp),bytes=rp.stat().st_size))
    for scope,name in [(BASE,n) for n in ('DESIGN_LOCK.json','ROSTER_LOCK.json','REFERENCE_LOCK.json','CPU_CONTRACTS.json','PREPARATION_READBACK.json','RUNTIME.json','STUDENT_GLOBAL_BARRIER.json','ROOT_READBACK.json')]:
        pins.append(dict(path=str(scope.relative_to(ROOT)/name),sha256=sha(scope/name),bytes=(scope/name).stat().st_size))
    for scope,names in [(LR1,('DESIGN_LOCK.json','RUNTIME.json','STUDENT_GLOBAL_BARRIER.json','ROOT_READBACK.json','GLOBAL_PREDICTION_BARRIER.json','POSTSEAL_SCORE_ROWS.json')),
                        (PARENT,('RUNTIME006.json','STUDENT_GLOBAL_BARRIER.json','ROOT_READBACK.json','GLOBAL_PREDICTION_BARRIER.json','REFERENCE_LOCK.json'))]:
        for name in names:
            p=scope/name
            if p.exists(): pins.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    write_once(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(schema='stvg-soft-ep-lr2-prediction-barrier-v1',status='complete',count=32,
        files=pins,student_barrier_sha256=sha(BASE/'STUDENT_GLOBAL_BARRIER.json'),root_readback_sha256=sha(BASE/'ROOT_READBACK.json'),
        scorer_sha256=sha(ROOT/'scripts/measure_stvg_soft_ep_lr2_v1.py'),GT_read=False))
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__': main()
