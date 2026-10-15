#!/usr/bin/env python3
"""Sealed 32-source scoring for the Soft EP LR-x2 trial, plus matched LR1 contrasts."""
from __future__ import annotations
import json, math, os, statistics, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1/lr2'
LR1=ROOT/'artifacts/stvg_soft_ep_opd_v1/mu025'
PARENT=ROOT/'artifacts/stvg_evidence_constrained_v1/round2_prompt009'
ARMS=('F','T','S_M0','TS_M0','S_M1','TS_M1','S_M2','TS_M2','S_M0matched','TS_M0matched')
METRICS=('m_vIoU','m_tIoU','m_sIoU_dense_GT','m_sIoU_supervised_GT','m_sIoU_unsupervised_GT')
CONTRASTS=(
 ('S_M2_minus_S_M1','S_M2','S_M1'),('TS_M2_minus_TS_M1','TS_M2','TS_M1'),
 ('S_M2_minus_S_M0','S_M2','S_M0'),('TS_M2_minus_TS_M0','TS_M2','TS_M0'),
 ('S_M1_minus_S_M0','S_M1','S_M0'),('TS_M1_minus_TS_M0','TS_M1','TS_M0'),
 ('S_M1_minus_S_M0matched','S_M1','S_M0matched'),('TS_M1_minus_TS_M0matched','TS_M1','TS_M0matched'),
 ('S_M2_minus_S_M0matched','S_M2','S_M0matched'),('TS_M2_minus_TS_M0matched','TS_M2','TS_M0matched'),
 ('S_M0matched_minus_S_M0','S_M0matched','S_M0'),('TS_M0matched_minus_TS_M0','TS_M0matched','TS_M0'),
 ('S_M0matched_minus_F','S_M0matched','F'),('TS_M0matched_minus_T','TS_M0matched','T'))
LR1_CONTRASTS=(('LR2_M1_vs_LR1_M1_S','S_M1','S_M1'),('LR2_M1_vs_LR1_M1_TS','TS_M1','TS_M1'),
 ('LR2_M2_vs_LR1_M2_S','S_M2','S_M2'),('LR2_M2_vs_LR1_M2_TS','TS_M2','TS_M2'))
EPS=1e-10

def need(ok,msg):
    if not ok: raise RuntimeError(msg)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def rootpath(s):
    p=(ROOT/s).resolve(); need(p==ROOT or ROOT in p.parents,f'path escapes repository: {s}'); return p
def filemap(rows,label):
    need(isinstance(rows,list),label+'.files must be a list'); out={}
    for x in rows:
        need(isinstance(x,dict) and isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str),'bad '+label+' file row')
        p=rootpath(x['path']); need(p.is_file() and sha(p)==x['sha256'],label+' SHA mismatch '+x['path'])
        if x.get('bytes') is not None: need(p.stat().st_size==int(x['bytes']),label+' byte mismatch '+x['path'])
        need(x['path'] not in out,'duplicate '+label+' path'); out[x['path']]=x
    return out
def load_pt(p):
    import torch
    import numpy as n
    safe=[n.ndarray,n.dtype]
    for mod in (getattr(n,'core',None),getattr(n,'_core',None)):
        ma=getattr(mod,'multiarray',None) if mod is not None else None
        for name in ('_reconstruct','scalar'):
            if ma is not None and hasattr(ma,name): safe.append(getattr(ma,name))
    safe += [kind for t in (n.float16,n.float32,n.float64,n.int8,n.int16,n.int32,n.int64,n.uint8,n.bool_) for kind in (n.dtype(t).type,type(n.dtype(t)))]
    with torch.serialization.safe_globals(safe): return torch.load(p,map_location='cpu',weights_only=True)
def write_once(path,obj):
    path=Path(path); need(not path.exists(),'refusing repeat GT scoring: '+path.name); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp'); need(not tmp.exists(),'stale temp path '+tmp.name)
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); os.replace(tmp,path)

def gates():
    rt=read(BASE/'RUNTIME.json'); need(rt.get('status')=='locked' and rt.get('GT_read') is False,'LR2 runtime not locked/GT-free')
    for p,h in {**rt['pins'],**rt['protected']}.items(): need(sha(rootpath(p))==h,'runtime pin mismatch: '+p)
    design=read(BASE/'DESIGN_LOCK.json'); rosterdoc=read(BASE/'ROSTER_LOCK.json'); refdoc=read(BASE/'REFERENCE_LOCK.json')
    need(design.get('status')=='locked' and design.get('GT_read') is False and design['response_settings']['mu']==0.25,'LR2 design invalid')
    need(rosterdoc.get('status')=='locked' and rosterdoc.get('count')==32 and rosterdoc.get('GT_read') is False,'LR2 roster invalid')
    need(refdoc.get('status')=='locked' and refdoc.get('count')==32 and refdoc.get('GT_read') is False,'LR2 reference lock invalid')
    roster=rosterdoc['records']; refs={r['case_id']:r for r in refdoc['records']}; aliases=[r['probe_alias'] for r in roster]
    need(len(aliases)==32 and len(set(aliases))==32 and set(aliases)==set(refs),'LR2 roster/reference mismatch')
    need(sum(len(refs[c]['anchors']) for c in aliases)==49,'LR2 support count mismatch')
    sbp=BASE/'STUDENT_GLOBAL_BARRIER.json'; sb=read(sbp)
    need(sb.get('status')=='complete' and sb.get('count')==32 and sb.get('GT_read') is False and sb.get('runtime_sha256')==sha(BASE/'RUNTIME.json'),'LR2 student barrier invalid')
    sf=filemap(sb.get('files'),'LR2 student'); expected={str((BASE/'student'/f'{c}.pt').relative_to(ROOT)) for c in aliases}
    need(set(sf)==expected,'LR2 student file set mismatch')
    rp=BASE/'ROOT_READBACK.json'; rr=read(rp)
    need(rr.get('status')=='pass' and rr.get('count')==32 and rr.get('GT_read') is False,'independent pre-GT readback invalid')
    gp=BASE/'GLOBAL_PREDICTION_BARRIER.json'; gb=read(gp)
    need(gb.get('status')=='complete' and gb.get('count')==32 and gb.get('GT_read') is False,'LR2 prediction seal invalid')
    need(gb.get('student_barrier_sha256')==sha(sbp) and gb.get('root_readback_sha256')==sha(rp),'prediction seal misses student/root bindings')
    gf=filemap(gb.get('files'),'LR2 prediction seal')
    need(expected<=set(gf) and all(gf[p]['sha256']==sf[p]['sha256'] for p in expected),'prediction-sealed student file mismatch')
    need(gb.get('scorer_sha256')==sha(Path(__file__)),'scorer source differs from sealed pin')
    for scope,name in [(LR1,'mu025'),(PARENT,'parent M0')]:
        b=read(scope/'STUDENT_GLOBAL_BARRIER.json'); need(b.get('status')=='complete' and b.get('count')==32 and b.get('GT_read') is False,name+' barrier invalid')
    lbar=read(LR1/'GLOBAL_PREDICTION_BARRIER.json'); need(lbar.get('status')=='complete' and lbar.get('count')==32 and lbar.get('GT_read') is False,'mu025 global seal invalid')
    lr1rows=read(LR1/'POSTSEAL_SCORE_ROWS.json')
    need(lr1rows.get('status')=='complete' and lr1rows.get('GT_read') is True and lr1rows.get('case_count')==32,'mu025 completed scored baseline missing')
    old={(r['case_id'],r['arm']):r for r in lr1rows['rows']}
    need(len(old)==256 and all((c,a) in old for c in aliases for a in ('S_M1','TS_M1','S_M2','TS_M2')),'mu025 LR1 comparison rows incomplete')
    from scripts.stvg_evidence_common_v1 import load_case as load_source
    cases={}
    for rec in roster:
        case,src=load_source(rec); need(case.get('GT_read') is False and case.get('source_case_sha256')==refs[rec['probe_alias']]['source_case_sha256'],'source binding mismatch '+rec['probe_alias'])
        cases[rec['probe_alias']]=case
    return dict(design=design,roster=roster,refs=refs,cases=cases,sfiles=sf,prediction_files=gf,student_barrier=sb,
        old_rows=old,lr1_rows=lr1rows,bindings={'runtime_sha256':sha(BASE/'RUNTIME.json'),'design_sha256':sha(BASE/'DESIGN_LOCK.json'),
            'roster_sha256':sha(BASE/'ROSTER_LOCK.json'),'reference_lock_sha256':sha(BASE/'REFERENCE_LOCK.json'),
            'student_barrier_sha256':sha(sbp),'root_readback_sha256':sha(rp),'prediction_seal_sha256':sha(gp),
            'lr1_runtime_sha256':sha(LR1/'RUNTIME.json'),'lr1_student_barrier_sha256':sha(LR1/'STUDENT_GLOBAL_BARRIER.json'),
            'lr1_global_prediction_barrier_sha256':sha(LR1/'GLOBAL_PREDICTION_BARRIER.json'),'lr1_score_rows_sha256':sha(LR1/'POSTSEAL_SCORE_ROWS.json'),
            'parent_global_prediction_barrier_sha256':sha(PARENT/'GLOBAL_PREDICTION_BARRIER.json')})

def aggregate(rows):
    by={(r['case_id'],r['arm']):r for r in rows}; result=[]
    for ds in ('hc2','vidstg'):
        cids=sorted({r['case_id'] for r in rows if r['dataset']==ds}); need(len(cids)==16,ds+' denominator mismatch')
        for arm in ARMS:
            for metric in METRICS:
                vals=[]; unknown=[]
                for cid in cids:
                    value=by[cid,arm]['metrics'].get(metric)
                    if value is None: unknown.append(cid)
                    else: need(math.isfinite(float(value)),'nonfinite '+metric); vals.append(float(value))
                result.append(dict(dataset=ds,arm=arm,metric=metric,n_valid=len(vals),n_unknown=len(unknown),unknown_case_ids=unknown,
                    mean=math.fsum(vals)/len(vals) if vals else None,median=statistics.median(vals) if vals else None,min=min(vals) if vals else None,max=max(vals) if vals else None))
    return result
def paired(rows):
    by={(r['case_id'],r['arm']):r for r in rows}; result=[]
    for ds in ('hc2','vidstg'):
        cids=sorted({r['case_id'] for r in rows if r['dataset']==ds})
        for name,left,right in CONTRASTS:
            for metric in METRICS:
                vals={}; unknown=[]
                for cid in cids:
                    a=by[cid,left]['metrics'].get(metric); b=by[cid,right]['metrics'].get(metric)
                    if a is None or b is None: unknown.append(cid)
                    else: vals[cid]=100*(float(a)-float(b))
                result.append(dict(dataset=ds,contrast=name,comparison_arm=left,baseline_arm=right,metric=metric,n_pairs=len(vals),unknown_count=len(unknown),
                    unknown_case_ids=unknown,positive_case_ids=sorted(c for c,v in vals.items() if v>EPS),negative_case_ids=sorted(c for c,v in vals.items() if v< -EPS),
                    neutral_case_ids=sorted(c for c,v in vals.items() if abs(v)<=EPS),neutral_tolerance_pp=EPS,
                    mean_delta_pp=math.fsum(vals.values())/len(vals) if vals else None,per_source_delta_pp=vals))
    return result
def cross_trial(rows,old_rows,aliases):
    current={(r['case_id'],r['arm']):r for r in rows}; out=[]
    for ds in ('hc2','vidstg'):
        cids=sorted(c for c in aliases if current[c,'F']['dataset']==ds)
        for name,newarm,oldarm in LR1_CONTRASTS:
            for metric in METRICS:
                vals={}; unknown=[]
                for cid in cids:
                    new=current[cid,newarm]['metrics'].get(metric); old=old_rows[cid,oldarm]['metrics'].get(metric)
                    if new is None or old is None: unknown.append(cid)
                    else:
                        need(math.isfinite(float(old)),'nonfinite saved LR1 metric')
                        vals[cid]=100*(float(new)-float(old))
                out.append(dict(dataset=ds,contrast=name,comparison_arm=newarm,baseline_trial='mu025_LR1',baseline_arm=oldarm,metric=metric,
                    n_pairs=len(vals),unknown_count=len(unknown),unknown_case_ids=unknown,
                    positive_case_ids=sorted(c for c,v in vals.items() if v>EPS),negative_case_ids=sorted(c for c,v in vals.items() if v< -EPS),
                    neutral_case_ids=sorted(c for c,v in vals.items() if abs(v)<=EPS),neutral_tolerance_pp=EPS,
                    mean_delta_pp=math.fsum(vals.values())/len(vals) if vals else None,per_source_delta_pp=vals,
                    interpretation='matched-case cross-trial comparison against already sealed mu=.25 LR1 scores; not an extra arm in this trial'))
    return out

def main():
    outputs=[BASE/'GT_EXPOSURE.json',BASE/'POSTSEAL_SCORE_ROWS.json',BASE/'POSTSEAL_SCORE_SUMMARY.json',BASE/'POSTSEAL_PAIRED_CONTRASTS.json',BASE/'CROSS_TRIAL_LR1_CONTRASTS.json',BASE/'CPU_COMPLETION.json']
    need(not any(p.exists() for p in outputs),'score outputs already exist; refusing a second GT pass')
    locked=gates()
    exposure=dict(schema='stvg-soft-ep-lr2-gt-exposure-v1',status='after_lr2_prediction_seal_and_independent_cpu_readback',bindings=locked['bindings'],GT_read=True,time=time.time())
    write_once(outputs[0],exposure)
    from scripts import review_stvg_sparse_r1_v2 as truth_reader
    from scripts import score_stvg_native_instance_nativeclock005 as scorer
    targets,_=truth_reader._truths_after_seal(locked['cases']); official,independent,contract=truth_reader._metric_functions()
    rows=[]; payloads={}
    for rec in locked['roster']:
        cid=rec['probe_alias']; path=BASE/'student'/(cid+'.pt'); item=locked['sfiles'][str(path.relative_to(ROOT))]
        need(sha(path)==item['sha256'],'student changed after gate '+cid); z=load_pt(path); payloads[cid]=z
        for arm in ARMS:
            metrics,audit=scorer._score_student_arm(z['arms'][arm],locked['cases'][cid],targets[cid],set(locked['refs'][cid]['anchors'][i]['position'] for i in range(len(locked['refs'][cid]['anchors']))),official,independent)
            tag=None
            for prefix in ('S_','TS_'):
                if arm.startswith(prefix): tag=arm[len(prefix):]; break
            fit=z['fits'].get(tag) if tag in ('M1','M2','M0matched') else None
            fs={'kind':'reference_fit','optimizer_steps':fit.get('optimizer_steps'),'proposed_steps':fit.get('proposed_optimizer_steps',fit.get('proposed_optimizer_steps')),
                'rejected_steps':fit.get('rejected_optimizer_steps'),'declared_steps':fit.get('config',{}).get('steps'),'active_parameters':fit.get('active_parameters'),
                'algorithm':fit.get('algorithm',fit.get('mode'))} if fit else {'kind':'exact_parent_reuse' if arm in ('S_M0','TS_M0') else 'no_fit'}
            rows.append(dict(case_id=cid,dataset=z['dataset'],arm=arm,status='scored',metrics=metrics,support=audit['support'],metric_audit=audit['metric_audit'],fit=fs))
    need(len(rows)==320,'expected 32x10 score rows')
    sums=aggregate(rows); pairs=paired(rows); cross=cross_trial(rows,locked['old_rows'],[r['probe_alias'] for r in locked['roster']])
    need(len(sums)==100 and len(pairs)==140 and len(cross)==40,'aggregate/contrast cardinality mismatch')
    common=dict(schema='stvg-soft-ep-lr2-postseal-v1',status='complete',case_count=32,row_count=320,metric_names=list(METRICS),
        bindings=locked['bindings'],GT_read=True,bootstrap_executed=False)
    write_once(outputs[1],dict(common,rows=rows)); write_once(outputs[2],dict(common,summary=sums))
    write_once(outputs[3],dict(common,contrast_definitions=[list(x) for x in CONTRASTS],paired_contrasts=pairs))
    write_once(outputs[4],dict(schema='stvg-soft-ep-lr2-cross-trial-contrasts-v1',status='complete',case_count=32,row_count=40,
        baseline_trial='artifacts/stvg_soft_ep_opd_v1/mu025',metric_names=list(METRICS),bindings=locked['bindings'],contrasts=cross,
        bootstrap_executed=False,GT_read=True))
    write_once(outputs[5],dict(schema='stvg-soft-ep-lr2-cpu-completion-v1',status='complete',case_count=32,row_count=320,summary_cell_count=100,
        paired_contrast_count=140,cross_trial_contrast_count=40,fit_invocations=69,GT_read=True,CPU_only=True,model_loaded=False,GPU_used=False,
        bootstrap_executed=False,metric_contract_sha256=sha(truth_reader.METRIC_CONTRACT),bindings=locked['bindings'],time=time.time()))
if __name__=='__main__': main()
