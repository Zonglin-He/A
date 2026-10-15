#!/usr/bin/env python3
"""Post-seal scoring for the fixed 32-source Soft EP-OPD comparison."""
from __future__ import annotations
import json, math, os, statistics, time, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1'
PARENT=ROOT/'artifacts/stvg_evidence_constrained_v1/round2_prompt009'
ARMS=('F','T','S_M0','TS_M0','S_M1','TS_M1','S_M2','TS_M2')
METRICS=('m_vIoU','m_tIoU','m_sIoU_dense_GT','m_sIoU_supervised_GT','m_sIoU_unsupervised_GT')
CONTRASTS=(('S_M2_minus_S_M1','S_M2','S_M1'),('TS_M2_minus_TS_M1','TS_M2','TS_M1'),
 ('S_M2_minus_S_M0','S_M2','S_M0'),('TS_M2_minus_TS_M0','TS_M2','TS_M0'),
 ('S_M1_minus_S_M0','S_M1','S_M0'),('TS_M1_minus_TS_M0','TS_M1','TS_M0'),
 ('S_M0_minus_F','S_M0','F'),('TS_M0_minus_T','TS_M0','T'),
 ('S_M1_minus_F','S_M1','F'),('TS_M1_minus_T','TS_M1','T'),
 ('S_M2_minus_F','S_M2','F'),('TS_M2_minus_T','TS_M2','T'))
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
    p=(ROOT/s).resolve()
    need(p==ROOT or ROOT in p.parents,f'path escapes repository: {s}')
    return p
def filemap(rows,label):
    need(isinstance(rows,list),f'{label}.files must be a list'); out={}
    for x in rows:
        need(isinstance(x,dict) and isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str),f'bad {label} file row')
        rel=x['path']; p=rootpath(rel); need(p.is_file() and sha(p)==x['sha256'],f'{label} SHA mismatch: {rel}')
        if x.get('bytes') is not None: need(p.stat().st_size==int(x['bytes']),f'{label} byte mismatch: {rel}')
        need(rel not in out,f'duplicate {label} path: {rel}'); out[rel]=x
    return out
def load_pt(path):
    import torch
    # These local tensor artifacts contain NumPy arrays; retain restricted unpickling.
    import numpy as n
    safe=[n.ndarray,n.dtype]
    for mod in (getattr(n,'core',None),getattr(n,'_core',None)):
        ma=getattr(mod,'multiarray',None) if mod is not None else None
        for name in ('_reconstruct','scalar'):
            if ma is not None and hasattr(ma,name): safe.append(getattr(ma,name))
    safe += [kind for t in (n.float16,n.float32,n.float64,n.int8,n.int16,n.int32,n.int64,n.uint8,n.bool_) for kind in (n.dtype(t).type,type(n.dtype(t)))]
    with torch.serialization.safe_globals(safe): return torch.load(path,map_location='cpu',weights_only=True)
def xyxy(cxcywh):
    x=np.asarray(cxcywh.detach().cpu().double().numpy() if hasattr(cxcywh,'detach') else cxcywh,dtype=np.float64)
    return np.concatenate((x[:,:2]-x[:,2:]/2,x[:,:2]+x[:,2:]/2),axis=1)
def same_array(a,b):
    x,y=np.asarray(a),np.asarray(b)
    return x.shape==y.shape and x.dtype==y.dtype and np.array_equal(x,y)
def _gates():
    rt=read(BASE/'RUNTIME.json'); need(rt.get('status')=='locked' and rt.get('GT_read') is False,'runtime is not locked GT-free')
    need(isinstance(rt.get('pins'),dict) and isinstance(rt.get('protected'),dict),'runtime pins malformed')
    for rel,h in {**rt['pins'],**rt['protected']}.items(): need(sha(rootpath(rel))==h,f'runtime pin mismatch: {rel}')
    design=read(BASE/'DESIGN_LOCK.json'); roster_doc=read(BASE/'ROSTER_LOCK.json'); ref_doc=read(BASE/'REFERENCE_LOCK.json')
    need(design.get('status')=='locked' and design.get('GT_read') is False,'design lock invalid')
    need(roster_doc.get('status')=='locked' and roster_doc.get('GT_read') is False and roster_doc.get('count')==32,'roster lock invalid')
    need(ref_doc.get('status')=='locked' and ref_doc.get('GT_read') is False and ref_doc.get('count')==32,'reference lock invalid')
    roster=roster_doc['records']; refs={r['case_id']:r for r in ref_doc['records']}
    ids=[r['probe_alias'] for r in roster]; need(len(set(ids))==32 and set(ids)==set(refs),'roster/reference membership mismatch')
    need(sum(len(refs[c]['anchors']) for c in ids)==int(design['reference_anchor_count']),'reference anchor total mismatch')
    sbp=BASE/'STUDENT_GLOBAL_BARRIER.json'; sb=read(sbp)
    need(sb.get('status')=='complete' and sb.get('count')==32 and sb.get('GT_read') is False and sb.get('runtime_sha256')==sha(BASE/'RUNTIME.json'),'student barrier invalid')
    sfiles=filemap(sb.get('files'),'student barrier'); expected={str((BASE/'student'/f'{c}.pt').relative_to(ROOT)) for c in ids}
    need(set(sfiles)==expected,'student barrier file set mismatch')
    rp=BASE/'ROOT_READBACK.json'; rr=read(rp)
    need(rr.get('status')=='pass' and rr.get('count')==32 and rr.get('GT_read') is False,'independent pre-GT readback invalid')
    gp=BASE/'GLOBAL_PREDICTION_BARRIER.json'; gb=read(gp)
    need(gb.get('status')=='complete' and gb.get('count')==32 and gb.get('GT_read') is False,'prediction seal invalid')
    need(gb.get('student_barrier_sha256')==sha(sbp) and gb.get('root_readback_sha256')==sha(rp),'prediction seal does not bind student/root readbacks')
    gfiles=filemap(gb.get('files'),'prediction seal'); need(expected <= set(gfiles) and all(gfiles[p]['sha256']==sfiles[p]['sha256'] for p in expected),'prediction seal file set differs')
    need(gb.get('scorer_sha256')==sha(Path(__file__)),'scorer changed after prediction seal')
    from scripts.stvg_evidence_common_v1 import load_case as _load_case
    import torch
    parent_bar=read(PARENT/'STUDENT_GLOBAL_BARRIER.json'); need(parent_bar.get('status')=='complete' and parent_bar.get('count')==32 and parent_bar.get('GT_read') is False,'parent barrier invalid')
    pfiles=filemap(parent_bar.get('files'),'parent student barrier'); cases={}; payloads={}; refs_out={}
    for rec in roster:
        cid=rec['probe_alias']; ref=refs[cid]; case,srcpath=_load_case(rec); case=dict(case); case.setdefault('dataset',rec['dataset']); case.setdefault('query_ordinal',rec['query_ordinal'])
        need(case.get('GT_read') is False and case.get('dataset')==rec['dataset'] and case.get('source_case_sha256')==ref.get('source_case_sha256'),f'source identity mismatch: {cid}')
        rel=str((BASE/'student'/f'{cid}.pt').relative_to(ROOT)); path=rootpath(rel); receipt=read(path.with_suffix('.receipt.json'))
        need(receipt.get('status')=='complete' and receipt.get('GT_read') is False and receipt.get('case_id')==cid and receipt.get('sha256')==sha(path),'student receipt mismatch: '+cid)
        z=load_pt(path); need(z.get('schema')=='stvg-soft-ep-case-v1' and z.get('case_id')==cid and z.get('dataset')==rec['dataset'] and z.get('GT_read') is False,'student payload identity mismatch: '+cid)
        need(z.get('reference_records')==ref and z.get('source_case_sha256')==case['source_case_sha256'] and z.get('source_file_sha256')==sha(srcpath) and z.get('runtime_sha256')==sha(BASE/'RUNTIME.json') and z.get('source_restored') is True,'student custody mismatch: '+cid)
        pp=PARENT/'student'/f'{cid}.pt'; prel=str(pp.relative_to(ROOT)); need(prel in pfiles and z.get('parent_student_sha256')==pfiles[prel]['sha256'] and sha(pp)==pfiles[prel]['sha256'],'parent prediction mismatch: '+cid)
        parent=load_pt(pp); need(parent.get('GT_read') is False and parent.get('source_case_sha256')==case['source_case_sha256'],'parent source identity mismatch: '+cid)
        arms=z['arms']; need(set(arms)==set(ARMS),'arm set mismatch: '+cid)
        for dst,src in [('F','F'),('T','T'),('S_M0','S_new_crop'),('TS_M0','TS_new_crop')]:
            need(same_array(arms[dst]['boxes'],parent['arms'][src]['boxes']) and list(arms[dst]['interval'])==list(parent['arms'][src]['interval']),f'M0/control reuse mismatch {cid}/{dst}')
        need(list(map(int,z['frame_ids']))==list(map(int,case['frame_ids'])),'source frame grid mismatch: '+cid)
        pos=[int(a['position']) for a in ref['anchors']]; fits=z['fits']; outs=z['readouts']; need(set(fits)=={'M1','M2'} and set(outs)=={'M1','M2'},'fit/readout keys invalid: '+cid)
        for m in ('M1','M2'):
            if not pos:
                need(fits[m] is None and outs[m] is None and np.array_equal(arms['S_'+m]['boxes'],arms['F']['boxes']) and np.array_equal(arms['TS_'+m]['boxes'],arms['T']['boxes']),f'empty-reference fallback mismatch: {cid}/{m}'); continue
            f= fits[m]; o=outs[m]
            need(f.get('GT_read') is False and f.get('mode')==m and f.get('active_parameters')==1792 and f.get('config')==design['configs'][rec['dataset']] and f.get('config',{}).get('steps')==5 and f.get('observation_positions')==pos and len(f.get('path',[]))==5,'fit contract mismatch: '+cid+'/'+m)
            need(o['boxes'].dtype==f['final_cached_decoder_boxes'].dtype and torch.equal(o['boxes'].detach().cpu(),f['final_cached_decoder_boxes'].detach().cpu()),'full/cached boxes mismatch: '+cid+'/'+m)
            want=xyxy(o['boxes']); need(np.array_equal(np.asarray(arms['S_'+m]['boxes'],dtype=np.float64),want) and np.array_equal(np.asarray(arms['TS_'+m]['boxes'],dtype=np.float64),want),'scored arm/readout mismatch: '+cid+'/'+m)
            need(list(arms['S_'+m]['interval'])==list(arms['F']['interval']) and list(arms['TS_'+m]['interval'])==list(arms['T']['interval']),'paired interval mismatch: '+cid+'/'+m)
        cases[cid]=case; payloads[cid]=z; refs_out[cid]=pos
    return dict(design=design,roster=roster,refs=refs,cases=cases,payloads=payloads,positions=refs_out,
        bindings={'runtime_sha256':sha(BASE/'RUNTIME.json'),'design_lock_sha256':sha(BASE/'DESIGN_LOCK.json'),'roster_sha256':sha(BASE/'ROSTER_LOCK.json'),'reference_lock_sha256':sha(BASE/'REFERENCE_LOCK.json'),'parent_student_barrier_sha256':sha(PARENT/'STUDENT_GLOBAL_BARRIER.json'),'student_barrier_sha256':sha(sbp),'root_readback_sha256':sha(rp),'prediction_seal_sha256':sha(gp)})

def aggregate(rows):
    by={(r['case_id'],r['arm']):r for r in rows}; out=[]
    for ds in ('hc2','vidstg'):
        cids=sorted({r['case_id'] for r in rows if r['dataset']==ds}); need(len(cids)==16,f'{ds} denominator mismatch')
        for arm in ARMS:
            for metric in METRICS:
                vals=[]; unknown=[]
                for c in cids:
                    v=by[c,arm]['metrics'].get(metric)
                    if v is None: unknown.append(c)
                    else: need(math.isfinite(float(v)),f'nonfinite {metric}'); vals.append(float(v))
                out.append(dict(dataset=ds,arm=arm,metric=metric,n_valid=len(vals),n_unknown=len(unknown),unknown_case_ids=unknown,mean=math.fsum(vals)/len(vals) if vals else None,median=statistics.median(vals) if vals else None,min=min(vals) if vals else None,max=max(vals) if vals else None))
    return out
def paired(rows):
    by={(r['case_id'],r['arm']):r for r in rows}; out=[]
    for ds in ('hc2','vidstg'):
        cids=sorted({r['case_id'] for r in rows if r['dataset']==ds})
        for name,left,right in CONTRASTS:
            for met in METRICS:
                vals={}; unknown=[]
                for c in cids:
                    a=by[c,left]['metrics'].get(met); b=by[c,right]['metrics'].get(met)
                    if a is None or b is None: unknown.append(c)
                    else: vals[c]=100*(float(a)-float(b))
                out.append(dict(dataset=ds,contrast=name,comparison_arm=left,baseline_arm=right,metric=met,n_pairs=len(vals),unknown_count=len(unknown),unknown_case_ids=unknown,positive_case_ids=sorted(c for c,v in vals.items() if v>EPS),negative_case_ids=sorted(c for c,v in vals.items() if v< -EPS),neutral_case_ids=sorted(c for c,v in vals.items() if abs(v)<=EPS),neutral_tolerance_pp=EPS,mean_delta_pp=math.fsum(vals.values())/len(vals) if vals else None,per_source_delta_pp=vals))
    return out
def write_once(path,obj):
    need(not path.exists(),f'refusing to overwrite {path.name}'); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp'); need(not tmp.exists(),f'stale temp {tmp.name}')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); os.replace(tmp,path)
def main():
    outpaths=[BASE/'GT_EXPOSURE.json',BASE/'POSTSEAL_SCORE_ROWS.json',BASE/'POSTSEAL_SCORE_SUMMARY.json',BASE/'POSTSEAL_PAIRED_CONTRASTS.json',BASE/'CPU_COMPLETION.json']
    need(not any(p.exists() for p in outpaths),'score artifacts already exist; refusing a second GT pass')
    locked=_gates()
    exposure=dict(schema='stvg-soft-ep-gt-exposure-v1',status='after_complete_student_seal_and_independent_root_readback',bindings=locked['bindings'],GT_read=True,time=time.time())
    write_once(outpaths[0],exposure)
    from scripts import review_stvg_sparse_r1_v2 as truth_reader
    from scripts import score_stvg_native_instance_nativeclock005 as scorer
    targets, _=truth_reader._truths_after_seal(locked['cases']); official,independent,contract=truth_reader._metric_functions()
    rows=[]
    for cid in [r['probe_alias'] for r in locked['roster']]:
        z=locked['payloads'][cid]; case=locked['cases'][cid]
        for arm in ARMS:
            metrics,audit=scorer._score_student_arm(z['arms'][arm],case,targets[cid],set(locked['positions'][cid]),official,independent)
            fit_tag=arm.split('_',1)[1] if arm.startswith(('S_','TS_')) else None
            fit=(z['fits'].get(fit_tag) if fit_tag in ('M1','M2') else None)
            fs={'kind':'reference_fit','optimizer_steps':fit.get('optimizer_steps'),'proposed_steps':fit.get('proposed_optimizer_steps'),'rejected_steps':fit.get('rejected_optimizer_steps'),'declared_steps':fit['config']['steps'],'active_parameters':fit.get('active_parameters')} if fit else {'kind':'exact_parent_reuse' if arm in ('S_M0','TS_M0') else 'no_fit'}
            rows.append(dict(case_id=cid,dataset=z['dataset'],arm=arm,status='scored',metrics=metrics,support=audit['support'],metric_audit=audit['metric_audit'],fit=fs))
    need(len(rows)==256,'expected exactly 32x8 rows'); sums=aggregate(rows); pairs=paired(rows)
    need(len(sums)==80 and len(pairs)==120,'summary/paired cardinality mismatch')
    common=dict(schema='stvg-soft-ep-postseal-v1',status='complete',case_count=32,row_count=256,metric_names=list(METRICS),bindings=locked['bindings'],GT_read=True,bootstrap_executed=False)
    write_once(outpaths[1],dict(common,rows=rows)); write_once(outpaths[2],dict(common,summary=sums)); write_once(outpaths[3],dict(common,paired_contrasts=pairs))
    write_once(outpaths[4],dict(schema='stvg-soft-ep-cpu-completion-v1',status='complete',case_count=32,row_count=256,summary_cell_count=80,paired_contrast_count=120,contrast_definitions=[list(x) for x in CONTRASTS],GT_read=True,CPU_only=True,model_loaded=False,GPU_used=False,bootstrap_executed=False,metric_contract_sha256=sha(truth_reader.METRIC_CONTRACT),bindings=locked['bindings'],time=time.time()))
if __name__=='__main__': main()
