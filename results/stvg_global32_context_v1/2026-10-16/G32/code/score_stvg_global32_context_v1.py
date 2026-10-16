"""Matched32 postseal comparison with exact core A; all parents and signs retained."""
from pathlib import Path
import gc,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_global32_context_v1 import BASE,CORE,read,put,sha,verify
def aggregate(rows):
    summaries=[];pairs=[]
    for ds in ('hc2','vidstg'):
        rr=[r for r in rows if r['dataset']==ds];ids=sorted({r['case_id'] for r in rr});arms=list(dict.fromkeys(r['arm'] for r in rr));idx={(r['case_id'],r['arm']):r['metrics'] for r in rr}
        assert len(ids)==16 and len(idx)==len(rr)==16*len(arms)
        for a in arms:
            for metric in idx[ids[0],a]:
                v=[idx[c,a][metric] for c in ids if idx[c,a][metric] is not None];assert all(math.isfinite(x) for x in v)
                summaries.append(dict(dataset=ds,arm=a,metric=metric,valid=len(v),unknown=16-len(v),mean=math.fsum(v)/len(v) if v else None))
        for a,b in (('G32','A'),('G32','Temporal'),('G32','Frozen'),('A','Temporal'),('G32_Spatial','A_Spatial')):
            for metric in idx[ids[0],a]:
                v={c:100*(idx[c,a][metric]-idx[c,b][metric]) for c in ids if idx[c,a][metric] is not None and idx[c,b][metric] is not None}
                pairs.append(dict(dataset=ds,comparison=a+'_minus_'+b,metric=metric,valid=len(v),unknown=16-len(v),mean_delta_pp=math.fsum(v.values())/len(v) if v else None,positive=sum(x>1e-6 for x in v.values()),negative=sum(x< -1e-6 for x in v.values()),neutral=sum(abs(x)<=1e-6 for x in v.values()),per_source_delta_pp=v))
    return summaries,pairs
def contracts():
    rows=[dict(case_id=ds+str(i),dataset=ds,arm=a,metrics=dict(m_vIoU=i/1000+(0.01 if a=='G32' else 0),nullable=None if i%2 else .2)) for ds in ('hc2','vidstg') for i in range(16) for a in ('Frozen','Temporal','A','A_Spatial','G32','G32_Spatial')]
    s,p=aggregate(rows);assert all(r['unknown']==8 for r in s if r['metric']=='nullable')
    assert all(abs(r['mean_delta_pp']-1)<1e-10 for r in p if r['comparison']=='G32_minus_A' and r['metric']=='m_vIoU')
    try:aggregate(rows[:-1])
    except AssertionError:pass
    else:raise AssertionError('missing row accepted')
    print('GLOBAL32_POSTSEAL_SCHEMA_CONTRACTS_PASS_NO_GT')
def main():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_spatial_controls256_v1 import validate_barrier
    verify();tick=time.perf_counter();b=BASE/'trials/G32';r=read(b/'ROOT_READBACK.json');bar=read(b/'GLOBAL_PREDICTION_BARRIER.json');st=read(b/'STUDENT_GLOBAL_BARRIER.json')
    assert r['status']=='pass' and r['count']==32 and not r['GT_read'];assert bar['status']==st['status']=='complete' and bar['count']==st['count']==32 and not bar['GT_read'] and not st['GT_read']
    assert bar['root_sha256']==sha(b/'ROOT_READBACK.json') and r['barrier_sha256']==sha(b/'STUDENT_GLOBAL_BARRIER.json') and bar['files']==st['files']
    files={f['case_id']:f for f in bar['files']};rr=read(BASE/'ROSTER_LOCK.json')['records'];assert len(files)==32 and set(files)=={r['probe_alias'] for r in rr}
    for f in files.values():assert sha(ROOT/f['path'])==f['sha256'] and sha(ROOT/f['receipt_path'])==f['receipt_sha256']
    control=validate_barrier('A');assert read(CORE/'trials/A/CPU_COMPLETION002.json')['status']=='complete'
    # GT is accessed only after this new32 full prediction seal. Existing core
    # A/F/T scalar results are independently checked, not used for decisions.
    cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions()
    rows=[];usage=[];adopted_rows=[];geometry=[];refs={r['case_id']:r for r in read(b/'REFERENCE_LOCK.json')['records']};core_result=read(CORE/'trials/A/POSTSEAL_RESULTS002.json');original={(r['case_id'],r['arm']):r['metrics'] for r in core_result['rows']};core_geometry={(r['case_id'],r['slot']):r for r in core_result['geometry']}
    for rec in rr:
        cid=rec['probe_alias'];ds=rec['dataset'];case=cases[cid];d=torch.load(ROOT/files[cid]['path'],map_location='cpu',weights_only=False);a=torch.load(ROOT/control[cid]['path'],map_location='cpu',weights_only=False)
        assert d['temporal_record']==a['temporal_record'];assert np.array_equal(d['arms']['F']['boxes'],a['arms']['F']['boxes']) and np.array_equal(d['arms']['T']['boxes'],a['arms']['T']['boxes'])
        pp=CORE/'spatial/E1/proposals'/(cid+'.json');pool=read(pp) if pp.exists() else dict(windows=[]);common=[w['source_grid_position'] for w in pool['windows']];adopted=[x['position'] for x in refs[cid]['anchors']]
        for name,pred in [('Frozen',d['arms']['F']),('Temporal',d['arms']['T']),('A',a['arms']['TS_M2']),('A_Spatial',a['arms']['S_M2']),('G32',d['arms']['TS_M2']),('G32_Spatial',d['arms']['S_M2'])]:
            values,audit=metric._score_student_arm(pred,case,targets[cid],common,official,independent)
            if name in ('Frozen','Temporal','A','A_Spatial'):assert values==original[cid,name]
            rows.append(dict(case_id=cid,dataset=ds,arm=name,metrics=values,GT_read=True))
            values,_=metric._score_student_arm(pred,case,targets[cid],adopted,official,independent);adopted_rows.append(dict(case_id=cid,dataset=ds,arm=name,metrics=values,GT_read=True))
        for w in pool['windows']:
            cg=core_geometry[cid,w['slot']];anchor=next((x for x in refs[cid]['anchors'] if x['position']==w['source_grid_position']),None)
            selected=next((x for x in cg['candidates'] if anchor and x['candidate_id']==anchor['candidate_id']),None)
            geometry.append(dict(case_id=cid,dataset=ds,slot=w['slot'],known_GT=cg['known_GT'],core_A_adopted_candidate_id=cg['adopted_candidate_id'],core_A_adopted_geometry=cg['adopted_geometry'],G32_adopted_candidate_id=anchor['candidate_id'] if anchor else None,G32_adopted_geometry=selected['diagnostic'] if selected else None,GT_read=True))
        fit=d['fits']['M2'];usage.append(dict(case_id=cid,dataset=ds,anchors=len(adopted),actual_fit=fit is not None,accepted_steps=fit['optimizer_steps'] if fit else 0,nonzero_final=any(not torch.equal(v.cpu(),fit['final_state'][k].cpu()) for k,v in fit['initial_state'].items()) if fit else False,fit_seconds=fit.get('instance_study_fit_seconds') if fit else None,decisions=refs[cid]['decisions'],GT_read=False));del d,a;gc.collect()
    summary,pairs=aggregate(rows);a_s,a_p=aggregate(adopted_rows)
    put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,per_domain=16,rows=rows,summary=summary,paired=pairs,actual_support_rows=adopted_rows,actual_support_summary=a_s,actual_support_paired=a_p,usage=usage,geometry=geometry,GT_bindings=bindings,GT_read=True,bootstrap=False,semantic_status='pending separate prelocked arm-blind32 pixel review',prediction_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'),core_A_prediction_sha256=sha(CORE/'trials/A/GLOBAL_PREDICTION_BARRIER.json')))
    put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_read=True,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),wall_seconds=time.perf_counter()-tick))
if __name__=='__main__':contracts() if sys.argv[1]=='contracts' else main()
