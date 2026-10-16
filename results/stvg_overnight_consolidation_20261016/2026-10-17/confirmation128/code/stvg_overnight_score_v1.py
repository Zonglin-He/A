"""Full-cohort post-seal scoring and frozen development gates; no online labels."""
from pathlib import Path
import gc,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
read,sha,put=c.read,c.sha,c.write
def seal(base):
    b=base/'trials/G32';r=read(b/'ROOT_READBACK.json');g=read(b/'GLOBAL_PREDICTION_BARRIER.json');s=read(b/'STUDENT_GLOBAL_BARRIER.json');n=read(base/'ROSTER_LOCK.json')['count']
    assert r['status']=='pass' and g['status']==s['status']=='complete' and r['count']==g['count']==s['count']==n and not any(x['GT_read'] for x in (r,g,s))
    assert g['root_sha256']==sha(b/'ROOT_READBACK.json') and r['barrier_sha256']==sha(b/'STUDENT_GLOBAL_BARRIER.json') and g['files']==s['files']
    f={r['case_id']:r for r in g['files']};assert len(f)==n and set(f)=={r['probe_alias'] for r in read(base/'ROSTER_LOCK.json')['records']}
    for x in f.values():assert sha(ROOT/x['path'])==x['sha256'] and sha(ROOT/x['receipt_path'])==x['receipt_sha256']
    return f
def aggregate(rows,n):
    summaries=[];pairs=[];arms=list(dict.fromkeys(r['arm'] for r in rows))
    for ds in ('hc2','vidstg'):
        rr=[r for r in rows if r['dataset']==ds];ids=sorted({r['case_id'] for r in rr});idx={(r['case_id'],r['arm']):r['metrics'] for r in rr};assert len(ids)==n//2 and len(idx)==len(rr)==n//2*len(arms)
        for arm in arms:
            for metric in idx[ids[0],arm]:
                vals=[idx[i,arm][metric] for i in ids if idx[i,arm][metric] is not None];assert all(math.isfinite(v) for v in vals)
                summaries.append(dict(dataset=ds,arm=arm,metric=metric,valid=len(vals),unknown=n//2-len(vals),mean=math.fsum(vals)/len(vals) if vals else None))
        for ai,a in enumerate(arms):
            for b in arms[:ai]:
                for metric in idx[ids[0],a]:
                    v={i:100*(idx[i,a][metric]-idx[i,b][metric]) for i in ids if idx[i,a][metric] is not None and idx[i,b][metric] is not None}
                    pairs.append(dict(dataset=ds,comparison=a+'_minus_'+b,metric=metric,valid=len(v),unknown=n//2-len(v),mean_delta_pp=math.fsum(v.values())/len(v) if v else None,positive=sum(x>1e-6 for x in v.values()),negative=sum(x< -1e-6 for x in v.values()),neutral=sum(abs(x)<=1e-6 for x in v.values()),per_source_delta_pp=v))
    return summaries,pairs
def score(scope):
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    c.verify();base=c.BASE/scope;cfg=read(base/'SCOPE_CONFIG.json');core=ROOT/cfg['core_scope'];files=seal(base);rr=read(base/'ROSTER_LOCK.json')['records'];n=len(rr);control=cfg['local_control'];old=c.score(control);oldidx={(x['case_id'],x['arm']):x['metrics'] for x in old['rows']}
    from scripts.score_stvg_spatial_controls256_v1 import validate_barrier
    controls=validate_barrier(control);cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();refs={r['case_id']:r for r in read(base/'trials/G32/REFERENCE_LOCK.json')['records']};rows=[];adopted_rows=[];usage=[];tick=time.perf_counter()
    for rec in rr:
        cid=rec['probe_alias'];d=torch.load(ROOT/files[cid]['path'],map_location='cpu',weights_only=False);a=torch.load(ROOT/controls[cid]['path'],map_location='cpu',weights_only=False);assert d['temporal_record']==a['temporal_record']
        for k in ('F','T'):assert np.array_equal(d['arms'][k]['boxes'],a['arms'][k]['boxes']) and np.array_equal(d['arms'][k]['interval'],a['arms'][k]['interval'])
        pp=core/'spatial/E1/proposals'/(cid+'.json');pool=read(pp) if pp.exists() else dict(windows=[]);common=[x['source_grid_position'] for x in pool['windows']];adopted=[x['position'] for x in refs[cid]['anchors']]
        for name,pred in [('Frozen',d['arms']['F']),('Temporal',d['arms']['T']),('LocalB',a['arms']['TS_M2']),('LocalB_Spatial',a['arms']['S_M2']),('Candidate',d['arms']['TS_M2']),('Candidate_Spatial',d['arms']['S_M2'])]:
            vals,_=metric._score_student_arm(pred,cases[cid],targets[cid],common,official,independent)
            if name in ('Frozen','Temporal','LocalB','LocalB_Spatial'):assert vals==oldidx[cid,{'LocalB':control,'LocalB_Spatial':control+'_Spatial'}.get(name,name)]
            rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=vals,GT_read=True));vals,_=metric._score_student_arm(pred,cases[cid],targets[cid],adopted,official,independent);adopted_rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=vals,GT_read=True))
        f=d['fits']['M2'];usage.append(dict(case_id=cid,dataset=rec['dataset'],anchors=len(adopted),actual_fit=f is not None,accepted_steps=f['optimizer_steps'] if f else 0,nonzero_final=any(not torch.equal(v.cpu(),f['final_state'][k].cpu()) for k,v in f['initial_state'].items()) if f else False,GT_read=False));del d,a;gc.collect()
    s,p=aggregate(rows,n);s2,p2=aggregate(adopted_rows,n);put(base/'POSTSEAL_RESULTS.json',dict(status='complete',count=n,per_domain=n//2,rows=rows,summary=s,paired=p,actual_support_rows=adopted_rows,actual_support_summary=s2,actual_support_paired=p2,usage=usage,GT_bindings=bindings,prediction_sha256=sha(base/'trials/G32/GLOBAL_PREDICTION_BARRIER.json'),semantic_status='pending separate prelocked arm-blind pixel review; not inferred from metrics',GT_read=True,bootstrap=False))
    put(base/'CPU_COMPLETION.json',dict(status='complete',count=n,GT_read=True,result_sha256=sha(base/'POSTSEAL_RESULTS.json'),wall_seconds=time.perf_counter()-tick));put(base/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready_for_independent_root_view_public_archive',count=n,GT_read=True))
def contracts():
    rows=[dict(case_id=ds+str(i),dataset=ds,arm=a,metrics=dict(m_vIoU=.1+(a=='Candidate')*.01,unknown=None if i%2 else .2)) for ds in ('hc2','vidstg') for i in range(16) for a in ('Frozen','Temporal','LocalB','Candidate')]
    s,p=aggregate(rows,32);assert all(abs(x['mean_delta_pp']-1)<1e-12 for x in p if x['comparison']=='Candidate_minus_LocalB' and x['metric']=='m_vIoU')
    for bad in (rows[:-1],rows+rows[:1]):
        try:aggregate(bad,32)
        except AssertionError:pass
        else:raise AssertionError('bad coverage accepted')
    print('NIGHT_POSTSEAL_AGGREGATION_3_CONTRACTS_PASS_NO_GT')
if __name__=='__main__':contracts() if sys.argv[1]=='contracts' else score(sys.argv[1])
