"""Fixed32 sealed prediction scoring and complete paired summaries."""
from pathlib import Path
import gc,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,LITE,read,put,sha
def aggregate(rows):
    summary=[];paired=[];idx={(r['case_id'],r['arm']):r for r in rows};arms=list(dict.fromkeys(r['arm']for r in rows))
    comparisons=[(arms[-1],a)for a in arms[:-1]]if arms[-1]=='Corrected_Temporal'else[('E1','E0'),('E1','E2'),('E0','Temporal'),('E1','Temporal'),('E1','Frozen'),('E2','Temporal')]
    for ds in ('hc2','vidstg'):
        ids=sorted({r['case_id']for r in rows if r['dataset']==ds});assert len(ids)==16
        for arm in arms:
            for key in idx[ids[0],arm]['metrics']:
                vals=[idx[c,arm]['metrics'][key]for c in ids];valid=[v for v in vals if v is not None]
                summary.append(dict(dataset=ds,arm=arm,metric=key,count=16,valid=len(valid),unknown=len(vals)-len(valid),mean=math.fsum(valid)/len(valid)if valid else None))
        for a,b in comparisons:
            for key in idx[ids[0],a]['metrics']:
                vals={c:100*(idx[c,a]['metrics'][key]-idx[c,b]['metrics'][key])for c in ids if idx[c,a]['metrics'][key]is not None and idx[c,b]['metrics'][key]is not None};vv=list(vals.values())
                paired.append(dict(dataset=ds,comparison=a+'_minus_'+b,metric=key,count=len(vals),unknown=16-len(vals),mean_delta_pp=math.fsum(vv)/len(vv)if vv else None,positive=sum(v>1e-6 for v in vv),negative=sum(v< -1e-6 for v in vv),neutral=sum(abs(v)<=1e-6 for v in vv),per_source_delta_pp=vals))
    return summary,paired
def temporal():
    import torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    b=BASE/'temporal';bar=read(b/'GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete'and bar['count']==32 and bar['GT_read']is False and bar['root_sha256']==sha(b/'ROOT_READBACK.json')
    rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in rr};files={r['case_id']:r for r in bar['files']}
    targets,gt=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[]
    for rec in rr:
        cid=rec['probe_alias'];f=files[cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);case=cases[cid]
        intervals={'Frozen':case['Native_interval'],'Legacy_Temporal':d['legacy_interval'],'Corrected_Temporal':d['selected_interval']}
        for name,interval in intervals.items():
            score,audit=metric._score_student_arm(dict(boxes=case['native_boxes_xyxy_normalized'],interval=interval),case,targets[cid],[],official,independent)
            rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=score,fit_status='actual_fit'if d['fit']else'original_fallback',GT_read=True))
        del d;gc.collect()
    summary,paired=aggregate(rows);put(b/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=paired,GT_read=True,prediction_barrier_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'),GT_bindings=gt))
    put(b/'CPU_COMPLETION.json',dict(status='complete',count=32,row_count=len(rows),GT_read=True,result_sha256=sha(b/'POSTSEAL_RESULTS.json')))
    print(json.dumps({'status':'complete','paired':[p for p in paired if p['metric']=='m_vIoU']}))
if __name__=='__main__':{'temporal':temporal}[sys.argv[1]]()
