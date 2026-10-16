"""Official/independent metric cross-check after all 128 predictions are sealed."""
from pathlib import Path
import sys,time,math,statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *
from scripts.measure_stvg_soft_ep_v1 import ARMS,METRICS,CONTRASTS
def main():
 import torch
 from scripts import review_stvg_sparse_r1_v2 as truth_reader
 from scripts import score_stvg_native_instance_nativeclock005 as scorer
 b=BASE/'trials'/sys.argv[1];assert not(b/'GT_EXPOSURE.json').exists();rt=read(b/'RUNTIME.json')
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
 bar=read(b/'GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete'and bar['count']==128 and not bar['GT_read']and bar['scorer_sha256']==sha(__file__)and bar['runtime_sha256']==sha(b/'RUNTIME.json')
 rosterdoc=read(BASE/'ROSTER_LOCK.json');rs=rosterdoc['records'];assert rosterdoc['status']=='locked'and rosterdoc['count']==128 and not rosterdoc['GT_read']and len(rs)==128
 aliases=[r['probe_alias']for r in rs];assert len(set(aliases))==128==len({(r['dataset'],r['parent_sha256'])for r in rs});assert all(sum(r['dataset']==ds for r in rs)==64 for ds in ['hc2','vidstg'])
 sb=read(b/'STUDENT_GLOBAL_BARRIER.json');assert sb['status']=='complete'and sb['count']==128 and not sb['GT_read']and sb['runtime_sha256']==sha(b/'RUNTIME.json')
 expected={str((b/'student'/(c+'.pt')).relative_to(ROOT))for c in aliases};sf={x['path']:x for x in sb['files']};gf={x['path']:x for x in bar['files']};assert len(sf)==len(sb['files'])==128 and set(sf)==expected and len(gf)==len(bar['files'])and {p for p in gf if p.endswith('.pt')}==expected
 for x in bar['files']:
  p=ROOT/x['path'];assert p.is_file()and sha(p)==x['sha256']
  if 'bytes'in x:assert p.stat().st_size==x['bytes']
 assert all(sf[p]['sha256']==gf[p]['sha256']for p in expected)
 root=read(b/'ROOT_READBACK.json');assert root['status']=='pass'and root['count']==128 and not root['GT_read']and root['runtime_sha256']==sha(b/'RUNTIME.json')and root['barrier_sha256']==sha(b/'STUDENT_GLOBAL_BARRIER.json')
 for n in ['ROOT_READBACK.json','STUDENT_GLOBAL_BARRIER.json','RUNTIME.json','DESIGN_LOCK.json']:assert gf[str((b/n).relative_to(ROOT))]['sha256']==sha(b/n)
 binding=dict(runtime_sha256=sha(b/'RUNTIME.json'),prediction_seal_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'),root_readback_sha256=sha(b/'ROOT_READBACK.json'),roster_sha256=sha(BASE/'ROSTER_LOCK.json'),reference_sha256=sha(BASE/'REFERENCE_LOCK.json'))
 cases={};payload={};refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']}
 for r in roster():
  cid=r['probe_alias'];cases[cid],_=load_case(r);p=b/'student'/(cid+'.pt');assert sha(p)==sf[str(p.relative_to(ROOT))]['sha256']==read(p.with_suffix('.receipt.json'))['sha256'];payload[cid]=torch.load(p,map_location='cpu',weights_only=False);d=payload[cid];assert set(d['arms'])==set(ARMS)and d['case_id']==cid and d['dataset']==r['dataset']and d['source_case_sha256']==cases[cid]['source_case_sha256']and d['runtime_sha256']==sha(b/'RUNTIME.json')and not d['GT_read']and d['reference_records']==refs[cid]
 write(b/'GT_EXPOSURE.json',dict(status='all128_predictions_sealed_and_independently_read',GT_read=True,bindings=binding,time=time.time()))
 truth,_=truth_reader._truths_after_seal(cases);official,independent,contract=truth_reader._metric_functions();rows=[]
 for r in roster():
  cid=r['probe_alias'];z=payload[cid];case=cases[cid];pos={a['position']for a in refs[cid]['anchors']}
  for arm in ARMS:
   metrics,audit=scorer._score_student_arm(z['arms'][arm],case,truth[cid],pos,official,independent)
   rows.append(dict(case_id=cid,dataset=r['dataset'],arm=arm,metrics=metrics,support=audit['support'],metric_audit=audit['metric_audit']))
 assert len(rows)==1024;by={(r['case_id'],r['arm']):r for r in rows};summary=[];paired=[]
 for ds in ['hc2','vidstg']:
  ids=[r['probe_alias']for r in roster()if r['dataset']==ds];assert len(ids)==64
  for arm in ARMS:
   for metric in METRICS:
    values=[by[c,arm]['metrics'][metric]for c in ids if by[c,arm]['metrics'][metric]is not None];unknown=[c for c in ids if by[c,arm]['metrics'][metric]is None];assert all(math.isfinite(v)for v in values)
    summary.append(dict(dataset=ds,arm=arm,metric=metric,n_valid=len(values),n_unknown=len(unknown),unknown_case_ids=unknown,mean=math.fsum(values)/len(values)if values else None,median=statistics.median(values)if values else None,min=min(values)if values else None,max=max(values)if values else None))
  for name,left,right in CONTRASTS:
   for metric in METRICS:
    v={c:100*(by[c,left]['metrics'][metric]-by[c,right]['metrics'][metric])for c in ids if by[c,left]['metrics'][metric]is not None and by[c,right]['metrics'][metric]is not None};unknown=[c for c in ids if c not in v]
    paired.append(dict(dataset=ds,contrast=name,metric=metric,n_pairs=len(v),unknown_count=len(unknown),unknown_case_ids=unknown,mean_delta_pp=math.fsum(v.values())/len(v)if v else None,per_source_delta_pp=v,positive_case_ids=[c for c,x in v.items()if x>1e-10],negative_case_ids=[c for c,x in v.items()if x< -1e-10],neutral_case_ids=[c for c,x in v.items()if abs(x)<=1e-10]))
 common=dict(status='complete',count=128,row_count=1024,bindings=binding,GT_read=True,bootstrap_executed=False)
 write(b/'POSTSEAL_SCORE_ROWS.json',dict(common,rows=rows));write(b/'POSTSEAL_SCORE_SUMMARY.json',dict(common,summary=summary));write(b/'POSTSEAL_PAIRED_CONTRASTS.json',dict(common,paired_contrasts=paired));write(b/'CPU_COMPLETION.json',dict(common,summary_count=len(summary),paired_count=len(paired),CPU_only=True,model_loaded=False,GPU_used=False,time=time.time()))
 print('scored128',len(rows),len(summary),len(paired))
if __name__=='__main__':main()
