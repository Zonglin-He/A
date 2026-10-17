"""Independent arithmetic on sealed scalar rows only; no model or GT reader."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_spatial_caption_repair_v1 import FIXED,read,put,sha,verify
def main():
 verify();d=read(FIXED/'POSTSEAL_RESULTS.json');c=read(FIXED/'CPU_COMPLETION.json');assert c['status']=='complete'and c['count']==256 and c['result_sha256']==sha(FIXED/'POSTSEAL_RESULTS.json')
 idx={(r['case_id'],r['arm']):r for r in d['rows']};assert len(idx)==len(d['rows']);groups=collections.defaultdict(list)
 for r in d['rows']:
  for k,v in r['metrics'].items():
   assert v is None or math.isfinite(v)
   groups[r['dataset'],r['arm'],k].append(v)
 summary=0;nulls=0
 for s in d['summary']:
  a=groups[s['dataset'],s['arm'],s['metric']];v=[v for v in a if v is not None];assert len(a)==128 and len(v)==s['valid']and 128-len(v)==s['unknown'];m=math.fsum(v)/len(v)if v else None;assert m==s['mean'];summary+=1;nulls+=len(a)-len(v)
 arms=list(dict.fromkeys(r['arm']for r in d['rows']));pairs=0;points=0
 for p in d['paired']:
  options=[(a,b)for a in arms for b in arms if a+'_minus_'+b==p['comparison']];assert len(options)==1;a,b=options[0];ids=[cid for cid,arm in idx if arm==a and idx[cid,arm]['dataset']==p['dataset']];vals={}
  for cid in ids:
   x=idx[cid,a]['metrics'][p['metric']];y=idx[cid,b]['metrics'][p['metric']]
   if x is not None and y is not None:vals[cid]=100*(x-y)
  assert vals==p['per_source_delta_pp']and len(vals)==p['valid']and 128-len(vals)==p['unknown'];assert (math.fsum(vals.values())/len(vals)if vals else None)==p['mean_delta_pp'];assert (sum(x>1e-6 for x in vals.values()),sum(x< -1e-6 for x in vals.values()),sum(abs(x)<=1e-6 for x in vals.values()))==(p['positive'],p['negative'],p['neutral']);pairs+=1;points+=len(vals)
 for cid in {cid for cid,a in idx}:
  assert idx[cid,'S_A_Native']['metrics']['m_tIoU']==idx[cid,'S_B_Native']['metrics']['m_tIoU']==idx[cid,'Frozen']['metrics']['m_tIoU']
  for a in ('A','B'):
   assert idx[cid,'T_S_'+a+'_Native_evidence']['metrics']['m_tIoU']==idx[cid,'Full_'+a+'_Adapted']['metrics']['m_tIoU']==idx[cid,'Temporal']['metrics']['m_tIoU']
   assert abs(idx[cid,'S_'+a+'_Native']['metrics']['m_sIoU_dense_GT']-idx[cid,'T_S_'+a+'_Native_evidence']['metrics']['m_sIoU_dense_GT'])<2e-12
 out=dict(status='pass',count=256,rows=len(idx),metric_values=sum(len(r['metrics'])for r in d['rows']),null_metric_values=nulls,summary_cells=summary,paired_cells=pairs,paired_source_values=points,all_signed_outcomes_and_unknowns_checked=True,Temporal_and_fixed_support_invariants=True,result_sha256=sha(FIXED/'POSTSEAL_RESULTS.json'),new_GT_reader_calls=0,new_model_calls=0,GT_read=False)
 put(FIXED/'ROOT_INDEPENDENT_SCALAR_READBACK003.json',out);print(json.dumps(out))
if __name__=='__main__':main()
