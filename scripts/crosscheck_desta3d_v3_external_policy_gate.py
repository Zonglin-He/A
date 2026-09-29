"""Independent NumPy aggregation, resampling, tails and boolean routing readback."""
import sys,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_external_policy_provider_v2 import D,read,write,sha

def main(stage):
 import numpy as np
 dest=D/('evaluation' if stage=='policy001' else 'wrong_evaluation');r=read(dest/'REPORT.json');assert sha(dest/'REPORT.json')==read(dest/'COMPLETE.json')['report_sha'];start=time.monotonic();errors=[];checks=0
 metrics=['tIoU','sIoU','vIoU'];arms=r['arms'];mat={}
 for a,x in arms.items():
  assert len(x['rows'])==len({str(z['source']) for z in x['rows']})==16
  mat[a]={k:np.array([z['metrics'][k] for z in sorted(x['rows'],key=lambda z:str(z['source']))],dtype=np.float64) for k in metrics}
  for k in metrics:
   errors.append(abs(float(np.mean(mat[a][k]))-x['summary']['parent_macro'][k]));checks+=1
 def verify(comp,a,b):
  nonlocal checks
  for k,c in comp.items():
   v=mat[a][k]-mat[b][k];rng=np.random.default_rng(c['bootstrap_seed']);ix=rng.integers(0,16,(c['bootstrap_replicates'],16));samples=np.sum(v[ix],axis=1)/16
   want={'mean_delta_pp':100*float(np.sum(v)/16),'worst_delta_pp':100*float(min(v)),'best_delta_pp':100*float(max(v)),'positive_parents':int(sum(v>1e-10)),'negative_parents':int(sum(v< -1e-10)),'zero_parents':int(sum(abs(v)<=1e-10)),'severe_loss_below_minus5pp':int(sum(v< -.05))}
   for key,val in want.items():errors.append(abs(val-c[key]));checks+=1
   errors.extend(abs(np.percentile(100*samples,[2.5,97.5])-c['bootstrap_ci95_pp']));checks+=2
 for a,c in r['comparisons'].items():
  verify(c,a,'B1')
  for k in metrics:
   b=mat['B1'][k];v=mat[a][k];diff=100*(v-b);t=r['query_tails'][a][k];ret=r['B1_good_retention'][a][k]
   assert t['query_harm_gt5pp']==int(sum(diff< -5)) and t['positive']==int(sum(diff>0)) and t['negative']==int(sum(diff<0)) and t['zero']==int(sum(diff==0))
   assert ret==dict(eligible=int(sum(b>.5)),retained=int(sum((b>.5)&(v>.5))));checks+=6
 for a,c in r['correct_minus_wrong'].items():verify(c,a,a+'_wrong')
 collapse=any(float(np.mean(mat['TS'][k]-mat['B1'][k]))*100<=-1 and sum(mat['TS'][k]-mat['B1'][k]< -1e-10)>=12 for k in ['tIoU','sIoU'])
 positive=[a for a,k in [('T','tIoU'),('S','sIoU')] if np.mean(mat[a][k]-mat['B1'][k])>0]
 preliminary=np.mean(mat['TS']['vIoU']-mat['B1']['vIoU'])>0 and bool(positive) and not collapse
 assert bool(preliminary)==r['preliminary_pass'] and bool(collapse)==r['TS_systematic_collapse'] and positive==r['positive_corresponding_branches']
 if 'TS_wrong' in mat:
  final=preliminary and np.mean(mat['TS']['vIoU']-mat['TS_wrong']['vIoU'])>0 and any(a+'_wrong' in mat and np.mean(mat[a][k]-mat[a+'_wrong'][k])>0 for a,k in [('T','tIoU'),('S','sIoU')] if a in positive)
  assert bool(final)==r['final_pass']
 else:assert r['final_pass'] is (None if preliminary else False)
 assert max(errors,default=0)<1e-10
 write(dest/'ROOT_SCORE_CROSSCHECK.json',dict(status='passed',checks=checks,maximum_absolute_error=max(errors,default=0),CPU_seconds=time.monotonic()-start,decision=r['decision'],report_sha=sha(dest/'REPORT.json')));print('CROSSCHECK',checks,max(errors,default=0),r['decision'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['policy001','wrong001']);main(p.parse_args().stage)
