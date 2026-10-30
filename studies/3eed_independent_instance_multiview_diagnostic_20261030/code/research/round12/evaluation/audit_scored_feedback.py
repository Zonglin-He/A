"""Independent R12 scalar reduction of alreadysealed scores; no new IoU/GT IO."""
from pathlib import Path
import argparse,json,hashlib,collections,math
import numpy as np
ARMS=('source','grabcut_direct','single_instance_direct','multiview_instance_direct','wrongview_instance_direct','single_instance_half','multiview_instance_half','multiview_maskmatch_only_direct')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
class Compare:
 def __init__(self):self.n=0;self.max=0.;self.fail=[]
 def check(self,a,b,key):
  self.n+=1
  if a is None or b is None:
   if a is not b:self.fail.append({'field':key,'expected':a,'actual':b})
  elif isinstance(a,(int,float,np.number)) and isinstance(b,(int,float,np.number)):
   e=abs(float(a)-float(b));self.max=max(self.max,e)
   if not math.isfinite(e) or e>1e-11:self.fail.append({'field':key,'maxdiff':e})
  elif a!=b:self.fail.append({'field':key,'expected':a,'actual':b})
 def tree(self,a,b,key):
  if isinstance(a,dict):
   for k,v in a.items():
    if k not in b:self.fail.append({'field':key+'.'+k,'missing':True})
    else:self.tree(v,b[k],key+'.'+k)
  elif isinstance(a,list):
   self.check(len(a),len(b),key+'.len')
   for j,(x,y) in enumerate(zip(a,b)):self.tree(x,y,key+'.'+str(j))
  else:self.check(a,b,key)
def avg(x):return float(np.mean(x)) if len(x) else None

def scene_stats(rr,x):
 grouped=collections.defaultdict(list)
 for r,v in zip(rr,x):grouped[r['scene']].append(float(v))
 g=np.array([avg(grouped[k]) for k in sorted(grouped)])
 ci=np.quantile(np.random.default_rng(1201).choice(g,(2000,len(g))).mean(1),[.025,.975]).tolist() if len(g)>1 else None
 return {'n':len(x),'scene_clusters':len(g),'micro_mean':avg(x),'equal_scene_mean':avg(g),'equal_scene_descriptive_CI95':ci,'bootstrap_seed':1201,'bootstrap_draws':2000}
def signed(x,tol):
 x=np.asarray(x,float);p=x[x>tol];n=x[x< -tol]
 return {'n':len(x),'mean':avg(x),'sum':math.fsum(map(float,x)),'positive_count':len(p),'negative_count':len(n),'ties_within_tol':int((abs(x)<=tol).sum()),'positive_sum':math.fsum(map(float,p)),'negative_sum':math.fsum(map(float,n)),'gain_sum_exceeds_abs_harm_sum':math.fsum(map(float,p))>abs(math.fsum(map(float,n))),'min':float(x.min()) if len(x) else None,'max':float(x.max()) if len(x) else None,'tolerance':tol}
def population(row,name,arm):
 if name=='all':return True
 if name=='single_MV_union_changed':return row['arms']['single_instance_direct']['actual_changed'] or row['arms']['multiview_instance_direct']['actual_changed']
 if name=='single_MV_common_eligible':return row['arms']['single_instance_direct']['accepted_any'] and row['arms']['multiview_instance_direct']['accepted_any']
 if name=='own_changed_descriptive':return row['arms'][arm]['actual_changed']
 raise ValueError('Unknown locked population '+name)
def main(a):
 seal=read(a.scored/'SEALED.json')
 for rel,h in seal['files'].items():
  p=(a.scored/rel).resolve();assert p.is_relative_to(a.scored.resolve()) and sha(p)==h
 out=read(a.scored/'RESULTS_PRIVATE.json');assert out['state']=='R12_146_independent_instance_multiview_scored_after_complete_seal' and out['n']==146 and seal['n_samples']==146
 assert read(a.scored/'SCALAR_SUMMARY.json')=={k:v for k,v in out.items() if k!='rows_PRIVATE'}
 rows=out['rows_PRIVATE'];assert len(rows)==len({r['id'] for r in rows})==146 and collections.Counter(r['platform'] for r in rows)=={'drone':50,'quad':96}
 assert all(tuple(r['arms'])==ARMS for r in rows);C=Compare()
 for t in out['summary']:
  rr=[r for r in rows if r['platform']==t['platform'] and population(r,t['population'],t['arm'])];v=[r['arms'][t['arm']] for r in rr];n=len(v);iou=[x['iou'] for x in v];ce=[x['coordinate_errors']['center_L2_m'] for x in v];se=[x['coordinate_errors']['size_L2_m'] for x in v]
  actual={'n':n,'scene_clusters':len({r['scene'] for r in rr}),'mean_iou':avg(iou),'equal_scene_mean_iou':scene_stats(rr,iou)['equal_scene_mean'],'Acc25':sum(x>.25 for x in iou)/n if n else None,'Acc50':sum(x>.5 for x in iou)/n if n else None,'equal_scene_Acc25':scene_stats(rr,[float(x>.25) for x in iou])['equal_scene_mean'],'equal_scene_Acc50':scene_stats(rr,[float(x>.5) for x in iou])['equal_scene_mean'],'center_L2_m_mean':avg(ce),'equal_scene_center_L2_m_mean':scene_stats(rr,ce)['equal_scene_mean'],'size_L2_m_mean':avg(se),'equal_scene_size_L2_m_mean':scene_stats(rr,se)['equal_scene_mean'],'accepted_any_n':sum(x['accepted_any'] for x in v),'actual_changed_n':sum(x['actual_changed'] for x in v),'actual_changed_axis_counts':[sum(x['actual_changed_axes'][j] for x in v) for j in range(6)],'status_histogram':dict(collections.Counter(x['status'] for x in v))};C.tree(actual,t,'summary.'+t['platform']+'.'+t['arm']+'.'+t['population'])
 for p in out['paired_contrasts']:
  rr=[r for r in rows if r['platform']==p['platform'] and population(r,p['population'],p['arm'])];arm,ref=p['arm'],p['reference'];d=[r['arms'][arm]['iou']-r['arms'][ref]['iou'] for r in rr];ce=[r['arms'][ref]['coordinate_errors']['center_L2_m']-r['arms'][arm]['coordinate_errors']['center_L2_m'] for r in rr];se=[r['arms'][ref]['coordinate_errors']['size_L2_m']-r['arms'][arm]['coordinate_errors']['size_L2_m'] for r in rr]
  C.tree({'n':len(rr),'IoU_gain':signed(d,1e-8),'center_error_reduction_m':signed(ce,1e-6),'size_error_reduction_m':signed(se,1e-6),'IoU_scene':scene_stats(rr,d),'center_scene':scene_stats(rr,ce),'size_scene':scene_stats(rr,se)},p,'paired.'+p['platform']+'.'+arm+'.'+ref+'.'+p['population'])
 result={'state':'PASS_R12_INDEPENDENT_SCALAR_REDUCTION_AUDIT' if not C.fail else 'FAIL_R12_INDEPENDENT_SCALAR_REDUCTION_AUDIT','n':146,'platform_n':{'drone':50,'quad':96},'results_sha256':sha(a.scored/'RESULTS_PRIVATE.json'),'scored_seal_sha256':sha(a.scored/'SEALED.json'),'prediction_seal_sha256':seal['prediction_seal_sha256'],'release_sha256':seal['release_sha256'],'auditor_sha256':sha(Path(__file__)),'numeric_and_exact_checks':C.n,'max_numeric_difference':C.max,'failures':C.fail,'all_sealed_scored_files_rehashed_and_scalar_projection_exact':True,'coverage':'All8arms/allpopulations/strictAcc25&50/micro+equal-scene mIoU,Acc,center,size/signs/sceneCI/negative,zero,NAstatuscounts. Legacy8th isunfilteredsamegate aggregation, notanchor-only.','no_new_GT_metadata_IoU_model_GPU_or_methodchanges':True};assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'state':result['state'],'checks':C.n,'maxdiff':C.max,'sha256':sha(a.output)}));assert not C.fail
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scored',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
