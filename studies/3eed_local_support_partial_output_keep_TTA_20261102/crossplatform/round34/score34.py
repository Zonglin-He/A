"""Post-seal CPU task scoring; this file is never imported by inference code."""
import sys,argparse,numpy as np
from common34 import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['diagnostic','U128'],required=True);a=p.parse_args();check_plan();pred=output(a.split);verify(pred);closure=BASE/('closure_'+a.split+'_attempt001');verify(closure);c=read(closure/'CLOSURE_AUDIT.json');assert c['state']=='PASSED' and c['observations']==(10 if a.split=='diagnostic' else 128) and c['full_model_forwards']==(20 if a.split=='diagnostic' else 256);audit=read(EXP/('CPU_AUDIT_'+a.split+'.json'));assert audit['state']=='PASSED' and audit['prediction_seal_sha256']==sha(pred/'SEALED.json')
 # Task annotation access begins only here, after prediction + closure + arithmetic seals.
 sys.path.insert(0,str(ROOT/'research/round27'));from evaluate27 import ious
 records=[]
 for r in rows(a.split):
  q=load_npz(pred/'predictions_PRIVATE'/(r['id']+'.npz'));z=feature(r);assert q['arm_names'].tolist()==ARMS;assert np.array_equal(q['boxes'],z['boxes'][q['queries']]);uq,inv=np.unique(q['queries'],return_inverse=True);v=ious(r,z['boxes'][uq])[inv];records.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'split':r.get('split',a.split),'queries':dict(zip(ARMS,map(int,q['queries']))),'IoU':dict(zip(ARMS,map(float,v)))})
 summary={};neg=[];scenes=[];comparisons=[]
 for split in sorted({r['split'] for r in records}):
  rs=[r for r in records if r['split']==split];summary[split]={};pairs=[('h_scoped_R30',r) for r in ['source','R30_ungated','noun_scoped_R30','h_samegate_QP','h_global_mean','h_global_norm']]
  for pl in ['drone','quad']:
   rr=[r for r in rs if r['platform']==pl];summary[split][pl]={}
   for arm in ARMS:
    v=np.array([r['IoU'][arm] for r in rr]);d=v-np.array([r['IoU']['source'] for r in rr]);summary[split][pl][arm]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'delta_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in rr),'scene_clusters':len({r['scene_PRIVATE'] for r in rr})}
   for r in rr:
    for arm in ARMS:
     for ref in ['source','R30_ungated','h_samegate_QP','noun_samegate_QP']:
      d=r['IoU'][arm]-r['IoU'][ref]
      if d< -1e-10:neg.append({'alias':r['alias'],'platform':pl,'split':split,'arm':arm,'reference':ref,'delta_pp':100*d})
   for j,s in enumerate(sorted({r['scene_PRIVATE'] for r in rr})):
    ss=[r for r in rr if r['scene_PRIVATE']==s];scenes.append({'split':split,'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{arm:float(np.mean([r['IoU'][arm] for r in ss])*100) for arm in ARMS}})
  clusters={pl:[[r for r in rs if r['platform']==pl and r['scene_PRIVATE']==s] for s in sorted({r['scene_PRIVATE'] for r in rs if r['platform']==pl})] for pl in ['drone','quad']};rng=np.random.default_rng(34)
  for arm,ref in pairs:
   by={pl:float(np.mean([r['IoU'][arm]-r['IoU'][ref] for r in rs if r['platform']==pl])*100) for pl in ['drone','quad']};effectclusters=sum(any(abs(r['IoU'][arm]-r['IoU'][ref])>1e-12 for r in c) for cc in clusters.values() for c in cc);dist=[]
   for _ in range(2000):
    d=[]
    for pl,cc in clusters.items():
     sample=[r for j in rng.integers(0,len(cc),len(cc)) for r in cc[j]];d.append(np.mean([r['IoU'][arm]-r['IoU'][ref] for r in sample]))
    dist.append(100*np.mean(d))
   adequate=effectclusters>=2 and all(len(cc)>=2 for cc in clusters.values());comparisons.append({'split':split,'arm':arm,'reference':ref,'platform_gain_pp':by,'balanced_gain_pp':float(np.mean(list(by.values()))),'clusters':{pl:len(cc) for pl,cc in clusters.items()},'active_effect_clusters':effectclusters,'descriptive_scene_bootstrap95_pp':np.quantile(dist,[.025,.975]).tolist() if adequate else None,'conditional_resampling_quantiles_pp':np.quantile(dist,[.025,.975]).tolist(),'between_scene_generalization_not_estimated_if_a_platform_has_one_scene':True,'diagnostic_splits_exposed_not_independent_confirmation':split!='U128'})
  summary[split]['balanced_mIoU_percent']={arm:float(np.mean([summary[split][pl][arm]['mIoU_percent'] for pl in ['drone','quad']])) for arm in ARMS}
 prefix=a.split;dump(EXP/(prefix+'_ALL_ROWS_PRIVATE.json'),records);dump(EXP/(prefix+'_SUMMARY.json'),summary);dump(EXP/(prefix+'_ALL_NEGATIVES.json'),neg);dump(EXP/(prefix+'_SCENE_ROWS.json'),scenes);dump(EXP/(prefix+'_PAIRED_COMPARISONS.json'),comparisons);dump(EXP/(prefix+'_SCORING_RECEIPT.json'),{'state':'COMPLETED','prediction_seal_sha256':sha(pred/'SEALED.json'),'closure_seal_sha256':sha(closure/'SEALED.json'),'n':len(records),'no_selection_or_feedback_filtering':True,'primary_fixed_h_key_no_k_sweep':True});print('SCORE34',a.split,summary)
if __name__=='__main__':main()
