"""Current labels after immutable prediction seals; all methods and negative tails."""
import sys,numpy as np,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'research/round27')]
from common30 import *
from evaluate27 import ious

def main():
 verify(BASE/'screen_attempt001');verify(BASE/'selectedS64_attempt001');verify(BASE/'memoryAllS64_attempt001');verify(BASE/'closureS64_attempt001');verify(BASE/'evidence_review22_attempt001');sel=read(EXP/'SELECTED_BETA_LOCK.json');fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');fids={f['id_PRIVATE'] for f in fb};records=[]
 for split in ['A208','DEV64','S64']:
  for r in rows(split):
   z=feature(r);q={'source':int(z['scores'].argmax())}
   if split in ['A208','DEV64']:
    for arm,name in [('preference_only','b0'),('keep_selected',sel['selected']['name'])]:
     p=load_npz(BASE/'screen_attempt001'/(r['platform']+'_'+name+('_A' if split=='A208' else '_DEV'))/'predictions_PRIVATE'/(r['id']+'.npz'));ix=int(p['query']);assert np.array_equal(p['box'],z['boxes'][ix]);q[arm]=ix
    if split=='A208':
     p=load_npz(ROOT/'crossplatform/round29/direct208_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));q['same_feedback_direct']=int(p['queries'][p['arm_names'].tolist().index('full_pair_group')])
   else:
    p=load_npz(BASE/'selectedS64_attempt001/predictions_PRIVATE'/(r['id']+'.npz'))
    for name,ix,b in zip(p['arm_names'],p['queries'],p['boxes']):assert np.array_equal(b,z['boxes'][ix]);q[str(name)]=int(ix)
    p=load_npz(BASE/'memoryAllS64_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));ix=p['arm_names'].tolist().index('all_A_zeroU_memory');q['all_A_zeroU_memory']=int(p['queries'][ix])
   names=list(q);vv=ious(r,z['boxes'][np.array(list(q.values()))]);records.append({'alias':r['alias'],'id_PRIVATE':r['id'],'split':split,'platform':r['platform'],'scene_PRIVATE':r['scene'],'has_accepted_feedback':r['id'] in fids,'q':q,'ious':dict(zip(names,map(float,vv)))})
 dump(EXP/'ALL336_TASK_ROWS_PRIVATE.json',records);summary={};neg=[]
 for split in ['A208','DEV64','S64']:
  summary[split]={}
  for pl in ['drone','quad']:
   rr=[r for r in records if r['split']==split and r['platform']==pl];summary[split][pl]={}
   for arm in rr[0]['q']:
    v=np.array([r['ious'][arm] for r in rr]);d=v-np.array([r['ious']['source'] for r in rr]);summary[split][pl][arm]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['q'][arm]!=r['q']['source'] for r in rr)}
   for r in rr:
    for arm in r['q']:
     for ref in ['source','preference_only']+(['same_feedback_direct'] if split=='A208' else ['all_A_zeroU_memory'] if split=='S64' else []):
      if r['ious'][arm]<r['ious'][ref]-1e-10:neg.append({'alias':r['alias'],'split':split,'platform':pl,'arm':arm,'reference':ref,'delta_pp':100*(r['ious'][arm]-r['ious'][ref])})
 strata={}
 for pl in ['drone','quad']:
  strata[pl]={}
  for flag in [True,False]:
   rr=[r for r in records if r['split']=='A208' and r['platform']==pl and r['has_accepted_feedback']==flag];strata[pl]['feedback' if flag else 'no_feedback']={}
   for arm in rr[0]['q']:
    d=np.array([r['ious'][arm]-r['ious']['source'] for r in rr]);strata[pl]['feedback' if flag else 'no_feedback'][arm]={'n':len(rr),'mIoU_percent':float(np.mean([r['ious'][arm] for r in rr])*100),'gain_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['q'][arm]!=r['q']['source'] for r in rr)}
 # Platform-stratified scene bootstrap. Do not treat frames as independent scenes.
 rng=np.random.default_rng(30);S=[r for r in records if r['split']=='S64'];clusters={p:[[r for r in S if r['platform']==p and r['scene_PRIVATE']==s] for s in sorted({r['scene_PRIVATE'] for r in S if r['platform']==p})] for p in ['drone','quad']};ci={}
 for arm,ref in [('keep_selected','source'),('keep_selected','preference_only'),('keep_selected','same_A_memory'),('keep_selected','all_A_zeroU_memory')]:
  v=[]
  for _ in range(5000):
   means=[]
   for p,cc in clusters.items():
    rs=[r for j in rng.integers(0,len(cc),len(cc)) for r in cc[j]];means.append(np.mean([r['ious'][arm]-r['ious'][ref] for r in rs]))
   v.append(100*np.mean(means))
  ci[arm+'_vs_'+ref]={'balanced_gain_pp':float(np.mean([np.mean([r['ious'][arm]-r['ious'][ref] for r in S if r['platform']==p]) for p in ['drone','quad']])*100),'scene_bootstrap95_pp':np.quantile(v,[.025,.975]).tolist(),'clusters':{p:len(c) for p,c in clusters.items()},'descriptive_small_reused_scenes':True}
 scenes=[]
 for p,cc in clusters.items():
  for j,rs in enumerate(cc):scenes.append({'platform':p,'scene_alias':p+'_s'+str(j),'n':len(rs),'mIoU_percent':{a:float(np.mean([r['ious'][a] for r in rs])*100) for a in rs[0]['q']}})
 dump(EXP/'TASK_SUMMARY.json',summary);dump(EXP/'A_FEEDBACK_STRATA.json',strata);dump(EXP/'ALL_NEGATIVES.json',neg);dump(EXP/'SCENE_CLUSTER_COMPARISONS.json',ci);dump(EXP/'S64_SCENE_RESULTS.json',scenes);print(json.dumps({'task':summary,'strata':strata,'cluster':ci},indent=2))
if __name__=='__main__':main()
