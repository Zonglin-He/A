"""Task labels enter here only after both A/B prediction seals and closure."""
import sys,numpy as np,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round29'),str(ROOT/'research/round27')]
from common29 import *
from evaluate27 import ious

def main():
 verify(BASE/'closureR64_attempt001');records=[]
 for split,rr,sub in [('A208',rows()[:208],'A208_predictions'),('R64',read(EXP.parent/'input/R64_PRIVATE.json'),'R64_predictions')]:
  for folder in ['ranknet_attempt001','ranknet_logodds_attempt001']:verify(BASE/folder/sub)
  for r in rr:
   sel={};boxes={}
   for folder,label in [('ranknet_attempt001','probability'),('ranknet_logodds_attempt001','logodds')]:
    z=load_npz(BASE/folder/sub/'predictions_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(z['boxes'],feature(r)['boxes'][z['queries']])
    for i,n in enumerate(z['arm_names']):
     name='TTA_'+label if n=='ranknet_group' else n
     if name in sel:assert sel[name]==int(z['queries'][i])
     sel[name]=int(z['queries'][i]);boxes[name]=z['boxes'][i]
   if split=='A208':
    z=load_npz(BASE/'direct208_attempt001/predictions_PRIVATE'/(r['id']+'.npz'))
    for n in ['legacy_G','source_anchor','old_pair_group']:
     i=z['arm_names'].tolist().index(n);sel[n]=int(z['queries'][i]);boxes[n]=z['boxes'][i]
   names=list(sel);v=ious(r,np.stack([boxes[n] for n in names]));records.append({'split':split,'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'q':sel,'ious':dict(zip(names,map(float,v)))})
 dump(EXP/'ALL272_TASK_ROWS_PRIVATE.json',records);summary={};neg=[];comparisons={}
 for split in ['A208','R64']:
  summary[split]={}
  for p in ['drone','quad']:
   rs=[r for r in records if r['split']==split and r['platform']==p];summary[split][p]={}
   for arm in rs[0]['q']:
    v=np.array([r['ious'][arm] for r in rs]);d=v-np.array([r['ious']['source'] for r in rs]);summary[split][p][arm]={'n':len(rs),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'changed':sum(r['q'][arm]!=r['q']['source'] for r in rs)}
    for r in rs:
     if r['ious'][arm]<r['ious']['source']-1e-10:neg.append({'split':split,'alias':r['alias'],'platform':p,'arm':arm,'delta_pp':100*(r['ious'][arm]-r['ious']['source'])})
  for arm in summary[split]['drone']:
   comparisons[split+'_'+arm]={'balanced_mIoU_percent':float(np.mean([summary[split][p][arm]['mIoU_percent'] for p in ['drone','quad']]))}
 # Scene clustered paired descriptive uncertainty, not iid frame CI.
 rng=np.random.default_rng(29);R=[r for r in records if r['split']=='R64'];clusters={p:[ [r for r in R if r['platform']==p and r['scene_PRIVATE']==s] for s in sorted({r['scene_PRIVATE'] for r in R if r['platform']==p})] for p in ['drone','quad']};ci={}
 for arm,ref in [('TTA_probability','source'),('TTA_logodds','source'),('same_A_memory','source'),('TTA_probability','same_A_memory'),('TTA_logodds','same_A_memory')]:
  ds=[]
  for _ in range(5000):
   means=[]
   for p,cc in clusters.items():
    rs=[r for j in rng.integers(0,len(cc),len(cc)) for r in cc[j]];means.append(np.mean([r['ious'][arm]-r['ious'][ref] for r in rs]))
   ds.append(100*np.mean(means))
  delta=float(np.mean([np.mean([r['ious'][arm]-r['ious'][ref] for r in R if r['platform']==p]) for p in ['drone','quad']])*100);ci[arm+'_vs_'+ref]={'balanced_gain_pp':delta,'scene_bootstrap95_pp':np.quantile(ds,[.025,.975]).tolist(),'scene_clusters':{'drone':len(clusters['drone']),'quad':len(clusters['quad'])},'descriptive_small_reused_scenes':True}
 dump(EXP/'TASK_SUMMARY.json',summary);dump(EXP/'ALL_NEGATIVES.json',neg);dump(EXP/'SCENE_CLUSTER_COMPARISONS.json',ci);dump(EXP/'BALANCED_SUMMARY.json',comparisons);print(json.dumps({'task':summary,'cluster':ci},indent=2))
if __name__=='__main__':main()
