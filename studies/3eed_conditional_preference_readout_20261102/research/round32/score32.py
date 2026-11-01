"""Selected-family postseal task scoring. T128 never selects a method."""
import sys,numpy as np,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round32'),str(ROOT/'research/round27')]
from common32 import *
from predictT32 import selected_specs
from evaluate27 import ious

def main():
 verify(BASE/'selectedT128_attempt001');verify(BASE/'closureT128_attempt001');specs=selected_specs();records=[]
 for split in ['A208','DEV64','R64','S64','T128']:
  for r in rows(split):
   z=feature(r)
   if split=='T128':
    p=load_npz(BASE/'selectedT128_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));names=p['arm_names'].tolist();qs=p['queries'];assert np.array_equal(p['boxes'],z['boxes'][qs])
   else:
    q={'source':int(z['scores'].argmax())};native=load_npz(BASE/'scope400_endpoint_attempt001/predictions_PRIVATE'/(r['id']+'_noun_interaction.npz'));q['R30_ungated']=int(native['queries'][native['arm_names'].tolist().index('R30_ungated')])
    for name,spec in specs.items():
     folder='scope400_endpoint_attempt001' if spec['support']=='endpoint' else 'scope400_attempt001';p=load_npz(BASE/folder/'predictions_PRIVATE'/(r['id']+'_'+spec['key']+'.npz'));arm='k'+str(spec['k'])+'_'+spec['arm'];q[name]=int(p['queries'][p['arm_names'].tolist().index(arm)])
    if split=='A208':
     p=load_npz(ROOT/'crossplatform/round29/direct208_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));q['same_information_direct_G']=int(p['queries'][p['arm_names'].tolist().index('full_pair_group')])
    names=list(q);qs=np.array(list(q.values()))
   # Repeated arms may choose the same query; they are not independent evidence.
   uq,inv=np.unique(qs,return_inverse=True);v=ious(r,z['boxes'][uq])[inv];records.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'split':split,'q':dict(zip(names,map(int,qs))),'ious':dict(zip(names,map(float,v)))})
 dump(EXP/'ALL528_TASK_ROWS_PRIVATE.json',records);summary={};neg=[]
 for split in ['A208','DEV64','R64','S64','T128']:
  summary[split]={}
  for pl in ['drone','quad']:
   rr=[x for x in records if x['split']==split and x['platform']==pl];summary[split][pl]={}
   for arm in rr[0]['q']:
    vv=np.array([x['ious'][arm] for x in rr]);d=vv-np.array([x['ious']['source'] for x in rr]);summary[split][pl][arm]={'n':len(rr),'mIoU_percent':float(vv.mean()*100),'Acc25_percent':float((vv>.25).mean()*100),'Acc50_percent':float((vv>.5).mean()*100),'delta_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(x['q'][arm]!=x['q']['source'] for x in rr)}
   for x in rr:
    for arm in x['q']:
     for ref in ['source','R30_ungated','same_gate_QP_memory','DEVbest_global_amount']:
      if x['ious'][arm]<x['ious'][ref]-1e-10:neg.append({'alias':x['alias'],'split':split,'platform':pl,'arm':arm,'reference':ref,'delta_pp':100*(x['ious'][arm]-x['ious'][ref])})
 T=[x for x in records if x['split']=='T128'];clusters={p:[[x for x in T if x['platform']==p and x['scene_PRIVATE']==s] for s in sorted({x['scene_PRIVATE'] for x in T if x['platform']==p})] for p in ['drone','quad']};rng=np.random.default_rng(32);ci={}
 for ref in ['source','R30_ungated','same_gate_QP_memory','DEVbest_global_amount','global_norm_matched','R30_allA_zeroU_memory']:
  dist=[]
  for _ in range(5000):
   m=[]
   for pl,cc in clusters.items():
    rr=[x for j in rng.integers(0,len(cc),len(cc)) for x in cc[j]];m.append(np.mean([x['ious']['scoped_TTA']-x['ious'][ref] for x in rr]))
   dist.append(100*np.mean(m))
  active_clusters=sum(any(abs(x['ious']['scoped_TTA']-x['ious'][ref])>1e-12 for x in cc) for groups in clusters.values() for cc in groups)
  ci['scoped_TTA_vs_'+ref]={'active_effect_clusters':active_clusters,'interval_generalizability':'not_available_if_less_than_two_effect_clusters;Dronebetween-scenevariation_always_unestimated','conditional_resampling_quantiles_not_population_CI':np.quantile(dist,[.025,.975]).tolist(),'balanced_gain_pp':float(np.mean([np.mean([x['ious']['scoped_TTA']-x['ious'][ref] for x in T if x['platform']==p]) for p in ['drone','quad']])*100),'descriptive_scene_bootstrap95_pp':np.quantile(dist,[.025,.975]).tolist() if active_clusters>=2 else None,'clusters':{p:len(c) for p,c in clusters.items()},'Drone_only_one_fixed_cluster_cannot_estimate_between_Drone_scene_uncertainty':True}
 scenes=[]
 for pl,cc in clusters.items():
  for j,rr in enumerate(cc):scenes.append({'platform':pl,'scene_alias':pl+'_s'+str(j),'n':len(rr),'mIoU_percent':{a:float(np.mean([x['ious'][a] for x in rr])*100) for a in rr[0]['q']}})
 dump(EXP/'TASK_SUMMARY.json',summary);dump(EXP/'ALL_NEGATIVES.json',neg);dump(EXP/'T128_SCENE_COMPARISONS.json',ci);dump(EXP/'T128_SCENE_RESULTS.json',scenes);print('R32_T128_TASK')
 for pl in ['drone','quad']:
  print(pl,json.dumps(summary['T128'][pl]))
 print('CLUSTER',json.dumps(ci))
if __name__=='__main__':main()
