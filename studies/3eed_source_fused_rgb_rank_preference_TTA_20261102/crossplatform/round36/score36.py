"""Sealed fixed 2x2 scores; native per-model baselines and all negative controls."""
import sys,numpy as np
from common36 import *
def main():
 check_plan();verify(BASE/'fit272_attempt001');verify(BASE/'closure4_attempt001');assert read(EXP/'CPU_AUDIT.json')['state']=='PASSED';sys.path.insert(0,str(ROOT/'research/round27'));from evaluate27 import ious
 from endpoint_scope32 import parent_scores
 out=BASE/'fit272_attempt001';records=[];negative=[];summary={}
 for split in ['A208','DEV64']:
  summary[split]={}
  for mode in MODES:
   summary[split][mode]={}
   for pl in ['drone','quad']:
    rr=[r for r in rows(split) if r['platform']==pl];local=[]
    for r in rr:
     p=load_npz(out/'predictions_PRIVATE'/(r['id']+'_'+mode+'.npz'));z=feature(r);q={'fused_frozen':int(p['queries'][0]),'rank_TTA':int(p['queries'][1]),'sameinfo_memory':int(p['queries'][2]),'XYZ_source':int(z['scores'].argmax()),'XYZ_R30':grouped_readout(r,parent_scores(r),z['scores'])};uq,inv=np.unique(list(q.values()),return_inverse=True);v=ious(r,z['boxes'][uq])[inv];rec={'alias':r['alias'],'id_PRIVATE':r['id'],'mode':mode,'platform':pl,'split':split,'scene_PRIVATE':r['scene'],'queries':q,'IoU':dict(zip(q,map(float,v)))};records.append(rec);local.append(rec)
     for arm in q:
      for ref in ['fused_frozen','sameinfo_memory','XYZ_source','XYZ_R30']:
       if arm!=ref and rec['IoU'][arm]<rec['IoU'][ref]-1e-10:negative.append({'alias':r['alias'],'mode':mode,'platform':pl,'split':split,'arm':arm,'reference':ref,'delta_pp':100*(rec['IoU'][arm]-rec['IoU'][ref])})
    for arm in local[0]['queries']:
     vv=np.array([r['IoU'][arm] for r in local]);d=vv-np.array([r['IoU']['fused_frozen'] for r in local]);summary[split][mode].setdefault(pl,{})[arm]={'n':len(local),'mIoU_percent':float(vv.mean()*100),'Acc25_percent':float((vv>.25).mean()*100),'Acc50_percent':float((vv>.5).mean()*100),'delta_vs_own_fused_frozen_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['fused_frozen'] for r in local)}
   summary[split][mode]['balanced_mIoU_percent']={arm:float(np.mean([summary[split][mode][pl][arm]['mIoU_percent'] for pl in ['drone','quad']])) for arm in ['fused_frozen','rank_TTA','sameinfo_memory','XYZ_source','XYZ_R30']}
 mechanism={};scenes=[]
 for split in ['A208','DEV64']:
  mechanism[split]={}
  for pl in ['drone','quad']:
   a={r['id_PRIVATE']:r for r in records if r['split']==split and r['mode']=='rgb' and r['platform']==pl};b={r['id_PRIVATE']:r for r in records if r['split']==split and r['mode']=='no_rgb' and r['platform']==pl};assert a.keys()==b.keys();diff=[(a[i]['IoU']['rank_TTA']-a[i]['IoU']['fused_frozen'])-(b[i]['IoU']['rank_TTA']-b[i]['IoU']['fused_frozen']) for i in a];mechanism[split][pl]={'paired_delta_TTA_RGB_minus_delta_TTA_noRGB_pp':float(np.mean(diff)*100),'n':len(diff),'model_source_weights_differ_input_only_causality_not_claimed':True}
  mechanism[split]['balanced_paired_interaction_pp']=float(np.mean([mechanism[split][pl]['paired_delta_TTA_RGB_minus_delta_TTA_noRGB_pp'] for pl in ['drone','quad']]))
 for (split,mode,pl,scene) in sorted({(r['split'],r['mode'],r['platform'],r['scene_PRIVATE']) for r in records}):
  rr=[r for r in records if (r['split'],r['mode'],r['platform'],r['scene_PRIVATE'])==(split,mode,pl,scene)];scenes.append({'split':split,'mode':mode,'platform':pl,'scene_PRIVATE':scene,'n':len(rr),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in rr])*100) for a in rr[0]['queries']}})
 dev=summary['DEV64']['rgb']['balanced_mIoU_percent'];passes=dev['rank_TTA']>dev['fused_frozen'] and dev['rank_TTA']>dev['sameinfo_memory'];decision={'state':'COMPLETED','fixed_DEV_gate_pass':passes,'decision':'report root before further registration' if passes else 'stop this fixed rank configuration; no FINAL or interface expansion','RGB_TTA_vs_own_frozen_pp':dev['rank_TTA']-dev['fused_frozen'],'RGB_TTA_vs_sameinfo_memory_pp':dev['rank_TTA']-dev['sameinfo_memory'],'primary_paired_interaction_pp':mechanism['DEV64']['balanced_paired_interaction_pp'],'no_U_FINAL_access':True,'all_negative_cases_retained':True}
 dump(EXP/'SUMMARY.json',summary);dump(EXP/'ALL_ROWS_PRIVATE.json',records);dump(EXP/'ALL_NEGATIVES.json',negative);dump(EXP/'PAIRED_MECHANISM.json',mechanism);dump(EXP/'SCENE_ROWS_PRIVATE.json',scenes);dump(EXP/'DECISION.json',decision);print('R36_DECISION',decision);print('DEV36',summary['DEV64'])
if __name__=='__main__':main()
