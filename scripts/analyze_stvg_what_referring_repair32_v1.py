"""Post-seal reference-quality analysis and independent actual-frame IoU checks."""
import os,sys,collections,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_what_referring_repair32_v1 import *

def main():
 import torch,numpy as np
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32;assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube
 gt,spans,_=truths('vidstg',dict(parents=[r['query_ordinal']for r in p['records']if r['alias']in FOCUS]));plans=read(ROOT/'artifacts/decota_paper_experiments_v1/vidstg/PLAN.json')['rows'];old={r['alias']:r for r in read(P/'RESULTS_PUBLIC.json')['rows']};frames=read(P/'FRAME_DIAGNOSTICS_PUBLIC.json')['rows'];details=[];checks=collections.Counter()
 for a in FOCUS:
  z=load(D/'predictions'/(a+'.pt'));q=z['query_ordinal'];row=plans[q];W,H=row['input']['width'],row['input']['height'];sel=z['estimate']['selected'];ac=0
  def iou(b,f):
   b=np.asarray(b);x=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]*[W,H,W,H];y=np.asarray(gt[q][f]);area=lambda z:float(np.prod(np.maximum(0,z[2:]-z[:2])));inter=float(np.prod(np.maximum(0,np.minimum(x[2:],y[2:])-np.maximum(x[:2],y[:2]))));return inter/max(area(x)+area(y)-inter,1e-12)
  for (_,pos),v in sorted(z['expert']['observations'].items()):
   pr=v['probe'];known=z['frame_ids'][pos]in gt[q];bs=np.asarray(pr['boxes']);ious=[iou(b,z['frame_ids'][pos])for b in bs]if known else None;prev=next(r for r in frames if r['alias']==a and r['position']==pos)
   if known:assert abs(max(ious)-prev['retained_pool_best_IoU'])<1e-12;checks['independent_reference_IoU']+=1
   k=sel[ac]if pr['accepted']else None;ac+=int(pr['accepted']);details.append(dict(alias=a,position=pos,GT_known=known,admitted=pr['accepted'],E_selected_candidate=k,E_used=pr['accepted']and k>=0,E_reference_IoU=ious[k]if known and k is not None and k>=0 else None,baseline_IoU=prev['E_IoU'],top_score_IoU=prev['top_score_IoU'],pool_best_IoU=prev['retained_pool_best_IoU'],rejection_reason=pr['reason']))
  for arm,b in z['outputs'].items():
   tube=DenseTube(b,row,gt[q],spans[q],clip=False);s=tube.score(z['interval']);
   for k in ['v','t','s']:assert abs(s[k]-old[a]['values'][arm][k])<2e-10;checks['independent_full_tube_scores']+=1
   vals=[iou(b[j],f)for j,f in enumerate(z['frame_ids'])if f in gt[q]and j not in z['planned_positions']];assert abs(np.mean(vals)-old[a]['values'][arm]['unobserved_actual_s'])<1e-12;checks['independent_unobserved_actual_s']+=1
 known_ad=[r for r in details if r['admitted']and r['GT_known']];used_known=[r for r in details if r['E_used']and r['GT_known']]
 result=dict(count=20,rows=details,counts=dict(admitted=sum(r['admitted']for r in details),E_used=sum(r['E_used']for r in details),known_admitted=len(known_ad),known_admitted_top_better_than_E=sum(r['top_score_IoU']>r['baseline_IoU']for r in known_ad),unknown_admitted=sum(r['admitted']and not r['GT_known']for r in details),known_E_used=len(used_known),known_E_used_better_than_E=sum(r['E_reference_IoU']>r['baseline_IoU']for r in used_known)),checks=dict(checks),scope='offline diagnostics after seal, no feedback or checkpoint selection',GT_unknown_not_negative=True)
 put(P/'POST_SEAL_REFERENCE_QUALITY.json',result)
 put(P/'SCIENTIFIC_READBACK_AUDIT.json',dict(status='pass',checks=dict(checks),all_sealed_predictions_unchanged=True,no_models_GPU_or_selection=True));print(result['counts'],dict(checks),flush=True)
if __name__=='__main__':main()
