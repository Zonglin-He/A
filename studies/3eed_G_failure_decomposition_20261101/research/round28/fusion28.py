"""Offline sealed-score readout counterfactuals. No model, optimizer or new G."""
import sys,numpy as np,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round28'),str(ROOT/'research/round5/audit')]
from common28 import *
from independent_core_v2 import candidate_ious,transform_gt
C_VALUES=[-2,-1,-.5,-.25,0,.25,.5,1,2]
def main():
 out=BASE/'fusion528_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();stats=collections.Counter();rows_private=[]
 priorrows={x['alias']:x for x in read(ROOT/'research/round27/experiment/ALL528_ROWS_PRIVATE.json')}
 for r in rows():
  z=load_npz(ROOT/'crossplatform/round27/scores528_attempt001/samples_PRIVATE'/(r['id']+'.npz'));s=z['source'].astype(float);v=z['cues'][z['cue_names'].tolist().index('crop_scene_joint')];mapped=z['mapped'];q0=int(s.argmax());f=s+2*v;q=int(f.argmax());assert q==priorrows[r['alias']]['q']['crop_scene_joint'];M=np.flatnonzero(mapped);U=np.flatnonzero(~mapped);qm=int(M[f[M].argmax()]) if len(M) else q0;qu=int(U[s[U].argmax()]) if len(U) else None
  pred={'source':q0,'joint_original':q,'legacy_G':priorrows[r['alias']]['q']['legacy_complex'],'mapped_competition_only':qm,'source_anchor_zero':qm}
  pp=paths(r);prep=read(pp['prep']/'cases_PRIVATE'/(r['id']+'.json'));a=load_npz(pp['prep']/'arrays_PRIVATE'/(r['id']+'.npz'));e=read(ROOT/'crossplatform/round27/joint528_attempt001/cases_PRIVATE'/(r['id']+'.json'));pool=prep['mask_pool_PRIVATE'];ev=np.array([e['targets_PRIVATE'][str(m)]['crop_scene_joint']['log_odds_yes_no'] for m in pool])
  def shifted(c):
   if not len(M):return q0
   if qu is None:return qm
   ms=f[qm]+2*c;us=s[qu]
   return qm if ms>us or ms==us and qm<qu else qu
  for c in C_VALUES:pred['offset_'+str(c)]=shifted(c)
  for name,c in [('median_zero',-float(np.median(ev)) if len(ev) else 0.),('mean_zero',-float(ev.mean()) if len(ev) else 0.)]:pred[name]=shifted(c)
  assert pred['offset_0']==q
  if len(M):
   assert mapped[q0],'record sourcewinner-unmapped rather than use anchor equivalence'
   anchored=f.copy();anchored[M]-=2*v[q0];qq=int(anchored.argmax())
   # Floating tie can differ within a theoretically equal group; recorded separately.
   if qq!=qm:assert abs(anchored[qq]-anchored[qm])<=1e-12;stats['anchor_numeric_ties']+=1
  qq=np.array(list(pred.values()));np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),names=np.array(list(pred)),queries=qq,boxes=z['boxes'][qq])
  rec={'alias':r['alias'],'platform':r['platform'],'stage':r['stage'],'scene_PRIVATE':r['scene'],'id_PRIVATE':r['id'],'has_pool':bool(len(M)),'pool_instances':len(pool),'mapped_q_n':len(M),'q0_mapped':bool(mapped[q0]),'original_mapped':bool(mapped[q]),'mapped_best_q_PRIVATE':qm,'unmapped_best_q_PRIVATE':qu,'positive_offset_switch_threshold':float((s[qu]-f[qm])/2) if len(M) and qu is not None else None,'mask_e_min':float(ev.min()) if len(ev) else None,'mask_e_max':float(ev.max()) if len(ev) else None,'mask_e_mean':float(ev.mean()) if len(ev) else None,'mask_e_std':float(ev.std()) if len(ev) else None,'all_mask_e_positive':bool((ev>0).all()) if len(ev) else None,'q':pred};rows_private.append(rec);stats['n']+=1;stats['with_pool']+=len(M)>0;stats['sourcewinner_mapped_with_pool']+=bool(len(M) and mapped[q0]);stats['original_unmapped_with_pool']+=bool(len(M) and not mapped[q]);stats['mapped_only_changes']+=qm!=q
 dump(out/'PROCESS_STATS.json',dict(stats));dump(out/'ROWS_PRIVATE.json',rows_private);seal(out,'R28_counterfactual_predictions_before_current_diagnostic_scoring',n=528,no_model_or_parameter_update=True,not_new_confirmation=True)
 # Postseal CPU task scoring; diagnostics, not independent test selection.
 idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};records=[]
 for r,x in zip(rows(),rows_private):
  z=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));ix=idx[r['id']];p=ix['meta_path']
  if p not in containers:assert sha(p)==ix['metadata_container_sha256'];containers[p]=read(p)
  gt=transform_gt(containers[p]['ground_info'][ix['object_index']]['bbox_3d'],r['platform']);b=z['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);vals=candidate_ious(b,gt)[0];records.append(dict(x,ious=dict(zip(z['names'],map(float,vals)))))
 dump(EXP/'FUSION_ROWS_PRIVATE.json',records);summary={};neg=[]
 for phase,rr in [('ALL528',records),('OLD400',records[:400]),('REPLAY256',records[208:464]),('Q64',records[464:])]:
  summary[phase]={}
  for p in ['drone','quad']:
   pr=[x for x in rr if x['platform']==p];summary[phase][p]={}
   for arm in pr[0]['q']:
    vv=np.array([x['ious'][arm] for x in pr]);delta=vv-np.array([x['ious']['joint_original'] for x in pr]);summary[phase][p][arm]={'n':len(pr),'mIoU_percent':float(vv.mean()*100),'delta_vs_original_pp':float(delta.mean()*100),'gains':int((delta>1e-10).sum()),'harms':int((delta< -1e-10).sum()),'query_changes':sum(x['q'][arm]!=x['q']['joint_original'] for x in pr)}
 for r in records:
  for arm in r['q']:
   if r['ious'][arm]<r['ious']['joint_original']-1e-10:neg.append({'alias':r['alias'],'platform':r['platform'],'stage':r['stage'],'arm':arm,'delta_pp':100*(r['ious'][arm]-r['ious']['joint_original'])})
 dump(EXP/'FUSION_SUMMARY.json',summary);dump(EXP/'FUSION_ALL_NEGATIVES.json',neg);print('FUSION28',dict(stats));print('MAIN_EFFECTS',{p:{a:round(np.mean([summary[p][pl][a]['mIoU_percent'] for pl in ['drone','quad']]),6) for a in ['source','joint_original','legacy_G','mapped_competition_only','median_zero','mean_zero']} for p in ['OLD400','Q64','ALL528']})
if __name__=='__main__':main()
