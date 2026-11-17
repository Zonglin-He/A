"""Analysis only: regroup sealed predictions and already-computed per-box IoUs.

No neural model/torch/CUDA, raw media/annotations, geometric IoU calculation,
training, new predictions, loss/rule modification or FINAL access.
"""
from pathlib import Path
import json,hashlib,collections,csv,io
import numpy as np
R=Path(__file__).resolve().parents[3];O=Path(__file__).resolve().parent
def rd(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def metric(rows,k,weighted=True):
 parts=[]
 for name in sorted({r['direction'] for r in rows}):
  rr=[r for r in rows if r['direction']==name];w=np.array([r['weight'] if weighted else 1. for r in rr]);y=np.array([r[k] for r in rr]);parts.append(np.array([np.average(y,weights=w),np.average(y>.25,weights=w),np.average(y>.5,weights=w)]))
 x=np.mean(parts,axis=0)*100
 return dict(zip(['mIoU','Acc25','Acc50'],map(float,x)))
def table(path,rows):
 if not rows:return
 s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows);path.write_text(s.getvalue())
def main():
 assert rd(R/'research/round134/STATE.json')['execution_open'] is False
 reg=rd(R/'research/round134/INPUT_REGISTRY_PRIVATE.json');rows=[];files={};stages=['original_source','prepared_frozen_A','foundation_A','foundation_B','frozen_FAB','LOG3','R134','A256_oracle','S_oracle','C_oracle','selected_group_oracle']
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];p=R/'crossplatform/round134/formal'/name/'samples_PRIVATE'/(sid+'.npz');evp=R/e['evidence_path'];ip=R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz');p97=R/'crossplatform/round97'/name/'samples_PRIVATE'/(sid+'.npz');p131=R/'crossplatform/round131/formal'/name/'features_PRIVATE'/(sid+'.npz')
  a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));ev=np.load(evp);y=np.load(ip)['ious'];a97=np.load(p97);p97s=dict(zip(a97['arm_names'],a97['scores']));f131=np.load(p131);assert np.array_equal(a['source_boxes'],a97['source_boxes']);assert y.shape==(256,) and np.isfinite(y).all();S,C,ass=ev['S'],ev['C'],ev['assignment'];assert np.all(~C|S);q0=int(ps['zero_F'].argmax());qa=int(ps['partial_pair_F'].argmax());best=int(y.argmax());B=bool(f131['valid_B'].any());gs=ass[qa];maskS=S[ass];maskC=C[ass]
  vals={'original_source':float(y[p97s['original_source'].argmax()]),'prepared_frozen_A':float(y[p97s['prepared_frozen'].argmax()]),'foundation_A':float(y[f131['pA'].argmax()]),'foundation_B':float(y[f131['pB'].argmax()]),'frozen_FAB':float(y[q0]),'LOG3':float(y[ps['partial3_F'].argmax()]),'R134':float(y[qa]),'A256_oracle':float(y.max()),'S_oracle':float(y[maskS].max()) if maskS.any() else 0.,'C_oracle':float(y[maskC].max()) if maskC.any() else 0.,'selected_group_oracle':float(y[ass==gs].max())}
  if not B:assert np.allclose(f131['pA'],f131['pB'],atol=1e-7) and np.allclose(f131['pA'],ps['zero_F'],atol=1e-7)
  classes={}
  for th in [.25,.5]:
   label=str(th)
   if vals['R134']>th:cat='final_pass'
   elif vals['A256_oracle']<=th:cat='A256_no_passing_box'
   elif not B:cat='passing_box_but_no_usable_B'
   elif vals['S_oracle']<=th:cat='B_available_but_passing_box_outside_S'
   elif vals['C_oracle']<=th:cat='passing_box_in_S_but_outside_C'
   else:cat='passing_box_in_C_but_final_wrong'
   classes[label]=cat
  rank=np.argsort(-ps['partial_pair_F'],kind='stable');rank0=np.argsort(-ps['zero_F'],kind='stable');good25=y>.25;good50=y>.5;finalgap=vals['A256_oracle']-vals['R134'];groupgap=vals['A256_oracle']-vals['selected_group_oracle'];withingap=vals['selected_group_oracle']-vals['R134'];assert finalgap>=-1e-12 and groupgap>=-1e-12 and withingap>=-1e-12
  rows.append({'source':e['source'],'target':e['target'],'direction':name,'id_PRIVATE':sid,'frame_PRIVATE':e['frame_PRIVATE'],'content_group_PRIVATE':e['content_group_PRIVATE'],'scene_PRIVATE':e['row']['scene'],'weight':e['sampling_weight'],'B_available':B,'partial_active':e['active'],'pair_active':e['pair_active'],'actual_update':bool(e['active'] or e['pair_active']),'S_groups':int(S.sum()),'C_groups':int(C.sum()),'q0_in_S':bool(S[ass[q0]]),'q0_in_C':bool(C[ass[q0]]),'qa_in_S':bool(S[gs]),'qa_in_C':bool(C[gs]),'top_challenged':bool((ev['lose']==ass[q0]).any()),**vals,'geometry_gap':1-vals['A256_oracle'],'selection_gap':finalgap,'between_geometric_group_gap':groupgap,'within_geometric_group_gap':withingap,'class25':classes['0.25'],'class50':classes['0.5'],'bestA_rank_final':int(np.flatnonzero(rank==best)[0])+1,'bestA_rank_zero':int(np.flatnonzero(rank0==best)[0])+1,'first_good25_rank_final':int(np.flatnonzero(good25[rank])[0])+1 if good25.any() else None,'first_good50_rank_final':int(np.flatnonzero(good50[rank])[0])+1 if good50.any() else None,'new_vs_zero_IoU':vals['R134']-vals['frozen_FAB'],'new_vs_LOG3_IoU':vals['R134']-vals['LOG3']})
  for path in [p,evp,ip,p97,p131]:files[str(path.relative_to(R))]=sha(path)
 assert len(rows)==1798
 # Explicit design weights: equal directions, original within-direction sampling weights.
 for r in rows:r['macro_weight']=r['weight']/sum(x['weight'] for x in rows if x['direction']==r['direction'])/6
 assert abs(sum(r['macro_weight'] for r in rows)-1)<1e-12
 metrics={ag:{s:metric(rows,s,ag=='design_weighted') for s in stages} for ag in ['design_weighted','raw']}
 for ag in metrics:
  published={r['arm']:r for r in rd(R/'research/round134/scores_attempt001/AGGREGATE_ROWS.json') if r['aggregation']==ag and r['group']=='all_six_equal'}
  for s,arm in [('frozen_FAB','zero_F'),('LOG3','partial3_F'),('R134','partial_pair_F')]:
   for m,old in [('mIoU','mIoU_percent'),('Acc25','Acc25_percent'),('Acc50','Acc50_percent')]:assert abs(metrics[ag][s][m]-published[arm][old])<1e-10
 gaps={k:float(sum(r[k]*r['macro_weight'] for r in rows)*100) for k in ['geometry_gap','selection_gap','between_geometric_group_gap','within_geometric_group_gap']};gaps['total_gap_to_unit_IoU']=100-metrics['design_weighted']['R134']['mIoU'];assert abs(gaps['geometry_gap']+gaps['selection_gap']-gaps['total_gap_to_unit_IoU'])<1e-10;assert abs(gaps['between_geometric_group_gap']+gaps['within_geometric_group_gap']-gaps['selection_gap'])<1e-10
 classes=[]
 for label in ['class25','class50']:
  for cat in ['final_pass','A256_no_passing_box','passing_box_but_no_usable_B','B_available_but_passing_box_outside_S','passing_box_in_S_but_outside_C','passing_box_in_C_but_final_wrong']:
   rr=[r for r in rows if r[label]==cat];classes.append({'threshold':label,'category':cat,'n':len(rr),'percent_of_1798':len(rr)/1798*100,'design_weighted_mass_percent':sum(r['macro_weight'] for r in rr)*100,'selection_regret_contribution_pp':sum(r['selection_gap']*r['macro_weight'] for r in rr)*100,'geometry_gap_contribution_pp':sum(r['geometry_gap']*r['macro_weight'] for r in rr)*100})
 loss=[]
 for base in ['original_source','prepared_frozen_A','foundation_A','frozen_FAB','LOG3']:
  d=np.array([r['R134']-r[base] for r in rows]);w=np.array([r['macro_weight'] for r in rows]);loss.append({'baseline':base,'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'positive_contribution_pp':float(np.dot(np.maximum(d,0),w)*100),'negative_contribution_pp':float(np.dot(np.minimum(d,0),w)*100),'net_pp':float(np.dot(d,w)*100),'GT_only_no_harm_selection_upper_bound':float(sum(max(r['R134'],r[base])*r['macro_weight'] for r in rows)*100),'Acc25_rescues':sum(r[base]<=.25 and r['R134']>.25 for r in rows),'Acc25_harms':sum(r[base]>.25 and r['R134']<=.25 for r in rows)})
 coverage={}
 for scope in ['all','B_available','actual_update','pair_active']:
  rr=rows if scope=='all' else [r for r in rows if r[scope]];zero_wrong=[r for r in rr if r['frozen_FAB']<=.25];recover=[r for r in zero_wrong if r['A256_oracle']>.25];coverage[scope]={'n':len(rr),'zero_Acc25_wrong':len(zero_wrong),'zero_wrong_A256_recoverable':len(recover),'recoverable_good_in_S':sum(r['S_oracle']>.25 for r in recover),'recoverable_good_in_C':sum(r['C_oracle']>.25 for r in recover),'recoverable_C_good_top_challenged':sum(r['C_oracle']>.25 and r['top_challenged'] for r in recover),'actual_update_count':sum(r['actual_update'] for r in rr)}
 ranking={}
 for th in [.25,.5]:
  eligible=[r for r in rows if r['A256_oracle']>th];k='first_good25_rank_final' if th==.25 else 'first_good50_rank_final';ranking[str(th)]={'A256_eligible_n':len(eligible),'good_in_top1':sum(r[k]<=1 for r in eligible),'good_in_top5':sum(r[k]<=5 for r in eligible),'good_in_top10':sum(r[k]<=10 for r in eligible),'good_in_top20':sum(r[k]<=20 for r in eligible),'good_in_top50':sum(r[k]<=50 for r in eligible),'median_first_good_rank':float(np.median([r[k] for r in eligible]))}
 bydirection=[]
 for name in sorted({r['direction'] for r in rows}):
  rr=[r for r in rows if r['direction']==name];entry={'direction':name,'n':len(rr)}
  for s in stages:
   for m,v in metric(rr,s).items():entry[s+'__'+m]=v
  for k in ['geometry_gap','selection_gap','between_geometric_group_gap','within_geometric_group_gap']:entry[k]=sum(r[k]*r['weight'] for r in rr)/sum(r['weight'] for r in rr)*100
  bydirection.append(entry)
 dump(O/'ALL_ROWS_PRIVATE.json',rows);dump(O/'METRICS.json',metrics);dump(O/'GAP_DECOMPOSITION.json',gaps);dump(O/'THRESHOLD_PARTITION.json',classes);dump(O/'TTA_GAIN_LOSS.json',loss);dump(O/'SUPERVISION_COVERAGE.json',coverage);dump(O/'RANKING.json',ranking);table(O/'SIX_DIRECTION_METRICS.csv',bydirection)
 table(O/'THRESHOLD_PARTITION.csv',classes);table(O/'TTA_GAIN_LOSS.csv',loss)
 summary={'analysis_only':True,'n':len(rows),'new_model_or_training_runs':0,'new_predictions':0,'new_raw_GT_or_media_reads':0,'new_geometric_IoU_computations':0,'method_code_changed':False,'FINAL_access':False,'candidate_oracle_is_posthoc_GT_only':True,'S_C_are_supervision_not_hard_final_filters':True,'metrics':metrics,'gap_decomposition':gaps,'threshold_partition':classes,'TTA_gain_loss':loss,'supervision_coverage':coverage,'ranking':ranking}
 dump(O/'SUMMARY.json',summary);dump(O/'READ_RECEIPT_PRIVATE.json',{'cache_files_read':len(files),'files':files,'new_experiment':False,'method_changes':False,'FINAL_access':False})
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
