"""Postseal influence table. Task evidence is diagnostic, never a runtime filter."""
import sys,numpy as np,collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round31'),str(ROOT/'research/round27')]
from common31 import *
from evaluate27 import ious

def main():
 out=BASE/'loo19_attempt002';verify(out);specs=read(out/'MODEL_SPECS_PRIVATE.json');allrows=[dict(r,split=split) for split in ['A208','DEV64','R64','S64'] for r in rows(split)];lookup={r['id']:r for r in allrows};pred={x['name']:load_npz(out/'predictions_PRIVATE'/(x['name']+'.npz')) for x in specs};task={};source={};cache={}
 for r in allrows:
  z=feature(r);q0=int(z['scores'].argmax());qs={q0}
  for x in specs:
   if x['platform']!=r['platform']:continue
   a=pred[x['name']];j=a['row_ids_PRIVATE'].tolist().index(r['id']);qs.add(int(a['queries'][j]))
  q=sorted(qs);v=ious(r,z['boxes'][q]);cache[r['id']]=dict(zip(q,map(float,v)));source[r['id']]=cache[r['id']][q0]
 records=[]
 for spec in specs:
  a=pred[spec['name']];values={}
  for i,rid in enumerate(a['row_ids_PRIVATE'].tolist()):
   r=lookup[rid];q=int(a['queries'][i]);values[rid]=cache[rid][q]
  task[spec['name']]=values
 for spec in specs:
  if spec['kind']!='leave_one_observation':continue
  base=task[spec['platform']+'_all'];minus=task[spec['name']];a=pred[spec['name']];baseline=pred[spec['platform']+'_all']
  for i,rid in enumerate(a['row_ids_PRIVATE'].tolist()):
   r=lookup[rid];delta=100*(minus[rid]-base[rid]);records.append({'feedback_alias':spec['omitted_alias'],'platform':spec['platform'],'model':spec['name'],'target_alias':r['alias'],'target_id_PRIVATE':rid,'target_scene_PRIVATE':r['scene'],'split':r['split'],'is_removed_observation':rid==spec['omitted_id_PRIVATE'],'source_iou':source[rid],'all_iou':base[rid],'minus_iou':minus[rid],'delta_remove_pp':delta,'all_query_PRIVATE':int(baseline['queries'][i]),'minus_query_PRIVATE':int(a['queries'][i]),'query_changed':int(a['queries'][i])!=int(baseline['queries'][i]),'baseline_gain_vs_source_pp':100*(base[rid]-source[rid]),'minus_gain_vs_source_pp':100*(minus[rid]-source[rid])})
 dump(EXP/'ALL_INFLUENCE_ROWS_PRIVATE.json',records);summary=[];fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');fids={f['id_PRIVATE'] for f in fb};r30={r['id_PRIVATE']:r for r in read(ROOT/'research/round30/experiment/ALL336_TASK_ROWS_PRIVATE.json')}
 for spec in specs:
  if spec['kind']!='leave_one_observation':continue
  rs=[r for r in records if r['model']==spec['name']];s={'feedback_alias':spec['omitted_alias'],'platform':spec['platform'],'removed_preferences':spec['original_N']-spec['active_preferences'],'subsets':{}}
  for split in ['A208','DEV64','R64','S64','R64_S64']:
   rr=[r for r in rs if (r['split'] in ['R64','S64'] if split=='R64_S64' else r['split']==split) and not r['is_removed_observation']];dd=np.array([r['delta_remove_pp'] for r in rr]);base=np.array([r['all_iou'] for r in rr]);s['subsets'][split]={'n':len(rr),'mIoU_all_percent':float(base.mean()*100),'mIoU_minus_percent':float(np.mean([r['minus_iou'] for r in rr])*100),'mean_delta_remove_pp':float(dd.mean()),'improves_if_removed':int((dd>1e-8).sum()),'worsens_if_removed':int((dd< -1e-8).sum()),'changed_query':sum(r['query_changed'] for r in rr),'baseline_harms_recovered_pp_sum':float(sum(max(r['minus_iou']-r['all_iou'],0)*100 for r in rr if r['baseline_gain_vs_source_pp']< -1e-8)),'baseline_gains_lost_pp_sum':float(sum(max(r['all_iou']-r['minus_iou'],0)*100 for r in rr if r['baseline_gain_vs_source_pp']>1e-8))}
  for subset in ['A_feedback_others','A_no_feedback']:
   rr=[r for r in rs if r['split']=='A208' and not r['is_removed_observation'] and ((r['target_id_PRIVATE'] in fids) if subset=='A_feedback_others' else r['target_id_PRIVATE'] not in fids)];s['subsets'][subset]={'n':len(rr),'mean_delta_remove_pp':float(np.mean([r['delta_remove_pp'] for r in rr])),'improves_if_removed':sum(r['delta_remove_pp']>1e-8 for r in rr),'worsens_if_removed':sum(r['delta_remove_pp']< -1e-8 for r in rr)}
  summary.append(s)
 # R30 baseline reproduction at prediction+task level on shared A/DEV/S.
 matches=0
 for pl in ['drone','quad']:
  a=pred[pl+'_all']
  for i,rid in enumerate(a['row_ids_PRIVATE'].tolist()):
   if rid not in r30:continue
   assert int(a['queries'][i])==r30[rid]['q']['keep_selected'];assert task[pl+'_all'][rid]==r30[rid]['ious']['keep_selected'];matches+=1
 dump(EXP/'INFLUENCE19_SUMMARY.json',summary);dump(EXP/'BASELINE_REPRODUCTION.json',{'shared_A_DEV_S_predictions_exact':matches,'R30_state_maxerror':[x['baseline_R30_state_maxerror'] for x in read(out/'FIT_RECEIPTS_PRIVATE.json') if x['kind']=='all'],'all_model_diagnostic_only':True});witness=[]
 for alias in ['S_drone_005','S_quad_016']:
  rr=[r for r in records if r['target_alias']==alias];witness.append({'target_alias':alias,'baseline_iou':rr[0]['all_iou'],'source_iou':rr[0]['source_iou'],'influence_sorted':sorted([{k:r[k] for k in ['feedback_alias','minus_iou','delta_remove_pp','query_changed']} for r in rr],key=lambda r:-r['delta_remove_pp'])})
 dump(EXP/'REQUESTED_WITNESSES.json',witness);print('R31_INFLUENCE_SUMMARY')
 for s in summary:print(s['feedback_alias'],'R+S_delta',round(s['subsets']['R64_S64']['mean_delta_remove_pp'],5),'S_delta',round(s['subsets']['S64']['mean_delta_remove_pp'],5),'Srecovered/lost',s['subsets']['S64']['baseline_harms_recovered_pp_sum'],s['subsets']['S64']['baseline_gains_lost_pp_sum'])
 print('WITNESSES',json.dumps(witness))
if __name__=='__main__':main()
