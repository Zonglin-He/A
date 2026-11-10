"""Postseal task and memory-process analysis; never feeds back to runtime."""
import json,csv,io
from common64 import *
def main():
 H=ROOT/'research/round64';p=read(H/'experiment/FORMAL_AUDIT_AND_SCORE_PLAN_LOCK.json');tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');summary=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];pred=ROOT/p['prediction_root'];orders=read(ROOT/p['orders']);case={r['id_PRIVATE']:read(pred/'cases_PRIVATE'/(r['id_PRIVATE']+'.json')) for r in tasks};table=[];scene_rows=[];process=[];stat={};adopt=True
 for o in p['order_names']:
  ref=o+'__r59_direct_all';main=o+'__memory';shu=o+'__shuffled_values';g=summary['balanced_mIoU_percent'][main]-summary['balanced_mIoU_percent'][ref];gs=summary['balanced_mIoU_percent'][main]-summary['balanced_mIoU_percent'][shu];adopt=adopt and g>1e-10 and gs>1e-10
  for name in p['per_order_arms']:
   arm=o+'__'+name;rr={'order':o,'method':name,'Drone_mIoU':summary['drone'][arm]['mIoU_percent'],'Quad_mIoU':summary['quad'][arm]['mIoU_percent'],'balanced_mIoU':summary['balanced_mIoU_percent'][arm],'Acc25':float(np.mean([summary[pl][arm]['Acc25_percent'] for pl in ['drone','quad']])),'Acc50':float(np.mean([summary[pl][arm]['Acc50_percent'] for pl in ['drone','quad']])),'gain_vs_R59_pp':summary['balanced_mIoU_percent'][arm]-summary['balanced_mIoU_percent'][ref],'improved_vs_R59':sum(r['IoU'][arm]>r['IoU'][ref]+1e-10 for r in tasks),'worsened_vs_R59':sum(r['IoU'][arm]<r['IoU'][ref]-1e-10 for r in tasks),'query_changes_vs_R59':sum(r['queries'][arm]!=r['queries'][ref] for r in tasks)};table.append(rr)
   for pl in ['drone','quad']:
    for j,sce in enumerate(sorted({r['scene_PRIVATE'] for r in tasks if r['platform']==pl})):
     tt=[r for r in tasks if r['platform']==pl and r['scene_PRIVATE']==sce];scene_rows.append({'order':o,'method':name,'platform':pl,'scene_alias':('D' if pl=='drone' else 'Q')+'_S'+str(j+1).zfill(2),'n':len(tt),'mIoU':float(np.mean([r['IoU'][arm] for r in tt])*100),'gain_vs_R59_pp':float(np.mean([r['IoU'][arm]-r['IoU'][ref] for r in tt])*100)})
  st={'memory_gain_vs_R59_pp':g,'memory_gain_vs_shuffled_pp':gs,'reset_exact_R59':all(r['queries'][o+'__reset_each_input']==r['queries'][ref] and r['IoU'][o+'__reset_each_input']==r['IoU'][ref] for r in tasks),'R58_accepted':0,'R58_rejected':0}
  for pl in ['drone','quad']:
   seq=orders[o][pl]
   for pos,sid in enumerate(seq):
    r=case[sid][o];item={'order':o,'platform':pl,'stream_position':pos,'pool_before':r['pool_before'],'pool_after':r['pool_after'],'write_proposals':r['write_proposals'],'replacements':r['replacements'],'evictions':r['evictions']}
    for kind in ['memory','shuffled_values','no_explicit_text']:
     mm=r['memory'][kind];item[kind+'_matched_views']=sum(mm[v]['matched_candidates']>0 for v in 'ABC');item[kind+'_matched_candidates']=sum(mm[v]['matched_candidates'] for v in 'ABC');item[kind+'_median_correction_norm']=float(np.mean([mm[v]['median_correction_norm'] for v in 'ABC']))
    if 'R58_step' in r:st['R58_accepted' if r['R58_step']['accepted'] else 'R58_rejected']+=1
    process.append(item)
  stat[o]=st
 means=[]
 for name in p['per_order_arms']:
  rr=[r for r in table if r['method']==name];means.append({'method':name,**{k:float(np.mean([r[k] for r in rr])) for k in ['Drone_mIoU','Quad_mIoU','balanced_mIoU','Acc25','Acc50','gain_vs_R59_pp']},'order_min_mIoU':min(r['balanced_mIoU'] for r in rr),'order_max_mIoU':max(r['balanced_mIoU'] for r in rr)})
 for fname,rows in [('ALL_ORDER_METHOD_ROWS',table),('ALL_ORDER_SCENE_ROWS',scene_rows),('ALL_STREAM_PROCESS_SCALARS',process),('MEAN_ACROSS_FIXED_ORDERS',means)]:dump(H/(fname+'.json'),rows)
 decision={'state':'R64_COMPLETED_DEVELOPMENT_MEMORY_CANDIDATE' if adopt else 'R64_COMPLETED_REJECT_THIS_MEMORY_TRANSFER_RECIPE','new_TTA_adopted':False,'locked_development_criterion_passed':bool(adopt),'order_checks':stat,'FINAL384_accessed':False,'source_retrained':False,'postscore_retuning':False,'scope':'Three orders of the same historically exposed DEV336; not 1008 independent observations or new test generalization.','action':'Do not expand capacity, retune alpha or rescue via order selection. Keep R59 and original R58 references; next work requires a different evidence-backed transfer mechanism.' if not adopt else 'Keep as development candidate only; no final holdout accessed and no robust-generalization claim.'};dump(H/'DECISION.json',decision);print(json.dumps({'decision':decision,'means':means},indent=2))
if __name__=='__main__':main()
