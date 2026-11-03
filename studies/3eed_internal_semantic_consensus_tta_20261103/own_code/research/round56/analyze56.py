"""Postsealed descriptive task/role/stage diagnostics, no raw GT/geometry/models."""
from scipy.special import logsumexp
from common56 import *
def main():
 r=ROOT/'research/round56';score=r/'score_dev336_attempt001';tasks=read(score/'ALL_TASK_ROWS_PRIVATE.json');oracle=read(score/'OFFLINE_EARLY_ORACLE_PUBLIC_SAFE.json');rows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');summary=read(score/'TASK_SUMMARY.json')['ALL336'];oldlabels=read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json');out=r/'analysis_attempt002';assert not out.exists();out.mkdir();metrics=[];fitrows=[];roles=[];topk=[]
 for arm in ARMS:
  v=np.array([t['IoU'][arm] for t in tasks]);base=np.array([t['IoU']['source'] for t in tasks]);direct=np.array([t['IoU']['semantic_consensus'] for t in tasks]);metrics.append({'arm':arm,'drone_mIoU_percent':summary['drone'][arm]['mIoU_percent'],'quad_mIoU_percent':summary['quad'][arm]['mIoU_percent'],'balanced_mIoU_percent':summary['balanced_mIoU_percent'][arm],'gain_vs_source_pp':float((v-base).mean()*100),'gain_vs_semantic_consensus_pp':float((v-direct).mean()*100),'balanced_Acc25_percent':float((v>.25).mean()*100),'balanced_Acc50_percent':float((v>.5).mean()*100),'improved_vs_source':int((v-base>1e-10).sum()),'worsened_vs_source':int((v-base< -1e-10).sum()),'query_changes_vs_source':sum(t['queries'][arm]!=t['queries']['source'] for t in tasks)})
 for i,(row,t,label) in enumerate(zip(rows,tasks,oldlabels)):
  sid=row['id'];rec=read(ROOT/'crossplatform/round56/adapted336_attempt001/cases_PRIVATE'/(sid+'.json'));cap=load(ROOT/'crossplatform/round56/semantic_capture336_attempt001/samples_PRIVATE'/(sid+'.npz'),['stage_role_z','role_masks']);z=cap['stage_role_z'][-1].astype(float)-np.log(cap['role_masks'].sum(1))[:,None];q=t['queries']['source'];gap=float(z[0,q]-np.max(z[1:,q])) if len(z)>1 else None;roles.append({'row':f'DEV336_{i+1:05d}','platform':row['platform'],'source_IoU':t['IoU']['source'],'reference_role_dominates_source_Top1':bool(gap is not None and gap<0),'target_minus_max_reference_role_score':gap,'n_roles':len(z),'not_semantic_identity_GT':True})
  for prefix,stem in [('', 'TTA'),('shuffle_','shuffle_TTA')]:
   for k in [0,2]:
    s=rec['statistics'][stem]['trace'][k];arm=prefix+'tta'+str(s['step']);fitrows.append({'row':f'DEV336_{i+1:05d}','platform':row['platform'],'arm':arm,'initial_KL':rec['statistics'][stem]['initial_KL'],**s,'source_IoU':t['IoU']['source'],'direct_IoU':t['IoU']['semantic_consensus' if not prefix else 'shuffled_consensus'],'adapted_IoU':t['IoU'][arm]})
  pred=load(ROOT/'crossplatform/round56/adapted336_attempt001/samples_PRIVATE'/(sid+'.npz'),['scores']);iou=np.array(label['candidate_IoUs_PRIVATE']);assert label['id_PRIVATE']==sid
  for arm in ['source','proposal']+['decoder'+str(j) for j in range(5)]+['early_consensus','semantic_consensus']:
   order=np.argsort(-pred['scores'][ARMS.index(arm)],kind='stable');topk.append({'platform':row['platform'],'arm':arm,**{f'Top{k}_hit25':bool((iou[order[:k]]>.25).any()) for k in [1,5,10]}})
 mechanics=[]
 for arm in ['tta1','tta3','shuffle_tta1','shuffle_tta3']:
  rr=[x for x in fitrows if x['arm']==arm];mechanics.append({'arm':arm,'n':len(rr),'nonzero_updates':sum(x['parameter_delta_L2']>0 for x in rr),'KL_lower':sum(x['KL_after']<x['initial_KL'] for x in rr),'median_initial_KL':float(np.median([x['initial_KL'] for x in rr])),'median_final_KL':float(np.median([x['KL_after'] for x in rr])),'median_gradient_norm':float(np.median([x['gradient_norm'] for x in rr])),'median_parameter_delta_L2':float(np.median([x['parameter_delta_L2'] for x in rr]))})
 oracles=[]
 for pl in ['drone','quad','balanced_all']:
  oo=oracle if pl=='balanced_all' else [x for x in oracle if x['platform']==pl];rl=roles if pl=='balanced_all' else [x for x in roles if x['platform']==pl];oracles.append({'platform':pl,'n':len(oo),'source_mIoU_percent':float(np.mean([x['source_IoU'] for x in oo])*100),'offline_early_Top1_oracle_mIoU_percent':float(np.mean([x['early_Top1_oracle_IoU'] for x in oo])*100),'offline_source_plus_early_oracle_mIoU_percent':float(np.mean([max(x['source_IoU'],x['early_Top1_oracle_IoU']) for x in oo])*100),'source_failed25_earlier_Top1_can_rescue':sum(x['source_failed25_early_can_rescue'] for x in oo),'source_IoU_improvable_by_earlier_Top1':sum(x['early_Top1_oracle_IoU']>x['source_IoU']+1e-10 for x in oo),'all256_oracle_hit25_n':sum(x['all256_oracle_IoU']>.25 for x in oo),'reference_dominates_source_Top1':sum(x['reference_role_dominates_source_Top1'] for x in rl),'role_dominance_not_identity_truth':True,'offline_oracle_never_used_in_inference':True})
 tk=[]
 for pl in ['drone','quad']:
  for arm in ['source','proposal']+['decoder'+str(j) for j in range(5)]+['early_consensus','semantic_consensus']:
   rr=[x for x in topk if x['platform']==pl and x['arm']==arm];tk.append({'platform':pl,'arm':arm,'n':len(rr),**{f'Top{k}_hit25_percent':float(np.mean([x[f'Top{k}_hit25'] for x in rr])*100) for k in [1,5,10]}})
 scenes=[];sc=read(score/'ALL_SCENE_SUMMARIES_PRIVATE.json')
 for pl in ['drone','quad']:
  names=sorted({x['scene_PRIVATE'] for x in sc if x['platform']==pl})
  for x in sc:
   if x['scope']=='ALL336' and x['platform']==pl:scenes.append({'platform':pl,'scene_alias':('D' if pl=='drone' else 'Q')+'_S'+str(names.index(x['scene_PRIVATE'])+1).zfill(2),'n':x['n'],'mIoU_percent':x['mIoU_percent']})
 for name,v in [('ARM_METRICS',metrics),('OPTIMIZATION_DIAGNOSTICS',mechanics),('EARLY_ORACLE_AND_ROLE_DIAGNOSTICS',oracles),('STAGE_TOPK_DIAGNOSTICS',tk),('ALL_FIT_DIAGNOSTICS_PUBLIC_SAFE',fitrows),('ALL_ROLE_DIAGNOSTICS_PUBLIC_SAFE',roles),('ALL_SCENE_METRICS',scenes)]:dump(out/(name+'.json'),v)
 print({'metrics':metrics,'mechanics':mechanics,'oracle':oracles},flush=True)
if __name__=='__main__':main()
