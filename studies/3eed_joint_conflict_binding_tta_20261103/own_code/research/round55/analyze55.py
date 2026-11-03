"""Post-seal descriptive analysis of frozen task scalars and predictions; no GT I/O."""
from common55 import *
from scipy.special import softmax,logsumexp
from adapt55 import distributions
import torch

def main():
 score=ROOT/'research/round55/score_dev336_attempt001';rows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');tasks=read(score/'ALL_TASK_ROWS_PRIVATE.json');summ=read(score/'TASK_SUMMARY.json')['ALL336'];diags=read(score/'ALL_CANDIDATE_IOUS_PRIVATE.json');out=ROOT/'research/round55/analysis_attempt001';assert not out.exists();out.mkdir();pred=ROOT/'crossplatform/round55/adapted336_attempt002';obs=ROOT/'crossplatform/round55/conflicts336_attempt001';metrics=[];mechanics=[];publicfits=[];detail=[]
 for arm in ARMS:
  v=np.array([t['IoU'][arm] for t in tasks]);base=np.array([t['IoU']['source'] for t in tasks]);direct=np.array([t['IoU']['real_direct'] for t in tasks]);metrics.append({'arm':arm,'drone_mIoU_percent':summ['drone'][arm]['mIoU_percent'],'quad_mIoU_percent':summ['quad'][arm]['mIoU_percent'],'balanced_mIoU_percent':summ['balanced_mIoU_percent'][arm],'gain_vs_source_pp':float((v-base).mean()*100),'gain_vs_real_direct_pp':float((v-direct).mean()*100),'balanced_Acc25_percent':float((v>.25).mean()*100),'balanced_Acc50_percent':float((v>.5).mean()*100),'improved_vs_source':int((v-base>1e-10).sum()),'worsened_vs_source':int((v-base< -1e-10).sum()),'query_changes':sum(t['queries'][arm]!=t['queries']['source'] for t in tasks)})
 for no,(row,t,d) in enumerate(zip(rows,tasks,diags)):
  sid=row['id'];assert d['id_PRIVATE']==sid;rec=read(pred/'cases_PRIVATE'/(sid+'.json'));e=load(obs/'samples_PRIVATE'/(sid+'.npz'),['conflicts','role_masks']);l=load(ROOT/'crossplatform/round53/latent336_attempt001/samples_PRIVATE'/(sid+'.npz'),['query','tokens']);f=load(pred/'fitted_PRIVATE'/(sid+'.npz'));logits=l['query'].astype(float)@l['tokens'].astype(float).T/.07;z0=np.stack([logsumexp(logits[:,m],-1)-logsumexp(logits[:,~m],-1) for m in e['role_masks']]);p0=softmax(z0,axis=-1);ious=np.array(d['candidate_IoUs_PRIVATE']);q0=t['queries']['source'];best=int(ious.argmax());C=e['conflicts'][0]
  assert all(abs(ious[t['queries'][a]]-t['IoU'][a])<1e-10 for a in ARMS)
  strength=1-np.prod(1-np.einsum('rj,rij->ri',p0[1:],C),axis=0) if len(C) else np.zeros(256)
  det={'row':f'DEV336_{no+1:05d}','platform':row['platform'],'source_IoU':float(ious[q0]),'oracle_IoU':float(ious.max()),'source_top1_conflict_strength':float(strength[q0]),'oracle_box_conflict_strength':float(strength[best]),'source_wrong25_oracle_recoverable':bool(ious[q0]<=.25 and ious.max()>.25),'best_box_more_penalized_than_source':bool(strength[best]>strength[q0]+1e-10),'source25good_and_conflicted':bool(ious[q0]>.25 and strength[q0]>1e-10),'localization_proxy_only_not_instance_or_reference_gold':True};detail.append(det)
  for vi,variant in enumerate(VARIANTS):
   tr=rec['statistics'][variant];ss=tr['steps'];z3=f[variant+'_3__role_scores'];c=e['conflicts'][vi];pt=torch.from_numpy(z3);pp=distributions(pt,torch.from_numpy(c),torch.from_numpy(z0)).numpy();pbase=softmax(z0,axis=-1);known=np.r_[c.any(0).any(-1)[None],c.any(1)] if len(c) else np.zeros_like(z0,bool)
   entry={'row':det['row'],'platform':row['platform'],'variant':variant,'active':ss[0]['active'],'initial_loss':tr['initial_loss'],'final_loss':ss[-1]['loss_after'],'initial_conflict_loss':tr['initial_loss'],'final_conflict_loss':ss[-1]['conflict_after'],'final_keep':ss[-1]['keep_after'],'delta_L2':ss[-1]['parameter_delta_L2'],'first_gradient_norm':ss[0]['gradient_norm'],'target_distribution_L1':float(abs(pp[0]-pbase[0]).sum()),'mean_reference_distribution_L1':float(abs(pp[1:]-pbase[1:]).sum(1).mean()) if len(pp)>1 else 0.,'reference_top1_changes':int((pp[1:].argmax(1)!=pbase[1:].argmax(1)).sum()) if len(pp)>1 else 0,'initial_top1_target_conflict_mass':tr['initial_top1_target_conflict_mass'],'source_IoU':t['IoU']['source'],'direct_IoU':t['IoU'][variant+'_direct'],'tta3_raw_IoU':t['IoU'][variant+'_tta3_raw'],'tta3_joint_IoU':t['IoU'][variant+'_tta3_joint']};publicfits.append(entry)
 for variant in VARIANTS:
  rr=[x for x in publicfits if x['variant']==variant];active=[x for x in rr if x['active']]
  rec={'variant':variant,'n':336,'active':len(active),'nonzero_updates':sum(x['delta_L2']>0 for x in rr),'objective_lower_active':sum(x['final_loss']<x['initial_loss'] for x in active),'conflict_lower_active':sum(x['final_conflict_loss']<x['initial_conflict_loss'] for x in active),'reference_argmax_changes':sum(x['reference_top1_changes'] for x in rr)}
  for key in ['initial_loss','final_loss','final_conflict_loss','final_keep','delta_L2','first_gradient_norm','target_distribution_L1','mean_reference_distribution_L1','initial_top1_target_conflict_mass']:
   rec[key+'_median_active']=float(np.median([x[key] for x in active])) if active else None
  mechanics.append(rec)
 coverage=[]
 for pl in ['drone','quad','balanced_all']:
  rr=detail if pl=='balanced_all' else [x for x in detail if x['platform']==pl]
  recover=[x for x in rr if x['source_wrong25_oracle_recoverable']];good=[x for x in rr if x['source_IoU']>.25]
  coverage.append({'platform':pl,'n':len(rr),'source_wrong25_but_any_candidate25':len(recover),'recoverable_source_top1_has_any_conflict':sum(x['source_top1_conflict_strength']>1e-10 for x in recover),'recoverable_best_has_more_conflict_than_source':sum(x['best_box_more_penalized_than_source'] for x in recover),'source25good':len(good),'source25good_has_any_conflict':sum(x['source25good_and_conflicted'] for x in good),'oracle25_but_best_has_any_conflict':sum(x['oracle_IoU']>.25 and x['oracle_box_conflict_strength']>1e-10 for x in rr),'no_candidate25':sum(x['oracle_IoU']<=.25 for x in rr),'diagnostic_not_identity_or_reference_truth':True})
 scene=read(score/'ALL_SCENE_SUMMARIES_PRIVATE.json');scenes=[]
 for pl in ['drone','quad']:
  names=sorted({x['scene_PRIVATE'] for x in scene if x['platform']==pl})
  for x in scene:
   if x['scope']=='ALL336' and x['platform']==pl:scenes.append({'platform':pl,'scene_alias':('D' if pl=='drone' else 'Q')+'_S'+str(names.index(x['scene_PRIVATE'])+1).zfill(2),'n':x['n'],'mIoU_percent':x['mIoU_percent']})
 for name,value in [('ARM_METRICS',metrics),('OPTIMIZATION_AND_ROLE_DIAGNOSTICS',mechanics),('CONFLICT_LOCALIZATION_PROXY_COVERAGE',coverage),('ALL_FIT_DIAGNOSTICS_PUBLIC_SAFE',publicfits),('ALL_OBSERVATION_LOCALIZATION_PROXY_PUBLIC_SAFE',detail),('ALL_SCENE_METRICS',scenes)]:dump(out/(name+'.json'),value)
 print(json.dumps({'metrics':metrics,'mechanics':mechanics,'coverage':coverage}),flush=True)
if __name__=='__main__':main()
