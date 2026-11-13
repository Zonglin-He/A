"""Saved-only source hardness and annotation-supported pair margins; no selection."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round91';P=R/'crossplatform/round91/trained_source_attempt001';read=lambda p:json.loads(p.read_text());rows=read(H/'SOURCE_PAIR_DISCRIMINATION_ROWS.json');coverage=read(P/'PAIR_COVERAGE_ROWS.json');hard=[];summary=[]
for b in ['R76','R89continue']:
 for sp in ['TRAIN','VAL']:
  x=[r for r in coverage if r['base']==b and r['row'].startswith(sp)];hard.append({'base':b,'split':sp,'n':len(x),'pair_eligible_n':sum(r['pair_count']>0 for r in x),'native_top1_is_accepted_negative_n':sum(r['native_top1_is_accepted_negative'] for r in x),'pair_count':sum(r['pair_count'] for r in x)})
 for h in ['query_disc','full_ce','full_disc','no_ref_disc','no_local_disc']:
  x=[r for r in rows if r['base']==b and r['head']==h];n=sum(r['pair_count'] for r in x);summary.append({'base':b,'head':h,'eligible_examples':len(x),'pair_count':n,'positive_margin_percent':100*sum(r['positive_margin_n'] for r in x)/n,'training_margin_satisfied_percent':100*sum(r['above_training_margin_n'] for r in x)/n,'pair_weighted_mean_margin':sum(r['mean_margin']*r['pair_count'] for r in x)/n})
 vals=read(R/'research/round76/input/SOURCE_VAL128_PRIVATE.json');old=np.load(R/'crossplatform/round89/trained_source_attempt001/val_PRIVATE/continue_e1.npz')['scores'][:,0];margins=[];nex=0
 for i,r in enumerate(vals):
  pairs=np.load(P/'pairs_PRIVATE'/(b+'_'+r['id']+'.npz'))['pairs_PRIVATE']
  if not len(pairs):continue
  p=np.load(R/'crossplatform/round91/source_features640_attempt001/samples_PRIVATE'/(r['id']+'.npz'))['0_baseline'] if b=='R76' else old[i];s=np.log(np.maximum(p,1e-300));margins.extend((s[pairs[:,0]]-s[pairs[:,1]]).tolist());nex+=1
 v=np.array(margins);summary.append({'base':b,'head':'native_baseline','eligible_examples':nex,'pair_count':len(v),'positive_margin_percent':float((v>0).mean()*100),'training_margin_satisfied_percent':float((v>=.2).mean()*100),'pair_weighted_mean_margin':float(v.mean())})
for n,x in [('SOURCE_HARDNESS_ROWS',hard),('SOURCE_DISCRIMINATION_SUMMARY_ROWS',summary)]:(H/(n+'.json')).write_text(json.dumps(x,indent=2)+'\n')
print({'source_hardness_rows':len(hard),'source_discrimination_rows':len(summary),'new_models_or_GT_scoring':False})
hist=read(P/'HISTORY.json');out=[]
for b in ['R76','R89continue']:
 for h in ['query_disc','full_ce','full_disc','no_ref_disc','no_local_disc']:
  rr=[x for x in hist if x['base']==b and x['head']==h];best=max(rr,key=lambda x:(x['source_val_mIoU_percent'],-x['epoch']));out.append({'base':b,'head':h,'own_best_epoch_source_only_diagnostic':best['epoch'],'own_best_source_mIoU_percent':best['source_val_mIoU_percent'],'shared_selected_epoch':read(P/'SELECTED_SOURCE_ONLY.json')['shared_epoch'],'used_to_change_target_checkpoint':False})
(H/'SOURCE_CHECKPOINT_DIAGNOSTIC_ROWS.json').write_text(json.dumps(out,indent=2)+'\n')
