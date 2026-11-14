"""Register the fixed existing mainline; source-only preparation, no target selection."""
from pathlib import Path
import json,hashlib,collections,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round97';rd=lambda p:json.loads(p.read_text())
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def choose(rows,n,salt):
 by=collections.defaultdict(list)
 for r in rows:by[r['scene']].append(r)
 for sc in by:by[sc].sort(key=lambda r:hashlib.sha256((salt+'|'+str(r['suffix'])+'|'+r['id']).encode()).hexdigest())
 order=sorted(by,key=lambda s:hashlib.sha256((salt+'scene|'+s).encode()).hexdigest());out=[]
 while len(out)<n:
  for sc in order:
   if by[sc]:out.append(by[sc].pop(0))
   if len(out)==n:break
 return out
def rows97(pl,split):
 ix=rd(R/f'research/round96/experiment/OFFICIAL_{pl.upper()}_{split.upper()}_ORDER_PRIVATE.json');out=[]
 for i,a in enumerate(ix):
  mp=Path(a['meta_path']);assert mp.is_relative_to(R/'crossplatform/round96/data_view');meta=rd(mp);obj=meta['ground_info'][a['object_index']];assert hashlib.sha256(obj['caption'].encode()).hexdigest()==a['caption_sha256'];caption=' '.join(obj['caption'].replace(',',' ,').split());identity=pl+'|'+str(mp.relative_to(R/'crossplatform/round96/data_view'))+'|'+str(a['object_index']);sid=hashlib.sha256(identity.encode()).hexdigest()[:20];point=mp.parent/('lidar.npy' if pl=='waymo' else 'lidar.bin');im=mp.parent/'image.jpg';assert point.is_file() and im.is_file()
  out.append({'id':sid,'platform':pl,'scene':a['scene'],'pointpath':str(point),'metapath':str(mp),'imagepath':str(im),'caption':caption,'suffix':int(a['frame'].split('_')[-1]) if pl=='waymo' else None,'seed':int(hashlib.sha256(('R97point|'+identity).encode()).hexdigest()[:16],16),'object_index':a['object_index'],'anno_index':i})
 return out
def main():
 assert not (H/'experiment/DESIGN_LOCK.json').exists();targets={};sources={};contracts={}
 for pl in ['waymo','drone','quad']:
  rr=rows97(pl,'val');assert len(rr)=={'waymo':3782,'drone':2792,'quad':2736}[pl];dump(H/f'input/TARGET_{pl.upper()}_PRIVATE.json',rr);targets[pl]=len(rr)
  if pl!='waymo':
   tt=rows97(pl,'train');tr=choose(tt,512,'R76train');va=choose(rr,128,'R76val');assert not {r['scene'] for r in tr}&{r['scene'] for r in va};assert not {r['pointpath'] for r in tr}&{r['pointpath'] for r in va}
   for s,v in [('TRAIN512',tr),('VAL128',va)]:dump(H/f'input/SOURCE_{pl.upper()}_{s}_PRIVATE.json',v)
   sources[pl]={'train_n':512,'val_n':128,'train_scenes':len({r['scene'] for r in tr}),'val_scenes':len({r['scene'] for r in va})}
  for p in (H/'input').glob('*.json'):contracts[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
 p={'method':'R91_R76_full_disc_physical_bt','mainline_choice':'user-selected R91 architecture and original physical BT, not stopped R95 LM recipe or posthoc best method','sources':rd(R/'research/round96/SOURCE_MODEL_COLLECTION_PRIVATE.json')['models'],'target_n':targets,'source_preparation':sources,'Waymo_preparation_reused':'crossplatform/round91/trained_source_attempt001 selected R76_full_disc, R76 bidirectional e06','new_source_preparation':'R76 bidirectional only: 512/128,8epochs,AdamW3e-4/wd1e-4/clip1,BF16training/FP32sourceVAL,sourceVAL earliest max; R91 full_disc only:4epochs,AdamW1e-4/wd1e-4/clip1,sourceVAL earliest max; no ten-branch architecture search','source_original_weights_frozen':True,'target_original_A256_fixed':True,'target_sampling':'per-expression deterministic seed, same points for original source, prepared zero, all frozen views and TTA; R96 batch-sampling scores retained as historical cohort reference only','TTA':'episodic original102498 params; two masks76001/76002; physical .1P+.1RGB+.01depth; SGD clipped1 step .5; Armijo1e-4,tol1e-10,9scales .5/2**k; original R91 dynamic readout; no new loss','arms':['original_source','prepared_frozen','direct_mean3','direct_feature3','physical_raw','physical_bt','wrong_physical_raw','wrong_physical_bt'],'no_GT_for_target_update_gate_or_selection':True,'target_score_after_all_predictions_sealed':True,'new_hyperparameter_search':False,'FINAL_input':False,'all_negatives_and_unknown_retained':True,'primary_aggregation':'six directed tasks equal weight; each target platform mean of its two incoming sources; per-direction and anonymous-scene rows also retained','inputs_sha256':contracts}
 dump(H/'experiment/DESIGN_LOCK.json',p);(H/'ITERATION_RECORD.md').write_text('# R97 fixed six-direction TTA\n\nClaim: fixed existing R91 task recipe transfers across independently trained source models. No method search. Missing Drone/Quad-specific source-prepared modules are trained only on the matching source with the old losses/architecture; original source100 remains frozen. Waymo preparation reused. All target VAL expressions outside protected frames retained. Matched zero controls, wrong binding and raw/BT negative controls, prediction sealing before scores. Distinct deterministic point sampling from R96 is disclosed and matched original-source scores recomputed. No FINAL.\n')
 g=rd(R/'research/STATE.json');g.update(current_round='R97',run_state='R97_FIXED_SIX_DIRECTION_TTA_REGISTERED',manual_active_state='round97/STATE.json',current_stage='R97_SOURCE_SPECIFIC_PREPARATION');g['directions']['crossplatform'].update(latest_round='R97',active_state='round97/STATE.json',state='R97_FIXED_SIX_DIRECTION_TTA_REGISTERED',active_experiment='fixed_R91_six_transfer',R97_method=p['method']);dump(R/'research/STATE.json',g);print({'targets':targets,'new_source_preparation':sources,'no_target_GT_used':True})
if __name__=='__main__':main()
