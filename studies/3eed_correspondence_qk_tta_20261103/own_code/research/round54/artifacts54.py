"""Reproducible post-seal scalar tables and entity-mask diagnostic only."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round54';sys.path.insert(0,str(ROOT/'research/round53'))
from support53 import entity_masks
def read(p):return json.loads(Path(p).read_text())
def dump_identical(p,x):
 p=Path(p)
 if p.exists():assert read(p)==x,'A frozen derived value changed'
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 rows=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');arms=list(rows[0]['IoU']);records=[];components={a:[] for a in arms if a not in ['source','zero_step'] and not a.endswith('_1')};losses={a:{k:[] for k in ['RGB_abs','prior','mass','stability','logit_keep','keep']} for a in ['free','qk','free_RGBshuffle','qk_RGBshuffle','qk_noobs']};aliases={p:sorted({r['scene_PRIVATE'] for r in rows if r['platform']==p}) for p in ['drone','quad']}
 for i,row in enumerate(rows):
  sid=row['id_PRIVATE'];main=read(ROOT/'crossplatform/round54/adapted336_attempt001/cases_PRIVATE'/(sid+'.json'))['statistics']['trajectories'];strong=read(ROOT/'crossplatform/round54/free20_control336_attempt001/cases_PRIVATE'/(sid+'.json'))['details']
  for a in losses:
   d=main[a]['final_terms'];losses[a]['RGB_abs'].append(abs(d['RGB']))
   for k in ['prior','mass','stability','logit_keep','keep']:losses[a][k].append(d[k]*(.25 if k=='stability' else .01 if k=='logit_keep' else 1))
  for arm in arms:
   if arm in ['source','zero_step']:continue
   if arm.endswith('_20'):
    r=strong[arm];diag=r['final_diagnostics'];final=diag['loss_terms'];name='free_RGBshuffle' if 'RGBshuffle' in arm else 'free';initial=main[name]['steps'][0]['loss_terms'];loss0=r['initial_loss'];loss1=r['final_loss']
   else:
    name,n=arm.rsplit('_',1);r=main[name];diag=r['steps'][int(n)-1];initial=r['steps'][0]['loss_terms'];final=r['final_terms'] if n=='3' else None;loss0=r['initial_loss'];loss1=r['final_loss'] if n=='3' else None
   records.append({'row':f'DEV336_{i+1:05d}','platform':row['platform'],'scene_alias':('D' if row['platform']=='drone' else 'Q')+'_S'+str(aliases[row['platform']].index(row['scene_PRIVATE'])+1).zfill(2),'arm':arm,'initial_loss':loss0,'final_loss':loss1,'gradient_norm':diag['gradient_norm_before_clipping'],'optimized_state_delta_L2':diag['parameter_delta_L2'],'assignment_KL':diag['assignment_KL'],'source_UNKNOWN_mass':diag['source_unknown_mass'],'updated_UNKNOWN_mass':diag['unknown_mass'],'max_query_mass':diag['max_query_mass'],'new_mass_collapse_flag':diag['new_mass_collapse_flag'],'IoU':row['IoU'][arm],'IoU_delta_pp':(row['IoU'][arm]-row['IoU']['source'])*100,'one_step_loss_terms_not_post_step':arm.endswith('_1')})
   if final is not None:
    coefficients={'RGB':0. if 'noobs' in arm else 1.,'geometry':0. if 'noobs' in arm else .25,'graph':0. if 'noobs' in arm else .1,'prior':1.,'mass':1.,'stability':.25,'keep':1.,'logit_keep':.01};d={k:coefficients[k]*(initial[k]-final[k]) for k in coefficients};d['total_loss_decrease']=loss0-loss1;components[arm].append(d)
 summary={a:{k:{'mean':float(np.mean([d[k] for d in values])),'median':float(np.median([d[k] for d in values]))} for k in values[0]} for a,values in components.items()};dump_identical(H/'analysis_attempt001/WEIGHTED_COMPONENT_LOSS_DECREASE.json',summary);dump_identical(H/'analysis_attempt001/ALL_FIT_DIAGNOSTICS_PUBLIC_SAFE.json',records)
 final_summary={a:{k:{'mean':float(np.mean(v)),'median':float(np.median(v)),'max':float(np.max(v))} for k,v in d.items()} for a,d in losses.items()};dump_identical(H/'analysis_attempt001/FINAL_WEIGHTED_LOSS_TERMS.json',final_summary)
 registered=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');counts={'n':336,'all_positive_tokens_inside_T':0,'any_positive_token_inside_T':0,'positive_tokens_outside_T':0,'positive_tokens_touch_reference':0,'no_positive_overlap_T':0}
 for row in registered:
  sid=row['id'];graph=read(ROOT/'crossplatform/round51/grouped336_attempt001/attributes_PRIVATE'/(sid+'.json'))['text_graph_unchanged'];case=read(ROOT/'crossplatform/round53/latent336_attempt001/cases_PRIVATE'/(sid+'.json'))
  with np.load(ROOT/'crossplatform/round53/latent336_attempt001/samples_PRIVATE'/(sid+'.npz')) as z:mask=z['token_mask'].astype(bool)
  entities,_=entity_masks(graph,row['caption'],np.asarray(case['token_offsets_PRIVATE']));target=entities['T'];ref=np.zeros_like(target)
  for name,m in entities.items():
   if name!='T':ref|=m
  counts['all_positive_tokens_inside_T']+=bool((mask&target).sum()==mask.sum());counts['any_positive_token_inside_T']+=bool((mask&target).any());counts['positive_tokens_outside_T']+=bool((mask&~target).any());counts['positive_tokens_touch_reference']+=bool((mask&ref).any());counts['no_positive_overlap_T']+=not bool((mask&target).any())
 counts.update(postscore_descriptive_only=True,no_model_method_or_predictions_changed=True,no_new_GT_or_model_calls=True,parser_entity_spans_not_semantic_identity_gold=True);dump_identical(H/'analysis_attempt001/READOUT_ENTITY_ALIGNMENT_DIAGNOSTIC.json',counts);print('POSTSEAL_ARTIFACTS54 all derived values reproduced exactly; no new fit/score')
if __name__=='__main__':main()
