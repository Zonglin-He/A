"""Independent sealed-output, prompt-contract, E and metric aggregation audit."""
import os,sys,time,collections,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_what_referring_repair32_v1 import D,P,PRE,FOCUS,verify,read,put,load,sha,rewrite

def contract():
 def parses(words,role='obj'):
  tok=[dict(id=i+1,text=t,lemma=t.lower(),upos=pos,head=h,deprel=d)for i,(t,pos,h,d)in enumerate(words)]
  return dict(old=dict(phrase=''),context=dict(eligible=False,reason='unresolved_wh_category',tokens=tok))
 # Negation and stranded preposition cannot be dropped or reverse target role.
 x=parses([('what','PRON',7,'obl'),('is','AUX',7,'aux'),('the','DET',4,'det'),('person','NOUN',7,'nsubj'),('not','PART',7,'advmod'),('yet','ADV',7,'advmod'),('standing','VERB',0,'root'),('beside','ADP',1,'case'),('?','PUNCT',7,'punct')]);old=copy.deepcopy(x);y,inf=rewrite(x);assert y['context']['context']=='the object that the person not yet is standing beside.';assert x==old
 # Better-attached auxiliary negation moves with auxiliary, all words retained.
 x=parses([('what','PRON',7,'obj'),('does','AUX',7,'aux'),('not','PART',2,'advmod'),('the','DET',5,'det'),('person','NOUN',7,'nsubj'),('ever','ADV',7,'advmod'),('hold','VERB',0,'root'),('?','PUNCT',7,'punct')]);y,_=rewrite(x);assert 'does not hold' in y['context']['context'] and y['context']['context'][4:10]=='object'
 return dict(source_words_preserved=True,synthetic_negation_and_preposition=True,input_not_mutated=True)

def main():
 import torch,numpy as np
 from methods.decota_final_simplified_v1.observations import pool_gate,uniform_positions,target_tokens
 from vg_tta.stvg_error_state_correction_v1 import encode
 from vg_tta.stvg_dino_supported_correction32_v1 import estimate
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');x=read(P/'RESULTS_PUBLIC.json');c=collections.Counter();cc=contract();conditions=read(D/'CONDITIONS_PRIVATE.json');cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json');prev={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-19/reference_capacity32/RESULTS_PUBLIC.json')['rows']}
 from transformers import AutoProcessor
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 proc=AutoProcessor.from_pretrained(ROOT/EXPERT_SNAPSHOT,local_files_only=True)
 for rec in bar['files']:
  assert sha(ROOT/rec['path'])==rec['sha256'];z=load(ROOT/rec['path']);a=z['alias'];r=next(r for r in p['records']if r['alias']==a);s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));v=next(v for v in x['rows']if v['alias']==a);assert z['interval']==e['interval']and np.array_equal(z['outputs']['E_MBR'],e['E']);c['baseline_time_exact']+=1
  subj=read(ROOT/f'artifacts/decota_paper_experiments_v1/{r["dataset"]}/subjects/{r["query_ordinal"]:05}.json');par,info=rewrite(subj['parses']);assert par==conditions[a]['parses']and info==z['condition_info'];assert z['GT_read']==False;c['condition_rebuilt_without_labels']+=1
  for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
   u,w=v['values']['E_MBR'][k],prev[a]['values']['E_MBR'][k];assert u is None and w is None or u is not None and w is not None and abs(u-w)<1e-12;c['previous_baseline_metric_exact']+=1
  if a in FOCUS:
   ctx=par['context'];tt=proc.tokenizer(ctx['context'],return_offsets_mapping=True);ent=target_tokens(ctx['context'],tt['offset_mapping'],ctx['span']);assert len(tt['input_ids'])<=256 and all(ctx['context'][max(lo,4):min(hi,10)]for i,(lo,hi)in enumerate(tt['offset_mapping'])if i in ent);assert not s['expert']['observations'];ex=z['expert'];prior=torch.load(ROOT/r['prior_input'],map_location='cpu',weights_only=False);assert uniform_positions(s['frame_ids'],prior['indices'],4)==ex['positions4']==e['planned_positions'];pp=[];ss=[]
   for (_,pos),o in ex['observations'].items():
    pr=o['probe'];again=pool_gate({k:w for k,w in pr.items()if k not in {"accepted","reason","margin","unary","admission_winner"}});assert again['reason']==pr['reason']and again['accepted']==pr['accepted'];assert o['receipt']['text']==ctx['context']and o['receipt']['inputs']and o['receipt']['target_span']==[4,10];c['real_forward_receipts_same_slots_and_gate']+=1
    if pr['accepted']:pp.append(encode(np.asarray(pr['boxes'],float))-encode(z['Source'])[pos]);ss.append(np.asarray(pr['target_scores'],float))
   cfg=cal['source_domains']['hc2'];es=estimate(pp,ss,cfg['scale'],cfg['truncation'],1/16);assert np.array_equal(apply(z['Source'],es['delta']),z['outputs']['Repair_E_MBR']);assert es['delta']==z['estimate']['delta']if isinstance(es['delta'],list)else np.array_equal(es['delta'],z['estimate']['delta']);c['E_recomputed_from_new_references']+=1
  else:assert par==subj['parses']and np.array_equal(z['outputs']['E_MBR'],z['outputs']['Repair_E_MBR']);c['unchanged_normal_branch_exact']+=1
 for name,g in x['groups'].items():
  rr=[r for r in x['rows']if r['dataset']==('hc2'if name=='HC16'else'vidstg')]if name!='Affected5'else[r for r in x['rows']if r['changed_condition']]
  for k,su in g['paired'].items():
   dif=np.array([100*(r['values']['Repair_E_MBR'][k]-r['values']['E_MBR'][k])for r in rr if r['values']['E_MBR'][k]is not None and r['values']['Repair_E_MBR'][k]is not None]);assert len(dif)==su['parents']and abs(dif.mean()-su['mean'])<1e-12;ix=np.random.default_rng(20261018).integers(len(dif),size=(10000,len(dif)));assert np.allclose(np.quantile(dif[ix].mean(1),[.025,.975]),su['CI95'],rtol=0,atol=1e-12);c['independent_mean_bootstrap']+=1
 failures=[read(f)for f in D.glob('FAILURE_*.json')];assert all(f['stage']=='score'and f['type']=='FileNotFoundError'and 'GLOBAL_PREDICTION_BARRIER' in f['error']for f in failures);assert c['real_forward_receipts_same_slots_and_gate']==20 and bar['new_DINO']==20;assert not torch.cuda.is_initialized();verify()
 put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=dict(c),synthetic_contract=cc,all_new_GT_after_seal=True,preseal_scoring_attempts_blocked=len(failures),GPU_audit=False,registries_and_original_inputs_unchanged=True,new_condition_key_namespace=True));print('ROOT AUDIT PASS',dict(c),flush=True)
if __name__=='__main__':main()
