"""Independent metadata and source-AST readback. No parser/model execution."""
import os,sys,time,ast,collections,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_dino_coverage_audit32_v1 import D,P,PRE,PAPER,LEGACY,FOCUS,verify,read,put,load,sha

def main():
 import torch,numpy as np
 torch.set_num_threads(4);start=time.perf_counter();p=verify(True);x=read(P/'RESULTS_PUBLIC.json');source=ROOT/'methods/decota_final_simplified_v1/observations.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='observations');loop=next(n for n in fn.body if isinstance(n,ast.For));gate=loop.body[0];assert isinstance(gate,ast.If)and ast.unparse(gate.test)=="not eligible and (not s1['phrase'])"and len(gate.body)==1 and isinstance(gate.body[0],ast.Continue)
 assert any(isinstance(n,ast.Try)for n in loop.body[1:]);assert next(n for n in loop.body if isinstance(n,ast.Try)).lineno>gate.lineno;assert read(PAPER/'RUNTIME_LOCK.json')['pins'][str(source.relative_to(ROOT))]==sha(source);c=collections.Counter();ct=collections.Counter();details=[]
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];row=next(v for v in x['rows']if v['alias']==a);subject=read(PAPER/ds/'subjects'/f'{q:05}.json');cxt=subject['parses']['context'];s1=subject['parses']['old'];original=PAPER/('t1_ours_hc2'if ds=='hc2'else't1_ours_vid')/'inputs/clean'/f'{q:05}.npz'
  with np.load(original,allow_pickle=False)as ar:md=json.loads(ar['metadata'].tobytes())
  origin_ex=md['expert'];n=len(origin_ex['observations']);ac=sum(v['probe']['accepted']for v in origin_ex['observations']);reasons=collections.Counter(v['probe']['reason']for v in origin_ex['observations']);assert(n,ac,n-ac)==(row['original_DINO_forwards'],row['accepted_frames'],row['rejected_frames'])and dict(reasons)==row['rejection_reasons'];s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));cond=load(PRE/'predictions'/(a+'.pt'));assert len(s['expert']['observations'])==n and origin_ex['new_DINO']==n
  expected_skip=(cxt['eligible']is False)and s1['phrase']=='';assert row['classification']=='pre_call_target_text_gate_skip'if expected_skip else row['classification'].startswith('executed_')
  if a in FOCUS:
   assert expected_skip and n==0 and origin_ex['actual_observation_positions']==[]and len(origin_ex['positions4'])==4;assert cond['compute']['encoding_frames']==0 and cond['active']is False;assert e['estimate']['selected']==[]and e['estimate']['support_frames']==0 and np.array_equal(s['Source'].double().numpy(),e['E']);assert cxt['tokens'][0]['lemma']=='what';c['focus_pre_call_skip_not_forward_rejection']+=1
  c['origin_receipt_metadata_counts']+=1;ct[ds+'_planned']+=4;ct[ds+'_forwarded']+=n;ct[ds+'_accepted']+=ac;ct[ds+'_rejected']+=n-ac;ct[ds+'_conditional_encode']+=cond['compute']['encoding_frames'];details.append(dict(alias=a,planned=4,called=n,accepted=ac,pre_call_skip=expected_skip))
 for ds,g in x['groups'].items():
  for f,k in [('planned_frames','planned'),('actual_original_DINO_frames','forwarded'),('accepted_frames','accepted'),('rejected_frames','rejected'),('later_conditional_encoding_frames','conditional_encode')]:assert g[f]==ct[ds+'_'+k];c['independent_budget_totals']+=1
 prev=read(ROOT/'results/stvg_error_state/2026-10-19/reference_capacity32/RESULTS_PUBLIC.json');rr=[r for r in prev['rows']if r['alias']in FOCUS];contrib=sum(100*(r['values']['GT_E_known4']['v']-r['values']['E_MBR']['v'])for r in rr)/16;assert abs(contrib-x['groups']['vidstg']['full_Vid16_GT_E_capacity_contribution_from_no_original_DINO_pp'])<1e-12
 assert ct['vidstg_forwarded']==44 and ct['vidstg_rejected']==2 and ct['hc2_forwarded']==64;assert not torch.cuda.is_initialized();verify(True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=dict(c),source_AST_skip_before_model_try=True,origin_source_runtime_pin_exact=True,independent_budget_counts=dict(ct),saved_GT_capacity_readback_exact=True,no_new_GT_or_models_or_GPU=True,CPU_seconds=time.perf_counter()-start));print('independent coverage audit pass',dict(c),dict(ct),flush=True)
if __name__=='__main__':main()
