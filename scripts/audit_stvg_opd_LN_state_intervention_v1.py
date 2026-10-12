"""Independent full state/objective/noise/source/dense/statistics root readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_LN_state_intervention_common_v1 import *
from scripts.score_stvg_opd_LN_state_intervention_v1 import require_seal,assemble,rows_read,sign
def near(a,b):
 if isinstance(a,dict):assert set(a)==set(b);return sum(near(v,b[k]) for k,v in a.items())
 if isinstance(a,list):assert len(a)==len(b);return sum(near(x,y) for x,y in zip(a,b))
 if a is None or isinstance(a,(str,bool,int)):assert a==b,(a,b)
 else:assert abs(a-b)<2e-10*(1+abs(a)),(a,b)
 return 1
def independent_estimate(values):
 v=np.array([v for v in values if v is not None],float);n=len(v)
 if not n:return dict(defined_parents=0,mean=None,ci95=None)
 index=np.random.default_rng(SEED).integers(n,size=(BOOT,n));weights=np.array([(index==k).sum(1) for k in range(n)]).T
 means=weights@v/n;ordered=sorted(means);ci=np.quantile(ordered,[.025,.975])
 return dict(defined_parents=n,mean=math.fsum(v)/n,ci95=ci.tolist())
def independent_summary(rr):
 out=dict(parents=64,fits=192,bootstrap=BOOT,seed=SEED,historically_exposed_panel=True,datasets={})
 for ds in ['hc2','vidstg']:
  rows=[r for r in rr if r['dataset']==ds];assert len(rows)==32;out['datasets'][ds]={}
  for mask in ['own_U','common_U','observed','dense_s','dense_v']:
   arms={};contrasts={}
   for a in ARMS:
    values=[r[a+'_'+mask+'_delta'] for r in rows];defined=[v for v in values if v is not None]
    arms[a]=dict(total_parents=32,unknown_parents=32-len(defined),positive_parents=len([v for v in defined if v>EPS]),negative_parents=len([v for v in defined if v< -EPS]),neutral_parents=len([v for v in defined if abs(v)<=EPS]),harm5pp=len([v for v in defined if v<-.05]),harm20pp=len([v for v in defined if v<-.2]),estimate=independent_estimate(values))
   for a,b in [('Source','Inherited_order1'),('Source','Inherited_order2'),('Inherited_order1','Inherited_order2')]:
    pairs=[(r[a+'_'+mask+'_delta'],r[b+'_'+mask+'_delta']) for r in rows];valid=[(x,y) for x,y in pairs if x is not None and y is not None]
    flips=[(x,y) for x,y in valid if (x>EPS and y< -EPS) or (y>EPS and x< -EPS)];d=[y-x for x,y in valid]
    transitions={}
    for x,y in pairs:
     k=sign(x)+'_'+sign(y);transitions[k]=transitions.get(k,0)+1
    contrasts[a+'_to_'+b]=dict(total_parents=32,defined_pairs=len(valid),unknown_pairs=32-len(valid),transition_matrix=transitions,
     signed_second_minus_first=independent_estimate(d),absolute_difference=independent_estimate([abs(v) for v in d]),sign_flips=len(flips),
     flip_both_at_least1pp=len([1 for x,y in flips if abs(x)>=.01 and abs(y)>=.01]),flip_both_at_least5pp=len([1 for x,y in flips if abs(x)>=.05 and abs(y)>=.05]),
     differences_at_least5pp=len([1 for v in d if abs(v)>=.05]),differences_at_least20pp=len([1 for v in d if abs(v)>=.2]))
   out['datasets'][ds][mask]=dict(arms=arms,contrasts=contrasts)
 return out
def root():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();bar=require_seal()
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old,audit
 activate_old()
 from scripts.run_decota_paper_main_v1 import read_row,unpack_expert
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare,mathematical_readback
 from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
 compares=dict(tensors=0,tensor_coordinates=0,scalar_values=0);count=collections.Counter();d=read(BASE/'DESIGN_LOCK.json')
 for r in d['records']:
  pair=pair_for(r);ss=states(pair);inp=normalized(load(ROOT/r['originals'][0]['input_path']));ex=unpack_expert(inp['expert']);raw=read_row(r['dataset'],r['query_ordinal']);common_noise=None
  for arm in ARMS:
   p=BASE/'formal'/f"{r['index']:05}"/(arm+'_repeat0.pt');h=sha(p);assert h==bar['files'][str(p.relative_to(ROOT))];z=load(p);f=z['fit']
   assert digest(f['initial'])==digest(ss[arm])==r['initial_hashes'][arm] and torch.count_nonzero(f['initial'][Q])==0
   assert z['source_model_sha256']==inp['source_model_state_sha256']==pair['source_model_sha256'] and z['pixel_sha256']==inp['pixel_sha256']==pair['pixel_sha256']
   assert z['frame_ids']==inp['frame_ids']==raw['frame_ids'] and z['Native_interval']==inp['interval'] and z['evidence_sha256']==digest(inp['expert'])
   assert f['config']==r['config'] and z['Adam_reset'] and z['Query_Residual_reset'] and z['no_cross_query_state_writeback']
   assert not f['geometry']['rounds'] or f['geometry']['rounds'][0]['Adam_before']['state']=={}
   assert audit(f,ex)==z['audit']['objective'];m=mathematical_readback(f);assert m==z['audit']['Adam'];count['independent_Adam_scalar_checks']+=m['scalar_checks']
   trace,coords=noise(f,raw['key']);assert trace==z['audit']['noise'];count['sample_coordinates']+=coords
   if common_noise is None:common_noise=trace
   assert trace==common_noise
   if arm=='Source':assert torch.equal(f['before'],inp['native_boxes'])
   else:equal(bare(f),bare(load(ROOT/r['originals'][ARMS.index(arm)-1]['path'])['fit']),compares);count['exact_original_inherited_fits']+=1
   if r['index'] in d['qualification_indices']:
    q=load(BASE/'qualification'/f"{r['index']:05}"/(arm+'_repeat0.pt'));equal(bare(f),bare(q['fit']),compares);equal(f['geometry'],q['fit']['geometry'],compares);count['exact_qualified_formal_fits']+=1
   count['fits']+=1;count['rounds']+=len(f['rounds']);count['empty_fits']+=f['empty'];count['backwards']+=f['gradient_calls'];assert sha(p)==h
  status(BASE/'ROOT_STATUS.json',dict(status='independent_full_source_state_math_noise_readback',done=count['fits'],total=192,CPU_only=True,GT_read=False,time=time.time()))
 assert count['fits']==192 and count['rounds']==4800 and count['exact_original_inherited_fits']==128 and count['exact_qualified_formal_fits']==12
 rr,c,provenance=assemble(True);saved=rows_read(OUT/'ROWS.jsonl.gz');scalars=near(rr,saved);scalars+=near(independent_summary(rr),read(OUT/'SUMMARY.json'))
 # Rebind the inherited results to all original P9 labels after the new global seal.
 old=rows_read(ROOT/'results/stvg_opd_evidence_actionability_audit/2026-10-12/ROWS.jsonl.gz');lookup={(r['parent_id'],r['order'],r['scope']):r for r in old}
 for r,row in zip(d['records'],rr):
  for order in ['order1','order2']:
   prior=lookup[r['parent_id'],order,'Omega0']
   for mask in ['own_U','common_U']:
    scalars+=near(row['Inherited_'+order+'_'+mask+'_delta'],prior[mask+'_delta']);scalars+=near(row['Inherited_'+order+'_'+mask+'_frames'],prior[mask+'_frames'])
 write(BASE/'ROOT_READBACK.json',dict(status='pass',scope='all192 actual complete controlled fits, exact LN transplant/QR0/fresh Adam/source/input/evidence/common noise/128 original full fits/12 qualified formal; independent real-frame/dense/parent bootstrap and all anonymous outcomes',
  counts=dict(count),full_comparisons=compares,independent_scalar_checks=scalars,GT_provenance=provenance,parents=64,fits=192,CPU_only=True,GT_read=True,
  independent_full_decoder_Jacobian=False,time=time.time()))
 s=read(OUT/'SUMMARY.json');decision={}
 for ds in ['hc2','vidstg']:
  cs=s['datasets'][ds]['own_U']['contrasts'];supported=[]
  for key,v in cs.items():
   ci=v['absolute_difference']['ci95'];material=v['flip_both_at_least1pp']>0 or v['differences_at_least5pp']>0
   if ci is not None and ci[0]>0 and material:supported.append(key)
  decision[ds]=dict(fixed_panel_material_LN_state_dependence_supported=bool(supported),supported_contrasts=supported)
 write(BASE/'MECHANISM_DECISION.json',dict(status='actual_predeclared_fixed_panel_intervention_evaluated',datasets=decision,
  scope='causal receiving-LN transplant for these matched historical queries, not held-out generalization or a deployable method',
  gate_implemented=False,unlabelled_predictor_established=False,new_method_frozen=False,production_promoted=False,automatic_further_experiments=False,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json',dict(status='pass',parents=64,fits=192,scalar_checks=scalars,positive_negative_neutral_unknown_complete=True,CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':root()
