"""Independent anonymous-result reaggregation and sealed sparse-output checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
D=ROOT/'artifacts/stvg_error_state32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/cached32'
checks=0
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):
 global checks
 checks+=1
 if a is None or b is None:assert a is b,(a,b)
 else:assert abs(float(a)-float(b))<2e-10,(a,b)
def mean(xs):
 v=[float(x)for x in xs if x is not None];return np.mean(v)if v else None
def paired(v,p):
 v=np.array([x for x in v if x is not None]);same(len(v),p['parents'])
 if not len(v):assert p['mean']is None and p['CI95']is None;return
 same(np.mean(v),p['mean'])
 for k,s in [('positive',np.sum(v>1e-10)),('negative',np.sum(v< -1e-10)),('neutral',np.sum(abs(v)<=1e-10)),('severe_harm_below_minus5pp',np.sum(v< -5))]:same(s,p[k])
 assert np.array_equal(v,np.array(p['per_parent_pp']))
 boot=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)))
 for a,b in zip(np.quantile(v[boot].mean(1),[.025,.975]),p['CI95']):same(a,b)
def delta(r,b,ref,control,ep):
 v=r['values'][b][ref];g=v['GlobalError'];c=v[control]
 if ep in ['v','s']:a,z=g['clocks']['final_MBR'][ep],c['clocks']['final_MBR'][ep]
 else:a,z=g['supports'][ep]['s'],c['supports'][ep]['s']
 return None if a is None or z is None else 100*(a-z)
def group(rows,g):
 for b,refs in g.items():
  for ref,s in refs.items():
   same(len(rows),s['queries']);same(len(rows),s['parents'])
   same(sum(r['reference_counts'][ref]==0 for r in rows),s['no_reference_queries'])
   same(sum(r['reference_counts'][ref]for r in rows),s['total_references'])
   for arm,m in s['metrics'].items():
    values=[r['values'][b][ref][arm]for r in rows]
    for c,vals in m['clocks'].items():
     for k,v in vals.items():same(mean([r['clocks'][c][k]for r in values]),v)
    for support,v in m['supports'].items():
     same(mean([r['supports'][support]['s']for r in values]),v['s'])
     same(sum(r['supports'][support]['s']is not None for r in values),v['evaluable_parents'])
     same(sum(r['supports'][support]['frames']for r in values),v['known_frames'])
    for k in ['anchor_z_RMSE','anchor_IoU','unobserved_center_residual_RMSE','unobserved_log_size_residual_RMSE']:same(mean([r[k]for r in values]),m[k])
   for control,eps in s['global_minus'].items():
    for ep,p in eps.items():paired([delta(r,b,ref,control,ep)for r in rows],p)

def main():
 from scripts.audit_stvg_error_state32_v1 import correction_check,sealed
 complete=read(D/'CPU_COMPLETION.json');assert complete['status']=='complete'and complete['count']==32
 assert sha(P/'RESULTS.json')==complete['result_sha256']
 x=read(P/'RESULTS.json');decision=read(P/'DECISION.json');rows=x['rows']
 assert len(rows)==32 and len({(r['dataset'],r['parent_alias'])for r in rows})==32
 mathematical=0;max_math=0.;contribution_checks=0;noops=0;base_exact=0
 plan=read(D/'PLAN_LOCK.json');lock=read(D/'RUNTIME_LOCK.json')
 for path,h in {**lock['pins'],**lock['registries']}.items():assert sha(ROOT/path)==h
 dbar=read(D/'DINO_FULL_BARRIER.json');obar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
 assert not dbar['GT_read']and obar['DINO_full_barrier_sha256']==sha(D/'DINO_FULL_BARRIER.json')
 assert obar['all_predictions_sealed_before_dense_GT_evaluation']
 assert max((D/p).stat().st_mtime_ns for p in ['PLAN_LOCK.json','RUNTIME_LOCK.json'])<=min((ROOT/f['path']).stat().st_mtime_ns for f in dbar['files'])
 assert max((ROOT/f['path']).stat().st_mtime_ns for f in dbar['files'])<=(D/'DINO_FULL_BARRIER.json').stat().st_mtime_ns
 assert (D/'DINO_FULL_BARRIER.json').stat().st_mtime_ns<=min((D/'oracle_sparse_inputs'/(r['alias']+'.pt')).stat().st_mtime_ns for r in plan['records'])
 assert (D/'GLOBAL_PREDICTION_BARRIER.json').stat().st_mtime_ns<=(D/'CPU_COMPLETION.json').stat().st_mtime_ns
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];assert len(rr)==16;group(rr,x['groups'][ds])
  for key,bins in x['strata'][ds].items():
   assert set(bins)=={r[key]for r in rr}
   for label,g in bins.items():group([r for r in rr if r[key]==label],g)
 for r in rows:
  alias=r['alias'];z=sealed(D/'dino_full'/(alias+'.pt'));o=sealed(D/'oracle'/(alias+'.pt'));sparse=sealed(D/'oracle_sparse_inputs'/(alias+'.pt'))
  old=sealed(ROOT/'artifacts/stvg_dino_rollback32_v1/predictions'/(alias+'.pt'));source=sealed(ROOT/'artifacts/stvg_dino_rollback32_v1/source'/(alias+'.pt'))
  assert np.array_equal(z['bases']['Before'],old['fit']['before'].double().numpy());base_exact+=1
  xy=np.asarray(source['native_boxes_xyxy_normalized'],float)
  assert np.array_equal(z['bases']['Source'],np.column_stack(((xy[:,:2]+xy[:,2:])/2,xy[:,2:]-xy[:,:2])));base_exact+=1
  assert z['planned_positions']==list(old['expert']['positions4'])
  anchors=old['expert']['anchors']['single4'];assert z['reference_positions']==[int(a['position'])for a in anchors]
  assert np.array_equal(z['reference_targets'],np.array([a['box']for a in anchors],float).reshape(-1,4))
  refs=dict(DINO_full=dict(positions=z['reference_positions'],targets=z['reference_targets']),**o['references'])
  assert o['references']['GT_common']['positions']==o['references']['DINO_common']['positions']
  for ref,t in o['references'].items():
   assert t['positions']==sparse['references'][ref]['positions']and np.array_equal(t['targets'],sparse['references'][ref]['targets'])
  for b in ['Before','Source']:
   for ref,t in refs.items():
    c=z['corrections'][b]if ref=='DINO_full'else o['corrections'][ref][b]
    audit=correction_check(z['bases'][b],t['positions'],t['targets'],c);mathematical+=1;max_math=max(max_math,audit['max_error']);noops+=audit['exact_noop']
    vals=r['values'][b][ref];same(len(t['positions']),r['reference_counts'][ref])
    for arm in ['LocalAnchor','GlobalError']:
     for clock in vals['Base']['clocks']:same(vals[arm]['clocks'][clock]['t'],vals['Base']['clocks'][clock]['t'])
     parts=r['contributions'][b][ref][arm]
     same(sum(parts.values()),100*(vals[arm]['clocks']['final_MBR']['v']-vals['Base']['clocks']['final_MBR']['v']));contribution_checks+=1
    for support in ['unobserved_decoder','unobserved_decoder_in_MBR','far_unobserved_decoder']:same(vals['LocalAnchor']['supports'][support]['s'],vals['Base']['supports'][support]['s'])
    same(r['contributions'][b][ref]['LocalAnchor']['unobserved_decoder'],0)
    if not t['positions']:
     for arm in ['LocalAnchor','GlobalError']:assert vals[arm]==vals['Base']
 flags={}
 for ref in ['DINO_full','GT_planned','GT_common','DINO_common']:
  flags[ref]={ds:all(x['groups'][ds]['Before'][ref]['global_minus']['Base'][ep]['mean']is not None and x['groups'][ds]['Before'][ref]['global_minus']['Base'][ep]['mean']>0 for ep in ['v','unobserved_decoder'])for ds in ['hc2','vidstg']}
 assert flags==decision['positive_unobserved_and_v_means']
 assert decision['GT_common_positive_both']and not decision['DINO_common_positive_both']and not decision['DINO_full_positive_both']
 assert not decision['new_models_or_OPD_authorized']
 result=dict(status='pass',count=32,anonymous_aggregate_CI_count_support_checks=checks,
   original_Before_Source_exact_checks=base_exact,all_reference_constant_local_checks=mathematical,
   constant_LS_max_error=max_math,exact_noop_reference_base_pairs=noops,
   full_v_contribution_decomposition_checks=contribution_checks,
   all_unobserved_local_decoder_IoUs_equal_Base=True,all_clocks_tIoU_invariant=True,
   sparse_oracle_outputs_only_no_dense_truth_read=True,source_metadata_read_for_exact_baseline=True,dense_groundtruth_files_read=False,
   common_reference_positions_matched=True,seal_bindings_code_order_and_file_timestamp_order_pass=True,
   source_and_runtime_pins_exact=True,all32_queries_retained=True,unknowns_preserved=True,
   decision_flags=flags,new_OPD_not_authorized=True,result_sha256=sha(P/'RESULTS.json'),script_sha256=sha(__file__))
 target=P/'ROOT_AGGREGATE_AUDIT.json';assert not target.exists();target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
