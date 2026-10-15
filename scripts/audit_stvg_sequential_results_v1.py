"""Complete independent scalar/count audit after finite coordinate and confirmation scoring."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_coordinate_v1';H=B/'confirmation';PUB=ROOT/'results/stvg_weightfree_coordinate/2026-10-15'
def main():
 import torch,numpy as np
 phases=read(B/'DESIGN_LOCK.json')['stages']+read(B/'TEMPORAL_DESIGN_LOCK002.json')['stages'];n=fields=0;official_error=0.;actual=logical=0;temporal_fits=temporal_steps=temporal_zero=0;phase_counts=[];reuse=empty=0;previous_spatial=read(B/'DESIGN_LOCK.json')['initial_configs'];previous_temporal={ds:read(B/'TEMPORAL_DESIGN_LOCK002.json')['initial'] for ds in ['hc2','vidstg']}
 for stage in phases+[dict(name='confirmation')]:
  name=stage['name'];d=B/name;rows=read(d/'POSTSEAL_ROWS.json')['rows'];public=read(PUB/('CONFIRMATION_ALL_ROWS.json'if name=='confirmation'else name+'_ALL_ROWS.json'))['rows'];assert len(public)==len(rows)
  for raw,out in zip(rows,public):
   assert(raw['case_id'],raw['dataset'],raw['arm'],raw['metrics'])==(out['case_id'],out['dataset'],out['arm'],out['metrics']);n+=1
  summary=read(d/'POSTSEAL_SUMMARY.json')['summary']
  for x in summary:
   key=x.get('metric','m_vIoU');vals=[r['metrics'][key]for r in rows if r['dataset']==x['dataset']and r['arm']==x['arm']and r['metrics'][key]is not None];expected=math.fsum(vals)/len(vals)if vals else None
   assert x['count']==len(vals) and ((expected is None and x['mean']is None)or abs(expected-x['mean'])<1e-12);fields+=1
  stage_actual=stage_logical=0;expected_keys=set();stage_reuse=stage_empty=0
  bar=d/('STUDENT_GLOBAL_BARRIER.json'if(d/'STUDENT_GLOBAL_BARRIER.json').exists()else'GLOBAL_PREDICTION_BARRIER.json')
  for item in read(bar)['files']:
   if not item['path'].endswith('.pt'):continue
   assert sha(ROOT/item['path'])==item['sha256'];z=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False)
   expected_keys.update((z['case_id'],arm)for arm in z['arms'])
   for receipt in z.get('receipts',{}).values():
    stage_reuse+=int(receipt['status'].startswith('exact_'));stage_empty+=int(receipt['status']=='no_reference_no_update')
   if 'actual_fit_count'in item:
    stage_actual+=item['actual_fit_count'];stage_logical+=len(z['fits'])
   else:
    for fit in z['fits'].values():
     temporal_fits+=1;temporal_steps+=fit['optimizer_steps']
     if fit['config']['lr']==0:assert fit['optimizer_steps']==0 and torch.count_nonzero(fit['path'][-1]['theta'])==0;temporal_zero+=1
  keys=[(r['case_id'],r['arm'])for r in rows];assert len(keys)==len(set(keys))and set(keys)==expected_keys
  actual+=stage_actual;logical+=stage_logical;reuse+=stage_reuse;empty+=stage_empty;phase_counts.append(dict(stage=name,new_spatial_fits=stage_actual,logical_spatial_fit_records=stage_logical,exact_reuse_records=stage_reuse,no_reference_records=stage_empty))
  if name!='confirmation':
   cfg=read(d/'STAGE_LOCK.json');sel=read(d/'SELECTION_LOCK.json');parameter=cfg['parameter']
   prev=previous_temporal if name.startswith('T')else previous_spatial
   for ds,recipes in cfg['recipes'].items():
    assert recipes[cfg['incumbent'][ds]]==prev[ds]
    for v,recipe in recipes.items():assert {k:x for k,x in recipe.items()if k!=parameter}=={k:x for k,x in prev[ds].items()if k!=parameter}
    arm=lambda v:('TS_'if name.startswith('T')else'T_')+v
    vals={v:math.fsum(r['metrics']['m_vIoU']for r in rows if r['dataset']==ds and r['arm']==arm(v))/16 for v in recipes};mx=max(vals.values());ties=[v for v in vals if abs(vals[v]-mx)<=1e-12];expected=cfg['incumbent'][ds]if cfg['incumbent'][ds]in ties else min(ties,key=lambda v:recipes[v][parameter]);assert sel['selected'][ds]==expected and sel['configs'][ds]==recipes[expected];fields+=2
   if name.startswith('T'):previous_temporal=sel['configs']
   else:previous_spatial=sel['configs']
 # Independently check every parent-paired delta; no bootstrap by user instruction.
 rows=read(H/'POSTSEAL_ROWS.json')['rows'];lookup={(r['dataset'],r['case_id'],r['arm']):r['metrics']['m_vIoU']for r in rows}
 pairs=read(H/'PARENT_PAIRED_DELTAS.json');assert pairs['bootstrap_executed']is False
 for r in pairs['records']:
  ids=sorted({x['case_id']for x in rows if x['dataset']==r['dataset']});delta={c:(lookup[r['dataset'],c,r['left']]-lookup[r['dataset'],c,r['right']])*100 for c in ids}
  assert delta==r['per_source_delta_pp'] and abs(math.fsum(delta.values())/len(delta)-r['mean_delta_pp'])<1e-10
  assert sum(v < -1e-10 for v in delta.values())==r['negative_sources'] and min(delta.values())==r['worst_delta_pp'];fields+=4
 # Exact counts distinguish reused records from actual GPU optimization.
 write(B/'PUBLIC_SCALAR_AUDIT.json',dict(status='pass',phase_case_arm_rows=n,summary_selection_contrast_fields_recomputed=fields,all_positive_negative_neutral_unknown_preserved=True,exact_public_private_scalar_mapping=True,new_spatial_fits=actual,logical_spatial_fit_records=logical,exact_reuse_records=reuse,no_reference_records=empty,cross_stage_full_config_inheritance=True,unique_complete_case_arm_sets=True,development_shared_head_fits=temporal_fits,development_shared_head_optimizer_steps=temporal_steps,projection_only_zero_lr_fits=temporal_zero,phase_counts=phase_counts,confirmation_used_for_selection=False,bootstrap_executed=False))
 print('scalar audit',n,fields,actual,logical,temporal_fits,temporal_steps,temporal_zero)
if __name__=='__main__':main()
