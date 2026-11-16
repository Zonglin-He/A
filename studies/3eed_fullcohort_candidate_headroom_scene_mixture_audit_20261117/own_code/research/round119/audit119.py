"""Separate pure-Python arithmetic audit; no diagnostic helper import."""
from pathlib import Path
import json,hashlib,math
R=Path(__file__).resolve().parents[2];H=R/'research/round119';O=H/'analysis_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rec=rd(H/'DIAGNOSTIC_RECEIPT.json');assert rec['n']==18620 and rec['new_GT_scoring']==0;rows=[];oldn=0
 for f in O.glob('*_ALL_ROWS_PRIVATE.json'):
  rr=rd(f);name=f.stem.replace('_ALL_ROWS_PRIVATE','');old={r['id_PRIVATE']:r for r in rd(R/'research/round100/scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'))};assert len(old)==len(rr)
  for r in rr:
   x=old[r['id_PRIVATE']];assert r['scene_PRIVATE']==x['scene_PRIVATE'] and r['oracle_IoU']==x['arms']['original_source']['oracle_IoU_diagnostic'] and all(v==x['arms'][k]['IoU'] for k,v in r['values'].items());oldn+=1
  rows+=rr
 assert oldn==18620;assert {c:sum(x['cohort']==c for x in rows) for c in ['DEV','EXP','REMAINDER']}==rec['cohort_partition']
 for x in rd(O/'METHOD_ROWS.json')+rd(O/'SCENE_ROWS_PRIVATE.json'):
  rr=[r for r in rows if r['source']==x['source'] and r['target']==x['target'] and (x['cohort']=='FULL' or r['cohort']==x['cohort']) and ('scene_PRIVATE' not in x or r['scene_PRIVATE']==x['scene_PRIVATE'])];n=len(rr);yy=[r['values'][x['arm']] for r in rr];oo=[r['oracle_IoU'] for r in rr];assert n==x['n'] and abs(100*sum(yy)/n-x['mIoU_percent'])<1e-9 and abs(100*sum(oo)/n-x['oracle_mIoU_percent'])<1e-9
  for t in [25,50]:
   hit=sum(y>t/100 for y in yy);covered=sum(y>t/100 for y in oo);rank=sum(y<=t/100 and o>t/100 for y,o in zip(yy,oo));assert n-covered==x[f'no_good_candidate_at{t}'] and rank==x[f'covered_but_not_selected_at{t}'] and n-covered+rank==n-hit and abs(hit/n*100-x[f'Acc{t}_percent'])<1e-9 and abs(covered/n*100-x[f'oracle_Acc{t}_percent'])<1e-9
 for x in rd(O/'PAIRED_ROWS.json'):
  rr=[r for r in rows if r['source']==x['source'] and r['target']==x['target'] and (x['cohort']=='FULL' or r['cohort']==x['cohort'])];dd=[r['values'][x['method']]-r['values'][x['control']] for r in rr];assert len(dd)==x['n'] and abs(100*sum(dd)/len(dd)-x['gain_pp'])<1e-9 and sum(d>1e-12 for d in dd)==x['better'] and sum(d< -1e-12 for d in dd)==x['worse']
  for t in [25,50]:
   rescue=[r for r in rr if r['values'][x['method']]>t/100 and r['values'][x['control']]<=t/100];damage=[r for r in rr if r['values'][x['control']]>t/100 and r['values'][x['method']]<=t/100];assert len(rescue)==x[f'rescue_at{t}'] and len(damage)==x[f'damage_at{t}'] and sum(r['values']['original_source']<=t/100 for r in rescue)==x[f'rescue_of_original_source_failure_at{t}'] and sum(r['values']['original_source']>t/100 for r in rescue)==x[f'restore_original_source_hit_at{t}']
 methods=rd(O/'METHOD_ROWS.json')
 for x in rd(O/'AGGREGATE_ROWS.json'):
  mm=[m for m in methods if m['cohort']==x['cohort'] and m['arm']==x['arm'] and (x['group']=='all_six_equal' or m['target']==x['group'])]
  for k,v in x.items():
   if isinstance(v,(int,float)):assert abs(sum(m[k] for m in mm)/len(mm)-v)<1e-9
 for x in rd(O/'COMPOSITION_ROWS.json'):
  rr=[r for r in rows if r['source']==x['source'] and r['target']==x['target']];cr=[r for r in rr if r['cohort']==x['cohort']];scs={r['scene_PRIVATE'] for r in cr};base=sum(r['values'][x['arm']] for r in rr)/len(rr)*100;sub=sum(r['values'][x['arm']] for r in cr)/len(cr)*100;standard=0
  for sc in scs:
   fr=[r for r in rr if r['scene_PRIVATE']==sc];weight=sum(r['scene_PRIVATE']==sc for r in cr)/len(cr);standard+=weight*sum(r['values'][x['arm']] for r in fr)/len(fr)*100
  assert abs(standard-base-x['scene_mixture_component_pp'])<1e-9 and abs(sub-standard-x['within_scene_component_pp'])<1e-9 and abs(sub-base-x['difference_pp'])<1e-9
 out={'state':'PASS_ALL_18620_EXISTING_SCALARS_ELIGIBILITY_PARTITION_PAIRS_SCENES_COMPOSITION_AND_MACRO','n':oldn,'method_rows':len(methods),'new_GT_scoring':0,'new_model_or_features':0,'FINAL_access':False};(H/'INDEPENDENT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
