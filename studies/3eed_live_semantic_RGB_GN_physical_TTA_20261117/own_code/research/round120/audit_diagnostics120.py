"""Saved-query/scalar crosschecks for postscore diagnostics; no model or taskGT."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round120';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=rd(H/'experiment/ZERO_NUMERICS_DIAGNOSTIC_LOCK.json');assert sha(H/'zero_numerics120.py')==lock['file_sha256'];assert sha(R/'crossplatform/round120/MATRIX_SEALED.json')==lock['source_matrix_seal_sha256'];rows=[r for f in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)];by={(r['cohort'],r['source'],r['target'],r['id_PRIVATE']):r for r in rows};diag=rd(H/'zero_numerics_attempt001/ALL_ROWS_PRIVATE.json');assert len(diag)==2150;assert len({(r['cohort'],r['source'],r['target'],r['id_PRIVATE']) for r in diag})==2150
 for r in diag:
  x=by[(r['cohort'],r['source'],r['target'],r['id_PRIVATE'])];assert not x['protected_alias_fallback'] and r['zero_CPU_query_PRIVATE']==x['arms']['context_zero_F']['query_PRIVATE'] and r['updated_query_PRIVATE']==x['arms']['depth_trust_live_F']['query_PRIVATE'];assert r['zero_backend_Top1_differs']==(r['zero_CPU_query_PRIVATE']!=r['zero_GPU_query_PRIVATE']);assert r['update_changes_CPU_zero_Top1']==(r['zero_CPU_query_PRIVATE']!=r['updated_query_PRIVATE']);assert r['update_changes_GPU_zero_Top1']==(r['zero_GPU_query_PRIVATE']!=r['updated_query_PRIVATE']);v={a['query_PRIVATE']:a['IoU'] for a in x['arms'].values()}
  for field,k in [('zero_CPU_IoU_reused','zero_CPU_query_PRIVATE'),('zero_GPU_IoU_reused','zero_GPU_query_PRIVATE'),('updated_IoU_reused','updated_query_PRIVATE')]:assert r[field]==v.get(r[k])
 for s in rd(H/'ZERO_NUMERICS_SUMMARY.json')['cohorts']:
  rr=[r for r in diag if r['cohort']==s['cohort']];assert len(rr)==s['n_nonalias'] and sum(r['zero_backend_Top1_differs'] for r in rr)==s['zero_backend_Top1_disagreements'] and sum(r['update_changes_CPU_zero_Top1'] for r in rr)==s['updates_change_CPU_baseline'] and sum(r['update_changes_GPU_zero_Top1'] for r in rr)==s['updates_change_actual_GPU_zero'];assert max(r['zero_CPU_GPU_max_probability_error'] for r in rr)==s['max_zero_CPU_GPU_probability_error'] and max(r['canonical_CPU_replay_error'] for r in rr)==s['max_zero_CPU_replay_error']
 for s in rd(H/'DECISION_EQUIVALENCE_SUMMARY.json')['rows']:
  rr=[r for r in rows if r['cohort']==s['cohort']];assert len(rr)==s['n'] and sum(r['arms'][s['method']]['query_PRIVATE']!=r['arms'][s['control']]['query_PRIVATE'] for r in rr)==s['query_changes'] and sum(abs(r['arms'][s['method']]['IoU']-r['arms'][s['control']]['IoU'])>1e-12 for r in rr)==s['IoU_changes']
 out={'state':'PASS_POSTSCORE_SAVED_QUERY_AND_OLD_SCALAR_DIAGNOSTICS','n':2304,'zero_native_head_checks':2150,'new_GT_scoring':0,'new_models':0,'original_method_unchanged':True,'FINAL_access':False};(H/'DIAGNOSTIC_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
