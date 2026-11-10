"""Independent derivation of totals, paired-policy process and adoption criterion."""
from common65 import *
H=ROOT/'research/round65';p=read(H/'experiment/EXPERIMENT_SPEC_LOCK.json');tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');method=read(H/'ALL_METHOD_ROWS.json');scene=read(H/'ALL_SCENE_ROWS.json');process=read(H/'ALL_SAMPLING_PROCESS_ROWS.json');transport=read(H/'ALL_TRANSPORT_PROCESS_ROWS.json');agg=read(H/'POLICY_PROCESS_SUMMARY.json');rows=read(ROOT/p['manifest']);err=0.
def close(a,b):
 global err
 err=max(err,abs(float(a)-float(b)));assert abs(float(a)-float(b))<1e-11
assert len(method)==11 and len(scene)==55 and len(process)==2016 and len(transport)==1344
for r in method:
 a=r['method'];x=np.array([t['IoU'][a] for t in tasks]);b=np.array([t['IoU']['old_R59'] for t in tasks]);close(r['balanced_mIoU'],x.mean()*100);close(r['Acc25'],(x>.25).mean()*100);close(r['Acc50'],(x>.5).mean()*100);close(r['gain_vs_old_R59_pp'],(x-b).mean()*100);assert r['improved_vs_old_R59']==int((x-b>1e-10).sum()) and r['worsened_vs_old_R59']==int((x-b< -1e-10).sum()) and r['query_changes_vs_old_R59']==sum(t['queries'][a]!=t['queries']['old_R59'] for t in tasks)
 for pl in ['drone','quad']:close(r[pl.capitalize()+'_mIoU'],np.mean([t['IoU'][a] for t in tasks if t['platform']==pl])*100)
for r in scene:
 pl=r['platform'];s=sorted({t['scene_PRIVATE'] for t in tasks if t['platform']==pl})[int(r['scene_alias'][-2:])-1];rr=[t for t in tasks if t['platform']==pl and t['scene_PRIVATE']==s];assert len(rr)==r['n'];close(r['mIoU'],np.mean([t['IoU'][r['method']] for t in rr])*100);close(r['gain_vs_old_R59_pp'],np.mean([t['IoU'][r['method']]-t['IoU']['old_R59'] for t in rr])*100)
for r in process:
 i=int(r['row'][4:])-1;row=rows[i];assert r['platform']==row['platform'];cap=read(ROOT/p['new_capture_root']/'cases_PRIVATE'/(row['id']+'.json'))[r['policy']+'_'+r['view']];pr=read(ROOT/p['prepared_root']/'cases_PRIVATE'/(row['id']+'.json'));assert r['reference_available']==pr['reference_available'] and r['global_fallback_draws']==pr['views'][r['policy']+'_'+r['view']]['global_fallback_draws']
 for k,v in cap.items():assert r[k]==v
for r in transport:
 row=rows[int(r['row'][4:])-1];cap=read(ROOT/p['prediction_root']/'cases_PRIVATE'/(row['id']+'.json'))[r['policy']];assert r['platform']==row['platform'] and r['B_known_mass']==cap['B_known_mass'] and r['C_known_mass']==cap['C_known_mass']
for r in agg:
 rr=[x for x in process if x['platform']==r['platform'] and x['policy']==r['policy']];assert len(rr)==336 and r['inputs']==168 and r['views']==336
 close(r['mean_raw_coverage_percent'],np.mean([x['raw_fraction_covered'] for x in rr])*100);close(r['mean_new_unique_vs_A'],np.mean([x['new_unique_vs_A'] for x in rr]))
 for a,b in [('competition','focus'),('context','context')]:
  close(r['mean_'+a+'_draw_fraction_percent'],np.mean([x[b+'_draws']/16384 for x in rr])*100);close(r['mean_'+a+'_seed_fraction_percent'],np.mean([x[b+'_seed_n']/x['seed_n'] for x in rr])*100)
 assert r['empty_pool_global_fallback_draws_total']==sum(x['global_fallback_draws'] for x in rr);assert r['reference_available_inputs']==sum(x['reference_available'] for x in rr)//2;tt=[x for x in transport if x['platform']==r['platform'] and x['policy']==r['policy']];close(r['mean_known_mass'],np.mean([x[v+'_known_mass'] for x in tt for v in 'BC']))
d=read(H/'DECISION.json');m={r['method']:r['balanced_mIoU'] for r in method};passed=all(m['directed_R59']>m[a]+1e-10 for a in ['old_R59','common_R59','random_R59','coverage_R59']);assert d['locked_development_criterion_passed']==passed and d['new_TTA_adopted']==False
for a,gain in d['gains_pp'].items():close(gain,m['directed_R59']-m[a])
out=H/'independent_derived_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R65_ALL_METHOD_SCENE_POINT_SEED_TRANSPORT_AND_DECISION_SCALARS','method_rows':len(method),'scene_rows':len(scene),'point_seed_process_rows':len(process),'transport_process_rows':len(transport),'process_aggregate_rows':len(agg),'max_scalar_error':err,'program_sha256':sha(__file__)};dump(out/'AUDIT_RESULT.json',result);print(result)
