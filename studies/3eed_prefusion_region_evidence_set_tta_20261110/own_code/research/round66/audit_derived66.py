"""Independent saved-only audit of derived task, feedback and paired summaries.
No producer imports, no model, no new prediction/scoring or selection.
"""
from pathlib import Path
import json, hashlib, math, numbers
import numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round66'
def read(p):return json.loads(Path(p).read_text())
def eq(a,b,p='$'):
 if isinstance(b,dict):
  assert set(a)==set(b),(p,'keys')
  for k in b:eq(a[k],b[k],p+'.'+k)
 elif isinstance(b,list):
  assert len(a)==len(b),(p,'len')
  for i,(x,y) in enumerate(zip(a,b)):eq(x,y,p+str(i))
 elif isinstance(b,numbers.Real) and type(b) is not bool:assert math.isclose(a,b,abs_tol=2e-12,rel_tol=1e-12),(p,a,b)
 else:assert a==b,(p,a,b)
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
alltasks=[];allflags=[];support={};pins={}
for version,suf,taskroot,design in [('v1','','score_dev336_attempt001','EXPERIMENT_SPEC_LOCK.json'),('v2','_QUERY','score_matched336_attempt001','MATCHED_QUERY_DESIGN_LOCK_v2.json')]:
 tt=read(H/taskroot/'ALL_TASK_ROWS_PRIVATE.json');plan=read(H/'experiment'/design);mr=read(H/('ALL_METHOD_ROWS'+suf+'.json'));sr=read(H/('ALL_SCENE_ROWS'+suf+'.json'));gr=read(H/('ALL_EVIDENCE_PROCESS_ROWS'+suf+'.json'));fr=read(H/('ALL_FIT_PROCESS_ROWS'+suf+'.json'));flags=[]
 assert len(tt)==len(fr)==336 and len({t['id_PRIVATE'] for t in tt})==336
 for m in mr:
  arm=m['method'];x=np.array([t['IoU'][arm] for t in tt]);base=np.array([t['IoU']['R59'] for t in tt]);dc=x-base
  for pl,key in [('drone','Drone_mIoU'),('quad','Quad_mIoU')]:eq(m[key],np.mean([t['IoU'][arm] for t in tt if t['platform']==pl])*100)
  for key,value in [('balanced_mIoU',x.mean()*100),('Acc25',(x>.25).mean()*100),('Acc50',(x>.5).mean()*100),('gain_vs_R59_pp',dc.mean()*100),('improved_vs_R59',int((dc>1e-10).sum())),('worsened_vs_R59',int((dc< -1e-10).sum()))]:eq(m[key],value)
 for s in sr:
  pl=s['platform'];sv=sorted({t['scene_PRIVATE'] for t in tt if t['platform']==pl});index=int(s['scene_alias'][-2:])-1;subset=[t for t in tt if t['platform']==pl and t['scene_PRIVATE']==sv[index]];eq(s['n'],len(subset));eq(s['mIoU'],np.mean([t['IoU'][s['method']] for t in subset])*100)
 for i,(t,f) in enumerate(zip(tt,fr)):
  e=npz(R/plan['evidence_root']/'samples_PRIVATE'/(t['id_PRIVATE']+'.npz'));flags.append(bool(e['update_allowed']));eq(f['update_eligible'],flags[-1]);eq(f['postseal_R59_gain_pp'],100*(t['IoU']['R59_tta1']-t['IoU']['R59']));eq(f['postseal_C_gain_pp'],100*(t['IoU']['C_tta1']-t['IoU']['C_source']))
  gg=[g for g in gr if g['row']=='DEV_'+str(i+1).zfill(5)];assert len(gg)==len(e['members'])
  for j,g in enumerate(gg):
   eq(g['positive'],bool(e['positive'][j]));eq(g['negative'],bool(e['negative'][j]));eq(g['valid_both_views'],bool(e['valid'][:,j].all()));eq(g['E_A'],float(e['E_views'][0,j]));eq(g['E_B'],float(e['E_views'][1,j]));eq(g['full_only'],float(e['full_only'][j]))
 criteria=read(H/'experiment'/('ADOPTION_CRITERIA_LOCK.json' if version=='v1' else 'ADOPTION_CRITERIA_QUERY_LOCK_v2.json'));scores={m['method']:m['balanced_mIoU'] for m in mr};dec=read(H/('DECISION'+suf+'.json'))
 for key,arm,ck in [('G_development_gate_passed','R59_E_direct','G_requires_R59_E_direct_exceed'),('TTA_R59_gate_passed','R59_tta1','TTA_requires_R59_tta1_exceed'),('TTA_heldout_C_gate_passed','C_tta1','TTA_also_requires_C_tta1_exceed')]:eq(dec[key],all(scores[arm]>scores[a]+1e-10 for a in criteria[ck]))
 eq(dec['TTA_development_gate_passed'],dec['TTA_R59_gate_passed'] and dec['TTA_heldout_C_gate_passed'])
 positive=[g for g in gr if g['positive']];good=[g for g in positive if g['full_only']>0];support[version]={'positive_E_groups':len(positive),'positive_E_mean_corrected_full_support_positive':len(good),'positive_E_mean_corrected_full_support_nonpositive':len(positive)-len(good),'positive_E_with_full_support_positive_task_good':sum(g['postseal_best_task_IoU']>.25 for g in good),'all_positive_E_task_good':sum(g['postseal_best_task_IoU']>.25 for g in positive),'not_a_filtered_method_or_new_TTA_run':True}
 alltasks.append(tt);allflags.append(flags)
 for name in ['ALL_METHOD_ROWS','ALL_SCENE_ROWS','ALL_EVIDENCE_PROCESS_ROWS','ALL_FIT_PROCESS_ROWS','DECISION']:path=H/(name+suf+'.json');pins[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
a,b=alltasks;f1,f2=allflags;strata=[];pair=[];neg=[]
for arm in a[0]['IoU']:
 diff=np.array([y['IoU'][arm]-x['IoU'][arm] for x,y in zip(a,b)]);pair.append({'method':arm,'v2_minus_v1_pp':float(diff.mean()*100),'improved':int((diff>1e-10).sum()),'worsened':int((diff< -1e-10).sum()),'query_changes':sum(x['queries'][arm]!=y['queries'][arm] for x,y in zip(a,b))})
for i,(x,y) in enumerate(zip(a,b)):
 assert x['id_PRIVATE']==y['id_PRIVATE']
 for arm in a[0]['IoU']:
  d=y['IoU'][arm]-x['IoU'][arm]
  if d< -1e-10:neg.append({'row':'DEV_'+str(i+1).zfill(5),'platform':x['platform'],'method':arm,'v1_IoU':x['IoU'][arm],'v2_IoU':y['IoU'][arm],'v2_minus_v1_pp':d*100})
# Public output order follows registered ARMS, not JSON key ordering.
actual=read(H/'PAIRED_METHOD_COMPARISONS.json');eq({v['method']:v for v in actual},{v['method']:v for v in pair});actual=read(H/'ALL_PAIRED_V2_V1_NEGATIVES.json');eq(sorted(actual,key=lambda x:(x['row'],x['method'])),sorted(neg,key=lambda x:(x['row'],x['method'])))
for x in [False,True]:
 for y in [False,True]:
  ix=[i for i in range(336) if f1[i]==x and f2[i]==y];row={'v1_update_eligible':x,'v2_update_eligible':y,'n':len(ix)}
  for ver,tt in [('v1',a),('v2',b)]:
   for scope,arm,ref in [('R59','R59_tta1','R59'),('C','C_tta1','C_source')]:
    vals=[tt[i]['IoU'][arm]-tt[i]['IoU'][ref] for i in ix];row[ver+'_'+scope+'_within_stratum_gain_pp']=sum(vals)/len(vals)*100 if vals else None;row[ver+'_'+scope+'_full336_contribution_pp']=sum(vals)/336*100
  strata.append(row)
eq(read(H/'PAIRED_ELIGIBILITY_STRATA.json'),strata);support['posthoc_v1_updates_restricted_to_v2_eligible_full336_contribution_R59_pp']=sum(s['v1_R59_full336_contribution_pp'] for s in strata if s['v2_update_eligible']);support['posthoc_v2_updates_full336_gain_R59_pp']=sum(s['v2_R59_full336_contribution_pp'] for s in strata);support['this_is_saved_scalar_stratification_not_method_selection']=True;eq(read(H/'POSTRESULT_SUPPORT_SIGN_DIAGNOSTICS.json'),support)
out=H/'independent_derived_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R66_INDEPENDENT_DERIVED_METHOD_SCENE_GATE_PAIR_AND_SUPPORT_AUDIT','n_per_run':336,'v1_eligible':sum(f1),'v2_eligible':sum(f2),'method_rows':32,'scene_rows':160,'paired_negatives':len(neg),'input_sha256':pins,'new_model_calls':0,'new_predictions':0,'new_task_scoring':0,'posthoc_rule_selected':False};(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='input_sha256'})
