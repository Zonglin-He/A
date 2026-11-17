"""Postseal eligible-scope coverage and first-gradient diagnostic; no new task IoUs."""
from common148 import *
import numpy as np,collections

def main():
 assert rd(H/'SCALAR_AUDIT.json')['status']=='PASS';rr=rd(H/'ALL_ROWS_PRIVATE.json');ds=[];rows=[]
 for r in rr:
  if not r['has_B']:continue
  src,tgt,sid=r['source'],r['target'],r['id_PRIVATE'];name=src+'_to_'+tgt;old=np.load(R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz'));new=np.load(B/'evidence_PRIVATE'/name/(sid+'.npz'));ass,S,C=old['assignment'],old['S'],new['C'];oldC=old['C'];y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];good=np.array([y[g].max()>.25 for g in gs]);qzero=r['queries_PRIVATE']['norm_joint_zero'];qb=max(np.flatnonzero(C[ass]),key=lambda i:(float(new['pB'][i]),-int(i)));qo=max(np.flatnonzero(oldC[ass]),key=lambda i:(float(old['pB'][i]),-int(i)));rec={'source':src,'target':tgt,'eligible':bool(new['eligible']),'recoverable_zero_error':bool(y.max()>.25 and y[qzero]<=.25),'good_S':bool((S&good).any()),'old_good_C':bool((oldC&good).any()),'new_good_C':bool((C&good).any()),'old_excludes_good_zero':bool(y[qzero]>.25 and not oldC[ass[qzero]]),'new_excludes_good_zero':bool(y[qzero]>.25 and not C[ass[qzero]]),'old_teacher_good25':bool(y[qo]>.25),'new_teacher_good25':bool(y[qb]>.25),'teacher_query_changed':int(qb)!=int(qo),'same_C':bool(np.array_equal(C,oldC))}
  if not bool(new['compatible_parent']):
   aa=np.load(R/'crossplatform/round145/method'/name/'states_PRIVATE'/(sid+'.npz'));bb=np.load(B/'method'/name/'states_PRIVATE'/(sid+'.npz'));a=aa['norm_TTA_gradient1'].astype(float);b=bb['mass_TTA_gradient1'].astype(float);na,nb=np.linalg.norm(a),np.linalg.norm(b);rec['first_gradient_cosine']=float(a@b/(na*nb)) if na>0 and nb>0 else None;rec['gradient_norm_ratio']=float(nb/na) if na>0 else None
  rows.append(rec)
 for src,tgt in sorted({(x['source'],x['target']) for x in rows}):
  ss=[x for x in rows if (x['source'],x['target'])==(src,tgt) and x['eligible']];ee=[x for x in ss if x['recoverable_zero_error']];gg=[x['first_gradient_cosine'] for x in ss if x.get('first_gradient_cosine') is not None];ds.append({'source':src,'target':tgt,'eligible_n':len(ss),'recoverable_zero_errors_eligible':len(ee),'recoverable_good_in_S':sum(x['good_S'] for x in ee),'recoverable_old_good_C':sum(x['old_good_C'] for x in ee),'recoverable_new_good_C':sum(x['new_good_C'] for x in ee),'old_excluded_good_zero':sum(x['old_excludes_good_zero'] for x in ss),'new_excluded_good_zero':sum(x['new_excludes_good_zero'] for x in ss),'old_teacher_good25':sum(x['old_teacher_good25'] for x in ss),'new_teacher_good25':sum(x['new_teacher_good25'] for x in ss),'teacher_query_changed':sum(x['teacher_query_changed'] for x in ss),'changed_C_first_gradient_n':len(gg),'mean_first_gradient_cosine':float(np.mean(gg)) if gg else None,'min_first_gradient_cosine':min(gg) if gg else None})
 out={'status':'POSTSEAL_DESCRIPTIVE_ONLY','target_models_or_GTgeometry_not_rerun':True,'rows':ds,'new_outputs_or_runtime_filters':False,'same_eligible_denominators_for_old_and_new_C':True,'teacher_good25_boxproxy_not_instance_identity':True};dump(H/'ELIGIBLE_COVERAGE_POSTSEAL.json',out);O=H/'publication/public_export_attempt001';dump(O/'ELIGIBLE_COVERAGE_POSTSEAL.json',out);p=H/'REPORT.md';p.write_text(p.read_text()+'\n\n## Matched eligible-only scope and gradient diagnostic\n\n'+json.dumps(out,indent=2)+'\n\nThese useexistingsealedcandidateIoUscalars onlyafterpredictions/scalarsaudited. Allrowsbelowsharethe764eligibleinputs, avoidingmixingeligiblecoveragewithall1798/noBerrorcounts. FirstgradientcosinescompareactualR145/R148initialgradients withidenticalreader/input/sourceinit/budget; onlyC/conditionalmasschanged. Cosineisdescriptive, notGTalignment oradeploymentgate. TeacherIoUthresholdsareboxqualityproxies, notverifiedidentity.\n');(O/'REPORT.md').write_text(p.read_text());q=O/'own_code/research/round148/context148.py';q.write_text(Path(__file__).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));tokens=set()
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  for ob in [e['row'],e['A_sensor']]+([e['pair_PRIVATE']['B']] if e['has_B'] else []):
   for k in ['id','caption','scene','imagepath','pointpath','metapath']:
    v=ob.get(k)
    if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json'):
  for k in ['id','caption','scene','imagepath','pointpath','metapath']:
   v=e['entry']['row'].get(k)
   if isinstance(v,str) and len(v)>=12:tokens.add(v)
 for p in O.rglob('*'):
  if p.is_file():assert 'LOCAL_HOME/' not in p.read_text() and not any(t in p.read_text() for t in tokens),p
 dump(H/'POSTSEAL_EXPORT_AUDIT.json',{'status':'PASS','private_literals_scanned':len(tokens),'all_public_files_rescanned':True});dump(O/'POSTSEAL_EXPORT_AUDIT.json',rd(H/'POSTSEAL_EXPORT_AUDIT.json'));dump(O/'MANIFEST.json',{'prefix':'studies/3eed_source_calibrated_cumulative_candidate_mass_TTA_20261118','files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']});print(out,flush=True)
if __name__=='__main__':main()
