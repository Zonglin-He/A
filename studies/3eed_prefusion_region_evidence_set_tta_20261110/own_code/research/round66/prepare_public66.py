"""Pin an explicit dual-version saved-scalar/code-only public package."""
from pathlib import Path
import argparse,json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round66';E=H/'experiment'
def rd(p):return json.loads(Path(p).read_text())
def sh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n')
a=argparse.ArgumentParser();a.add_argument('stage',choices=['export','audit']);a.add_argument('--attempt',type=int,default=1);args=a.parse_args();suf='' if args.attempt==1 else '_attempt'+str(args.attempt).zfill(3)
if args.stage=='export':
 p={'prefix':'studies/3eed_prefusion_region_evidence_set_tta_20261110','no_network_models_GT_or_media':True,'scalar_runs':[{'label':'REGION_V1','score_root':'research/round66/score_dev336_attempt001','independent_audit':'research/round66/independent_audit_attempt001/AUDIT_RESULT.json','saved_scalar_audit':'research/round66/independent_scalar_audit_attempt001/AUDIT_RESULT.json'},{'label':'MATCHED_V2','score_root':'research/round66/score_matched336_attempt001','independent_audit':'research/round66/independent_matched_audit_attempt001/AUDIT_RESULT.json','saved_scalar_audit':'research/round66/scalar_matched_audit_attempt001/AUDIT_RESULT.json'}],'private_registry':'research/round50/final_inputs_attempt001/PRIVATE_PRIVACY_TOKEN_REGISTRY.json','publication_base_snapshot':'research/round66/publication/BASE_SNAPSHOT_PRIVATE.json','report':'research/round66/REPORT.md','decision':'research/round66/DECISION_FINAL.json','public_fixed_configuration':'research/round66/PUBLIC_FIXED_CONFIGURATION.json','normalized_costs':'research/round66/NORMALIZED_COSTS.json','normalized_history':'research/round66/NORMALIZED_HISTORY.json','own_code_allowlist':[x.relative_to(R).as_posix() for x in sorted(H.glob('*.py'))]+['tools/manual_r66_guard.py'],'auxiliary_saved_scalar_reports':[],'auxiliary_text_reports':[{'source':'research/round66/ITERATION_RECORD.md','public_path':'protocol/ITERATION_RECORD.md'}],'inputs_sha256':{}}
 for suffix,folder in [('', 'REGION_V1'),('_QUERY','MATCHED_V2')]:
  for name in ['ALL_METHOD_ROWS','ALL_SCENE_ROWS','ALL_EVIDENCE_PROCESS_ROWS','ALL_FIT_PROCESS_ROWS']:
   p['auxiliary_saved_scalar_reports'].append({'source':'research/round66/'+name+suffix+'.json','public_path':'results/'+folder+'/'+name+'.csv','table_csv':True})
  p['auxiliary_saved_scalar_reports'].append({'source':'research/round66/POSTSEAL_TASK_FEEDBACK_DIAGNOSTICS'+suffix+'.json','public_path':'results/'+folder+'/POSTSEAL_TASK_FEEDBACK_DIAGNOSTICS.json'})
 for name in ['PAIRED_METHOD_COMPARISONS','ALL_PAIRED_V2_V1_NEGATIVES','PAIRED_ELIGIBILITY_STRATA']:
  p['auxiliary_saved_scalar_reports'].append({'source':'research/round66/'+name+'.json','public_path':'results/'+name+'.csv','table_csv':True})
 p['auxiliary_saved_scalar_reports'].append({'source':'research/round66/POSTRESULT_SUPPORT_SIGN_DIAGNOSTICS.json','public_path':'results/POSTRESULT_SUPPORT_SIGN_DIAGNOSTICS.json'})
 for folder,name in [('independent_evidence_audit_attempt001','V1_EVIDENCE'),('independent_query_evidence_audit_attempt001','V2_EVIDENCE'),('independent_derived_audit_attempt001','DERIVED')]:p['auxiliary_saved_scalar_reports'].append({'source':'research/round66/'+folder+'/AUDIT_RESULT.json','public_path':'audits/'+name+'.json'})
 for f in ['ADOPTION_CRITERIA_LOCK.json','ADOPTION_CRITERIA_QUERY_LOCK_v2.json']:p['auxiliary_saved_scalar_reports'].append({'source':'research/round66/experiment/'+f,'public_path':'protocol/'+f})
 paths=[p[k] for k in ['private_registry','publication_base_snapshot','report','decision','public_fixed_configuration','normalized_costs','normalized_history']]+p['own_code_allowlist']+[x['source'] for x in p['auxiliary_saved_scalar_reports']+p['auxiliary_text_reports']]
 for run in p['scalar_runs']:paths += [run['independent_audit'],run['saved_scalar_audit']]+[x.relative_to(R).as_posix() for x in (R/run['score_root']).glob('*.json')]
 for path in paths:p['inputs_sha256'][path]=sh(R/path)
 path=E/('LOCAL_PUBLIC_EXPORT_PLAN_LOCK'+suf+'.json');put(path,p);put(E/('LOCAL_PUBLIC_EXPORT_AUTH'+suf+'.json'),{'state':'ROOT_AUTHORIZED_R66_LOCAL_SAVED_SCALAR_WHITELIST_EXPORT','program_sha256':sh(H/'publish66.py'),'plan_sha256':sh(path),'private_registry_sha256':sh(R/p['private_registry']),'publication_base_snapshot_sha256':sh(R/p['publication_base_snapshot'])})
else:
 author='research/round66/experiment/LOCAL_PUBLIC_EXPORT_PLAN_LOCK'+suf+'.json';p0=rd(R/author);folder='public_export_attempt'+str(args.attempt).zfill(3);p={'no_GT_models_geometry_media_network_or_original_manifest_read':True,'public_package':'research/round66/publication/'+folder,'manifest':'research/round66/publication/'+folder+'/MANIFEST.json','payload':'research/round66/publication/TREE_PAYLOAD_LOCAL_attempt'+str(args.attempt).zfill(3)+'.json','publication_plan':author,'private_registry':p0['private_registry'],'expected_registry_tokens':29937,'output_directory':'research/round66/independent_public_audit_attempt'+str(args.attempt).zfill(3),'inputs_sha256':dict(p0['inputs_sha256'])}
 for rel in [author,p['manifest'],p['payload']]:p['inputs_sha256'][rel]=sh(R/rel)
 path=E/('INDEPENDENT_PUBLIC_AUDIT_PLAN_LOCK'+suf+'.json');put(path,p);put(E/('INDEPENDENT_PUBLIC_AUDIT_AUTH'+suf+'.json'),{'state':'ROOT_AUTHORIZED_R66_INDEPENDENT_LOCAL_PUBLIC_EXPORT_AUDIT_SAVED_ONLY','program_sha256':sh(H/'audit_public66.py'),'plan_sha256':sh(path),'manifest_sha256':sh(R/p['manifest']),'payload_sha256':sh(R/p['payload']),'private_registry_sha256':sh(R/p['private_registry']),'base_snapshot_sha256':sh(R/p0['publication_base_snapshot'])})
print({'stage':args.stage,'plan':str(path.relative_to(R)),'sha256':sh(path)})
