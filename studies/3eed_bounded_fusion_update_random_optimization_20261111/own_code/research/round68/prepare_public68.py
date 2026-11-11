"""Explicit single-run scalar/own-code whitelist; no raw model artifacts."""
from pathlib import Path
import argparse,json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round68';E=H/'experiment'
def rd(p):return json.loads(Path(p).read_text())
def sh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n')
a=argparse.ArgumentParser();a.add_argument('stage',choices=['export','audit']);a.add_argument('--attempt',type=int,default=1);args=a.parse_args();suf='' if args.attempt==1 else '_attempt'+str(args.attempt).zfill(3)
if args.stage=='export':
 p={'prefix':'studies/3eed_bounded_fusion_update_random_optimization_20261111','no_network_models_GT_or_media':True,'scalar_runs':[{'label':'DEV336','score_root':'research/round68/score_dev336_attempt001','independent_audit':'research/round68/independent_audit_attempt001/AUDIT_RESULT.json','saved_scalar_audit':'research/round68/independent_scalar_audit_attempt001/AUDIT_RESULT.json'}],'private_registry':'research/round50/final_inputs_attempt001/PRIVATE_PRIVACY_TOKEN_REGISTRY.json','publication_base_snapshot':'research/round68/publication/BASE_SNAPSHOT_PRIVATE.json','report':'research/round68/REPORT.md','decision':'research/round68/DECISION.json','public_fixed_configuration':'research/round68/PUBLIC_FIXED_CONFIGURATION.json','normalized_costs':'research/round68/NORMALIZED_COSTS.json','normalized_history':'research/round68/NORMALIZED_HISTORY.json','own_code_allowlist':[x.relative_to(R).as_posix() for x in sorted(H.glob('*.py'))]+['tools/manual_r68_guard.py'],'auxiliary_saved_scalar_reports':[],'auxiliary_text_reports':[{'source':'research/round68/ITERATION_RECORD.md','public_path':'protocol/ITERATION_RECORD.md'}],'inputs_sha256':{}}
 for name in ['ALL_METHOD_ROWS','ALL_SCENE_ROWS','ALL_SCALE_PROCESS_ROWS','RANDOM_SCENE_SUMMARY']:p['auxiliary_saved_scalar_reports'].append({'source':'research/round68/'+name+'.json','public_path':'results/'+name+'.csv','table_csv':True})
 p['auxiliary_saved_scalar_reports'].append({'source':'research/round68/RANDOM_REPEAT_SUMMARY.json','public_path':'results/RANDOM_REPEAT_SUMMARY.json'})
 p['auxiliary_saved_scalar_reports'].append({'source':'research/round68/SELECTED_DEVELOPMENT_CONFIG.json','public_path':'protocol/SELECTED_DEVELOPMENT_CONFIG.json'})
 for folder,name in [('independent_capture_audit_attempt001','CAPTURE'),('independent_derived_audit_attempt001','DERIVED')]:p['auxiliary_saved_scalar_reports'].append({'source':'research/round68/'+folder+'/AUDIT_RESULT.json','public_path':'audits/'+name+'.json'})
 paths=[p[k] for k in ['private_registry','publication_base_snapshot','report','decision','public_fixed_configuration','normalized_costs','normalized_history']]+p['own_code_allowlist']+[x['source'] for x in p['auxiliary_saved_scalar_reports']+p['auxiliary_text_reports']]
 for run in p['scalar_runs']:paths += [run['independent_audit'],run['saved_scalar_audit']]+[x.relative_to(R).as_posix() for x in (R/run['score_root']).glob('*.json')]
 for path in paths:p['inputs_sha256'][path]=sh(R/path)
 path=E/('LOCAL_PUBLIC_EXPORT_PLAN_LOCK'+suf+'.json');put(path,p);put(E/('LOCAL_PUBLIC_EXPORT_AUTH'+suf+'.json'),{'state':'ROOT_AUTHORIZED_R68_LOCAL_SAVED_SCALAR_WHITELIST_EXPORT','program_sha256':sh(H/'publish68.py'),'plan_sha256':sh(path),'private_registry_sha256':sh(R/p['private_registry']),'publication_base_snapshot_sha256':sh(R/p['publication_base_snapshot'])})
else:
 author='research/round68/experiment/LOCAL_PUBLIC_EXPORT_PLAN_LOCK'+suf+'.json';p0=rd(R/author);folder='public_export_attempt'+str(args.attempt).zfill(3);p={'no_GT_models_geometry_media_network_or_original_manifest_read':True,'public_package':'research/round68/publication/'+folder,'manifest':'research/round68/publication/'+folder+'/MANIFEST.json','payload':'research/round68/publication/TREE_PAYLOAD_LOCAL_attempt'+str(args.attempt).zfill(3)+'.json','publication_plan':author,'private_registry':p0['private_registry'],'expected_registry_tokens':29937,'output_directory':'research/round68/independent_public_audit_attempt'+str(args.attempt).zfill(3),'inputs_sha256':dict(p0['inputs_sha256'])}
 for rel in [author,p['manifest'],p['payload']]:p['inputs_sha256'][rel]=sh(R/rel)
 path=E/('INDEPENDENT_PUBLIC_AUDIT_PLAN_LOCK'+suf+'.json');put(path,p);put(E/('INDEPENDENT_PUBLIC_AUDIT_AUTH'+suf+'.json'),{'state':'ROOT_AUTHORIZED_R68_INDEPENDENT_LOCAL_PUBLIC_EXPORT_AUDIT_SAVED_ONLY','program_sha256':sh(H/'audit_public68.py'),'plan_sha256':sh(path),'manifest_sha256':sh(R/p['manifest']),'payload_sha256':sh(R/p['payload']),'private_registry_sha256':sh(R/p['private_registry']),'base_snapshot_sha256':sh(R/p0['publication_base_snapshot'])})
print({'stage':args.stage,'sha256':sh(path)})
