"""Dual-cohort saved-scalar whitelist; never models, GT or network."""
from pathlib import Path
import argparse,json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round69';E=H/'experiment'
def rd(p):return json.loads(Path(p).read_text())
def sh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n')
a=argparse.ArgumentParser();a.add_argument('stage',choices=['export','audit']);a.add_argument('--attempt',type=int,default=1);args=a.parse_args();suf='' if args.attempt==1 else '_attempt'+str(args.attempt).zfill(3)
if args.stage=='export':
 p={'prefix':'studies/3eed_fixed_method_expanded_validation_20261111','no_network_models_GT_or_media':True,'scalar_runs':[{'label':co,'score_root':'research/round69/score_'+co.lower()+'_derived001','independent_audit':'research/round69/independent_audit_attempt001/AUDIT_RESULT.json','saved_scalar_audit':'research/round69/independent_scalar_audit_attempt001/AUDIT_RESULT.json'} for co in ['VAL1280','TRAIN1024']],'private_registry':'research/round69/publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json','publication_base_snapshot':'research/round69/publication/BASE_SNAPSHOT_PRIVATE.json','report':'research/round69/REPORT.md','decision':'research/round69/DECISION.json','public_fixed_configuration':'research/round69/PUBLIC_FIXED_CONFIGURATION.json','normalized_costs':'research/round69/NORMALIZED_COSTS.json','normalized_history':'research/round69/NORMALIZED_HISTORY.json','own_code_allowlist':[x.relative_to(R).as_posix() for x in sorted(H.glob('*.py'))]+['tools/manual_r69_guard.py'],'auxiliary_saved_scalar_reports':[],'auxiliary_text_reports':[{'source':'research/round69/ITERATION_RECORD.md','public_path':'protocol/ITERATION_RECORD.md'},{'source':'research/round69/READOUT_AND_COHORT_NOTES.md','public_path':'protocol/READOUT_AND_COHORT_NOTES.md'},{'source':'research/round69/CONCLUSIONS.md','public_path':'CONCLUSIONS.md'}],'inputs_sha256':{}}
 for name in ['ALL_METHOD_ROWS','ALL_SCENE_ROWS','ALL_UPDATE_PROCESS_ROWS','SOURCE_RESTORATION_BREAKDOWN','ALL_FRAME_CLUSTER_ROWS','FRAME_BALANCED_SUMMARY']:
  p['auxiliary_saved_scalar_reports'].append({'source':'research/round69/'+name+'.json','public_path':'results/'+name+'.csv','table_csv':True})
 for name in ['RANDOM_REPEAT_SUMMARY','COHORT_DIAGNOSTICS','DATA_SELECTION_AUDIT','PUBLIC_PROVENANCE','CONCLUSION_NUMERIC_AUDIT']:
  p['auxiliary_saved_scalar_reports'].append({'source':'research/round69/'+name+'.json','public_path':'results/'+name+'.json'})
 for folder,name in [('independent_data_audit_attempt001','DATA'),('independent_gpu_audit_attempt001','GPU_SUBSET'),('independent_audit_attempt001','ALL_INPUT_CPU'),('independent_scalar_audit_attempt001','GEOMETRY_SCALARS'),('independent_derived_audit_attempt001','DERIVED')]:
  p['auxiliary_saved_scalar_reports'].append({'source':'research/round69/'+folder+'/AUDIT_RESULT.json','public_path':'audits/'+name+'.json'})
 p['auxiliary_saved_scalar_reports'].append({'source':'research/round69/attempts/score_parser_preflight001/PREFLIGHT_RESULT.json','public_path':'audits/SYNTHETIC_NUMERIC_PARSER_PREFLIGHT.json'})
 paths=[p[k] for k in ['private_registry','publication_base_snapshot','report','decision','public_fixed_configuration','normalized_costs','normalized_history']]+p['own_code_allowlist']+[x['source'] for x in p['auxiliary_saved_scalar_reports']+p['auxiliary_text_reports']]
 for run in p['scalar_runs']:paths += [run['independent_audit'],run['saved_scalar_audit']]+[x.relative_to(R).as_posix() for x in (R/run['score_root']).glob('*.json')]
 for path in paths:p['inputs_sha256'][path]=sh(R/path)
 path=E/('LOCAL_PUBLIC_EXPORT_PLAN_LOCK'+suf+'.json');put(path,p);put(E/('LOCAL_PUBLIC_EXPORT_AUTH'+suf+'.json'),{'state':'ROOT_AUTHORIZED_R69_LOCAL_SAVED_SCALAR_WHITELIST_EXPORT','program_sha256':sh(H/'publish69.py'),'plan_sha256':sh(path),'private_registry_sha256':sh(R/p['private_registry']),'publication_base_snapshot_sha256':sh(R/p['publication_base_snapshot'])})
else:
 author='research/round69/experiment/LOCAL_PUBLIC_EXPORT_PLAN_LOCK'+suf+'.json';p0=rd(R/author);folder='public_export_attempt'+str(args.attempt).zfill(3);p={'no_GT_models_geometry_media_network_or_original_manifest_read':True,'public_package':'research/round69/publication/'+folder,'manifest':'research/round69/publication/'+folder+'/MANIFEST.json','payload':'research/round69/publication/TREE_PAYLOAD_LOCAL_attempt'+str(args.attempt).zfill(3)+'.json','publication_plan':author,'private_registry':p0['private_registry'],'expected_registry_tokens':len(rd(R/p0['private_registry'])['tokens']),'output_directory':'research/round69/independent_public_audit_attempt'+str(args.attempt).zfill(3),'inputs_sha256':dict(p0['inputs_sha256'])}
 for rel in [author,p['manifest'],p['payload']]:p['inputs_sha256'][rel]=sh(R/rel)
 path=E/('INDEPENDENT_PUBLIC_AUDIT_PLAN_LOCK'+suf+'.json');put(path,p);put(E/('INDEPENDENT_PUBLIC_AUDIT_AUTH'+suf+'.json'),{'state':'ROOT_AUTHORIZED_R69_INDEPENDENT_LOCAL_PUBLIC_EXPORT_AUDIT_SAVED_ONLY','program_sha256':sh(H/'audit_public69.py'),'plan_sha256':sh(path),'manifest_sha256':sh(R/p['manifest']),'payload_sha256':sh(R/p['payload']),'private_registry_sha256':sh(R/p['private_registry']),'base_snapshot_sha256':sh(R/p0['publication_base_snapshot'])})
print({'stage':args.stage,'sha256':sh(path)})
