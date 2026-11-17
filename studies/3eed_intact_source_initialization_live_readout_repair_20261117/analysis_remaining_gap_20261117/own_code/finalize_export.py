from common import *
import re,csv
def main():
 check();assert rd(H/'REJECTION_VECTOR_AUDIT.json')['status']=='PASS';O=H/'publication/public_export_attempt001'
 # Preserve every already-audited scalar file; add complete posthoc audit code.
 for f in rd(O/'MANIFEST.json')['files']:assert sha(O/f['path'])==f['sha256']
 for p in H.glob('*.py'):
  q=O/'own_code'/p.name;q.write_text(p.read_text().replace('WORKSPACE_ROOT','WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 dump(O/'REJECTION_VECTOR_AUDIT.json',rd(H/'REJECTION_VECTOR_AUDIT.json'));dump(O/'FINAL_ANALYSIS_CODE_LOCK.json',{'posthoc_analysis_not_deployment_code':True,'files':{str(p.relative_to(ROOT)):sha(p) for p in H.glob('*.py')},'source_modules':{n:sha(ROOT/n) for n in ['research/round76/model76.py','research/round90/core90.py','research/round91/core91.py']}})
 p=O/'REPRODUCTION.md';p.write_text(p.read_text()+'Additional independent rejection-vector check: guard.py check_rejections.py verifies all74 rejected cases repeat the exact gradient vectors across all3 steps and retain exact zero delta; finalize_export.py appends its public audit and complete posthoc scripts without altering any scalar file.\n')
 dump(O/'MANIFEST.json',{'prefix':'studies/3eed_intact_source_initialization_live_readout_repair_20261117/analysis_remaining_gap_20261117','files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']})
 tokens=set()
 for e in rd(P/'INPUT_REGISTRY_PRIVATE.json'):
  for k in ['id','caption','scene','imagepath','pointpath','metapath']:
   v=e['row'].get(k)
   if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 files=[p for p in O.rglob('*') if p.is_file()]
 for p in files:
  assert p.suffix in ['.csv','.json','.py','.md'];txt=p.read_text();assert 'LOCAL_HOME/' not in txt and not any(t in txt for t in tokens),p;assert not re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',txt)
 assert sum(len(list(csv.DictReader(p.open()))) for p in (O/'tasks').glob('*.csv'))==1798;dump(H/'PUBLIC_EXPORT_FINAL_AUDIT.json',{'status':'PASS','files':len(files),'bytes':sum(p.stat().st_size for p in files),'private_literals_scanned':len(tokens),'all_existing_scalar_files_unchanged':True,'raw_data_features_weights_media_coordinates_or_IDs_exported':False});print(rd(H/'PUBLIC_EXPORT_FINAL_AUDIT.json'))
if __name__=='__main__':main()
