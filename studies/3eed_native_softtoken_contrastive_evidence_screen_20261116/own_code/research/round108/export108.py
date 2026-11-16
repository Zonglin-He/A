"""Anonymized all-arm scalar export, source code, complete matched controls."""
from pathlib import Path
import json,csv,io,hashlib,argparse
ap=argparse.ArgumentParser();ap.add_argument("--attempt",type=int,default=1);arg=ap.parse_args();tag=str(arg.attempt).zfill(3)
R=Path(__file__).resolve().parents[2];H=R/'research/round108';B=R/'crossplatform/round108';P=H/'publication';O=P/('public_export_attempt'+tag);S=H/'scores_attempt001';PREFIX='studies/3eed_native_softtoken_contrastive_evidence_screen_20261116';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rr):
 if not rr:return
 p.parent.mkdir(exist_ok=True,parents=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;O.mkdir(exist_ok=False,parents=True);scmap={};fitn=0
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));sc=sorted({r['scene_PRIVATE'] for r in rr});scmap[name]=sc;tasks=[];fits=[]
  for i,r in enumerate(rr):
   alias='ROW'+str(i+1).zfill(6);st=r['status'];a={'row':alias,'source':r['source'],'target':r['target'],'scene_cluster':'CLUSTER'+str(sc.index(r['scene_PRIVATE'])+1).zfill(3),'protected_alias_fallback':st['protected_alias_fallback'],'updated_output_enabled':False,'seconds':st['seconds']}
   for k,v in r['arms'].items():
    for metric,val in v.items():
     if metric!='query_PRIVATE':a[k+'__'+metric]=val
   tasks.append(a)
   for label,v in r['fits'].items():fits.append({'row':alias,'source':r['source'],'target':r['target'],'fit':label,**{k:x for k,x in v.items() if k!='fit_view_labels'}})
  table(O/'tasks'/(name+'.csv'),tasks);table(O/'fit_endpoints'/(name+'.csv'),fits);fitn+=len(fits)
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(name+'.csv'),rd(S/(name+'.json')))
 scenes=[]
 for x in rd(S/'ALL_SCENE_ROWS_PRIVATE.json'):
  name=x['source']+'_to_'+x['target'];sc=x['scene_PRIVATE'];scenes.append({('scene_cluster' if k=='scene_PRIVATE' else k):('CLUSTER'+str(scmap[name].index(sc)+1).zfill(3) if k=='scene_PRIVATE' else v) for k,v in x.items()})
 table(O/'results/ALL_SCENE_ROWS.csv',scenes)
 for f in ['REPORT.md','LITERATURE.md','ENGINEERING.json','NATIVE_AUDIT.json','SCALAR_AUDIT.json','COMPLEMENTARITY_DIAGNOSTIC.json','CONCENTRATION_SUMMARY.json','DECISION.json']:(O/f).write_bytes((H/f).read_bytes())
 config=rd(H/'experiment/DESIGN_LOCK.json');config['source_head_authority']=config.pop('weights');dump(O/'FIXED_CONFIGURATION.json',config);dump(O/'COUNTS.json',{'directed_inputs':768,'unique_target_expressions':384,'arms':29,'readouts':22272,'fit_endpoint_rows':fitn,'all_negatives_unknowns_fallbacks_retained':True,'raw_media_captions_IDs_coordinates_features_weights_excluded':True,'FINAL_access':False})
 files={R/p for p in rd(H/'experiment/CODE_LOCK.json')['files']};files.update(H.glob('*.py'))
 for f in sorted(files):
  p=O/'own_code'/f.relative_to(R);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 fs=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in fs]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'prefix':PREFIX,'own_code_files':[str(f.relative_to(R)) for f in sorted(files)]});print('EXPORT108',len(fs)+1)
if __name__=='__main__':main()
