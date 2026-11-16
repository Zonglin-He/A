"""Anonymous full-cohort scalar/negative-tail export, no original identifiers."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round119';O=H/'analysis_attempt001';P=H/'publication';D=P/'public_export_attempt002';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(p,rr):
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'INDEPENDENT_AUDIT.json')['n']==18620 and rd(H/'POSTSTRATIFICATION_AUDIT.json')['method_rows']==450;D.mkdir(parents=True,exist_ok=False);mapping={};rowmap={}
 for f in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')):
  name=f.stem.replace('_ALL_ROWS_PRIVATE','');rr=rd(f);sc=sorted({r['scene_PRIVATE'] for r in rr});mapping[name]=sc;tasks=[];rowmap[name]={}
  for i,r in enumerate(rr):
   tag=f'ROW{i+1:06d}';rowmap[name][r['id_PRIVATE']]=tag;x={'row':tag,'source':r['source'],'target':r['target'],'scene_cluster':f"CLUSTER{sc.index(r['scene_PRIVATE'])+1:03d}",'cohort':r['cohort'],'oracle_IoU':r['oracle_IoU'],'unknown_roles':r['unknown_roles'],'visible_seeds':r['visible_seeds']};x.update({k+'__IoU':v for k,v in r['values'].items()});tasks.append(x)
  for j in range(0,len(tasks),1024):table(D/'tasks'/name/(f'part{j//1024+1:02d}.csv'),tasks[j:j+1024])
 for name in ['METHOD_ROWS','PAIRED_ROWS','COMPOSITION_ROWS','AGGREGATE_ROWS','R118_POSTSTRATIFIED_METHOD_ROWS','R118_POSTSTRATIFIED_AGGREGATE_ROWS']:table(D/'results'/(name+'.csv'),rd(O/(name+'.json')))
 for fn in ['SCENE_ROWS_PRIVATE','R118_SCENE_SCALARS_PRIVATE','POSTSTRATIFICATION_WEIGHTS_PRIVATE']:
  allrows=rd(O/(fn+'.json'))
  for name,sc in mapping.items():
   src,_,tgt=name.split('_');rows=[{k:v for k,v in x.items() if k!='scene_PRIVATE'}|{'scene_cluster':f"CLUSTER{sc.index(x['scene_PRIVATE'])+1:03d}"} for x in allrows if x['source']==src and x['target']==tgt]
   for j in range(0,len(rows),1024):table(D/'scenes'/fn.replace('_PRIVATE','')/name/(f'part{j//1024+1:02d}.csv'),rows[j:j+1024])
 cases=[]
 for r in rd(O/'GOOD_FAILURE_CASES_PRIVATE.json'):
  name=r['source']+'_to_'+r['target'];cases.append({k:v for k,v in r.items() if k!='id_PRIVATE'}|{'row':rowmap[name][r['id_PRIVATE']]})
 table(D/'results/GOOD_FAILURE_CASES.csv',cases)
 for n in ['SUMMARY','DIAGNOSTIC_RECEIPT','INDEPENDENT_AUDIT','POSTSTRATIFICATION_RECEIPT','POSTSTRATIFICATION_AUDIT']:dump(D/(n+'.json'),rd(H/(n+'.json')))
 for n in ['DESIGN_LOCK','POSTSTRATIFICATION_EXTENSION_LOCK']:dump(D/(n+'.json'),rd(H/'experiment'/(n+'.json')))
 (D/'REPORT.md').write_text((H/'REPORT.md').read_text());sources=[]
 for p in sorted(list(H.glob('*.py'))+[R/'tools/exploration_guard_round119.py']):
  q=D/'own_code'/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(p.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));sources.append({'source':str(p.relative_to(R)),'public':str(q.relative_to(D))})
 files=sorted(p for p in D.rglob('*') if p.is_file());dump(D/'MANIFEST.json',{'prefix':'studies/3eed_fullcohort_candidate_headroom_scene_mixture_audit_20261117','files':[{'path':str(p.relative_to(D)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'sources':sources});print('EXPORT119_COMPLETE',len(files)+1,flush=True)
if __name__=='__main__':main()
