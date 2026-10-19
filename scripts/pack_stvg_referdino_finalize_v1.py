"""Lossless public scalar shards; raw geometry, features and captions stay private."""
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
D=R/'artifacts/stvg_referdino_finalize_v1';P=R/'results/stvg_error_state/2026-10-19/referdino_final'
def main():
 records=[]
 import shutil
 selection_input=D/'TEMPERATURE_SELECTION_INPUT.json';shutil.copyfile(P/'TEMPERATURE_RESULTS_PUBLIC.json',selection_input);assert sha(selection_input)==read(D/'SELECTION_LOCK.json')['results_sha256']
 for f in sorted(P.glob('*STEP_DIAGNOSTICS_PUBLIC.json')):
  z=read(f);backup=D/'public_original'/f.name;put(backup,z);rows=z['rows'];shards=[]
  for k in range(0,len(rows),48):
   name=f.stem+f'_SHARD_{k//48:03}.json';payload=dict(rows=rows[k:k+48]);dest=P/'steps'/name;put(dest,payload);shards.append(dict(path='steps/'+name,sha256=sha(dest),rows=len(payload['rows'])))
  rebuilt=[r for sh in shards for r in read(P/sh['path'])['rows']];assert rebuilt==rows;put(f,dict(status='lossless_all_steps_sharded',total_fit_rows=len(rows),total_steps=sum(len(r['steps'])for r in rows),shards=shards),True);records.append(dict(path=f.name,source_rows=len(rows),logical_reconstruction_exact=True))
 for f in P.rglob('*.json'):
  z=read(f);tmp=f.with_name(f.name+'.tmp');tmp.write_text(json.dumps(z,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(f)
 for index in P.glob('*STEP_DIAGNOSTICS_PUBLIC.json'):
  z=read(index)
  for rec in z['shards']:rec['sha256']=sha(P/rec['path'])
  put(index,z,True)
 public_selection=read(P/'SELECTION_PUBLIC.json');public_selection.update(original_selection_input_SHA256=sha(selection_input),exported_temperature_results_SHA256=sha(P/'TEMPERATURE_RESULTS_PUBLIC.json'),format_only_change_to_results=True);put(P/'SELECTION_PUBLIC.json',public_selection,True)
 files=[str(f.relative_to(R))for f in (R/'methods/stvg_referdino_opd_v1').glob('*')if f.is_file()]+[str(f.relative_to(R))for f in P.rglob('*')if f.is_file()]+[str(f.relative_to(R))for f in (R/'scripts').glob('*stvg_referdino_final*.py')]+['protocols/stvg_referdino_finalize_v1.md','README.md','REVIEW_START_HERE.md','methods/RESEARCH_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json','methods/paper/STVG_DINO_P7A_ONLINE_V1.json','methods/paper/STVG_T2_B_EPISODIC_V1.json']
 files=sorted(set(files)-{'scripts/publish_stvg_referdino_finalize_v1.py'});put(P/'SCALAR_EXPORT_AUDIT.json',dict(status='pass',records=records,all_scalar_step_rows_reconstruct_exactly=True,raw_candidate_geometry_and_media_not_exported=True));files.append(str((P/'SCALAR_EXPORT_AUDIT.json').relative_to(R)));put(D/'PUBLIC_FILES_FINAL.json',dict(files=files));print('PACKED',len(files),'files',flush=True)
if __name__=='__main__':main()
