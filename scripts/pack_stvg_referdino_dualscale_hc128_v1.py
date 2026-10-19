"""Lossless columnar per-step export and explicit public allowlist."""
from pathlib import Path
import sys,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.run_stvg_referdino_dualscale_hc128_v1 import D,P

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def compact(f,obj):f.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')

def main():
 assert read(P/'RESULTS_PUBLIC.json')==read(D/'DEVELOPMENT_SELECTION_INPUT.json')
 for folder,label,chunk in [(P,'development',144),(P/'confirmation','confirmation',80)]:
  assert read(folder/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';source=folder/'STEP_DIAGNOSTICS_PUBLIC.json';full=read(source);assert 'rows'in full;raw=D/'publication_raw_scalars'/label;raw.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,raw/'STEP_DIAGNOSTICS_FULL_ORIGINAL.json');rows=full['rows'];columns=list(rows[0]['steps'][0]);rebuilt=[];shards=[]
  for part,start in enumerate(range(0,len(rows),chunk)):
   subset=rows[start:start+chunk];encoded=[]
   for row in subset:
    row_columns=list(row['steps'][0]);assert all(set(v)==set(row_columns)for v in row['steps']);entry=dict(**{k:v for k,v in row.items()if k!='steps'},steps=[[v[c]for c in row_columns]for v in row['steps']]);
    if row_columns!=columns:entry['step_columns']=row_columns
    encoded.append(entry)
   name=f'STEP_DIAGNOSTICS_PART_{part:02}.json';f=folder/name;assert not f.exists();compact(f,dict(format='lossless columnar step fields',columns=columns,rows=encoded));rr=read(f)
   for row in rr['rows']:rebuilt.append(dict(**{k:v for k,v in row.items()if k not in ['steps','step_columns']},steps=[dict(zip(row.get('step_columns',rr['columns']),v))for v in row['steps']]))
   shards.append(dict(path=name,fits=len(subset),steps=sum(len(v['steps'])for v in subset),bytes=f.stat().st_size,sha256=sha(f)))
  assert rebuilt==rows;h=hashlib.sha256(canonical(rows)).hexdigest();index={k:v for k,v in full.items()if k!='rows'};index.update(format='lossless columnar shards; reconstruct each step by zip(columns,values)',total_fits=len(rows),total_steps=sum(len(v['steps'])for v in rows),logical_rows_canonical_SHA256=h,shards=shards);put(source,index,True);put(folder/'SCALAR_EXPORT_AUDIT.json',dict(status='pass',all_steps_losslessly_reconstructed=True,total_fits=len(rows),total_steps=index['total_steps'],logical_rows_canonical_SHA256=h,columns=columns,shards=shards,no_prediction_configuration_or_metric_change=True))
  whitespace=[]
  for f in folder.glob('*.json'):
   if f.stat().st_size>512000 and not f.name.startswith('STEP_DIAGNOSTICS_PART_'):
    obj=read(f);before=sha(f);logical=hashlib.sha256(canonical(obj)).hexdigest();shutil.copyfile(f,raw/f.name);compact(f,obj);assert read(f)==obj;whitespace.append(dict(path=f.name,original_byte_SHA256=before,export_byte_SHA256=sha(f),canonical_logical_SHA256=logical,change='JSON whitespace only'))
  audit=read(folder/'SCALAR_EXPORT_AUDIT.json');audit['JSON_whitespace_only_exports']=whitespace;put(folder/'SCALAR_EXPORT_AUDIT.json',audit,True)
 sel=read(P/'SELECTION_PUBLIC.json');sel['original_selection_input_byte_SHA256']=sha(D/'DEVELOPMENT_SELECTION_INPUT.json');sel['exported_development_results_byte_SHA256']=sha(P/'RESULTS_PUBLIC.json');sel['canonical_logical_results_SHA256']=hashlib.sha256(canonical(read(P/'RESULTS_PUBLIC.json'))).hexdigest();sel['selection_snapshot_preserved_private_and_metrics_unchanged']=True;put(P/'SELECTION_PUBLIC.json',sel,True)
 dec=read(P/'DECISION_PUBLIC.json');dec['status']='development_and_frozen_confirmation_complete';dec['confirmation']='confirmation/DECISION_PUBLIC.json';dec['structure_iteration_finished']=True;put(P/'DECISION_PUBLIC.json',dec,True)
 (P/'ROUND_REVIEW.md').write_text('''# Final scientific and engineering review\n\nOne predeclared three-policy selection round. The changed package is actual-corrected box coordinates, target-size-dependent near/far mixture exploration, and its exact mixture-density NLL; the old Gaussian KL/control-variate was not reused. This is not an isolated causal contrast of coverage alone. Same frozen teacher/content interface and all3 seeds, equal steps/candidates, fresh initial-proposal Fixed and current-proposal Live.\n\nAll development predictions sealed before new development scoring; one old/new policy selected and hash-locked using the declared development ranking. Reserved32 confirmation acquired same8 teacher observations, not a new expert; all confirmation outputs sealed before GT, no confirmation-based switching. Historical data exposure disclosed. Independent saved-artifact noise, density/gradient/Adam, coordinates/path, evaluator and parent-bootstrap audits; no claim of a second GPU replication. All full signed/unknown parent rows and step scalars retained, positive cases and severe losses equally preserved.\n\nThe final judgement must follow full-vIoU, unobserved actual-node sIoU, Live-vs-Direct/freshFixed/oldLive intervals, seed consistency and cost together. Selection ranking alone, lower NLL, stronger anchor fitting or a greater exploratory range does not establish OPD value. This round finishes structure selection, without automatic additional search or production promotion. See both reports for actual outcomes.\n''')
 files=['protocols/stvg_referdino_dualscale_source_baseline_v1.md','vg_tta/stvg_referdino_dualscale_hc128_v1.py','protocols/stvg_referdino_dualscale_hc128_v1.md']+[str(f.relative_to(ROOT))for f in ROOT.joinpath('scripts').glob('*stvg_referdino_dualscale*_v1.py')if not f.name.startswith('publish_')]+[str(f.relative_to(ROOT))for f in sorted(P.rglob('*'))if f.is_file()];assert len(files)==len(set(files));put(D/'PUBLIC_FILES_FINAL.json',dict(files=files));print('PUBLIC PACK',len(files),'files, actual update counts retained',flush=True)
if __name__=='__main__':main()
