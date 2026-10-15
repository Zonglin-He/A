"""Explicit anonymous public export for the finite per-domain coordinate study."""
import sys,json,shutil,hashlib,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_coordinate_v1'
def main():
 mode=sys.argv[1];assert mode in ['interim','progress','final'];out=B/('public_'+mode);out.mkdir(exist_ok=True);items=[]
 def emit(path,data=None,source=None):
  p=out/path;p.parent.mkdir(parents=True,exist_ok=True)
  if source is not None:shutil.copyfile(source,p)
  elif isinstance(data,str):p.write_text(data)
  else:p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
  items.append(dict(path=path,bytes=p.stat().st_size,sha256=sha(p),encoding='base64'if p.suffix=='.png'else'utf-8'))
 prefix='results/stvg_weightfree_coordinate/2026-10-15/'
 for p in sorted((ROOT/'scripts').glob('*.py')):
  if ('stvg_weightfree_coordinate'in p.name or 'stvg_coordinate_'in p.name or p.name in ['diagnose_stvg_weightfree_temporal_v1.py','prepare_stvg_sequential_public_v1.py','audit_stvg_sequential_results_v1.py']):emit('scripts/'+p.name,source=p)
 emit('protocols/stvg_weightfree_coordinate_v1.md',source=ROOT/'protocols/stvg_weightfree_coordinate_v1.md')
 for n in ['NO_BOOTSTRAP_AUTHORITY004.json','DESIGN_LOCK.json','TEMPORAL_DESIGN_LOCK002.json','CAUSE_READBACK.json','CPU_CONTRACTS.json','CPU_ORIGINAL_FIT_EQUIVALENCE.json','ROOT_ORIGINAL_TEMPORAL_ARITHMETIC.json','PREEXECUTION_REVIEW_RESOLUTION003.json']:
  emit(prefix+n,source=B/n)
 # Public cohort information includes counts and opaque root identity only, never original roster fields.
 emit(prefix+'COHORT_SCOPE.json',dict(status='locked',sources=64,sources_per_domain=32,development_per_domain=16,confirmation_per_domain=16,one_query_per_parent=True,parents_disjoint=True,historical_project_exposure=True,prior_scored32_reclassified_as_development=True,roster_sha256=sha(B/'ROSTER_LOCK.json'),confirmation_roster_sha256=sha(B/'CONFIRMATION_ROSTER_LOCK.json'),not_Table1=True))
 done=[];total=0
 for stage in read(B/'DESIGN_LOCK.json')['stages']+read(B/'TEMPORAL_DESIGN_LOCK002.json')['stages']:
  d=B/stage['name']
  if not(d/'SELECTION_LOCK.json').exists():continue
  raw=read(d/'POSTSEAL_ROWS.json')['rows'];rows=[dict(case_id=x['case_id'],dataset=x['dataset'],arm=x['arm'],metrics=x['metrics'])for x in raw]
  emit(prefix+stage['name']+'_ALL_ROWS.json',dict(status='complete',development=True,rows=rows,all_signs_and_nulls_preserved=True))
  emit(prefix+stage['name']+'_SELECTION.json',source=d/'SELECTION_AUDIT.json')
  emit(prefix+stage['name']+'_STAGE_LOCK.json',source=d/'STAGE_LOCK.json')
  if(d/'ROOT_READBACK.json').exists():emit(prefix+stage['name']+'_ROOT.json',source=d/'ROOT_READBACK.json')
  count=sum(x['actual_fit_count']for x in read(d/'STUDENT_GLOBAL_BARRIER.json')['files'])if(d/'STUDENT_GLOBAL_BARRIER.json').exists()else 0;total+=count;done.append(dict(stage=stage['name'],new_spatial_fits=count))
 emit(prefix+'TEMPORAL_DIAGNOSIS_ALL_ROWS.json',source=B/'temporal_diagnosis/POSTSEAL_ROWS.json')
 if mode!='final':
  if (B/'JOINT_METHOD_LOCK.json').exists():emit(prefix+'JOINT_METHOD_LOCK.json',source=B/'JOINT_METHOD_LOCK.json')
  text='# Sequential tuning: interim evidence, expanded 64-source cohort\n\nThis is a running study, not a completed confirmation or Table1 result. Each domain has16 development and16 frozen confirmation parents. The first32 already had measured results and are explicitly development now. All64 are historically exposed project sources.\n\nThe previous method had temporal ranking and update failures, and spatial pseudo-reference improvement did not guarantee tube improvement. Independent NumPy replay of28 original temporal fits matched every saved grid decode and Adam path (504 checks, max theta error about2.08e-13).19/28 final regularized preference objectives exceeded their initial values.\n\nEach coordinate inherits the previous winner separately per domain; all32 predictions seal before development scoring. Parameters are selected within a fixed finite grid, not proven globally optimal. The additional32 are excluded from selection. The original QA observation clock is held fixed; tuned temporal output does not silently move spatial supervision.\n\nCompleted stages:\n\n'
  for x in done:
   text+=f"- {x['stage']}: {x['new_spatial_fits']} new spatial fits.\n"
   for r in read(B/x['stage']/'SELECTION_AUDIT.json')['records']:text+=f"  {r['dataset']}: {r['parameter']}={r['config'][r['parameter']]}, development joint vIoU={100*r['scores'][r['selected']]:.6f}%.\n"
  text+='\nAll positive, negative, neutral and missing values for completed stages are included. No confirmation results are available in this interim report. The current user instruction cancels Bootstrap; no resampling or confidence intervals will be computed. A zero temporal learning-rate winner means final-output Projection-only; the original QA head still provides the fixed spatial observation clock. Later parameters at zero lr are algebraically inactive, not evidence that they never matter. The early own-cgroup initialization failure occurred before GPU inference and was resolved in separate002 supervisor; originals retained. The confirmation input-reader hookup and zero-lr Projection-only accounting were corrected before those stages executed. CURRENT and the old queues remain unchanged.\n'
  emit('docs/STVG_WEIGHTFREE_COORDINATE_INTERIM.md',text)
 else:
  allowed=['SPATIAL_COORDINATES.png','TEMPORAL_COORDINATES.png','CONFIRMATION_COST_COUNTS.json','CONFIRMATION_ALL_ROWS.json','CONFIRMATION_PARENT_PAIRED_DELTAS.json','SPATIAL_DESIGN.json','TEMPORAL_DESIGN.json','JOINT_METHOD_LOCK.json','CAUSE_READBACK.json','TEMPORAL_DIAGNOSIS_ALL_ROWS.json','REPORT.md','CONFIRMATION.png']+[x['name']+suffix for x in read(B/'DESIGN_LOCK.json')['stages']+read(B/'TEMPORAL_DESIGN_LOCK002.json')['stages']for suffix in ['_ALL_ROWS.json','_SELECTION.json']]
  for p in [ROOT/'results/stvg_weightfree_coordinate/2026-10-15'/n for n in allowed]:
   rel=prefix+p.name
   if rel not in {x['path']for x in items}:emit(rel,source=p)
  emit('docs/STVG_WEIGHTFREE_COORDINATE_REPORT.md',source=ROOT/'docs/STVG_WEIGHTFREE_COORDINATE_REPORT.md')
  for p in [B/'ROOT_POLICY_ARITHMETIC_READBACK.json',B/'ROOT_TEMPORAL_ARITHMETIC_READBACK.json',B/'ROOT_VISUAL_REVIEW.json',B/'ROOT_ANONYMOUS_SCALAR_READBACK.json',B/'PUBLIC_SCALAR_AUDIT.json',B/'ROOT_ENGINEERING_RECOVERY_READBACK.json',B/'confirmation/ROOT_READBACK.json',B/'confirmation/ROOT_INPUT_RENDER_READBACK.json']:
   emit(prefix+('CONFIRMATION_'if p.parent.name=='confirmation'else'')+p.name,source=p)
 # Content safety: no raw data assets, credentials or private query texts included.
 def audit_json(v):
  if isinstance(v,dict):
   prohibited={'query_text','caption','raw_output','prompt','input_videos','video_path','weights','boxes','box','bbox','masks','tensor','payload','gt_boxes','coordinates','frame_map'}
   for k,x in v.items():
    assert k.lower() not in prohibited,k
    audit_json(x)
  elif isinstance(v,list):
   for x in v:audit_json(x)
  elif isinstance(v,str):assert not v.startswith(('/home/','/tmp/','file://')),v[:50]
 for item in items:
  p=out/item['path'];assert p.suffix in ['.json','.md','.py','.png']
  if p.suffix=='.py':ast.parse(p.read_text())
  elif p.suffix=='.json':audit_json(read(p))
 assert len({i['path']for i in items})==len(items)
 write(B/('PUBLIC_'+mode.upper()+'_MANIFEST.json'),items)
 write(B/('ROOT_PUBLIC_'+mode.upper()+'_BINDINGS.json'),dict(status='pass',scope=mode,file_count=len(items),bytes=sum(x['bytes']for x in items),completed_stages=done,development_new_spatial_fits=total,confirmation_new_spatial_fits=(sum(x['actual_fit_count']for x in read(B/'confirmation/STUDENT_GLOBAL_BARRIER.json')['files'])if mode=='final'else 0),total_new_spatial_fits=(total+sum(x['actual_fit_count']for x in read(B/'confirmation/STUDENT_GLOBAL_BARRIER.json')['files'])if mode=='final'else total),private_media_queries_gt_coordinates_weights_excluded=True))
 print(mode,len(items),sum(x['bytes']for x in items),done)
if __name__=='__main__':main()
