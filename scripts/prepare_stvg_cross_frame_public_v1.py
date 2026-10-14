"""Explicit anonymous export, retaining every sealed metric and declared contrast."""
import sys,shutil,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_cross_frame_common_v1 import BASE,read,write,sha,verify
from scripts.report_stvg_cross_frame_v1 import alias
def clean(x):
 if isinstance(x,dict):return {('case' if k=='case_id' else alias(k) if k.startswith(('hc2_00','vidstg_00')) else k):(alias(v)if k=='case_id'else clean(v))for k,v in x.items()}
 if isinstance(x,list):return [clean(v)for v in x]
 return x
def main():
 verify();out=ROOT/'results/stvg_cross_frame/2026-10-15';out.mkdir(parents=True,exist_ok=True);paths=[]
 def save(n,d):
  p=out/n;write(p,clean(d));paths.append(p)
 rows=read(BASE/'POSTSEAL_ROWS.json')['rows']
 public=[dict(case=alias(r['case_id']),dataset=r['dataset'],arm=r['arm'],metrics=r['metrics'],support_counts={k:v for k,v in r['support'].items()if k.endswith('_count')or k=='source_grid_positions'},official_independent_max_abs_error=r['metric_audit']['official_independent_max_abs_error'])for r in rows]
 save('all_results.json',dict(status='complete',historically_exposed_development=True,GT_read=True,rows=public))
 save('summary.json',read(BASE/'POSTSEAL_SUMMARY.json'))
 for name in ['METHOD_SELECTION_LOCK.json','SPEED_VARIANT_LOCK.json','SPEED_VARIANT_DECISION.json','FINAL_METHOD_FREEZE.json','ROOT_ACTUAL_GATE_READBACK.json','ROOT_POSTSEAL_METRIC_READBACK.json','ROOT_VISUAL_REVIEW.json']:
  save(name,read(BASE/name))
 ind=read(BASE/'ROOT_INDEPENDENT_ACTUAL_READBACK.json')
 save('independent_math_readback.json',{k:ind[k]for k in ['status','scope','counts','max_abs_errors','limitations','GT_read','GPU_or_CUDA_used','cuda_initialized','model_loaded']})
 refs=read(BASE/'REFERENCE_LOCK.json')['records']
 save('reference_support.json',dict(status='complete',GT_read=False,records=[dict(case=alias(r['case_id']),dataset=r['dataset'],train_count=len(r['training_anchors']),validation_count=len(r['validation_anchors']))for r in refs]))
 bar=read(BASE/'STUDENT_GLOBAL_BARRIER.json')
 save('execution.json',dict(status='complete',cases=8,new_source_captures=7,new_fits=7,new_final_full_model_redecodes=7,inherited_no_update_cases=1,source_model_loads=bar['source_model_load_count'],incremental_student_worker_wall_seconds=bar['wall_seconds'],new_teacher_calls=0,new_tracking=0,new_GT_oracle_fits=0,component_cost_scope='incremental Student only; prior expert cost is real and excluded',GT_read=False))
 sources={p for p in read(BASE/'STUDENT_RUNTIME.json')['pins']if p.startswith(('scripts/','vg_tta/','protocols/'))and Path(p).suffix in ['.py','.md']}
 sources.update(str(p.relative_to(ROOT))for p in (ROOT/'scripts').glob('*stvg_cross_frame*.py'))
 sources.update(['scripts/run_stvg_temporal_two_candidate_v1.py','protocols/stvg_cross_frame_validation_v1.md'])
 save('source_bindings.json',dict(status='pass',sources={p:sha(ROOT/p)for p in sorted(sources)},runtime_sha256=sha(BASE/'STUDENT_RUNTIME.json'),prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),reference_lock_sha256=sha(BASE/'REFERENCE_LOCK.json'),independent_readback_sha256=sha(BASE/'ROOT_INDEPENDENT_ACTUAL_READBACK.json')))
 for p in paths:
  text=p.read_text()
  assert all(s not in text for s in ['/home/wwww/','query_text','raw_output','physical_frame_ids','boxes_xyxy','teacher_boxes','video_path']),p
 assert read(out/'all_results.json')['rows']==public
 assert read(out/'summary.json')==clean(read(BASE/'POSTSEAL_SUMMARY.json'))
 paths +=[ROOT/p for p in sorted(sources)]
 paths +=[ROOT/'docs/STVG_CROSS_FRAME_VALIDATION_REPORT.md',BASE/'figures/cross_frame_validation.pdf',BASE/'figures/cross_frame_validation-1.png',BASE/'figures/cross_frame_validation-2.png']
 stage=BASE/'public_stage';stage.mkdir(exist_ok=False);manifest=[]
 for p in paths:
  rel=str(p.relative_to(ROOT));dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  manifest.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),encoding='base64'if p.suffix in ['.pdf','.png']else'utf-8'))
 write(BASE/'PUBLIC_MANIFEST.json',manifest)
 write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(status='pass',row_count=len(public),summary_count=len(read(BASE/'POSTSEAL_SUMMARY.json')['summary']),contrast_count=len(read(BASE/'POSTSEAL_SUMMARY.json')['contrasts']),source_count=len(sources),files=len(manifest),bytes=sum(x['bytes']for x in manifest),all_positive_negative_neutral_unknown_preserved=True,private_coordinates_queries_media_excluded=True))
 print('Public manifest:',len(manifest),'files')
if __name__=='__main__':main()
