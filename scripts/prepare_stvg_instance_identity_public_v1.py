"""Explicit anonymous scalar/source export of the eight-source identity/update study."""
import sys,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_identity_common_v1 import BASE,OLD,read,write,sha,verify
OUT=ROOT/'results/stvg_instance_identity/2026-10-14'
def alias(cid):return ('HC' if cid.startswith('hc2')else'V')+f'{int(cid[-3:])+1:02d}'
def main():
 verify();OUT.mkdir(parents=True,exist_ok=True);paths=[]
 def save(name,d):
  p=OUT/name;write(p,d);paths.append(p)
 rows=read(BASE/'POSTSEAL_ROWS.json')['rows']
 public=[dict(case=alias(r['case_id']),dataset=r['dataset'],arm=r['arm'],metrics=r['metrics'],status=r['status'],support_counts={k:v for k,v in r['support'].items()if k.endswith('_count')or k=='source_grid_positions'},official_independent_max_abs_error=r['metric_audit']['official_independent_max_abs_error'])for r in rows]
 save('all_results.json',dict(status='complete',historically_exposed_development=True,GT_read=True,rows=public))
 save('summary.json',read(BASE/'POSTSEAL_SUMMARY.json'))
 diagnostic=read(BASE/'STEP_DIAGNOSTIC_READBACK.json')
 save('saved_step_diagnostic.json',{k:diagnostic[k]for k in ['status','counts','cases','limitations','readout_semantics','GT_read']})
 import torch
 references=read(BASE/'REFERENCE_LOCK.json')['records'];evidence=[];updates=[]
 for r in references:
  cid=r['case_id'];g=next(x for x in read(BASE/'TEACHER_GEOMETRY_ONLY_READOUT.json')['records']if x['case_id']==cid)
  evidence.append(dict(case=alias(cid),dataset=r['dataset'],when_status=r['when']['parsed_status'],who_syntactic=r['who']['id_syntax_valid'],who_inventory=r['who']['id_in_mask_inventory'],who_qa_present=r['who']['nonempty_mask_in_qa_interval'],identity_correctness='unknown',strict_anchor_count=len(r['strict_original']['anchors']),raw_anchor_count=len(r['raw_decoupled']['anchors']),supported_anchor_count=len(r['presence_supported']['anchors']),presence_weights=[a['reliability']['presence_fraction']for a in r['presence_supported']['anchors']],geometry_mask_counts={k:sum(g[k]['teacher_mask_present'])for k in ['strict_original','raw_decoupled']},grid_count=len(g['frame_ids_physical'])))
  p=BASE/'student'/(cid+'.pt');d=torch.load(p,map_location='cpu',weights_only=False)
  updates.append(dict(case=alias(cid),dataset=r['dataset'],receipts=d['receipts'],rounds={k:f['rounds']for k,f in d['fits'].items()if k in ['loss_weighted_iou','validated_loss_weighted_iou']},actual_fit_count=d['actual_fit_count'],accepted_optimizer_steps=d['optimizer_steps'],proposed_optimizer_steps=d['proposed_optimizer_steps'],final_model_redecodes=d['actual_final_redecode_calls'],source_capture_seconds=d['source_capture_seconds'],config=d['configuration']))
 save('identity_support.json',dict(status='complete',GT_read=False,records=evidence))
 save('actual_update_steps_cost.json',dict(status='complete',GT_read=False,records=updates,cost_scope='new source capture and matched Student updates only; prior detector/tracker/R1 cost not zero and not rerun'))
 save('execution_counts.json',read(BASE/'EXECUTION_COUNTS.json'))
 sources=set()
 for rel,h in read(BASE/'STUDENT_RUNTIME.json')['pins'].items():
  assert sha(ROOT/rel)==h
  if rel.startswith(('scripts/','vg_tta/','methods/','protocols/')) and Path(rel).suffix in ['.py','.md'] :sources.add(rel)
 for pattern in ['*stvg_instance*.py']:
  sources.update(str(p.relative_to(ROOT))for p in (ROOT/'scripts').glob(pattern))
 sources.update(['protocols/stvg_instance_identity_update_v1.md','protocols/stvg_instance_identity_P1_update_lock_v1.md'])
 save('source_bindings.json',dict(status='pass',sources={p:sha(ROOT/p)for p in sorted(sources)},student_runtime_sha256=sha(BASE/'STUDENT_RUNTIME.json'),prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),previous_public_commit='6f75661953d25385450aac3c5de5fb7acfeefc82'))
 for name in ['ROOT_ACTUAL_UPDATE_READBACK.json','ROOT_RAW_IOU_READBACK.json','ROOT_INDEPENDENT_GEOMETRY_READBACK.json','ROOT_POSTSEAL_READBACK.json','TECHNICAL_DECISION.json','ROOT_VISUAL_REVIEW.json']:
  value=read(BASE/name)
  if name=='ROOT_ACTUAL_UPDATE_READBACK.json':
   def clean(x):
    if isinstance(x,dict):return {('case'if k=='case_id'else k):(alias(v)if k=='case_id'else clean(v))for k,v in x.items()if k not in ['observation_positions','physical_frame_ids']}
    if isinstance(x,list):return [clean(v)for v in x]
    return x
   value=clean(value)
  save(name,value)
 # Full scalars only; coordinates, frame identities, private queries/media and raw model payloads are excluded.
 for p in paths:
  s=p.read_text()
  assert all(x not in s for x in ['/home/wwww/','query_text','raw_output','source_case_path','physical_frame_ids','boxes_xyxy','teacher_boxes','video_path']),p
 paths +=[ROOT/p for p in sorted(sources)]
 paths +=[ROOT/'docs/STVG_INSTANCE_IDENTITY_UPDATE_REPORT.md',BASE/'figures/instance_identity_update.pdf',BASE/'figures/instance_identity_update-1.png',BASE/'figures/instance_identity_update-2.png']
 stage=BASE/'public_stage';stage.mkdir(exist_ok=False);manifest=[]
 for p in paths:
  rel=str(p.relative_to(ROOT));dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  manifest.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),encoding='base64'if p.suffix in ['.pdf','.png']else'utf-8'))
 write(BASE/'PUBLIC_MANIFEST.json',manifest)
 write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(status='pass',row_count=len(public),summary_count=len(read(BASE/'POSTSEAL_SUMMARY.json')['summary']),contrast_count=len(read(BASE/'POSTSEAL_SUMMARY.json')['contrasts']),source_count=len(sources),files=len(manifest),bytes=sum(x['bytes']for x in manifest),all_positive_negative_neutral_unknown_preserved=True,private_coordinates_queries_media_excluded=True))
 print(json.dumps(read(BASE/'PUBLIC_SCALAR_AUDIT.json')))
if __name__=='__main__':main()
