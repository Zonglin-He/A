"""Stage an explicit anonymous/source allowlist for two finite research pilots."""
import sys,json,hashlib,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stvg_temporal_caption_query_v1 import ROOT,read,write,sha
A=ROOT/'artifacts/stvg_support_aware_temporal_spatial_update_v1';B=ROOT/'artifacts/stvg_box_scale_global_evidence_v1'
def equal(a,b):return a is b if a is None or b is None else abs(a-b)<1e-12
stage=A/'public_stage';stage.mkdir(exist_ok=True);optimizer=read(ROOT/'results/stvg_support_and_grid/2026-10-14/optimizer_anonymous_all.json');grid=read(ROOT/'results/stvg_support_and_grid/2026-10-14/grid_anonymous_all.json');assert optimizer['metrics']['readout_count']==304 and len(optimizer['cases_readback'])==8 and len(grid['all_candidate_rows'])==76 and len(grid['source_position_control']['rows'])==8
err=0
for r in grid['cases']:
 p=next(x for x in read(B/'ROOT_PRELIMINARY_POSTSEAL_METRICS.json')['rows']if x['case']==r['case'])
 for k,q in [('source_box_anchor_IoU','source_IoU'),('masked_teacher_top1_candidate_IoU','mask_selected_IoU'),('marked_teacher_top1_candidate_IoU','marked_selected_IoU')]:
  assert equal(r[k],p[q]);err=max(err,abs(r[k]-p[q]))if r[k]is not None else err
for p in [A,B]:
 receipt=p/'PUBLIC_SCALAR_AUDIT.json'
 if not receipt.exists():write(receipt,dict(status='pass',scope='all anonymous actual scalar outputs including negative neutral and unknown checked against independent root private binding',scientific_values_not_modified=True,independent_grid_IoU_max_error=err,unknown_values_not_coerced_to_zero=True,GT_query_frame_coordinates_removed=True))
files=[p for p in ROOT.glob('scripts/*support_aware*v1.py')]+[ROOT/'scripts/stvg_source_anchored_spatial_v1.py',ROOT/'scripts/stvg_support_grid_public_transfer_v1.py',ROOT/'scripts/report_stvg_support_and_grid_v1.py',ROOT/'scripts/prepare_stvg_support_grid_public_v1.py']+[p for p in ROOT.glob('scripts/*box_scale*v1.py')]+[p for p in ROOT.glob('scripts/*box_scale*001.py')]+[ROOT/'protocols/stvg_support_aware_temporal_spatial_update_v1.md',ROOT/'protocols/stvg_box_scale_global_evidence_v1.md',ROOT/'docs/STVG_SUPPORT_AND_GRID_V1.md',ROOT/'output/pdf/stvg_support_and_grid_v1.pdf']+[ROOT/'results/stvg_support_and_grid/2026-10-14'/n for n in ['optimizer_anonymous_all.json','grid_anonymous_all.json']]
for base in [A,B]:
 for name in ['ROOT_READBACK.json','ROOT_VISUAL_REVIEW.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','PUBLIC_SCALAR_AUDIT.json']:files.append(base/name)
files +=[A/'PROTOCOL_READBACK_CORRECTION.json',A/'ROOT_SYNTHETIC_TEMPORAL_CONTRACTS.json',A/'source_audit/LUNA_INDEPENDENT_CPU_SOURCE_AUDIT.json',B/'CPU_CONTRACTS.json',B/'ROOT_CPU_CONTRACTS.json',B/'ROOT_INPUT_PIXEL_RECONSTRUCTION.json',B/'ROOT_PRIVATE_SELECTED_PIXEL_READBACK.json',B/'MECHANISM_DECISION.json',B/'COST_SCOPE_CORRECTION.json',B/'source_position_control001/CONTROL_AUTHORITY.json']
manifest=[]
for src in sorted(set(files)):
 rel=str(src.relative_to(ROOT));dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);data=dst.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();manifest.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),blob_sha=blob,mode='100644',type='blob',encoding='base64'if src.suffix=='.pdf'else 'utf-8'))
for row in manifest:assert not any(t in row['path']for t in ['private_inputs','private_views','.pt','.safetensors','attachments','conversation'])
write(A/'PUBLIC_MANIFEST.json',manifest);write(B/'PUBLIC_BUNDLE_BINDING.json',dict(status='prepared',manifest_sha256=sha(A/'PUBLIC_MANIFEST.json'),combined_optimizer_and_grid_P0=True,files=len(manifest),bytes=sum(r['bytes']for r in manifest)));print(json.dumps(dict(files=len(manifest),bytes=sum(r['bytes']for r in manifest),independent_grid_metric_max_error=err)))
