"""Only source-cache exact qualification; no G, masks, predictions, GT or torch."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'research/round44/qualification';OUT=ROOT/'crossplatform/round44/source_exact_gate_attempt001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def compare(a,b):
 shape=a.shape==b.shape;dtype=a.dtype==b.dtype;numeric=bool(np.array_equal(a,b)) if shape else False;byte=shape and dtype and a.tobytes(order='C')==b.tobytes(order='C');finite=bool(np.isfinite(a).all() and np.isfinite(b).all());return {'exact':bool(shape and dtype and numeric and byte and finite),'shape_R24':list(a.shape),'shape_R37':list(b.shape),'dtype_R24':str(a.dtype),'dtype_R37':str(b.dtype),'numeric_equal':numeric,'raw_bytes_equal':bool(byte),'finite':finite,'max_abs_difference_diagnostic_only':float(abs(a.astype(float)-b.astype(float)).max()) if shape and finite else None}
def main():
 p=read(EXP/'CPU_GATE_PLAN_LOCK.json');i=read(EXP/'CPU_GATE_IMPLEMENTATION_LOCK.json');assert i['plan_sha256']==sha(EXP/'CPU_GATE_PLAN_LOCK.json')
 for f,h in dict(p['input_sha256'],**i['program_sha256']).items():assert sha(ROOT/f)==h,f
 rr=read(ROOT/'research/round24/input/D208_PRIVATE.json');assert len(rr)==208 and len({r['id'] for r in rr})==208;old=ROOT/'crossplatform/round24/capture336_attempt001';new=ROOT/'crossplatform/round37/prepared208_attempt001';s0=read(old/'SEALED.json');s1=read(new/'SEALED.json');OUT.mkdir(exist_ok=False);records=[]
 for r in rr:
  p0=old/'samples_PRIVATE'/(r['id']+'.npz');p1=new/'arrays_PRIVATE'/(r['id']+'.npz');assert sha(p0)==s0['files'][str(p0.relative_to(old))] and sha(p1)==s1['files'][str(p1.relative_to(new))]
  # np.load is lazy; do not decode masks/RGB/pointcloud/features or metadata GT.
  with np.load(p0,allow_pickle=False) as a,np.load(p1,allow_pickle=False) as b:
   assert a['boxes'].shape==(256,6) and a['scores'].shape==(256,);assert b['source_boxes'].shape==(256,6) and b['source_scores'].shape==(256,);cb=compare(a['boxes'],b['source_boxes']);cs=compare(a['scores'],b['source_scores'])
  records.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'boxes':cb,'scores':cs,'pass':cb['exact'] and cs['exact'],'R24_npz_sha256':sha(p0),'R37_npz_sha256':sha(p1)})
 dump(OUT/'ALL208_EXACT_COMPARISONS_PRIVATE.json',records);passed=all(r['pass'] for r in records);summary={'state':'PASS_EXACT_SOURCE_ARRAY_COMPATIBILITY_ONLY' if passed else 'STOP_ORIGINAL_GROUP_REUSE_SOURCE_ARRAY_MISMATCH','n':208,'boxes_exact':sum(r['boxes']['exact'] for r in records),'scores_exact':sum(r['scores']['exact'] for r in records),'platform':{pl:{'n':sum(r['platform']==pl for r in records),'passed':sum(r['platform']==pl and r['pass'] for r in records)} for pl in ['drone','quad']},'all_original_input_hashes_valid':True,'can_reuse_source_arrays_for_later_plan':passed,'new_rank_prediction_or_taskscore_or_GT_GPU':False,'mask_semantic_identity_or_G_reliability_verified':False,'no_tolerance_remap_or_source_change':True};dump(OUT/'RESULT.json',summary);files={str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file()};dump(OUT/'SEALED.json',{'state':'R44_NO_GT_SOURCE_COMPATIBILITY_GATE_SEALED','files':files,'plan_sha256':sha(EXP/'CPU_GATE_PLAN_LOCK.json'),'program_lock_sha256':sha(EXP/'CPU_GATE_IMPLEMENTATION_LOCK.json'),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});dump(EXP/'RESULT.json',dict(summary,seal_sha256=sha(OUT/'SEALED.json')));print('R44_SOURCE_GATE',summary)
if __name__=='__main__':main()
