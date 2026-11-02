"""R50 metadata/source-array inventory only; no models, GT, media or FINAL IO."""
from pathlib import Path
import collections,datetime,hashlib,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
EXP=ROOT/'research/round50/experiment'
OUT=HERE/'source_metadata_inventory_attempt001'
INPUTS={
 'full_manifest':'research/round8/experiment/full5912_gpu_inputs.json',
 'old336':'research/round24/input/ALL336_PRIVATE.json',
 'R8seal':'crossplatform/round8/full5912_attempt001/SEALED.json',
 'R49seal':'crossplatform/round49/stage336_attempt001/SEALED.json',
 'R12index':'research/round12/data_audit/FRAME_INDEX_PRIVATE.json',
 'A64index':'research/round17/input/A_FRAME_INDEX_PRIVATE.json',
 'P64rows':'research/round26/input/P64_PRIVATE.json',
 'P64index':'research/round26/input/P_FRAME_INDEX_PRIVATE.json',
 'Q64rows':'research/round27/input/Q64_PRIVATE.json',
 'Q64index':'research/round27/input/Q_FRAME_INDEX_PRIVATE.json',
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def triple(r):return tuple(r[k] for k in ('imagepath','pointpath','metapath'))
def comparison(a,b):
 a=np.asarray(a);b=np.asarray(b);shape=a.shape==b.shape;dtype=a.dtype==b.dtype
 return {'shape_equal':shape,'dtype_equal':dtype,'array_equal':bool(shape and np.array_equal(a,b)),
         'bitexact':bool(shape and dtype and a.tobytes()==b.tobytes()),
         'different_elements':int(np.count_nonzero(a!=b)) if shape else None,
         'max_abs_difference':float(np.max(np.abs(a.astype(float)-b.astype(float)))) if shape and a.size else None}
def main():
 start=time.perf_counter();assert not OUT.exists();auth=read(EXP/'ROOT_INDEPENDENT_METADATA_INVENTORY_AUTHORIZATION.json')
 assert auth['state']=='ROOT_AUTHORIZED_R50_INDEPENDENT_METADATA_AND_SOURCE_EXACT_INVENTORY_NO_GT'
 assert auth['program_sha256']==sha(__file__)
 assert set(auth['inputs_sha256'])==set(INPUTS.values())
 for rel,h in auth['inputs_sha256'].items():assert 'round46' not in rel and 'FINAL' not in rel and sha(ROOT/rel)==h
 OUT.mkdir(parents=True);full=read(ROOT/INPUTS['full_manifest']);old=read(ROOT/INPUTS['old336']);assert len(full)==5912 and len(old)==336
 fi={r['id']:r for r in full};assert len(fi)==5912 and len({triple(r) for r in full})==5912
 assert collections.Counter(r['platform'] for r in full)=={'drone':2984,'quad':2928}
 s8=read(ROOT/INPUTS['R8seal']);s49=read(ROOT/INPUTS['R49seal']);r8=ROOT/'crossplatform/round8/full5912_attempt001';r49=ROOT/'crossplatform/round49/stage336_attempt001'
 assert s8['n_samples']==5912 and s8['manifest_sha256']==sha(ROOT/INPUTS['full_manifest'])
 assert len(list(r8.glob('*.npz')))==5912
 records=[];source_files={}
 for i,r in enumerate(old):
  rid=r['id'];rr=fi[rid];assert all(rr[k]==r[k] for k in ('caption','platform','scene','imagepath','pointpath','metapath','seed'))
  p8=r8/(rid+'.npz');p49=r49/'samples_PRIVATE'/(rid+'.npz');assert sha(p8)==s8['files'][p8.name] and sha(p49)==s49['files'][str(p49.relative_to(r49))]
  source_files[str(p8.relative_to(ROOT))]=sha(p8);source_files[str(p49.relative_to(ROOT))]=sha(p49)
  with np.load(p8,allow_pickle=False) as x,np.load(p49,allow_pickle=False) as y:
   b8=x['source_boxes_raw6'].copy();p8s=x['source_text_scores256'].copy();b49=y['source_boxes'].copy();p49s=y['source_scores'].copy()
  assert b8.shape==b49.shape==(256,6) and p8s.shape==p49s.shape==(256,) and all(np.isfinite(x).all() for x in (b8,p8s,b49,p49s))
  records.append({'row':f'D{i+1:04d}','id_PRIVATE':rid,'platform':r['platform'],'stage':r['stage'],'boxes':comparison(b8,b49),'scores':comparison(p8s,p49s)})
 caches=[('R12',read(ROOT/INPUTS['R12index']),ROOT/'crossplatform/round12/masks501_attempt001'),('A64',read(ROOT/INPUTS['A64index']),ROOT/'crossplatform/round15/masks64_attempt001')]
 for tag in ('P64','Q64'):
  rr=read(ROOT/INPUTS[tag+'rows']);idx=read(ROOT/INPUTS[tag+'index']);route={r['frame_alias']:dict(idx[r['frame_alias']],**{k:r[k] for k in ('imagepath','pointpath','metapath')}) for r in rr};assert len(route)==64
  caches.append((tag,route,ROOT/('crossplatform/round26/masksP64_attempt001' if tag=='P64' else 'crossplatform/round27/masksQ64_attempt001')))
 cachetriples={};cache_bindings={};cachecounts={}
 for tag,idx,folder in caches:
  ss=read(folder/'SEALED.json');cache_bindings[str((folder/'SEALED.json').relative_to(ROOT))]=sha(folder/'SEALED.json');cachecounts[tag]=len(idx)
  for key,fr in idx.items():
   p=folder/(key+'.npz');assert sha(p)==ss['files'][p.name];cache_bindings[str(p.relative_to(ROOT))]=sha(p)
   t=triple(fr);cachetriples.setdefault(t,[]).append(tag)
 covered=[r for r in full if triple(r) in cachetriples]
 per={}
 for pl in ('drone','quad'):
  rr=[r for r in full if r['platform']==pl];pr=[r for r in records if r['platform']==pl]
  per[pl]={'n':len(rr),'scene_clusters':len({r['scene'] for r in rr}),'covered_by_four_known_mask_camera_caches':sum(triple(r) in cachetriples for r in rr),'old336_source_compared':len(pr),'boxes_bitexact_n':sum(r['boxes']['bitexact'] for r in pr),'scores_bitexact_n':sum(r['scores']['bitexact'] for r in pr)}
 both=sum(r['boxes']['bitexact'] and r['scores']['bitexact'] for r in records)
 result={'state':'COMPLETED_R50_INDEPENDENT_SOURCE_AND_METADATA_INVENTORY_NO_GT','n_full':5912,'unique_ids':len(fi),'unique_pathtriples':5912,'source_compare_n':336,'both_boxes_scores_bitexact_n':both,'R8_source_reuse_exact_gate_PASS':both==336,'source_arrays_not_relabelled_or_tolerance_remapped':True,'full_model_recapture_required_for_R49canonical_if_gate_fails':both!=336,'per_platform':per,'known_mask_cache_frame_counts':cachecounts,'four_known_mask_camera_cache_unique_triplets':len(cachetriples),'full_rows_covered_by_four_known_mask_camera_caches':len(covered),'full_rows_outside_these_four_caches':5912-len(covered),'not_all_repository_caches_exhaustively_claimed':True,'R8_source_NPZ_count':5912,'no_query_token_feature_reuse_assumed_from_R8':True,'no_source_argmax_or_new_readout_prediction_computed':True,'no_GT_media_rawXYZ_models_or_protected_FINAL_reads':True,'program_sha256':sha(__file__),'seconds':time.perf_counter()-start}
 dump(OUT/'ROWS_PRIVATE.json',records);dump(OUT/'INPUT_AND_SOURCE_BINDINGS_PRIVATE.json',dict(auth['inputs_sha256'],**source_files,**cache_bindings));dump(OUT/'SUMMARY.json',result);dump(OUT/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()},'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print(json.dumps(result))
if __name__=='__main__':main()
