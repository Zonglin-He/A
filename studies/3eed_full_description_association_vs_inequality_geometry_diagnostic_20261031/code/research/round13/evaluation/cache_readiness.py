"""R13 schema/hash-only R12 cache reuse audit; no labels/models/geometry scoring."""
from pathlib import Path
import hashlib,json,numpy as np
B=Path(__file__).resolve().parents[3];R=B/'research/round13/evaluation';S=B/'crossplatform/round12/source146_attempt001';G=B/'crossplatform/round12/feedback146_attempt001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
ss=read(S/'SEALED.json');gs=read(G/'SEALED.json');checks=0
for root,s in [(S,ss),(G,gs)]:
 for rel,h in s['files'].items():
  p=(root/rel).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==h;checks+=1
rows=read(S/'manifest_input.json');selected=[];schemas={}
required={'base_xyz':((1,256,3),'float32'),'center_features':((1,288,256),'float32'),'size_features':((1,288,256),'float32'),'boxes_raw6':((1,256,6),'float32'),'text_scores':((1,256),'float32')}
for row in rows:
 sid=row['id'];cp=S/'source_cache'/(sid+'.npz');obs=S/'observations'/(sid+'.npz')
 with np.load(cp,allow_pickle=False) as z:
  for key,(shape,dtype) in required.items():assert z[key].shape==shape and str(z[key].dtype)==dtype;checks+=1
  assert int(z['query_index'])==int(z['text_scores'][0].argmax());checks+=1
  assert str(z['source_checkpoint_sha256'])==ss['source_checkpoint_sha256'];checks+=1
  keys=set(z.files);assert not any('token' in k or 'proj_quer' in k or 'top10' in k for k in keys);checks+=1
  pcsha=z['point_sample_sha256'].copy();ixsha=z['raw_indices_sha256'].copy();box=z['boxes_raw6'][0].copy();q=int(z['query_index'])
 with np.load(obs,allow_pickle=False) as z:
  assert z['sampled_points_model'].shape==(16384,3) and z['raw_indices'].shape==(16384,);assert np.array_equal(pcsha,z['point_sample_sha256']) and np.array_equal(ixsha,z['raw_indices_sha256']);checks+=2
 with np.load(G/(sid+'.npz'),allow_pickle=False) as z:
  names=list(z['arm_names']);assert np.array_equal(box,z['source_boxes_raw6']) and int(z['q_source'])==q;checks+=1
  if any(bool(z['actual_changed_axes'][names.index(a)].any()) for a in ('single_instance_direct','multiview_instance_direct')):selected.append(sid)
assert len(selected)==8
# Reuse is within the same immutable forward, never an identity claim across processes.
out={'state':'PASS_R13_R12_CACHE_SCHEMA_REUSE_READINESS','source_seal_sha256':sha(S/'SEALED.json'),'regression_prediction_seal_sha256':sha(G/'SEALED.json'),'auditor_sha256':sha(Path(__file__)),'audited_source_cases':len(rows),'checks':checks,'derived_changed_union_n':8,'selection_uses_only_sealed_changed_flags_not_task_score':True,'cache_schemas':{k:{'shape':v[0],'dtype':v[1]} for k,v in required.items()},'reusable':['immutable256 original raw6 boxes/scores/q0 for external candidate association','actual source center/size features for later separately authorized head checks','sampled16384 modelXYZ/rawindices only with original text/input/checkpoint hashes','R12 frontend independent masks and original pixels if asset/config/image byte matches'],'not_cached':['fulltoken scores or projected token/query embeddings','native GPU top10 tie order','full raw point cloud or new observation/caption forward'],'rules':['A may choose another slot only in the same saved source pool; association_only output must equal that slot raw6 bitexact.','Original model encoded full caption but cached readout sums firstnoun positions. A full-caption external helper is a new computation, not already present in cache.','Changed grounder text/sampling/numerics requires new source forward; cross-process query slot identity is not transferable.','B fixes q0 and sample/source bindings; geometry requiring whole rawP must separately bind raw file/camera/mask, not relabel cached16k as whole sensor P.','Old8 are output-conditioned exposed regression only. New108 need fresh strictFP32 source caches.'],'no_GT_annotation_scoredrows_IoU_image_pointfile_model_GPU_access':True}
(R/'CACHE_REUSE_READINESS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'checks':checks,'old_union':8,'sha256':sha(R/'CACHE_REUSE_READINESS.json')}))
