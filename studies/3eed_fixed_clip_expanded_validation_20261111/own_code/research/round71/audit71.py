"""All-input independent score math/baseline/unknown-pattern audit, no labels."""
from common71 import *
from scipy.special import softmax
import time
p=read(H/'experiment/AUDIT_PLAN_LOCK.json');pin(p);out=H/'independent_audit_attempt001';assert not out.exists();out.mkdir();pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');start=time.perf_counter();worst=0.;count=0;validcounts=[];unknown=0;seals={};media=read(ROOT/p['media_hashes']);metadata={x['id']:x for x in read(ROOT/'research/round69/input/SCORING_INDEX_PRIVATE.json')};metadata_checks={}
def verified(root,rel,keys=None):
 key=str(root)
 if key not in seals:seals[key]=read(root/'SEALED.json')
 assert sha(root/rel)==seals[key]['files'][rel];return load(root/rel,keys)
for rr,dev in [(rows(),False),(devrows(),True)]:
 for r in rr:
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';prep=ROOT/('crossplatform/round70/prepared336_attempt001' if dev else p['prepared_root']);clip=ROOT/('crossplatform/round70/clip336_attempt001' if dev else p['clip_root']);baseline=ROOT/('crossplatform/round70/predicted336_attempt001' if dev else p['baseline_root']);a=verified(prep,rel);c=verified(clip,rel);
  if not dev:
   receipt=read(prep/'cases_PRIVATE'/(sid+'.json'));assert receipt['image_sha256']==media[sid]['image_sha256'] and receipt['point_sha256']==media[sid]['point_sha256'];meta=metadata[sid];assert meta['meta_path']==r['metapath'];metadata_checks.setdefault(r['metapath'],sha(r['metapath']));assert metadata_checks[r['metapath']]==meta['metadata_container_sha256']
  b=verified(baseline,rel,['arm_names','scores','source_boxes','source_scores']);z=verified(pred,rel);assert np.array_equal(a['boxes'],z['source_boxes']) and np.array_equal(b['source_boxes'],z['source_boxes']);assert np.array_equal(a['scores'],z['source_scores']) and np.array_equal(b['source_scores'],z['source_scores']);assert np.array_equal(z['queries'],z['scores'].argmax(1));assert np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']]);base=np.log(np.maximum(a['scores'].astype(float),1e-300));orig=dict(zip(b['arm_names'],b['scores']))
  for j,k in enumerate(ARMS[:3]):assert np.array_equal(z['scores'][j],orig[k])
  rproj=a['project_reliability'];valid=np.flatnonzero(rproj>0);assert np.array_equal(valid,np.flatnonzero(a['rect_valid']));assert np.all((rproj>=0)&(rproj<=1));vf=c['visual'];assert np.isfinite(vf).all() and np.allclose(np.linalg.norm(vf[valid],axis=2),1,atol=2e-5);s32=vf.mean(1)@c['subject_text'];s64=np.einsum('ij,j->i',vf.astype(float).mean(1),c['subject_text'].astype(float));g=a['query_group'];assert g.shape==(256,) and set(g)==set(range(len(rproj)));validcounts.append(len(valid));unknown+=len(valid)==0
  for shuffle in [False,True]:
   native=s32.copy();ind=s64.copy()
   if shuffle:
    pp=np.random.default_rng(int(hashlib.sha256(b'R70Avisualbinding').hexdigest()[:16],16)).permutation(len(valid));native[valid]=s32[valid][pp];ind[valid]=s64[valid][pp];assert np.array_equal(np.sort(native[valid]),np.sort(s32[valid]));assert np.array_equal(native[rproj==0],s32[rproj==0])
   avg=np.sum(rproj*native)/rproj.sum() if len(valid) else 0.;avg64=sum(rproj*ind)/sum(rproj) if len(valid) else 0.;d32=rproj*(native-avg)*float(c['logit_scale']);d64=rproj*(ind-avg64)*float(c['logit_scale']);actual=z['scores'][3+int(shuffle)];expected=softmax(base+d32[g]);assert np.array_equal(actual,expected);approx=softmax(base+d64[g]);err=float(np.max(abs(actual-approx)));assert err<=.5*np.max(abs(d32[g]-d64[g]))+1e-12;worst=max(worst,err)
  if dev:assert np.array_equal(z['scores'][3],orig['A_projectonly_joint'])
  count+=1
rec={'state':'PASS_R71_ALL2640_INDEPENDENT_READOUT_AND_MATCHED_VISUAL_CONTROL','n':count,'arm_names':ARMS,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'cached_DEV336_exact_R70_readout':True,'permutation_preserves_valid_mask_weights_and_visual_score_multiset':True,'new_projection_groups_total':int(sum(validcounts[:2304])),'no_visual_evidence_inputs':int(unknown),'max_float64_independent_probability_error':worst,'prediction_seal_sha256':sha(pred/'SEALED.json'),'no_task_label_lookup':True,'original_metadata_container_hashes_verified':len(metadata_checks),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',rec);print(rec)
