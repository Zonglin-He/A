"""UNEXECUTED post-seal full5912 readout audit; no GT or author algorithm import.

Checks every fixed-source three-arm score/selection against independently
reconstructed old-W arithmetic, original relation functions and mask pixels.
Raw XYZ/projection, pretrained networks and task correctness are not rerun.
"""
from pathlib import Path
import ast,collections,hashlib,json,re,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'research/round50/experiment';HERE=Path(__file__).resolve().parent
OUT=HERE/'full5912_audit_attempt001';ARMS=['source','old_G','wideentry_oldW']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy(order='K') for k in z.files}
def exact(a,b,name):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name
SEALS={};VERIFIED=set()
def sealmap(folder):
 folder=Path(folder)
 if folder not in SEALS:SEALS[folder]=read(folder/'SEALED.json')
 return SEALS[folder]
def member(folder,rel):
 folder=Path(folder);p=folder/rel;assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
 if p not in VERIFIED:assert sha(p)==sealmap(folder)['files'][rel];VERIFIED.add(p)
 return p
def original_relation():
 ns={'np':np}
 for rel,names in [('crossplatform/round20/detect.py',{'rect_iou'}),('crossplatform/round23/rank23.py',{'sigmoid','relation_value'})]:
  tree=ast.parse((ROOT/rel).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in nodes}==names
  assert not any(isinstance(x,(ast.Import,ast.ImportFrom)) for n in nodes for x in ast.walk(n))
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen-original-pure-reference>','exec'),ns)
 return ns['relation_value']
def mask_objects(masks):
 h,w=map(int,masks['original_hw']);objects=[];valid=[]
 for m,mask in enumerate(masks['masks_bool']):
  yy,xx=np.nonzero(mask);defined=len(xx)>0;center=np.array([xx.mean(),yy.mean()]) if defined else None
  if defined:center=(center+.5)/np.array([w,h])
  objects.append({'mask_centroid_norm2':center,'touches_image_border':bool(defined and (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())),'detector_box_xyxy':np.asarray(masks['boxes_xyxy'][m],float)})
  x1,y1,x2,y2=map(float,masks['boxes_xyxy'][m]);valid.append(bool(defined and x1<x2 and y1<y2 and min(w,int(x2))>max(0,int(x1)) and min(h,int(y2))>max(0,int(y1))))
 return w,h,objects,np.asarray(valid,bool)
def weights(pool,S,C,I,boxes):
 valid=(S>=3)&(C>=.15)&(I>=.1)&(boxes[:,3:]>0).all(1)[:,None];quality=np.where(valid,C*np.sqrt(np.maximum(I,0)),0.)
 W=quality[:,pool].copy() if len(pool) else np.zeros((256,0));W/=np.maximum(W.sum(1,keepdims=True),1e-12)
 return W,quality,valid
def readout(s,W,pool,e,pr):
 v=np.zeros(256);p=np.zeros(256);r=np.zeros(256);mapped=W.sum(1)>0
 if len(pool):
  cue=W@e[pool];rel=W@pr[pool];q0=int(s.argmax());av=cue[q0] if mapped[q0] else float(np.median(e[pool]));ar=rel[q0] if mapped[q0] else np.median(pr[pool],axis=0)
  v[mapped]=np.clip((cue[mapped]-av)/2,-2,2);p[mapped]=np.clip(rel[mapped,0]-ar[0],-2,2);r[mapped]=np.clip(rel[mapped,1]-ar[1],-2,2)
 return s.astype(float)+2*v+2*p+2*r,v,p,r
def old_classset(category):
 if category=='person':return {0}
 if category in ('bicycle','bike','cyclist','biker'):return {1}
 if category=='motorcycle':return {3}
 if category=='bus':return {5}
 if category in ('truck','trailer'):return {7}
 if category in ('car','vehicle','sedan','van','taxi','suv','jeep'):return {2,7}
 return set()
def original_lexicon():
 tree=ast.parse((ROOT/'crossplatform/round7/regression_adapter.py').read_text())
 node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LEXICON' for t in n.targets))
 value=ast.literal_eval(node.value);assert isinstance(value,tuple) and all(isinstance(x,str) for x in value);return value
def category_from_text(caption,g,lexicon):
 end=g['base_text'].get('subject',{}).get('span',[0,len(caption)])[1];match=re.search(r'\b(?:'+'|'.join(lexicon)+r')\b',caption[:end],re.I)
 if not match:return g['category'].lower()
 word=match.group().lower();return 'person' if word in ('pedestrian','person','people','man','woman','child','boy','girl','walker','worker') else word
def main():
 start=time.perf_counter();auth=read(EXP/'ROOT_INDEPENDENT_FULL_AUDIT_AUTHORIZATION.json');lp=EXP/'FULL_INDEPENDENT_AUDIT_PLAN_LOCK.json';plan=read(lp)
 assert auth['state']=='ROOT_AUTHORIZED_R50_INDEPENDENT_FULL5912_ARITHMETIC_AUDIT_NO_GT' and auth['program_sha256']==sha(__file__) and auth['independent_plan_sha256']==sha(lp)
 assert plan['n']==5912 and plan['arms']==ARMS and plan['no_GT_rawXYZ_or_models'] is True
 for rel,h in {**plan['programs_sha256'],**plan['inputs_sha256']}.items():assert 'FINAL384' not in rel and 'round46/final' not in rel and sha(ROOT/rel)==h
 producer_path=ROOT/plan['producer_prediction_plan'];producer=read(producer_path);assert producer['arms']==ARMS and producer['n']==5912 and producer['manifest']=='research/round8/experiment/full5912_gpu_inputs.json'
 pred=ROOT/producer['output'];run=ROOT/producer['runroot'];assert pred.is_relative_to(ROOT/'crossplatform/round50') and run.is_relative_to(ROOT/'crossplatform/round50')
 assert auth['prediction_seal_sha256']==sha(pred/'SEALED.json');ps=sealmap(pred);assert ps['n']==5912 and ps['arms']==ARMS
 for folder,h in producer['sealed_roots'].items():assert sha(ROOT/folder/'SEALED.json')==h
 for f,h in ps['files'].items():assert sha(pred/f)==h
 rows=read(ROOT/producer['manifest']);assert len(rows)==5912 and len({r['id'] for r in rows})==5912 and collections.Counter(r['platform'] for r in rows)=={'drone':2984,'quad':2928}
 identity=lambda r:tuple(r[k] for k in ('imagepath','pointpath','metapath','caption'))
 assert len({identity(r) for r in rows})==5912
 devrows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');dev={identity(r):r for r in devrows};assert len(dev)==336
 evidence_index=read(ROOT/producer['evidence_case_index']);assert set(evidence_index)=={r['id'] for r in rows}
 source=ROOT/'crossplatform/round8/full5912_attempt001';source_seal=sealmap(source);assert source_seal['n_samples']==5912 and source_seal['manifest_sha256']==sha(ROOT/producer['manifest'])
 assert source_seal['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 relation=original_relation();lexicon=original_lexicon();assert not OUT.exists();OUT.mkdir(parents=True);flags=[];bridges=0;seed_equal=0;totals=collections.Counter()
 try:
  for i,row in enumerate(rows):
   rid=row['id'];a=load(member(run/'prepared','arrays_PRIVATE/'+rid+'.npz'));c=read(member(run/'prepared','cases_PRIVATE/'+rid+'.json'));g=read(member(run/'text','cases_PRIVATE/'+rid+'.json'));masks=load(member(run/'frames','masks_PRIVATE/'+rid+'.npz'));z=load(member(pred,'samples_PRIVATE/'+rid+'.npz'));case=read(member(pred,'cases_PRIVATE/'+rid+'.json'))
   srcpath=member(source,rid+'.npz')
   with np.load(srcpath,allow_pickle=False) as ss:sb=ss['source_boxes_raw6'].copy(order='K');sp=ss['source_text_scores256'].copy(order='K');source_image_sha=ss['image_file_sha256'].tobytes().hex();assert str(ss['sample_id'])==rid
   exact(a['source_boxes'],sb,'prepared original R8 boxes');exact(a['source_scores'],sp,'prepared original R8 scores');exact(z['source_boxes'],sb,'prediction original R8 boxes');exact(z['source_scores'],sp,'prediction original R8 scores')
   assert sb.shape==(256,6) and sp.shape==(256,) and np.isfinite(sb).all() and np.isfinite(sp).all()
   ref=evidence_index[rid];ep=ROOT/ref['file'];assert sha(ep)==ref['sha256'];roots=[ROOT/f for f in producer['sealed_roots'] if ep.is_relative_to(ROOT/f)];assert roots
   assert any(sealmap(f)['files'].get(ep.relative_to(f).as_posix())==ref['sha256'] for f in roots);ev=read(ep)
   assert ev['id_PRIVATE']==rid and ev['frame_bindings']==c['bindings'] and c['bindings']['image_sha256']==source_image_sha
   for rel,h in ev['input_bindings_PRIVATE'].items():
    assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and 'FINAL384' not in rel and 'round46/final' not in rel and sha(ROOT/rel)==h
   assert hashlib.sha256(g['subject_query'].encode()).hexdigest()==c['actual_subject_query_sha256']==ev['actual_subject_query_sha256'];assert case['evidence_file_sha256']==ref['sha256'] and case['geometry_file_sha256']==sha(run/'prepared/arrays_PRIVATE'/(rid+'.npz')) and case['text_file_sha256']==sha(run/'text/cases_PRIVATE'/(rid+'.json')) and case['mask_file_sha256']==sha(run/'frames/masks_PRIVATE'/(rid+'.npz'))
   w,h,objects,crop=mask_objects(masks);M=len(objects);assert c['id_PRIVATE']==case['id_PRIVATE']==rid and c['platform']==case['platform']==row['platform']
   assert c['bindings']['mask_cache_sha256']==case['mask_file_sha256']
   exact(a['mask_sha256'],np.asarray([hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest() for m in masks['masks_bool']]),'mask pixel identity/order')
   exact(a['detector_boxes'],masks['boxes_xyxy'],'native detector boxes');exact(a['native_detector_class'],masks['class_ids'],'native predicted classes')
   S=a['legacy_support'];C=a['legacy_coverage'];I=a['legacy_projection_iou'];counts=a['strict_counts'];N=a['allvisible'];u=a['unassigned'];amb=a['ambiguous']
   assert S.shape==C.shape==I.shape==counts.shape==(256,M) and all(np.isfinite(x).all() for x in (C,I)) and (S>=0).all() and ((C>=0)&(C<=1)).all() and ((I>=0)&(I<=1)).all() and np.array_equal(counts.sum(1)+u+amb,N)
   exact(crop,a['crop_valid'],'crop eligibility');old=a['old_pool'];wide=a['wide_pool'];assert c['mask_pool_PRIVATE']==old.tolist() and c['wide_pool_PRIVATE']==wide.tolist()
   expectedwide=np.array(sorted(set(old.tolist())|{m for m in range(M) if crop[m] and counts[:,m].any()}),np.int64);exact(wide,expectedwide,'wide entrance');assert crop[old].all()
   ow,quality,valid=weights(old,S,C,I,sb);ww,_,_=weights(wide,S,C,I,sb);exact(ow,a['old_weights'],'old W');exact(ww,a['wide_old_weights'],'wide W');exact(quality,a['legacy_quality'],'old quality');exact(valid,a['legacy_valid'],'old geometry valid')
   # Reconstruct coarse/source supplementation; the original pair cache is
   # preserved, but its 3D median separation is not a fresh geometry audit.
   assert c['category']==category_from_text(row['caption'],g,lexicon);classes=old_classset(c['category']);labels=masks['class_ids'];target=[m for m in range(M) if valid[:,m].any() and (not classes or int(labels[m]) in classes)];target=sorted(target,key=lambda m:(-float(np.max(sp*quality[:,m])),m))[:6];q0=int(sp.argmax());source_mask=int(quality[q0].argmax()) if M and quality[q0].max()>0 else None
   if source_mask is not None:target.append(source_mask)
   for pair in c['legacy_preparation_PRIVATE']['pairs']:
    mm=pair['mask_indices_PRIVATE'];assert len(mm)==2 and all(0<=m<M for m in mm) and pair['qA_PRIVATE']==q0;target.extend(mm)
   exact(old,np.asarray(sorted(set(target)),np.int64),'old entrance metadata');assert c['source_mask_PRIVATE']==source_mask
   assert set(ev['targets_PRIVATE'])==set(map(str,wide));e=np.full(M,np.nan);pr=np.zeros((M,2));target_prompts=set()
   for m in wide:
    views=ev['targets_PRIVATE'][str(m)];assert len(views)==2;deltas=[]
    for v in views:
     assert v['visual_evidence']==v['full']['log_odds_yes_no']-v['gray']['log_odds_yes_no'];deltas.append(v['visual_evidence'])
     for mode in ('full','gray'):
      assert v[mode]['generated_tokens']==0 and np.isfinite(v[mode]['log_odds_yes_no']) and 0<v[mode]['input_tokens']<=8192
      target_prompts.add(v[mode]['prompt_sha256'])
    e[m]=float(np.mean(deltas));pr[m]=relation(g,objects[m],ev,w,h)
   assert len(target_prompts)<=1 and np.isfinite(e[wide]).all()
   assert set(ev['references'])<={ref['entity'] for ref in g['references']}
   if not len(old):assert ev['references']=={}, 'No new reference acquisition for wide-only pool'
   for observations in ev['references'].values():
    assert len(observations)<=3
    for observation in observations:
     assert observation['visual_evidence']==observation['full']['log_odds_yes_no']-observation['gray']['log_odds_yes_no']
     assert observation['full']['prompt_sha256']==observation['gray']['prompt_sha256']
     for mode in ('full','gray'):assert observation[mode]['generated_tokens']==0 and np.isfinite(observation[mode]['log_odds_yes_no'])
   exact(z['image_e'],e,'all target evidence');exact(z['raw_mask_PR'],pr,'original relation evidence');exact(z['old_pool'],old,'pred oldpool');exact(z['wide_pool'],wide,'pred widepool');exact(z['old_weights'],ow,'pred oldW');exact(z['wide_old_weights'],ww,'pred wideW')
   os,ov,op,orr=readout(sp,ow,old,e,pr);ns,nv,np_,nr=readout(sp,ww,wide,e,pr);scores=np.stack([sp.astype(float),os,ns]);queries=np.argmax(scores,axis=1).astype(np.int64)
   exact(z['scores'],scores,'all three full scores');exact(z['queries'],queries,'all three selected query');exact(z['selected_original_boxes'],sb[queries],'all three original boxes');assert z['arm_names'].tolist()==ARMS
   for field,val in [('vision',[np.zeros(256),ov,nv]),('position',[np.zeros(256),op,np_]),('relation',[np.zeros(256),orr,nr])]:exact(z[field],np.stack(val),field)
   before=dev.get(identity(row));isdev=before is not None;assert case['is_DEV336']==isdev
   if before:
    assert row['platform']==before['platform'] and row['scene']==before['scene'];seed_equal+=row.get('seed')==before.get('seed')
    dz=load(member(ROOT/'crossplatform/round50/predicted336_attempt001','samples_PRIVATE/'+before['id']+'.npz'))
    exact(scores,dz['scores'][:3],'DEV three score bridge');exact(queries,dz['queries'][:3],'DEV three query bridge');exact(sb[queries],dz['selected_original_boxes'][:3],'DEV three box bridge');exact(sb,dz['source_boxes'],'DEV source box pool');exact(sp,dz['source_scores'],'DEV source scores');bridges+=1
   totals.update(n=1,old_masks=len(old),wide_masks=len(wide),weight_cells=256*(len(old)+len(wide)))
   flags.append({'row':f'F{i+1:05d}','platform':row['platform'],'DEV336':isdev,'old_pool_count':len(old),'wide_pool_count':len(wide),'old_q0_mapped':bool(ow[q0].sum()>0),'all_arithmetic_scores_and_boxes_bitexact':True})
   if (i+1)%256==0:print('INDEPENDENT_FULL50',i+1,flush=True)
 except BaseException as exc:
  dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_n':len(flags),'no_GT_read_or_automatic_retry':True});raise
 assert len(flags)==5912 and bridges==336;dump(OUT/'ANONYMOUS_ROWS.json',flags);result={'state':'PASS_R50_INDEPENDENT_FULL5912_FIXED_SOURCE_THREE_READOUT_AUDIT_NO_GT','n':5912,'arms':3,'source256_box_score_pools_bitexact':5912,'full_score_values_bitexact':5912*3*256,'original_selected_boxes_bitexact':17736,'DEV336_three_arm_score_query_box_bridges_bitexact':bridges,'DEV_join_by_rawtriple_caption_platform_scene_not_ID_or_alias':True,'DEV336_metadata_seed_equal_n':seed_equal,'DEV336_metadata_seed_mismatch_n':336-seed_equal,'seed_mismatch_if_any_requires_explanation_before_GT_score':True,'old_and_wide_weight_cells_recomputed':totals['weight_cells'],'old_pool_masks':totals['old_masks'],'wide_pool_masks':totals['wide_masks'],'original_relation_from_mask_pixels_recomputed':True,'full_data_not_new_independent_confirmation':True,'no_author_R50_pure_algorithm_import':True,'memory_layout_preserved_order_K_and_old_weights_reconstructed':True,'rawXYZ_projection_and_pair_median_separation_not_independently_recomputed':True,'frozen_network_values_not_recomputed_or_semantic_identity_certified':True,'no_GT_rawRGB_rawXYZ_models_or_protected_FINAL_specific_artifacts_read':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(OUT/'AUDIT_RESULT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
