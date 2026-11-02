"""R51 independent post-seal contract/numeric audit draft. NO GT or models.

Does not import assemble51, safe_predict51 or evidence51 as a reference.
Replays known-cue fusion, source-reference probabilities, fixed endpoints and
numeric-call closure from immutable source/evidence bundles. Root runs it.
"""
from pathlib import Path
import argparse,hashlib,json,math,time,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
FACTORS=(.5,1.,2.)
MAIN=['source','external_G','replace_instances','replace_appearance_c05','replace_appearance_c1','replace_appearance_c2','replace_references','replace_all_c05','replace_all_c1','replace_all_c2']
SAFE=['source','replace_all_c05','replace_all_c1','replace_all_c2']
NOASSOCIATION=['no_instance_association_c05','no_instance_association_c1','no_instance_association_c2']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def path(s):
 p=Path(s);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve();assert 'round46/final' not in str(p) and 'FINAL384' not in str(p);return p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b,name):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name

def attributes(a,entity):
 wanted=next((r['recognized_color_words'] for r in a['entity_attribute_requests'] if r['entity']==entity),[]);vals=[];known=[]
 for r in a['instances']:
  x=[r['color_logodds_relative_scene_visible'].get(c) for c in wanted];x=[v for v in x if v is not None and np.isfinite(v)]
  vals.append(float(np.mean(x))*r['color_evidence_reliability'] if x else 0.);known.append(bool(x))
 return np.asarray(vals,float),np.asarray(known,bool)

def literal_fusion(s,w,e,pr):
 V=np.zeros(256);P=np.zeros(256);R=np.zeros(256);mapped=w.sum(1)>0
 if len(e):
  vv=w@e;rr=w@pr;q=int(s.argmax());av=vv[q] if mapped[q] else float(np.median(e));ar=rr[q] if mapped[q] else np.median(pr,axis=0)
  V[mapped]=np.clip((vv[mapped]-av)/2,-2,2);P[mapped]=np.clip(rr[mapped,0]-ar[0],-2,2);R[mapped]=np.clip(rr[mapped,1]-ar[1],-2,2)
 return s.astype(float)+2*V+2*P+2*R

def measured(s,w,e,pr,ek,pk):
 # Independent expression retains producer's arithmetic order for exact audit.
 e=np.asarray(e,float);pr=np.asarray(pr,float);ek=np.asarray(ek,bool);pk=np.asarray(pk,bool);w=np.asarray(w,float)
 value=np.c_[np.where(ek,e,0.),np.where(pk,pr,0.)];known=np.c_[ek,pk]
 mass=w@known.astype(float);available=mass>0;projected=w@value;cue=np.zeros((3,256));anchor=np.zeros(3);q=int(s.argmax())
 for j in range(3):
  if known[:,j].any():
   anchor[j]=projected[q,j] if available[q,j] else float(np.median(value[known[:,j],j]))
   have=available[:,j];cue[j,have]=np.clip((projected[have,j]-anchor[j])/(2. if j==0 else 1.),-2,2)
 assert np.all(cue[~available.T]==0),'unknown became signed cue'
 return s.astype(float)+2*cue[0]+2*cue[1]+2*cue[2],cue,available.T,anchor,mass.T

def spatial(raw,graph,references,W,H,old_masks=None,domain='raw3d'):
 result=np.zeros((len(raw['instance_rect_xyxy']),2));known=np.zeros_like(result,dtype=bool)
 for j,rect in enumerate(raw['instance_rect_xyxy']):
  if old_masks is None:
   lo,hi=raw['instance_crop_mask_pixel_offsets'][j:j+2];px=raw['instance_crop_mask_pixel_xy'][lo:hi]
   if not len(px) or ((px[:,0]==0)|(px[:,1]==0)|(px[:,0]==W-1)|(px[:,1]==H-1)).any():continue
   x=float(((px.mean(0)+.5)/[W,H])[0])
  else:
   mask=old_masks[j];yx=np.argwhere(mask)
   if not len(yx) or mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():continue
   x=float(((yx[:,::-1].mean(0)+.5)/[W,H])[0])
  glob=[];rel=[]
  for g in graph.get('global_relations',[]):
   if g.get('frame')!='camera_image':continue
   p=g['predicate']
   if p in ('GLOBAL_LEFT','LEFT_OF_CENTER'):glob.append(float(-np.tanh((x-.5)/.15)))
   elif p in ('GLOBAL_RIGHT','RIGHT_OF_CENTER'):glob.append(float(np.tanh((x-.5)/.15)))
   elif p=='GLOBAL_CENTER':glob.append(1-4*abs(x-.5))
  if glob:result[j,0]=np.mean(glob);known[j,0]=True
  for g in graph.get('reference_relations',[]):
   if g.get('frame')!='camera_image' or g.get('predicate') not in ('LEFT_OF','RIGHT_OF'):continue
   weighted=0.;total=0.
   for h in references.get(g['subject'],{}).get('hypotheses',[]):
    if h.get('xy_norm2') is None or (h.get('instance_domain')==domain and h.get('instance_index')==j):continue
    if h.get('rect') is not None:
     a=np.asarray(h['rect'],float);b=np.asarray(rect,float);intersection=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-intersection
     if union>0 and intersection/union>=.3:continue
    weight=float(h['probability'])
    if weight<=0:continue
    direction=1 if g['predicate']=='RIGHT_OF' else -1
    weighted+=weight*float(np.tanh(direction*(h['xy_norm2'][0]-x)/(.15+.1*h.get('width_norm',0.))));total+=weight
   if total>0:rel.append(weighted/(1+total));known[j,1]=True
  if rel:result[j,1]=np.mean(rel)
 return result,known

def check_reference_distributions(raw,attrs,records,stored):
 checked=0
 for entity in records[0]:
  z=stored[entity+'_source_logits'];p0=stored[entity+'_source_probability'];finite=np.isfinite(z)
  assert not np.isnan(z).any();mass=np.zeros(len(z));keep=[];u=1.;other=0.
  if finite.any():
   mx=max(0.,float(z[finite].max()));mass[finite]=np.exp(np.clip(z[finite]-mx,-700,0));unknown=np.exp(-mx);den=unknown+mass.sum();mass/=den
   u=float(unknown/den);keep=sorted(np.flatnonzero(finite),key=lambda j:(-float(mass[j]),int(j)))[:3];other=float(max(0.,1-u-sum(mass[j] for j in keep)))
  exact(mass,p0,'all-reference-source-probability')
  attr,known=attributes(attrs,entity)
  for fi,c in enumerate(FACTORS):
   r=records[fi][entity];d=r['distribution'];assert d['source_top3_instance_ids']==[int(j) for j in keep]
   assert d['source_unknown_probability']==u and d['source_truncated_probability']==other
   assert d['source_probabilities']==mass.tolist()
   hs=r['hypotheses'];assert [h['instance_index'] for h in hs]==[int(j) for j in keep]
   adjusted=np.array([mass[j]*np.exp(np.clip(c*attr[j],-8,8)) if known[j] else mass[j] for j in keep]);den=u+other+float(adjusted.sum());adjusted/=den
   assert [h['probability'] for h in hs]==[float(v) for v in adjusted]
   assert d['unknown_probability']==float((u+other)/den) and d['retained_probability']==float(adjusted.sum())
   assert d['unknown_probability']+d['retained_probability']<=1+1e-12
   for h in hs:
    j=h['instance_index'];assert h['source_probability']==float(mass[j]) and h['uncertainty_mass_explicit'] is True
    assert h['rect']==raw['instance_rect_xyxy'][j].tolist()
   checked+=1
 return checked

def check_visual(folder,worklist):
 seal=read(folder/'SEALED.json');assert seal['state']=='R51_DIAGNOSTIC_NUMERIC_VISUAL_COMPLETE_NO_GT'
 for rel,h in seal['files'].items():assert sha(folder/rel)==h
 work=read(worklist);bindings=read(folder/'RUN_BINDINGS.json');assert bindings['worklist_sha256']==sha(worklist)
 completed=0;unknown=0;values={};calls={}
 for cp in (folder/'calls_PRIVATE').glob('*.json'):
  r=read(cp);assert r['state']=='COMPLETED';b=r['binding'];wi=b['work_index'];v=r['value'];raw=r['raw']
  assert 0<=wi<len(work) and b['plan_sha256']==bindings['plan_sha256']
  assert b['work_item_sha256']==hashlib.sha256(json.dumps(work[wi],sort_keys=True).encode()).hexdigest()
  assert float(np.float32(raw['yes_lse_float32'])-np.float32(raw['no_lse_float32']))==raw['logodds_float32']==v['log_odds_yes_no']
  assert v['image_sha256']==b['image_sha256'] and v['prompt_sha256']==b['prompt_sha256'] and v['generated_tokens']==0
  key=(wi,b['view'],b['mode']);assert key not in calls;calls[key]=v;completed+=1
 call_item_indices={key[0] for key in calls}
 for i,w in enumerate(work):
  rel=f'items_PRIVATE/{i:07d}.json';r=read(folder/rel);assert sha(folder/rel)==seal['files'][rel] and r['work_index']==i
  key=(w['id_PRIVATE'],w['kind'],w.get('reference_entity'),w['source_instance_id']);assert key not in values
  if r['status'].startswith('UNKNOWN'):
   assert r['views']==[] and i not in call_item_indices;values[key]=None;unknown+=1;continue
  assert r['status']=='COMPLETED' and w['status']=='READY'
  nview=2 if w['recipe']=='TARGET_DUAL_CROP_FULL_GRAY' else 1;assert len(r['views'])==nview
  for j,v in enumerate(r['views']):
   assert calls[(i,j,'full')]==v['full'] and calls[(i,j,'gray')]==v['gray']
   assert v['visual_evidence']==v['full']['log_odds_yes_no']-v['gray']['log_odds_yes_no']
  values[key]=r['views']
 counts=read(folder/'COUNTS.json');assert counts['physical_completed_call_files']==completed<=counts['registered_call_cap']
 assert counts['model_unchanged'] is True and counts['new_DINO_forwards']==counts['optimizer_steps']==counts['generated_tokens']==0
 return values,{'work_items':len(work),'UNKNOWN_items':unknown,'physical_COMPLETED_numeric_calls':completed}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_INDEPENDENT_PREDICTION_AUDIT_NO_GT' and auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(args.plan)
 assert plan['n']==336 and plan['no_GT_models_or_score'] is True
 for g in ('programs_sha256','inputs_sha256'):
  for f,h in plan[g].items():assert sha(path(f))==h,('pin',f)
 roots={k:path(plan[k]) for k in ('prediction_root','safe_prediction_root','safe_original_root','noassociation_root','rawsafe_root','canonical_source_root','r50_prediction_root','target_visual_root','reference_visual_root','hybrid_root')}
 seals={k:read(p/'SEALED.json') for k,p in roots.items()}
 for k,p in roots.items():assert plan['inputs_sha256'][(p/'SEALED.json').relative_to(ROOT).as_posix()]==sha(p/'SEALED.json')
 assert auth['prediction_seal_sha256']==sha(roots['prediction_root']/'SEALED.json')
 def bound(k,rel):
  p=roots[k]/rel;assert sha(p)==seals[k]['files'][rel];return p
 out=path(plan['output_directory']);assert not out.exists();out.mkdir(parents=True);done=[];start=time.perf_counter();numeric=0;refs=0;fixture_rows=[]
 try:
  rows=read(path(plan['manifest']));assert len(rows)==336 and plan['manifest']=='research/round24/input/ALL336_PRIVATE.json'
  allowed_raw_points={path(r['pointpath']) for r in rows}
  allowed_raw_inputs=allowed_raw_points|{path(r['imagepath']) for r in rows}
  for f in ('crossplatform/round51/relation_fixture51.py','crossplatform/round51/evidence51_fidelity.py','crossplatform/round23/rank23.py','crossplatform/round20/detect.py'):assert f in plan['programs_sha256']
  sys.path.insert(0,str(ROOT/'crossplatform/round51'));from relation_fixture51 import compare_legacy_fixture
  arms=seals['prediction_root']['arm_names'];assert arms[:10]==MAIN and len(arms)==16 and arms[13:]==NOASSOCIATION
  # Final-safe input graph is an exact four-input whitelist, separate from all hybrid roots.
  sp=read(path(plan['safe_prediction_plan']));assert sp['inputs_sha256']=={sp['manifest']:plan['inputs_sha256'][sp['manifest']],**{sp[k]+'/SEALED.json':sha(path(sp[k])/'SEALED.json') for k in ('prepared_root','capture_root','reference_root')}}
  allowed_safe_roots=tuple(path(sp[k]) for k in ('prepared_root','capture_root','reference_root'));safe_manifest_path=path(sp['manifest']);safe_reference_root=path(sp['reference_root'])
  assert sp['no_external_vision_cache_inputs'] is True and seals['safe_prediction_root']['external_vision_outputs_or_mask_derived_caches_read'] is False
  allowed_roles={'REGISTERED_RAW_INPUTS','PREPARED_ROOT_SEAL','CAPTURE_ROOT_SEAL','REFERENCE_ROOT_SEAL','RAW_SAFE_POINT_INSTANCES','RAW_SAFE_BINDINGS','MEASURED_RGB_AND_PURE_TEXT','SAME_MODEL_INPUT_BRIDGE','SUBJECT_GEOMETRY_SCORE_BRIDGE_ONLY','ORIGINAL_RAW_XYZ_FOR_REFERENCE_SUPPORT','SAME_MODEL_REFERENCE_GRAPH_BINDINGS','SAME_GROUNDER_REFERENCE_HEADSPAN_ONLY','REFERENCE_SAMPLING_AND_MASK_RECEIPT'}
  assert seals['safe_prediction_root']['previous_safe_seal_sha256']==sha(roots['safe_original_root']/'SEALED.json')
  ledger=read(bound('safe_original_root','INPUT_LEDGER_PRIVATE.json'))
  for item in ledger['items']:
   assert item['role'] in allowed_roles
   p=path(item['path_PRIVATE']);assert sha(p)==item['sha256']
   if item['role']=='ORIGINAL_RAW_XYZ_FOR_REFERENCE_SUPPORT':assert p in allowed_raw_points
   else:assert any(p.is_relative_to(folder) for folder in allowed_safe_roots) or p==safe_manifest_path
  repair=read(bound('safe_prediction_root','READ_BINDINGS_PRIVATE.json'));assert repair['no_original_media_or_refsupport_recompute'] is True
  for item in repair['files']:
   p=path(item['path_PRIVATE']);assert sha(p)==item['sha256'] and (p.is_relative_to(roots['safe_original_root']) or p.is_relative_to(roots['rawsafe_root']))
  nal=read(bound('noassociation_root','INPUT_LEDGER_PRIVATE.json'))
  na_roles={'RAW_SAFE_BINDINGS','PURE_GRAPH_SCENE_COLOR','RAW_SAFE_VISIBLE_POINTS_SOURCE_ONLY','ORIGINAL_RGB','ORIGINAL_RAW_XYZ','SAME_MODEL_REFERENCE_INPUTS','SAME_MODEL_REFERENCE_BOXES_LOGODDS'}
  for item in nal['files']:
   assert item['role'] in na_roles;pp=path(item['path_PRIVATE']);assert sha(pp)==item['sha256']
   if item['role'] in {'ORIGINAL_RGB','ORIGINAL_RAW_XYZ'}:assert pp in allowed_raw_inputs
   else:assert pp.is_relative_to(roots['rawsafe_root']) or pp.is_relative_to(safe_reference_root)
  target,tc=check_visual(roots['target_visual_root'],path(plan['target_worklist']));refvis,rc=check_visual(roots['reference_visual_root'],path(plan['reference_worklist']))
  work=read(path(plan['reference_worklist']));source_w={(w['id_PRIVATE'],w['reference_entity'],w['source_instance_id']):w for w in work};assert len(source_w)==len(work)
  statusrows=read(bound('prediction_root','READOUT_STATUS_PRIVATE.json'));assert len(statusrows)==336 and len({r['id'] for r in statusrows})==336
  status={r['id']:r['arm_statuses'] for r in statusrows};assert set(status)=={r['id'] for r in rows}
  for row in rows:
   sid=row['id'];p=load(bound('prediction_root','samples_PRIVATE/'+sid+'.npz'));safe=load(bound('safe_prediction_root','samples_PRIVATE/'+sid+'.npz'))
   raw=load(bound('rawsafe_root','arrays_PRIVATE/'+sid+'.npz'));attrs=read(bound('rawsafe_root','attributes_PRIVATE/'+sid+'.json'))
   old=load(bound('r50_prediction_root','samples_PRIVATE/'+sid+'.npz'))
   originalsafe=load(bound('safe_original_root','samples_PRIVATE/'+sid+'.npz'));na=load(bound('noassociation_root','samples_PRIVATE/'+sid+'.npz'))
   with np.load(bound('canonical_source_root','samples_PRIVATE/'+sid+'.npz'),allow_pickle=False) as z:canonical_boxes=z['boxes'].copy();canonical_scores=z['scores'].copy()
   for z in (p,safe):exact(z['source_boxes'],canonical_boxes,'canonical sourceboxes');exact(z['source_scores'],canonical_scores,'canonical sourcescores')
   assert p['arm_names'].tolist()==arms and p['queries'].shape==(len(arms),) and np.issubdtype(p['queries'].dtype,np.integer)
   assert np.isfinite(p['scores']).all() and p['scores'].shape==(len(arms),256)
   assert np.array_equal(p['queries'],p['scores'].argmax(1)) and ((p['queries']>=0)&(p['queries']<256)).all()
   exact(p['selected_original_boxes'],canonical_boxes[p['queries']],'original selected')
   exact(p['scores'][0],old['scores'][0],'source F bridge');oi=old['arm_names'].tolist().index('wideentry_oldW')
   exact(p['scores'][1],old['scores'][oi],'external G F bridge');assert p['queries'][1]==old['queries'][oi]
   exact(p['scores'][7:10],safe['scores'][1:],'safe copied final F')
   exact(p['scores'][10:13],safe['no_relation_scores'],'safe copied no-R F');exact(p['scores'][13:],na['scores'],'copied noassociation F')
   for field in ('source_boxes','source_scores','instance_attribute_e','instance_attribute_known','query_instance_conditional_weights','no_relation_scores'):exact(safe[field],originalsafe[field],'fidelity unchanged '+field)
   exact(safe['components'][:,:2],originalsafe['components'][:,:2],'fidelity V/P unchanged')
   for field in ('source_boxes','source_scores'):exact(na[field],safe[field],'noassociation canonical '+field)
   assert na['arm_names'].tolist()==NOASSOCIATION
   for fi,factor in enumerate(FACTORS):
    nf,nc,nk,an,mn=measured(canonical_scores,np.eye(256),factor*na['query_attribute_e'],na['raw_PR'][fi],na['query_attribute_known'],na['raw_PR_known'][fi])
    exact(nf,na['scores'][fi],'independent identity-W noassociation fusion');exact(nc,na['components'][fi],'identity-W cue');assert np.array_equal(nk,na['cue_query_known'][fi]);exact(an,na['cue_anchors'][fi],'identity-W anchor');exact(mn,na['cue_query_known_weight_mass'][fi],'identity-W mass')
   nacase=read(bound('noassociation_root','cases_PRIVATE/'+sid+'.json'));assert nacase['external_vision_outputs_or_mask_derived_caches_read'] is False and nacase['RGB_retained'] is True and nacase['instance_clustering_ownership_and_dedup_removed'] is True
   nad=read(bound('noassociation_root','references_PRIVATE/'+sid+'.json'))
   for ent,d in nad.items():
    prob=np.asarray(d['source_probabilities'],float);assert prob.shape==(256,) and np.isfinite(prob).all() and (prob>=0).all()
    top=sorted(np.flatnonzero(prob>0),key=lambda q:(-float(prob[q]),int(q)))[:3]
    assert top==d['source_top3_queries_PRIVATE'] and d['duplicate_query_confidence_intentionally_not_removed'] is True
    assert abs(float(prob.sum())+d['source_unknown_probability']-1)<1e-12
    for ref_factor in d['hypotheses_by_factor']:
     hh=ref_factor['hypotheses'];assert [h['instance_index'] for h in hh]==top
     assert abs(sum(h['probability'] for h in hh)+ref_factor['unknown_probability']-1)<1e-12
     for h in hh:assert h['source_probability']==prob[h['instance_index']]
   e,ek=attributes(attrs,'T');exact(e,safe['instance_attribute_e'],'safe attribute e');assert np.array_equal(ek,safe['instance_attribute_known'])
   refjson=read(bound('safe_prediction_root','references_PRIVATE/'+sid+'.json'));refarr=load(bound('safe_prediction_root','references_PRIVATE/'+sid+'.npz'))
   refs+=check_reference_distributions(raw,attrs,refjson['references_by_factor'],refarr)
   for entity,entry in refjson['references_by_factor'][0].items():
    keep=entry['distribution']['source_top3_instance_ids'];ww=[w for w in work if w['id_PRIVATE']==sid and w['reference_entity']==entity]
    assert [w['source_instance_id'] for w in ww]==(keep if keep else [-1])
    for w in ww:
     j=w['source_instance_id'];assert w['source_probability']==(float(refarr[entity+'_source_probability'][j]) if j>=0 else 0.)
   prep_case=read(bound('rawsafe_root','cases_PRIVATE/'+sid+'.json'));iw,ih=prep_case['image_size_wh']
   for fi,factor in enumerate(FACTORS):
    ipr,ipk=spatial(raw,attrs['text_graph_unchanged'],refjson['references_by_factor'][fi],iw,ih)
    exact(ipr,safe['raw_PR'][fi],'independent faithful typed spatial operator');assert np.array_equal(ipk,safe['raw_PR_known'][fi])
    F,C,K,A,M=measured(canonical_scores,raw['query_instance_conditional_weights'],factor*e,safe['raw_PR'][fi],ek,safe['raw_PR_known'][fi])
    exact(F,safe['scores'][fi+1],'independent safe known fusion');exact(C,safe['components'][fi],'independent safe cues');assert np.array_equal(K,safe['cue_query_known'][fi]);exact(A,safe['cue_anchors'][fi],'cue anchor');exact(M,safe['cue_query_known_weight_mass'][fi],'cue known mass')
    pr=safe['raw_PR'][fi].copy();pk=safe['raw_PR_known'][fi].copy();pr[:,1]=0.;pk[:,1]=False
    nr,*_=measured(canonical_scores,raw['query_instance_conditional_weights'],factor*e,pr,ek,pk);exact(nr,safe['no_relation_scores'][fi],'independent no-R')
   # Reconstruct hybrid raw evidence from separate sealed parents, not only
   # the producer's stored final score/selected query.
   case=read(bound('prediction_root','cases_PRIVATE/'+sid+'.json'))
   ctrl=load(bound('hybrid_root','controls_PRIVATE/'+sid+'.npz'));ma=read(bound('hybrid_root','oldmask_attributes_PRIVATE/'+sid+'.json'))
   oldrefs=read(bound('hybrid_root','controls_PRIVATE/'+sid+'.json'))['old_references'];fixture_objects=[]
   for j,mask in enumerate(ctrl['masks_bool']):
    yx=np.argwhere(mask);defined=bool(len(yx));fixture_objects.append({'mask_geometry_defined':defined,'touches_image_border':bool(defined and (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())),'mask_centroid_norm2':(yx[:,::-1].mean(0)+.5)/[iw,ih] if defined else None,'detector_box_xyxy':ctrl['boxes_xyxy'][j]})
   fixture=compare_legacy_fixture(attrs['text_graph_unchanged'],fixture_objects,{'references':oldrefs},(iw,ih),ctrl['raw_mask_PR'])
   if fixture['state'].startswith('NOT_COMPARABLE'):
    fixture_rows.append({'row':len(done),'state':fixture['state']})
   else:
    assert fixture['max_abs_difference']<=1e-12,'Literal R23 relation fixture exceeds registered diagnostic tolerance'
    fw=ctrl['wide_old_weights'];fp=ctrl['wide_pool'];fe=ctrl['image_e'][fp]
    fbase=literal_fusion(canonical_scores,fw,fe,fixture['baseline'][fp]);fnow=literal_fusion(canonical_scores,fw,fe,fixture['current'][fp])
    fixture_rows.append({'row':len(done),'state':'PASS_LITERAL_RELATION_FIXTURE_TOLERANCE_1e_12','max_raw_PR_error':fixture['max_abs_difference'],'raw_PR_byteexact':fixture['literal_original_vs_new_byteexact'],'saved_raw_PR_byteexact':fixture['literal_original_vs_saved_raw_PR_byteexact'],'max_final_F_error':float(np.max(np.abs(fbase-fnow))),'argmax_same':int(fbase.argmax())==int(fnow.argmax())})
   me,mk=attributes(ma,'T');pool=ctrl['wide_pool'];ow=ctrl['wide_old_weights'];graph=attrs['text_graph_unchanged']
   targete=np.zeros(len(e));targetknown=np.zeros(len(e),bool)
   for j in range(len(e)):
    vv=target[(sid,'target',None,j)]
    if vv is not None:targete[j]=float(np.mean([v['visual_evidence'] for v in vv]));targetknown[j]=True
   exact(targete,p['replace_instances_e'],'fresh target image evidence');assert np.array_equal(targetknown,p['replace_instances_e_known'])
   mpr,mpk=spatial(raw,graph,{k:{'hypotheses':v} for k,v in case['legacy_reference_hypotheses_PRIVATE'].items()},iw,ih)
   exact(mpr,p['replace_instances_raw_PR'],'M raw spatial');assert np.array_equal(mpk,p['replace_instances_raw_PR_known'])
   fm,cm,km,am,mm=measured(canonical_scores,raw['query_instance_conditional_weights'],targete,mpr,targetknown,mpk)
   exact(fm,p['scores'][2],'independent replace-instance full score')
   old_geometry={'instance_rect_xyxy':ctrl['boxes_xyxy']}
   gp=any(g.get('predicate') in {'GLOBAL_LEFT','LEFT_OF_CENTER','GLOBAL_RIGHT','RIGHT_OF_CENTER','GLOBAL_CENTER'} for g in graph.get('global_relations',[]))
   oldpk=np.asarray([gp and bool(mask.any()) and not (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()) for mask in ctrl['masks_bool']],bool)
   for fi,factor in enumerate(FACTORS):
    hs=case['appearance_reference_hypotheses_PRIVATE'][fi];dd=case['appearance_ref_diagnostics_PRIVATE'][fi]
    for entity,rr in dd.items():
     assert len(rr)==len(hs[entity])
     for d,h in zip(rr,hs[entity]):
      assert d['Qwen_value_used'] is False and d['factor']==factor
      conf=d['detector_confidence'];sig=lambda z:float(1/(1+np.exp(-np.clip(z,-30,30))))
      weight=conf*sig(factor*d['color_e']) if d['color_known'] else conf
      assert weight==d['weight']==h['probability']
      if d['color_known']:assert d['exclusive_visible_pixels']>=8
    qpr,qpk=spatial(old_geometry,graph,{k:{'hypotheses':v} for k,v in hs.items()},iw,ih,ctrl['masks_bool'],'oldmask')
    qpr[:,0]=ctrl['raw_mask_PR'][:,0];qpk[:,0]=oldpk
    exact(qpr,p['replace_appearance_raw_PR'][fi],'Q raw spatial');assert np.array_equal(qpk,p['replace_appearance_raw_PR_known'][fi])
    fq,cq,kq,aq,mq=measured(canonical_scores,ow,factor*me[pool],qpr[pool],mk[pool],qpk[pool]);exact(fq,p['scores'][3+fi],'independent replace-appearance full score')
   fresh_refs=case['new_reference_hypotheses_PRIVATE']
   original_request_gate=bool(len(ctrl['old_pool']));assert case['Donly_original_reference_request_gate']==original_request_gate and case['all_reference_collection_costs_retained'] is True
   if not original_request_gate:
    assert not fresh_refs and not case['reference_mass']
    assert case['Donly_reference_items_collected_but_not_consumed']==sum(k[0]==sid for k in refvis)
    assert not p['replace_references_raw_PR_known'][:,1].any() and not p['replace_references_raw_PR'][:,1].any()
   for entity,hh in fresh_refs.items():
    for h in hh:
     j=h['instance_index'];vv=refvis[(sid,'reference',entity,j)];assert vv is not None
     ew=float(np.mean([v['visual_evidence'] for v in vv]));fw=float(np.mean([v['full']['log_odds_yes_no'] for v in vv]));mass=source_w[(sid,entity,j)]['source_probability']
     sig=lambda z:float(1/(1+np.exp(-np.clip(z,-30,30))))
     assert h['probability']==mass*sig(fw)*sig(ew)
   dpr,dpk=spatial(old_geometry,graph,{k:{'hypotheses':v} for k,v in fresh_refs.items()},iw,ih,ctrl['masks_bool'],'oldmask');dpr[:,0]=ctrl['raw_mask_PR'][:,0];dpk[:,0]=oldpk
   exact(dpr,p['replace_references_raw_PR'],'D raw spatial');assert np.array_equal(dpk,p['replace_references_raw_PR_known'])
   fd,cd,kd,ad,md=measured(canonical_scores,ow,ctrl['image_e'][pool],dpr[pool],np.ones(len(pool),bool),dpk[pool]);exact(fd,p['scores'][6],'independent replace-reference full score')
   names=p['hybrid_arm_names'].tolist();assert names==MAIN[2:6]+['replace_references']
   for j,arm in enumerate(names):
    C=p['hybrid_components'][j];known=p['hybrid_cue_query_known'][j]
    assert np.array_equal(C[~known],np.zeros_like(C[~known]))
    F=canonical_scores.astype(float)+2*C[0]+2*C[1]+2*C[2];exact(F,p['scores'][arms.index(arm)],'hybrid expression')
   # Fresh target e is independently matched to its source instance and UNKNOWN.
   counts=0
   for j in range(len(e)):
    key=(sid,'target',None,j);assert key in target;counts+=target[key] is None
   assert case['new_target_unknown_count']==counts and case['Qonly_Qwen_numeric_feedback_used'] is False
   assert set(status[sid])==set(arms)
   for arm,ss in status[sid].items():assert set(ss)=={'unknown','fallback','no_parameter_update'} and all(type(v) is bool for v in ss.values()) and ss['no_parameter_update'] is True
   for j,arm in enumerate(names):
    unavailable=not bool(p['hybrid_cue_query_known'][j].any());assert status[sid][arm]['unknown']==status[sid][arm]['fallback']==unavailable
   for fi,arm in enumerate(SAFE[1:]):
    unavailable=not bool(safe['cue_query_known'][fi].any());assert status[sid][arm]['unknown']==status[sid][arm]['fallback']==unavailable
   if len(arms)==16:
    for fi,arm in enumerate(arms[10:13]):
     unavailable=not bool(safe['cue_query_known'][fi,:2].any());assert status[sid][arm]['unknown']==status[sid][arm]['fallback']==unavailable
    for fi,arm in enumerate(NOASSOCIATION):
     assert status[sid][arm]==nacase['arm_statuses'][arm]
   numeric+=p['scores'].size;done.append(sid)
  dump(out/'LITERAL_RELATION_FIXTURE_ROWS.json',fixture_rows)
  result={'state':'PASS_R51_INDEPENDENT_SEALED_PREDICTION_CONTRACT_AND_NUMERIC_AUDIT_NO_GT','n':336,'arm_names':arms,'prediction_seal_sha256':sha(roots['prediction_root']/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'no_GT_read':True,'models_or_optimizers_run':False,'numeric_score_values_checked':numeric,'reference_factor_distributions_checked':refs,'target_visual':tc,'reference_visual':rc,'safe_external_dependency_ledger_clean':True,'known_cue_unavailable_exact_zero_verified':True,'all_baselines_and_safe_copy_endpoints_exact':True,'literal_relation_fixture_rows':len(fixture_rows),'literal_relation_fixture_comparable':sum('max_raw_PR_error' in r for r in fixture_rows),'literal_relation_fixture_argmax_differences':sum(not r.get('argmax_same',True) for r in fixture_rows),'hybrid_independent_numeric_scope':'Fusion and typed spatial math, fresh target mean/full-gray closure, source quality times reference gates, appearance scalar gate formula; RGB descriptor/point connectivity upstream not rebuilt','raw_connected_components_or_calibration_truth_not_recomputed':True,'program_sha256':sha(__file__),'plan_sha256':sha(args.plan),'seconds':time.perf_counter()-start}
  dump(out/'AUDIT_RESULT.json',result);dump(out/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}});print(json.dumps(result),flush=True)
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_n':len(done),'no_GT_or_auto_retry':True});raise
if __name__=='__main__':main()
