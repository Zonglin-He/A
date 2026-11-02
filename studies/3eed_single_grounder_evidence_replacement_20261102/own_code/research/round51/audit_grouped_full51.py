"""Independent full grouped-carrier audit draft, NO GT or models.

Independently reconstructs proposal complete-link groups, actual-point medoids,
visible carriers and all-row numeric fusion. Only hash-selected two rows per
platform reread original XYZ for reference support and no-association geometry.
RGB color descriptors are sealed measurements, not independently remeasured.
No producer algorithms are imported. Root separately authorizes execution.
"""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit51 import attributes,measured,spatial
from audit_full51 import check_refs,query_objects,query_spatial,source_mass
MANIFEST='research/round8/experiment/full5912_gpu_inputs.json'
DEV_MANIFEST='research/round24/input/ALL336_PRIVATE.json'
SIX=('imagepath','pointpath','metapath','caption','platform','scene')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def path(s):
 p=(ROOT/s).resolve() if not Path(s).is_absolute() else Path(s).resolve()
 assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p)
 return p
def load(p,fields=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if fields is None else fields)}
def exact(a,b,name):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name

def reconstruct_groups_and_medoid(a):
 """Independent set arithmetic/greedy complete-link, without producer import."""
 boxes=a['source_boxes'];off=a['query_support_point_offsets'];flat=a['query_support_point_indices'];cc=a['query_component_counts']
 assert boxes.shape==(256,6) and off.shape==(257,) and off[0]==0 and off[-1]==len(flat) and (np.diff(off)>=0).all()
 bags=[np.sort(flat[off[q]:off[q+1]].astype(np.int64)) for q in range(256)]
 sets=[frozenset(map(int,b)) for b in bags];n=np.array([len(b) for b in bags]);assert all(len(s)==len(b) for s,b in zip(sets,bags))
 eligible=(n>=6)&(cc>=1)&(boxes[:,3:]>0).all(1)
 cache={}
 def pair(q,r):
  key=tuple(sorted((int(q),int(r))))
  if key not in cache:
   overlap=len(sets[q]&sets[r]);union=int(n[q]+n[r]-overlap)
   diff=np.abs(boxes[q,:3].astype(np.float64)-boxes[r,:3].astype(np.float64))
   centers=bool((boxes[q,3:]>0).all() and (boxes[r,3:]>0).all() and (diff<=boxes[q,3:].astype(np.float64)/2).all() and (diff<=boxes[r,3:].astype(np.float64)/2).all())
   J=overlap/union if union else 0.
   ok=bool(eligible[q] and eligible[r] and centers and J>=.7 and overlap/n[q]>=.75 and overlap/n[r]>=.75)
   cache[key]=(ok,J)
  return cache[key]
 order=sorted(np.flatnonzero(eligible),key=lambda q:(-int(n[q]),int(bags[q][0]),int(q)))
 groups=[];ids=np.full(256,-1,np.int64);matches=[[] for _ in range(256)]
 for q in order:
  available=[j for j,members in enumerate(groups) if all(pair(q,r)[0] for r in members)]
  matches[q]=available
  if len(available)>1:continue
  if available:g=available[0];groups[g].append(int(q))
  else:g=len(groups);groups.append([int(q)])
  ids[q]=g
 means=np.full(256,np.nan);medoids=[]
 for members in groups:
  ordered=sorted(members)
  for q in ordered:means[q]=float(np.mean(np.asarray([pair(q,r)[1] for r in ordered],np.float64)))
  top=max(means[q] for q in ordered);medoids.append(min(q for q in ordered if means[q]==top))
 return groups,ids,np.asarray(medoids,np.int64),means,bags,order,matches

def audit_carrier(a,gi,c):
 groups,ids,medoids,means,bags,order,matches=reconstruct_groups_and_medoid(a)
 assert [g['group_id'] for g in gi['groups']]==list(range(len(groups)))
 assert [g['members'] for g in gi['groups']]==groups and gi['query_group_ids']==ids.tolist()
 assert gi['construction_order']==list(map(int,order)) and gi['compatible_group_ids_at_insertion']==matches
 assert gi['semantic_identity_certified'] is False and gi['canonical_members_keep_original_bags_no_cross_group_union'] is True
 exact(c['query_group_ids'],ids,'independent complete-link query bindings')
 exact(c['group_medoid_queries'],medoids,'independent whole-member Jaccard medoids')
 exact(c['member_mean_point_jaccard'],means,'independent actual-point Jaccard means')
 W=np.zeros((256,len(groups)),np.float64);have=ids>=0;W[np.flatnonzero(have),ids[have]]=1.
 exact(c['query_instance_conditional_weights'],W,'one-hot/ambiguous-zero propagation')
 vis=a['visible_raw_point_indices'];pixels=a['visible_pixels_xy'];uv=a['visible_uv']
 assert len(np.unique(vis))==len(vis)
 for m,q in enumerate(medoids):
  lo,hi=c['instance_point_offsets'][m:m+2];exact(c['instance_point_indices'][lo:hi],bags[q],'actual medoid bag without union')
  chosen=np.isin(vis,bags[q],assume_unique=True);px=pixels[chosen]
  lo,hi=c['instance_visible_point_offsets'][m:m+2];exact(c['instance_visible_point_indices'][lo:hi],vis[chosen],'shared visible points retained')
  lo,hi=c['instance_crop_mask_pixel_offsets'][m:m+2];exact(c['instance_crop_mask_pixel_xy'][lo:hi],px,'actual sparse visible pixels')
  rect=np.r_[px.min(0),px.max(0)+1] if len(px) else np.zeros(4,np.int64)
  exact(c['instance_rect_xyxy'][m],rect,'sparse-pixel rectangle')
  center=uv[chosen].mean(0) if len(px) else np.full(2,np.nan)
  exact(c['instance_visible_centroid_uv'][m],center,'actual visible UV centroid')
 return len(groups)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_INDEPENDENT_FULL_GROUP_MEDOID_AUDIT_NO_GT'
 assert auth['plan_sha256']==sha(args.plan) and auth['program_sha256']==sha(__file__)
 assert plan['state']=='ROOT_LOCKED_R51_INDEPENDENT_FULL_GROUP_MEDOID_AUDIT' and plan['n']==5912 and plan['manifest']==MANIFEST
 assert plan['no_GT_models_or_optimizer'] is True and plan['selected_factor']==.5
 assert plan['rawsupport_recompute_scope']=='HASH_MIN_N_PER_PLATFORM' and plan['rawsupport_n_per_platform']==2 and plan['rawsupport_hash_salt']
 for kind in ('programs_sha256','inputs_sha256'):
  for f,h in plan[kind].items():assert sha(path(f))==h,('pin',f)
 for f in ('research/round51/audit_grouped_full51.py','research/round51/audit51.py','research/round51/audit_full51.py'):
  assert plan['programs_sha256'][f]==sha(ROOT/f)
 keys=['prediction_root','safe_root','noassociation_root','prepared_root','reference_root','r50_prediction_root','canonical_source_root']
 if plan.get('dev_overlap_bridge'):
  assert plan['inputs_sha256'][DEV_MANIFEST]==sha(ROOT/DEV_MANIFEST);keys.append('dev_grouped_root')
 roots={k:path(plan[k]) for k in keys};seals={k:read(v/'SEALED.json') for k,v in roots.items()}
 for k,v in roots.items():assert plan['inputs_sha256'][(v/'SEALED.json').relative_to(ROOT).as_posix()]==sha(v/'SEALED.json')
 assert auth['prediction_seal_sha256']==sha(roots['prediction_root']/'SEALED.json')
 roles=plan['arm_roles'];arms=plan['arm_names'];assert set(roles)=={'source','external','measured','no_relation','no_association'}
 assert len(arms)==5 and set(roles.values())==set(arms) and seals['prediction_root']['arm_names']==arms
 assert seals['prediction_root']['state']=='R51_SELECTED_GROUPED_FULL5912_FIVE_READOUTS_ASSEMBLED_NO_GT'
 assert seals['prediction_root']['n']==seals['safe_root']['n']==seals['noassociation_root']['n']==5912
 assert seals['safe_root']['state']=='R51_FULL_GROUP_MEDOID_G_PREDICTIONS_NO_GT'
 assert seals['safe_root']['appearance_factor_values']==seals['noassociation_root']['appearance_factor_values']==[.5]
 assert seals['safe_root']['external_vision_outputs_or_mask_derived_caches_read'] is False
 assert seals['noassociation_root']['external_vision_outputs_or_mask_derived_caches_read'] is False
 assert plan['canonical_source_root']=='crossplatform/round8/full5912_attempt001'
 rows=read(ROOT/MANIFEST);assert len(rows)==5912 and len({r['id'] for r in rows})==5912
 def hashkey(r):return hashlib.sha256((plan['rawsupport_hash_salt']+'\n'+json.dumps([r[k] for k in SIX],ensure_ascii=False,separators=(',',':'))).encode()).hexdigest()
 raw_ids={r['id'] for p in ('drone','quad') for r in sorted([r for r in rows if r['platform']==p],key=hashkey)[:2]};assert len(raw_ids)==4
 devlookup={}
 if plan.get('dev_overlap_bridge'):
  devrows=read(ROOT/DEV_MANIFEST);assert len(devrows)==336
  for r in devrows:
   key=tuple(r[k] for k in SIX);assert key not in devlookup;devlookup[key]=r['id']
 out=path(plan['output_directory']);assert not out.exists();out.mkdir(parents=True)
 dump(out/'RAW_SCOPE_SELECTION_PRIVATE.json',{'scope':'HASH_MIN_N_PER_PLATFORM','selected_ids_PRIVATE':sorted(raw_ids),'hash_salt':plan['rawsupport_hash_salt'],'n_per_platform':2})
 def member(k,rel):
  p=roots[k]/rel;assert sha(p)==seals[k]['files'][rel],('sealed member',k,rel);return p
 statuses=read(member('prediction_root','READOUT_STATUS_PRIVATE.json'));assert len(statuses)==5912
 status={r['id']:r['arm_statuses'] for r in statuses};assert set(status)=={r['id'] for r in rows}
 byid={r['id']:r for r in rows};allowed={'RAW_SAFE_INPUT_BINDINGS','RAW_SAFE_GLOBAL_VISIBILITY_SOURCE_ONLY','CANONICAL_SOURCE_ONLY_BRIDGE','PURE_TEXT_FIELDS_ONLY','ORIGINAL_RGB','ORIGINAL_RAW_XYZ','SAME_GROUNDER_REFERENCE_BINDINGS','REFERENCE_STAGE_ORIGINAL_R8_SUBJECT_BRIDGE','SAME_GROUNDER_REFERENCE_BOXES_LOGODDS','REFERENCE_SAMPLING_AND_MASK_RECEIPT'}
 for item in read(member('safe_root','INPUT_LEDGER_PRIVATE.json'))['files']:
  assert item['role'] in allowed
  p=path(item['path_PRIVATE']);sid=item['id_PRIVATE']
  if item['role'] in ('ORIGINAL_RGB','ORIGINAL_RAW_XYZ'):assert p==Path(byid[sid]['imagepath' if item['role']=='ORIGINAL_RGB' else 'pointpath']).resolve()
  else:assert any(p.is_relative_to(roots[k]) for k in ('prepared_root','reference_root','canonical_source_root'))
 na_allowed={'RAW_SAFE_BINDINGS','PURE_GRAPH_SCENE_COLOR','RAW_SAFE_VISIBLE_POINTS_SOURCE_ONLY','ORIGINAL_RGB','ORIGINAL_RAW_XYZ','SAME_MODEL_REFERENCE_INPUTS','SAME_MODEL_REFERENCE_BOXES_LOGODDS'}
 for item in read(member('noassociation_root','INPUT_LEDGER_PRIVATE.json'))['files']:
  assert item['role'] in na_allowed
  p=path(item['path_PRIVATE']);sid=item['id_PRIVATE']
  if item['role'] in ('ORIGINAL_RGB','ORIGINAL_RAW_XYZ'):assert p==Path(byid[sid]['imagepath' if item['role']=='ORIGINAL_RGB' else 'pointpath']).resolve()
  else:assert p.is_relative_to(roots['prepared_root']) or p.is_relative_to(roots['reference_root'])
 n=collections.Counter();start=time.perf_counter();devseen=set()
 try:
  for row in rows:
   sid=row['id'];assembled=load(member('prediction_root','samples_PRIVATE/'+sid+'.npz'))
   safe=load(member('safe_root','samples_PRIVATE/'+sid+'.npz'));na=load(member('noassociation_root','samples_PRIVATE/'+sid+'.npz'))
   fields=['source_boxes','source_scores','visible_raw_point_indices','visible_pixels_xy','visible_uv','query_support_point_offsets','query_support_point_indices','query_component_counts']
   prepared=load(member('prepared_root','arrays_PRIVATE/'+sid+'.npz'),fields)
   case=read(member('prepared_root','cases_PRIVATE/'+sid+'.json'));w,h=case['image_size_wh']
   carrier=load(member('safe_root','carriers_PRIVATE/'+sid+'.npz'));gi=read(member('safe_root','groups_PRIVATE/'+sid+'.json'))
   n['carrier_groups']+=audit_carrier(prepared,gi,carrier)
   attrs=read(member('safe_root','attributes_PRIVATE/'+sid+'.json'));graph=attrs['text_graph_unchanged'];assert case['pure_text_graph_sha256']==read(member('safe_root','cases_PRIVATE/'+sid+'.json'))['pure_text_graph_sha256']
   source=load(member('canonical_source_root',sid+'.npz'),['source_boxes_raw6','source_text_scores256','point_sample_sha256','raw_indices_sha256'])
   boxes=source['source_boxes_raw6'];s=source['source_text_scores256']
   for a in (assembled,safe,na,prepared):exact(a['source_boxes'],boxes,'R8 original boxes');exact(a['source_scores'],s,'R8 original native scores')
   external=load(member('r50_prediction_root','samples_PRIVATE/'+sid+'.npz'),['source_boxes','source_scores','arm_names','scores','wide_pool'])
   exact(external['source_boxes'],boxes,'same-source external boxes');exact(external['source_scores'],s,'same-source external scores')
   expected={'source':s.astype(float),'external':external['scores'][external['arm_names'].tolist().index(plan['r50_external_arm'])],
             'measured':safe['scores'][1],'no_relation':safe['no_relation_scores'][0],'no_association':na['scores'][0]}
   assert assembled['arm_names'].tolist()==arms and assembled['scores'].shape==(5,256) and np.isfinite(assembled['scores']).all()
   for key,f in expected.items():exact(assembled['scores'][arms.index(roles[key])],f,'five-arm exact copy '+key)
   for a in (assembled,safe,na):
    assert np.array_equal(a['queries'],a['scores'].argmax(1));exact(a['selected_original_boxes'],boxes[a['queries']],'selected original source boxes')
   refcase=read(member('reference_root','cases_PRIVATE/'+sid+'.json'));assert refcase['graph_sha256']==case['pure_text_graph_sha256']
   assert refcase['point_sample_sha256']==source['point_sample_sha256'].tobytes().hex() and refcase['point_indices_sha256']==source['raw_indices_sha256'].tobytes().hex()
   refinputs={}
   for ref in graph['references']:
    route=refcase['references'][ref['entity']];rp=member('reference_root',route['file']);assert sha(rp)==route['sha256']
    refinputs[ref['entity']]=(load(rp,['boxes','logodds']),route['mask']['reference_usable'])
   xyz=None
   if sid in raw_ids:
    pp=Path(row['pointpath']);assert sha(pp)==case['raw_bindings']['point_sha256']
    x=np.load(pp,allow_pickle=False) if pp.suffix=='.npy' else np.fromfile(pp,np.float32).reshape(-1,4)
    xyz=np.ascontiguousarray(x[:,:3],np.float32)
    if row['platform']=='drone':xyz[:,2]+=np.float32(1.8)
    else:assert row['platform']=='quad'
    n['raw_recomputed_frames']+=1
   records=read(member('safe_root','references_PRIVATE/'+sid+'.json'))['references_by_factor'];assert len(records)==1
   refstored=load(member('safe_root','references_PRIVATE/'+sid+'.npz'));check_refs(carrier,attrs,records[0],refstored,refinputs,xyz,.5)
   n['reference_distributions']+=len(refinputs)
   if xyz is not None:n['raw_reference_support_vectors']+=len(refinputs)
   e,ek=attributes(attrs,'T');exact(e,safe['instance_attribute_e'],'independent color extraction');assert np.array_equal(ek,safe['instance_attribute_known'])
   exact(carrier['query_instance_conditional_weights'],safe['query_instance_conditional_weights'],'frozen one-hot groups')
   pr,pk=spatial(carrier,graph,records[0],w,h)
   exact(pr,safe['raw_PR'][0],'faithful weighted/(1+total) relation');assert np.array_equal(pk,safe['raw_PR_known'][0])
   f,c,k,anchor,mass=measured(s,carrier['query_instance_conditional_weights'],.5*e,pr,ek,pk)
   for x,y,label in ((f,safe['scores'][1],'F'),(c,safe['components'][0],'VPR'),(anchor,safe['cue_anchors'][0],'anchors'),(mass,safe['cue_query_known_weight_mass'][0],'knownmass')):exact(x,y,label)
   assert np.array_equal(k,safe['cue_query_known'][0])
   npr=pr.copy();npr[:,1]=0.;npk=pk.copy();npk[:,1]=False
   nr,nc,*_=measured(s,carrier['query_instance_conditional_weights'],.5*e,npr,ek,npk)
   exact(nr,safe['no_relation_scores'][0],'noR fullscores');exact(nc[:2],c[:2],'noR does not change V/P');assert not nc[2].any()
   exact(safe['scores'][2],nr,'third safe arm noR');exact(safe['no_relation_selected_original_boxes'],boxes[safe['no_relation_queries']],'noR original boxes')
   count=na['query_visible_support_counts'];assert not (na['query_attribute_known']&(count<8)).any()
   objects=None
   if xyz is not None:
    objects,counts=query_objects(boxes,xyz[prepared['visible_raw_point_indices']],prepared['visible_pixels_xy'],w,h)
    exact(counts,count,'noassociation direct visible-point counts');n['raw_noassociation_queries']+=256
   nad=read(member('noassociation_root','references_PRIVATE/'+sid+'.json'));narefs={}
   for entity,d in nad.items():
    a,usable=refinputs[entity];valid=(a['boxes'][:,3:]>0).all(1)&np.isfinite(a['logodds'])
    if not usable:valid[:]=False
    p,u,other,keep=source_mass(a['logodds'],valid)
    assert d['source_probabilities']==p.tolist() and d['source_top3_queries_PRIVATE']==list(map(int,keep))
    assert d['source_unknown_probability']==u and d['source_truncated_probability']==other
    assert len(d['hypotheses_by_factor'])==1;entry=d['hypotheses_by_factor'][0];hs=entry['hypotheses']
    assert [z['instance_index'] for z in hs]==list(map(int,keep)) and abs(sum(z['probability'] for z in hs)+entry['unknown_probability']-1)<1e-12
    for z,q in zip(hs,keep):
     assert z['source_probability']==float(p[q])
     if xyz is not None:
      oo,_=query_objects(a['boxes'][[q]],xyz[prepared['visible_raw_point_indices']],prepared['visible_pixels_xy'],w,h)
      assert z['xy_norm2']==(oo[0]['xy'].tolist() if oo[0]['xy'] is not None else None) and z['rect']==oo[0]['rect'].tolist()
    narefs[entity]=hs
   npr,npk=na['raw_PR'][0],na['raw_PR_known'][0]
   if objects is not None:
    npr,npk=query_spatial(objects,graph,narefs);exact(npr,na['raw_PR'][0],'scoped noassociation relation');assert np.array_equal(npk,na['raw_PR_known'][0])
   nf,nc,nk,an,mn=measured(s,np.eye(256),.5*na['query_attribute_e'],npr,na['query_attribute_known'],npk)
   for x,y,label in ((nf,na['scores'][0],'noassociationF'),(nc,na['components'][0],'noassociationVPR'),(an,na['cue_anchors'][0],'noassociationanchors'),(mn,na['cue_query_known_weight_mass'][0],'noassociationknownmass')):exact(x,y,label)
   assert np.array_equal(nk,na['cue_query_known'][0])
   sc=read(member('safe_root','cases_PRIVATE/'+sid+'.json'));nac=read(member('noassociation_root','cases_PRIVATE/'+sid+'.json'))
   req={roles['source']:{'unknown':False,'fallback':False,'no_parameter_update':True},
        roles['external']:{'unknown':len(external['wide_pool'])==0,'fallback':len(external['wide_pool'])==0,'no_parameter_update':True},
        roles['measured']:sc['arm_statuses']['grouped_replace_all_c05'],roles['no_relation']:sc['arm_statuses']['grouped_no_relation_c05'],
        roles['no_association']:nac['arm_statuses'][na['arm_names'][0]]}
   assert status[sid]==req
   for st in status[sid].values():assert set(st)=={'unknown','fallback','no_parameter_update'} and all(type(v) is bool for v in st.values())
   if tuple(row[k] for k in SIX) in devlookup:
    did=devlookup[tuple(row[k] for k in SIX)];dev=load(member('dev_grouped_root','samples_PRIVATE/'+did+'.npz'))
    for key in ('source_boxes','source_scores','scores','queries','selected_original_boxes','components','raw_PR','raw_PR_known','cue_query_known','cue_anchors','cue_query_known_weight_mass','instance_attribute_e','instance_attribute_known','query_instance_conditional_weights','group_medoid_queries'):
     exact(safe[key],dev[key],'six-field DEV overlap exact '+key)
    devseen.add(did)
   n['n']+=1;n['assembled_query_checks']+=5
   if n['n']%128==0:print('AUDIT_GROUPED_FULL51',n['n'],5912,flush=True)
  assert n['raw_recomputed_frames']==4
  if plan.get('dev_overlap_bridge'):assert devseen==set(devlookup.values())
  result={'state':'PASS_R51_INDEPENDENT_FULL_GROUP_MEDOID_AUDIT_NO_GT','n':5912,'arm_names':arms,
   'prediction_seal_sha256':sha(roots['prediction_root']/'SEALED.json'),'all_rows_groups_medoid_visiblecarriers_fusion_originalboxes_verified':True,
   'canonical_source_256_boxes_and_readout_verified':True,
   'all_unknown_fallback_and_zero_update_rows_retained':True,
   'rawsupport_recompute_scope':'HASH_MIN_N_PER_PLATFORM','rawsupport_recomputed_frames':4,
   'raw_noassociation_geometry_all_rows_verified':False,'all_reference_support_logits_recomputed':False,
   'RGB_measurement_limit':'sealed raw-RGB descriptors reused; independent attribute extraction/fusion only; no second RGB measurement',
   'dev_six_field_exact_overlap_rows':len(devseen),'no_semantic_or_foreground_identity_certification':True,
   'counts':dict(n),'seconds':time.perf_counter()-start,'GT_models_optimizer_used':False,'plan_sha256':sha(args.plan),'program_sha256':sha(__file__)}
  dump(out/'AUDIT_RESULT.json',result);dump(out/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
 except BaseException as error:
  dump(out/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'completed_n':n['n'],'automatic_retry':False,'GT_read':False});raise

if __name__=='__main__':main()
