"""Independent full R51 post-seal audit draft; no GT, models or optimizer.

Uses independent DEV auditor arithmetic, never producer evidence/readout code.
Original raw XYZ is permitted only for reference support/query geometry checks;
RGB color measurements are reused from sealed preparation, not remeasured.
Root must issue a separate actual full-audit grant before any data access.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
# Independent prior auditor, not a producer. Its main() never runs on import.
from audit51 import attributes,measured,spatial

MANIFEST='research/round8/experiment/full5912_gpu_inputs.json'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def path(s):
 p=(ROOT/s).resolve() if not Path(s).is_absolute() else Path(s).resolve()
 assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p)
 return p
def load(p,fields=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if fields is None else fields)}
def exact(a,b,name):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name

def source_mass(z,valid):
 mass=np.zeros(len(z),float);u=1.;keep=[];other=0.
 if valid.any():
  mx=max(0.,float(z[valid].max()));mass[valid]=np.exp(np.clip(z[valid]-mx,-700,0));u=np.exp(-mx);den=u+mass.sum();mass/=den;u=float(u/den)
  keep=sorted(np.flatnonzero(valid),key=lambda j:(-float(mass[j]),int(j)))[:3]
  other=float(max(0.,1-u-sum(mass[j] for j in keep)))
 return mass,u,other,keep

def ref_logits(raw_model,carrier,boxes,z,usable):
 result=np.full(len(carrier['instance_rect_xyxy']),-np.inf)
 if not usable:return result
 for j in range(len(result)):
  lo,hi=carrier['instance_point_offsets'][j:j+2];ids=carrier['instance_point_indices'][lo:hi]
  if len(ids)<6:continue
  points=raw_model[ids]
  inside=(np.abs(points[None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(-1)
  count=inside.sum(1);qualified=(count>=3)&(count/max(len(ids),1)>=.15)&(boxes[:,3:]>0).all(1)
  if qualified.any():result[j]=np.max(z[qualified])
 return result

def check_refs(carrier,attrs,records,saved,refinputs,model_raw,factor):
 for entity,entry in records.items():
  proposals,usable=refinputs[entity]
  z=saved[entity+'_source_logits']
  if model_raw is not None:
   recomputed=ref_logits(model_raw,carrier,proposals['boxes'],proposals['logodds'],usable)
   exact(recomputed,z,'independent scoped-instance ref point-support logits')
  p,u,other,keep=source_mass(z,np.isfinite(z));exact(p,saved[entity+'_source_probability'],'source reference mass')
  d=entry['distribution'];assert d['source_top3_instance_ids']==list(map(int,keep))
  assert d['source_unknown_probability']==u and d['source_truncated_probability']==other and d['source_probabilities']==p.tolist()
  av,ak=attributes(attrs,entity)
  weights=np.array([p[j]*np.exp(np.clip(factor*av[j],-8,8)) if ak[j] else p[j] for j in keep]);den=u+other+float(weights.sum());weights/=den
  hs=entry['hypotheses'];assert [h['instance_index'] for h in hs]==list(map(int,keep))
  assert [h['probability'] for h in hs]==list(map(float,weights))
  assert d['unknown_probability']==float((u+other)/den) and d['retained_probability']==float(weights.sum())
  for h in hs:
   j=h['instance_index'];assert h['source_probability']==float(p[j]) and h['uncertainty_mass_explicit'] is True
   assert h['rect']==carrier['instance_rect_xyxy'][j].tolist()
   xy=carrier['instance_visible_centroid_uv'][j];expected=xy.tolist() if np.isfinite(xy).all() else None
   assert h['xy_pixel']==expected

def query_objects(boxes,visible,px,W,H):
 objects=[];counts=[]
 for q,b in enumerate(boxes):
  have=(np.abs(visible.astype(np.float64)-b[:3])<=b[3:]/2).all(1) if (b[3:]>0).all() else np.zeros(len(px),bool)
  xy=px[have];defined=bool(len(xy));rect=np.r_[xy.min(0),xy.max(0)+1] if defined else np.zeros(4)
  objects.append({'visible':defined,'xy':(xy.mean(0)+.5)/np.array([W,H]) if defined else None,
    'rect':rect,'border':bool(defined and ((xy[:,0]==0)|(xy[:,1]==0)|(xy[:,0]==W-1)|(xy[:,1]==H-1)).any())})
  counts.append(len(xy))
 return objects,np.asarray(counts)

def query_spatial(objects,graph,references):
 # Independent scalar transcription of original R23 equations. Geometry was
 # just reconstructed from actual raw-visible support, not stored producer PR.
 pr=np.zeros((256,2));known=np.zeros((256,2),bool)
 for i,obj in enumerate(objects):
  if not obj['visible'] or obj['border']:continue
  x=float(obj['xy'][0]);glob=[];rel=[]
  for r in graph.get('global_relations',[]):
   if r.get('frame')!='camera_image':continue
   p=r['predicate']
   if p in ('GLOBAL_LEFT','LEFT_OF_CENTER'):glob.append(float(-np.tanh((x-.5)/.15)))
   elif p in ('GLOBAL_RIGHT','RIGHT_OF_CENTER'):glob.append(float(np.tanh((x-.5)/.15)))
   elif p=='GLOBAL_CENTER':glob.append(1-4*abs(x-.5))
  if glob:pr[i,0]=np.mean(glob);known[i,0]=True
  for r in graph.get('reference_relations',[]):
   if r.get('frame')!='camera_image' or r.get('predicate') not in ('LEFT_OF','RIGHT_OF'):continue
   weighted=0.;total=0.
   for h in references.get(r['subject'],[]):
    if h.get('xy_norm2') is None:continue
    a=np.asarray(h['rect'],float);b=np.asarray(obj['rect'],float)
    overlap=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    area=(a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-overlap
    if overlap/max(area,1e-12)>=.3:continue
    w=float(h['probability'])
    if w<=0:continue
    direction=1 if r['predicate']=='RIGHT_OF' else -1
    weighted+=w*float(np.tanh(direction*(h['xy_norm2'][0]-x)/(.15+.1*h.get('width_norm',0.))));total+=w
   if total>0:rel.append(weighted/(1+total));known[i,1]=True
  if rel:pr[i,1]=np.mean(rel)
 return pr,known

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_INDEPENDENT_FULL_PREDICTION_AUDIT_NO_GT' and auth['plan_sha256']==sha(args.plan) and auth['program_sha256']==sha(__file__)
 assert plan['state']=='ROOT_LOCKED_R51_INDEPENDENT_FULL_PREDICTION_AUDIT' and plan['n']==5912 and plan['manifest']==MANIFEST
 assert plan['no_GT_models_or_optimizer'] is True
 scope=plan['rawsupport_recompute_scope'];assert scope in ('ALL','HASH_MIN_N_PER_PLATFORM')
 if scope=='HASH_MIN_N_PER_PLATFORM':
  assert plan['rawsupport_n_per_platform']==2 and isinstance(plan['rawsupport_hash_salt'],str) and plan['rawsupport_hash_salt']
 factor=plan['selected_factor'];assert factor in (.5,1.,2.)
 roles=plan['arm_roles'];assert set(roles)=={'source','external','measured','no_relation','no_association'}
 arms=plan['arm_names'];assert len(arms)==5 and set(roles.values())==set(arms)
 for group in ('programs_sha256','inputs_sha256'):
  for f,h in plan[group].items():assert sha(path(f))==h,('pin',f)
 for f in ('research/round51/audit_full51.py','research/round51/audit51.py'):
  assert plan['programs_sha256'][f]==sha(ROOT/f)
 keys=('prediction_root','safe_root','noassociation_root','prepared_root','reference_root','r50_prediction_root','canonical_source_root')
 roots={k:path(plan[k]) for k in keys};seals={k:read(v/'SEALED.json') for k,v in roots.items()}
 for k,v in roots.items():assert plan['inputs_sha256'][(v/'SEALED.json').relative_to(ROOT).as_posix()]==sha(v/'SEALED.json')
 assert auth['prediction_seal_sha256']==sha(roots['prediction_root']/'SEALED.json')
 assert seals['prediction_root']['arm_names']==arms and seals['prediction_root']['n']==5912
 assert seals['safe_root']['appearance_factor_values']==seals['noassociation_root']['appearance_factor_values']==[factor]
 assert seals['safe_root']['external_vision_outputs_or_mask_derived_caches_read'] is False
 assert seals['noassociation_root']['external_vision_outputs_or_mask_derived_caches_read'] is False
 assert plan['canonical_source_root']=='crossplatform/round8/full5912_attempt001'
 rows=read(ROOT/MANIFEST);assert len(rows)==5912 and len({r['id'] for r in rows})==5912
 if scope=='ALL':raw_ids={r['id'] for r in rows}
 else:
  def selection_key(r):
   key=[r[k] for k in ('imagepath','pointpath','metapath','caption','platform','scene')]
   return hashlib.sha256((plan['rawsupport_hash_salt']+'\n'+json.dumps(key,ensure_ascii=False,separators=(',',':'))).encode()).hexdigest()
  raw_ids={r['id'] for platform in ('drone','quad') for r in sorted([x for x in rows if x['platform']==platform],key=selection_key)[:2]}
  assert len(raw_ids)==4
 safeplan=read(path(plan['safe_prediction_plan']))
 assert set(safeplan['inputs_sha256'])=={MANIFEST,safeplan['prepared_root']+'/SEALED.json',safeplan['reference_root']+'/SEALED.json'}
 assert safeplan['appearance_factor_values']==[factor] and safeplan['no_external_vision_cache_inputs'] is True
 out=path(plan['output_directory']);assert not out.exists();out.mkdir(parents=True)
 dump(out/'RAW_SCOPE_SELECTION_PRIVATE.json',{'scope':scope,'selected_ids_PRIVATE':sorted(raw_ids),
     'hash_salt':plan.get('rawsupport_hash_salt'),'selected_before_prediction_arrays':True})
 def member(key,rel):
  p=roots[key]/rel;assert sha(p)==seals[key]['files'][rel];return p
 statuses=read(member('prediction_root','READOUT_STATUS_PRIVATE.json'));assert len(statuses)==5912 and len({r['id'] for r in statuses})==5912
 status={r['id']:r['arm_statuses'] for r in statuses};assert set(status)=={r['id'] for r in rows}
 byid={r['id']:r for r in rows}
 safe_roles={'REGISTERED_RAW_INPUTS','PREPARED_ROOT_SEAL','REFERENCE_ROOT_SEAL','RAW_SAFE_POINT_INSTANCES',
  'RAW_SAFE_BINDINGS','MEASURED_RGB_AND_PURE_TEXT','ORIGINAL_RAW_XYZ_FOR_REFERENCE_SUPPORT',
  'SAME_MODEL_REFERENCE_GRAPH_BINDINGS','ORIGINAL_R8_SUBJECT_SOURCE_BRIDGE_ONLY',
  'SAME_GROUNDER_REFERENCE_HEADSPAN_ONLY','REFERENCE_SAMPLING_AND_MASK_RECEIPT'}
 for item in read(member('safe_root','INPUT_LEDGER_PRIVATE.json'))['items']:
  assert item['role'] in safe_roles
  p=path(item['path_PRIVATE'])
  if item['role']=='ORIGINAL_RAW_XYZ_FOR_REFERENCE_SUPPORT':assert p==Path(byid[item['id_PRIVATE']]['pointpath']).resolve()
  else:assert p.is_relative_to(roots['prepared_root']) or p.is_relative_to(roots['reference_root']) or p==ROOT/MANIFEST
 na_roles={'RAW_SAFE_BINDINGS','PURE_GRAPH_SCENE_COLOR','RAW_SAFE_VISIBLE_POINTS_SOURCE_ONLY','ORIGINAL_RGB','ORIGINAL_RAW_XYZ','SAME_MODEL_REFERENCE_INPUTS','SAME_MODEL_REFERENCE_BOXES_LOGODDS'}
 for item in read(member('noassociation_root','INPUT_LEDGER_PRIVATE.json'))['files']:
  assert item['role'] in na_roles
  p=path(item['path_PRIVATE'])
  if item['role'] in ('ORIGINAL_RGB','ORIGINAL_RAW_XYZ'):
   field='imagepath' if item['role']=='ORIGINAL_RGB' else 'pointpath'
   assert p==Path(byid[item['id_PRIVATE']][field]).resolve()
  else:assert p.is_relative_to(roots['prepared_root']) or p.is_relative_to(roots['reference_root'])
 count={'n':0,'query_checks':0,'reference_distributions':0,'actual_ref_support_vectors':0,'noassociation_query_geometry':0,'raw_scope_frames':0};start=time.perf_counter()
 try:
  for row in rows:
   sid=row['id'];p=load(member('prediction_root','samples_PRIVATE/'+sid+'.npz'))
   safe=load(member('safe_root','samples_PRIVATE/'+sid+'.npz'));na=load(member('noassociation_root','samples_PRIVATE/'+sid+'.npz'))
   raw=load(member('prepared_root','arrays_PRIVATE/'+sid+'.npz'));attrs=read(member('prepared_root','attributes_PRIVATE/'+sid+'.json'))
   case=read(member('prepared_root','cases_PRIVATE/'+sid+'.json'));W,H=case['image_size_wh'];graph=attrs['text_graph_unchanged']
   with np.load(member('canonical_source_root',sid+'.npz'),allow_pickle=False) as original:
    boxes=original['source_boxes_raw6'].copy();s=original['source_text_scores256'].copy()
    original_point_sample_sha256=original['point_sample_sha256'].tobytes().hex()
    original_raw_indices_sha256=original['raw_indices_sha256'].tobytes().hex()
   for a in (p,safe,na,raw):exact(a['source_boxes'],boxes,'originalR8boxes');exact(a['source_scores'],s,'originalR8scores')
   baseline=load(member('r50_prediction_root','samples_PRIVATE/'+sid+'.npz'),['arm_names','scores','queries','source_boxes','source_scores','wide_pool'])
   exact(baseline['source_boxes'],boxes,'external same source pool');exact(baseline['source_scores'],s,'external same source score')
   assert p['arm_names'].tolist()==arms and p['scores'].shape==(5,256) and np.isfinite(p['scores']).all()
   assert np.array_equal(p['queries'],p['scores'].argmax(1))
   exact(p['selected_original_boxes'],boxes[p['queries']],'assembled original selected boxes')
   role_index={k:arms.index(v) for k,v in roles.items()}
   external_index=baseline['arm_names'].tolist().index(plan['r50_external_arm'])
   expected={'source':baseline['scores'][0],'external':baseline['scores'][external_index],
             'measured':safe['scores'][1],'no_relation':safe['no_relation_scores'][0],'no_association':na['scores'][0]}
   for k,v in expected.items():exact(p['scores'][role_index[k]],v,'exactcopy '+k)
   xyz=None
   if sid in raw_ids:
    pointpath=Path(row['pointpath']);assert sha(pointpath)==case['raw_bindings']['point_sha256']
    xyz=np.load(pointpath,allow_pickle=False) if pointpath.suffix=='.npy' else np.fromfile(pointpath,np.float32).reshape(-1,4)
    xyz=np.ascontiguousarray(xyz[:,:3],np.float32)
    if row['platform']=='drone':xyz[:,2]+=np.float32(1.8)
    else:assert row['platform']=='quad'
    count['raw_scope_frames']+=1
   refcase=read(member('reference_root','cases_PRIVATE/'+sid+'.json'));assert refcase['graph_sha256']==case['pure_text_graph_sha256']
   assert refcase['point_sample_sha256']==original_point_sample_sha256 and refcase['point_indices_sha256']==original_raw_indices_sha256
   refinputs={}
   for ref in graph['references']:
    entry=refcase['references'][ref['entity']];rp=member('reference_root',entry['file']);assert sha(rp)==entry['sha256']
    refinputs[ref['entity']]=(load(rp,['boxes','logodds']),entry['mask']['reference_usable'])
   records=read(member('safe_root','references_PRIVATE/'+sid+'.json'))['references_by_factor'];assert len(records)==1
   stored=load(member('safe_root','references_PRIVATE/'+sid+'.npz'))
   check_refs(raw,attrs,records[0],stored,refinputs,xyz,factor)
   count['reference_distributions']+=len(refinputs)
   if xyz is not None:count['actual_ref_support_vectors']+=len(refinputs)
   e,ek=attributes(attrs,'T');exact(e,safe['instance_attribute_e'],'independent measured subject color extraction');assert np.array_equal(ek,safe['instance_attribute_known'])
   exact(safe['query_instance_conditional_weights'],raw['query_instance_conditional_weights'],'original conditional association')
   pr,pk=spatial(raw,graph,records[0],W,H)
   exact(pr,safe['raw_PR'][0],'independent main faithful spatial');assert np.array_equal(pk,safe['raw_PR_known'][0])
   f,c,k,a,m=measured(s,raw['query_instance_conditional_weights'],factor*e,pr,ek,pk)
   for x,y,name in ((f,safe['scores'][1],'safeF'),(c,safe['components'][0],'safecues'),(a,safe['cue_anchors'][0],'safeanchors'),(m,safe['cue_query_known_weight_mass'][0],'safeknownmass')):exact(x,y,name)
   assert np.array_equal(k,safe['cue_query_known'][0])
   npr=pr.copy();npr[:,1]=0.;npk=pk.copy();npk[:,1]=False
   nr,*_=measured(s,raw['query_instance_conditional_weights'],factor*e,npr,ek,npk);exact(nr,safe['no_relation_scores'][0],'noR')
   assert np.array_equal(safe['no_relation_queries'],safe['no_relation_scores'].argmax(1))
   exact(safe['no_relation_selected_original_boxes'],boxes[safe['no_relation_queries']],'noR selected original boxes')
   # Independently recover subject-query visible geometry for no-association.
   objects=None
   if xyz is not None:
    visible=xyz[raw['visible_raw_point_indices']];pixels=raw['visible_pixels_xy']
    objects,support=query_objects(boxes,visible,pixels,W,H);exact(support,na['query_visible_support_counts'],'direct-query visible counts')
    count['noassociation_query_geometry']+=256
   else:support=na['query_visible_support_counts']
   assert not (na['query_attribute_known']&(support<8)).any(),'unsupported query color was known'
   nad=read(member('noassociation_root','references_PRIVATE/'+sid+'.json'));nrrefs={}
   for entity,d in nad.items():
    proposals,usable=refinputs[entity];valid=(proposals['boxes'][:,3:]>0).all(1)&np.isfinite(proposals['logodds'])
    if not usable:valid[:]=False
    mass,u,other,keep=source_mass(proposals['logodds'],valid)
    assert d['source_probabilities']==mass.tolist() and d['source_top3_queries_PRIVATE']==list(map(int,keep))
    assert d['source_unknown_probability']==u and d['source_truncated_probability']==other
    assert len(d['hypotheses_by_factor'])==1
    hh=d['hypotheses_by_factor'][0]['hypotheses'];assert [h['instance_index'] for h in hh]==list(map(int,keep))
    assert abs(sum(h['probability'] for h in hh)+d['hypotheses_by_factor'][0]['unknown_probability']-1)<1e-12
    for h,q in zip(hh,keep):
     assert h['source_probability']==float(mass[q])
     if xyz is not None:
      obj,_=query_objects(proposals['boxes'][[q]],visible,pixels,W,H)
      expectedxy=obj[0]['xy'].tolist() if obj[0]['xy'] is not None else None
      assert h['xy_norm2']==expectedxy and h['rect']==obj[0]['rect'].tolist()
    nrrefs[entity]=hh
   npr,npk=na['raw_PR'][0],na['raw_PR_known'][0]
   if objects is not None:
    npr,npk=query_spatial(objects,graph,nrrefs)
    exact(npr,na['raw_PR'][0],'noassociation independently recomputed scoped spatial');assert np.array_equal(npk,na['raw_PR_known'][0])
   nf,nc,nk,an,mn=measured(s,np.eye(256),factor*na['query_attribute_e'],npr,na['query_attribute_known'],npk)
   exact(nf,na['scores'][0],'noassociationF');exact(nc,na['components'][0],'noassociationcues');exact(an,na['cue_anchors'][0],'noassocanchors');exact(mn,na['cue_query_known_weight_mass'][0],'noassocmass');assert np.array_equal(nk,na['cue_query_known'][0])
   for z in (safe,na):assert np.array_equal(z['queries'],z['scores'].argmax(1));exact(z['selected_original_boxes'],boxes[z['queries']],'safe/noassoc selected')
   scase=read(member('safe_root','cases_PRIVATE/'+sid+'.json'));ncase=read(member('noassociation_root','cases_PRIVATE/'+sid+'.json'))
   assert status[sid][roles['measured']]==scase['arm_statuses'][safe['arm_names'][1]]
   assert status[sid][roles['no_association']]==ncase['arm_statuses'][na['arm_names'][0]]
   nr_unknown=not bool(safe['cue_query_known'][0,:2].any());external_empty=len(baseline['wide_pool'])==0
   assert status[sid][roles['source']]=={'unknown':False,'fallback':False,'no_parameter_update':True}
   assert status[sid][roles['external']]=={'unknown':external_empty,'fallback':external_empty,'no_parameter_update':True}
   assert status[sid][roles['no_relation']]=={'unknown':nr_unknown,'fallback':nr_unknown,'no_parameter_update':True}
   for st in status[sid].values():assert set(st)=={'unknown','fallback','no_parameter_update'} and all(type(v) is bool for v in st.values())
   count['n']+=1;count['query_checks']+=5
   if count['n']%128==0:print('AUDIT_FULL51',count['n'],5912,flush=True)
  assert count['raw_scope_frames']==len(raw_ids)
  result={'state':'PASS_R51_INDEPENDENT_FULL_PREDICTION_AUDIT_NO_GT','n':5912,'arm_names':arms,
    'prediction_seal_sha256':sha(roots['prediction_root']/'SEALED.json'),
    'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,
    'source_external_exact_bridges':5912,'selected_factor':factor,'counts':count,
    'all5912_source_distribution_top3_fusion_selected_original_boxes_verified':True,
    'rawsupport_recompute_scope':scope,'rawsupport_recomputed_frames':len(raw_ids),
    'all_new_reference_source_point_support_logits_independently_recomputed':scope=='ALL',
    'noassociation_visible_query_geometry_independently_recomputed_all':scope=='ALL',
    'noassociation_spatial_scope':'raw scoped frames independently recomputed; others sealed PR with independent fusion',
    'color_measurement_limit':'sealed pixel color measurements reused; extraction/fusion audited, no second RGB measurement',
    'GT_or_models_or_optimizer_used':False,'seconds':time.perf_counter()-start,
    'plan_sha256':sha(args.plan),'program_sha256':sha(__file__)}
  dump(out/'AUDIT_RESULT.json',result);dump(out/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
 except BaseException as error:
  dump(out/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'completed_n':count['n'],'GT_read':False,'automatic_retry':False});raise

if __name__=='__main__':main()
