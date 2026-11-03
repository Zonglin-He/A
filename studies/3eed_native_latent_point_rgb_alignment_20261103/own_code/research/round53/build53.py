"""Fixed soft-support association comparison; all predictions precede GT."""
import argparse,hashlib,json,sys,time,copy
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'crossplatform/round51'))
from support53 import pixel_descriptors,entity_masks,seed_weights,local_lift,weighted_measurement,permutation
from evidence51_fidelity import attr_evidence,raw_relations,propagate_measured,object_geometry
from noassociation51 import pixel_color_record
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (keys or z.files)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();plan=read(a.plan)
 for rel,h in {**plan['programs_sha256'],**plan['inputs_sha256']}.items():assert sha(ROOT/rel)==h,rel
 rows=read(ROOT/plan['manifest']);arms=plan['arms'];out=ROOT/plan['output_root'];assert not out.exists();out.mkdir(parents=True)
 for name in ['samples_PRIVATE','supports_PRIVATE','cases_PRIVATE']:(out/name).mkdir()
 roots={k:ROOT/plan[k] for k in ['latent_root','grouped_root','prepared_root','external_root']};seals={k:read(v/'SEALED.json') for k,v in roots.items()}
 def member(k,rel,keys=None):
  f=roots[k]/rel;assert sha(f)==seals[k]['files'][rel],(k,rel)
  return load(f,keys) if rel.endswith('.npz') else read(f)
 stats=[];statuses=[];start=time.perf_counter();media={};metadata={}
 for row in rows:
  sid=row['id'];tick=time.perf_counter();g=member('grouped_root','samples_PRIVATE/'+sid+'.npz');carrier=member('grouped_root','carriers_PRIVATE/'+sid+'.npz');attrs=member('grouped_root','attributes_PRIVATE/'+sid+'.json');references=member('grouped_root','references_PRIVATE/'+sid+'.json')['references_by_factor'][0]
  lat=member('latent_root','samples_PRIVATE/'+sid+'.npz');lc=member('latent_root','cases_PRIVATE/'+sid+'.json');prepared=member('prepared_root','arrays_PRIVATE/'+sid+'.npz',['visible_raw_point_indices','visible_pixels_xy','visible_uv']);pc=member('prepared_root','cases_PRIVATE/'+sid+'.json')
  assert lat['boxes'].tobytes()==g['source_boxes'].tobytes() and lat['scores'].tobytes()==g['source_scores'].tobytes()
  keys=(row['imagepath'],row['pointpath']);bindings=pc['raw_bindings']
  if keys not in media:
   assert sha(row['imagepath'])==bindings['image_sha256'] and sha(row['pointpath'])==bindings['point_sha256']
   with Image.open(row['imagepath']) as im:rgb=np.array(im.convert('RGB'))
   f=Path(row['pointpath']);raw=np.load(f,allow_pickle=False) if f.suffix=='.npy' else np.fromfile(f,np.float32).reshape(-1,4);raw=np.ascontiguousarray(raw[:,:3],np.float32)
   media={keys:(rgb,raw)}
  rgb,raw=media[keys];W,H=pc['image_size_wh'];assert rgb.shape[:2]==(H,W)
  offsets=np.asarray(lc['token_offsets_PRIVATE']);masks,mask_receipt=entity_masks(attrs['text_graph_unchanged'],row['caption'],offsets);entities=list(masks);M=len(carrier['group_medoid_queries']);baseobjects=object_geometry(carrier,(W,H))
  allpx=prepared['visible_pixels_xy'];scene=attrs['scene_visible_color_soft_support'];nscene=attrs['n_scene_visible_pixels'];assert nscene==len(allpx)
  idxmap={int(r):j for j,r in enumerate(prepared['visible_raw_point_indices'])};observations={v:{e:[] for e in entities} for v in ['pre','native','shuffle']};objects={v:[] for v in observations};refgeom={v:{e:[] for e in entities} for v in observations};support_archive={};case_counts={'no_seed_groups':0,'anchor_outside_box_groups':0,'geometry_descriptor_max_error':0.,'groups':M}
  for m,medoid in enumerate(carrier['group_medoid_queries']):
   lo,hi=carrier['instance_crop_mask_pixel_offsets'][m:m+2];px=carrier['instance_crop_mask_pixel_xy'][lo:hi];vlo,vhi=carrier['instance_visible_point_offsets'][m:m+2];ids=carrier['instance_visible_point_indices'][vlo:vhi];assert len(ids)==len(px)
   visibility_positions=np.asarray([idxmap[int(r)] for r in ids],np.int64);assert np.array_equal(prepared['visible_pixels_xy'][visibility_positions],px);uv=prepared['visible_uv'][visibility_positions]
   xyz=raw[ids].copy()
   if row['platform']=='drone':xyz[:,2]+=np.float32(1.8)
   desc=pixel_descriptors(rgb[px[:,1],px[:,0]]);uniform=weighted_measurement(desc,np.ones(len(px)),scene,nscene,m)
   for name,val in uniform['color_logodds_relative_scene_visible'].items():
    old=attrs['instances'][m]['color_logodds_relative_scene_visible'][name]
    assert (val is None)==(old is None)
    if val is not None:case_counts['geometry_descriptor_max_error']=max(case_counts['geometry_descriptor_max_error'],abs(val-old));assert abs(val-old)<1e-11
   nn,dist=local_lift(lat['fp2_xyz'],xyz,g['source_boxes'][int(medoid)]);pre,semantic=seed_weights(lat,masks,int(medoid));anchor=lat['fp2_xyz'][int(lat['query_points_sample_inds'][int(medoid)])];case_counts['anchor_outside_box_groups']+=int(not (np.abs(anchor-g['source_boxes'][int(medoid),:3])<=g['source_boxes'][int(medoid),3:]/2).all())
   case_counts['no_seed_groups']+=nn is None;perm=permutation(sid,int(medoid))
   for v in observations:
    weights={}
    for ent in entities:
     invalid=nn is None or (v in ['native','shuffle'] and semantic[ent] is None)
     if invalid:w=np.ones(len(px));item=copy.deepcopy(attrs['instances'][m])
     else:
      sw=pre if v=='pre' else semantic[ent] if v=='native' else semantic[ent][perm];w=sw[nn];item=weighted_measurement(desc,w,scene,nscene,m)
     observations[v][ent].append(item);weights[ent]=w
     if len(px) and w.sum()>0:pos=np.average(px.astype(float),axis=0,weights=w);u=np.average(uv.astype(float),axis=0,weights=w)
     else:pos=u=None
     refgeom[v][ent].append(u)
     support_archive[v+'__'+ent+'__'+str(m)]=w
    obj=copy.deepcopy(baseobjects[m]);wt=weights['T']
    if len(px) and wt.sum()>0:obj['xy']=(np.average(px.astype(float),axis=0,weights=wt)+.5)/[W,H]
    objects[v].append(obj)
   support_archive['nearest_seed__'+str(m)]=nn if nn is not None else np.empty(0,np.int64)
   support_archive['nearest_distance__'+str(m)]=dist if dist is not None else np.empty(0)
  source=g['source_scores'];scores={'source':source.astype(float)};cues={};known={};refs_by_variant={}
  scores['external_G']=member('external_root','samples_PRIVATE/'+sid+'.npz',['scores','arm_names'])
  ex=scores['external_G'];scores['external_G']=ex['scores'][ex['arm_names'].tolist().index('external_G')]
  for name,subset in [('geometry_full','VPR'),('geometry_V','V'),('geometry_VR','VR')]:
   f=source.astype(float)
   for i,c in enumerate('VPR'):
    if c in subset:f=f+2*g['components'][0,i]
   scores[name]=f;known[name]=g['cue_query_known'][0,[i for i,c in enumerate('VPR') if c in subset]].any()
  for v in observations:
   measured={ent:{**attrs,'instances':observations[v][ent]} for ent in entities};e,ek=attr_evidence(measured['T'],'T');refs={}
   for ent,info in references.items():
    distribution=info['distribution'];keep=distribution['source_top3_instance_ids'];mass=np.asarray(distribution.get('source_probabilities',[0.]*M));ae,ak=attr_evidence(measured[ent],ent)
    adjusted=np.array([mass[j]*np.exp(np.clip(.5*ae[j],-8,8)) if ak[j] else mass[j] for j in keep]);den=distribution['source_unknown_probability']+distribution['source_truncated_probability']+adjusted.sum()
    assert den>0;adjusted/=den;hs=[]
    oldhs={h['instance_index']:h for h in info['hypotheses']}
    for j,prob in zip(keep,adjusted):
     h=copy.deepcopy(oldhs[j]);h['probability']=float(prob);point=refgeom[v][ent][j];h['xy_norm2']=(point/np.array([W,H])).tolist() if point is not None else None;hs.append(h)
    refs[ent]=hs
   pr,pk=raw_relations(attrs['text_graph_unchanged'],objects[v],refs,(W,H));full,cs,ki=propagate_measured(source,g['query_instance_conditional_weights'],.5*e,pr,ek,pk);cues[v]=cs;refs_by_variant[v]=refs
   for suffix,subset in [('full','VPR'),('V','V'),('VR','VR')]:
    f=source.astype(float)
    for i,c in enumerate('VPR'):
     if c in subset:f=f+2*cs[i]
    name=v+'_'+suffix;scores[name]=f;known[name]=ki['query_known'][[i for i,c in enumerate('VPR') if c in subset]].any()
  stack=np.stack([scores[arm] for arm in arms]);q=stack.argmax(1).astype(np.int64)
  assert np.array_equal(stack[arms.index('geometry_full')],g['scores'][g['arm_names'].tolist().index('grouped_replace_all_c05')])
  np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.asarray(arms),scores=stack,queries=q,selected_original_boxes=g['source_boxes'][q],source_boxes=g['source_boxes'],source_scores=source,components=np.stack([cues[v] for v in ['pre','native','shuffle']]))
  np.savez_compressed(out/'supports_PRIVATE'/(sid+'.npz'),**support_archive)
  fallback=case_counts['no_seed_groups']>0;unknown_entity=not mask_receipt['T']['available'];st={}
  for arm in arms:
   native=arm.startswith(('native_','shuffle_'));st[arm]={'unknown':bool(not known.get(arm,True) or (native and unknown_entity)),'fallback':bool(fallback and arm.startswith(('pre_','native_','shuffle_')) or (native and unknown_entity)),'no_parameter_update':True}
  statuses.append({'id':sid,'arm_statuses':st});record={'id_PRIVATE':sid,'platform':row['platform'],'entity_masks':mask_receipt,'carrier_and_rectangles_fixed':True,'reference_source_group_mass_and_top3_fixed':True,'all_reference_phrases_keep_independent_slot_sets':True,'counts':case_counts,'seconds':time.perf_counter()-tick,'reference_hypotheses_PRIVATE':refs_by_variant};dump(out/'cases_PRIVATE'/(sid+'.json'),record);stats.append(record)
  if len(stats)%16==0:print('SUPPORT53',len(stats),336,flush=True)
 dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'COSTS.json',{'seconds':time.perf_counter()-start,'n':len(stats),'model_forwards':0,'optimizer_steps':0})
 dump(out/'SEALED.json',{'state':'R53_LATENT_POINT_RGB_ASSOCIATION_PREDICTIONS_NO_GT','n':len(stats),'arm_names':arms,'plan_sha256':sha(a.plan),'files':{f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file()}})
 print(json.dumps({'state':'SEALED','n':len(stats),'arms':len(arms),'seconds':time.perf_counter()-start}),flush=True)
if __name__=='__main__':main()
