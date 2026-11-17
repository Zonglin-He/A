"""Metadata-only grouped expansion: all captions per sampled physical frame."""
from pathlib import Path
import json,hashlib,collections,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round131';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();resolve=lambda p:str(Path(p).resolve())
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def allocate(groups,total):
 keys=sorted(groups);k={s:1 for s in keys};assert total>=len(keys);remaining=total-len(keys)
 while remaining:
  cap={s:len(groups[s])-k[s] for s in keys if k[s]<len(groups[s])};assert cap;weights={s:len(groups[s]) for s in cap};den=sum(weights.values());ideal={s:remaining*weights[s]/den for s in cap};add={s:min(cap[s],int(ideal[s])) for s in cap};used=sum(add.values())
  for s,v in add.items():k[s]+=v
  remaining-=used
  if remaining and used==0:
   order=sorted(cap,key=lambda s:(-ideal[s],hashlib.sha256(('R131allocation|'+s).encode()).hexdigest()))
   for s in order:
    if remaining and k[s]<len(groups[s]):k[s]+=1;remaining-=1
 return k
def main():
 F=rd(R/'research/round105/FRAMES_PRIVATE.json');registry=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');seen_meta={resolve(e['row']['metapath']) for e in registry};seen_points={resolve(e['row']['pointpath']) for e in registry}
 for f in ['research/round105/FULL_PAIRS_PRIVATE.json','research/round109/PAIRS_PRIVATE.json']:
  for p in rd(R/f):
   for v in ['A','B']:seen_meta.add(resolve(p[v]['metapath']));seen_points.add(resolve(p[v]['pointpath']))
 oldhash={f['image_sha256'] for f in F if resolve(f['metapath']) in seen_meta and f['image_sha256']};anchors=[];directions=[];summary=[];pools={};byplatform={}
 for pl in ['waymo','drone','quad']:
  rows=rd(R/f'research/round97/input/TARGET_{pl.upper()}_PRIVATE.json');byframe=collections.defaultdict(list)
  for r in rows:byframe[resolve(r['metapath'])].append(r)
  ordinary=[f for f in F if f['platform']==pl and resolve(f['metapath']) not in seen_meta and resolve(f['pointpath']) not in seen_points];groups=collections.defaultdict(list)
  for f in ordinary:groups[f['scene_PRIVATE']].append(f)
  k=allocate(groups,256);selected=[];strata=[]
  for sc,ff in sorted(groups.items()):
   ff.sort(key=lambda x:hashlib.sha256(('R131frame|'+pl+'|'+x['frame_PRIVATE']).encode()).hexdigest());chosen=ff[:k[sc]];selected+=chosen;strata.append({'target':pl,'scene_PRIVATE':sc,'population_frames':len(ff),'selected_frames':len(chosen),'population_expressions':sum(len(byframe[resolve(f['metapath'])]) for f in ff),'selected_expressions':sum(len(byframe[resolve(f['metapath'])]) for f in chosen),'inverse_frame_sampling_weight':len(ff)/len(chosen)})
   for f in chosen:
    weight=len(ff)/len(chosen);anchors.append({'A':f,'B_pair':None,'sampling_weight':weight,'selected_all_captions':True,'known_RGB_content_seen_in_development':bool(f['image_sha256'] in oldhash),'protected_alias_fallback':not f['eligible_for_extra_media']})
    for src in ['waymo','drone','quad']:
     if src==pl:continue
     for row in byframe[resolve(f['metapath'])]:directions.append({'cohort':'EXT','source':src,'target':pl,'row':row,'frame_PRIVATE':f['frame_PRIVATE'],'sampling_weight':weight,'protected_alias_fallback':not f['eligible_for_extra_media']})
  pools[pl]=ordinary;byplatform[pl]=byframe;summary.append({'target':pl,'population_frames':len(ordinary),'population_expressions':sum(len(byframe[resolve(f['metapath'])]) for f in ordinary),'selected_frames':len(selected),'selected_expressions':sum(len(byframe[resolve(f['metapath'])]) for f in selected),'recording_clusters':len(groups),'protected_fallback_frames':sum(not f['eligible_for_extra_media'] for f in selected),'known_RGB_content_overlap_old_development_frames':sum(f['image_sha256'] in oldhash for f in selected)})
 pairs=[];lookup={f['frame_PRIVATE']:f for ff in pools.values() for f in ff}
 for ar in anchors:
  A=ar['A']
  if not A['eligible_for_extra_media']:continue
  eligible=[f for f in pools[A['platform']] if f['scene_PRIVATE']==A['scene_PRIVATE'] and f['suffix']==A['suffix'] and f['eligible_for_extra_media'] and f['frame_PRIVATE']!=A['frame_PRIVATE']];field='timestamp_counter' if A['timestamp_counter'] is not None and any(x['timestamp_counter'] is not None for x in eligible) else 'folder_ordinal';eligible=[f for f in eligible if f[field] is not None and A[field] is not None];eligible.sort(key=lambda f:(abs(f[field]-A[field]),f['frame_PRIVATE']));ha=sha(A['imagepath']);A=dict(A,image_sha256=ha);ar['A']=A
  for f in eligible:
   hb=sha(f['imagepath'])
   if hb==ha:continue
   pid=A['frame_PRIVATE']+'_R131_B';ar['B_pair']=pid;pairs.append({'pair_PRIVATE':pid,'kind':'same_scene_distinct_sensor_candidate','A':A,'B':dict(f,image_sha256=hb),'counter_difference':f[field]-A[field],'counter_unit_seconds_verified':False});break
 # Frame groups carry every expression and both source directions. No GT/score selection.
 assert len(anchors)==768 and not any(resolve(a['A']['metapath']) in seen_meta for a in anchors);assert all(resolve(p[v]['metapath']) not in seen_meta for p in pairs for v in ['A','B'])
 dump(H/'ANCHORS_PRIVATE.json',anchors);dump(H/'PAIRS_PRIVATE.json',pairs);dump(H/'INPUT_REGISTRY_PRIVATE.json',directions);dump(H/'STRATA_PRIVATE.json',[r for pl in ['waymo','drone','quad'] for r in []])
 # Rebuild stratum counts from selection, without source-direction duplication.
 strata=[]
 for pl in ['waymo','drone','quad']:
  for sc in sorted({f['scene_PRIVATE'] for f in pools[pl]}):
   ff=[f for f in pools[pl] if f['scene_PRIVATE']==sc];aa=[a for a in anchors if a['A']['platform']==pl and a['A']['scene_PRIVATE']==sc];strata.append({'target':pl,'scene_PRIVATE':sc,'population_frames':len(ff),'selected_frames':len(aa),'population_expressions':sum(len(byplatform[pl][resolve(f['metapath'])]) for f in ff),'selected_expressions':sum(len(byplatform[pl][resolve(a['A']['metapath'])]) for a in aa),'inverse_frame_sampling_weight':len(ff)/len(aa)})
 dump(H/'STRATA_PRIVATE.json',strata);out={'selected_physical_frames':768,'expressions':len(directions)//2,'directed_inputs':len(directions),'pairs_attempted':len(pairs),'groups':summary,'selector':'minimum one frame/available recording, remaining allocation proportional to population frame counts with caps; fixed hash sample within recording; all captions and both source directions grouped','ordinary_pool_excludes':'all old DEV/EXP anchor and attempted B/C physical paths, using existing manifests; old FINAL frame identities were already outside R97 whitelist','primary_metric':'six-direction mean of per-direction inverse-frame-sampling-weighted expression means; raw six-direction mean also reported, no task label weight selection','historical_full_VAL_scores_exposed':True,'new_scene_claim':False,'recording_disjoint_from_old_development':False,'no_C_generated':'main F_AB only, fixed R130 recipe; no C fitting/selection or extra-view claim','protected_content_alias_policy':'selected protected aliases keep archived prepared-frozen output for all four arms; no raw media/features/newGT indexing','known_RGB_duplicate_frames':'not falsely treated as new imagery; scene and RGB-family dependence disclosed; no image-content-driven performance filter','no_GT_or_model_used':True};dump(H/'INPUT_SUMMARY.json',out)
 files=['ANCHORS_PRIVATE.json','PAIRS_PRIVATE.json','INPUT_REGISTRY_PRIVATE.json','STRATA_PRIVATE.json'];dump(H/'experiment/INPUT_LOCK.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{n:sha(H/n) for n in files},'selection_code_sha256':sha(Path(__file__)),'summary':out,'no_task_GT':True,'FINAL_access':False});print(json.dumps(out,indent=2))
if __name__=='__main__':main()
