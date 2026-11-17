"""Corrected pre-score input units: exact point-content groups, all aliases/captions."""
from pathlib import Path
import json,hashlib,collections,datetime
from select131 import allocate,sha,resolve
R=Path(__file__).resolve().parents[2];H=R/'research/round131';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 content=rd(H/'CONTENT_GROUPS_PRIVATE.json');ordinary=[g for g in content if not g['old_development_content'] and not g['protected_content_group']];anchors=[];registry=[];strata=[];summary=[];poolframes={}
 for pl in ['waymo','drone','quad']:
  groups=collections.defaultdict(list);rows=rd(R/f'research/round97/input/TARGET_{pl.upper()}_PRIVATE.json');byframe=collections.defaultdict(list)
  for r in rows:byframe[resolve(r['metapath'])].append(r)
  for g in ordinary:
   if g['platform']==pl:assert len(g['scenes_PRIVATE'])==1;groups[g['scenes_PRIVATE'][0]].append(g)
  budget=min(128,sum(map(len,groups.values())));alloc=allocate(groups,budget);selected=[];poolframes[pl]=[f for gg in groups.values() for g in gg for f in g['frames']]
  for scene,gg in sorted(groups.items()):
   gg.sort(key=lambda g:hashlib.sha256(('R131cleanContent|'+pl+'|'+g['content_group_PRIVATE']).encode()).hexdigest());chosen=gg[:alloc[scene]];selected+=chosen;w=len(gg)/len(chosen);strata.append({'target':pl,'scene_PRIVATE':scene,'population_content_groups':len(gg),'selected_content_groups':len(chosen),'population_frame_paths':sum(len(g['frames']) for g in gg),'selected_frame_paths':sum(len(g['frames']) for g in chosen),'population_expressions':sum(len(byframe[resolve(f['metapath'])]) for g in gg for f in g['frames']),'selected_expressions':sum(len(byframe[resolve(f['metapath'])]) for g in chosen for f in g['frames']),'inverse_group_sampling_weight':w})
   for g in chosen:
    for f in g['frames']:
     assert f['eligible_for_extra_media'];ar={'A':f,'B_pair':None,'content_group_PRIVATE':g['content_group_PRIVATE'],'sampling_weight':w,'selected_all_captions_and_aliases':True,'protected_alias_fallback':False};anchors.append(ar)
     for src in ['waymo','drone','quad']:
      if src==pl:continue
      for r in byframe[resolve(f['metapath'])]:registry.append({'cohort':'EXT','source':src,'target':pl,'row':r,'frame_PRIVATE':f['frame_PRIVATE'],'content_group_PRIVATE':g['content_group_PRIVATE'],'sampling_weight':w,'protected_alias_fallback':False})
  summary.append({'target':pl,'population_content_groups':sum(map(len,groups.values())),'selected_content_groups':len(selected),'population_frame_paths':len(poolframes[pl]),'selected_frame_paths':sum(len(g['frames']) for g in selected),'selected_expressions':sum(len(byframe[resolve(f['metapath'])]) for g in selected for f in g['frames']),'recording_clusters':len(groups),'known_old_point_content_overlap':0,'protected_content_groups_selected':0})
 pairs=[]
 for ar in anchors:
  A=ar['A'];ha=sha(A['imagepath']);A=dict(A,image_sha256=ha);ar['A']=A;others=[f for f in poolframes[A['platform']] if f['scene_PRIVATE']==A['scene_PRIVATE'] and f['suffix']==A['suffix'] and f['content_group_PRIVATE']!=A['content_group_PRIVATE']];field='timestamp_counter' if A['timestamp_counter'] is not None and any(f['timestamp_counter'] is not None for f in others) else 'folder_ordinal';others=[f for f in others if f[field] is not None and A[field] is not None];others.sort(key=lambda f:(abs(f[field]-A[field]),f['frame_PRIVATE']))
  for f in others:
   hb=sha(f['imagepath'])
   if hb==ha:continue
   pid=A['frame_PRIVATE']+'_R131_B';ar['B_pair']=pid;pairs.append({'pair_PRIVATE':pid,'kind':'same_scene_distinct_sensor_candidate','A':A,'B':dict(f,image_sha256=hb),'counter_difference':f[field]-A[field],'counter_unit_seconds_verified':False});break
 # Every selected content group includes all its whitelisted frame aliases and captions.
 assert len(registry)==2*sum(s['selected_expressions'] for s in summary);assert all(p['A']['content_group_PRIVATE']!=p['B']['content_group_PRIVATE'] for p in pairs)
 dump(H/'ANCHORS_PRIVATE.json',anchors);dump(H/'PAIRS_PRIVATE.json',pairs);dump(H/'INPUT_REGISTRY_PRIVATE.json',registry);dump(H/'STRATA_PRIVATE.json',strata)
 out={'selected_content_groups':sum(s['selected_content_groups'] for s in summary),'selected_frame_paths':len(anchors),'expressions':len(registry)//2,'directed_inputs':len(registry),'pairs_attempted':len(pairs),'groups':summary,'selector':'up to128 raw-point-content groups perplatform; minimum one/remaining recording then proportional group allocation; fixed hash withinrecording; allframealiases/captions and bothsource directions together','ordinary_pool':'exclude entire raw-point SHA groups touching old DEV/EXP A/B/C paths or known protected-image aliases; no model/GT/registration-result filtering','primary_metric':'six-direction mean of inverse-content-group-sampling-weighted expression means; raw means and target/recording results also retained','population_scope':'remaining ordinary content-disjoint point groups, not original fullVAL; no new scene/blind claim','historical_full_VAL_exposed':True,'FINAL_access':False,'new_C_generated':False,'no_source_training':True,'old_attempt':'path-only1696 plan cancelled before any GT;239 selected paths repeated old point content; all logs/partial predictions retained separately','GT_or_model_used_for_selector':False};dump(H/'INPUT_SUMMARY.json',out);files=['ANCHORS_PRIVATE.json','PAIRS_PRIVATE.json','INPUT_REGISTRY_PRIVATE.json','STRATA_PRIVATE.json','CONTENT_GROUPS_PRIVATE.json'];dump(H/'experiment/INPUT_LOCK.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_code_path':'select_content131.py','selection_code_sha256':sha(Path(__file__)),'files':{n:sha(H/n) for n in files},'summary':out,'no_task_GT':True,'FINAL_access':False});print(json.dumps(out,indent=2))
if __name__=='__main__':main()
