"""Metadata-only two separate interventions: fixed top3 B and source-AB log-gap."""
from common132 import *
import collections,functools
@functools.lru_cache(None)
def imagehash(p):return sha(p)
def candidates(A,pool,maximum):
 other=[f for f in pool if f['scene_PRIVATE']==A['scene_PRIVATE'] and f['suffix']==A['suffix'] and f['content_group_PRIVATE']!=A['content_group_PRIVATE']];field='timestamp_counter' if A['timestamp_counter'] is not None and any(f['timestamp_counter'] is not None for f in other) else 'folder_ordinal';other=[f for f in other if A[field] is not None and f[field] is not None];other.sort(key=lambda f:(abs(f[field]-A[field]),f['frame_PRIVATE']));out=[];seen_groups=set();seen_images={A['image_sha256']}
 for f in other:
  ih=f['image_sha256'] or imagehash(f['imagepath'])
  if ih in seen_images or f['content_group_PRIVATE'] in seen_groups:continue
  seen_groups.add(f['content_group_PRIVATE']);seen_images.add(ih);out.append((dict(f,image_sha256=ih),f[field]-A[field]))
  if len(out)==maximum:break
 return out
def main():
 basecheck();content=rd(R132/'research/round131/CONTENT_GROUPS_PRIVATE.json');oldanchors=rd(R132/'research/round131/ANCHORS_PRIVATE.json');oldpairs={p['pair_PRIVATE']:p for p in rd(R132/'research/round131/PAIRS_PRIVATE.json')};clean=[f for g in content if not g['old_development_content'] and not g['protected_content_group'] for f in g['frames']];target=[];used={a['content_group_PRIVATE'] for a in oldanchors}|{p['B']['content_group_PRIVATE'] for p in oldpairs.values()}
 for a in oldanchors:
  A=a['A'];pool=[f for f in clean if f['platform']==A['platform']];cc=candidates(A,pool,3);pp=[]
  assert bool(cc)==bool(a['B_pair'])
  for rank,(B,dt) in enumerate(cc,1):
   if rank==1:
    old=oldpairs[a['B_pair']];assert B['frame_PRIVATE']==old['B']['frame_PRIVATE'] and B['image_sha256']==old['B']['image_sha256'];pair=old
   else:pair={'pair_PRIVATE':A['frame_PRIVATE']+'_R132_B'+str(rank),'kind':'same_scene_distinct_sensor_candidate','A':A,'B':B,'counter_difference':dt,'counter_unit_seconds_verified':False}
   pp.append({'rank':rank,'pair':pair,'registration_reuse':str((R132/'crossplatform/round131/registration_PRIVATE'/(pair['pair_PRIVATE']+'.json')).relative_to(R132)) if rank==1 else None})
  target.append({'A':A,'candidates':pp})
 # Calibration is source-VAL only, source-own model; no R131 A/B content or protected content.
 frame={str(Path(f['metapath']).resolve()):(f,g) for g in content for f in g['frames']};source=[];sourcegroups={};excluded=collections.Counter()
 for e in rd(R132/'research/round121/SOURCE_CACHE_REGISTRY_PRIVATE.json'):
  f,g=frame[str(Path(e['row']['metapath']).resolve())]
  if g['protected_content_group']:excluded['protected_content']+=1;continue
  if g['content_group_PRIVATE'] in used:excluded['R131_A_or_B_content']+=1;continue
  assert f['eligible_for_extra_media'];A=dict(f,image_sha256=f['image_sha256'] or imagehash(f['imagepath']));pool=[ff for gg in content if gg['platform']==e['source'] and not gg['protected_content_group'] and gg['content_group_PRIVATE'] not in used for ff in gg['frames']];cc=candidates(A,pool,1);pair=None
  if cc:
   B,dt=cc[0];pair={'pair_PRIVATE':'SRC_'+A['frame_PRIVATE']+'_R132_B1','kind':'same_scene_distinct_sensor_candidate','A':A,'B':B,'counter_difference':dt,'counter_unit_seconds_verified':False};sourcegroups[pair['pair_PRIVATE']]=pair
  source.append({'entry':e,'A':A,'pair_PRIVATE':pair['pair_PRIVATE'] if pair else None})
 dump(H132/'TARGET_B3_CANDIDATES_PRIVATE.json',target);dump(H132/'SOURCE_CALIBRATION_INPUTS_PRIVATE.json',source);dump(H132/'SOURCE_PAIRS_PRIVATE.json',list(sourcegroups.values()))
 summary={'target_anchors':len(target),'target_directed_inputs':1798,'max_B_candidates':3,'first_B_bitexact_original_selection':True,'target_candidate_count':dict(collections.Counter(len(a['candidates']) for a in target)),'source_calibration_n':len(source),'source_platform_n':dict(collections.Counter(e['entry']['source'] for e in source)),'source_unique_pairs':len(sourcegroups),'source_exclusions':dict(excluded),'source_R131_A_B_content_overlap':0,'source_protected_content_overlap':0,'source_labels_read_for_selection':False,'target_GT_used':False,'FINAL_access':False};dump(H132/'INPUT_SUMMARY.json',summary)
 rules={'utc':utc(),'scope':'R1311798 exposed fixed inputs; cached diagnosis then two separate single-factor arms, no B3+loggap combination','B3':'at most3 preordered distinct point-content/RGB candidates; B1 exactlyR131; take first original two-way/cycle geometry pass, no gate relaxation/no GT/no scored preference; skip later attempts after success','B3_information_controls':'zero/direct/correct/wrong all share the same selected B; retain original4scores, new-coverage subgroup and full denominator; charge all extra registration cost','loggap_qualification':'all600 C=S cases have a view max<=tau; frozen absolute-gap scale mechanism observed, proceed with logratio alternative','loggap':'original B only; C=union overA/B of logmax(groupmeanp)-log(groupmeanp)<=kappa; preserve unknown if flat; no forced negatives, no intersection','kappa':'one pooled90th percentile of source-own registered-A/B per-view gap to best Acc25 good supported group, after original n4/coverage.5 qualification; source labels only after source predictions sealed; no target/task/coverage-based choice; no perplatform threshold','source_calibration':'existing sourceVAL128perplatform; exclude all known protected-content and R131 A/B content from A/B; maximumone nearest B with original geometry gates; no source training or GTgradient; reuse sourceA fixedcontext/prefix/GT IoUs','unchanged':'three source states,A256,grouping,pointUp4896,rho0/3steps/featurekeep1/Armijo9,original A context and F_AB reader','no_fallback':'if no valid sourceAB calibration gap stoploggap branch and document blockage; no target-derived kappa or sourceA-only proxy','selection_and_scoring':'all candidates/kappa/method code locked before new target predictions; all target arms sealed before any new task scoring; use cached sealed A256 IoUs, no new GT geometry/FINAL access','decision':'report signed gains vs arm-ownzero/direct/wrong and vs R131; retain working factor only if weighted same-information comparisons and R131partial gain allpositive; no six-direction-positive demand; no postscore tuning','resource':'oneGPUexclusive,4registrationCPUworkers,eachguard<=10800seconds; one sourcecal and one target matrix, no radius/step/layer grid','source_training':False,'new_experts':False,'FINAL_access':False,'files':{p.name:sha(p) for p in [H132/'TARGET_B3_CANDIDATES_PRIVATE.json',H132/'SOURCE_CALIBRATION_INPUTS_PRIVATE.json',H132/'SOURCE_PAIRS_PRIVATE.json',H132/'DIAGNOSIS_NO_GT_SEAL.json']}}
 dump(H132/'experiment/DESIGN_LOCK.json',rules);print(summary)
if __name__=='__main__':main()
