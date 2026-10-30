"""R12 independent input byte/selection audit; no GT/pixel/point decoding."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,collections,re,time
B=Path(__file__).resolve().parents[3];D=B/'research/round12/data_audit';O=Path(__file__).resolve().parent
SALT='R12_independent_instance_temporal_v1';WRONG='R12_wrongscene_reference_v1';KEYS={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text())
h=lambda s:hashlib.sha256(s.encode()).hexdigest()
def main():
 start=time.perf_counter();receipt=read(D/'INPUT_RECEIPT.json');plan=read(D/'INPUT_PLAN_LOCK.json');index=read(D/'PHYSICAL_INPUT_HASH_INDEX_PRIVATE.json');full=read(B/'research/round8/experiment/full5912_gpu_inputs.json');fm={r['id']:r for r in full};hm={r['id_PRIVATE']:r for r in index['rows_PRIVATE']};assert len(fm)==len(hm)==5912 and set(fm)==set(hm)
 for field,name in [('input_plan_sha256','INPUT_PLAN_LOCK.json'),('anchor_manifest_sha256','ANCHOR146_GPU_INPUTS.json'),('pairs_sha256','PAIRS_PRIVATE.json'),('frame_index_sha256','FRAME_INDEX_PRIVATE.json'),('mask_frame_inputs_sha256','MASK_FRAME_INPUTS.json')]:assert receipt[field]==sha(D/name)
 assert plan['source_population_sha256']==sha(B/'research/round8/experiment/full5912_gpu_inputs.json') and plan['physical_input_index_sha256']==sha(D/'PHYSICAL_INPUT_HASH_INDEX_PRIVATE.json') and plan['input_metadata_table_sha256']==sha(D/'TARGET5912_INPUT_FIELDS_PRIVATE.json')
 prior_paths={'R9_160':'research/round9/experiment/diagnostic160_gpu_inputs.json','R10_160':'research/round10/input_prepare/diagnostic160_gpu_inputs.json','R11_320':'research/round11/input_prepare/diagnostic320_gpu_inputs.json','R8_40':'research/round8/experiment/smoke40_gpu_inputs.json','R7_20':'research/round7/experiment/smoke_gpu_inputs.json'};prior=[]
 for name,p in prior_paths.items():assert sha(B/p)==plan['prior_manifest_sha256'][name];prior+=read(B/p)
 assert len(prior)==len({r['id'] for r in prior})==700
 group=collections.defaultdict(list)
 for sid,x in hm.items():
  assert (x['platform'],x['scene_PRIVATE'],x['image_path_PRIVATE'],x['point_path_PRIVATE'])==(fm[sid]['platform'],fm[sid]['scene'],fm[sid]['imagepath'],fm[sid]['pointpath'])
  assert x['physical_group']==h('R12_physical_frame_v1|'+x['image_sha256']+'|'+x['point_sha256']);group[x['physical_group']].append(x)
 assert len(group)==1753 and all(len({(r['platform'],r['scene_PRIVATE']) for r in rr})==1 for rr in group.values())
 canonical={g:min(rr,key=lambda r:r['id_PRIVATE']) for g,rr in group.items()};excluded={hm[r['id']]['physical_group'] for r in prior};assert len(excluded)==588
 keys=sorted({(r['platform'],r['scene_PRIVATE']) for r in hm.values()});assert len(keys)==5
 selected=[];availability=[]
 for key in keys:
  pool=[x for g,x in canonical.items() if g not in excluded and (x['platform'],x['scene_PRIVATE'])==key];take=sorted(pool,key=lambda x:(h(SALT+'|'+x['id_PRIVATE']),x['id_PRIVATE']))[:32];selected+=take;availability.append((key,len(pool),len(take)))
 anchors=read(D/'ANCHOR146_GPU_INPUTS.json');assert len(anchors)==146 and anchors==[fm[x['id_PRIVATE']] for x in selected] and all(set(x)==KEYS for x in anchors)
 assert len({x['physical_group'] for x in selected})==146 and not {x['physical_group'] for x in selected}&excluded and not {r['id'] for r in anchors}&{r['id'] for r in prior}
 assert collections.Counter(r['platform'] for r in anchors)=={'drone':50,'quad':96} and sorted(n for _,_,n in availability)==[18,32,32,32,32]
 timelines=collections.defaultdict(dict)
 for g,x in canonical.items():
  key=(x['platform'],x['scene_PRIVATE']);stamp=x['input_timestamp'];assert stamp not in timelines[key] or timelines[key][stamp]['physical_group']==g;timelines[key][stamp]=x
 choices={};used={x['physical_group'] for x in selected};gaps=collections.Counter();missing=collections.Counter()
 for x in selected:
  vals=timelines[x['platform'],x['scene_PRIVATE']];t=x['input_timestamp'];before=[v for k,v in vals.items() if k<t];after=[v for k,v in vals.items() if k>t];wrong=[v for v in canonical.values() if v['platform']==x['platform'] and v['scene_PRIVATE']!=x['scene_PRIVATE']]
  ch={'previous':max(before,key=lambda v:v['input_timestamp']) if before else None,'next':min(after,key=lambda v:v['input_timestamp']) if after else None,'wrongview':min(wrong,key=lambda v:(h(WRONG+'|'+x['id_PRIVATE']+'|'+v['physical_group']),v['physical_group'])) if wrong else None};choices[x['physical_group']]=ch
  for role,v in ch.items():
   if v is None:missing[role]+=1
   else:
    assert v['physical_group']!=x['physical_group'];used.add(v['physical_group'])
    if role!='wrongview':gaps[str(abs(v['input_timestamp']-t))]+=1
 ordered=sorted([canonical[g] for g in used],key=lambda x:(x['platform'],x['scene_PRIVATE'],x['input_timestamp'],x['physical_group']));aliases={r['physical_group']:f'input_frame_{i+1:05d}' for i,r in enumerate(ordered)}
 frames=read(D/'FRAME_INDEX_PRIVATE.json');mask=read(D/'MASK_FRAME_INPUTS.json');pairs=read(D/'PAIRS_PRIVATE.json');assert len(frames)==len(ordered)==len(mask)==501
 assert list(frames)==[aliases[x['physical_group']] for x in ordered] and all(set(r)=={'frame_alias','imagepath'} and re.fullmatch(r'input_frame_[0-9]{5}',r['frame_alias']) for r in mask)
 assert mask==[{'frame_alias':alias,'imagepath':row['imagepath']} for alias,row in frames.items()]
 for x in ordered:
  f=frames[aliases[x['physical_group']]];row=fm[x['id_PRIVATE']]
  for key in ('platform','imagepath','pointpath','metapath'):assert f[key]==row[key]
  assert f['image_sha256']==x['image_sha256'] and f['point_sha256']==x['point_sha256'] and f['physical_group_sha256']==x['physical_group'] and f['timestamp']==x['input_timestamp']
 assert len(pairs['rows_PRIVATE'])==146;shared=collections.Counter()
 for x,pair in zip(selected,pairs['rows_PRIVATE']):
  assert pair['anchor_id_PRIVATE']==x['id_PRIVATE'] and pair['anchor_frame_alias']==aliases[x['physical_group']]
  for role,v in choices[x['physical_group']].items():
   actual=pair[role]
   if v is None:assert actual is None;continue
   assert actual['frame_alias']==aliases[v['physical_group']] and actual['counter_delta']==v['input_timestamp']-x['input_timestamp'] and actual['different_content_proved'] and not actual['physical_time_seconds_or_sync_verified']
   assert actual['same_scene']==(role!='wrongview') and v['platform']==x['platform'] and actual['absolute_counter_gap']==abs(actual['counter_delta']) and actual['within5_counter_steps']==(abs(actual['counter_delta'])<=5 if role!='wrongview' else None)
   if x['image_sha256']==v['image_sha256']:shared[role+'_same_image_bytes']+=1
   if x['point_sha256']==v['point_sha256']:shared[role+'_same_point_bytes']+=1
 assert dict(gaps)==receipt['companion_counter_gap_histogram'] and dict(missing)==receipt['companion_missing_roles']
 expected={}
 for x in ordered+[hm[r['id']] for r in prior]:
  for key,field in [('image_path_PRIVATE','image_sha256'),('point_path_PRIVATE','point_sha256')]:
   if x[key] in expected:assert expected[x[key]]==x[field]
   expected[x[key]]=x[field]
 totalbytes=sum(Path(p).stat().st_size for p in expected)
 def verify(item):p,digest=item;assert sha(p)==digest;return 1
 with ThreadPoolExecutor(max_workers=2) as pool:hashed=sum(pool.map(verify,expected.items()))
 assert (B/'crossplatform/round12/capture_source.py').read_text()==(B/'crossplatform/round11/capture_source.py').read_text().replace('R11','R12')
 out={'state':'PASS_R12_INDEPENDENT_INPUT_SELECTION_CONTENT_AND_SOURCE_CONTRACT','n_anchors':146,'by_platform':{'drone':50,'quad':96},'scene_quota_sizes':[n for _,_,n in availability],'scene_available_group_counts':[n for _,n,_ in availability],'exact_content_groups_all5912':1753,'prior_pathcases':700,'prior_excluded_exact_content_groups':588,'new_anchor_prior_ID_overlap':0,'new_anchor_prior_exact_content_overlap':0,'mask_frontend_alias_n':501,'mask_inputs_exact_two_keys':True,'all_hash_rank_selection_counter_candidates_wrongscene_rules_replayed':True,'companion_gaps':dict(gaps),'missing_roles':dict(missing),'shared_anchor_support_channel_counts':dict(shared),'actual_input_files_rehashed':hashed,'actual_input_bytes_rehashed':totalbytes,'actual501frontend_and700prior_pathcases_bytehashes_checked':True,'source_R11_to_R12_only_stringchange':True,'source_capture_sha256':sha(B/'crossplatform/round12/capture_source.py'),'input_receipt_sha256':sha(D/'INPUT_RECEIPT.json'),'input_plan_sha256':sha(D/'INPUT_PLAN_LOCK.json'),'anchor_manifest_sha256':sha(D/'ANCHOR146_GPU_INPUTS.json'),'pair_manifest_sha256':sha(D/'PAIRS_PRIVATE.json'),'mask_frame_inputs_sha256':sha(D/'MASK_FRAME_INPUTS.json'),'frame_index_sha256':sha(D/'FRAME_INDEX_PRIVATE.json'),'auditor_sha256':sha(Path(__file__)),'limitations':['Exact JPEG+pointfile bytes differ; reencoding or shared physical moments/objects are not ruled out.','Counter order is retrieval only, not seconds/ego clock/sync.','Support can overlap historical studies; anchor content exclusions strict.','D32+18 requires bothmicro/equal-scene, Q32x3; only2/3sceneclusters.','Providedpose unverified, not used by selection or primaryregistration.'],'no_GT_annotation_fields_pixels_pointarrays_model_GPU_or_scoring_decoded':True,'seconds':time.perf_counter()-start};path=O/'INPUT146_INDEPENDENT_AUDIT.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'files_rehashed':hashed,'seconds':out['seconds'],'sha256':sha(path)}))
if __name__=='__main__':main()
