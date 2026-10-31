"""Independent R13 input-only replay; no labels/media decode/model/IoU."""
from pathlib import Path
import hashlib,json,collections
import numpy as np
B=Path(__file__).resolve().parents[3];D=B/'research/round13/input';V=B/'research/round13/evaluation'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0
def ck(x):
 global checks
 assert x;checks+=1
plan=read(D/'INPUT_PLAN_LOCK.json');ck(plan['program_sha256']==sha(B/'research/round13/prepare_inputs.py'))
for n,h in plan['files'].items():ck(sha(D/n)==h)
full=read(B/'research/round8/experiment/full5912_gpu_inputs.json');fm={r['id']:r for r in full};hp=B/'research/round12/data_audit/PHYSICAL_INPUT_HASH_INDEX_PRIVATE.json';hm={r['id_PRIVATE']:r for r in read(hp)['rows_PRIVATE']};ck(sha(hp)==plan['physical_hash_index_sha256'])
ck(sha(B/'research/round8/experiment/full5912_gpu_inputs.json')==plan['source_population_sha256'])
excluded=set()
for p,h in plan['prior_manifest_sha256'].items():
 ck(sha(B/p)==h);excluded.update(hm[r['id']]['physical_group'] for r in read(B/p))
ck(len(excluded)==588)
of=B/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json';oldframes=read(of);ck(sha(of)==plan['R12_all_frontend_frame_index_sha256']);excluded.update(x['physical_group_sha256'] for x in oldframes.values());ck(len(excluded)==914)
groups=collections.defaultdict(list)
for r in hm.values():groups[r['physical_group']].append(r)
canonical={g:min(rr,key=lambda x:x['id_PRIVATE']) for g,rr in groups.items()};fresh=[];scene_results=[]
for i,(platform,scene) in enumerate(sorted({(r['platform'],r['scene_PRIVATE']) for r in hm.values()})):
 available=[r for g,r in canonical.items() if g not in excluded and (r['platform'],r['scene_PRIVATE'])==(platform,scene)]
 ordered=sorted(available,key=lambda r:(hashlib.sha256((plan['salt']+'|'+r['id_PRIVATE']).encode()).hexdigest(),r['id_PRIVATE']))
 fresh.extend(fm[r['id_PRIVATE']] for r in ordered[:32]);scene_results.append({'platform':platform,'scene_alias':f'scene_{i+1:02d}','available':len(available),'quota':min(32,len(available))})
ck(read(D/'NEW108_GPU_INPUTS.json')==fresh);ck([x['available'] for x in scene_results]==[440,0,297,12,90]);ck([x['quota'] for x in scene_results]==[32,0,32,12,32])
rows=read(D/'ALL116_GPU_INPUTS.json');cases=read(D/'CASE116_PRIVATE.json');frames=read(D/'FRAME_INDEX_PRIVATE.json');mask=read(D/'MASK108_FRAME_INPUTS.json')
ck(rows[:108]==fresh and len(rows)==116 and len(cases)==116);ck(len({x['id'] for x in rows})==116)
keys={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
for row in rows:ck(set(row)==keys and row==fm[row['id']])
# Independent old-regression selection from sealed output change flags, no task scores.
G=B/'crossplatform/round12/feedback146_attempt001';gs=read(G/'SEALED.json');changed=set()
for r in read(B/'research/round12/data_audit/ANCHOR146_GPU_INPUTS.json'):
 p=G/(r['id']+'.npz');ck(sha(p)==gs['files'][p.name])
 with np.load(p,allow_pickle=False) as z:
  names=list(z['arm_names']);yes=any(z['actual_changed_axes'][names.index(a)].any() for a in ('single_instance_direct','multiview_instance_direct'))
  if yes:changed.add(r['id'])
ck(len(changed)==8 and {r['id'] for r in rows[108:]}==changed)
ck(len(frames)==116 and len(mask)==108 and {x['frame_alias'] for x in mask}=={c['frame_alias'] for c in cases[:108]})
mtpath=B/'research/round12/data_audit/TARGET5912_INPUT_FIELDS_PRIVATE.json';ck(sha(mtpath)==plan['input_metadata_sha256']);mt={r['meta_path_PRIVATE']:r['input_fields_PRIVATE'] for r in read(mtpath)['rows_PRIVATE']}
actualfiles={};newgroups=[];donors=collections.Counter();grouped=collections.defaultdict(list)
for i,(r,c) in enumerate(zip(rows,cases)):
 ck(c['id']==r['id'] and c['platform']==r['platform'] and c['scene_PRIVATE']==r['scene']);f=frames[c['frame_alias']];h=hm[r['id']]
 ck(set(f)=={'platform','scene_PRIVATE','pointpath','metapath','imagepath','image_sha256','point_sha256','metadata_sha256','physical_group_sha256','image_intrinsic','image_extrinsic','camera_distortion'})
 for a,b in [('imagepath','image_sha256'),('pointpath','point_sha256'),('metapath','metadata_sha256')]:
  ck(f[a]==r[a]);p=Path(r[a]);actualfiles[str(p)]=sha(p);ck(actualfiles[str(p)]==f[b])
 ck(f['image_sha256']==h['image_sha256'] and f['point_sha256']==h['point_sha256'] and f['physical_group_sha256']==h['physical_group'])
 inp=mt[r['metapath']]
 for key in ('image_intrinsic','image_extrinsic'):ck(f[key]==inp[key])
 ck(f['camera_distortion']==inp.get('camera_distortion'))
 if i<108:
  ck(c['role']=='new_content' and c['source_dir']=='crossplatform/round13/source108_attempt001' and c['mask_dir']=='crossplatform/round13/masks108_attempt001');ck(h['physical_group'] not in excluded);newgroups.append(h['physical_group'])
  ck(mask[i]=={'frame_alias':c['frame_alias'],'imagepath':r['imagepath']})
 else:
  ck(c['role']=='exposed_R12_all_changed_regression' and c['source_dir']=='crossplatform/round12/source146_attempt001' and c['mask_dir']=='crossplatform/round12/masks501_attempt001');ck(all(f[k]==oldframes[c['frame_alias']][k] for k in f))
 donor=fm[c['shuffled_donor_id_PRIVATE']];ck(c['shuffled_caption']==donor['caption']!=r['caption'] and (donor['platform'],donor['scene'])==(r['platform'],r['scene']));donors[donor['id']]+=1;grouped[c['role'],r['platform'],r['scene']].append(i)
ck(len(set(newgroups))==108)
# Independently replay fixed same-scene distinct-description donor selection.
for inds in grouped.values():
 if len(inds)==1:
  i=inds[0];r=rows[i];opts=[x for x in full if (x['scene'],x['platform'])==(r['scene'],r['platform']) and x['caption']!=r['caption']]
  donor=min(opts,key=lambda x:hashlib.sha256((plan['salt']+'|wrongtext|'+r['id']+'|'+x['id']).encode()).hexdigest());ck(cases[i]['shuffled_donor_id_PRIVATE']==donor['id'])
 else:
  ordered=sorted(inds,key=lambda i:hashlib.sha256((plan['salt']+'|text|'+rows[i]['id']).encode()).hexdigest())
  for j,i in enumerate(ordered):
   ds=[ordered[(j+k)%len(ordered)] for k in range(1,len(ordered)) if rows[ordered[(j+k)%len(ordered)]]['caption']!=rows[i]['caption']]
   donor=rows[ds[0]] if ds else min((x for x in full if (x['scene'],x['platform'])==(rows[i]['scene'],rows[i]['platform']) and x['caption']!=rows[i]['caption']),key=lambda x:x['id']);ck(cases[i]['shuffled_donor_id_PRIVATE']==donor['id'])
# Reused scripts describe real new forward by bound plan/manifest, not by state-name spelling.
source=B/'crossplatform/round12/capture_source.py';frontend=B/'crossplatform/round12/capture_masks.py'
ck(sha(source)=='4f640f27a4470195a6440818f12e8978231daa088a627689c524f86bb8399c9c');ck(sha(frontend)=='27a978ff72cc2fc162dcf61aedcea3cbaff03c4307b1c9db9c52ae32cf3e788a')
ck(all(__import__('re').fullmatch(r'input_frame_[0-9]{5}',c['frame_alias']) for c in cases))
result={'state':'PASS_R13_INPUT_SELECTION_AND_REUSED_CAPTURE_INTERFACE','input_plan_sha256':sha(D/'INPUT_PLAN_LOCK.json'),'auditor_sha256':sha(Path(__file__)),'checks':checks,'new_n':108,'new_platform_n':{'drone':32,'quad':76},'new_scene_n':{'drone':1,'quad':3},'scene_counts_aliased':scene_results,'excluded_exact_content_groups':914,'regression_n':8,'regression_matches_all_R12_single_MV_changed_union':True,'new_unique_content_groups':108,'actual_sensor_container_files_rehashed_without_decode':len(actualfiles),'all9keyvalues_original':True,'camera_fields_exact_allowlisted_input_copy':True,'shuffled_same_scene_different_description_all':True,'shuffled_unique_donors':len(donors),'max_donor_reuse':max(donors.values()),'shuffle_scope':'Different description is not a certified different target or contradiction; this is fixed input mismatch control.','source_capture_code_sha256':sha(source),'mask_capture_code_sha256':sha(frontend),'capture_state_name_scope':'Unchanged R12 producer writes inherited R12 state names on genuinely new108 files; new bound plans/inputhashes/outputdir establish R13 run provenance. Old8 files remain old actual forwards.','no_GT_annotations_scoredrows_new_IoU_model_GPU_media_or_point_decode':True,'issues':[]}
out=V/'INPUT_INDEPENDENT_AUDIT.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'state':result['state'],'checks':checks,'plan_sha256':sha(D/'INPUT_PLAN_LOCK.json'),'review_sha256':sha(out),'donors':len(donors),'maxreuse':max(donors.values())}))
