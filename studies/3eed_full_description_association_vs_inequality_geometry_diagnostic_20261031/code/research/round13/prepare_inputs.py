"""R13 input-only selection, strict exclusion of prior frontend content. No GT."""
from pathlib import Path
import collections,hashlib,json
B=Path(__file__).resolve().parents[2];D=B/'research/round13/input';D.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads((B/p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 p=D/n;assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');return p
def main():
 population='research/round8/experiment/full5912_gpu_inputs.json';full=read(population);fm={r['id']:r for r in full}
 hp='research/round12/data_audit/PHYSICAL_INPUT_HASH_INDEX_PRIVATE.json';hashed=read(hp)['rows_PRIVATE'];hm={r['id_PRIVATE']:r for r in hashed}
 prior=['research/round9/experiment/diagnostic160_gpu_inputs.json','research/round10/input_prepare/diagnostic160_gpu_inputs.json','research/round11/input_prepare/diagnostic320_gpu_inputs.json','research/round8/experiment/smoke40_gpu_inputs.json','research/round7/experiment/smoke_gpu_inputs.json']
 excluded={hm[r['id']]['physical_group'] for p in prior for r in read(p)};assert len(excluded)==588
 oldframes=read('research/round12/data_audit/FRAME_INDEX_PRIVATE.json');excluded.update(r['physical_group_sha256'] for r in oldframes.values());assert len(excluded)==914
 groups=collections.defaultdict(list)
 for r in hashed:groups[r['physical_group']].append(r)
 canonical={g:min(rs,key=lambda x:x['id_PRIVATE']) for g,rs in groups.items()}
 scene_keys=sorted({(r['platform'],r['scene_PRIVATE']) for r in hashed});chosen=[];counts=[];salt='R13_full_description_partial_observation_v1'
 for platform,scene in scene_keys:
  avail=[r for g,r in canonical.items() if g not in excluded and (r['platform'],r['scene_PRIVATE'])==(platform,scene)]
  chosen+=sorted(avail,key=lambda r:(hashlib.sha256((salt+'|'+r['id_PRIVATE']).encode()).hexdigest(),r['id_PRIVATE']))[:32]
  counts.append({'platform':platform,'scene_PRIVATE':scene,'available':len(avail),'quota':min(32,len(avail))})
 assert len(chosen)==108 and collections.Counter(r['platform'] for r in chosen)=={'drone':32,'quad':76}
 fresh=[fm[r['id_PRIVATE']] for r in chosen]
 oldanchors=read('research/round12/data_audit/ANCHOR146_GPU_INPUTS.json');oldmap={r['id']:r for r in oldanchors}
 gallery='research/round12/experiment/G_CHANGE_GALLERY_PLAN.json';reg=[oldmap[r['id_PRIVATE']] for r in read(gallery)['rows_PRIVATE']];assert len(reg)==8
 rows=fresh+reg;assert len({r['id'] for r in rows})==116
 metadata='research/round12/data_audit/TARGET5912_INPUT_FIELDS_PRIVATE.json';mt={r['meta_path_PRIVATE']:r['input_fields_PRIVATE'] for r in read(metadata)['rows_PRIVATE']}
 oldpairs={r['anchor_id_PRIVATE']:r for r in read('research/round12/data_audit/PAIRS_PRIVATE.json')['rows_PRIVATE']}
 frameindex={};cases=[];maskinputs=[]
 for i,r in enumerate(rows):
  role='new_content' if i<108 else 'exposed_R12_all_changed_regression';f=hm[r['id']]
  if i<108:
   alias=f'input_frame_{502+i:05d}';m=mt[r['metapath']]
   fr={'platform':r['platform'],'scene_PRIVATE':r['scene'],'pointpath':r['pointpath'],'metapath':r['metapath'],'imagepath':r['imagepath'],'image_sha256':f['image_sha256'],'point_sha256':f['point_sha256'],'metadata_sha256':sha(r['metapath']),'physical_group_sha256':f['physical_group'],'image_intrinsic':m['image_intrinsic'],'image_extrinsic':m['image_extrinsic'],'camera_distortion':m.get('camera_distortion')}
   maskinputs.append({'frame_alias':alias,'imagepath':r['imagepath']});maskdir='crossplatform/round13/masks108_attempt001';src='crossplatform/round13/source108_attempt001'
  else:
   alias=oldpairs[r['id']]['anchor_frame_alias'];fr={k:v for k,v in oldframes[alias].items() if k in ('platform','scene_PRIVATE','pointpath','metapath','imagepath','image_sha256','point_sha256','metadata_sha256','physical_group_sha256','image_intrinsic','image_extrinsic','camera_distortion')};maskdir='crossplatform/round12/masks501_attempt001';src='crossplatform/round12/source146_attempt001'
  frameindex[alias]=fr
  cases.append({'id':r['id'],'case_alias':('new_' if i<108 else 'regression_')+f'{(i+1 if i<108 else i-107):05d}','role':role,'platform':r['platform'],'scene_PRIVATE':r['scene'],'frame_alias':alias,'source_dir':src,'mask_dir':maskdir})
 # Same-scene cyclic derangement, locked before any new model output/GT.
 byscene=collections.defaultdict(list)
 for i,r in enumerate(rows):byscene[(cases[i]['role'],r['platform'],r['scene'])].append(i)
 for inds in byscene.values():
  if len(inds)==1:
   # Regression singleton uses a different same-scene original input query,
   # selected from input metadata only, never annotation or task performance.
   i=inds[0];r=rows[i];opts=[x for x in full if x['scene']==r['scene'] and x['platform']==r['platform'] and x['caption']!=r['caption']]
   donor=min(opts,key=lambda x:hashlib.sha256((salt+'|wrongtext|'+r['id']+'|'+x['id']).encode()).hexdigest())
   cases[i]['shuffled_caption']=donor['caption'];cases[i]['shuffled_donor_id_PRIVATE']=donor['id']
  else:
   ordered=sorted(inds,key=lambda i:hashlib.sha256((salt+'|text|'+rows[i]['id']).encode()).hexdigest())
   for j,i in enumerate(ordered):
    donors=[ordered[(j+k)%len(ordered)] for k in range(1,len(ordered)) if rows[ordered[(j+k)%len(ordered)]]['caption']!=rows[i]['caption']]
    if donors:donor=rows[donors[0]]
    else:donor=min((x for x in full if x['scene']==rows[i]['scene'] and x['platform']==rows[i]['platform'] and x['caption']!=rows[i]['caption']),key=lambda x:x['id'])
    cases[i]['shuffled_caption']=donor['caption'];cases[i]['shuffled_donor_id_PRIVATE']=donor['id']
 assert all(c['shuffled_caption']!=r['caption'] for c,r in zip(cases,rows))
 save('NEW108_GPU_INPUTS.json',fresh);save('ALL116_GPU_INPUTS.json',rows);save('FRAME_INDEX_PRIVATE.json',frameindex);save('MASK108_FRAME_INPUTS.json',maskinputs);save('CASE116_PRIVATE.json',cases)
 save('INPUT_PLAN_LOCK.json',{'state':'input_only_R13_new108_plus8_exposed_regression_locked_before_models','program_sha256':sha(__file__),'salt':salt,'source_population_sha256':sha(B/population),'physical_hash_index_sha256':sha(B/hp),'input_metadata_sha256':sha(B/metadata),'old_changed_gallery_sha256':sha(B/gallery),'prior_manifest_sha256':{p:sha(B/p) for p in prior},'R12_all_frontend_frame_index_sha256':sha(B/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json'),'excluded_content_groups':914,'quota_rule':'up to32 per scene, zero remaining Drone scene not backfilled','scenes_PRIVATE':counts,'new_n':108,'new_platform_n':{'drone':32,'quad':76},'new_scene_n':{'drone':1,'quad':3},'regression_n':8,'regression_selection':'ALL R12 single/MV changedflags union, outcome-exposed not confirmation','scope':'New bytecontent, historically research-exposed VAL scenes; no independent domain/generalization significance. No helpers/MV input in primary R13; isolate language identity and partial-positive box constraints.','caption_control':'fixed same-scene cyclic distinct-description input-only derangement; not labelled contradictory or guaranteed different target','no_GT_or_media_decode_or_pointarray_model_GPU':True,'files':{p.name:sha(p) for p in sorted(D.glob('*PRIVATE.json'))+sorted(D.glob('*INPUTS.json'))}})
 print(json.dumps({'new_n':108,'regression_n':8,'new_platforms':{'drone':32,'quad':76},'available_scene_counts':[x['available'] for x in counts],'input_plan_sha256':sha(D/'INPUT_PLAN_LOCK.json')}))
if __name__=='__main__':main()
