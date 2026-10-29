from pathlib import Path
import json,hashlib,requests,concurrent.futures,struct,zlib,time
R=Path(__file__).resolve().parent;P=R.parent;index=json.loads((P/'downloads/dataset_zip_index.json').read_text());prior_manifests=[P/'manifest.json',P/'replication36/manifest.json'];prior_rows=[s for m in prior_manifests for s in json.loads(m.read_text())];assert len(prior_rows)==45;assert json.loads((P/'surface_dev45/scored.json').read_text())['continue_fixed_new36'];old={str(Path(x['meta_path']).parent).split('/assets/')[-1]+'/' for x in prior_rows};selected=[]
for platform in ['waymo','drone','quad']:
 val=set((P/f'upstream/data/splits/{platform}_val.txt').read_text().splitlines());names=[a for a in index if a['name'].startswith(f'3eed_dataset/{platform}/') and a['name'].endswith('/meta_info.json') and a['name'].split('/')[2] in val and a['name'].rsplit('/',1)[0]+'/' not in old];names.sort(key=lambda a:hashlib.sha256(a['name'].encode()).hexdigest());chosen=names[:12];assert len(chosen)==12
 for a in chosen:
  prefix=a['name'].rsplit('/',1)[0]+'/';selected.extend(x for x in index if x['name'].startswith(prefix) and x['name'].endswith(('meta_info.json','lidar.bin','lidar.npy')))
selection=[a['name'] for a in selected];(R/'selected_paths.json').write_text(json.dumps(selection,indent=2));print('selected',len(selection),sum(a['compressed'] for a in selected),flush=True)
plan={'id':'3eed_unexposed_frames36_surface_v1','phase':'path_lock_before_download','n':36,'per_platform':12,'selection':'first12SHA256(meta-relative-path) in each official val platform, exclude all prior45frames; no correctness/class filtering','caption_rule':'firstground_info caption only, formatting identical to initialsource','scene_overlap':'same validation split; report overlap with old scenes and scene clustered statistics; not independent unseen-scene generalization','arms':['officialweights_textonly_source6Dyaw0','original_fixedpointPCAyaw7D','locked_threecandidate_boxsurfaceSDFyaw7D'],'pca_rules':'exact original8unique-crop-points/1.5predictedLoverW thresholds, sourcecenter/size fixed, XYprincipal covariance eigenvector; no new tuning or rollback','seed':20261029,'points':16384,'coordinate_preprocessing':'unchanged original officialsource preprocessing','output_extension':'6D→7D; unlabeled geometryrefinement, no networkTTA','labels':'source/PCA/surface predictions all sealed before GT scorer','selected_paths_sha256':hashlib.sha256((R/'selected_paths.json').read_bytes()).hexdigest(),'selected_paths':selection,'surface_development_plan_sha256':hashlib.sha256((P/'surface_dev45/PLAN_LOCK.json').read_bytes()).hexdigest(),'surface_refine_sha256':hashlib.sha256((P/'surface_dev45/refine.py').read_bytes()).hexdigest(),'old_geometry_plan_sha256':hashlib.sha256((P/'PLAN_GEOMETRY_PCA.json').read_bytes()).hexdigest(),'resources':'CPUdownload; one sourceGPU inference pass per frame; PCA CPU36samples; no LN/parameter/threshold changes'};(R/'PATH_PLAN_LOCK.json').write_text(json.dumps(plan,indent=2));(R/'PATH_PLAN_LOCK.sha256').write_text(hashlib.sha256((R/'PATH_PLAN_LOCK.json').read_bytes()).hexdigest()+'\n')
def fetch(a):
 dest=R/'assets'/a['name']
 if dest.exists() and dest.stat().st_size==a['size']:return a['name']
 start=a['offset'];end=start+30+len(a['name'].encode())+65536+a['compressed'];url='https://huggingface.co/datasets/RRRong/3EED/resolve/main/3eed_dataset.zip'
 for attempt in range(4):
  try:
   q=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=100);q.raise_for_status();b=q.content;assert q.status_code==206 and b[:4]==b'PK\x03\x04';nl,el=struct.unpack_from('<HH',b,26);b=b[30+nl+el:30+nl+el+a['compressed']];b=zlib.decompress(b,-15) if a['type']==8 else b;assert len(b)==a['size'];dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(b);return a['name']
  except Exception:
   if attempt==3:raise
   time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
 for name in ex.map(fetch,selected):print(name,flush=True)
samples=[]
for name in selection:
 if not name.endswith('meta_info.json'):continue
 path=R/'assets'/name;meta=json.loads(path.read_text());assert meta['ground_info'];plat=name.split('/')[1];samples.append({'id':hashlib.sha256(name.encode()).hexdigest()[:16],'platform':plat,'scene':name.split('/')[2],'meta_path':str(path),'lidar_path':str(path.with_name('lidar.npy' if plat=='waymo' else 'lidar.bin')),'text':' '.join(meta['ground_info'][0]['caption'].replace(',',' ,').split())+' . not mentioned','obj_index':0})
assert len(samples)==36 and len(set(s['id'] for s in samples))==36
(R/'manifest.json').write_text(json.dumps(samples,indent=2));old_scenes={(s['platform'],Path(s['meta_path']).parent.parent.name) for s in prior_rows};info={'n':len(samples),'platform_counts':{k:sum(s['platform']==k for s in samples) for k in ['waymo','drone','quad']},'scene_counts':{k:len(set(s['scene'] for s in samples if s['platform']==k)) for k in ['waymo','drone','quad']},'new_frames_old_scene_count':sum((s['platform'],s['scene']) in old_scenes for s in samples),'new_ids_disjoint':not set(s['id'] for s in samples)&set(s['id'] for s in prior_rows),'manifest_sha256':hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest()};(R/'cohort_info.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
