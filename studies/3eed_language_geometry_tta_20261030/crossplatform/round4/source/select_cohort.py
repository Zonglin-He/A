"""Path-only SHA cohort selection; never opens point/caption/numeric-label files."""
from pathlib import Path
import hashlib,json,datetime
R=Path(__file__).resolve().parent;P=R.parents[1];D=P/'dataset_full/3eed_dataset';SEED='3eed-round4-path-scene-v1-20261030'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(s):return hashlib.sha256((SEED+'|'+s).encode()).hexdigest()
def main():
 old=set();oldf=set();evidence={}
 for rel in ['manifest.json','replication36/manifest.json','replication36_surface/manifest.json']:
  p=P/rel;evidence[rel]=sha(p)
  for s in json.loads(p.read_text()):
   path=Path(s['meta_path']);old.add((s['platform'],path.parent.parent.name));oldf.add((s['platform'],path.parent.parent.name,path.parent.name))
 p=P/'round2/COHORT_PATH_LOCK.json';evidence['round2/COHORT_PATH_LOCK.json']=sha(p)
 for cohort in json.loads(p.read_text())['cohorts'].values():
  for s in cohort:old.add((s['platform'],s['scene']));oldf.add((s['platform'],s['scene'],s['frame']))
 # Round3 explicitly reuses precisely round2 dev18+confirm36; verify registration rather than infer.
 p=P/'round3/PLAN_DIAGNOSTIC_LOCK.json';evidence['round3/PLAN_DIAGNOSTIC_LOCK.json']=sha(p)
 assert 'oldround2 dev18+confirm36' in json.loads(p.read_text())['cached_diagnostic']['parents']
 chosen={'dev':[],'confirm':[]};desc={}
 for plat,split,nd,nc,nfd,nfc in [('waymo','val',12,24,2,2),('drone','train',1,2,4,4),('quad','train',2,2,2,4)]:
  sp=D/'splits'/f'{plat}_{split}.txt';evidence[f'splits/{plat}_{split}.txt']=sha(sp)
  available=[s.strip() for s in sp.read_text().splitlines() if s.strip() and (D/plat/s.strip()).is_dir()]
  fresh_all=sorted([s for s in available if (plat,s) not in old],key=lambda s:rank(plat+'/'+s))
  framepools={scene:[q for q in (D/plat/scene).iterdir() if q.is_dir() and (q/'meta_info.json').is_file() and (q/('lidar.npy' if plat=='waymo' else 'lidar.bin')).is_file()] for scene in fresh_all}
  insufficient={scene:len(pool) for scene,pool in framepools.items() if len(pool)<max(nfd,nfc)}
  fresh=[scene for scene in fresh_all if scene not in insufficient]
  vsc=[s.strip() for s in (D/'splits'/f'{plat}_val.txt').read_text().splitlines() if s.strip() and (D/plat/s.strip()).is_dir()]
  desc[plat]={'split':split,'available_scenes':len(available),'previously_scored_scenes_in_split':sum((plat,s) in old for s in available),'research_fresh_scenes_before_path_eligibility':len(fresh_all),'research_fresh_scenes':len(fresh),'insufficient_frame_path_scenes':insufficient,'validation_scenes_available':len(vsc),'validation_scenes_already_used':sum((plat,s) in old for s in vsc),'required_fresh_scenes':nd+nc,'model_exposure':'official union-training-exposed auxiliary' if split=='train' else 'official validation; research-fresh scenes; union checkpoint'}
  assert len(fresh)>=nd+nc,(plat,desc[plat])
  for part,scenes,n in [('dev',fresh[:nd],nfd),('confirm',fresh[nd:nd+nc],nfc)]:
   desc[plat][part+'_scenes']=scenes
   for scene in scenes:
    pool=framepools[scene]
    pool=sorted(pool,key=lambda q:rank(plat+'/'+scene+'/'+q.name));assert len(pool)>=n,(plat,scene,len(pool),n)
    for q in pool[:n]:
     key=q.relative_to(D).as_posix();chosen[part].append({'id':rank(key)[:16],'platform':plat,'split':split,'scene':scene,'frame':q.name,'meta_path':str(q/'meta_info.json'),'lidar_path':str(q/('lidar.npy' if plat=='waymo' else 'lidar.bin'))})
 assert not {(s['platform'],s['scene']) for s in chosen['dev']}&{(s['platform'],s['scene']) for s in chosen['confirm']}
 out={'id':'3eed_round4_fresh_research_scenes_v1','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_seed':SEED,'selection':'SHA256(seed|platform/scene), dev prefix then confirm prefix; frames SHA256(seed|platform/scene/frame), fixed count each scene. File existence only; eligibility requires >=max(dev frames per scene, confirm frames per scene) paths; insufficient scenes recorded before allocation. No metadata/point files opened. Missing caption/invalid data later FAILS, no silent replacements.','old_scene_count':len(old),'old_frame_count':len(oldf),'old_scene_list':sorted(map(list,old)),'evidence_sha256':evidence,'platforms':desc,'cohorts':chosen,'P0_CPU_val_provenance_limit':'P0 retained item indices but not source scene IDs for its four CPU validation item checks. Recovering their annotation-order mapping would require opening metadata before this path-only lock. They are not falsely claimed excluded. P0 sourceval loader audit was not model inference/scoring. This residual prior CPU-label exposure must remain disclosed.','caption_rule':'Only ground_info[0].caption. Preparers open JSON containers with labels but never reference bbox/class numeric fields; model reads standalone GT-free caches.','sampling':'3 independent default_rng resamples of raw points, 16384 XYZ each, replacement iff raw N<16384. No artificial rotations beyond official coordinates.','scope':'Primary Waymo val fresh research scenes; Drone/Quad train split fresh research scenes only, official union-training-exposed auxiliary mechanism checks; not held-out target validation.','gpu_authorized':False}
 dest=R/'PATH_LOCK.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');(R/'PATH_LOCK.sha256').write_text(sha(dest)+'\n');print(json.dumps({'sha256':sha(dest),'dev_n':len(chosen['dev']),'confirm_n':len(chosen['confirm']),'platforms':desc},indent=2))
if __name__=='__main__':main()
