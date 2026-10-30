"""Path-only cohort lock. Never reads sample metadata or LiDAR."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone

R=Path(__file__).resolve().parent; P=R.parent; DATA=P/'dataset_full/3eed_dataset'
SEED='3eed-round2-20261030-cohort-v1'
def order(x): return hashlib.sha256((SEED+'|'+x).encode()).hexdigest()
def main():
    old=[]
    for f in ['manifest.json','replication36/manifest.json','replication36_surface/manifest.json']:
        old += json.loads((P/f).read_text())
    oldframes={(s['platform'],Path(s['meta_path']).parent.parent.name,Path(s['meta_path']).parent.name) for s in old}
    oldscenes={(p,s) for p,s,_ in oldframes}
    selected={'dev':[],'confirm':[]}; description={}
    for plat in ['waymo','drone','quad']:
        scenes=[s.strip() for s in (DATA/'splits'/f'{plat}_val.txt').read_text().splitlines() if s.strip() and (DATA/plat/s.strip()).is_dir()]
        scenes=sorted(scenes,key=lambda s:order(plat+'/'+s))
        if plat=='waymo':
            fresh=[s for s in scenes if (plat,s) not in oldscenes]; conf=fresh[:12]; dev=fresh[12:18]
        elif plat=='drone': conf=scenes[:1]; dev=scenes[1:2]
        else: conf=scenes[:2]; dev=scenes[2:3]
        assert conf and dev and not set(conf)&set(dev)
        description[plat]={'available_val_scene_dirs':len(scenes),'old_exposed_scene_count':sum((plat,s) in oldscenes for s in scenes),'dev_scenes':dev,'confirm_scenes':conf,'new_to_round1_dev_scenes':sum((plat,s) not in oldscenes for s in dev),'new_to_round1_confirm_scenes':sum((plat,s) not in oldscenes for s in conf)}
        for part,sc,n in [('dev',dev,6),('confirm',conf,12)]:
            pools=[]
            for scene in sc:
                frames=[p for p in (DATA/plat/scene).iterdir() if p.is_dir() and (p/'meta_info.json').is_file() and (plat,scene,p.name) not in oldframes]
                pools.append(sorted(frames,key=lambda p:order(plat+'/'+scene+'/'+p.name)))
            chosen=[]
            while len(chosen)<n:
                for pool in pools:
                    if pool and len(chosen)<n: chosen.append(pool.pop(0))
                assert any(pools) or len(chosen)==n
            for p in chosen:
                rel=p.relative_to(DATA).as_posix(); lp=p/('lidar.npy' if plat=='waymo' else 'lidar.bin'); assert lp.is_file()
                selected[part].append({'id':order(rel)[:16],'platform':plat,'scene':p.parent.name,'frame':p.name,'meta_path':str(p/'meta_info.json'),'lidar_path':str(lp)})
    plan={'id':'3eed_round2_cohort54_v1','locked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Path-only lock before round2 caption/geometry/GT scoring/model predictions','seed_string':SEED,'selection':'SHA256 ordering of officialval on-disk scenes; Waymo uses scenes never inround1; scene-disjoint dev-confirm; round-robin hash-orderedframes; exclude81old exactframes','dev_n':18,'confirm_n':36,'caption_rule':'First ground_info caption only; missing input/caption fails preparation, no silent replacement. No numericbbox/class-assisted selection.','old_frame_count':len(oldframes),'old_registry_sha256':hashlib.sha256(json.dumps(sorted(oldframes)).encode()).hexdigest(),'platform_scene_description':description,'cohorts':selected,'input_label_boundary':'Preparer reads official JSON container, extracts allowed input only. Adaptation reads separate cache. No numericbbox used for selection/teacher/hyperparameters; confirmation labels score only after allseals.','no_new_model_or_GT_scoring_yet':True}
    p=R/'COHORT_PATH_LOCK.json'; assert not p.exists();p.write_text(json.dumps(plan,indent=2)+'\n')
    h=hashlib.sha256(p.read_bytes()).hexdigest();(R/'COHORT_PATH_LOCK.sha256').write_text(h+'\n')
    print(json.dumps({'sha256':h,'dev':18,'confirm':36,'scenes':{k:{'dev':len(v['dev_scenes']),'confirm':len(v['confirm_scenes']),'previously_exposed':v['old_exposed_scene_count']} for k,v in description.items()}},indent=2))
if __name__=='__main__':main()
