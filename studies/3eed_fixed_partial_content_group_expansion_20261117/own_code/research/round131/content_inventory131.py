"""Hash-only content grouping, no decode/model/task labels; protects known aliases."""
from pathlib import Path
import json,hashlib,collections,concurrent.futures
R=Path(__file__).resolve().parents[2];H=R/'research/round131';rd=lambda p:json.loads(Path(p).read_text())
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4<<20),b''):h.update(x)
 return h.hexdigest()
def main():
 F=rd(R/'research/round105/FRAMES_PRIVATE.json');old={str(Path(e['row']['metapath']).resolve()) for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json')}
 for file in ['research/round105/FULL_PAIRS_PRIVATE.json','research/round109/PAIRS_PRIVATE.json']:
  for p in rd(R/file):
   for v in ['A','B']:old.add(str(Path(p[v]['metapath']).resolve()))
 missing=sorted({f['pointpath'] for f in F if f['point_sha256'] is None and f['eligible_for_extra_media']});assert all(f['point_sha256'] is not None for f in F if not f['eligible_for_extra_media'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:hashes=dict(zip(missing,pool.map(digest,missing)))
 groups=collections.defaultdict(list)
 for f in F:
  f=dict(f);f['point_sha256']=f['point_sha256'] or hashes[f['pointpath']];key=f['platform']+'|'+f['point_sha256'];f['content_group_PRIVATE']=hashlib.sha256(key.encode()).hexdigest()[:24];groups[f['content_group_PRIVATE']].append(f)
 records=[]
 for key,frames in groups.items():records.append({'content_group_PRIVATE':key,'platform':frames[0]['platform'],'frames':frames,'scenes_PRIVATE':sorted({f['scene_PRIVATE'] for f in frames}),'old_development_content':any(str(Path(f['metapath']).resolve()) in old for f in frames),'protected_content_group':any(not f['eligible_for_extra_media'] for f in frames)})
 (H/'CONTENT_GROUPS_PRIVATE.json').write_text(json.dumps(records,indent=2)+'\n');out=[]
 for pl in ['waymo','drone','quad']:
  gg=[g for g in records if g['platform']==pl];clean=[g for g in gg if not g['old_development_content'] and not g['protected_content_group']];out.append({'target':pl,'all_groups':len(gg),'old_groups':sum(g['old_development_content'] for g in gg),'protected_groups':sum(g['protected_content_group'] for g in gg),'clean_remaining_groups':len(clean),'clean_remaining_frame_paths':sum(len(g['frames']) for g in clean),'remaining_recordings':len({s for g in clean for s in g['scenes_PRIVATE']}),'cross_recording_groups':sum(len(g['scenes_PRIVATE'])>1 for g in clean)})
 (H/'CONTENT_INVENTORY_SUMMARY.json').write_text(json.dumps({'groups':out,'new_point_files_hashed_without_decoding':len(missing),'protected_files_newly_hashed_or_decoded':0,'no_task_GT_or_model_used':True},indent=2)+'\n');print(out)
if __name__=='__main__':main()
