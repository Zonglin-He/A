"""Metadata-only independent content-group and complete-expression audit."""
from pathlib import Path
import json,collections
R=Path(__file__).resolve().parents[2];H=R/'research/round131'
rd=lambda p:json.loads(Path(p).read_text())
def main():
 groups={g['content_group_PRIVATE']:g for g in rd(H/'CONTENT_GROUPS_PRIVATE.json')};reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');anchors=rd(H/'ANCHORS_PRIVATE.json');selected={a['content_group_PRIVATE'] for a in anchors};old=set()
 for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json'):old.add(str(Path(e['row']['metapath']).resolve()))
 for filename in ['research/round105/FULL_PAIRS_PRIVATE.json','research/round109/PAIRS_PRIVATE.json']:
  for p in rd(R/filename):
   for role in ['A','B']:old.add(str(Path(p[role]['metapath']).resolve()))
 count=collections.Counter();expressions=0
 for key in selected:
  g=groups[key];assert not g['old_development_content'] and not g['protected_content_group']
  assert not any(str(Path(f['metapath']).resolve()) in old or not f['eligible_for_extra_media'] for f in g['frames'])
  assert {a['A']['frame_PRIVATE'] for a in anchors if a['content_group_PRIVATE']==key}=={f['frame_PRIVATE'] for f in g['frames']}
  count[g['platform']]+=1
 for pl in ['waymo','drone','quad']:
  raw=rd(R/f'research/round97/input/TARGET_{pl.upper()}_PRIVATE.json');paths={str(Path(a['A']['metapath']).resolve()) for a in anchors if a['A']['platform']==pl};expected={r['id'] for r in raw if str(Path(r['metapath']).resolve()) in paths};expressions+=len(expected)
  for src in ['waymo','drone','quad']:
   if src!=pl:assert {e['row']['id'] for e in reg if e['source']==src and e['target']==pl}==expected
 assert dict(count)=={'waymo':128,'drone':128,'quad':128} and len(reg)==2*expressions==1798
 out={'status':'PASS_WHOLE_CONTENT_GROUPS_ALL_ALIASES_DESCRIPTIONS_BOTH_SOURCES','content_groups':len(selected),'frame_paths':len(anchors),'expressions':expressions,'direction_inputs':len(reg),'old_content_groups_selected':0,'protected_content_groups_selected':0,'new_media_or_GT_reads':False};(H/'GROUP_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
