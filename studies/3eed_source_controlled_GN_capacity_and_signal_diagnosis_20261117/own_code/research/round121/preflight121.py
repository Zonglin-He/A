"""Source-only cache qualification; no model/GT geometry/media or target inputs."""
from pathlib import Path
import json,hashlib,numpy as np,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round121';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 reg=[];summary=[];labelseal=rd(R/'research/round25/experiment/source_labels_attempt002/SEALED.json')
 for src in ['waymo','drone','quad']:
  file=R/('research/round76/input/SOURCE_VAL128_PRIVATE.json' if src=='waymo' else f'research/round97/input/SOURCE_{src.upper()}_VAL128_PRIVATE.json');rows=rd(file);assert len(rows)==128
  for r in rows:
   sid=r['id']
   if src=='waymo':files={'prefix':R/f'crossplatform/round76/source_prefix640_attempt001/samples_PRIVATE/{sid}.npz','geometry':R/f'crossplatform/round76/prepared976_attempt001/samples_PRIVATE/{sid}.npz','route':R/f'crossplatform/round89/source_features640_attempt001/routes_PRIVATE/{sid}.npz','label':R/f'research/round25/experiment/source_labels_attempt002/samples_PRIVATE/{sid}.npz'};assert sha(files['label'])==labelseal['files'][str(files['label'].relative_to(R/'research/round25/experiment/source_labels_attempt002'))]
   else:files={k:R/f'crossplatform/round97/{src}/samples_PRIVATE/{sid}_{suffix}.npz' for k,suffix in [('prefix','prefix'),('geometry','geometry'),('route','route'),('label','source_labels')]}
   assert all(f.is_file() for f in files.values())
   with np.load(files['prefix']) as z,np.load(files['geometry']) as g,np.load(files['route']) as route,np.load(files['label']) as lab:
    assert z['boxes'].shape==(256,6) and z['vis0'].shape==(1,1024,288) and z['xyz'].shape==(1,1024,3) and len(z['query_points_sample_inds'])==256;assert g['image'].shape==(256,384,3) and g['p2i'].shape[0]==1024;assert lab['ious'].shape==(256,) and np.isfinite(lab['ious']).all() and (lab['ious']>=0).all() and (lab['ious']<=1).all();inside=(abs(z['xyz'][0][None]-z['boxes'][:,None,:3])<=z['boxes'][:,None,3:]/2).all(-1);assert np.array_equal(inside,route['own_support'].astype(bool));assert np.array_equal(z['query_points_sample_inds'],g['query_anchor_indices'])
   frame=str(Path(r['metapath']).resolve());group=int(hashlib.sha256(('R121_source_frame_split|'+src+'|'+frame).encode()).hexdigest()[:16],16)%2;reg.append({'source':src,'row':r,'source_frame_group':group,'files':{k:{'path':str(v.relative_to(R)),'sha256':sha(v)} for k,v in files.items()}})
  rr=[x for x in reg if x['source']==src];frames={str(Path(x['row']['metapath']).resolve()) for x in rr};counts={str(g):sum(x['source_frame_group']==g for x in rr) for g in [0,1]};assert all(n>0 for n in counts.values());summary.append({'source':src,'n':len(rr),'scenes':len({x['row']['scene'] for x in rr}),'unique_frames':len(frames),'predeclared_frame_split_counts':counts})
 assert len(reg)==384;(H/'SOURCE_CACHE_REGISTRY_PRIVATE.json').write_text(json.dumps(reg,indent=2)+'\n');(H/'PREFLIGHT.json').write_text(json.dumps({'state':'PASS384_SOURCE_ONLY_EXISTING_CACHE_BOX_POINT_QUERY_ROLE_AND_LABEL_SCHEMAS','n':384,'sources':summary,'source_GT_values_for_schema_validation_only':True,'new_model_or_GT_geometry':False,'media_opened':False,'target_data_read':False,'FINAL_access':False,'no_source_weights_changed':True},indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':main()
