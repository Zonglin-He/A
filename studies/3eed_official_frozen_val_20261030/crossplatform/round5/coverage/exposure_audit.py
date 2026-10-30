from pathlib import Path
import json,hashlib,collections,zipfile
P=Path(__file__).resolve().parents[2];R=P/'round5/coverage';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=collections.defaultdict(set);evidence={}
for rel in ['manifest.json','replication36/manifest.json','replication36_surface/manifest.json']:
 p=P/rel;evidence[rel]=sha(p)
 for x in json.loads(p.read_text()):
  q=Path(x['meta_path']);old[(x['platform'],q.parent.parent.name,q.parent.name)].add(rel)
for rel in ['round2/COHORT_PATH_LOCK.json','round4/source/PATH_LOCK.json']:
 p=P/rel;evidence[rel]=sha(p)
 for part,rr in json.loads(p.read_text())['cohorts'].items():
  for x in rr:old[x['platform'],x['scene'],x['frame']].add(rel+':'+part)
frames=json.loads((R/'FRAMES_PRIVATE.json').read_text());prior_scene={(a,b) for a,b,c in old};counts={};rows=[]
for x in frames:
 key=x['platform'],x['scene'],x['frame'];exp='exact_frame_previously_used' if key in old else 'new_frame_in_previously_used_scene' if key[:2] in prior_scene else 'not_in_known_prior_scene_registries'
 rows.append({'platform':key[0],'scene':key[1],'frame':key[2],'research_exposure':exp,'prior_evidence':sorted(old.get(key,[]))})
for p in ['waymo','drone','quad']:
 rr=[x for x in rows if x['platform']==p];counts[p]={'frames':len(rr),'frame_exposure':dict(collections.Counter(x['research_exposure'] for x in rr)),'known_previously_used_scenes':len({x['scene'] for x in rr if x['research_exposure']!='not_in_known_prior_scene_registries'}),'known_unexposed_scenes':len({x['scene'] for x in rr if x['research_exposure']=='not_in_known_prior_scene_registries'})}
with zipfile.ZipFile(P/'downloads/3eed_dataset.zip') as z:
 names=z.namelist();coverage=json.loads((R/'COVERAGE.json').read_text());miss={p:{scene:any(name.startswith(f'3eed_dataset/{p}/{scene}/') for name in names) for scene in v['missing_val_scenes']} for p,v in coverage['summary'].items()}
assert not any(v for d in miss.values() for v in d.values())
f=R/'EXPOSURE_FRAMES_PRIVATE.json';f.write_text(json.dumps(rows,indent=2));out={'state':'path_only_exposure_audit','scope':'Known historical inference/scoring manifests, not a claim of globally unseen data','counts':counts,'known_prior_frame_union_all_splits':len(old),'known_prior_scene_union_all_splits':len(prior_scene),'evidence_sha256':evidence,'frames_sha256':sha(f),'source_code_sha256':sha(Path(__file__)),'missing_scene_any_members_in_original_officialZIP':miss,'labels_or_predictions_read':False,'caveats':['Round3 reused exactly round2 paths, not added as independent exposure.','Round4 Drone/Quad used TRAIN scenes; their4+8 frames each are excluded fromVAL pathmatches naturally.','P0 prior4 CPU validation checks did not retain source scene IDs; exact unknown exposure cannot be excluded.','Historical frame contains multiple expressions; previous methods generally used ground_info[0], so exactframe exposure does not imply every expression scored.','No subset here is a fresh preregistered confirmation cohort: this is complete officialVAL descriptive evaluation.']};(R/'EXPOSURE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
