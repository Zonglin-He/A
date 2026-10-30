"""Content identity of released RGB+point bytes, no pixels/P/GT decode."""
from pathlib import Path
import collections,hashlib,json,time
B=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
SOURCES={'R9_160':'research/round9/experiment/diagnostic160_gpu_inputs.json','R10_160':'research/round10/input_prepare/diagnostic160_gpu_inputs.json','R11_320':'research/round11/input_prepare/diagnostic320_gpu_inputs.json','R8_40':'research/round8/experiment/smoke40_gpu_inputs.json','R7_20':'research/round7/experiment/smoke_gpu_inputs.json'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8<<20),b''):h.update(z)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def main():
 start=time.perf_counter();fullpath=B/'research/round8/experiment/full5912_gpu_inputs.json';full=read(fullpath);meta=read(D/'TARGET5912_INPUT_FIELDS_PRIVATE.json');lookup={Path(r['meta_path_PRIVATE']).parent:r for r in meta['rows_PRIVATE']};rows=[];groups=collections.defaultdict(list);bytecount=0
 for i,r in enumerate(full):
  image=Path(r['imagepath']);point=Path(r['pointpath']);ih=sha(image);ph=sha(point);key=hashlib.sha256(('R12_physical_frame_v1|'+ih+'|'+ph).encode()).hexdigest();m=lookup[Path(r['metapath']).parent];f=m['input_fields_PRIVATE'];posehash=hashlib.sha256(json.dumps(f['pose'],sort_keys=True).encode()).hexdigest();rec={'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'image_path_PRIVATE':str(image),'point_path_PRIVATE':str(point),'meta_path_PRIVATE':r['metapath'],'folder_index':int(image.parent.name),'image_sha256':ih,'point_sha256':ph,'physical_group':key,'input_timestamp':f['timestamp'],'pose_payload_sha256':posehash,'caption_sha256':hashlib.sha256(r['caption'].encode()).hexdigest()};rows.append(rec);groups[key].append(rec);bytecount+=image.stat().st_size+point.stat().st_size
  if (i+1)%1500==0:print(json.dumps({'hashed':i+1,'n':5912,'elapsed_seconds':time.perf_counter()-start}),flush=True)
 assert len(rows)==5912;byid={r['id_PRIVATE']:r for r in rows};prior={k:read(B/p) for k,p in SOURCES.items()};allprior=[r for x in prior.values() for r in x];assert len(allprior)==len({r['id'] for r in allprior})==700;excluded={byid[r['id']]['physical_group'] for r in allprior}
 scene=[]
 for key in sorted({(r['platform'],r['scene_PRIVATE']) for r in rows}):
  pool=[r for r in rows if (r['platform'],r['scene_PRIVATE'])==key];content={r['physical_group'] for r in pool};available=content-excluded
  scene.append({'platform':key[0],'scene_PRIVATE':key[1],'path_rows':len(pool),'unique_RGB_point_content_groups':len(content),'prior700_excluded_content_groups':len(content&excluded),'available_unique_groups':len(available),'requested_anchors':32,'enough32':len(available)>=32})
 rounds={}
 for name,rr in prior.items():
  cc=[byid[r['id']]['physical_group'] for r in rr];rounds[name]={'path_cases':len(rr),'unique_RGB_point_groups':len(set(cc)),'extra_cases_sharing_physical_input':len(cc)-len(set(cc)),'within_group_different_caption_observed':any(len({byid[r['id']]['caption_sha256'] for r in rr if byid[r['id']]['physical_group']==g})>1 for g in set(cc))}
 overlap={a+'__'+b:len({byid[r['id']]['physical_group'] for r in prior[a]}&{byid[r['id']]['physical_group'] for r in prior[b]}) for j,a in enumerate(prior) for b in list(prior)[j+1:]}
 private={'state':'all5912_input_RGB_point_content_hash_index','program_sha256':sha(__file__),'population_manifest_sha256':sha(fullpath),'input_metadata_table_sha256':sha(D/'TARGET5912_INPUT_FIELDS_PRIVATE.json'),'rows_PRIVATE':rows};p=D/'PHYSICAL_INPUT_HASH_INDEX_PRIVATE.json';assert not p.exists();p.write_text(json.dumps(private,indent=2)+'\n')
 out={'state':'R12_inputcontent_duplicate_and_prior_exclusion_audit_complete','program_sha256':sha(__file__),'private_index_sha256':sha(p),'population_manifest_sha256':sha(fullpath),'path_cases':5912,'unique_content_groups':len(groups),'extra_path_cases_with_duplicate_RGB_point':5912-len(groups),'content_group_timestamp_conflicts':sum(len({r['input_timestamp'] for r in rr})>1 for rr in groups.values()),'content_group_pose_conflicts':sum(len({r['pose_payload_sha256'] for r in rr})>1 for rr in groups.values()),'same_timestamp_with_different_content_pair_count':None,'prior700_pathcases':700,'prior700_unique_contentgroups':len(excluded),'scene_summary_PRIVATE':scene,'prior_round_scope':rounds,'prior_round_content_overlap':overlap,'hash_bytes_total':bytecount,'elapsed_seconds':time.perf_counter()-start,'content_rule':'Exact identical JPEG AND pointfile bytes; conservative exact-duplicate notion, not guaranteed unique physical observations if same scene reencoded or differently sampled.','scope':'Inputfile bytes only; no pixels/P arrays, GTobject/captionpayload beyond existing inputmanifests, score/model/GPU. Old task-query metrics/predictions unchanged. Shared physical input can still represent different language queries, so this finding does not mechanically invalidate old query-case metrics.'};q=D/'PHYSICAL_INPUT_DUPLICATE_AUDIT.json';assert not q.exists();q.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('prior_round_scope','prior_round_content_overlap')},indent=2))
if __name__=='__main__':main()
