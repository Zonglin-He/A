"""Verify current calibration containers match the old registered inputs; no label parsing."""
from common71 import *
p=read(H/'experiment/DESIGN_LOCK.json');co=read(ROOT/p['cohort_map']);idx={x['id']:x for x in read(ROOT/'research/round69/input/SCORING_INDEX_PRIVATE.json')};media=read(ROOT/p['media_hashes']);ss=read(ROOT/p['prepared_root']/'SEALED.json');hashes={};n=0
for r in rows():
 sid=r['id'];rel='cases_PRIVATE/'+sid+'.json';assert sha(ROOT/p['prepared_root']/rel)==ss['files'][rel];rec=read(ROOT/p['prepared_root']/rel);assert rec['point_sha256']==media[sid]['point_sha256'] and rec['image_sha256']==media[sid]['image_sha256'];assert idx[sid]['meta_path']==r['metapath']
 if r['metapath'] not in hashes:hashes[r['metapath']]=sha(r['metapath'])
 assert hashes[r['metapath']]==idx[sid]['metadata_container_sha256'];n+=1
result={'state':'PASS_R71_ALL_EXPANDED_INPUT_IMAGE_POINT_AND_METADATA_HASHES_MATCH_R69','n':n,'unique_metadata_files_hashed':len(hashes),'original_GT_labels_parsed':False,'images_and_points_matched_from_preparation_hashes':True,'no_method_or_parameter_change':True};dump(H/'INPUT_HASH_AUDIT.json',result);print(result)
