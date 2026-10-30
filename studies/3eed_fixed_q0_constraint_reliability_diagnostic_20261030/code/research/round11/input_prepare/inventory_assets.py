"""Filename/stat and small config metadata only; no weights bytes/model imports."""
from pathlib import Path
import hashlib,json,os,re,collections
B=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
ROOTS=[B/'crossplatform',B/'avis',B/'affordance',Path.home()/'.cache/huggingface/hub',Path.home()/'.cache/torch/hub/checkpoints']
PRUNE={'env','.git','__pycache__','site-packages','dataset_full','dataset','datasets','JPEGImages','WAV','FEATAudios','public_export','predictions','source_cache','observations','sourcecache','cachefeatures'}
SUFFIXES={'.pth','.pt','.bin','.safetensors','.ckpt','.onnx'};TOKENS=re.compile(r'grounding.?dino|owlv(?:it|2)|(?:^|[\W_])sam(?:2|_vit|_hq|[\W_]|$)|clip|siglip|clap',re.I)
items=[];rootstats=[]
for root in ROOTS:
 visited=weights=0
 if root.is_dir():
  for base,dirs,files in os.walk(root,followlinks=False):
   dirs[:]=[x for x in dirs if x not in PRUNE and not x.startswith(('source160_','geometry160_','normal160_','gtcontrolled160_'))]
   visited+=1
   for name in files:
    p=Path(base)/name
    if p.suffix.lower() not in SUFFIXES:continue
    weights+=1
    relative=str(p.relative_to(root))
    if not TOKENS.search(relative):continue
    st=p.stat();items.append({'asset_path_PRIVATE':str(p),'scope_root_PRIVATE':str(root),'relative_path_PRIVATE':relative,'bytes_stat':st.st_size,'mtime_ns_stat':st.st_mtime_ns,'is_symlink':p.is_symlink(),'weight_payload_read':False,'candidate_identity_only_from_filename_parentdirectory':True})
 rootstats.append({'root_PRIVATE':str(root),'exists':root.is_dir(),'directories_visited':visited,'weight_filename_candidates_all_models':weights})
clip=B/'crossplatform/round8/assets/clip-vit-base-patch32';config=clip/'config.json';metadata=None
if config.is_file():
 raw=config.read_bytes();cfg=json.loads(raw);metadata={'config_sha256':hashlib.sha256(raw).hexdigest(),'model_type':cfg.get('model_type'),'architectures':cfg.get('architectures'),'image_size':cfg.get('vision_config',{}).get('image_size'),'patch_size':cfg.get('vision_config',{}).get('patch_size'),'projection_dim':cfg.get('projection_dim'),'text_context_positions':cfg.get('text_config',{}).get('max_position_embeddings'),'preprocessor_tokenizer_files_stat':{f:(clip/f).is_file() for f in ('preprocessor_config.json','tokenizer_config.json','vocab.json','merges.txt','tokenizer.json','special_tokens_map.json')},'role':'Frozen image/text crop similarity. No native language-conditioned 2D region detector/segmenter or identity proof.','weights_stat_only_no_current_payload_hash_or_safe_load_performed':True}
counts=collections.Counter()
for r in items:
 x=r['relative_path_PRIVATE'].lower()
 for name,pattern in [('GroundingDINO','grounding.?dino'),('OWL','owlv(?:it|2)'),('SAM',r'(?:^|[\W_])sam(?:2|_vit|_hq|[\W_]|$)'),('CLIP','clip'),('SigLIP','siglip'),('CLAP','clap')]:
  if re.search(pattern,x):counts[name]+=1
report={'state':'R11_scoped_local_frozen_association_asset_filename_config_inventory_no_model_load','inventory_program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'roots_PRIVATE':rootstats,'candidate_weights_PRIVATE':items,'candidate_filename_counts_by_family':dict(counts),'CLIP_smallconfig_metadata':metadata,'search_pruning':sorted(PRUNE),'scope_limit':'Scoped three worktrees and two standard cache locations only, not whole machine. Candidate filename/config does not prove checkpoint contents, usability, license or full tokenizer/load compatibility. Package OWL/SAM/DINO implementation code without pretrainedweights is not an installed usable detector.','association_assessment':{'local_CLIP_crop_similarity_available_by_filename_and_config':bool(metadata and any('clip-vit-base-patch32' in r['relative_path_PRIVATE'] for r in items)),'local_stronger_GroundingDINO_OWL_SAM_region_weights_found_in_scope':any(re.search(r'grounding.?dino|owlv(?:it|2)|(?:^|[\W_])sam(?:2|_vit|_hq|[\W_]|$)',r['relative_path_PRIVATE'],re.I) for r in items),'CLAP_audio_semantics_not_RGB_target_region_localizer':True,'q0_seed_CLIP_similarity_or_mask_stability_not_identity_confirmation':True},'no_weightbytes_safeheader_or_tensor_load_model_GPU_network_download_GT_scores_media_points_or_annotation_payload_access':True}
p=D/'LOCAL_ASSOCIATION_ASSET_INVENTORY.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'family_counts':dict(counts),'weight_candidates_matched':len(items),'clip_metadata':metadata,'stronger_region_weights_found':report['association_assessment']['local_stronger_GroundingDINO_OWL_SAM_region_weights_found_in_scope'],'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
