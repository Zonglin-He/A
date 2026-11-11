"""Input-only cohorts: exclude DEV/FINAL frames, repeated-place train sequences.
GT boxes/classes are not materialized; captions/sequence/time/file identity only.
"""
from pathlib import Path
import json,hashlib,re,collections,ijson,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round69';DATA=R/'crossplatform/dataset_full/3eed_dataset';SPLIT=R/'crossplatform/upstream/data/splits'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def norm(s):return ' '.join(s.replace(',',' ,').split())
def header_captions(path):
 header={};caps=[];j=-1
 with Path(path).open('rb') as f:
  for prefix,event,val in ijson.parse(f):
   if prefix=='ground_info.item' and event=='start_map':j+=1
   elif prefix=='ground_info.item.caption' and event=='string':caps.append((j,norm(val)))
   elif prefix in ['sequence_name','timestamp','image_path','lidar_path_proj'] and event in ['string','number']:header[prefix]=val
 return header,caps
full=read(R/'research/round8/experiment/full5912_gpu_inputs.json');dev=read(R/'research/round24/input/ALL336_PRIVATE.json')
# The protected manifest is used exclusively for exclusion identities, never
# opening its referenced captions/media/annotations or stored prediction/labels.
finalpath=R/'research/NIGHT_20261101/input/FINAL384_PRIVATE.json';final=read(finalpath);excluded={r['id'] for r in dev+final};excludedframes={str(Path(r['pointpath']).resolve()) for r in dev+final};protected_ids=[r['id'] for r in final];del final
canonical=lambda s:re.sub(r'_\d+$','',s);seqs={};stats=[];val_files={};headers_private={};index={};train=[];mediahash={}
for pl in ['drone','quad']:
 seqs[pl]={p:[v.strip() for v in (SPLIT/(pl+'_'+p+'.txt')).read_text().splitlines() if v.strip()] for p in ['train','val']};vseq=seqs[pl]['val'];allowed=[s for s in seqs[pl]['train'] if canonical(s) not in set(map(canonical,vseq))];seqs[pl]['allowed_train']=allowed
 # Raw point byte identity and image byte identity are checked across all VAL
 # and retained TRAIN frames, not only matching directory names/timestamps.
 valpoint=set();valimage=set();valorigin=set()
 for seq in vseq:
  for mp in sorted((DATA/pl/seq).glob('*/meta_info.json')):
   pp=mp.parent/'lidar.bin';ip=mp.parent/'image.jpg';hd,caps=header_captions(mp);headers_private[str(mp)]=hd
   valpoint.add(sha(pp));valimage.add(sha(ip));valorigin.add((hd.get('sequence_name'),str(hd.get('timestamp'))))
 for seq in allowed:
  nf=0;dup=0
  for mp in sorted((DATA/pl/seq).glob('*/meta_info.json')):
   pp=mp.parent/'lidar.bin';ip=mp.parent/'image.jpg';hd,caps=header_captions(mp);headers_private[str(mp)]=hd;ph=sha(pp);ih=sha(ip)
   if ph in valpoint or ih in valimage:dup+=1;continue
   if not pp.is_file() or not ip.is_file():raise FileNotFoundError(str(mp))
   nf+=1
   for obj,caption in caps:
    sid=hashlib.sha256(('R69train|'+pl+'|'+seq+'|'+mp.parent.name+'|'+str(obj)+'|'+caption).encode()).hexdigest()[:16];row={'id':sid,'platform':pl,'scene':seq,'pointpath':str(pp.resolve()),'metapath':str(mp.resolve()),'imagepath':str(ip.resolve()),'caption':caption,'suffix':None,'seed':int(hashlib.sha256(('R69_A|'+sid).encode()).hexdigest()[:16],16)};train.append(row);index[sid]={'meta_path':str(mp.resolve()),'object_index':obj,'metadata_container_sha256':sha(mp)};mediahash[sid]={'point_sha256':ph,'image_sha256':ih}
  stats.append({'platform':pl,'retained_train_sequence_alias':'T'+str(allowed.index(seq)+1),'frames_with_no_exact_VAL_duplicate':nf,'frames_excluded_exact_VAL_duplicate':dup})
 val_files[pl]={'VAL_unique_point_hashes':len(valpoint),'VAL_unique_image_hashes':len(valimage),'VAL_unique_sequence_local_time_keys':len(valorigin),'nominal_train_sequences':len(seqs[pl]['train']),'retained_train_sequences':len(allowed),'same_place_family_sequences_excluded':len(seqs[pl]['train'])-len(allowed)}
# Text-only runtime eligibility mirrors the existing fixed interface's <=256
# token requirement. It is not a source-model correctness/class/GT filter.
from transformers import RobertaTokenizerFast
tokenizer=RobertaTokenizerFast.from_pretrained(str(R/'crossplatform/upstream/data/roberta_base'),local_files_only=True)
rejected_text=[];validtrain=[]
for r in train:
 n=len(tokenizer(r['caption']+' . not mentioned')['input_ids'])
 if n>256 or not r['caption']:rejected_text.append(r['id'])
 else:validtrain.append(r)
hashcache=read(H/'experiment/VAL_MEDIA_HASH_CACHE_PRIVATE.json')
for row in full:
 for k in ['pointpath','imagepath']:
  if row[k] not in hashcache:hashcache[row[k]]=sha(row[k])
# Fingerprints are exclusion-only identity measurements, no image decoding.
protected=read(finalpath);excluded_point_hashes={sha(r['pointpath']) for r in dev+protected};excluded_image_hashes={sha(r['imagepath']) for r in dev+protected};del protected
val=[r for r in full if r['id'] not in excluded and str(Path(r['pointpath']).resolve()) not in excludedframes and hashcache[r['pointpath']] not in excluded_point_hashes and hashcache[r['imagepath']] not in excluded_image_hashes]
for r in validtrain:hashcache[r['pointpath']]=mediahash[r['id']]['point_sha256'];hashcache[r['imagepath']]=mediahash[r['id']]['image_sha256']
def select(rows,perplatform,salt):
 out=[];diag=[]
 # Deduplicate exactly repeated (media,description) records without consulting labels.
 unique={}
 for r in rows:
  key=(hashcache[r['pointpath']],hashcache[r['imagepath']],r['caption'])
  if key not in unique or r['id']<unique[key]['id']:unique[key]=r
 rows=list(unique.values())
 for pl in ['drone','quad']:
  rr=[r for r in rows if r['platform']==pl];by=collections.defaultdict(list)
  for r in rr:by[r['scene']].append(r)
  # 8 temporal bins per sequence; equal round-robin over nonempty scene/bin
  # strata, deterministic hash rank. Distinct raw-point-path per input.
  strata={}
  for scene,rs in sorted(by.items()):
   order=sorted({Path(r['pointpath']).parent.name for r in rs});lookup={s:i for i,s in enumerate(order)};bins=collections.defaultdict(list)
   for r in rs:bins[min(7,lookup[Path(r['pointpath']).parent.name]*8//max(len(order),1))].append(r)
   for k,group in bins.items():strata[(scene,k)]=sorted(group,key=lambda r:(hashlib.sha256((salt+'|'+r['id']).encode()).hexdigest(),r['id']))
  chosen=[];seen=set();media_counts=collections.Counter();limit=1
  while len(chosen)<perplatform:
   moved=False
   for key in sorted(strata):
    group=strata[key]
    while group and group[0]['pointpath'] in seen:group.pop(0)
    eligible_idx=next((j for j,x in enumerate(group) if media_counts[(hashcache[x['pointpath']],hashcache[x['imagepath']])]<limit),None)
    if eligible_idx is not None:
     x=group.pop(eligible_idx);chosen.append(x);seen.add(x['pointpath']);media_counts[(hashcache[x['pointpath']],hashcache[x['imagepath']])]+=1;moved=True
     if len(chosen)==perplatform:break
   if not moved:
    limit+=1
    if limit>4:raise RuntimeError(('insufficient_distinct_queries_with_max4_per_media',pl,len(chosen),perplatform))
  out+=chosen;diag.append({'platform':pl,'n':len(chosen),'scene_count':len({r['scene'] for r in chosen}),'unique_directory_records':len(seen),'unique_point_media':len({hashcache[r['pointpath']] for r in chosen}),'unique_image_media':len({hashcache[r['imagepath']] for r in chosen}),'max_descriptions_per_media_pair':max(media_counts.values())})
 return out,diag
valchosen,vd=select(val,640,'R69_VAL_FIXED1280');trainchosen,td=select(validtrain,512,'R69_TRAIN_FIXED1024');allrows=valchosen+trainchosen
assert len(allrows)==2304 and len({r['id'] for r in allrows})==2304 and not excluded&{r['id'] for r in allrows};assert not excludedframes&{r['pointpath'] for r in allrows};assert not {r['scene'] for r in valchosen}&{r['scene'] for r in trainchosen}
# Resolve VAL object positions using only normalized captions. No GT class/box.
for r in valchosen:
 hd,caps=header_captions(r['metapath']);match=[j for j,cap in caps if cap==r['caption']];assert len(match)==1,('caption_identity_ambiguous',r['id']);index[r['id']]={'meta_path':r['metapath'],'object_index':match[0],'metadata_container_sha256':sha(r['metapath'])}
sets={r['id']:('VAL1280' if i<1280 else 'TRAIN1024') for i,r in enumerate(allrows)}
for name,value in [('ALL2304_PRIVATE',allrows),('VAL1280_PRIVATE',valchosen),('TRAIN1024_PRIVATE',trainchosen),('SCORING_INDEX_PRIVATE',[{'id':r['id'],**index[r['id']]} for r in allrows]),('COHORT_BY_ID_PRIVATE',sets),('DATA_ORIGIN_HEADERS_PRIVATE',headers_private),('SELECTED_MEDIA_HASHES_PRIVATE',{r['id']:{'point_sha256':hashcache[r['pointpath']],'image_sha256':hashcache[r['imagepath']]} for r in allrows})]:dump(H/'input'/(name+'.json'),value)
summary={'state':'R69_COHORTS_LOCKED_BEFORE_MODEL_OR_GT_SCORING','VAL1280':vd,'TRAIN1024':td,'split_audit':val_files,'TRAIN_sequence_audit':stats,'DEV336_and_FINAL384_ID_overlap':0,'DEV336_and_FINAL384_pointpath_overlap':0,'cross_sequence_timestamp_not_used_for_exclusion_because_local_and_empty_origin_sequence':True,'exact_VAL_duplicate_train_frames_excluded':sum(x['frames_excluded_exact_VAL_duplicate'] for x in stats),'train_text_runtime_ineligible':len(rejected_text),'protected_manifest_used_only_for_exclusion_identity':True,'protected_media_bytes_hashed_for_exclusion_identity_only':True,'protected_media_decoded_or_inferred':False,'protected_predictions_or_labels_read':False,'scene_definition':'distinct_official_recording_sequences_and_distinct_place_name_families;not_certified_global_geographic_disjointness_or_absolute_wallclock','timestamp_semantics':'sequence_local_index_may_overlap_across_different_sequences;not_global_time','VAL_history':'ALL5912 historically source/other_methods scored, remaining observations are outside DEV/FINAL for this fixed-method validation, not never-exposed data','TRAIN_index':'all_caption_records_with_runtime_text_length_eligible;no_GTclass_correctness_or_box_filter','no_GT_boxes_classes_materialized_or_used_for_selection':True,'protected_manifest_sha256':sha(finalpath),'excluded_DEV_FINAL_point_hashes':len(excluded_point_hashes),'excluded_DEV_FINAL_image_hashes':len(excluded_image_hashes),'no_selected_VAL_media_fingerprint_in_DEV_FINAL':all(hashcache[r['pointpath']] not in excluded_point_hashes and hashcache[r['imagepath']] not in excluded_image_hashes for r in valchosen),'selected_media_hashes_private':True,'input_manifest_sha256':sha(H/'input/ALL2304_PRIVATE.json')};dump(H/'DATA_SELECTION_AUDIT.json',summary);print(json.dumps(summary,indent=2))
