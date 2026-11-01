import sys,hashlib,collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round23'))
from common23 import read,dump,sha
inp=ROOT/'research/round24/input';dev=read(ROOT/'research/round23/input/ALL272_PRIVATE.json');idx=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json');inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={r['image_file_sha256']:r for r in inv};full={r['id']:r for r in read(ROOT/'research/round8/experiment/full5912_gpu_inputs.json')};old=list(dev)
for name in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:old+=read(ROOT/name)
im={lookup[sha(r['imagepath'])]['decoded_pixel_sha256'] for r in old};pc={lookup[sha(r['imagepath'])]['canonical_XYZ_sha256'] for r in old};candidates=collections.defaultdict(list)
for alias,fr in idx.items():
 x=lookup.get(fr['image_sha256'])
 if x and x['decoded_pixel_sha256'] not in im and x['canonical_XYZ_sha256'] not in pc:candidates[(x['platform'],fr['scene_PRIVATE'])].append((hashlib.sha256(('R24_M64|'+x['pair_group_sha256']).encode()).hexdigest(),alias,x))
result=[];hashes=[];seenim=set();seenpc=set()
for platform in ['drone','quad']:
 pools=[sorted(v) for (p,s),v in sorted(candidates.items()) if p==platform];n=0
 print('AVAILABLE',platform,sum(map(len,pools)))
 while n<32:
  advanced=False
  for pool in pools:
   while pool and (pool[0][2]['decoded_pixel_sha256'] in seenim or pool[0][2]['canonical_XYZ_sha256'] in seenpc):pool.pop(0)
   if not pool:continue
   _,alias,x=pool.pop(0);seenim.add(x['decoded_pixel_sha256']);seenpc.add(x['canonical_XYZ_sha256']);result.append(dict(full[x['canonical_id_PRIVATE']],stage='M',alias=f'M_{platform}_{n:03d}',frame_alias=alias));hashes.append(x);n+=1;advanced=True
   if n==32:break
  assert advanced,'insufficient input-disjoint observations'
dump(inp/'D208_PRIVATE.json',dev[:208]);dump(inp/'V64_PRIVATE.json',dev[208:]);dump(inp/'M64_PRIVATE.json',result);dump(inp/'ALL336_PRIVATE.json',dev+result);dump(inp/'M_CONTENT_HASHES_PRIVATE.json',hashes)
dump(inp/'EXPOSURE.json',{'development':'D208 repeated exposure and K64 from R23 now layer-validation development','confirmation':'M64 decodedRGB and canonicalXYZ disjoint prior recent manifests; fixed by input hash and scene round robin before R24 results','historical_full_VAL_scoring':True,'not_pristine_blind_scenes':True,'n':64,'scenes':{p:len({r['scene'] for r in result if r['platform']==p}) for p in ['drone','quad']}})
spec={'question':'Does external G provide reliable geometry-conditioned instance preferences, and which ranking-update interface generalizes beyond fitted observations?','source':'Waymo-only epoch100 sha256 0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd','frozen':'all boxes,backbones,decoder,regression; model eval; no EMA','interfaces':['interaction residual ranking adapter 64 weights + intercept','last query projection Linear(288,64)','full query matching projection MLP 288-288-288-64','full text matching projection MLP 288-288-288-64'],'controls':['source','current Qwen G direct no-update','remove Qwen subject and reference checks, retain masks/DINO/pointmapping/globalposition','CLIP ViT-B32 replacing Qwen on identical crops/phrases','same A evidence nonparametric similarity memory on unseen observations'],'protocol':'per-platform shared state fit on unlabeled D208 G, choose interface/config on exposed V64 labels, freeze before M64 predictions/scoring; no M64 feedback used to update transfer model','optimization':'AdamW, candidate-level teacher-score distillation and top1 pair margin, source anchor, fixed finite LR/step grid; exact details locked before fits','selection':'balanced mIoU primary, aggregate net benefit, retain all harms; no all-negative/perfect gate','limits':'all target VAL historical labels exposed; M new content not new independent scene; no full Reg/TDA/RLCF reproduction claim','run':'registered_before_any_R24_model_output'}
dump(ROOT/'research/round24/experiment/PLAN_LOCK.json',spec)
print('M_SELECTED',len(result))
