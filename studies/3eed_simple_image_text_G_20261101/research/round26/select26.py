import sys,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round26'),str(ROOT/'crossplatform/round7')]
from common26 import *
from camera_projection import read_camera_allowlist
def main():
 inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={x['image_file_sha256']:x for x in inv};full={r['id']:r for r in read(ROOT/'research/round8/experiment/full5912_gpu_inputs.json')};old=read(ROOT/'research/round25/input/T400_PRIVATE.json');history=list(old)
 for name in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:history+=read(ROOT/name)
 ims={lookup[sha(r['imagepath'])]['decoded_pixel_sha256'] for r in history};pcs={lookup[sha(r['imagepath'])]['canonical_XYZ_sha256'] for r in history};idx=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json');known={x['image_sha256'] for x in idx.values()};pools=collections.defaultdict(list)
 for x in inv:
  if x['platform'] in ['drone','quad'] and x['image_file_sha256'] not in known and x['decoded_pixel_sha256'] not in ims and x['canonical_XYZ_sha256'] not in pcs:pools[(x['platform'],x['scene_PRIVATE'])].append((hashlib.sha256(('R26_P64|'+x['pair_group_sha256']).encode()).hexdigest(),x))
 result=[];hashes=[];frames={};seenim=set();seenpc=set()
 for platform in ['drone','quad']:
  groups=[sorted(v,key=lambda t:t[0]) for (p,s),v in sorted(pools.items()) if p==platform];n=0
  while n<32:
   advanced=False
   for pool in groups:
    while pool and (pool[0][1]['decoded_pixel_sha256'] in seenim or pool[0][1]['canonical_XYZ_sha256'] in seenpc):pool.pop(0)
    if not pool:continue
    _,x=pool.pop(0);r=dict(full[x['canonical_id_PRIVATE']],stage='P',alias=f'P_{platform}_{n:03d}',frame_alias=f'input_frame_{10000+len(result):05d}');seenim.add(x['decoded_pixel_sha256']);seenpc.add(x['canonical_XYZ_sha256']);result.append(r);hashes.append(x);f=read_camera_allowlist(r['metapath']);f=__import__('json').loads(__import__('json').dumps(f,default=float));frames[r['frame_alias']]={'platform':platform,'scene_PRIVATE':r['scene'],'image_sha256':sha(r['imagepath']),'point_sha256':sha(r['pointpath']),'metadata_sha256':sha(r['metapath']),'image_intrinsic':f['image_intrinsic'],'image_extrinsic':f['image_extrinsic'],'camera_distortion':f.get('camera_distortion')};n+=1;advanced=True
    if n==32:break
   assert advanced,'insufficient fresh content'
 inp=EXP.parent/'input';dump(inp/'D208_PRIVATE.json',old[:208]);dump(inp/'REPLAY192_PRIVATE.json',old[208:]);dump(inp/'OLD400_PRIVATE.json',old);dump(inp/'P64_PRIVATE.json',result);dump(inp/'ALL464_PRIVATE.json',old+result);dump(inp/'F128_PRIVATE.json',[r for r in old if r['stage']=='N']+result);dump(inp/'P_CONTENT_HASHES_PRIVATE.json',hashes);dump(inp/'P_FRAME_INDEX_PRIVATE.json',frames);dump(inp/'MASK_FRAMES_PRIVATE.json',[{'frame_alias':r['frame_alias'],'imagepath':r['imagepath']} for r in result])
 dump(inp/'EXPOSURE.json',{'development':'D208 repeatedly exposed target development, scalar lambda selection disclosed','replay192':'R23K64/R24M64/R25N64 already exposed; not independent confirmation','newP64':'32/platform outside prior501mask frames, decodedRGB/canonicalXYZ disjoint recent development; chosen by input hashes/scene round robin','full_VAL_historically_scored':True,'not_pristine_blind_or_new_scene_claim':True,'P_scene_clusters':{p:len({r['scene'] for r in result if r['platform']==p}) for p in ['drone','quad']}})
 dump(EXP/'PLAN_LOCK.json',{'question':'Can one image-text matching signal replace hand-added position/reference/gray/crop/residual rules?','source':'same Waymo-only100 frozen checkpoint and6Dcandidateboxes; no training or TTA','fixed':'same independent masks/candidatepool/2Dto3Dassociation across variants;GT never chooses candidates/gates','strict_first_change':'oldG minus explicit position/reference only; retain previous subject query,crops,gray,relative bounded visual residual','then':'independently remove gray,doublecrop and source-relative clipping; raw e mapped toqueries, F=source+lambda*e','fulltext':'do not relabel old subject-only cache as full-description joint scoring; add literal fulltext scene-marked candidate numericYesNo score','CLIP':'shared image/text cosine, native77 truncation audited and full-coverage chunk-average version distinct from cross-attention','development_lambdas':{'Qwen':[0,.1,.25,.5,1,2],'CLIP':[0,1,2,4,8,16]},'selection':'one shared lambda acrossplatforms pervariant on exposedD208; freeze beforeP64 scoring; no additional sweeps afterP','primary':'balancedP64mIoU,report replay400 and192 separately with Acc25/50,sceneclusters,all harms','no_more_source_fusion_training':True,'R25_internal_fusion':'retained separate reference, not claimed G/TTA gain'})
 print('P64_SELECTED',len(result))
if __name__=='__main__':main()
