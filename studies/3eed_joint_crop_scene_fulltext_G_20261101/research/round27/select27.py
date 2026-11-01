import sys,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round27'),str(ROOT/'crossplatform/round7')]
from common27 import *
from camera_projection import read_camera_allowlist
def main():
 inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={x['image_file_sha256']:x for x in inv};full={r['id']:r for r in read(ROOT/'research/round8/experiment/full5912_gpu_inputs.json')};old=read(ROOT/'research/round26/input/ALL464_PRIVATE.json');history=list(old)
 for name in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:history+=read(ROOT/name)
 ims={lookup[sha(r['imagepath'])]['decoded_pixel_sha256'] for r in history};pcs={lookup[sha(r['imagepath'])]['canonical_XYZ_sha256'] for r in history};idx=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json');known={x['image_sha256'] for x in idx.values()};pools=collections.defaultdict(list)
 for x in inv:
  if x['platform'] in ['drone','quad'] and x['image_file_sha256'] not in known and x['decoded_pixel_sha256'] not in ims and x['canonical_XYZ_sha256'] not in pcs:pools[(x['platform'],x['scene_PRIVATE'])].append((hashlib.sha256(('R27_Q64|'+x['pair_group_sha256']).encode()).hexdigest(),x))
 result=[];hashes=[];frames={};seenim=set();seenpc=set()
 for platform in ['drone','quad']:
  groups=[sorted(v,key=lambda t:t[0]) for (p,s),v in sorted(pools.items()) if p==platform];n=0
  while n<32:
   advanced=False
   for pool in groups:
    while pool and (pool[0][1]['decoded_pixel_sha256'] in seenim or pool[0][1]['canonical_XYZ_sha256'] in seenpc):pool.pop(0)
    if not pool:continue
    _,x=pool.pop(0);r=dict(full[x['canonical_id_PRIVATE']],stage='Q',alias=f'Q_{platform}_{n:03d}',frame_alias=f'input_frame_{11000+len(result):05d}');seenim.add(x['decoded_pixel_sha256']);seenpc.add(x['canonical_XYZ_sha256']);result.append(r);hashes.append(x);f=read_camera_allowlist(r['metapath']);f=__import__('json').loads(__import__('json').dumps(f,default=float));frames[r['frame_alias']]={'platform':platform,'scene_PRIVATE':r['scene'],'image_sha256':sha(r['imagepath']),'point_sha256':sha(r['pointpath']),'metadata_sha256':sha(r['metapath']),'image_intrinsic':f['image_intrinsic'],'image_extrinsic':f['image_extrinsic'],'camera_distortion':f.get('camera_distortion')};n+=1;advanced=True
    if n==32:break
   assert advanced,'insufficient fresh content'
 inp=EXP.parent/'input';dump(inp/'D208_PRIVATE.json',old[:208]);dump(inp/'REPLAY256_PRIVATE.json',old[208:]);dump(inp/'OLD464_PRIVATE.json',old);dump(inp/'OLD400_PRIVATE.json',old[:400]);dump(inp/'Q64_PRIVATE.json',result);dump(inp/'ALL528_PRIVATE.json',old+result);dump(inp/'ONLYQ64_PRIVATE.json',result);dump(inp/'Q_CONTENT_HASHES_PRIVATE.json',hashes);dump(inp/'Q_FRAME_INDEX_PRIVATE.json',frames);dump(inp/'MASK_FRAMES_PRIVATE.json',[{'frame_alias':r['frame_alias'],'imagepath':r['imagepath']} for r in result])

 dump(inp/'EXPOSURE.json',{'development':'D208 repeated target-label exposure; sharedlambda only','replay256':'all464previously scored,includingR26P64;not blind','Q64':'32each,input-hash+scene-roundrobin chosen,RGB/XYZdisjoint recent history;outside old501maskset','historical_fullVAL_exposure':True,'scene_overlap':True})
 print('Q64_SELECTED',len(result))
if __name__=='__main__':main()
