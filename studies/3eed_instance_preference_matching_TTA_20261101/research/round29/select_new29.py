import sys,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round29'))
from common29 import *
def main():
 inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={x['image_file_sha256']:x for x in inv};full={r['id']:r for r in read(ROOT/'research/round8/experiment/full5912_gpu_inputs.json')};history=list(rows())
 for name in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:history+=read(ROOT/name)
 ims={lookup[sha(r['imagepath'])]['decoded_pixel_sha256'] for r in history};pcs={lookup[sha(r['imagepath'])]['canonical_XYZ_sha256'] for r in history};known={x['image_sha256'] for x in read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json').values()};pools=collections.defaultdict(list)
 for x in inv:
  if x['platform'] in ['drone','quad'] and x['image_file_sha256'] not in known and x['decoded_pixel_sha256'] not in ims and x['canonical_XYZ_sha256'] not in pcs:pools[(x['platform'],x['scene_PRIVATE'])].append((hashlib.sha256(('R29_R64|'+x['pair_group_sha256']).encode()).hexdigest(),x))
 result=[];hashes=[];seenim=set();seenpc=set()
 for pl in ['drone','quad']:
  groups=[sorted(v,key=lambda t:t[0]) for (p,s),v in sorted(pools.items()) if p==pl];n=0
  while n<32:
   adv=False
   for pool in groups:
    while pool and (pool[0][1]['decoded_pixel_sha256'] in seenim or pool[0][1]['canonical_XYZ_sha256'] in seenpc):pool.pop(0)
    if pool and n<32:
     _,x=pool.pop(0);r=dict(full[x['canonical_id_PRIVATE']],stage='R',alias=f'R_{pl}_{n:03d}',frame_alias=f'input_frame_{12000+len(result):05d}');result.append(r);hashes.append(x);seenim.add(x['decoded_pixel_sha256']);seenpc.add(x['canonical_XYZ_sha256']);n+=1;adv=True
   assert adv
 assert not ims&seenim and not pcs&seenpc;dump(EXP.parent/'input/R64_PRIVATE.json',result);dump(EXP.parent/'input/R64_CONTENT_HASHES_PRIVATE.json',hashes);dump(EXP/'R64_EXPOSURE_LOCK.json',{'n':64,'platforms':dict(collections.Counter(r['platform'] for r in result)),'scene_clusters':{p:len({r['scene'] for r in result if r['platform']==p}) for p in ['drone','quad']},'RGB_XYZ_disjoint_prior528_and_declared_history':True,'B_not_used_for_feedback_or_fitting':True,'allVAL_historical_task_label_exposure':True,'scene_overlap':True,'no_B_feedback_forward_planned':True,'selector_sha256':sha(__file__)})
 print('R64_SELECTED',dict(collections.Counter(r['platform'] for r in result)))
if __name__=='__main__':main()
