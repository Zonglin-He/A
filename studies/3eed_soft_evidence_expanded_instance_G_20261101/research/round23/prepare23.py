"""Select fresh current-input observations by content hash, never by performance."""
import sys,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round23'))
from common23 import *
def main():
    inp=EXP.parent/'input';dev=read(ROOT/'research/round20/input/ALL208_PRIVATE.json');idx=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json');inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={r['image_file_sha256']:r for r in inv};full={r['id']:r for r in read(ROOT/'research/round8/experiment/full5912_gpu_inputs.json')};oldrows=[]
    for name in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json','research/round20/input/ALL208_PRIVATE.json']:oldrows+=read(ROOT/name)
    im={lookup[sha(r['imagepath'])]['decoded_pixel_sha256'] for r in oldrows};pc={lookup[sha(r['imagepath'])]['canonical_XYZ_sha256'] for r in oldrows};candidates=collections.defaultdict(list)
    for alias,fr in idx.items():
        x=lookup.get(fr['image_sha256'])
        if x and x['decoded_pixel_sha256'] not in im and x['canonical_XYZ_sha256'] not in pc:candidates[(x['platform'],fr['scene_PRIVATE'])].append((hashlib.sha256(('R23_K64|'+x['pair_group_sha256']).encode()).hexdigest(),alias,x))
    result=[];hashes=[];seen=set()
    for platform in ['drone','quad']:
        pools=[sorted(v) for (p,s),v in sorted(candidates.items()) if p==platform];n=0
        while n<32:
            advanced=False
            for pool in pools:
                while pool and pool[0][2]['decoded_pixel_sha256'] in seen:pool.pop(0)
                if not pool:continue
                _,alias,x=pool.pop(0);seen.add(x['decoded_pixel_sha256']);result.append(dict(full[x['canonical_id_PRIVATE']],stage='K',alias=f'K_{platform}_{n:03d}',frame_alias=alias));hashes.append(x);n+=1;advanced=True
                if n==32:break
            assert advanced,'Insufficient content-disjoint observations'
    dump(inp/'D208_PRIVATE.json',dev);dump(inp/'K64_PRIVATE.json',result);dump(inp/'ALL272_PRIVATE.json',dev+result);dump(inp/'K_CONTENT_HASHES_PRIVATE.json',hashes);dump(inp/'EXPOSURE.json',{'development':'208 already repeatedly exposed, target development allowed and disclosed','confirmation':'64 fixed before new outputs or method selection; decoded RGB and canonical XYZ disjoint prior recent development manifests','all_target_VAL_historically_scored':True,'not_pristine_blind_or_new_independent_scenes':True,'n':64,'per_platform':32,'scenes':{p:len({r['scene'] for r in result if r['platform']==p}) for p in ['drone','quad']}});print('K64_SELECTED',len(result))
if __name__=='__main__':main()
