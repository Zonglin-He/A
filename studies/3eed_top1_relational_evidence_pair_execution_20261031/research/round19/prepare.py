"""Input-only A64 development and new-content H32 registered evaluation."""
from pathlib import Path
import sys,hashlib,collections
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round19'))
from common import *
def main():
 inp=EXP.parent/'input';inp.mkdir(parents=True,exist_ok=False)
 A=read(ROOT/'research/round17/input/A64_PRIVATE.json');idx=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json');inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={x['image_file_sha256']:x for x in inv};full={r['id']:r for r in read(ROOT/'research/round8/experiment/full5912_gpu_inputs.json')};used=[]
 for p in ('research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json'):used+=read(ROOT/p)
 used_paths={r['imagepath'] for r in used};badim=set();badpc=set()
 for r in used:
  x=lookup[sha(r['imagepath'])];badim.add(x['decoded_pixel_sha256']);badpc.add(x['canonical_XYZ_sha256'])
 available=[]
 for alias,fr in idx.items():
  x=lookup.get(fr['image_sha256'])
  if x and x['decoded_pixel_sha256'] not in badim and x['canonical_XYZ_sha256'] not in badpc:
   r=full[x['canonical_id_PRIVATE']];assert r['imagepath']==fr['imagepath'];available.append((hashlib.sha256(('R19_H32|'+x['pair_group_sha256']).encode()).hexdigest(),r,alias,x))
 H=[];taken=set();hashes=[]
 for platform in ('drone','quad'):
  per=[]
  for rank,r,alias,x in sorted(available):
   if r['platform']!=platform or x['decoded_pixel_sha256'] in taken:continue
   per.append(dict(r,stage='H',alias=f'H_{platform}_{len(per):03d}',frame_alias=alias));taken.add(x['decoded_pixel_sha256']);hashes.append(x)
   if len(per)==16:break
  assert len(per)==16;H+=per
 A=[dict(r,stage='A',alias=f'A_{r["platform"]}_{i:03d}') for i,r in enumerate(A)]
 dump(inp/'A64_PRIVATE.json',A);dump(inp/'H32_PRIVATE.json',H);dump(inp/'ALL96_PRIVATE.json',A+H);dump(inp/'H_FRAME_INDEX_PRIVATE.json',{r['frame_alias']:idx[r['frame_alias']] for r in H});dump(inp/'H_CONTENT_HASHES_PRIVATE.json',hashes)
 dump(inp/'EXPOSURE.json',{'A64':'Exposed development cases and historical G used for diagnosis.','H32':'Input-hash selected32current-unused decoded RGB/XYZ; excludesR15source96/regression140,R17AB96,R18C32. Existing whole-image masks reused; no currentG/rules/fit/score used for selection.','all_target_VAL_historically_exposed':True,'not_blind_or_independent_scene_confirmation':True,'H_current_input_G_and_episodic_fit':'Permitted at evaluation, fixed rules; no cross-sample transfer.','H_counts':dict(collections.Counter(r['platform'] for r in H)),'H_scene_counts':{p:len({r['scene'] for r in H if r['platform']==p}) for p in ('drone','quad')}})
 dump(EXP/'AUTHORIZATION.json',{'instruction':'Execute cached ranking audit, current-Top1 explicit relational G, same-information direct rerank and frozen-box pairwise projection adaptation. Evidence destruction diagnostic only; preserve all negatives.','source_immutable':True,'boxes_frozen':True,'projection_only':True,'no_teacher_EMA':True,'no_cross_observation_transfer':True,'GT_only_after_prediction_seal':True})
 print('R19_REGISTERED',len(A),len(H))
if __name__=='__main__':main()
