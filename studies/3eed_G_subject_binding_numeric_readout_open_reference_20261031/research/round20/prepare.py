"""Fixed development144? D112 plus content-disjoint W32; no GT/model outputs."""
from pathlib import Path
import sys,hashlib,collections
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round20'))
from common import *
def main():
 inp=EXP.parent/'input';inp.mkdir(parents=True,exist_ok=False);dev=read(ROOT/'research/round19/input/ALL112_v2_PRIVATE.json');idx=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json');inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={r['image_file_sha256']:r for r in inv};full={r['id']:r for r in read(ROOT/'research/round8/experiment/full5912_gpu_inputs.json')};old=[]
 for name in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json','research/round19/input/ALL112_v2_PRIVATE.json']:old+=read(ROOT/name)
 badim={lookup[sha(r['imagepath'])]['decoded_pixel_sha256'] for r in old};badpc={lookup[sha(r['imagepath'])]['canonical_XYZ_sha256'] for r in old};avail=[]
 for alias,fr in idx.items():
  x=lookup.get(fr['image_sha256'])
  if x and x['decoded_pixel_sha256'] not in badim and x['canonical_XYZ_sha256'] not in badpc:avail.append((hashlib.sha256(('R20_W32|'+x['pair_group_sha256']).encode()).hexdigest(),alias,x))
 W=[];seen=set();hashes=[]
 for platform in ('drone','quad'):
  n=0
  for _,alias,x in sorted(avail):
   if x['platform']!=platform or x['decoded_pixel_sha256'] in seen:continue
   W.append(dict(full[x['canonical_id_PRIVATE']],stage='W',alias=f'W_{platform}_{n:03d}',frame_alias=alias));seen.add(x['decoded_pixel_sha256']);hashes.append(x);n+=1
   if n==16:break
  assert n==16
 dump(inp/'D112_PRIVATE.json',dev);dump(inp/'W32_PRIVATE.json',W);dump(inp/'ALL144_PRIVATE.json',dev+W);dump(inp/'W_CONTENT_HASHES_PRIVATE.json',hashes)
 dump(inp/'EXPOSURE.json',{'D112':'Prior R19 development observations, task and failure exposure acknowledged.','W32':'Fixed input-only hash selection, decodedRGB/XYZ excludeR15source96/regression140,R17AB96,R18C32,R19H32/V16; no currentoutput/GT used.','all_targetVAL_historically_scored':True,'not_pristine_blind_or_independent_scene_confirmation':True,'W_counts':dict(collections.Counter(r['platform'] for r in W)),'W_scene_counts':{p:len({r['scene'] for r in W if r['platform']==p}) for p in ('drone','quad')}})
 dump(EXP/'AUTHORIZATION.json',{'request':'Repair G subject attributes, reference localization and visual readout; real samples, negatives and direct rerank controls; verify supervision before TTA.','source_immutable':True,'boxes_frozen':True,'no_TTA_until_direct_supervision_gate':True,'no_GT_in_G':True,'no_teacher_EMA_or_source_retraining':True})
 print('R20_REGISTERED_D112_W32')
if __name__=='__main__':main()
