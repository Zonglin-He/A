"""Source-only module training inputs; target content selected before outputs."""
import sys,hashlib,collections,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round25'))
from common25 import *

def main():
 inp=EXP.parent/'input';result={};sourceframes={}
 for split in ['train','val']:
  order=read(ROOT/f'research/round6/source_data_audit/OFFICIAL_{split.upper()}_ANNO_ORDER_PRIVATE.json');containers={};rr=[]
  for i,x in enumerate(order):
   mp=x['meta_path'];assert sha(mp)==x['meta_sha256']
   if mp not in containers:containers[mp]=read(mp)
   m=containers[mp];caption=m['ground_info'][x['object_index']]['caption'];assert hashlib.sha256(caption.encode()).hexdigest()==x['caption_sha256'];sid=hashlib.sha256(('R25_source|'+split+'|'+mp+'|'+str(x['object_index'])).encode()).hexdigest()[:16];rr.append(dict(id=sid,platform='waymo',scene=x['scene'],pointpath=x['lidar_path'],metapath=mp,imagepath=str(Path(mp).parent/'image.jpg'),caption=caption,suffix=x['waymo_view_suffix'],seed=25000+i,stage='S' if split=='train' else 'SV',alias=('S' if split=='train' else 'SV')+f'_{i:04d}',object_index=x['object_index']))
   assert Path(rr[-1]['imagepath']).is_file()
  if split=='val':rr=sorted(rr,key=lambda x:hashlib.sha256(('R25_source_validation|'+x['id']).encode()).hexdigest())[:512]
  result[split]=rr;dump(inp/('S3682_PRIVATE.json' if split=='train' else 'SV512_PRIVATE.json'),rr)
 old=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');idx=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json');inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={r['image_file_sha256']:r for r in inv};full={r['id']:r for r in read(ROOT/'research/round8/experiment/full5912_gpu_inputs.json')};history=list(old)
 for name in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:history+=read(ROOT/name)
 im={lookup[sha(r['imagepath'])]['decoded_pixel_sha256'] for r in history};pc={lookup[sha(r['imagepath'])]['canonical_XYZ_sha256'] for r in history};pools=collections.defaultdict(list)
 for alias,fr in idx.items():
  x=lookup.get(fr['image_sha256'])
  if x and x['decoded_pixel_sha256'] not in im and x['canonical_XYZ_sha256'] not in pc:pools[(x['platform'],fr['scene_PRIVATE'])].append((hashlib.sha256(('R25_N64|'+x['pair_group_sha256']).encode()).hexdigest(),alias,x))
 fresh=[];hashes=[];seenim=set();seenpc=set()
 for platform in ['drone','quad']:
  groups=[sorted(v) for (p,s),v in sorted(pools.items()) if p==platform];n=0
  while n<32:
   advanced=False
   for group in groups:
    while group and (group[0][2]['decoded_pixel_sha256'] in seenim or group[0][2]['canonical_XYZ_sha256'] in seenpc):group.pop(0)
    if not group:continue
    _,alias,x=group.pop(0);seenim.add(x['decoded_pixel_sha256']);seenpc.add(x['canonical_XYZ_sha256']);fresh.append(dict(full[x['canonical_id_PRIVATE']],stage='N',alias=f'N_{platform}_{n:03d}',frame_alias=alias));hashes.append(x);n+=1;advanced=True
    if n==32:break
   assert advanced,'not enough fresh target content'
 dump(inp/'N64_PRIVATE.json',fresh);dump(inp/'N_CONTENT_HASHES_PRIVATE.json',hashes);dump(inp/'T400_PRIVATE.json',old+fresh);dump(inp/'ALL4594_PRIVATE.json',result['train']+result['val']+old+fresh)
 sourcegt={}
 for split in ['train','val']:
  orig=read(ROOT/f'research/round6/source_data_audit/OFFICIAL_{split.upper()}_ANNO_ORDER_PRIVATE.json');by={(x['meta_path'],x['object_index']):x for x in orig}
  for r in result[split]:sourcegt[r['id']]=by[(r['metapath'],r['object_index'])]['source_GT7_float32']
 dump(inp/'SOURCE_GT_PRIVATE.json',sourcegt)
 dump(inp/'EXPOSURE.json',{'source_train_n':3682,'source_validation_n':512,'target_old_exposed_n':336,'target_new_content_n':64,'target_full_VAL_historical_scoring':True,'new_target_not_pristine_blind_scenes':True,'no_target_supervised_training_or_model_selection_allowed':True,'target_scored_only_after_source_model_frozen':True,'source_train_scenes':len({x['scene'] for x in result['train']}),'source_validation_scenes':len({x['scene'] for x in result['val']}),'new_target_scene_clusters':{p:len({r['scene'] for r in fresh if r['platform']==p}) for p in ['drone','quad']}})
 dump(EXP/'PLAN_LOCK.json',{'claim':'Can source-trained small RGB-point-language feature fusion improve frozen-box candidate matching on new platforms without Qwen feedback?','source_model':'Waymo-only epoch100,sha256 0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd','train':'new fusion and matching parameters on3682Waymo source labels only; rest frozen','RGB':'frozen ImageNet ResNet34 layer2/layer3 dense features, calibrated candidate ROIAlign tokens; full language and point query condition cross-attention/gating','stage':'first establish source fusion and frozen cross-domain performance; no random-layer target TTA','controls':['original XYZ+text source','source-trained matching with same fusion capacity but no RGB','RGB feature fusion','same-information frozen visual/prototype direct-use where feasible','RGB-zero and RGB-permutation readouts of trained fusion'],'selection':'Waymo VAL512 only; target labels excluded; fixed finite training budget after pilot','targets':'400observations including64new content,32/platform; historical VAL exposure disclosed','pilot':'32 source observations for timing/projection, do not select based on target scores','budget':'cache frozen features; at most30 fusion epochs,2seeds or optimizer configs if necessary; source-only validation chooses','coordinate_policy':'native6D boxes/yaw0 all methods, no box refinement','all_negative_cases_retained':True,'run_state':'registered_before_R25_outputs'})
 print('SELECT25',{k:len(v) for k,v in result.items()},'targets',len(old+fresh))
if __name__=='__main__':main()
