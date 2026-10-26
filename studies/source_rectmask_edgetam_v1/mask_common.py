"""Only original Source predictions prompt EdgeTAM; no referring-expert input."""
from pathlib import Path
import os,sys,time,json,hashlib,shutil
R=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
sys.path.insert(0,str(R/'studies/fresh_external_comparison_v1'))
from edge_common_v1 import read,sha,atomic,identity,digest,save,rows,parent
S=R/'studies/edgetam_comparison_v1';sys.path.insert(0,str(S))
D=R/'outputs/source_rectmask_edgetam_v1';P=R/'results/source_rectmask_edgetam_v1'
ARMS=['Source','EdgeTAM'];PAIRS=[('EdgeTAM','Source')]
def validate():
 p=read(D/'plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
def resources(start):
 assert time.perf_counter()-start<259200 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:'))>=8*2**30
 assert sum(p.stat().st_size for p in D.rglob('*')if p.is_file())<160*2**30
def source_fields(cap):
 import numpy as np,torch
 from stvg.readout import prediction
 from stvg.positions import observation_positions
 native=prediction(cap['native_logits'],torch.as_tensor(cap['Source']),cap['records'],cap['frame_ids']);pos=observation_positions(cap['frame_ids'],native['indices']);assert list(pos)==list(cap['positions8'])
 assert np.array_equal(native['boxes'].cpu().numpy(),cap['Source'])
 return np.asarray(pos),np.asarray(native['physical_interval']),np.asarray(cap['Source'])[pos].copy()
def payload(ds,row):
 import numpy as np
 from stvg.decode import decode
 cap,origin=parent.selected.suite().load_parent(ds,row);frames,ids=decode(row['metadata'],ds);assert ids==cap['frame_ids'];pos,iv,prompts=source_fields(cap)
 arrays=dict(Source=np.asarray(cap['Source']).copy(),Source_interval=iv.copy(),frame_ids=np.asarray(ids),positions8=pos.copy())
 proof=dict(original_Source_parent=origin,actual_native_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),Source_box_prompts_sha256=hashlib.sha256(prompts.tobytes()).hexdigest(),Source_native_interval=iv.tolist(),Source_selected_indices=pos.tolist(),prompt_provider='original_Source_boxes_only',external_model='EdgeTAM_only',ReferDINO_input_used=False,current_method_event_used=False,GT_read=False)
 from adapter_v1 import rectangular_masks
 masks=rectangular_masks(prompts,*frames.shape[1:3]);proof.update(Source_rectangle_mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest(),rectangle_mask_pixels=[int(m.sum())for m in masks],mask_rule='native pixel centers in halfopen Source rectangle, clipped only by image support',prompt_provider='original_Source_boxes_rasterized_rectangular_masks_only')
 return frames,pos.tolist(),prompts,arrays,proof
def release():
 p=read(R/'outputs/source_prompt_edgetam_v1/complete_published_small_v1.json');assert p['all_remote_verified']
 old=R/'outputs/source_prompt_edgetam_v1';assert read(old/'GPU_exit.json')['code']==0
 for f in ['controller_v1.json','GPU_child.json']:
  w=read(old/f);v=Path('/proc')/str(w['pid'])/'stat';assert not v.exists()or int(v.read_text().split(') ',1)[1].split()[19])!=w['startticks']
 def guard(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path)):
   v=os.fsdecode(args[0]);assert not any(x in v for x in ['/teacher_cache/','/new_current_expert_cache/','/expert_payloads/']), 'No referring-expert input in Source rectangular-mask branch'
 sys.addaudithook(guard);return dict(Source_rectangular_mask_prompts_only=True,old_box_prompt_group_complete=True)
