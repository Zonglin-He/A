"""Only original Source predictions prompt EdgeTAM; no referring-expert input."""
from pathlib import Path
import os,sys,time,json,hashlib,shutil
R=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
sys.path.insert(0,str(R/'studies/fresh_external_comparison_v1'))
from edge_common_v1 import read,sha,atomic,identity,digest,save,rows,parent
S=R/'studies/edgetam_comparison_v1';sys.path.insert(0,str(S))
D=R/'outputs/source_prompt_edgetam_v1';P=R/'results/source_prompt_edgetam_v1'
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
 return frames,pos.tolist(),prompts,arrays,proof
def release():
 ref=R/'outputs/fresh_external_comparison_v1/referformer_v4';state=read(ref/'state_v4.json');assert state['status']=='all256_newexpert_small_comparison_scored_audited_pending_root_analysis_cost_publication'
 for f in ['controller_v4.json','GPU_child.json']:
  w=read(ref/f);p=Path('/proc')/str(w['pid'])/'stat';assert not p.exists()or int(p.read_text().split(') ',1)[1].split()[19])!=w['startticks']
 for ds in ['hc2','vidstg']:assert read(ref/'small_v1'/ds/'reporting_receipt.json')['status']=='verified_fixed128_expert_generalization_all_arrays_Source_teacher_and_means'
 def guard(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path)):
   p=os.fsdecode(args[0]);assert not any(z in p for z in ['/teacher_cache/','/new_current_expert_cache/','/expert_payloads/']), 'Source-prompt EdgeTAM must not read referring-expert payloads'
 sys.addaudithook(guard);return dict(actual_Refer256_audits_complete=True,Source_prompts_only=True)
