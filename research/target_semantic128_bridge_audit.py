"""Engineering regression of provider-path target head against prior128 sealed reads.
No candidate, GT, Source RGB/fusion/time, expert, or final prediction is added.
"""
import sys,os,json,time,gc
from pathlib import Path
R=Path(__file__).resolve().parents[1];PROVIDER=R.parent/'stvg-opd';sys.path.insert(0,str(R));sys.path.insert(0,str(PROVIDER))
import torch,numpy as np
from research.joint_event_v1.run import sha,read,put
from stvg.runtime import setup,lease,load_source
from stvg.spatial import SpatialGeometry,evidence_dp
from research.native_event_target_semantic_v1.run import head_read
OLD=R/'outputs/native_event_target_semantic_v1';P=R/'results/target_semantic128_shrink_v1';D=R/'outputs/target_semantic128_shrink_v1'
def main():
 setup();torch.set_num_threads(1);gpu=lease();p=read(OLD/'plan.json');s=read(OLD/'sealed.json');start=time.perf_counter();rows=[];count=0
 try:
  for ds in ['hc2','vidstg']:
   model=load_source(read(PROVIDER/'configs/default.json')['datasets'][ds]);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache()
   for item in [x for x in p['items']if x['dataset']==ds]:
    e=s['files'][item['alias']];assert sha(OLD/e['trace_path'])==e['trace_sha256'];old=torch.load(OLD/e['trace_path'],weights_only=False,map_location='cpu');capfile=PROVIDER/item['capture']['path'];assert sha(capfile)==item['capture']['sha256'];cap=torch.load(capfile,weights_only=False,map_location='cpu');g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);ids=cap['frame_ids'];aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final']
    for kind in ['Global','TargetContext']:
     prob,u,stats,mass=head_read(model,cap,aux,kind);count+=2;assert np.array_equal(prob,old['semantic_p'][kind])and np.array_equal(u,old['head_inputs'][kind]),(item['alias'],kind,'provider-path semantic mismatch')
    rows.append(item['alias'])
   del model;gc.collect();torch.cuda.empty_cache()
  out=dict(kind='Provider-path vs maintained-path semantic128 bitwise regression',code_sha256=sha(Path(__file__)),prior_plan_sha256=sha(OLD/'plan.json'),prior_seal_sha256=sha(OLD/'sealed.json'),count=len(rows),target_global_p_and_u_all_bitwise=True,ASA_head_forwards=count,new_Source_visual_or_fusion_or_time=0,new_expert=0,new_observations=0,new_final_predictions=0,GT_read=False,seconds=time.perf_counter()-start);put(P/'BRIDGE_AUDIT.json',out);put(D/'provider_bridge_receipt.json',out);print(json.dumps(out,indent=2))
 finally:gpu.close()
if __name__=='__main__':main()
