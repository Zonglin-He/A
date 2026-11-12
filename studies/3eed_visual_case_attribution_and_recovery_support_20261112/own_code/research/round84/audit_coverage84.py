"""Native CPU mask verification and direct map-support crosscheck on all cases."""
from core84 import *
torch.set_num_threads(4);rows=read(H84/'ORACLE_PROXY_SUPERVISION_ROWS.json');P=B84/'capture33_attempt001';checks=0
for c,v in zip(read(H84/'experiment/CASES33_PRIVATE.json'),rows):
 r=c['row'];sid=r['id'];g=load(casepath(r) if sid in DEV_IDS else ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(sid+'.npz'));d={k:torch.as_tensor(g[k]) for k in ['depth_valid','cell','visible','query_anchor_indices']};d['image']=torch.empty(0);ms=[masks(r,d,k) for k in [76001,76002,76003]];a=load(P/'samples_PRIVATE'/(sid+'.npz'));q=read(P/(c['case']+'_PRIVATE.json'))['chosen_PRIVATE']['oracle'];inside=in_box84(a['xyz'],a['boxes'][q]);fit=((ms[0][0]|ms[1][0]).numpy()&inside);held=ms[2][0].numpy()&inside;assert int(fit.sum())==v['fit_supervised_point_seeds_inside_oracle'] and int(held.sum())==v['third_supervised_point_seeds_inside_oracle']
 for j in range(3):
  observed=ms[j][0].numpy()&g['visible'];assert np.all(a['frozen_P_maps'][j][~observed]==0);assert not ms[j][0][d['query_anchor_indices'].long()].any();checks+=1
 if not fit.any():assert np.all(a['frozen_P_maps'][:2,inside]==0)
 if not held.any():assert np.all(a['frozen_P_maps'][2,inside]==0)
O=H84/'independent_supervision_support_audit_attempt001';O.mkdir();v={'state':'PASS_R84_NATIVE_MASK_ANCHOR_PROTECTION_AND_ZERO_DIRECT_POINT_RECOVERY_SUPPORT','n':33,'native_mask_checks':checks,'new_models_or_task_scores':0,'posthoc_oracle_proxy_not_instance_mask':True};dump(O/'AUDIT_RESULT.json',v);print(v)
