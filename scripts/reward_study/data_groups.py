import os
import json,numpy as np
from pathlib import Path
ROOT=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve();R=ROOT/'outputs/layout_reward_dev/research'
x=json.loads((R/'data_audit.json').read_text());m=json.loads((R/'development_official_membership.json').read_text());aa={q['id']:q for q in json.loads((R/'algorithm_audit.json').read_text())['queries']}
members=set(m['members']);changed=set(m['different']);groups={}
for name,sel in [('in_official67',members),('not_in_official61',None),('in_official_same_labels47',members-changed),('in_official_changed_boxes20',changed)]:
 qs=[q for q in x['queries'] if (q['id'] in sel if sel is not None else q['id'] not in members)]
 groups[name]=dict(n=len(qs),layout_minus_geometry_v_pp=float(np.mean([(q['metrics']['Full']['v']-q['metrics']['GeometryOPD']['v'])*100 for q in qs])),layout_minus_geometry_unobs_s_pp=float(np.mean([(q['metrics']['Full']['unobserved_actual_sIoU']-q['metrics']['GeometryOPD']['unobserved_actual_sIoU'])*100 for q in qs])),note='Original sealed development labels and existing scored metrics only; no labels replaced, no rescoring')
joint={}
for arm,ak in [('LayoutOPD','Full'),('LayoutDirect','LayoutDirect')]:
 rows=[]
 for q in x['queries']:
  if q['GT_anchors']==0:continue
  v=q['metrics'][ak];base=q['metrics']['WithoutOPD'];a=aa[q['id']]['arms'][arm]
  rows.append(dict(id=q['id'],public_id=q['public_id'],reward_delta=a['reward_delta'],obs_delta_pp=100*(v['observed_actual_sIoU']-base['observed_actual_sIoU']),unobs_delta_pp=100*(v['unobserved_actual_sIoU']-base['unobserved_actual_sIoU']),v_delta_pp=100*(v['v']-base['v']),gradient_norm=a['gradient_norm_mean'],final_state_norm=a['final_state_norm'],anchor_displacement=a['anchor_displacement'],unobserved_displacement=a['unobserved_displacement'],ESS=a.get('ESS_mean'),teacher_IoU=q['teacher_IoU'],GT_anchors=q['GT_anchors'],candidate_rho=q.get('ranking',{}).get('layout_spearman',{}).get('mean')))
 groups_arm={}
 for name,fun in [('reward_up_obs_up_unobs_down',lambda r:r['reward_delta']>0 and r['obs_delta_pp']>0 and r['unobs_delta_pp']<0),('reward_up_obs_down',lambda r:r['reward_delta']>0 and r['obs_delta_pp']<0),('reward_up_obs_up_unobs_up',lambda r:r['reward_delta']>0 and r['obs_delta_pp']>0 and r['unobs_delta_pp']>0),('all_reward_up',lambda r:r['reward_delta']>0)]:
  rr=[r for r in rows if fun(r)];groups_arm[name]=dict(n=len(rr),queries=[r['public_id'] for r in rr],means={f:float(np.mean([r[f] for r in rr if r[f] is not None])) for f in ['reward_delta','obs_delta_pp','unobs_delta_pp','v_delta_pp','gradient_norm','final_state_norm','anchor_displacement','unobserved_displacement','teacher_IoU','GT_anchors']})
 joint[arm]=dict(groups=groups_arm,rows=rows)
out=dict(development_distribution_groups=groups,reward_propagation_groups=joint);(R/'data_groups.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(groups,indent=2));print(json.dumps({a:v['groups']for a,v in joint.items()},indent=2))
