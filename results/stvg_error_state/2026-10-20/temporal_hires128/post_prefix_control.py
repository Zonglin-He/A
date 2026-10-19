"""Post-main-GT explanatory native conditional-end control, not a new candidate."""
from pathlib import Path
import sys,time,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
from scripts.score_stvg_where_when_event128_v1 import paired,avg
D=R/'artifacts/stvg_temporal_hires128_v1';P=Path(__file__).resolve().parent
p=read(D/'PLAN_LOCK.json');rows=[];files=[]
# Added after main aggregate was available: an explanatory attribution control.
# Neither prior selection nor sealed main outputs are changed.
for r in p['records']:
 z=load(D/'predictions'/f"{r['alias']}.pt");q=z['outputs']['Native']['q'];a0=z['outputs']['Native']['interval'][0];i=z['frame_ids'].index(a0);mass=q[i].sum();c=q.copy()if mass<=0 else np.zeros_like(q)
 if mass>0:c[i]=q[i]/mass
 files.append(save(D/'post_prefix_controls'/f"{r['alias']}.pt",dict(alias=r['alias'],q=c,interval=final_mbr(z['frame_ids'],c)['interval'],start_index=i,available=bool(mass>0),GT_read=False)))
put(D/'POST_PREFIX_CONTROL_BARRIER.json',dict(count=128,files=files,after_main_GT=True,explanatory_only=True,not_prespecified_or_confirmation=True,created_epoch=time.time(),code_sha256=sha(__file__),no_main_prediction_change=True))
from scripts.score_stvg_opd_p1_v1 import truths
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
gt,spans,_=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];checks=0
for r in p['records']:
 z=load(D/'predictions'/f"{r['alias']}.pt");n=load(D/'post_prefix_controls'/f"{r['alias']}.pt");vals={};exps={};i=n['start_index'];ids=np.array(z['frame_ids']);js=np.flatnonzero(n['q'][i]>0)
 for space,boxes in z['trajectories'].items():
  tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
  for name,iv in [('NativeConditionalEnd',n['interval']),('TeacherConditionalEnd',z['outputs']['StudentStartEnd']['interval'])]:
   v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;checks+=1;vals[space+'_'+name]=v
  q0=z['outputs']['Native']['q'][i];qh=z['observation']['q'][i]
  if q0.sum()>0 and qh.sum()>0:
   candidates=np.flatnonzero((q0+qh)>0);v=np.array([tube.score([ids[i],ids[j]+1])['v']for j in candidates]);t=np.array([tube.score([ids[i],ids[j]+1])['t']for j in candidates]);q0=q0[candidates]/q0.sum();qh=qh[candidates]/qh.sum();exps[space]=dict(v=float((qh-q0)@v),t=float((qh-q0)@t))
 for name in ['NativeConditionalEnd','TeacherConditionalEnd']:vals['LiveMean_'+name]={k:avg([vals[f'Live{s}_{name}'][k]for s in p['seeds']])for k in ['v','t','s']}
 exp={k:avg([exps.get('Live'+str(s),{}).get(k)for s in p['seeds']])for k in ['v','t']};rows.append(dict(alias=r['alias'],values=vals,common_student_start_expected_teacher_delta=exp,available=n['available']))
summary=dict(status='pass',posthoc_explanation=True,added_after_main_results=True,not_main_candidate=True,no_new_models=True,no_new_GT_annotations=True,all128_control_outputs_sealed_before_control_score=True,no_main_outputs_changed=True,official_checks=checks,rows=rows,point_teacher_minus_native_conditional={k:paired([100*(r['values']['LiveMean_TeacherConditionalEnd'][k]-r['values']['LiveMean_NativeConditionalEnd'][k])for r in rows])for k in ['v','t']},expected_teacher_minus_native_at_same_student_start={k:paired([100*r['common_student_start_expected_teacher_delta'][k]for r in rows if r['common_student_start_expected_teacher_delta'][k]is not None])for k in ['v','t']})
put(P/'POSTHOC_PREFIX_CONTROL.json',summary);print({k:v for k,v in summary.items()if k not in ['rows']})
