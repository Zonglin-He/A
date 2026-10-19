"""Post-seal interpretation from existing scalar results; no new GT/model access."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_dino_tb_checkpoint32_v1 import D,P,PRE,verify

def main():
 verify(True);assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';z=read(P/'RESULTS_PUBLIC.json');fr=read(P/'FRAME_DIAGNOSTICS_PUBLIC.json')['rows'];sensitivity={};cases=[]
 for ds,alias in [('hc2','NH012'),('vidstg','NV012')]:
  rr=[r for r in z['rows']if r['dataset']==ds];v=[100*(r['values']['B_E_MBR']['v']-r['values']['E_MBR']['v'])for r in rr if r['alias']!=alias];sensitivity[ds]=dict(omitted_for_sensitivity_only=alias,remaining_parents=len(v),remaining_mean_v_pp=sum(v)/len(v),main_result_keeps_all16=True)
 for a in ['NH001','NH003','NH012','NV001','NV011','NV012','NV016']:
  r=next(r for r in z['rows']if r['alias']==a);c=dict(alias=a,values=r['values'],frames=[f for f in fr if f['alias']==a],E_objectives={})
  old=load(PRE/'E'/(a+'.pt'));b=load(D/'predictions'/(a+'.pt'))
  for name,es in [('T',old['estimate']),('B',b['estimate'])]:c['E_objectives'][name]={k:es[k]for k in ['selected','objective','geometry_objective','score_objective','support_frames','rejected_frames','assignments_evaluated']}
  cases.append(c)
 losses=[dict(alias=f['alias'],position=f['position'],top_IoU=f['teachers']['B']['top_IoU'],selected_IoU=f['teachers']['B']['selected_IoU'],loss_pp=100*(f['teachers']['B']['selected_IoU']-f['teachers']['B']['top_IoU']))for f in fr if f['teachers']['B']['selected_IoU']is not None and f['teachers']['B']['top_IoU']-f['teachers']['B']['selected_IoU']>.05]
 put(P/'POST_SEAL_ANALYSIS.json',dict(status='completed',new_GT_model_or_scoring=0,sensitivity=sensitivity,B_selected_vs_actual_top_large_losses=losses,cases=cases,all_original_parent_and_actual_frame_denominators_preserved=True))
 put(P/'DECISION.json',dict(run='completed',measurement='valid',teacher_advantage=dict(HC='mean actual top positive, CI crosses zero; major helpful and harmful cases',Vid='actual top positive paired CI on eight known called parents, conditional support and concentrated gains'),whole_method='B unchanged-E full v gains uncertain in both panels, not reliable 2-3pp; no replacement promotion',reference_selection='B actual top .84718/.95637 replaced by zero-IoU alternatives in NH001/NV011; concrete selection losses remain',resource_decision='Close this finite checkpoint-only comparison; no OPD or new interface experiment automatically started',OPD_tested=False,stop_scope='does not establish OPD failure or universal B inferiority',no_target_GT_tuning=True))
 print(sensitivity,losses,flush=True)

if __name__=='__main__':main()
