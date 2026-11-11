"""Independent saved-table selection, concentration and repeat summary audit."""
from common68 import *
H=ROOT/'research/round68';p=read(H/'experiment/DESIGN_LOCK.json');tt=read(ROOT/p['score_root']/'ALL_TASK_ROWS_PRIVATE.json');methods=read(H/'ALL_METHOD_ROWS.json');scenes=read(H/'ALL_SCENE_ROWS.json');decision=read(H/'DECISION.json');repeat=read(H/'RANDOM_REPEAT_SUMMARY.json');process=read(H/'ALL_SCALE_PROCESS_ROWS.json')
for m in methods:
 x=np.array([t['IoU'][m['method']] for t in tt]);ref=np.array([t['IoU']['R59'] for t in tt]);delta=x-ref;assert abs(x.mean()*100-m['balanced_mIoU'])<1e-11 and m['improved']==int((delta>1e-10).sum()) and m['worsened']==int((delta< -1e-10).sum());assert abs(m['largest_positive_full336_contribution_pp']-delta.max()*100/336)<1e-11;assert abs(m['remainder_full336_gain_pp']-(delta.sum()-delta.max())*100/336)<1e-11
for s in scenes:
 available=sorted({t['scene_PRIVATE'] for t in tt if t['platform']==s['platform']});rr=[t for t in tt if t['platform']==s['platform'] and t['scene_PRIVATE']==available[int(s['scene_alias'][-2:])-1]];assert len(rr)==s['n'] and abs(np.mean([t['IoU'][s['method']] for t in rr])*100-s['mIoU'])<1e-11
by={m['method']:m for m in methods};values=[by['random_s'+str(j)]['balanced_mIoU'] for j in range(3)];assert values==repeat['seed_mIoU'];assert abs(repeat['mean_mIoU']-sum(values)/3)<1e-11;assert repeat['min_mIoU']==min(values) and repeat['max_mIoU']==max(values);assert abs(repeat['std_population']-np.std(values))<1e-11
for vals,prefix,key in [(GAMMAS,'G_','best_gamma'),(ETAS,'E_','best_eta')]:
 mx=max(by[prefix+tag(v)]['balanced_mIoU'] for v in vals);best=min(v for v in vals if by[prefix+tag(v)]['balanced_mIoU']>=mx-1e-10);assert decision[key]==best
candidates=[('R59_outer_gamma',decision['best_gamma'],by['G_'+tag(decision['best_gamma'])]['balanced_mIoU']),('random_SA3_policy_three_repeat_mean',None,repeat['mean_mIoU']),('R66_parameter_delta_eta',decision['best_eta'],by['E_'+tag(decision['best_eta'])]['balanced_mIoU'])];high=max(x[2] for x in candidates);selected=next(x for x in candidates if x[2]>=high-1e-10);assert (decision['selected_family'],decision['selected_coefficient'],decision['selected_development_mIoU'])==selected
for i,t in enumerate(tt):
 source=read(ROOT/p['scaled_root']/'cases_PRIVATE'/(t['id_PRIVATE']+'.json'))
 for j,eta in enumerate(ETAS):
  r=process[i*4+j];assert r['row']=='DEV_'+str(i+1).zfill(5) and r['eta']==eta
  for k in ['eligible','accepted','loss_before','loss_proposed','loss_final','parameter_delta_L2','C_used']:assert r[k]==source[tag(eta)][k]
out=H/'independent_derived_audit_attempt001';assert not out.exists();out.mkdir();dump(out/'AUDIT_RESULT.json',{'state':'PASS_R68_DEV_SELECTION_REPEATS_CONCENTRATION_AND_PROCESS_TABLES','n':336,'methods':len(methods),'scenes':len(scenes),'scale_process_rows':len(process),'selection_global_not_per_input':True,'random_best_seed_not_selected':True,'new_task_scores':0});print({'derived_audit':'PASS'})
