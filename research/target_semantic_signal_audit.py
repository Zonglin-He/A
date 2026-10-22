"""Postseal semantic signal and optimizer audits; no new task predictions."""
import os,sys,json
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from research.joint_event_v1.run import read,put,sha
from research.native_event_semantic_prototype_v1.core import loop
import numpy as np,torch
from scipy.special import log_expit

def auc(score,inside):
 a=score[inside];b=score[~inside]
 if not len(a)or not len(b):return None
 d=a[:,None]-b[None,:];return float(((d>0)+.5*(d==0)).mean())
def transition(L,sign=-1):
 # Both ends use event evidence; outside samples remain in the SAME offset.
 v=log_expit(L)+np.log(2);a=v.copy();b=v.copy();a[2:]+=log_expit(sign*L[:-2])+np.log(2);b[:-2]+=log_expit(sign*L[2:])+np.log(2);return a[:,None]+b[None,:]
def main():
 torch.set_num_threads(1);D=R/'outputs/native_event_target_semantic_v1';P=R/'results/native_event_target_semantic_v1';Q=R/'outputs/native_event_target_query_v1';plan=read(D/'plan.json');seal=read(D/'sealed.json');report=read(P/'report.json');assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json');assert read(Q/'sealed.json')['count']==128;qr=read(R/'results/native_event_target_query_v1/report.json');assert read(Q/'score_receipt.json')['report_sha256']==sha(R/'results/native_event_target_query_v1/report.json');rows=[];opt=[]
 for group,vv in report['domains'].items():
  ds='vidstg'if group=='vidstg_recheck'else'hc2'
  if group=='HC64':
   lf=R/'data/evaluation/development_labels.jsonl';assert sha(lf)==plan['development_labels_sha256'];gt={x['id']:x for x in map(json.loads,lf.read_text().splitlines())if x['dataset']=='hc2'}
  else:
   lf=plan['labels'][ds];f=R.parent/'stvg-opd'/lf['path'];assert sha(f)==lf['sha256'];gt={x['id']:x for x in map(json.loads,f.read_text().splitlines())}
  for item in [x for x in plan['items']if x['group']==group]:
   k=item['alias'];e=seal['files'][k];assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');q0=tr['q']['NativeG6'];ids=np.asarray(item['row']['metadata']['frame_ids']);truth=gt[item['key']]['interval'];inside=(ids>=truth[0])&(ids<truth[1]);evidence={};scores={};signals={}
   for kind,arm in [('Global','ParentGlobalSemanticLoop2'),('TargetContext','TargetSemanticLoop2'),('ReflectedContext','ReflectedSemanticLoop2'),('WrongTimeContext','WrongTimeSemanticLoop2')]:
    q,steps=loop(tr['semantic_p'][kind],q0);assert np.array_equal(q,tr['q'][arm]);L=steps[1]['L'];evidence[kind]=L;scores[kind]=steps[1]['S'];signals[kind]=dict(frame_event_auc=auc(L,inside),inside_mean_L=float(L[inside].mean())if inside.any()else None,outside_mean_L=float(L[~inside].mean())if(~inside).any()else None,outside_fraction_Lpositive=float((L[~inside]>0).mean())if(~inside).any()else None)
   scores.update(Transition=transition(evidence['TargetContext']),NoTransitionSigns=transition(evidence['TargetContext'],1),ReverseTransition=transition(evidence['TargetContext'][::-1]));i,j=np.triu_indices(len(ids),1);take=q0[i,j]>0;i=i[take];j=j[take];length=ids[j]+1-ids[i];inter=np.maximum(0,np.minimum(ids[j]+1,truth[1])-np.maximum(ids[i],truth[0]));target=inter/(length+truth[1]-truth[0]-inter);lp=np.log(q0[i,j]);win={a:0. for a in scores};total=0
   for t in range(len(i)):
    sel=np.flatnonzero((np.arange(len(i))>t)&(abs(length-length[t])<=.1*length[t])&(abs(lp-lp[t])<=.25)&(abs(target-target[t])>1e-10));direction=np.sign(target[sel]-target[t]);total+=len(sel)
    for a,S in scores.items():
     d=direction*(S[i[sel],j[sel]]-S[i[t],j[t]]);win[a]+=float(((d>1e-10)+.5*(abs(d)<=1e-10)).sum())
   row=next(x for x in vv['rows']if x['query']==k);rows.append(dict(query=k,group=group,duration_group=row['duration_group'],signal=signals,matched_pairs=total,matched_pair_accuracy={a:w/total if total else None for a,w in win.items()},target_time_delta_pp=100*(row['values']['TargetSemanticLoop2']['t']-row['values']['NativeG6']['t']),target_interval_length_delta_seconds=row['values']['TargetSemanticLoop2']['length_seconds']-row['values']['NativeG6']['length_seconds'],context_cells_mean=float(np.mean([x['selected_cells']for x in tr['support_statistics']['TargetContext']])),context_fallback_frames=sum(x['empty_ROI_fallback']for x in tr['support_statistics']['TargetContext']),target_reflected_overlap=float(np.mean([x['target_control_overlap']for x in tr['support_statistics']['TargetContext']]))))
   qe=read(Q/'sealed.json')['files'][k];assert sha(Q/qe['trace_path'])==qe['trace_sha256'];qt=torch.load(Q/qe['trace_path'],weights_only=False,map_location='cpu');values=next(x for x in qr['domains'][group]['rows']if x['query']==k)['values'];paths={}
   for arm,direct in [('TargetQuery','DirectTarget'),('ReflectedQuery','DirectReflected'),('GlobalSemanticQuery','DirectGlobal')]:
    path=qt['paths'][arm]['path'];paths[arm]=dict(initial_loss=path[0]['loss'],final_loss=path[-1]['loss'],loss_decreased=path[-1]['loss']<path[0]['loss'],final_normalized_delta_RMS=path[-1]['normalized_delta_RMS'],gradient_norm_max=max(x['gradient_norm']for x in path[:-1]),TV=float(abs(qt['q'][arm]-qt['q']['NativeG6']).sum()/2),teacher_TV=float(abs(qt['q'][direct]-qt['q']['NativeG6']).sum()/2),time_gain_G6_pp=100*(values[arm]['t']-values['NativeG6']['t']),teacher_time_gain_G6_pp=100*(values[direct]['t']-values['NativeG6']['t']))
   opt.append(dict(query=k,group=group,paths=paths))
 summary={};osummary={}
 for group in report['domains']:
  rr=[x for x in rows if x['group']==group];summary[group]=dict(frame_event_auc={kind:float(np.mean([x['signal'][kind]['frame_event_auc']for x in rr if x['signal'][kind]['frame_event_auc']is not None]))for kind in signals},matched_pair_accuracy={a:float(np.mean([x['matched_pair_accuracy'][a]for x in rr if x['matched_pair_accuracy'][a]is not None]))for a in scores},target_reflected_overlap_mean=float(np.mean([x['target_reflected_overlap']for x in rr])),context_cells_mean=float(np.mean([x['context_cells_mean']for x in rr])),empty_fallback_frames=sum(x['context_fallback_frames']for x in rr));qq=[x for x in opt if x['group']==group];osummary[group]={arm:dict(loss_decreased=sum(x['paths'][arm]['loss_decreased']for x in qq),queries=len(qq),median_final_RMS=float(np.median([x['paths'][arm]['final_normalized_delta_RMS']for x in qq])),median_TV=float(np.median([x['paths'][arm]['TV']for x in qq])),median_teacher_TV=float(np.median([x['paths'][arm]['teacher_TV']for x in qq])),teacher_time_helped=sum(x['paths'][arm]['teacher_time_gain_G6_pp']>1e-10 for x in qq),student_harmed_when_teacher_helped=sum(x['paths'][arm]['teacher_time_gain_G6_pp']>1e-10 and x['paths'][arm]['time_gain_G6_pp']< -1e-10 for x in qq))for arm in paths}
 put(P/'SIGNAL_AUDIT.json',dict(scope='POSTSEAL exposeddevelopmentdiagnostic; transition scores are exploratory candidate-ranking ONLY, no newtaskprediction/MBR oronlineGTselection',rows=rows,summary=summary));put(R/'results/native_event_target_query_v1/OPTIMIZATION_AUDIT.json',dict(scope='POSTSEAL fixedstepqueryoptimization audit; no newupdates orGTselection',rows=opt,summary=osummary));put(R/'outputs/overnight_event_loop_20261023/target_semantic_query_audit_receipt.json',dict(code_sha256=sha(Path(__file__)),target_plan_sha256=sha(D/'plan.json'),target_seal_sha256=sha(D/'sealed.json'),target_report_sha256=sha(P/'report.json'),query_plan_sha256=sha(Q/'plan.json'),query_seal_sha256=sha(Q/'sealed.json'),query_report_sha256=sha(R/'results/native_event_target_query_v1/report.json'),score_adapter_sha256=sha(R/'research/target_query_score_adapter.py'),new_model_calls=0,new_predictions=0,labels=plan['labels'],development_labels_sha256=plan['development_labels_sha256']));print(json.dumps(dict(signal=summary,optimization=osummary),indent=2))
if __name__=='__main__':main()
