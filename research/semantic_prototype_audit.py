"""Separately bound POSTSEAL diagnostic; never constructs online predictions."""
import sys,os,json,time
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records
import torch,numpy as np

def auc(score,y):
 a=score[y];b=score[~y]
 if not len(a)or not len(b):return None
 d=a[:,None]-b[None,:];return float(((d>0)+.5*(d==0)).mean())
def symkl(a,b):
 return float(np.mean((a-b)*np.log(a*(1-b)/(b*(1-a)))))
def main():
 torch.set_num_threads(1);PROVIDER=R.parent/'stvg-opd';stages=[('HC64',R/'outputs/native_event_semantic_prototype_hc64_v1',R/'results/native_event_semantic_prototype_hc64_v1'),('recheck',R/'outputs/native_event_semantic_recheck_v1',R/'results/native_event_semantic_recheck_v1')];allrows=[];bindings={}
 for label,D,P in stages:
  s=read(D/'sealed.json');p=read(D/'plan.json');rep=read(P/'report.json');assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json');bindings[label]=dict(plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),report_sha256=sha(P/'report.json'))
  if label=='HC64':
   lab=R/'data/evaluation/development_labels.jsonl';assert sha(lab)==p['labels_sha256'];gt={x['id']:x for x in map(json.loads,lab.read_text().splitlines())if x['dataset']=='hc2'};items=[dict(alias=x['id'],dataset='hc2',key=x['id'],row=x)for x in records()];values={x['query']:x for x in rep['rows']}
  else:
   gt={};items=p['items'];values={x['query']:x for v in rep['domains'].values()for x in v['rows']}
   for ds,lf in p['labels'].items():
    lab=PROVIDER/lf['path'];assert sha(lab)==lf['sha256'];gt.update({x['id']:x for x in map(json.loads,lab.read_text().splitlines())})
  for item in items:
   key=item['alias'];e=s['files'][key];assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');steps=tr['prototype_steps']['real']if label=='HC64'else tr['prototype_steps'];pp=tr['semantic_p']['motion']if label=='HC64'else tr['semantic_p'];ids=np.asarray(item['row']['metadata']['frame_ids']);truth=gt[item['key']]['interval'];inside=(ids>=truth[0])&(ids<truth[1]);pre=ids<truth[0];post=ids>=truth[1];L=steps[1]['L'];q0=tr['q']['NativeG6'];i,j=np.triu_indices(len(ids),1);sel=q0[i,j]>0;i=i[sel];j=j[sel];length=ids[j]+1-ids[i];over=np.maximum(0,np.minimum(ids[j]+1,truth[1])-np.maximum(ids[i],truth[0]));true=over/(length+truth[1]-truth[0]-over);lp=np.log(q0[i,j]);score=steps[1]['S'][i,j];wins=ties=total=0
   for t in range(len(i)):
    ix=np.flatnonzero((np.arange(len(i))>t)&(abs(length-length[t])<=.1*length[t])&(abs(lp-lp[t])<=.25)&(abs(true-true[t])>1e-10));d=np.sign(true[ix]-true[t])*(score[ix]-score[t]);wins+=int((d>1e-10).sum());ties+=int((abs(d)<=1e-10).sum());total+=len(ix)
   v=values[key];a='SemanticPrototypeLoop2';base=v['values']['NativeG6'];change={m:100*(v['values'][a][m]-base[m])for m in ['t','v']};stats=dict(semantic_class_count=pp.shape[1],mean_temporal_std=float(pp.std(0).mean()),frame_event_auc=auc(L,inside),event_vs_pre_auc=auc(L[inside|pre],inside[inside|pre]),event_vs_post_auc=auc(L[inside|post],inside[inside|post]),pre_post_mean_semantic_symmetricKL=symkl(pp[pre].mean(0),pp[post].mean(0))if pre.any()and post.any()else None,event_outside_semantic_symmetricKL=symkl(pp[inside].mean(0),pp[~inside].mean(0))if inside.any()and(~inside).any()else None,matched_pairs=total,matched_pair_accuracy=(wins+.5*ties)/total if total else None,TV=float(abs(tr['q'][a]-q0).sum()/2),second_minus_first_t_pp=100*(v['values'][a]['t']-v['values']['OneStepPrototype']['t']),interval_length_delta_seconds=v['values'][a]['length_seconds']-base['length_seconds'])
   allrows.append(dict(query=key,panel=label,dataset=item['dataset'],duration_group=v['duration_group'],gain=change,movement=v['movement'][a],statistics=stats))
 out=dict(scope='POSTSEALofflineonly, no new predictions; GTusedonlyforfailuremechanism notonlinefit/selection',rows=allrows,summary={})
 for group in ['HC64','hc2_recheck','vidstg_recheck']:
  rr=[x for x in allrows if x['panel']=='HC64']if group=='HC64'else[x for x in allrows if x['panel']=='recheck'and x['dataset']==group.split('_')[0]];summary={}
  for k in rr[0]['statistics']:
   vv=[x['statistics'][k]for x in rr if x['statistics'][k]is not None];summary[k]=float(np.mean(vv))if vv else None
  summary['short_harmed']=[x['query']for x in rr if x['duration_group']=='short_le5s'and x['gain']['t']< -1e-10];out['summary'][group]=summary
 put(R/'results/native_event_semantic_recheck_v1/MECHANISM_AUDIT.json',out);put(R/'outputs/native_event_semantic_recheck_v1/semantic_audit_receipt.json',dict(code_sha256=sha(Path(__file__)),bindings=bindings,labels_sha256=read(stages[0][1]/'plan.json')['labels_sha256'],fixed_recheck_labels=read(stages[1][1]/'plan.json')['labels'],output_sha256=sha(R/'results/native_event_semantic_recheck_v1/MECHANISM_AUDIT.json'),new_predictions=0,new_model_calls=0,postseal=True));print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
