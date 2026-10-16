"""Read-only saved-score/tensor audit. No model, inference, labels or resampling."""
from pathlib import Path
import sys,json,math,statistics,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write
SEARCH=ROOT/'artifacts/stvg_soft_ep_coordinate128_v1';BASE=ROOT/'artifacts/stvg_component_ablation128_v1'
def main():
 import torch,numpy as np
 torch.set_num_threads(2);completion=read(SEARCH/'SEARCH_COMPLETION006.json');assert len(completion['history'])==42
 sources={};max_error=0.;search_results=[]
 def bind(p):sources[str(p.relative_to(ROOT))]=sha(p)
 def reaggregate(rows,summary):
  nonlocal max_error
  for a in summary:
   metric=a['metric']
   if metric not in rows[0]['metrics']:continue
   vals=[r['metrics'][metric]for r in rows if r['dataset']==a['dataset']and r['arm']==a['arm']and r['metrics'][metric]is not None]
   assert len(vals)==a['n_valid']
   if vals:
    error=abs(math.fsum(vals)/len(vals)-a['mean']);assert error<1e-12;max_error=max(max_error,error)
 bind(SEARCH/'SEARCH_COMPLETION006.json')
 for tr in completion['history']:
  p=SEARCH/'trials'/tr;rows=read(p/'POSTSEAL_SCORE_ROWS.json')['rows'];summary=read(p/'POSTSEAL_SCORE_SUMMARY.json')['summary'];assert len(rows)==1024 and len({(r['case_id'],r['arm'])for r in rows})==1024
  assert read(p/'ROOT_READBACK.json')['status']=='pass'and read(p/'CPU_COMPLETION.json')['count']==128
  reaggregate(rows,summary)
  for n in ['POSTSEAL_SCORE_ROWS.json','POSTSEAL_SCORE_SUMMARY.json','ROOT_READBACK.json','CPU_COMPLETION.json']:bind(p/n)
 for ds in ['hc2','vidstg']:
  for mode in ['M0','M1','M2']:
   tr=completion['best_trial'][ds][mode];rows=read(SEARCH/'trials'/tr/'POSTSEAL_SCORE_ROWS.json')['rows'];arms={}
   for arm in ['F','T','TS_'+mode]:
    rr=[r for r in rows if r['dataset']==ds and r['arm']==arm];assert len(rr)==64
    arms[arm]=100*math.fsum(r['metrics']['m_vIoU']for r in rr)/64
   search_results.append(dict(dataset=ds,mode=mode,trial=tr,viou_percent=arms))
 baseline_rows=read(BASE/'trials/A000/POSTSEAL_ROWS.json')['rows'];baseline={(r['case_id'],r['arm']):r for r in baseline_rows};ablations=[];completed=[]
 for p in sorted((BASE/'trials').iterdir()):
  if not(p/'CPU_COMPLETION.json').exists():continue
  bar=read(p/'POSTSEAL_SCORE_BARRIER.json');cpu=read(p/'CPU_COMPLETION.json');root=read(p/'ROOT_READBACK.json')
  assert cpu['status']=='complete'and cpu['score_barrier_sha256']==sha(p/'POSTSEAL_SCORE_BARRIER.json')and root['status']=='pass'and root['count']==128
  for stem in ['rows','summary','paired']:assert bar[stem+'_sha256']==sha(ROOT/bar[stem+'_path'])
  rows=read(p/'POSTSEAL_ROWS.json')['rows'];summary=read(p/'POSTSEAL_SUMMARY.json')['summary'];assert len(rows)==512 and len({(r['case_id'],r['arm'])for r in rows})==512;reaggregate(rows,summary)
  cmp=read(p/'ACTUAL_BASELINE_COMPARISON.json')['records'];design=read(p/'DESIGN_LOCK.json');completed.append(p.name)
  for ds in ['hc2','vidstg']:
   rr=[r for r in rows if r['dataset']==ds and r['arm']=='TS_M2'];assert len(rr)==64
   delta={r['case_id']:100*(r['metrics']['m_vIoU']-baseline[r['case_id'],'TS_M2']['metrics']['m_vIoU'])for r in rr}
   vv=[x for x in cmp if x['dataset']==ds];assert len(vv)==64
   ablations.append(dict(trial=p.name,variant=design['variant'],dataset=ds,viou_percent=100*math.fsum(r['metrics']['m_vIoU']for r in rr)/64,mean_delta_pp=math.fsum(delta.values())/64,median_delta_pp=statistics.median(delta.values()),worst_delta_pp=min(delta.values()),positive=sum(v>1e-10 for v in delta.values()),negative=sum(v< -1e-10 for v in delta.values()),neutral=sum(abs(v)<=1e-10 for v in delta.values()),per_source_delta_pp=delta,reported_exact_prediction_count=sum(x['prediction_exact']for x in vv)))
  for n in ['DESIGN_LOCK.json','ROOT_READBACK.json','CPU_COMPLETION.json','POSTSEAL_ROWS.json','POSTSEAL_SUMMARY.json','POSTSEAL_PAIRED.json','POSTSEAL_SCORE_BARRIER.json','ACTUAL_BASELINE_COMPARISON.json']:bind(p/n)
 exact={};flat_skips=0;fit_cases=0
 # Independent actual-box/interval equality for the three apparent no-effect deletions.
 b=BASE/'trials/A000';bb=read(b/'STUDENT_GLOBAL_BARRIER.json')['files'];assert len(bb)==128
 for trial in ['A003','A009','A010']:
  if trial not in completed:continue
  other=BASE/'trials'/trial;files={x['case_id']:x for x in read(other/'STUDENT_GLOBAL_BARRIER.json')['files']};counts={'hc2':0,'vidstg':0}
  for row in bb:
   pp=ROOT/row['path'];qq=ROOT/files[row['case_id']]['path'];assert sha(pp)==row['sha256']and sha(qq)==files[row['case_id']]['sha256'];x=torch.load(pp,map_location='cpu',weights_only=False);y=torch.load(qq,map_location='cpu',weights_only=False)
   assert np.array_equal(x['arms']['TS_M2']['boxes'],y['arms']['TS_M2']['boxes'])and x['arms']['TS_M2']['interval']==y['arms']['TS_M2']['interval'];counts[x['dataset']]+=1
   if trial=='A003'and x['fits']['M2']is not None:
    fit_cases+=1;flat_skips+=sum(z.get('reason')=='no_information'for z in x['fits']['M2']['path'])
  assert counts=={'hc2':64,'vidstg':64};exact[trial]=counts;bind(other/'STUDENT_GLOBAL_BARRIER.json')
 result=dict(status='pass',scope='progress snapshot over sealed saved scores and selected actual output tensors; no new GT scoring/inference',time=time.time(),completed_search_slots=42,completed_component_trials=completed,max_summary_error=max_error,search_results=search_results,final_M2_configs={ds:completion['configs'][ds]['M2']for ds in ['hc2','vidstg']},final_M2_settings={ds:completion['response_settings'][ds]['M2']for ds in ['hc2','vidstg']},ablations=ablations,independent_exact_prediction_checks=exact,baseline_fit_case_count=fit_cases,baseline_flat_reward_skip_steps=flat_skips,source_hashes=sources,CPU_only=True,new_GT_read=False,new_model_loaded=False,new_GPU_used=False,bootstrap_executed=False,study_FINAL=False)
 out=BASE/'analysis'/('PROGRESS_READBACK_'+str(int(time.time()))+'.json');write(out,result)
 print(json.dumps(dict(path=str(out.relative_to(ROOT)),completed=len(completed),max_summary_error=max_error,exact=exact,baseline_fit_cases=fit_cases,flat_reward_skip_steps=flat_skips)))
if __name__=='__main__':main()
