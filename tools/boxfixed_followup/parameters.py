"""Parameter-only fixed128 follow-up. Calls immutable original worker/scorer.

No alternate propagation, candidates, features, reward, decoder or full entry.
Only output/plan directories are rebound to keep old artifacts immutable.
"""
import argparse,copy,json,hashlib,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup
from stvg.tube_runner import write
from tools.boxfixed_tube.profiles import profile_id
import tools.tubedetr_boxfixed_development as original
OLD_PUBLIC=original.PUBLIC;OLD_OUT=original.OUT;OLD_PLAN=original.PLAN
PUBLIC=ROOT/'results/tubedetr_boxfixed_followup';OUT=ROOT/'outputs/tubedetr_boxfixed_followup/parameters';PLAN=PUBLIC/'parameter_plan.json'
original.PUBLIC=PUBLIC;original.OUT=OUT;original.PLAN=PLAN


def prepare():
 old=json.loads(OLD_PLAN.read_text());d=copy.deepcopy(old);d.update(status='frozen_before_parameter_followup_scores',run_scope='fixed128 parameter-only; no full run',parent_plan_sha256=sha(OLD_PLAN),parent_final_seal_sha256=sha(OLD_PUBLIC/'final_seal.json'),posthoc_diagnosis_sha256=sha(PUBLIC/'diagnosis.json'),user_constraint='Only change the four existing scalar parameters; original source files are immutable',no_new_GPU_or_teacher_calls=True,full_evaluation_after_both128_profile_locks=False,new_full_allowed=False,max_profiles_per_dataset=10,admission_grid=[.4,.7,.9],guard_grid=[.4,1.,2.],link_grid=[.5,1.],gamma_grid=[0.,.0125,.025],selection_objective='same as prior: maximum complete128 equal-query Full m_vIoU; incumbent always eligible; exact ties retain incumbent',incumbents={ds:json.loads((OLD_PUBLIC/ds/'selected_profile.json').read_text())for ds in ['vidstg','hc2']})
 for f in ['tools/boxfixed_followup/__init__.py','tools/boxfixed_followup/diagnose.py','tools/boxfixed_followup/parameters.py']:d['code_sha256'][f]=sha(ROOT/f)
 assert not PLAN.exists(),'Never overwrite a frozen follow-up plan';write(PLAN,d)
 for ds in d['dataset_order']:
  root=OLD_OUT/ds/'captures';s=json.loads((root/'sealed.json').read_text());assert s['queries']==128 and s['plan_sha256']==sha(OLD_PLAN)
  files={q:{**e,'path':str(root/e['path']),'receipt_path':str(root/e['receipt_path'])}for q,e in s['files'].items()}
  for e in files.values():assert sha(Path(e['path']))==e['sha256'] and sha(Path(e['receipt_path']))==e['receipt_sha256']
  write(OUT/ds/'captures/sealed.json',dict(s,files=files,plan_sha256=sha(PLAN),reused_unchanged_raw_capture_seal_sha256=sha(root/'sealed.json'),raw_capture_parent_plan_sha256=sha(OLD_PLAN)))
 print('Frozen',sha(PLAN),flush=True)


def checked():
 p=original.checked();assert sha(OLD_PLAN)==p['parent_plan_sha256'];assert sha(OLD_PUBLIC/'final_seal.json')==p['parent_final_seal_sha256']
 for f,h in json.loads(OLD_PLAN.read_text())['code_sha256'].items():assert sha(ROOT/f)==h,f
 return p


def child(ds,stage,request):
 log=ROOT/'logs'/f'tubedetr_boxfixed_parameter_{ds}_{stage}.log';cmd=[sys.executable,'-B','-u','tools/boxfixed_followup/parameters.py','--stage',stage,'--dataset',ds,'--request',str(request)]
 write(OUT/'status.json',dict(status=stage,dataset=ds,request=str(request),new_GPU=False))
 with log.open('a')as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});write(ROOT/'runtime/tubedetr_boxfixed_parameter_followup.json',dict(controller_pid=os.getpid(),worker_pid=proc.pid,command=cmd,log=str(log),dataset=ds,stage=stage))
  if proc.wait():raise RuntimeError(f'{ds} {stage} failed')


def run():
 p=checked();locks={}
 for ds in p['dataset_order']:
  base=p['incumbents'][ds]['parameters'];current=base.copy();tested={};decisions=[];pub=PUBLIC/ds;pub.mkdir(exist_ok=True)
  def evaluate(params):
   key=profile_id(params)
   if key not in tested:
    req=OUT/ds/'requests'/(key+'.json');write(req,params);child(ds,'profile',req)
    if params==base:
     old=OLD_OUT/ds/'profiles'/p['incumbents'][ds]['profile_id'];ns=json.loads((OUT/ds/'profiles'/key/'sealed.json').read_text());os_=json.loads((old/'sealed.json').read_text());import numpy as np
     assert sha(old/'sealed.json')==p['incumbents'][ds]['dev_prediction_seal_sha256']
     for q,e in ns['files'].items():
      assert sha(old/os_['files'][q]['path'])==os_['files'][q]['sha256']
      with np.load(OUT/ds/'profiles'/key/e['path'])as a,np.load(old/os_['files'][q]['path'])as b:
       assert all(np.array_equal(a[k],b[k])for k in ['Source','BoxFixed','BoxFixedNative','Native_interval','Global6_interval','frame_ids','positions8','actions','confidence']),q
    child(ds,'score',req);tested[key]=json.loads((OUT/ds/'profiles'/key/'metrics.json').read_text())
   return tested[key]
  def choose():return max(tested.values(),key=lambda m:(m['means']['Full']['v'],m['parameters']==base,-sum(abs(m['parameters'][k]-base[k])for k in base)))['parameters'].copy()
  evaluate(base)
  for stage,key,grid,reason in [('admission','admission',p['admission_grid'],'已接受传播仍有错误；只提高既有cycle准入阈值，保持原置信公式和候选不变'),('protection','guard',p['guard_grid'],'原最优保护位于上界；提高既有保护权重检验是否减少不可靠替代'),('association','link',p['link_grid'],'只复查既有连接权重，不加新一致性项或新评分'),('initialization','gamma',p['gamma_grid'],'最后检验保守传播设置下既有初始化强度，包含gamma0控制')]:
   start=current.copy();batch=[evaluate({**start,key:x})for x in grid];current=choose();win=evaluate(current);reference=evaluate(base)
   decisions.append(dict(stage=stage,reason_before=reason,start=start,all_candidates=batch,selected=current,Full_gain_vs_incumbent_pp=win['means']['Full']['v']-reference['means']['Full']['v'],Full_minus_Temporal_pp=win['Full_minus_Temporal_pp'],unobserved_gain_vs_Temporal_pp=win['unobserved_sIoU']['Full']-win['unobserved_sIoU']['Temporal'],next_reason='按原完整128 m_vIoU目标选择，公开空间负例，不扩大阶段网格；不修改算法'))
   write(pub/'parameter_decisions.json',decisions)
  grid=p['admission_grid'];i=grid.index(current['admission']);neighbors=[grid[j]for j in [i-1,i+1]if 0<=j<len(grid)];batch=[evaluate({**current,'admission':x})for x in neighbors];current=choose();assert len(tested)<=p['max_profiles_per_dataset'];decisions.append(dict(stage='bounded_coupling_recheck',reason_before='所选保护/连接/gamma下只复查相邻准入，最多两个',all_candidates=batch,selected=current));write(pub/'parameter_decisions.json',decisions)
  winner=evaluate(current);receipt=dict(status='complete_fixed128_validation_only',parameters=current,profile_id=profile_id(current),plan_sha256=sha(PLAN),profiles_tested=len(tested),metrics=winner,prediction_seal_sha256=sha(OUT/ds/'profiles'/profile_id(current)/'sealed.json'),incumbent_128_predictions_bitexact=True,all_original_code_hashes_unchanged=True,new_model_or_teacher_calls=0,new_full_predictions=0,development_exposed=True,independent_confirmation=False,full_not_started=True)
  write(pub/'parameter_selection.json',receipt);locks[ds]=receipt
 write(PUBLIC/'parameter_summary.json',dict(status='complete',datasets=locks,no_full_run=True,only_existing_parameters_changed=True));files=[PLAN,PUBLIC/'parameter_summary.json',*[PUBLIC/ds/f for ds in p['dataset_order']for f in ['parameter_decisions.json','parameter_selection.json']]];write(PUBLIC/'parameter_final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in files},publication_pending=True));write(OUT/'status.json',dict(status='complete',both_fixed128=True,new_full=False))


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','run','profile','score'],required=True);a.add_argument('--dataset',choices=['vidstg','hc2']);a.add_argument('--request');args=a.parse_args()
 try:
  if args.stage=='prepare':prepare()
  elif args.stage=='run':run()
  elif args.stage=='profile':
   setup();checked();original.profile_worker(args.dataset,args.request)
  else:
   checked();original.score(args.dataset,json.loads(Path(args.request).read_text()))
 except Exception as e:
  write(OUT/'failure.json',dict(status='failed',stage=args.stage,error=type(e).__name__,message=str(e)));raise
