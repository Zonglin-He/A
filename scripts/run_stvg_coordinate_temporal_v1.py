"""Sequential CPU shared-head parameter search. GT lives only in separate scorer subprocesses."""
import sys,os,copy,subprocess,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
from scripts.run_stvg_weightfree_expand_v1 import BASE as E,roster,load_case,save_pt
BASE=ROOT/'artifacts/stvg_weightfree_coordinate_v1'
def run(name):
 sys.addaudithook(guard)
 import torch,numpy as np
 from scripts import stvg_table1_temporal_adapter_v1 as ta
 from scripts.stvg_coordinate_temporal_fit_v1 import fit
 torch.set_num_threads(2);b=BASE/name;lock=read(b/'STAGE_LOCK.json');files=[]
 for rec in roster():
  cid=rec['probe_alias'];case,cp=load_case(rec);t=torch.load(E/'temporal'/(cid+'.pt'),map_location='cpu',weights_only=False)['temporal'];old=torch.load(E/'student'/(cid+'.pt'),map_location='cpu',weights_only=False)
  spatial_stage=read(BASE/'DESIGN_LOCK.json')['stages'][-1]['name'];spatial_lock=read(BASE/spatial_stage/'SELECTION_LOCK.json');sp=torch.load(BASE/spatial_stage/'student'/(cid+'.pt'),map_location='cpu',weights_only=False);sp_arm=spatial_lock['selected'][rec['dataset']];sp_boxes=np.asarray(sp['arms'][sp_arm]['boxes'])
  arms={'F':copy.deepcopy(old['arms']['F']),'T':copy.deepcopy(old['arms']['T']),'TS_incumbent':dict(boxes=sp_boxes,interval=t['final_interval'])};fits={} 
  for variant,cfg in lock['recipes'][rec['dataset']].items():
   if 'temporal_fit'in t:
    rewards=[ta._alignment_reward(c['canonical_id'],r)for c,r in zip(case['temporal_candidates'],t['teacher_responses'])]
    f=fit(case,ta._hidden_from_source_case(case),case['temporal_candidates'],rewards,cfg);fits[variant]=f;iv=f['final_interval']
   else:iv=case['Native_interval']
   arms[variant]=dict(boxes=np.asarray(case['native_boxes_xyxy_normalized']),interval=iv);arms['TS_'+variant]=dict(boxes=sp_boxes,interval=iv)
  d=dict(case_id=cid,dataset=rec['dataset'],arms=arms,fits=fits,frame_ids=case['frame_ids'],reference_record=old['reference_record'],source_restored=True,GT_read=False,source_sha256=sha(cp),spatial_observation_clock='original QA unchanged')
  files.append(save_pt(b/'student'/(cid+'.pt'),d))
 write(b/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,stage_lock_sha256=sha(b/'STAGE_LOCK.json'),fit_source_sha256=sha(ROOT/'scripts/stvg_coordinate_temporal_fit_v1.py'),design_lock_sha256=sha(BASE/'TEMPORAL_DESIGN_LOCK002.json'),spatial_lock_sha256=sha(BASE/'SPATIAL_SELECTION_LOCK.json'),scorer_sha256=sha(__file__),GT_read=False))
 print(name,'sealed32')
def score(name):
 import torch,numpy as np
 from scripts import review_stvg_sparse_r1_v2 as scorer
 from scripts import score_stvg_native_instance_nativeclock005 as metric
 b=BASE/name;seal=read(b/'GLOBAL_PREDICTION_BARRIER.json');assert seal['scorer_sha256']==sha(__file__)
 assert seal['stage_lock_sha256']==sha(b/'STAGE_LOCK.json') and seal['fit_source_sha256']==sha(ROOT/'scripts/stvg_coordinate_temporal_fit_v1.py') and seal['design_lock_sha256']==sha(BASE/'TEMPORAL_DESIGN_LOCK002.json') and seal['spatial_lock_sha256']==sha(BASE/'SPATIAL_SELECTION_LOCK.json')
 for f in seal['files']:assert sha(ROOT/f['path'])==f['sha256']
 cases={r['probe_alias']:load_case(r)[0]for r in roster()};truth,_=scorer._truths_after_seal(cases);official,independent,_=scorer._metric_functions();rows=[]
 for rec in roster():
  cid=rec['probe_alias'];d=torch.load(b/'student'/(cid+'.pt'),map_location='cpu',weights_only=False);support={a['position']for a in d['reference_record']['original']['anchors']}
  for arm,pred in d['arms'].items():
   m,audit=metric._score_student_arm(pred,cases[cid],truth[cid],support,official,independent);rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=arm,metrics=m,metric_audit=audit['metric_audit']))
 summaries=[]
 for ds in ['hc2','vidstg']:
  for arm in sorted({r['arm']for r in rows if r['dataset']==ds}):
   vals=[r['metrics']['m_vIoU']for r in rows if r['dataset']==ds and r['arm']==arm];assert len(vals)==16;summaries.append(dict(dataset=ds,arm=arm,mean=math.fsum(vals)/16,count=16))
 write(b/'POSTSEAL_ROWS.json',dict(status='complete',rows=rows,GT_read=True));write(b/'POSTSEAL_SUMMARY.json',dict(status='complete',summary=summaries,GT_read=True,development_selection=True))
def controller():
 plan=read(BASE/'TEMPORAL_DESIGN_LOCK002.json');current={ds:dict(plan['initial'])for ds in ['hc2','vidstg']};py=ROOT/'.conda/tubedetr/bin/python'
 for stage in plan['stages']:
  b=BASE/stage['name'];b.mkdir(exist_ok=False);key=stage['parameter'];recipes={};inc={}
  for ds,cfg in current.items():
   recipes[ds]={}
   for j,v in enumerate(sorted(set(stage['values']+[cfg[key]]))):
    n=f'v{j:02}';recipes[ds][n]=dict(cfg,**{key:v})
    if v==cfg[key]:inc[ds]=n
  write(b/'STAGE_LOCK.json',dict(status='locked',recipes=recipes,incumbent=inc,parameter=key,GT_read=False,development_selection_used_GT=True))
  for command in ['run','score']:
   with (b/(command+'.log')).open('xb')as log:subprocess.run([str(py),'-B',__file__,command,stage['name']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  s=read(b/'POSTSEAL_SUMMARY.json')['summary'];selected={};audit=[]
  for ds in recipes:
   values={n:next(x['mean']for x in s if x['dataset']==ds and x['arm']=='TS_'+n)for n in recipes[ds]};mx=max(values.values());ties=[n for n in values if abs(values[n]-mx)<=1e-12];win=inc[ds]if inc[ds]in ties else min(ties,key=lambda n:recipes[ds][n][key]);current[ds]=recipes[ds][win];selected[ds]=win;audit.append(dict(dataset=ds,parameter=key,scores=values,selected=win,config=current[ds]))
  write(b/'SELECTION_AUDIT.json',dict(status='complete',records=audit,GT_read=True));write(b/'SELECTION_LOCK.json',dict(status='locked',configs=current,selected=selected,GT_read=False,selected_using_development_GT=True));print(stage['name'],audit,flush=True)
 write(BASE/'TEMPORAL_SELECTION_LOCK.json',dict(status='locked',configs=current,selected_using_development_GT=True,GT_read=False,observation_clock='original 5-step QA for spatial proposal locations; selected temporal config governs output interval',last_selection_sha256=sha(b/'SELECTION_LOCK.json')))
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
