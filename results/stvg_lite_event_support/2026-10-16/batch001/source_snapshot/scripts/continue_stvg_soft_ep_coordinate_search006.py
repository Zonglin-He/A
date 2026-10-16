"""Finite one-coordinate-at-a-time development search with equally tuned controls."""
from pathlib import Path
import sys,copy,json,math,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *
from scripts.lock_stvg_ecld_runtime002 import closure
PY=ROOT/'.conda/tubedetr/bin/python';DS=['hc2','vidstg'];MODES=['M0','M1','M2']
def fingerprint(cfg,settings,mode):return json.dumps([cfg,settings if mode!='M0'else None],sort_keys=True)
def source_pins():
 seeds=[ROOT/'scripts'/n for n in ['run_stvg_soft_ep_coordinate_trial003.py','stvg_soft_ep_coordinate_fit002.py','stvg_weightfree_coordinate_fit002.py','audit_stvg_soft_ep_step_sensitivity_v1.py','continue_stvg_soft_ep_coordinate_search006.py','launch_stvg_soft_ep_coordinate_trial003.py','review_stvg_soft_ep_coordinate_trial002.py','measure_stvg_soft_ep_coordinate_trial_v1.py','review_stvg_soft_ep_coordinate_inputs_v1.py']]
 paths=closure(seeds);r=read(BASE/'PREPARATION_RUNTIME.json');pins=dict(r['pins']);pins.update({str(p.relative_to(ROOT)):sha(p)for p in paths})
 for n in ['PREPARATION_RUNTIME.json','RECOVERY_RUNTIME002.json','SEARCH_POLICY002.json','CPU_GENERALIZED_FITTER_CONTRACTS.json','REFERENCE_LOCK.json','TEMPORAL_LOCK.json','ROOT_INPUT_ASSEMBLY.json','ROOT_COMPLETE_INPUT_READBACK.json','SOURCE_CAPTURE_BARRIER.json']:pins[str((BASE/n).relative_to(ROOT))]=sha(BASE/n)
 for n in ['ROOT_QUALIFICATION.json','CPU_CONTRACTS.json']:
  p=ROOT/'artifacts/stvg_soft_ep_opd_v1'/n;pins[str(p.relative_to(ROOT))]=sha(p)
 if (BASE/'SEARCH_EXECUTION_RUNTIME006.json').is_file():pins[str((BASE/'SEARCH_EXECUTION_RUNTIME006.json').relative_to(ROOT))]=sha(BASE/'SEARCH_EXECUTION_RUNTIME006.json')
 return pins,r['protected']
def objective(trial,ds,mode):
 s=read(BASE/'trials'/trial/'POSTSEAL_SCORE_SUMMARY.json')['summary'];r=next(x for x in s if x['dataset']==ds and x['arm']=='TS_'+mode and x['metric']=='m_vIoU');assert r['n_valid']==64 and r['n_unknown']==0;return r['mean']
def evaluate(configs,settings,history,coordinate,value):
 trial='F'+str(len(history)).zfill(3);b=BASE/'trials'/trial;b.mkdir(parents=True,exist_ok=False);reuse={ds:{}for ds in DS}
 for ds in DS:
  for mode in MODES:
   reuse[ds][mode]=None
   for old in history:
    d=read(BASE/'trials'/old/'DESIGN_LOCK.json')
    if d['family_reuse_from'][ds][mode]is None and fingerprint(d['configs'][ds][mode],d['response_settings'][ds].get(mode),mode)==fingerprint(configs[ds][mode],settings[ds].get(mode),mode):reuse[ds][mode]=old;break
 design=dict(status='locked',scope='128 exposed development one-coordinate matched M0/M1/M2',coordinate=coordinate,coordinate_value=value,configs=configs,response_settings=settings,family_reuse_from=reuse,GT_read=False)
 write(b/'DESIGN_LOCK.json',design);pins,protected=source_pins();pins[str((b/'DESIGN_LOCK.json').relative_to(ROOT))]=sha(b/'DESIGN_LOCK.json')
 for old in {x for d in reuse.values()for x in d.values()if x is not None}:
  ob=BASE/'trials'/old;assert read(ob/'CPU_COMPLETION.json')['status']=='complete'
  for x in read(ob/'STUDENT_GLOBAL_BARRIER.json')['files']:pins[x['path']]=x['sha256']
  for name in ['DESIGN_LOCK.json','RUNTIME.json','STUDENT_GLOBAL_BARRIER.json','GLOBAL_PREDICTION_BARRIER.json','CPU_COMPLETION.json']:pins[str((ob/name).relative_to(ROOT))]=sha(ob/name)
 write(b/'RUNTIME.json',dict(status='locked',pins=pins,protected=protected,GT_read=False))
 write(BASE/'SEARCH_STATUS006.json',dict(status='running',stage='student',trial=trial,coordinate=coordinate,value=value,completed_trial_count=len(history),GT_read=False,time=time.time()),mutable=True)
 subprocess.run(['systemd-run','--user','--wait','--collect','--unit=stvg-soft-ep-coordinate128-'+trial.lower()+'-001','--property=MemoryHigh=16G','--property=MemoryMax=20G','--property=MemorySwapMax=1G','--working-directory='+str(ROOT),str(PY),'-B','scripts/launch_stvg_soft_ep_coordinate_trial003.py',trial],cwd=ROOT,check=True)
 subprocess.run([str(PY),'-B','scripts/review_stvg_soft_ep_coordinate_trial002.py',trial],cwd=ROOT,check=True)
 subprocess.run([str(PY),'-B','scripts/measure_stvg_soft_ep_coordinate_trial_v1.py',trial],cwd=ROOT,check=True)
 subprocess.run([str(PY),'-B','scripts/audit_stvg_soft_ep_step_sensitivity_v1.py',trial],cwd=ROOT,check=True)
 history.append(trial);return trial
def verify_locked(path):
 r=read(path);assert r['status']=='locked'and r['GT_read']is False
 for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
 return r

def pidfd_wait(pid,start_ticks):
 import os,ctypes,select
 p=Path('/proc')/str(pid)/'stat'
 if not p.exists():return
 assert p.read_text().rsplit(')',1)[1].split()[19]==str(start_ticks),'PID identity changed'
 libc=ctypes.CDLL(None,use_errno=True);fd=libc.pidfd_open(pid,0)
 if fd<0:
  if ctypes.get_errno()==3:return
  raise OSError(ctypes.get_errno(),'pidfd_open')
 try:select.select([fd],[],[])
 finally:os.close(fd)

def refinement(cfg,settings,original,grid,coordinate):
 outputs=[]
 for side,label in [(-1,'lower'),(1,'upper')]:
  c=copy.deepcopy(cfg);s=copy.deepcopy(settings);values={ds:{}for ds in DS}
  for ds in DS:
   for m in MODES:
    if coordinate=='mu'and m!='M2'or coordinate=='response_ratio'and m=='M0':continue
    ov=cfg[ds][m][coordinate]/original[ds][coordinate]if coordinate in ['lr','sigma','tau']else cfg[ds][m][coordinate]if coordinate=='steps'else settings[ds][m][coordinate]
    idx=min(range(len(grid)),key=lambda i:abs(grid[i]-ov));assert abs(grid[idx]-ov)<1e-10
    near=idx+side;mid=ov
    if 0<=near<len(grid):mid=int(round((ov+grid[near])/2))if coordinate=='steps'else math.sqrt(ov*grid[near])
    values[ds][m]=mid
    if coordinate in ['lr','sigma','tau']:c[ds][m][coordinate]=original[ds][coordinate]*mid
    elif coordinate=='steps':c[ds][m]['steps']=mid
    else:s[ds][m][coordinate]=mid
  outputs.append((c,s,values,label))
 return outputs

EXECUTION_GT_READ=False
def main():
 global EXECUTION_GT_READ
 rt=verify_locked(BASE/'SEARCH_EXECUTION_RUNTIME006.json');assert not(BASE/'SEARCH_STATUS006.json').exists()
 policy=read(BASE/'SEARCH_DESIGN005.json');assert rt['pins'][str((BASE/'SEARCH_DESIGN005.json').relative_to(ROOT))]==sha(BASE/'SEARCH_DESIGN005.json')
 eb=BASE/'trials/F017';assert read(eb/'execution001/WORKER_EXIT.json')['exit_code']==0
 write(BASE/'SEARCH_STATUS006.json',dict(status='running',stage='F017_CPU_root_handoff',completed_trial_count=17,GT_read=False,time=time.time()),mutable=True)
 assert read(eb/'ROOT_READBACK.json')['status']=='pass'and read(eb/'ROOT_READBACK.json')['reader_source_sha256']==sha(ROOT/'scripts/review_stvg_soft_ep_coordinate_trial002.py')
 if not(eb/'CPU_COMPLETION.json').exists():
  try:subprocess.run([str(PY),'-B','scripts/measure_stvg_soft_ep_coordinate_trial_v1.py','F017'],cwd=ROOT,check=True)
  finally:EXECUTION_GT_READ=(eb/'GT_EXPOSURE.json').exists()
 if not(eb/'STEP_SENSITIVITY_READBACK.json').exists():subprocess.run([str(PY),'-B','scripts/audit_stvg_soft_ep_step_sensitivity_v1.py','F017'],cwd=ROOT,check=True)
 history=['D'+str(i).zfill(3)for i in range(12)]+['E012','E013','E014','F015','F016','F017']
 for tr in history:assert read(BASE/'trials'/tr/'CPU_COMPLETION.json')['status']=='complete'
 EXECUTION_GT_READ=True;coarse=read(BASE/'coordinate_selections005/lr_before_higher_extension.json');cfg=copy.deepcopy(coarse['configs']);settings=copy.deepcopy(coarse['response_settings']);best=copy.deepcopy(coarse['best_trial']);selections=[]
 original=read(ROOT/'artifacts/stvg_soft_ep_opd_v1/DESIGN_LOCK.json')['configs']
 def adopt(trials,coordinate):
  nonlocal cfg,settings
  for ds in DS:
   for m in MODES:
    winner=max([best[ds][m],*trials],key=lambda t:(objective(t,ds,m),-int(t[1:]),t[0]=='D'));best[ds][m]=winner;d=read(BASE/'trials'/winner/'DESIGN_LOCK.json');cfg[ds][m]=copy.deepcopy(d['configs'][ds][m])
    if m!='M0':settings[ds][m]=copy.deepcopy(d['response_settings'][ds][m])
  rec=dict(coordinate=coordinate,best_trial=copy.deepcopy(best),configs=copy.deepcopy(cfg),response_settings=copy.deepcopy(settings),means={ds:{m:objective(best[ds][m],ds,m)for m in MODES}for ds in DS},scope='all128 development selection only; no CURRENT/full',GT_read=True);selections.append(rec);write(BASE/'coordinate_selections006'/(coordinate+'.json'),rec)
 # The first three large-LR GPU trials are complete and reused byte-for-byte.
 ext=['F015','F016','F017']
 for tr,value in zip(ext,policy['vid_higher_absolute_lr'][:3]):
  frozen=read(BASE/'trials'/tr/'DESIGN_LOCK.json');expected=copy.deepcopy(cfg)
  for m in MODES:expected['vidstg'][m]['lr']=value
  assert frozen['configs']==expected and frozen['response_settings']==settings
 for value in policy['vid_higher_absolute_lr'][3:]:
  c=copy.deepcopy(cfg);s=copy.deepcopy(settings)
  for m in MODES:c['vidstg'][m]['lr']=value
  ext.append(evaluate(c,s,history,'vid_absolute_lr_extension',value))
 adopt(ext,'vid_higher_lr_extension')
 # Freeze both adjacent values before running either; neighbors are measured LR values.
 refines=[]
 for side,label in [(-1,'lower'),(1,'upper')]:
  c=copy.deepcopy(cfg);s=copy.deepcopy(settings);values={'vidstg':{}}
  for m in MODES:
   ov=cfg['vidstg'][m]['lr'];grid=sorted(set(read(BASE/'trials'/tr/'DESIGN_LOCK.json')['configs']['vidstg'][m]['lr']for tr in history));near=[v for v in grid if v<ov-1e-12]if side<0 else[v for v in grid if v>ov+1e-12];mid=math.sqrt(ov*(near[-1]if side<0 else near[0]))if near else ov
   c['vidstg'][m]['lr']=mid;values['vidstg'][m]=mid
  refines.append((c,s,values,label))
 out=[evaluate(c,s,history,'vid_higher_lr_refine_'+label,v)for c,s,v,label in refines];adopt(out,'vid_higher_lr_refined')
 original_design=read(BASE/'SEARCH_DESIGN.json')
 for coordinate in ['sigma','tau','steps','response_ratio']:
  grid=original_design['grids'][coordinate+'_multiplier'if coordinate in ['sigma','tau']else coordinate];trials=[]
  for value in grid:
   c=copy.deepcopy(cfg);s=copy.deepcopy(settings)
   for ds in DS:
    for m in MODES:
     if coordinate in ['sigma','tau']:c[ds][m][coordinate]=original[ds][coordinate]*value
     elif coordinate=='steps':c[ds][m]['steps']=int(value)
     elif m in ['M1','M2']:s[ds][m]['response_ratio']=value
   trials.append(evaluate(c,s,history,coordinate,value))
  adopt(trials,coordinate);frozen=refinement(cfg,settings,original,grid,coordinate)
  extra=[evaluate(c,s,history,coordinate+'_refine_'+label,v)for c,s,v,label in frozen];adopt(extra,coordinate+'_refined')
 assert len(history)==42
 write(BASE/'SEARCH_COMPLETION006.json',dict(status='complete_pending_independent_root_view_public_archive',history=history,selections=selections,best_trial=best,configs=cfg,response_settings=settings,scope='42-slot fixed coordinate development search; large LR measured with accepted/rollback diagnostics',independent_confirmation_complete=False,full_evaluation_started=False,CURRENT_promoted=False,bootstrap_executed=False,GT_read=True,time=time.time()))
 write(BASE/'SEARCH_STATUS006.json',dict(status='pending_actual_independent_root_view_public_archive',stage='root_handoff',completed_trial_count=len(history),GT_read=True,time=time.time()),mutable=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(BASE/'SEARCH_FAILURE006.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=EXECUTION_GT_READ,time=time.time()),mutable=True)
  write(BASE/'SEARCH_STATUS006.json',dict(status='failed',stage='required_root_review',GT_read=EXECUTION_GT_READ,time=time.time()),mutable=True);raise
