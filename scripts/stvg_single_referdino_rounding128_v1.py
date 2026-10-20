"""One user-requested rounded numerical configuration; frozen engines unchanged."""
import sys,time,subprocess,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import D as MD,P as MP,read,put,sha,verify as vm
D=MD/'rounding';P=MP/'rounding'
def verify():
 vm();p=read(D/'PLAN_LOCK.json')
 for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
 return p

def prepare():
 assert not(D/'PLAN_LOCK.json').exists();parent=read(MD/'initialization_coupled/PLAN_LOCK.json');selected=read(MP/'SELECTION_PUBLIC.json');cfg=dict(selected['datasets']['hc2']['config'],lr=.014,tau=.14)
 spec=dict(parent['datasets']['hc2'],gammas=[.5],coarse_configs={});p=dict(records=parent['records'],datasets={'hc2':spec},maximum_additional_primary_fits=128,maximum_additional_primary_updates=5120,shared_GPU_cap_seconds=7200,exact_selection_sha256=sha(MP/'SELECTION_PUBLIC.json'),acceptance=dict(maximum_complete_mean_loss_pp=.1,maximum_meaningful_parent_loss=2,maximum_severe_HC_loss_count=0),pins={f:sha(ROOT/f)for f in ['protocols/stvg_single_referdino_rounding128_revision004.md','scripts/stvg_single_referdino_rounding128_v1.py']})
 put(D/'PLAN_LOCK.json',p);put(D/'hc2/FINE_PLAN_LOCK.json',dict(configs={'RND':cfg},center_id='RND',center_config=cfg,steps=[40],alphas=[1.]));put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k!='records'});put(P/'CONFIG_PUBLIC.json',cfg)

def execute(mode):
 if mode=='worker':
  import scripts.run_stvg_single_referdino_initialization_coupled256_v1 as m
 elif mode in ['seal','score']:
  import scripts.score_stvg_single_referdino_initialization_coupled256_v1 as m
 else:
  import scripts.audit_stvg_single_referdino_initialization_coupled256_v1 as m
 m.D=D;m.P=P;m.verify=verify
 if mode=='worker':
  m.main('hc2','fine');l=read(MD/'GPU_LEDGER.json');v=l['invocations'][-1];assert v['phase']=='initialization_fine';v['engine_phase_name']=v['phase'];v['phase']='rounding_single_configuration';v['namespace']='rounding';put(MD/'GPU_LEDGER.json',l)
 else:getattr(m,'pregt'if mode=='pre-gt'else'scored'if mode=='scored'else mode)('hc2','fine')

def decision():
 import numpy as np
 assert read(P/'hc2/fine/ROOT_AUDIT_PUBLIC.json')['status']=='pass';r=read(P/'hc2/fine/RESULTS_PUBLIC.json');prior=read(MP/'SELECTED_FIVE_ROW_RESULTS_PUBLIC.json')['directions']['hc2'];chosen=read(MP/'SELECTION_PUBLIC.json')['datasets']['hc2'];a='RND_S40_A100';x=np.array([100*(v['values'][a]['v']-v['values']['E_G050']['v'])for v in r['rows']]);dv=100*(r['means'][a]['v']-prior['means']['Full']['v']);positive=int((x>=.1).sum());severe=int((x< -5).sum());accept=dv>=-.1 and positive>=chosen['meaningful_positive_parents']-2 and severe==0
 result=dict(status='completed_one_rounded_full128_configuration',adopt_rounded=bool(accept),exact_configuration=chosen['config'],rounded_configuration=read(P/'CONFIG_PUBLIC.json'),complete_mean_change_vs_exact_pp=dv,meaningful_improved=positive,meaningful_harmed=int((x<=-.1).sum()),any_improved=int((x>1e-10).sum()),any_harmed=int((x< -1e-10).sum()),unchanged=int((abs(x)<=1e-10).sum()),severe_losses=severe,rounded_full_vIoU_percent=100*r['means'][a]['v'],rounded_adaptation=r['paired'][a+'_minus_E_G050'],all128_signed_differences_vs_exact=[dict(alias=v['alias'],delta_pp=100*(v['values'][a]['v']-prior['rows'][i]['values']['Full']['v']))for i,v in enumerate(r['rows'])],not_independent_confirmation=True,additional_grid=False)
 put(P/'ROUNDING_DECISION_PUBLIC.json',result);print('ROUNDING_DECISION',accept,dv,positive,severe,flush=True)

def control():
 prepare();beg=time.time()
 for mode in ['worker','seal','pre-gt','score','scored','decision']:
  print('ROUNDING_CONTROL',mode,flush=True)
  with (D/(mode+'.log')).open('a')as f:q=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',__file__,mode],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert q.returncode==0,(mode,q.returncode)
 put(D/'COMPLETION.json',dict(status='completed_single_rounding_trial',wall=time.time()-beg));print('ROUNDING_COMPLETE',flush=True)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='control':
  try:control()
  except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(error=repr(e)));raise
 elif mode=='decision':decision()
 else:execute(mode)
