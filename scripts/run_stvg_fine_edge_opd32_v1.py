"""One bounded frozen-coarse ordered ROI feedback actual OPD experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,argparse,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_fine_edge_opd32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/fine_edge_opd32'
PRE=ROOT/'artifacts/stvg_discriminative_dense32_v1';OLD=ROOT/'artifacts/stvg_cosine_dense_opd32_v1';OLDP=ROOT/'results/stvg_error_state/2026-10-18/cosine_dense_opd32'
CODE=['vg_tta/stvg_fine_edge_opd32_v1.py','scripts/run_stvg_fine_edge_opd32_v1.py','scripts/continue_stvg_fine_edge_opd32_v1.py','protocols/stvg_fine_edge_opd32_v1.md']
CENTER=ROOT/'artifacts/stvg_fine_roi_scoped_opd32_v1'
FINE=ROOT/'artifacts/stvg_fine_feature32_v1';ROI=ROOT/'artifacts/stvg_roi_layout_opd32_v1'
VERIFIER=None

def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),epoch=time.time(),**kw),True)
def verify(live=False,full=False):
 global VERIFIER
 p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256']
 from main_method.runtime import PinVerifier,resource_guard
 if VERIFIER is None:VERIFIER=PinVerifier(r['pins'])
 VERIFIER.verify(full=full);resource_guard()
 if live:assert time.time()<p['deadline_epoch']
 return p

def prepare():
 from vg_tta.stvg_fine_edge_opd32_v1 import properties,ARMS
 old=read(CENTER/'PLAN_LOCK.json');assert read(CENTER/'ROOT_CLOSING_RECEIPT.json')['status']=='complete';D.mkdir(exist_ok=True,parents=True);P.mkdir(exist_ok=True,parents=True);put(D/'CPU_PROPERTIES.json',properties())
 p=dict(status='locked',records=old['records'],count=32,per_domain=16,arms=ARMS,configuration=old['configuration'],geometry='per-edge bounded latent Gaussians: edge_delta=.5Espan*tanh(.5a), derivative.25 at0; mean readout not physical expectation',candidate_rule='shared zero+4 quartile-stratified antithetic normal candidates per L/T/R/B; seed fine-edge32-v1|alias; per-edge5categorical',actual_fit_episodes=96,round_slots=192,candidate_evaluations_per_round_per_node=17,teacher='original ordered FineROI/equal anchor score mean, no mixture',scope='same previous MBR interpolation-neighbor boolean',new_GPU_seconds=0,new_Source_captures=0,new_DINO_calls=0,new_training=0,MBR_and_planned_fixed=True,hard_readout_not_gate=True,development_exposed=True,independent_confirmation=False,heartbeat_remains_paused=True,no_parameter_search=True,CPU_seconds_cap=900,created_epoch=time.time(),deadline_epoch=time.time()+1800,decision='Both-domain Edge_live−E and Edge_live−Center_live full-v/ALL unobserved actual-s positive; onpolicy evidence additionally Live over Fixed/Shuffle with intervals/tails. No auto expansion.')
 put(D/'PLAN_LOCK.json',p);files=CODE+['vg_tta/stvg_roi_layout_opd32_v1.py','vg_tta/stvg_fine_roi_scoped_opd32_v1.py','scripts/run_stvg_dino_local_residual32_v1.py','vg_tta/tastvg_oracle_event5_v1.py','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']
 for r in p['records']:
  files+=[str((root/'features'/(r['alias']+'.pt')).relative_to(ROOT))for root in [FINE,PRE]]+[str((CENTER/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))]
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in files}));verify(full=True);status('prepared',0,GT_read=False)

def measure():
 import torch,numpy as np
 from vg_tta.stvg_fine_edge_opd32_v1 import build,scope,noise,fit
 torch.set_num_threads(4);p=verify(True);start=time.perf_counter();files=[]
 for ix,r in enumerate(p['records']):
  verify(True);assert time.perf_counter()-start<p['CPU_seconds_cap'];a=r['alias'];at=time.perf_counter();x=load(FINE/'features'/(a+'.pt'));co=load(PRE/'features'/(a+'.pt'));old=load(CENTER/'predictions'/(a+'.pt'));E=x['E'].double();assert torch.equal(E,torch.as_tensor(old['outputs']['E_MBR']))and x['frame_ids']==old['frame_ids']and x['interval']==old['interval'];pos=[f['position']for f,k in zip(co['frames'],co['selected'])if k>=0];bb=[f['boxes'][k].tolist()for f,k in zip(co['frames'],co['selected'])if k>=0];assert pos==x['positions'];ctx=build(x['maps'],x['valid'],pos,bb);assert torch.equal(ctx['templates'],load(CENTER/'context'/(a+'.pt'))['templates']);_,allowed=scope(x['frame_ids'],x['planned_positions'],x['interval']);assert torch.equal(allowed,old['allowed']);eps=noise(a,len(E));fits={mode:fit(E,ctx,eps,p['configuration'],allowed,mode,old['permutation'])for mode in ['Edge_fixed','Edge_live','Edge_shuffle']};assert torch.equal(fits['Edge_fixed']['path'][1]['boxes'],fits['Edge_live']['path'][1]['boxes']);outputs=dict(E_MBR=E.numpy(),Center_live=old['outputs']['Fine_live'],**{k:f['final'].numpy()for k,f in fits.items()});assert all(np.array_equal(b[~allowed],E.numpy()[~allowed])and(b[:,2:]>0).all()for b in outputs.values());files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],active=bool(pos),positions=pos,allowed=allowed,frame_ids=x['frame_ids'],interval=x['interval'],planned_positions=x['planned_positions'],noise=eps,permutation=old['permutation'],fits=fits,outputs=outputs,GT_read=False,compute=dict(case_seconds=time.perf_counter()-at,fit_episodes=3,round_slots=6,training_candidate_evaluation_slots_per_node=34,edge_conditional_slots_per_node=40,illegal_training_candidates=sum(int((~t['legal']).sum())for f in fits.values()for t in f['rounds']),new_GPU=0,new_models=0,new_training=0))));status('three_edge_fits',ix+1,GT_read=False);print(a,ix+1,round(time.perf_counter()-start,3),flush=True)
 verify(full=True);verify(True);assert not torch.cuda.is_initialized()and time.perf_counter()-start<p['CPU_seconds_cap'];put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all32_before_NEW_GT=True,actual_fit_episodes=96,round_slots=192,CPU_prediction_seconds=time.perf_counter()-start));status('all32_five_outputs_sealed',32,GT_read=False)

def score():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_fine_edge_opd32_v1 import ARMS
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_NEW_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-19/fine_roi_scoped_opd32/RESULTS_PUBLIC.json')['rows']};rows=[]
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));m=metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q]);assert m['values']['E_MBR']==prior[a]['values']['E_MBR']and m['values']['Center_live']==prior[a]['values']['Fine_live'];fits={k:dict(state_norm=float(f['path'][-1]['state'].norm()),step_L2=[t['output_step_L2']for t in f['rounds']],updated_edge_slots=[int((o['allowed'][:,None]&t['edge_feedback']['informative']).sum())for t in f['rounds']])for k,f in o['fits'].items()};rows.append(dict(alias=a,dataset=ds,active=o['active'],allowed_nodes=int(o['allowed'].sum()),fits=fits,compute=o['compute'],target_area_bin=prior[a]['target_area_bin'],motion_bin=prior[a]['motion_bin'],**m))
 pairs=[(a,'E_MBR')for a in ARMS[1:]]+[('Edge_live',b)for b in ['Center_live','Edge_fixed','Edge_shuffle']];eps=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s']
 def agg(rr):return dict(count=len(rr),parents=len(rr),active=sum(r['active']for r in rr),metrics={a:{e:mean([r['values'][a][e]for r in rr])for e in eps}for a in ARMS},paired={a+'_minus_'+b:{e:paired([None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rr])for e in eps if e!='t'}for a,b in pairs})
 groups={ds:agg([r for r in rows if r['dataset']==ds])for ds in gt};strata={ds:{key:{str(k):agg([r for r in rows if r['dataset']==ds and r[key]==k])for k in sorted(set(r[key]for r in rows if r['dataset']==ds),key=str)}for key in ['target_area_bin','motion_bin','active']}for ds in gt};passed=all(groups[ds]['paired'][c][e]['mean']>0 for ds in gt for c in ['Edge_live_minus_E_MBR','Edge_live_minus_Center_live']for e in ['v','unobserved_actual_s']);put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,strata=strata,all_outputs_before_NEW_GT=True,development_exposed=True,independent_confirmation=False,actual_OPD=True));put(P/'DECISION.json',dict(status='evaluated',edge_route_passed=bool(passed),rule=p['decision'],onpolicy_direction=all(groups[ds]['paired']['Edge_live_minus_'+b]['v']['mean']>0 for ds in gt for b in ['Edge_fixed','Edge_shuffle']),no_auto_expansion=True));put(D/'CPU_SCORE_COMPLETION.json',dict(status='complete',count=32));print('edgegate',passed,{ds:groups[ds]['paired']['Edge_live_minus_Center_live']['v']['mean']for ds in gt},flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
