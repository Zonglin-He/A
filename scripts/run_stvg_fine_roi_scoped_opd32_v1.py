"""One bounded frozen-coarse ordered ROI feedback actual OPD experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,argparse,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_fine_roi_scoped_opd32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/fine_roi_scoped_opd32'
PRE=ROOT/'artifacts/stvg_discriminative_dense32_v1';OLD=ROOT/'artifacts/stvg_cosine_dense_opd32_v1';OLDP=ROOT/'results/stvg_error_state/2026-10-18/cosine_dense_opd32'
CODE=['vg_tta/stvg_fine_roi_scoped_opd32_v1.py','scripts/run_stvg_fine_roi_scoped_opd32_v1.py','scripts/continue_stvg_fine_roi_scoped_opd32_v1.py','protocols/stvg_fine_roi_scoped_opd32_v1.md']
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
 from vg_tta.stvg_fine_roi_scoped_opd32_v1 import properties,ARMS
 old=read(ROI/'PLAN_LOCK.json');assert read(ROI/'FINAL_COMPLETION.json')['status']=='complete';assert read(FINE/'REPORT_COMPLETION.json')['status']=='complete'
 D.mkdir(exist_ok=True,parents=True);P.mkdir(exist_ok=True,parents=True);put(D/'CPU_PROPERTIES.json',properties());files=CODE+['vg_tta/stvg_roi_layout_opd32_v1.py','vg_tta/stvg_discriminative_dense32_v1.py','scripts/run_stvg_dino_local_residual32_v1.py','vg_tta/tastvg_oracle_event5_v1.py','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']
 for r in old['records']:
  a=r['alias'];files+=[str((root/k/(a+'.pt')).relative_to(ROOT))for root,k in [(FINE,'features'),(ROI,'predictions'),(PRE,'features'),(PRE,'feedback')]]
 put(D/'PLAN_LOCK.json',dict(status='locked',records=old['records'],count=32,per_domain=16,arms=ARMS,configuration=old['configuration'],feature='actual cached ResNet layer3 stride16/1024, no new forward',mechanism='original ordered3x3 ROI scorer and original independent-center actual OPD; pure predicted-interval neighbor scope',scope='(right_neighbor>=start)&(left_neighbor<end), planned fixed, no tent amplitude',new_GPU_seconds=0,new_Source_captures=0,new_DINO_calls=0,new_model_training=0,new_head=0,actual_fit_episodes=96,round_slots=192,Coarse_reused_no_new_fits=True,MBR_fixed=True,development_exposed=True,independent_confirmation=False,no_parameter_search=True,no_GT_direct_gate=True,heartbeat_remains_paused=True,created_epoch=time.time(),deadline_epoch=time.time()+1800,CPU_seconds_cap=900,decision='Both-domain Fine_live−E and Fine_live−Coarse_live full-v/ALL unobserved actual-s positive; Fine_live full-v above Fixed/Shuffle/Direct. Confidence intervals/tails retained; no auto expansion.'))
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in files}));verify(full=True);status('prepared',0,GT_read=False)

def measure():
 import torch,numpy as np
 from vg_tta.stvg_fine_roi_scoped_opd32_v1 import build,scope,restore,scoped_direct,scoped_fit,interpolation_equivalence,MODES
 torch.set_num_threads(4);p=verify(True);start=time.perf_counter();files=[]
 for ix,r in enumerate(p['records']):
  verify(True);assert time.perf_counter()-start<p['CPU_seconds_cap'];a=r['alias'];at=time.perf_counter();x=load(FINE/'features'/(a+'.pt'));coarse=load(PRE/'features'/(a+'.pt'));old=load(ROI/'predictions'/(a+'.pt'));fb=load(PRE/'feedback'/(a+'.pt'))['feedback'];E=x['E'].double()
  assert x['frame_ids']==old['frame_ids']==coarse['frame_ids']and x['planned_positions']==old['planned_positions']and x['interval']==old['interval'];assert torch.equal(E,torch.as_tensor(old['outputs']['E_MBR']))and torch.equal(E,coarse['E'].double());assert x['maps'].shape[1]==1024 and x['maps'].shape[-2]>coarse['maps'].shape[-2]
  positions=[f['position']for f,k in zip(coarse['frames'],coarse['selected'])if k>=0];boxes=[f['boxes'][k].tolist()for f,k in zip(coarse['frames'],coarse['selected'])if k>=0];assert positions==x['positions'];assert len(boxes)==0 or torch.equal(torch.as_tensor(boxes),torch.stack(x['boxes']))
  relevant,allowed=scope(x['frame_ids'],x['planned_positions'],x['interval']);ctx=build(x['maps'],x['valid'],positions,boxes);crec=save(D/'context'/(a+'.pt'),dict(alias=a,positions=positions,templates=ctx['templates'],weights=ctx['weights'],relevant=relevant,allowed=allowed,GT_read=False));ep=fb['noise'];cfg=p['configuration'];perm=fb['permutation'];rd=scoped_direct(E,ctx,ep,cfg,x['planned_positions'],allowed);assert torch.equal(rd['candidates'],old['direct']['candidates'])
  ft=time.perf_counter();fits={k:scoped_fit(E,ctx,ep,cfg,x['planned_positions'],mode,perm,allowed)for k,mode in MODES.items()};fit_seconds=time.perf_counter()-ft;assert torch.equal(fits['Fine_fixed']['path'][1]['boxes'],fits['Fine_live']['path'][1]['boxes']);outputs=dict(E_MBR=E.numpy());proofs={}
  for new,key in [('Coarse_direct','Layout_readout'),('Coarse_fixed','Fixed_dense'),('Coarse_live','On_policy_dense'),('Coarse_shuffle','Shuffled_OPD')]:
   b=restore(E,old['outputs'][key],allowed);outputs[new]=b.numpy();proofs[new]=interpolation_equivalence(x['frame_ids'],x['interval'],b,old['outputs'][key]);assert proofs[new]['exact'];assert torch.equal(b[allowed],torch.as_tensor(old['outputs'][key])[allowed])
  outputs.update(Fine_direct=rd['boxes'].numpy(),**{k:f['final'].numpy()for k,f in fits.items()});assert all(np.array_equal(b[~allowed],E.numpy()[~allowed])and np.array_equal(b[:,2:],E.numpy()[:,2:])for b in outputs.values())
  if not positions:assert all(np.array_equal(b,E)for b in outputs.values())
  files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],active=bool(positions),frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],interval=x['interval'],relevant=relevant,allowed=allowed,positions=positions,permutation=perm,noise=ep,outputs=outputs,fits=fits,direct=rd,coarse_interpolation_proof=proofs,context_receipt=crec,GT_read=False,compute=dict(case_seconds=time.perf_counter()-at,three_fit_seconds=fit_seconds,new_GPU_seconds=0,new_Source=0,new_DINO=0,new_model_training=0,actual_fit_episodes=3,round_slots=6))))
  status('fine_ROI_three_actual_fits',ix+1,GT_read=False);print(a,ix+1,round(time.perf_counter()-start,3),flush=True)
 verify(full=True);verify(True);assert time.perf_counter()-start<p['CPU_seconds_cap']and not torch.cuda.is_initialized();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all32_before_NEW_GT=True,actual_fit_episodes=96,round_slots=192,CPU_prediction_seconds=time.perf_counter()-start));status('all32_nine_outputs_sealed',32,GT_read=False)

def score():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_fine_roi_scoped_opd32_v1 import ARMS
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_NEW_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-19/roi_layout_opd32/RESULTS_PUBLIC.json')['rows']};rows=[]
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));m=metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q]);old=prior[a];assert m['values']['E_MBR']==old['values']['E_MBR']
  for arm,key in [('Coarse_direct','Layout_readout'),('Coarse_fixed','Fixed_dense'),('Coarse_live','On_policy_dense'),('Coarse_shuffle','Shuffled_OPD')]:assert abs(m['values'][arm]['v']-old['values'][key]['v'])<1e-12
  diag={k:dict(updated_frames=[t['updated_frames']for t in f['rounds']],eta_norm=float(f['path'][-1]['eta'].norm()),output_delta_L2=float((f['final']-f['path'][0]['boxes']).norm()),step_L2=[t['output_step_L2']for t in f['rounds']],KL=[t['KL']for t in f['rounds']],second_proposal_center_norm=float(f['rounds'][1]['sampling_center'].norm()))for k,f in o['fits'].items()}
  rows.append(dict(alias=a,dataset=ds,active=o['active'],allowed_nodes=int(o['allowed'].sum()),retained_anchors=len(o['positions']),scope_interpolation_exact=all(v['exact']for v in o['coarse_interpolation_proof'].values()),fits=diag,compute=o['compute'],target_area_bin=old['target_area_bin'],motion_bin=old['motion_bin'],query_form=old['query_form'],**m))
 eps=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'];contrasts=[(a,'E_MBR')for a in ARMS[1:]]+[('Fine_live',a)for a in ['Coarse_live','Fine_fixed','Fine_shuffle','Fine_direct']]+[('Fine_direct','Coarse_direct'),('Fine_fixed','Coarse_fixed'),('Fine_shuffle','Coarse_shuffle')]
 def agg(rr):return dict(count=len(rr),parents=len(rr),active=sum(r['active']for r in rr),metrics={a:{e:mean([r['values'][a][e]for r in rr])for e in eps}for a in ARMS},paired={a+'_minus_'+b:{e:paired([None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rr])for e in eps if e!='t'}for a,b in contrasts})
 groups={ds:agg([r for r in rows if r['dataset']==ds])for ds in gt};strata={ds:{key:{str(k):agg([r for r in rows if r['dataset']==ds and r[key]==k])for k in sorted(set(r[key]for r in rows if r['dataset']==ds),key=str)}for key in ['target_area_bin','motion_bin','active']}for ds in gt};passed=all(groups[ds]['paired'][c][e]['mean']>0 for ds in gt for c in ['Fine_live_minus_E_MBR','Fine_live_minus_Coarse_live']for e in ['v','unobserved_actual_s'])and all(groups[ds]['paired']['Fine_live_minus_'+a]['v']['mean']>0 for ds in gt for a in ['Fine_fixed','Fine_shuffle','Fine_direct'])
 put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,strata=strata,all32_outputs_before_NEW_GT=True,development_exposed=True,independent_confirmation=False,actual_OPD=True,all96fits192slots_executed=True,config=p['configuration'],GT_scope_never_used=True));put(P/'DECISION.json',dict(status='evaluated',passed=bool(passed),rule=p['decision'],actual_OPD=True));put(D/'CPU_SCORE_COMPLETION.json',dict(status='complete',count=32));print('gate',passed,{ds:groups[ds]['paired']['Fine_live_minus_E_MBR']['v']['mean']for ds in gt},flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
