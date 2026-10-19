"""Frozen native trained Q/K capture, actual CPU OPD, then globally sealed GT score."""
from pathlib import Path
import os,sys,time,argparse,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_ca_query_opd32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/ca_query_opd32';PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1';CLOSED=ROOT/'artifacts/stvg_dino_query_closedloop32_v1';FEAT=ROOT/'artifacts/stvg_discriminative_dense32_v1';ANCHOR=ROOT/'artifacts/stvg_anchor_refresh32_v1';OLD=ROOT/'artifacts/stvg_cosine_dense_opd32_v1';VERIFIER=None
CODE=['vg_tta/stvg_ca_query_opd32_v1.py','scripts/run_stvg_ca_query_opd32_v1.py','scripts/continue_stvg_ca_query_opd32_v1.py','protocols/stvg_ca_query_opd32_v1.md']
def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),epoch=time.time(),**kw),True)
def verify(live=False,full=False):
 global VERIFIER
 p=read(D/'PLAN_LOCK.json');rt=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==rt['plan_sha256']
 from main_method.runtime import PinVerifier,resource_guard
 if VERIFIER is None:VERIFIER=PinVerifier(rt['pins'])
 VERIFIER.verify(full=full);resource_guard()
 if live:assert time.time()<p['deadline_epoch']
 return p

def prepare():
 from vg_tta.stvg_ca_query_opd32_v1 import properties,ARMS
 assert read(ANCHOR/'ROOT_CLOSING_RECEIPT.json')['status']=='complete';old=read(ANCHOR/'PLAN_LOCK.json');night=read(ROOT/'artifacts/stvg_opd_overnight_20261018_v1/STATE.json');D.mkdir(parents=True,exist_ok=True);P.mkdir(parents=True,exist_ok=True);put(D/'CPU_PROPERTIES.json',properties())
 p=dict(status='locked',records=old['records'],count=32,per_domain=16,arms=ARMS,new_Source_captures=0,new_DINO_calls=0,new_training_runs=0,new_heads=0,pretrained_weights_frozen=True,query_source='all6 anchor-conditioned final CA content half of each head, DINO vs E slots',metric='native QcKc/sqrt(full concat head_dim), no L2/extra tau; perhead visualsoftmax thenheadmean',likelihood='ROI mass/uniform visible ROI mass, anchor arithmeticmean; teacher prior*likelihood',scope='all actual unobserved nodes',center_scale=.25,step=.5,rounds=2,candidates_per_round=17,final_step=2,fit_episodes=160,round_slots=320,pilot_aliases=['NH001','NV001'],GPU_seconds_cap=900,CPU_adapt_seconds_cap=600,overnight_GPU_remaining_seconds=7200-night['new_GPU_spent_seconds'],created_epoch=time.time(),deadline_epoch=min(time.time()+1800,night['deadline_epoch']),MBR_fixed=True,E_widths_and_planned_fixed=True,GT_after_full32_prediction_seal=True,development_exposed=True,independent_confirmation=False,no_parameter_search=True,decision='Both-domain Dino_live minus E full-v and unobserved actual-s positive; Live full-v above Fixed/Direct/Shuffle/EqueryLive/CosineLikeLive; CIs/tails retained')
 put(D/'PLAN_LOCK.json',p);pins=dict(read(ANCHOR/'RUNTIME_LOCK.json')['pins'])
 for f in CODE+['external/TA-STVG/models/grounding_model/attention.py','vg_tta/stvg_discriminative_dense32_v1.py','vg_tta/stvg_soft_correspondence_v1.py']:pins[f]=sha(ROOT/f)
 for r in p['records']:
  for b in [ANCHOR,OLD]:
   f=b/'predictions'/(r['alias']+'.pt');pins[str(f.relative_to(ROOT))]=sha(f)
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));verify(full=True);status('prepared',0,GT_read=False)

def capture():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_opd_frozen_capture_cpu007 import transfer
 from vg_tta.stvg_ca_query_opd32_v1 import capture as cap,reconstruct,fields
 p=verify(True);lease=None
 while lease is None:
  verify(True)
  try:lease=gpu()
  except BlockingIOError:status('waiting_GPU_lease',0);time.sleep(15)
 start=time.perf_counter();model=None;domain=None;modelhash=None;files=[];pilot=[];partial={};a=None;complete=False
 def check():
  verify(True);assert time.perf_counter()-start+3<min(p['GPU_seconds_cap'],p['overnight_GPU_remaining_seconds']);assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  put(D/'GPU_RUNTIME.json',dict(pid=os.getpid(),start_epoch=time.time(),shared_lease=True));records=[next(r for r in p['records']if r['alias']==a)for a in p['pilot_aliases']]+[r for r in p['records']if r['alias']not in p['pilot_aliases']]
  for ix,r in enumerate(records):
   check();a=r['alias'];ds=r['dataset'];tick=time.perf_counter();status('trained_CA_capture',ix,alias=a,GT_read=False)
   if domain!=ds:
    if model is not None:assert state_hash(model.state_dict())==modelhash;del model;gc.collect();torch.cuda.empty_cache()
    model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;modelhash=state_hash(model.state_dict())
   old=load(ANCHOR/'predictions'/(a+'.pt'));x=load(FEAT/'features'/(a+'.pt'));rc=load(CLOSED/'replay_inputs'/(a+'.pt'));inputs=transfer(rc['spatial_inputs'],'cuda');decoder=model.ground_decoder.decoder;n=len(x['E']);positions=old['positions'];targets=old['targets'];assert torch.equal(rc['Source'],old['Source']);before=state_hash({str(v)+'/'+k:t for v,kw in enumerate(inputs)for k,t in kw.items()if torch.is_tensor(t)});partial={};errs=[]
   for name in ['native']+(['E','DINO']if positions else[]):
    check();obj=cap(decoder,inputs,n,positions,x['E'].numpy()[positions]if name=='E'else targets,name);partial[name]=obj;check();assert torch.equal(obj['boxes'],old['Source']if name=='native'else old['raw_calls']['persistent_'+name]['boxes'])
    for view,rec in enumerate(obj['records']):
     ref=old['raw_calls']['native_original']['all_six_outputs'][view]if name=='native'else torch.stack([v['raw_output']for v in old['raw_calls']['persistent_'+name]['traces'][view]['layers']]).transpose(1,2);assert torch.equal(obj['full_outputs'][view],ref);err=float((reconstruct(rec)-rec['attention'].double()).abs().max());assert err<=1e-5;errs.append(err)
     if name!='native':assert torch.equal(rec['k'],partial['native']['records'][view]['k'])
   assert state_hash({str(v)+'/'+k:t for v,kw in enumerate(inputs)for k,t in kw.items()if torch.is_tensor(t)})==before
   h,w=x['valid'].shape[-2:]
   for view,kw in enumerate(inputs):assert torch.equal((~kw['encoded_mask'][:,:h*w]).cpu(),x['valid'][view::2].flatten(1))
   ff=fields(partial,positions,x['valid'],h,w);assert not any(v.requires_grad or v.grad is not None for v in model.parameters());qd=[];qe=[]
   for pos in positions:
    v=pos%2;j=pos//2;re=partial['DINO']['records'][v];H=re['heads'];dh=re['head_dim'];qd.append(re['q'][0,j].reshape(H,dh)[:,:dh//2]);qe.append(partial['E']['records'][v]['q'][0,j].reshape(H,dh)[:,:dh//2])
   qdiff=float((torch.stack(qd)-torch.stack(qe)).abs().max())if qd else 0.;qscale=max(1.,float(torch.stack(qd).abs().max()))if qd else 1.;mdiff=float((ff['DINO']-ff['E']).abs().max())if positions else 0.;seconds=time.perf_counter()-tick
   files.append(save(D/'features'/(a+'.pt'),dict(alias=a,dataset=ds,positions=positions,targets=targets,conditions=partial,fields=ff,valid=x['valid'],E=x['E'].double(),frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],interval=x['interval'],attention_max_reconstruction_error=max(errs),query_content_max_DINO_E_difference=qdiff,query_content_scale=qscale,probability_field_max_DINO_E_difference=mdiff,GT_read=False,frozen_model_hash=modelhash,compute=dict(case_seconds=seconds,actual_trajectory_decoder_calls=len(partial),actual_PosDecoder_calls=2*len(partial),new_Source_captures=0,new_DINO_calls=0,backward_calls=0))))
   if ix<2:pilot.append(dict(alias=a,case_seconds=seconds,active=bool(positions),content_exceeds_FP32_relative1e_6=qdiff>1e-6*qscale,field_exceeds1e_8=mdiff>1e-8,max_attention_reconstruction_error=max(errs)))
   if ix==1:
    estimate=time.perf_counter()-start+15*sum(v['case_seconds']for v in pilot)+60;qualified=any(v['active']and(v['content_exceeds_FP32_relative1e_6']or v['field_exceeds1e_8'])for v in pilot)and estimate<p['GPU_seconds_cap'];put(D/'PILOT2_TIMING.json',dict(status='measured',cases=pilot,elapsed_seconds=time.perf_counter()-start,estimated_total_upper_seconds=estimate,qualified=qualified,not_a_guarantee=True))
    if not qualified:put(D/'PILOT_STOP.json',dict(status='stopped',reason='content/field has no FP32-scale change in supported pilots or budget infeasible',no_NEW_GT_scoring=True,preserved_count=len(files)));return
   print(a,ix+1,round(time.perf_counter()-start,3),max(errs),qdiff,mdiff,flush=True);partial={};del inputs,old,x,rc,ff;gc.collect();torch.cuda.empty_cache()
  assert state_hash(model.state_dict())==modelhash;verify(full=True);check();put(D/'GLOBAL_FEATURE_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,all32_features_before_adapt=True));status('all32_features_sealed',32,GT_read=False);complete=True
 finally:
  try:
   if not complete and partial:save(D/'partial'/(str(a)+'.pt'),dict(alias=a,conditions=partial,GT_read=False,complete=False))
  finally:
   try:put(D/'GPU_COST_RECEIPT.json',dict(status='released',GPU_allocation_seconds=time.perf_counter()-start+3,complete32=complete))
   finally:lease.close()

def adapt():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 from vg_tta.stvg_ca_query_opd32_v1 import noise,normalized_cosine,direct,fit,FIT_ARMS
 from vg_tta.stvg_soft_correspondence_v1 import propagate
 torch.set_num_threads(4);p=verify(True);bar=read(D/'GLOBAL_FEATURE_BARRIER.json');assert bar['count']==32
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 start=time.perf_counter();files=[]
 for ix,r in enumerate(p['records']):
  verify(True);assert time.perf_counter()-start<p['CPU_adapt_seconds_cap'];a=r['alias'];tick=time.perf_counter();o=load(D/'features'/(a+'.pt'));x=load(FEAT/'features'/(a+'.pt'));old=load(OLD/'predictions'/(a+'.pt'));E=o['E'];valid=o['valid'];eps=noise(a,len(E));permutation=old['permutation'];fields=dict(o['fields']);soft=propagate(x['maps'],valid,o['positions'],torch.as_tensor(o['targets']).double());fields['cosine']=normalized_cosine(soft,valid);dd=direct(E,fields['DINO'],valid,eps,o['planned_positions']);ed=direct(E,fields['E'],valid,eps,o['planned_positions']);fits={mode:fit(E,fields,valid,eps,o['planned_positions'],mode,permutation)for mode in FIT_ARMS};assert torch.equal(fits['Dino_fixed']['path'][1]['boxes'],fits['Dino_live']['path'][1]['boxes']);outputs=dict(E_MBR=E.numpy(),Dino_direct=dd['boxes'].numpy(),Equery_direct=ed['boxes'].numpy(),**{k:f['final'].numpy()for k,f in fits.items()});assert all(np.isfinite(v).all()for v in outputs.values());assert not torch.cuda.is_initialized()
  files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],frame_ids=o['frame_ids'],planned_positions=o['planned_positions'],interval=o['interval'],positions=o['positions'],active=bool(o['positions']),outputs=outputs,fields=fields,valid=valid,noise=eps,permutation=permutation,fits=fits,direct={'DINO':dd,'E':ed},GT_read=False,compute=dict(CPU_seconds=time.perf_counter()-tick,actual_fit_episodes=5,round_slots=10,informative_round_nodes=sum(t['updated_nodes']for f in fits.values()for t in f['rounds']),GPU_seconds=0,new_Source=0,new_DINO=0,new_training=0))))
  status('CPU_actual_OPD',ix+1,GT_read=False);print(a,ix+1,round(time.perf_counter()-start,3),flush=True)
 verify(full=True);verify(True);assert time.perf_counter()-start<p['CPU_adapt_seconds_cap'];put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all32_outputs_before_NEW_GT=True,actual_fit_episodes=160,round_slots=320,CPU_adapt_seconds=time.perf_counter()-start));status('all32_predictions_sealed',32,GT_read=False)

def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 torch.set_num_threads(4)
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_ca_query_opd32_v1 import ARMS
 p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_outputs_before_NEW_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-19/anchor_refresh32/RESULTS_PUBLIC.json')['rows']};rows=[]
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));f=load(D/'features'/(a+'.pt'));m=metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q])
  for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
   xx,yy=m['values']['E_MBR'][k],prior[a]['values']['E_MBR'][k];assert xx is None and yy is None or xx is not None and yy is not None and abs(xx-yy)<1e-12
  E=o['outputs']['E_MBR'];diag={key:dict(center_per_node_L2_mean=float(np.linalg.norm(v[:,:2]-E[:,:2],axis=1).mean()),whole_tube_L2=float(np.linalg.norm(v-E)),changed_nodes=int(np.any(v!=E,axis=1).sum()))for key,v in o['outputs'].items()};fdiag={k:dict(informative_node_rounds=sum(t['updated_nodes']for t in ff['rounds']),teacher_ESS_mean=float(torch.cat([1/t['feedback']['prob'].square().sum(1)for t in ff['rounds']]).mean()),visible_fraction_mean=float(torch.cat([t['feedback']['visible_fraction'].flatten()for t in ff['rounds']]).mean()))for k,ff in o['fits'].items()}
  rows.append(dict(alias=a,dataset=ds,active=o['active'],retained_anchors=len(o['positions']),diagnostics=diag,fit_diagnostics=fdiag,capture=dict(attention_reconstruction_error=f['attention_max_reconstruction_error'],content_DINO_E_difference=f['query_content_max_DINO_E_difference'],field_DINO_E_difference=f['probability_field_max_DINO_E_difference']),compute=dict(capture=f['compute'],adapt=o['compute']),**m))
 contrasts=[(a,'E_MBR')for a in ARMS if a!='E_MBR']+[('Dino_live',a)for a in ['Dino_fixed','Dino_direct','Dino_shuffle','Equery_live','Cosine_like_live']];groups={}
 for ds in gt:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=16,active=sum(r['active']for r in rr),metrics={a:{k:mean([r['values'][a][k]for r in rr])for k in ['v','t','s','observed_actual_s','unobserved_actual_s']}for a in ARMS},paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in ['v','s','observed_actual_s','unobserved_actual_s']}for a,b in contrasts})
 put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,GT_offline_after_seal=True,development_exposed=True,independent_confirmation=False,new_experts=0,new_training=0,actual_OPD=True));passed=all(groups[ds]['paired']['Dino_live_minus_E_MBR'][e]['mean']>0 for ds in gt for e in ['v','unobserved_actual_s'])and all(groups[ds]['paired']['Dino_live_minus_'+a]['v']['mean']>0 for ds in gt for a in ['Dino_fixed','Dino_direct','Dino_shuffle','Equery_live','Cosine_like_live']);put(P/'DECISION.json',dict(status='evaluated',passed=passed,rule=p['decision'],actual_OPD=True));put(D/'CPU_SCORE_COMPLETION.json',dict(status='complete',count=32));print('gate',passed,{ds:groups[ds]['paired']['Dino_live_minus_E_MBR']['v']['mean']for ds in gt},flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture','adapt','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),partials_preserved=True));raise
