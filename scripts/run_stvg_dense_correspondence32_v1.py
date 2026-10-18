"""Finite four-arm online mechanism panel; all predictions seal before any GT."""
from pathlib import Path
import sys,time,os,gc,collections,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha,save,load
D=ROOT/'artifacts/stvg_dense_correspondence32_v1'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
ARMS=['A_sparse','B_raw','C_rank','D_interpolation']

def prepare():
 import torch
 from vg_tta.stvg_dense_correspondence_opd_v1 import contract
 assert not(D/'PLAN_LOCK.json').exists()
 design=read(OLD/'DESIGN_LOCK.json');plans={ds:read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')for ds in ['hc2','vidstg']};excluded={ds:set()for ds in plans}
 for r in read(ROOT/'artifacts/stvg_dino_rollback32_v1/PLAN_LOCK.json')['records']:excluded[r['dataset']].add(r['parent'])
 for f in read(ROOT/'artifacts/stvg_interval_decision_v1/PLAN_LOCK.json')['files']:
  assert sha(ROOT/f['path'])==f['sha256'];z=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);ds=f['dataset'];excluded[ds].add(plans[ds]['rows'][z['query_ordinal']]['source']);del z
 for ds in plans:
  for q in design['datasets'][ds]['development_query_ordinals']:excluded[ds].add(plans[ds]['rows'][q]['source'])
 records=[]
 for ds in plans:
  used=set();bar=read(OLD/f'stages/P1_{ds}/PREDICTION_BARRIER.json')['inputs']
  for q in design['stages']['P1_'+ds]['orders']['order1']:
   row=plans[ds]['rows'][q];parent=row['source']
   if parent in excluded[ds]or parent in used:continue
   used.add(parent);di='vidstg_to_hc2'if ds=='hc2'else'hcstvg2_to_vidstg';rel=f'inputs/{di}/clean/{q:05}.pt';assert sha(OLD/rel)==bar[rel]
   records.append(dict(alias=('H'if ds=='hc2'else'V')+f'{len(used):03}',dataset=ds,parent=parent,query_ordinal=q,input=str((OLD/rel).relative_to(ROOT)),input_sha256=bar[rel],order=len(used)-1))
   if len(used)==16:break
  assert len(used)==16
 own=['scripts/run_stvg_dense_correspondence32_v1.py','vg_tta/stvg_dense_correspondence_opd_v1.py','vg_tta/stvg_soft_correspondence_v1.py','vg_tta/decota_spatial_opd_tunable_v1.py','vg_tta/c1_enabling_tricks_v1.py','vg_tta/spatial_online_state_v1.py','scripts/stvg_opd_paper_common_v1.py','scripts/run_decota_paper_main_v1.py','scripts/stvg_pipeline_speed_v1.py','scripts/stvg_opd_legacy_input_bridge003.py','protocols/stvg_dense_correspondence32_v1.md','methods/decota_spatial_opd_v1/configs.json']
 own +=[str(f.relative_to(ROOT))for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py')]
 put(D/'PLAN_LOCK.json',dict(status='locked_before_models_GT',count=32,records=records,excluded_parent_counts={ds:len(v)for ds,v in excluded.items()},selection='first eligible unique parent in preexisting P1 order1 excluding rollback32 P6_64 and old_tuning parents; no reference or outcome filtering',historical_full_GT_exposure=True,independent_confirmation=False,arms=ARMS,configs={ds:read(ROOT/'methods/decota_spatial_opd_v1/configs.json')['datasets'][ds]['config']for ds in plans},dense_coefficient=1.,dense_reduction='mean over actual decoder frames excluding all planned DINO positions',correspondence_temperature=.07,stream='own O-split branch carry from Source, same filtered order; independent reset per target',final_time='Source Native fixed',new_DINO_calls=0,new_Source_captures=32,max_fits=128,GPU_seconds_cap=3600,deadline_epoch=time.time()+7200,GT_before_all128_seal=False,gate=dict(C_minus_A_and_D_positive_both_domains=True,leave_one_parent_out_mean_positive=True,severe_current_harm_threshold_pp=-5.,severe_harm_count_allowance_over_A=1,CI_descriptive_not_sole_gate=True,no_automatic_search=True)))
 put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in own},registries={f:sha(ROOT/f)for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']},plan_sha256=sha(D/'PLAN_LOCK.json')))
 put(D/'CPU_CONTRACT_COMPLETION.json',contract());put(D/'STATUS.json',dict(status='prepared',done=0,total=32,GT_read=False),True)
 print('LOCKED32',[(r['alias'],r['query_ordinal'])for r in records],flush=True)

def verify():
 p=read(D/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch'];l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
 for f,h in {**l['pins'],**l['registries']}.items():assert sha(ROOT/f)==h,f
 from main_method.runtime import resource_guard
 resource_guard();return p

def measure():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for,frames_for,read_row,unpack_expert
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from vg_tta.stvg_soft_correspondence_v1 import propagate
 from vg_tta.stvg_dense_correspondence_opd_v1 import fit
 from vg_tta.decota_optimizer_posterior_r1_v1 import Energy
 from vg_tta.spatial_online_state_v1 import arrival
 from scripts.stvg_opd_paper_common_v1 import committed
 p=verify();lease=gpu();started=time.perf_counter();domain=None;model=None;cache=collections.OrderedDict();files=[];quals=[];previous={a:None for a in ARMS}
 try:
  for i,r in enumerate(p['records']):
   verify();assert time.perf_counter()-started<p['GPU_seconds_cap'];ds=r['dataset'];dest=D/'predictions'/(r['alias']+'.pt')
   if ds!=domain:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    domain=ds;model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());previous={a:None for a in ARMS};cache.clear()
   if dest.exists():
    z=load(dest);previous={a:z['arms'][a]['committed']for a in ARMS};files.append(read(dest.with_suffix('.receipt.json')));continue
   put(D/'STATUS.json',dict(status='running',stage='Source_capture',done=i,total=32,current=r['alias'],GT_read=False),True)
   assert sha(ROOT/r['input'])==r['input_sha256'];inp=normalized(torch.load(ROOT/r['input'],map_location='cpu',weights_only=False));assert inp['source_model_state_sha256']==mh
   row=read_row(ds,r['query_ordinal']);frames,ids,_=frames_for(ds,row,cache);assert ids==inp['frame_ids'];pixel=hashlib.sha256(frames.tobytes()).hexdigest();assert pixel==inp['pixel_sha256']
   appearance=[];masks=[]
   def ah(module,args,out):appearance.append(out.detach().float().cpu())
   def vh(module,args,out):
    x,mask,*_=out[0].decompose();masks.append((~mask).detach().cpu())
   hooks=[model.input_proj.register_forward_hook(ah),model.vis_encoder.register_forward_hook(vh)];tick=time.perf_counter()
   try:batch,recs,base=frozen_forward(model,frames,row)
   finally:
    for h in hooks:h.remove()
   native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids);assert torch.equal(native['boxes'],inp['native_boxes'])and native['physical_interval']==inp['interval']and native['indices']==inp['indices'];assert len(appearance)==len(masks)==2
   expert=unpack_expert(inp['expert'])
   for(_,pos),ob in expert['observations'].items():assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==ob['receipt']['rgb_sha256']
   H,W=appearance[0].shape[-2:];maps=torch.empty(len(ids),appearance[0].shape[1],H,W);valid=torch.empty(len(ids),H,W,dtype=torch.bool)
   for off in [0,1]:maps[off::2]=appearance[off];valid[off::2]=masks[off]
   support=Energy(expert,base.zero['boxes'],'top1');ap=[int(f[0])for f in support.frames];ab=[f[1][0].cpu()for f in support.frames];soft=propagate(maps.cuda(),valid.cuda(),ap,ab,p['correspondence_temperature']);unobs=[j for j in range(len(ids))if j not in expert['positions4']]
   if ap:
    order=np.argsort(ap);aa=np.array(ap)[order];bb=torch.stack(ab).numpy()[order];physical=np.array(ids);ref=np.stack([np.interp(physical,physical[aa],bb[:,c])for c in range(4)],1);ref=torch.tensor(ref,dtype=torch.float32)
   else:ref=torch.zeros(len(ids),4)
   feature_seconds=time.perf_counter()-tick;save(D/'features'/(r['alias']+'.pt'),dict(maps=maps,valid=valid,soft=soft,interpolated=ref,admitted_positions=ap,frame_ids=ids,records=recs,Source_zero=detached(base.zero,'cpu'),Source_initial=detached(base.initial,'cpu'),GT_read=False))
   del batch,appearance,masks;arms={};cfg=p['configs'][ds]
   for ai,a in enumerate(ARMS):
    verify();assert time.perf_counter()-started<p['GPU_seconds_cap'];put(D/'STATUS.json',dict(status='running',stage=a,done=i,total=32,arm_done=ai,current=r['alias'],GT_read=False),True)
    initial=arrival(base.initial,previous[a],'O-split');bt=time.perf_counter();z=fit(base,initial,expert,ids,row['key'],a,config=cfg,softmaps=soft,interpolated=ref,unobserved=unobs)
    # Independent full two-pass Decoder readout, without mutating mother Source.
    base.restore({n:v.to(base.initial[n])for n,v in z['state'].items()})
    with torch.no_grad():full=base.reference_values(capture=False)
    assert torch.equal(full['boxes'].cpu(),z['final']),('full readout',r['alias'],a)
    z['full_readout_exact']=True;z['committed']=committed(z['initial'],z['state'],cfg['writeback']);z['wall_seconds']=time.perf_counter()-bt;previous[a]=z['committed'];arms[a]=z
   assert state_hash(model.state_dict())==mh
   if not any(q['dataset']==ds for q in quals):quals.append(dict(dataset=ds,Source_complete_output_exact=True,DINO_pixels_exact=True,frozen_features=True,all_four_full_decoder_readouts_exact=True,active_parameters=1792))
   out=dict(alias=r['alias'],dataset=ds,query_ordinal=r['query_ordinal'],frame_ids=ids,Native_interval=native['physical_interval'],Source_boxes=native['boxes'],Source_initial=detached(base.initial,'cpu'),planned_positions=expert['positions4'],admitted_positions=ap,expert=detached(expert,'cpu'),arms=arms,GT_read=False,compute=dict(new_DINO_calls=0,new_Source_capture=1,feature_seconds=feature_seconds),input_sha256=r['input_sha256'],Source_state_sha256=mh)
   files.append(save(dest,out));print(r['alias'],i+1,'all_four_complete','seconds',time.perf_counter()-started,flush=True)
   del out,arms,z,base,frames,maps,valid,soft,ref,expert;gc.collect();torch.cuda.empty_cache()
  verify();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,fits=128,files=files,GT_read=False,GPU_lease_seconds=time.perf_counter()-started));put(D/'QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,rows=quals));put(D/'STATUS.json',dict(status='complete',stage='all_four_arms_sealed_before_GT',done=32,total=32,GT_read=False),True)
 finally:lease.close()

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','measure']);stage=a.parse_args().stage
 try:globals()[stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',type=type(e).__name__,error=str(e),partial_preserved=True));raise
