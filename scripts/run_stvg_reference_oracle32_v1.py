"""Explicitly GT-assisted matched reference x updater diagnostic, never production TTA."""
from pathlib import Path
import argparse,collections,copy,gc,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_reference_oracle32_v1'
OLD=ROOT/'artifacts/stvg_joint_event_critic_v1'
ARMS={'A':('DINO','Direct'),'B':('DINO','OPD'),'C':('GT_oracle','Direct'),'D':('GT_oracle','OPD')}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,x,mutable=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists()and not mutable:raise FileExistsError(p)
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def save(p,x):
 import torch
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
 t=p.with_name(p.name+'.tmp');torch.save(x,t);t.replace(p)
 r=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,status='sealed',GT_assisted_diagnostic=True)
 put(p.with_suffix('.receipt.json'),r);return r
def load(p):
 import torch
 assert sha(p)==read(p.with_suffix('.receipt.json'))['sha256']
 return torch.load(p,map_location='cpu',weights_only=False)
def status(stage,done,total=32,**kw):put(BASE/'STATUS.json',dict(status='running',stage=stage,done=done,total=total,pid=os.getpid(),timestamp=time.time(),**kw),True)
def prepare():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 records=read(OLD/'PLAN_LOCK.json')['target'];assert read(OLD/'TARGET_PREDICTION_BARRIER.json')['count']==32
 cfg=read(ROOT/'methods/decota_spatial_opd_v1/configs.json')['datasets']
 plan=dict(status='locked_before_new_models',scope='GT-assisted reference x updater oracle diagnostic, NOT deployable TTA',
  records=records,count=32,arms=ARMS,initial='same frozen Source1792 per query; QR/LN/Adam reset in every arm',LN_writeback=0,
  common_support='original admitted DINO positions intersect exact target-GT box frames; no GT interpolation or added positions',
  primary_time='same Source Native interval for all arms',secondary_time='same pre-existing frozen Source projection interval, CPU readout only',
  updater_budget='same maximum Adam rounds and lr within domain; original OPD no-information steps retained and counted',
  Direct_objective='original (5*L1+2*(1-GIoU)) sum /4; fixed final iterate',OPD='original Gaussian M32 antithetic detached IoU/control-variate; no rollback, response gate or annealing',
  configurations={ds:cfg[ds]['config']for ds in ['hc2','vidstg']},
  GT_exposure='all four arms use GT-available support; C/D additionally use target GT coordinates; all results oracle diagnosis only',
  target_metrics_not_for_tuning=True,retain_all32_and_nonempty25_separately=True,new_DINO_calls=0,new_Temporal_updates=0,
  CURRENT_unchanged=True,created_epoch=time.time(),deadline_epoch=time.time()+2*3600)
 put(BASE/'PLAN_LOCK.json',plan)
 targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in records if r['dataset']==ds]))for ds in ['hc2','vidstg']}
 files=[];support=[]
 for r in records:
  z=load(OLD/'target'/(r['alias']+'.pt'));gt=targets[r['dataset']][0][r['query_ordinal']];span=targets[r['dataset']][1][r['query_ordinal']]
  selected=[a for a in z['expert']['anchors']['single4']if z['frame_ids'][a['position']]in gt]
  experts={}
  for kind in ['DINO','GT_oracle']:
   e=dict(positions4=z['expert']['positions4'],anchors={'single4':[]},observations={},new_DINO=0,GT_assisted_diagnostic=True)
   for a in selected:
    a=copy.deepcopy(a);pos=a['position'];fid=z['frame_ids'][pos]
    if kind=='GT_oracle':
     x,y,x2,y2=gt[fid];w=z['input']['width'];h=z['input']['height'];assert x2>x and y2>y
     a['box']=[(x+x2)/(2*w),(y+y2)/(2*h),(x2-x)/w,(y2-y)/h]
    old=z['expert']['observations'][('original',pos)];probe=dict(position=pos,frame_id=fid,accepted=True,reason='matched_existing_admitted_support',margin=old['probe'].get('margin'),boxes=torch.tensor([a['box']],dtype=torch.float32),target_scores=[a['score']],candidate_ids=[0])
    e['anchors']['single4'].append(a);e['observations'][('original',pos)]=dict(probe=probe,receipt=copy.deepcopy(old['receipt']))
   e['actual_observation_positions']=[a['position']for a in selected];experts[kind]=e
  assert [a['position']for a in experts['DINO']['anchors']['single4']]==[a['position']for a in experts['GT_oracle']['anchors']['single4']]
  files.append(save(BASE/'inputs'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],experts=experts,original_payload_sha256=sha(OLD/'target'/(r['alias']+'.pt')),positions=[a['position']for a in selected],GT_assisted_diagnostic=True)))
  support.append(dict(source_alias=r['alias'],dataset=r['dataset'],original_admitted=len(z['expert']['anchors']['single4']),common_exact_GT_anchors=len(selected),empty=not selected))
 assert sum(s['common_exact_GT_anchors']for s in support)==67 and sum(s['empty']for s in support)==7
 put(BASE/'SUPPORT_READBACK.json',dict(status='pass',count=32,nonempty=25,empty=7,anchors=67,rows=support))
 put(BASE/'INPUT_BARRIER.json',dict(status='complete',count=32,files=files,GT_assisted_diagnostic=True))
 own=['scripts/run_stvg_reference_oracle32_v1.py','vg_tta/decota_spatial_opd_tunable_v1.py','methods/decota_spatial_opd_v1/configs.json','methods/decota_final_simplified_v1/objectives.py','scripts/stvg_pipeline_speed_v1.py','scripts/run_stvg_joint_event_critic_v1.py','vg_tta/c1_enabling_tricks_v1.py','vg_tta/decota_optimizer_posterior_r1_v1.py','methods/decota_final_simplified_v1/replay.py','scripts/run_decota_paper_main_v1.py']
 put(BASE/'RUNTIME_LOCK.json',dict(status='locked',plan_sha256=sha(BASE/'PLAN_LOCK.json'),input_barrier_sha256=sha(BASE/'INPUT_BARRIER.json'),pins={p:sha(ROOT/p)for p in own},registries={p:sha(ROOT/p)for p in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']}))
 status('prepared',0,GT_assisted_diagnostic=True)
 print(json.dumps(dict(status='prepared',sources=32,nonempty=25,empty=7,anchors=67)))
def verify():
 lock=read(BASE/'RUNTIME_LOCK.json');p=read(BASE/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch']
 assert sha(BASE/'PLAN_LOCK.json')==lock['plan_sha256']and sha(BASE/'INPUT_BARRIER.json')==lock['input_barrier_sha256']
 for f,h in {**lock['pins'],**lock['registries']}.items():assert sha(ROOT/f)==h,f
 from main_method.runtime import resource_guard
 resource_guard();return p
def measure():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for,frames_for,read_row
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from scripts.stvg_pipeline_speed_v1 import full_prefix_readout
 from scripts.run_stvg_joint_event_critic_v1 import direct_fit
 from vg_tta.decota_spatial_opd_tunable_v1 import fit as opd_fit
 from vg_tta.c1_enabling_tricks_v1 import TrickReplay
 assert 'tubedetr'in sys.prefix
 p=verify();lease=gpu();domain=None;model=None;cache=collections.OrderedDict();files=[];started=time.time();newfits=gradcalls=0
 try:
  for i,r in enumerate(p['records']):
   verify();ds=r['dataset'];alias=r['alias'];dest=BASE/'predictions'/(alias+'.pt')
   if dest.exists():load(dest);files.append(read(dest.with_suffix('.receipt.json')));continue
   status('measuring',i,GT_assisted_diagnostic=True)
   if domain!=ds:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    domain=ds;model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());cache.clear()
   row=read_row(ds,r['query_ordinal']);frames,ids,_=frames_for(ds,row,cache);batch,records,base=frozen_forward(model,frames,row)
   old=load(OLD/'target'/(alias+'.pt'));inp=load(BASE/'inputs'/(alias+'.pt'))
   native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
   assert torch.equal(native['boxes'],old['boxes'][0])and native['physical_interval']==old['native_interval'];assert ids==old['frame_ids']
   rgbchecks=0
   for (_,pos),ob in old['expert']['observations'].items():assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==ob['receipt']['rgb_sha256'];rgbchecks+=1
   initial=detached(base.initial,'cpu');assert sum(x.numel()for x in initial.values())==1792
   with torch.no_grad():assert torch.equal(TrickReplay(base,ids,row['key'],{}).values()['boxes'].cpu(),native['boxes'])
   empty=not inp['positions'];arms={};timings={}
   for arm,(reference,updater)in ARMS.items():
    base.restore(initial);assert all(torch.equal(base.state()[k].cpu(),v)for k,v in initial.items())
    expert=inp['experts'][reference];tick=time.perf_counter()
    if empty:
     result=dict(initial=initial,state=initial,before=native['boxes'],final=native['boxes'],empty=True,gradient_calls=0,positions=[],path=[],GT_used=True,fit_invoked=False)
     full=native
    else:
     newfits+=1
     if updater=='Direct':
      result=direct_fit(base,expert,ds);result.update(before=native['boxes'],final=result['cached_boxes'],positions=inp['positions'])
     else:result=opd_fit(base,initial,expert,ids,row['key'],'on_policy',config=p['configurations'][ds])
     assert torch.equal(result['before'],native['boxes'])and all(torch.equal(result['initial'][k],v)for k,v in initial.items())
     full=full_prefix_readout(base,ids,records,result['state']);assert torch.equal(full['boxes'],result['final'])
     result['fitter_generic_GT_flag']=result['GT_used']if'GT_used'in result else result.get('GT_read',False)
     result.update(GT_used=True,GT_support_selected=True,oracle_coordinates=reference=='GT_oracle',fit_invoked=True)
     gradcalls+=result['gradient_calls']
    timings[arm]=time.perf_counter()-tick
    arms[arm]=dict(reference=reference,updater=updater,fit=detached(result,'cpu'),complete_output=detached(full,'cpu'),GT_assisted_diagnostic=True)
    assert all(torch.equal(base.state()[k].cpu(),v)for k,v in initial.items())
   assert state_hash(model.state_dict())==mh
   prior=load(ROOT/'artifacts/stvg_dino_rollback32_v1/predictions'/(alias+'.pt'))
   payload=dict(alias=alias,dataset=ds,query_ordinal=r['query_ordinal'],input=old['input'],frame_ids=ids,initial=initial,Source_boxes=native['boxes'],Native_interval=native['physical_interval'],projection_interval=prior['arms']['Full_P7A']['interval'],positions=inp['positions'],arms=arms,model_hash=mh,original_DINO_RGB_checks=rgbchecks,GT_assisted_diagnostic=True,metrics_used_for_update_or_selection=False,LN_writeback=0,compute=dict(new_Source_captures=1,new_DINO_calls=0,arm_seconds=timings))
   files.append(save(dest,detached(payload,'cpu')));status('measuring',i+1,GT_assisted_diagnostic=True)
   print(alias,i+1,'support',len(inp['positions']),'fits',newfits,'grad',gradcalls,flush=True)
   del frames,batch,base,old,inp,arms,payload,native,prior,result,full;gc.collect();torch.cuda.empty_cache()
  put(BASE/'PREDICTION_BARRIER.json',dict(status='complete',count=32,logical_arm_outputs=128,files=files,GT_assisted_diagnostic=True,new_Source_captures=32,new_DINO_calls=0,actual_nonempty_fit_calls=newfits,gradient_calls=gradcalls,wall_seconds=time.time()-started))
  status('predictions_complete',32,GT_assisted_diagnostic=True)
 finally:lease.close()
def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 from scripts.run_decota_paper_main_v1 import read_row
 p=read(BASE/'PLAN_LOCK.json');bar=read(BASE/'PREDICTION_BARRIER.json');assert bar['count']==32
 payloads=[load(ROOT/f['path'])for f in bar['files']]
 targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
 rows=[];checks=0
 for z in payloads:
  ds=z['dataset'];q=z['query_ordinal'];gt=targets[ds][0][q];span=targets[ds][1][q];row=read_row(ds,q)
  values={}
  for arm,boxes in [('Source',z['Source_boxes'])]+[(k,a['complete_output']['boxes'])for k,a in z['arms'].items()]:
   d=DenseTube(boxes.numpy(),row,gt,span,clip=ds=='hc2');values[arm]={}
   for support,iv in [('Native',z['Native_interval']),('projection',z['projection_interval'])]:
    m=official(boxes.numpy(),row,gt,span,iv,ds);assert max(abs(m[k]-d.score(iv)[k])for k in ['v','t','s'])<2e-10;checks+=3
    m.update(R_at_0p3=float(m['v']>.3),R_at_0p5=float(m['v']>.5));values[arm][support]=m
   masks={'observed':np.isin(d.fids,[z['frame_ids'][i]for i in z['positions']]),'native':np.isin(d.fids,z['frame_ids'])}
   values[arm]['geometry_support']={k:dict(count=int(mask.sum()),mean_IoU=float(d.iou[mask].mean())if mask.any()else None)for k,mask in [('observed',masks['observed']),('unobserved_native',masks['native']&~masks['observed']),('benchmark_interpolated',~masks['native'])]}
  rows.append(dict(source_alias=z['alias'],dataset=ds,common_anchors=len(z['positions']),empty=not z['positions'],values=values))
 groups={}
 for ds in ['hc2','vidstg']:
  groups[ds]={}
  for subset in ['all32','nonempty25']:
   rr=[r for r in rows if r['dataset']==ds and(subset=='all32'or not r['empty'])]
   groups[ds][subset]=dict(count=len(rr),means={a:{t:{m:float(np.mean([r['values'][a][t][m]for r in rr]))for m in ['v','t','s','R_at_0p3','R_at_0p5']}for t in ['Native','projection']}for a in ['Source',*ARMS]})
 put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,groups=groups,rows=rows,GT_assisted_diagnostic=True,no_deployable_TTA_claim=True,metric_consistency_checks=checks))
 put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,logical_arm_outputs=128,GT_assisted_diagnostic=True,independent_root_audit_pending=True))
 print(json.dumps(dict(status='complete',groups=groups)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','measure','score']);a=p.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  put(BASE/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),all_partial_preserved=True));raise
