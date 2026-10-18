"""Finite source training -> source qualification -> conditional five-arm target test."""
from pathlib import Path
import sys,os,time,collections,argparse,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
from scripts.run_stvg_dfine_source_capture_v1 import D,P,TARGET,verify as capture_verify,status
from vg_tta.stvg_dfine_refinement32_v1 import Refiner,prepare_query,pack,distribution_loss,outputs
from vg_tta.stvg_discriminative_dense32_v1 import noise
V=None

def verify():
 global V
 from main_method.runtime import PinVerifier,resource_guard
 p=capture_verify();l=read(D/'TRAINING_RUNTIME_LOCK.json')
 if V is None:V=PinVerifier(l['pins'])
 V.verify();resource_guard();return p

def lock():
 p=capture_verify();bar=read(D/'SOURCE_FEATURE_BARRIER.json');assert bar['count']==118
 files=['vg_tta/stvg_dfine_refinement32_v1.py','scripts/run_stvg_dfine_refinement32_v1.py','scripts/audit_stvg_dfine_refinement32_v1.py','protocols/stvg_dfine_refinement32_v1.md','protocols/stvg_dfine_refinement32_source_readout_v1.md']
 files +=[str((D/'source_supervision'/(r['alias']+'.pt')).relative_to(ROOT))for r in p['source']]
 files +=[f['path']for f in bar['files']]
 for r in p['target']:files +=[str((TARGET/'features'/(r['alias']+'.pt')).relative_to(ROOT)),str((TARGET/'feedback'/(r['alias']+'.pt')).relative_to(ROOT))]
 put(D/'TRAINING_RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in files},source_barrier_sha256=sha(D/'SOURCE_FEATURE_BARRIER.json'),plan_sha256=sha(D/'PLAN_LOCK.json')));print('TRAINING_LOCK',len(files),flush=True)

def source_data(records):
 data=[]
 for r in records:
  x=load(D/'source_features'/(r['alias']+'.pt'));y=load(D/'source_supervision'/(r['alias']+'.pt'));z=prepare_query(x);gt=y['gt'];ids=x['frame_ids'];known=np.array([i for i,f in enumerate(ids)if int(f)in gt],dtype=int);truth=torch.zeros(len(ids),4)
  for i in known:
   b=np.array(gt[int(ids[i])])/np.array([r['input']['width'],r['input']['height']]*2);truth[i]=torch.tensor(np.r_[(b[:2]+b[2:])/2,b[2:]-b[:2]])
  data.append(dict(record=r,raw=x,query=z,label=y,known=known,truth=truth,noise=noise('dfine-source-v1|'+r['alias'],len(ids))))
 return data

def lease_wait():
 from scripts.run_decota_paper_main_v1 import gpu
 while True:
  verify()
  try:return gpu()
  except BlockingIOError:status('waiting_shared_GPU_lease',0);time.sleep(10)


def current_budget():return read(D/'SOURCE_GPU_STOP_LATEST.json')['actual_GPU_elapsed_seconds']

def evaluate(model,data,cfg,device):
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 rows=[];packets=[];model.eval()
 for item in data:
  r=item['record'];x=item['raw'];z=item['query'];eps=noise('dfine-source-v1|'+r['alias'],len(z['E']));result=outputs(model,z,r['alias'],eps,cfg,x['planned_positions'],device);out=dict(alias=r['alias'],dataset=r['dataset'],frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],interval=x['interval'],**result)
  aux=result['paths']['Direct_refine2'][0]['teacher']['stage_boxes'][-1].double().numpy();aux[x['planned_positions']]=x['E'][x['planned_positions']].numpy();out['outputs']['Teacher_boundary_mean2']=aux if z['active']else x['E'].numpy();vals=metrics(out,dict(input=r['input'],frame_ids=x['frame_ids']),item['label']['gt'],item['label']['span']);base=vals['values']['E_MBR'];final=vals['values']['Direct_refine2'];known=item['known'];bb=torch.tensor(result['outputs']['Direct_refine2']);
  from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
  initial_iou=np.array([iou_gt(x['E'][i].numpy(),item['label']['gt'][int(x['frame_ids'][i])],dict(input=r['input']),r['dataset'])for i in known]);refined_iou=np.array([iou_gt(bb[i].numpy(),item['label']['gt'][int(x['frame_ids'][i])],dict(input=r['input']),r['dataset'])for i in known]);accuracy={}
  for name,mask in [('E_good_ge075',initial_iou>=.75),('E_poor_lt05',initial_iou<.5)]:accuracy[name]=dict(nodes=int(mask.sum()),delta_IoU_pp=float(100*(refined_iou[mask]-initial_iou[mask]).mean())if mask.any()else None)
  rows.append(dict(alias=r['alias'],dataset=r['dataset'],active=z['active'],values=vals['values'],accuracy=accuracy,stage_invalid=result['paths']['Direct_refine2'][0]['teacher']['stage_invalid'],sample_invalid_fraction=[p['teacher']['invalid_fraction']for p in result['paths']['Direct_refine2']]))
  packets.append(out)
 return rows,packets

def train():
 p=verify();from scripts.run_decota_paper_main_v1 import gpu
 lease=lease_wait();begin=time.perf_counter();spent=current_budget();remaining=3600-spent;assert remaining>0;put(D/'TRAIN_GPU_RUNTIME001.json',dict(pid=os.getpid(),remaining_seconds=remaining,already_spent_seconds=spent,start_epoch=time.time()));summaries=[];totalmodels=0
 def guard():verify();assert spent+time.perf_counter()-begin<3600,'total newly held GPU budget reached'
 try:
  for ds in ['hc2','vidstg']:
   guard();status('source_head_train',totalmodels,source_domain=ds);dd=source_data([r for r in p['source']if r['dataset']==ds]);tr=[v for v in dd if v['record']['split']=='source_train'];va=[v for v in dd if v['record']['split']=='source_validation'];active=[v for v in tr if v['query']['active']and len(v['known'])];assert active
   residual=[]
   from vg_tta.stvg_dfine_refinement32_v1 import edges
   for item in tr:
    ix=item['known'];E=item['query']['E'][ix].float();T=item['truth'][ix];residual.extend(((edges(T)-edges(E))/torch.cat((E[:,2:],E[:,2:]),-1).clamp_min(1e-5)).abs().flatten().tolist())
   radius=float(np.clip(np.quantile(residual,.95),.25,2.));torch.manual_seed(p['seed']);np.random.seed(p['seed']);model=Refiner(radius).cuda();initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(p['seed']);history=[];best=None;cfg=p['target_configuration'];valfiles=[];start=time.perf_counter()
   for epoch in range(1,31):
    model.train();losses=[];aux=[]
    for step in range(6):
     guard();chosen=rng.integers(len(active),size=64);items=[active[i]for i in chosen];positions=[int(v['known'][rng.integers(len(v['known']))])for v in items];batch=pack([v['query']for v in items],positions,'cuda');cur=batch['E'].clone();truth=torch.stack([v['truth'][i]for v,i in zip(items,positions)]).cuda();kinds=rng.integers(3,size=64)
     for j,(v,i,kind)in enumerate(zip(items,positions,kinds)):
      if kind==1:cur[j]=v['raw']['Source'][i].float().cuda()
      elif kind==2:cur[j,:2]+=v['noise'][epoch%2,i,int(rng.integers(17))].float().cuda()*.25*cur[j,2:]
     opt.zero_grad(set_to_none=True);st=model(batch,cur);loss,extra=distribution_loss(st,batch['E'],truth,model.bins);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()));aux.append(extra)
    record=dict(epoch=epoch,mean_loss=float(np.mean(losses)),outside_bins=float(np.mean([a['outside_bins']for a in aux])),invalid_stage_fraction=float(np.mean([a['invalid_stage_fraction']for a in aux])))
    if epoch%5==0:
     guard();rows,packets=evaluate(model,va,cfg,'cuda');score=float(np.mean([r['values']['Direct_refine2']['v']for r in rows]));dv=float(np.mean([r['values']['Direct_refine2']['v']-r['values']['E_MBR']['v']for r in rows]));sv=[r['values']['Direct_refine2']['unobserved_actual_s']-r['values']['E_MBR']['unobserved_actual_s']for r in rows if r['values']['Direct_refine2']['unobserved_actual_s']is not None and r['values']['E_MBR']['unobserved_actual_s']is not None];assert sv,'No source-val actual-node GT support';s=float(np.mean(sv));record.update(validation_v=score,validation_delta_v_pp=100*dv,validation_delta_unobserved_s_pp=100*s)
     file=D/'source_validation'/ds/f'epoch{epoch:02}.pt';valfiles.append(save(file,dict(epoch=epoch,rows=rows,packets=packets,source_validation_only=True)));save(D/'models'/ds/f'epoch{epoch:02}.pt',dict(state={k:v.detach().cpu()for k,v in model.state_dict().items()},radius=radius,epoch=epoch,parameters=sum(v.numel()for v in model.parameters()),source_domain=ds))
     if best is None or score>best['validation_v']:best=dict(epoch=epoch,validation_v=score,delta_v_pp=100*dv,delta_unobserved_s_pp=100*s)
     print(ds,'epoch',epoch,'loss',record['mean_loss'],'val_dv_pp',100*dv,'val_ds_pp',100*s,flush=True)
    history.append(record);put(D/'TRAINING_PROGRESS.json',dict(dataset=ds,epoch=epoch,history=history,GPU_total_spent_seconds=spent+time.perf_counter()-begin),True)
   assert best;changed=sum(not torch.equal(initial[k],v.cpu())for k,v in model.state_dict().items());assert changed>0
   summaries.append(dict(source_domain=ds,training_parents=43,validation_parents=16,active_training_parents=len(active),radius=radius,parameters=sum(v.numel()for v in model.parameters()),selected=best,history=history,source_qualification_pass=best['delta_v_pp']>0 and best['delta_unobserved_s_pp']>0,teacher_checkpoint_sha256=sha(D/'models'/ds/f"epoch{best['epoch']:02}.pt"),validation_files=valfiles,seconds=time.perf_counter()-start,changed_state_tensors=changed));totalmodels+=1;del model,opt,dd,tr,va,active;gc.collect();torch.cuda.empty_cache()
  gate=all(s['source_qualification_pass']for s in summaries);put(P/'SOURCE_TRAINING.json',dict(status='complete',models=2,rows=summaries,seed=20261018,one_configuration=True,source_only=True));put(P/'SOURCE_GATE.json',dict(status='pass'if gate else'not_passed',passed=gate,both_source_domains_required=True,criterion='center-only direct2 complete vIoU and unobserved actual sIoU each improve E',selected=[s['selected']for s in summaries],target_not_started_when_source_gate_recorded=True));put(D/'TRAINING_COMPLETION.json',dict(status='complete',models=2,source_gate_passed=gate));status('source_gate_pass'if gate else'source_gate_not_passed_stop_before_target',2)
 finally:
  elapsed=time.perf_counter()-begin;lease.close();put(D/'TRAIN_GPU_STOP.json',dict(actual_GPU_elapsed_seconds=elapsed,cumulative_GPU_seconds=spent+elapsed,all_partials_preserved=True))

def target():
 p=verify();assert read(P/'SOURCE_GATE.json')['passed'];spent=read(D/'TRAIN_GPU_STOP.json')['cumulative_GPU_seconds'];from scripts.run_decota_paper_main_v1 import gpu
 lease=lease_wait();begin=time.perf_counter();models={};cfg=p['target_configuration'];files=[]
 try:
  for i,r in enumerate(p['target']):
   verify();assert spent+time.perf_counter()-begin<3600;ds=r['dataset'];source='vidstg'if ds=='hc2'else'hc2';a=r['alias'];status('target_five_arm_inference',i,alias=a)
   s=next(v for v in read(P/'SOURCE_TRAINING.json')['rows']if v['source_domain']==source)
   if source not in models:
    cp=load(D/'models'/source/f"epoch{s['selected']['epoch']:02}.pt");m=Refiner(cp['radius']).cuda().eval().requires_grad_(False);m.load_state_dict(cp['state']);models[source]=m
   x=load(TARGET/'features'/(a+'.pt'));old=load(TARGET/'feedback'/(a+'.pt'));z=prepare_query(x);out=outputs(models[source],z,a,old['feedback']['noise'],cfg,x['planned_positions'],'cuda');assert np.array_equal(out['outputs']['E_MBR'],x['E'].numpy())
   for b in out['outputs'].values():assert np.array_equal(b[:,2:],x['E'][:,2:].numpy())and np.array_equal(b[x['planned_positions']],x['E'][x['planned_positions']].numpy())
   files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=ds,frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],interval=x['interval'],**out,GT_read=False,source_checkpoint_sha256=s['teacher_checkpoint_sha256'])));print('TARGET',a,i+1,flush=True)
  put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all32_before_new_target_GT=True,historical_target_development_exposure=True,new_target_DINO_calls=0))
 finally:elapsed=time.perf_counter()-begin;lease.close();put(D/'TARGET_GPU_STOP.json',dict(actual_GPU_elapsed_seconds=elapsed,cumulative_GPU_seconds=spent+elapsed))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['lock','train','target']);x=a.parse_args()
 try:globals()[x.stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=x.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
