"""Source-only finite TTFT calibration; immutable original adaptation math.

Inference, saved-state GPU replay audit, official scoring and CPU aggregation
audit are separate processes. No target scores are used to select a setting.
"""
from pathlib import Path
import os,sys,json,time,hashlib,gc,traceback,argparse
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
R=Path(__file__).resolve().parents[2]; S=Path(__file__).parent
sys.path[:0]=[str(S),str(R),str(R/'tools')]
import numpy as np,torch
import preflight_gpu as probe
from core import adapt,decoder_boxes,to_device,state_digest
from model_utils import cpu_source,spatial_modules,restore
from full import atomic,sha,identity,read
from stvg.prior import native_prior
from stvg.prototype import one_step
from stvg.representation import extract
from stvg.current_shared_roi import map_interval
from stvg.readout import prediction as native_readout
D=R/'outputs/expert_guided_ttft_v1'; C=D/'source_calibration_v1'
P=R/'results/expert_guided_ttft_v1/source_calibration_v1'
B=R/'outputs/spatial_temporal_six_v1/source_validation'
COHORT=R/'outputs/spatial_temporal_six_v1/source_validation_cohort_v1.json'
CONFIG=D/'calibration_selection_preparation_v1.json'
PREFLIGHT=D/'source_gradient_preflight_v2'
TARGETS=['hc2','vidstg']
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def save(p,x):probe.save(p,x)
def resource(start):
 probe.resources(start)
 assert sum(f.stat().st_size for f in C.rglob('*')if f.is_file())<160*2**30
def config_list(target):return read(CONFIG)['configs_in_tiebreak_order'][target]
def finish(ctx,boxes):
 f,pool=extract(ctx,boxes);q,pr=one_step(f,ctx['q0'],kappa=0,beta=1)
 return np.asarray(map_interval(ctx['frame_ids'],q)),f,q,pr,pool
def prepare():
 assert not(C/'plan.json').exists()
 z=read(PREFLIGHT/'source_gradient_preflight_complete.json')
 assert z['counts']==dict(Source=2,expert=0,optimizer_steps=10)
 assert read(PREFLIGHT/'actual_exit.json')['actual_tool_wait']
 files=list(S.glob('*.py'))+[D/'source_calibration_preflight_v1.json',COHORT,CONFIG,PREFLIGHT/'source_gradient_preflight_complete.json',PREFLIGHT/'actual_exit.json',PREFLIGHT/'execution_v2.json',R/'outputs/maintenance/cuda_repair_20261104/post_reboot_verification.json']
 files += [R/f for f in read(PREFLIGHT/'execution_v2.json')['code_and_weights_sha256']]
 for t in TARGETS:files += [B/t/'sealed.json']
 cohort=read(COHORT)
 for t in TARGETS:
  assert len(cohort[t]['rows'])==36
  dest=PREFLIGHT/'source_gradient_preflight'/t/cohort[t]['rows'][0]['id']
  files += [dest/'context.pt',dest/'context_checkpoint.json',dest/'passed.json',dest/'paid_Source_checkpoint.json']
 plan=dict(schema='expert_guided_TTFT_source36_each_direction_calibration_v1',
  scope='72 correct source-trained model/source-domain validation queries. Metadata preselected; no same-direction source-target video overlap; reverse-direction previous target exposure disclosed. Not the historical training36. Target labels never select configurations.',
  configs={t:config_list(t)for t in TARGETS},selection='Per direction maximum unrounded query-equal source-vIoU, first config on exact ties. All72*8 predictions sealed and separate GPU saved-state replay audited before source-only labels. Root review before target release.',
  branch='Full pos_fc+PosDecoder+bbox_embed, 14344208 unique scalars/288 tensors; fixed video/text/temporal encoders, cached actual frames_cls and spatial kwargs; actual fresh spatial forwards with gradients, L1+GIoU on same raw8 ReferDINO boxes. No E/shared correction/transport/candidate/DP. Adapted all-frame boxes pooled on native pretext grid, same one-pair kappa0/beta1/context1.5 posterior/MAP.',
  optimization='Four episodic AdamW streams/query to20, checkpoints5+20; every stream starts original parameters and fresh optimizer. lr1e-5/1e-4 x official source L1/GIoU ratio or1:1; eval dropout, clip.1, wd0/foreachFalse. All 8 configs reported, no target best step.',
  new_Source_cap=70,preflight_Source_reused=2,new_expert=0,new_optimizer_steps_cap=5760,preflight_optimizer_steps_separate=10,
  evidence='Every paid Source checkpoint BEFORE numerical validation. All context/source/frames_cls/native logits/records/RGB/8teacher refs SHA retained. All adapted state step5/20 saved and cold-forward exact in acquisition, then separate GPU process replays every saved state without optimization; CPU independently rederives every feature/prototype/interval and official scalar/means/selection. Full paid failure evidence retained, no fallback or silent retry.',
  precision='Original Source224 visualFP16/FP32 heads and gradients, cudnnTF32True/matmulFalse, Torch4. No AMP/changed backbone. Decoder replay must be bitwise Source before every stream and exact Source after every reset.',
  resources='1 GPU owner+at most1 root CPU job; CPU14-15 nice19 idleIO, RAM8GiB/disk80GiB/private160GiB/72h; GPU lease+live protected-service identity gate. Source acquisition/cached update/checkpoint wall distinct, nested costs not additive, not deployment timing.',
  no_paper_or_production_or_old_bound_code_changes=True,
  code_and_input_sha256={str(f.relative_to(R)):sha(f)for f in dict.fromkeys(files)})
 plan['fingerprint']=digest(plan);atomic(C/'plan.json',plan)
 atomic(P/'protocol.json',{k:v for k,v in plan.items()if k!='code_and_input_sha256'})
 print(plan['fingerprint'],flush=True)
def check():
 p=read(C/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for f,h in p['code_and_input_sha256'].items():assert sha(R/f)==h,f
 return p
def new_context(model,target,row,folder,parser,ds):
 from stvg.decode import decode
 from stvg.native import frozen_native_forward
 from stvg.representation import observe_native_grid
 from stvg.tensors import detached
 ref=read(B/target/'sealed.json')['files'][row['id']]
 assert sha(B/ref['native_path'])==ref['native_sha256']
 reference=load(B/ref['native_path']);frames,ids=decode(row['metadata'],ds)
 assert ids==reference['frame_ids']
 parsed={'subject':parser(row['metadata']['caption'])['subject']}
 atomic(folder/'attempt.json',dict(worker=identity(),before_Source=True,GT_read=False,RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),video_sha256=row['metadata']['video_sha256'],reference_native_sha256=ref['native_sha256']))
 classes=[];hook=model.ground_decoder.pos_fc.register_forward_pre_hook(lambda m,a:classes.append(detached(a[0],'cpu')))
 torch.set_num_threads(4);torch.manual_seed(20260920);np.random.seed(20260920);model.cuda();torch.cuda.synchronize();tick=time.perf_counter()
 try:
  with torch.no_grad(),observe_native_grid(model)as grids:batch,recs,native=frozen_native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses=parsed))
 finally:hook.remove()
 torch.cuda.synchronize();secs=time.perf_counter()-tick
 paid=dict(Source=native.zero['boxes'].cpu().numpy(),native_logits=detached(native.zero['logits'],'cpu'),records=recs,spatial_inputs=detached(native.spatial_inputs,'cpu'),frames_cls_calls=classes,native_local_grid=grids,frame_ids=ids,GT_read=False)
 save(folder/'paid_Source_complete.pt',paid);atomic(folder/'paid_Source_checkpoint.json',dict(worker=identity(),sha256=sha(folder/'paid_Source_complete.pt'),Source_seconds=secs,checkpoint_before_validation=True,GT_read=False))
 assert np.array_equal(paid['Source'],reference['Source'])
 assert len(paid['native_logits'])==len(reference['native_logits'])==2
 assert all(torch.equal(a,b)for a,b in zip(paid['native_logits'],reference['native_logits']))and recs==reference['records']and len(classes)>=4
 ctx={k:paid[k]for k in ['Source','spatial_inputs','native_local_grid','frame_ids','native_logits','records']};ctx['frames_cls']=[classes[-3],classes[-1]];ctx['q0']=native_prior(recs,paid['native_logits'],ids)
 for c,kw in zip(ctx['frames_cls'],ctx['spatial_inputs']):
  with torch.no_grad():v=model.ground_decoder.pos_fc(c.cuda()).sigmoid().unsqueeze(1)
  assert torch.equal(v.cpu(),kw['pred_boxes'])
 save(folder/'context.pt',ctx);atomic(folder/'context_checkpoint.json',dict(sha256=sha(folder/'context.pt'),actual_Source_before_validation_sha256=sha(folder/'paid_Source_complete.pt'),Source_exact=True,native_logits_records_exact=True,frames_cls_actual_readonly_observer=True,GT_read=False))
 model.cpu();del batch,native,frames,paid,classes,reference;gc.collect();torch.cuda.empty_cache()
 return ctx,secs
def acquire():
 plan=check();assert not(C/'worker.json').exists();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
 from stvg.runtime import setup,lease
 from stvg.query import QuerySubjectParser
 probe.foreign();lock=lease();setup();torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
 atomic(C/'worker.json',identity());start=time.perf_counter();cohort=read(COHORT);counts=dict(new_Source=0,reused_Source=0,new_expert=0,optimizer_steps=0,queries=0);allfiles={};model=None
 try:
  parser=QuerySubjectParser(R/'checkpoints/stanza')
  for target in TARGETS:
   resource(start);probe.foreign();tick=time.perf_counter();model,meta=cpu_source(target)
   atomic(C/target/'model_load.json',dict(worker=identity(),seconds=time.perf_counter()-tick,checkpoint_load=meta.get('checkpoint_load')))
   ds=cohort[target]['source_domain'];files={}
   for i,row in enumerate(cohort[target]['rows']):
    resource(start);probe.foreign();qstart=time.perf_counter();key=row['id'];folder=C/target/'paid'/key;assert not folder.exists();folder.mkdir(parents=True)
    e=read(B/target/'sealed.json')['files'][key];assert sha(B/e['input_path'])==e['input_sha256'];teacher=load(B/e['input_path']);assert not teacher['GT_read']
    if i==0:
     source=PREFLIGHT/'source_gradient_preflight'/target/key;cp=source/'context.pt';assert sha(cp)==read(source/'context_checkpoint.json')['sha256'];ctx=load(cp);counts['reused_Source']+=1;source_seconds=None
     atomic(folder/'reuse_context.json',dict(path=str(cp.relative_to(R)),sha256=sha(cp),preflight_paid_Source_checkpoint_sha256=sha(source/'paid_Source_checkpoint.json'),new_Source=0,inherited_Source_nonzero=True))
    else:
     ctx,source_seconds=new_context(model,target,row,folder,parser,ds);counts['new_Source']+=1;cp=folder/'context.pt'
    assert list(ctx['frame_ids'])==list(teacher['frame_ids']);model.cpu();torch.cuda.empty_cache();torch.set_num_threads(4)
    ids=np.asarray(ctx['frame_ids']);pos=np.asarray(teacher['positions8']);boxes=np.asarray(teacher['native_boxes']);assert boxes.shape==(8,4)
    nativeiv=native_readout(ctx['native_logits'],torch.from_numpy(ctx['Source']),ctx['records'],ids.tolist())['physical_interval']
    iv,f,q,pr,pool=finish(ctx,ctx['Source']);arrays=dict(frame_ids=ids,positions8=pos,Source=ctx['Source'].copy(),Source_interval=np.asarray(nativeiv),SourceTR=ctx['Source'].copy(),SourceTR_interval=iv)
    states={};trace={};configs=config_list(target)
    for j in [0,2,4,6]:
     tick=time.perf_counter();result=adapt(model,ctx,boxes,pos,configs[j],checkpoints=(5,20),scratch=folder/'states'/f'stream{j//2}');streamsecs=time.perf_counter()-tick;counts['optimizer_steps']+=20
     for cfgidx,step in [(j,5),(j+1,20)]:
      out=result[step];a=f'Config{cfgidx}';iv,_,_,_,_=finish(ctx,out['boxes']);arrays[a]=out['boxes'];arrays[a+'_interval']=iv;sp=folder/'states'/f'stream{j//2}'/f'step{step}.pt';states[a]=dict(path=str(sp.relative_to(C)),sha256=sha(sp),parameter_sha256=out['adapted_parameter_sha256'])
      trace[a]={k:v for k,v in out.items()if k not in ['boxes','parameter_update_norms']};trace[a]['shared_stream_total_seconds']=streamsecs
     del result;gc.collect();torch.cuda.empty_cache()
    out=C/target/'predictions'/(key+'.npz');out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
    with out.open('wb')as f:np.savez_compressed(f,**arrays)
    with np.load(out)as z:assert set(z.files)==set(arrays)and all(np.array_equal(z[k],a)for k,a in arrays.items())
    tp=folder/'adaptation_trace.json';atomic(tp,trace)
    rec=dict(path=str(out.relative_to(C)),sha256=sha(out),context_path=str(cp.relative_to(R)),context_sha256=sha(cp),teacher_path=str((B/e['input_path']).relative_to(R)),teacher_sha256=e['input_sha256'],states=states,trace_path=str(tp.relative_to(C)),trace_sha256=sha(tp),Source_seconds=source_seconds,reused_preflight_Source=i==0,GT_read=False,query_wall_seconds=time.perf_counter()-qstart,worker=identity())
    rp=C/target/'receipts'/(key+'.json');atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(C)),receipt_sha256=sha(rp));counts['queries']+=1
    atomic(C/'progress.json',dict(target=target,completed_in_domain=i+1,total_in_domain=36,counts=counts,worker=identity(),GT_read=False));del ctx,teacher,arrays;gc.collect()
   atomic(C/target/'sealed.json',dict(files=files,queries=36,source_domain=ds,GT_read=False,plan_fingerprint=plan['fingerprint'],worker=identity()));allfiles[target]=dict(seal_sha256=sha(C/target/'sealed.json'));model.cpu();del model;model=None;gc.collect();torch.cuda.empty_cache()
  assert counts==dict(new_Source=70,reused_Source=2,new_expert=0,optimizer_steps=5760,queries=72)
  atomic(C/'sealed.json',dict(status='all72_source_queries_all8configs_sealed_before_labels',files=allfiles,counts=counts,elapsed_wall_seconds=time.perf_counter()-start,worker=identity(),plan_fingerprint=plan['fingerprint'],GT_read=False));print('Source TTFT72 x8 sealed',flush=True)
 except BaseException:atomic(C/'failure.json',dict(error=traceback.format_exc(),counts=counts,worker=identity(),all_paid_state_preserved=True));raise
 finally:
  if model is not None:model.cpu()
  lock.close()
def seals():
 z=read(C/'sealed.json');assert z['counts']['queries']==72 and not z['GT_read']
 for t in TARGETS:assert z['files'][t]['seal_sha256']==sha(C/t/'sealed.json')
 return z
def checked_record(target,key):
 e=read(C/target/'sealed.json')['files'][key]
 for p,h in [('path','sha256'),('trace_path','trace_sha256'),('receipt_path','receipt_sha256')]:assert sha(C/e[p])==e[h]
 for p,h in [('context_path','context_sha256'),('teacher_path','teacher_sha256')]:assert sha(R/e[p])==e[h]
 return e
def gpu_audit():
 check();seals();assert not(C/'GPU_audit.json').exists();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
 from stvg.runtime import setup,lease
 probe.foreign();lock=lease();setup();torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();cases=0;model=None
 try:
  for target in TARGETS:
   tick=time.perf_counter();model,meta=cpu_source(target);atomic(C/target/'audit_model_load.json',dict(worker=identity(),seconds=time.perf_counter()-tick))
   original={n:{k:v.clone()for k,v in m.state_dict().items()}for n,m in spatial_modules(model).items()}
   for mod in spatial_modules(model).values():mod.cuda()
   for key in read(C/target/'sealed.json')['files']:
    resource(start);probe.foreign();e=checked_record(target,key);ctx=load(R/e['context_path']);device=to_device(ctx,'cuda')
    with np.load(C/e['path'])as z:
     restore(model,original)
     with torch.no_grad():before=decoder_boxes(model,device)
     assert np.array_equal(before.cpu().numpy(),z['Source'])
     for a,se in e['states'].items():
      assert sha(C/se['path'])==se['sha256'];state=load(C/se['path']);assert state_digest(state)==se['parameter_sha256'];restore(model,state)
      with torch.no_grad():boxes=decoder_boxes(model,device).cpu().numpy()
      assert np.array_equal(boxes,z[a]);cases+=1
     restore(model,original)
     with torch.no_grad():after=decoder_boxes(model,device).cpu().numpy()
     assert np.array_equal(after,z['Source'])
    del ctx,device;gc.collect();torch.cuda.empty_cache()
   model.cpu();del model;model=None;gc.collect();torch.cuda.empty_cache()
  assert cases==576
  atomic(C/'GPU_audit.json',dict(status='independent_process_all576_saved_adapted_states_native_Source_replay_and_reset_bitwise_exact',cases=cases,new_Source=0,new_expert=0,optimizer_steps=0,elapsed_wall_seconds=time.perf_counter()-start,worker=identity(),seal_sha256=sha(C/'sealed.json')));print('576 saved adapted states independently GPU exact',flush=True)
 finally:
  if model is not None:model.cpu()
  lock.close()
def label_reader(ds):
 # Import only in a post-seal CPU process. Inference never loads these modules.
 sys.path.insert(0,str(R/'studies/shared_corrected_roi_v1'))
 import shared_common
 return shared_common.labels(ds)
def scalar(boxes,ids,iv,pos,meta,g,ds,independent=False):
 from stvg.metrics import official,DenseTube,box_iou
 from stvg.metrics_base import xyxy
 gt={int(k):v for k,v in g['boxes'].items()};span=g['interval'];row=dict(input=meta,frame_ids=list(map(int,ids)))
 if independent:
  # Independent explicit dense interpolation, normalized-to-pixel conversion,
  # intersection/union, half-open temporal support and volume denominator.
  b=np.asarray(boxes,dtype=np.float64);wh=np.array([meta['width'],meta['height']]);p=np.concatenate([(b[:,:2]-b[:,2:]/2)*wh,(b[:,:2]+b[:,2:]/2)*wh],1)
  if ds=='hc2':p=np.maximum(p,0)
  ff=np.asarray(sorted(gt));v=np.array([np.interp(ff,ids,p[:,k])for k in range(4)]).T;bb=np.array([gt[int(f)]for f in ff]);inter=np.maximum(np.minimum(v[:,2:],bb[:,2:])-np.maximum(v[:,:2],bb[:,:2]),0).prod(1);den=np.maximum(v[:,2:]-v[:,:2],0).prod(1)+np.maximum(bb[:,2:]-bb[:,:2],0).prod(1)-inter;vv=np.divide(inter,den,out=np.zeros_like(inter),where=den>0);vv[(ff<ids[0])|(ff>ids[-1])]=0
  a,b=map(int,iv);lo,hi=span;overlap=max(0,min(b,hi)-max(a,lo));value=dict(t=overlap/(b-a+hi-lo-overlap),v=float(vv[(ff>=max(a,lo))&(ff<min(b,hi))].sum()/max(max(b,hi)-min(a,lo),1)),s=float(vv.mean()))
 else:
  value=official(boxes,row,gt,span,iv,ds);v=DenseTube(boxes,row,gt,span,clip=ds=='hc2').score(iv);assert max(abs(value[k]-v[k])for k in ['v','t','s'])<1e-12
 p=xyxy(boxes,meta['width'],meta['height']);p=np.maximum(p,0)if ds=='hc2'else p;observed=set(map(int,pos));ii=[i for i,f in enumerate(ids)if int(f)in gt and i not in observed];value['unobserved_actual_sIoU']=float(np.mean([box_iou(p[i],gt[int(ids[i])])for i in ii]))if ii else None
 value.update(R03=float(value['v']>=.3),R05=float(value['v']>=.5))
 return value
def report(rows):
 sys.path.insert(0,str(R/'studies/reviewer_key_controls_v1'))
 from statistics_v1 import stats,pair_stats
 out={}
 for t in TARGETS:
  rr=[r for r in rows if r['target']==t];arms=list(rr[0]['values']);means={a:{k:float(np.mean([r['values'][a][k]for r in rr if r['values'][a][k]is not None]))if any(r['values'][a][k]is not None for r in rr)else None for k in rr[0]['values'][a]}for a in arms};selected=max(range(8),key=lambda j:(means[f'Config{j}']['v'],-j));pairs={f'Config{j}-SourceTR':pair_stats(rr,f'Config{j}','SourceTR')for j in range(8)}
  out[t]=dict(queries=len(rr),parents=len({r['parent']for r in rr}),source_domain=rr[0]['source_domain'],means=means,paired=pairs,selected_index=selected,selected_config=config_list(t)[selected],selection_only_source_labels=True,target_release=False)
 return out
def derive(independent=False):
 seals();assert read(C/'GPU_audit.json')['seal_sha256']==sha(C/'sealed.json');rows=[];checks=0;maxerr=0
 for target,v in read(COHORT).items():
  ds=v['source_domain'];labels=label_reader(ds)
  for row in v['rows']:
   key=row['id'];e=checked_record(target,key);g=labels[key]
   with np.load(C/e['path'])as z:
    if independent:
     ctx=load(R/e['context_path']);teacher=load(R/e['teacher_path']);assert np.array_equal(z['Source'],ctx['Source'])and np.array_equal(z['frame_ids'],ctx['frame_ids'])and np.array_equal(z['positions8'],teacher['positions8'])
     for a in ['SourceTR']+[f'Config{j}'for j in range(8)]:
      iv,_,_,_,_=finish(ctx,z[a]);assert np.array_equal(iv,z[a+'_interval'])
     nativeiv=native_readout(ctx['native_logits'],torch.from_numpy(ctx['Source']),ctx['records'],ctx['frame_ids'])['physical_interval'];assert np.array_equal(z['Source_interval'],nativeiv)
    values={}
    for a in ['Source','SourceTR']+[f'Config{j}'for j in range(8)]:
     values[a]=scalar(z[a],z['frame_ids'],z[a+'_interval'],z['positions8'],row['metadata'],g,ds,independent);checks+=1
   rows.append(dict(target=target,source_domain=ds,id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
 return rows,checks
def score():
 check();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and not(C/'score_receipt.json').exists();torch.set_num_threads(1);start=time.perf_counter();rows,n=derive();rep=report(rows);atomic(C/'private_rows.json',rows);atomic(P/'report.json',rep);atomic(C/'score_receipt.json',dict(official_checks=n,seal_sha256=sha(C/'sealed.json'),GPU_audit_sha256=sha(C/'GPU_audit.json'),rows_sha256=sha(C/'private_rows.json'),report_sha256=sha(P/'report.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start));print('all source calibration scored pending independent CPU audit',flush=True)
def audit():
 check();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and not(C/'audit_receipt.json').exists();torch.set_num_threads(1);start=time.perf_counter();rows,n=derive(True);old=read(C/'private_rows.json');assert len(rows)==len(old)==72;mx=0
 for a,b in zip(rows,old):
  assert all(a[k]==b[k]for k in ['target','id','parent','duration_group'])
  for arm,v in a['values'].items():
   for f,x in v.items():
    y=b['values'][arm][f];assert(x is None)==(y is None)
    if x is not None:mx=max(mx,abs(x-y));assert abs(x-y)<1e-12
 rep=report(old);assert rep==read(P/'report.json');ind=report(rows)
 assert all(ind[t]['selected_index']==rep[t]['selected_index']for t in TARGETS)
 atomic(C/'audit_receipt.json',dict(status='independent_CPU_all_features_prototypes_intervals_input_file_SHA_official_vs_manual_dense_scalar_means_and_source_selection_verified_PENDING_ROOT',manual_scalar_checks=n,max_scalar_discrepancy=mx,report_sha256=sha(P/'report.json'),score_receipt_sha256=sha(C/'score_receipt.json'),GPU_audit_sha256=sha(C/'GPU_audit.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,target_release=False));print('source calibration independent CPU audit passed pending ROOT',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','acquire','gpu_audit','score','audit']);x=a.parse_args();globals()[x.stage]()
