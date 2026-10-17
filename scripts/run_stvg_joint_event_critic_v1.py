"""Finite source-supervised visual event-critic experiment, separate from CURRENT."""
from pathlib import Path
import argparse,collections,gc,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_joint_event_critic_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,x,mutable=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and not mutable:raise FileExistsError(p)
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def save(p,x):
 import torch
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
 t=p.with_name(p.name+'.tmp');torch.save(x,t);t.replace(p)
 r=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,status='sealed')
 put(p.with_suffix('.receipt.json'),r);return r
def load(p):
 import torch
 p=Path(p);assert sha(p)==read(p.with_suffix('.receipt.json'))['sha256']
 return torch.load(p,map_location='cpu',weights_only=False)
def status(stage,done,total,**kw):put(BASE/'STATUS.json',dict(status='running',stage=stage,done=done,total=total,pid=os.getpid(),timestamp=time.time(),**kw),True)
def verify():
 lock=read(BASE/'PLAN_LOCK.json');assert time.time()<lock['deadline_epoch'],'finite pilot deadline reached'
 for f,h in read(BASE/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f
 from main_method.runtime import resource_guard
 resource_guard();return lock
def prepare():
 import torch
 from methods.decota_final_simplified_v1.observations import QuerySubjectParser,visual_query,parse_context
 torch.set_num_threads(4);assert not(BASE/'PLAN_LOCK.json').exists()
 old=read(ROOT/'artifacts/stvg_dino_rollback32_v1/PLAN_LOCK.json');target=[];excluded={d:set() for d in ['hc2','vidstg']}
 for r in old['records']:
  excluded[r['dataset']].add(r['parent']);target.append({**r,'split':'target_development'})
 parser=QuerySubjectParser(ROOT/'.cache/stanza');source=[];provenance={};removed=[]
 for ds,di in [('hc2','hc2_to_vid'),('vidstg','vid_to_hc1')]:
  f=ROOT/'artifacts/decota_paper_execution_20260917/vitta'/di/'SOURCE_MANIFEST.json';m=read(f)
  assert m['official_split']=='train';provenance[ds]=dict(manifest=str(f.relative_to(ROOT)),manifest_sha256=sha(f),annotation=m['annotation'],annotation_sha256=m['annotation_sha256'])
  available=[]
  for r in m['records']:
   if r['input']['source'] in excluded[ds]:removed.append(dict(dataset=ds,parent=r['input']['source'],reason='parent_overlaps_target_pilot'));continue
   assert Path(r['input']['video_path']).is_file();available.append(r)
  available.sort(key=lambda r:hashlib.sha256(('joint-event-v1|'+r['input']['source']).encode()).hexdigest())
  assert len(available)>=48;selected=available[:48];assert len({r['input']['source'] for r in selected})==48
  for i,r in enumerate(selected):
   q=dict(r['input']);assert sha(q['video_path'])==q['video_sha256']
   if 'fps' not in q:
    rate=subprocess.check_output([str(ROOT/'.conda/tubedetr/bin/ffprobe'),'-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate','-of','default=nw=1:nk=1',q['video_path']],text=True).strip();a,b=rate.split('/');q['fps']=float(a)/float(b)
   parses=parser(q['caption']);parses.update(old=visual_query(parser,q['caption']),context=parse_context(parser,q['caption']))
   source.append(dict(alias=('SH' if ds=='hc2' else 'SV')+f'{i+1:03}',dataset=ds,source_domain=ds,parent=q['source'],input=q,frame_ids=q['frame_ids'],key=r['key'],annotation_key=r['official_annotation_key'],parses=parses,split='source_train' if i<36 else 'source_validation'))
 own=['scripts/run_stvg_joint_event_critic_v1.py','vg_tta/stvg_joint_event_critic_v1.py','protocols/stvg_joint_event_critic_v1.md','tests/test_stvg_joint_event_critic_v1.py']
 pins=read(ROOT/'artifacts/stvg_dino_rollback32_v1/RUNTIME_LOCK.json')['pins']
 pins={k:v for k,v in pins.items() if Path(ROOT/k).exists()};pins.update({p:sha(ROOT/p) for p in own})
 for ds,p in provenance.items():pins[p['manifest']]=p['manifest_sha256']
 put(BASE/'PLAN_LOCK.json',dict(status='locked',scope='source-supervised finite cross-domain visual tube critic',created_epoch=time.time(),deadline_epoch=time.time()+4*3600,target=target,source=source,source_provenance=provenance,excluded_source_parent_overlaps=removed,source_counts=dict(train_per_domain=36,validation_per_domain=12),target_per_domain=16,critic=dict(seed=20261017,epochs=80,batch_queries=4,lr=.001,weight_decay=.0001),CURRENT_unchanged=True,official_full_not_authorized_by_this_pilot=True))
 put(BASE/'RUNTIME_LOCK.json',dict(status='locked',plan_sha256=sha(BASE/'PLAN_LOCK.json'),pins=pins))
 put(BASE/'DATA_SPLIT_READBACK.json',dict(status='pass',source_count=96,target_count=32,source_train_count=72,source_validation_count=24,parent_overlap_each_direction=0,removed_overlaps=len(removed),source_GT_not_network_input=True,target_GT_training=False))
 status('prepared',0,128,GT_read=False)

def direct_fit(base,expert,ds):
 import torch
 from methods.decota_final_simplified_v1.objectives import SpatialLoss
 from methods.decota_final_simplified_v1.tensors import detached
 initial=base.state();base.restore(base.initial);assert all(torch.equal(base.state()[n],v)for n,v in base.initial.items())
 lossfn=SpatialLoss(expert['anchors']['single4'],base.zero['boxes']);params=[v for _,v in base.named]
 steps,lr=(40,.01)if ds=='hc2' else(10,.03)
 opt=torch.optim.Adam(params,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
 trace=[]
 try:
  for k in range(1 if lossfn.empty else steps+1):
   opt.zero_grad(set_to_none=True);values=base.values();loss=lossfn(values['boxes']);assert torch.isfinite(loss)
   trace.append(dict(step=k,loss=float(loss.detach())))
   if lossfn.empty or k==steps:break
   loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all()for v in params);opt.step()
  state=detached(base.state(),'cpu');cached=values['boxes'].detach().cpu()
  return dict(state=state,initial=detached(base.initial,'cpu'),cached_boxes=cached,trace=trace,steps=steps,gradient_calls=0 if lossfn.empty else steps,lr=lr,empty=lossfn.empty,episodic=True,LN_writeback=0,optimizer='Adam',objective='5*L1+2*(1-GIoU)',GT_read=False)
 finally:base.restore(initial)

def capture(which):
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for,frames_for,read_row
 from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from scripts.stvg_pipeline_speed_v1 import full_prefix_readout
 from vg_tta.stvg_joint_event_critic_v1 import temporal_candidates,extract_visual
 p=verify();records=p['target']if which=='target'else p['source']
 if which=='source':assert read(BASE/'CANDIDATE_HEADROOM.json')['continue_source_pilot']
 lease=gpu();model=None;domain=None;files=[];dino=None;cache=collections.OrderedDict();started=time.perf_counter()
 if which=='source':dino=SpatialExpert(ROOT/EXPERT_SNAPSHOT);eh=state_hash(dino.model.state_dict())
 try:
  for i,r in enumerate(records):
   verify();alias=r['alias'];dest=BASE/which/(alias+'.pt')
   if dest.exists():load(dest);files.append(read(dest.with_suffix('.receipt.json')));continue
   ds=r['dataset']
   if ds!=domain:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    domain=ds
    # Target uses the opposite-domain checkpoint; source trains the same critic
    # on that checkpoint's own training domain, not on target domain labels.
    source_ds=('vidstg'if ds=='hc2'else'hcstvg2') if which=='target' else ('vidstg'if ds=='vidstg'else'hcstvg2')
    model=model_for(source_ds);mh=state_hash(model.state_dict());cache.clear()
   row=read_row(ds,r['query_ordinal'])if which=='target'else dict(input=r['input'],frame_ids=r['frame_ids'],parses=r['parses'],key=r['key'],ordinal=i)
   frames,ids,_=frames_for(ds,row,cache);tick=time.perf_counter();batch,recs,base=frozen_forward(model,frames,row)
   native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids)
   if which=='target':
    prior=load_external(ROOT/'artifacts/stvg_dino_rollback32_v1/predictions'/(alias+'.pt'))
    source=load_external(ROOT/'artifacts/stvg_dino_rollback32_v1/source'/(alias+'.pt'))
    assert np.array_equal(source['frame_ids'],ids) and source['Native_interval']==native['physical_interval']
    nb=native['boxes'].double();xy=torch.cat((nb[:,:2]-nb[:,2:]/2,nb[:,:2]+nb[:,2:]/2),1).numpy()
    assert np.array_equal(xy,source['native_boxes_xyxy_normalized'])
    expert=prior['expert'];calls=0;expert_origin='exact restored Source/Native/input hash reuse'
    # Every planned observed pixel is checked against the stored expert receipt.
    for (_,pos),v in expert['observations'].items():assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==v['receipt']['rgb_sha256']
   else:expert=observations(dino,row['parses'],frames,ids,native['indices'],audit=True);calls=expert['new_DINO'];expert_origin='new_source_DINO'
   candidates=temporal_candidates(native['logits'],recs,ids,native['physical_interval'])
   adaptation_domain=ds if which=='target'else('vidstg'if ds=='hc2'else'hc2')
   fit=direct_fit(base,expert,adaptation_domain);full=full_prefix_readout(base,ids,recs,fit['state'])
   fit['adaptation_domain_config']=adaptation_domain
   assert all(torch.equal(base.state()[n],v)for n,v in base.initial.items())
   features=[extract_visual(base,b,ids,row['input']['fps'])for b in [native['boxes'],full['boxes']]]
   payload=dict(alias=alias,dataset=ds,split=r['split'],query_ordinal=r.get('query_ordinal'),parent=r['parent'],key=row['key'],input={k:v for k,v in row['input'].items()if k!='caption'},frame_ids=ids,native_interval=native['physical_interval'],candidates=candidates,boxes=[native['boxes'],full['boxes']],features=features,expert=detached(expert,'cpu'),expert_origin=expert_origin,fit=fit,complete_decoder_readout=detached(full,'cpu'),cached_vs_complete_max_box_difference=float((full['boxes']-fit['cached_boxes']).abs().max()),complete_Source_boxes_exact=True,model_hash=mh,compute=dict(capture_fit_feature_seconds=time.perf_counter()-tick,new_DINO_calls=calls),GT_read=False)
   files.append(save(dest,detached(payload,'cpu')));assert state_hash(model.state_dict())==mh
   status(which+'_candidates',i+1,len(records),GT_read=False);print(which,alias,i+1,len(records),flush=True)
   del frames,base,batch,fit,full,features,payload,native,expert;gc.collect();torch.cuda.empty_cache()
  if dino is not None:assert state_hash(dino.model.state_dict())==eh
  put(BASE/(which.upper()+'_PREDICTION_BARRIER.json'),dict(status='complete',count=len(files),files=files,GT_read=False,wall_seconds=time.perf_counter()-started))
 finally:lease.close()
def load_external(p):
 import torch
 assert sha(p)==read(p.with_suffix('.receipt.json'))['sha256'];return torch.load(p,map_location='cpu',weights_only=False)

def evaluate_payload(z,gt,span):
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
 row=dict(frame_ids=z['frame_ids'],input=z['input']);out=[]
 for boxes in z['boxes']:
  rr=[];ind=DenseTube(boxes.numpy(),row,gt,span,clip=z['dataset']=='hc2')
  for interval in z['candidates']['intervals']:
   v=official(boxes.numpy(),row,gt,span,interval,z['dataset']);v2=ind.score(interval)
   assert all(abs(v[k]-v2[k])<2e-10 for k in ['v','t','s']);rr.append(v)
  out.append(rr)
 return out

def target_labels(headroom=False):
 import numpy as np
 p=verify();bar=read(BASE/'TARGET_PREDICTION_BARRIER.json');assert bar['count']==32
 # All immutable candidate payload hashes checked before ANY target GT read.
 zs=[load(ROOT/r['path'])for r in bar['files']]
 from scripts.score_stvg_opd_p1_v1 import truths
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['target']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
 rows=[]
 for z in zs:
  dense,spans,_=gt[z['dataset']];vals=evaluate_payload(z,dense[z['query_ordinal']],spans[z['query_ordinal']]);v=np.array([[a['v']for a in row]for row in vals]);baseline=float(v[1,0]);oracle=float(v.max())
  rows.append(dict(alias=z['alias'],dataset=z['dataset'],values=vals,oracle_v=oracle,direct_native_v=baseline,headroom_v=oracle-baseline))
 if headroom:
  groups={d:dict(count=16,improvable=sum(r['headroom_v']>1e-8 for r in rows if r['dataset']==d),mean_headroom_pp=100*float(np.mean([r['headroom_v']for r in rows if r['dataset']==d])))for d in ['hc2','vidstg']}
  put(BASE/'CANDIDATE_HEADROOM.json',dict(status='complete',count=32,groups=groups,rows=rows,continue_source_pilot=all(g['improvable']>0 for g in groups.values()),GT_read=True,oracle_only_not_deployable=True,all_candidates_locked_before_GT=True))
 return rows

def source_labels():
 import ijson
 p=verify();b=read(BASE/'SOURCE_PREDICTION_BARRIER.json');assert b['count']==96
 zs={z['alias']:z for z in [load(ROOT/r['path'])for r in b['files']]};labels=[]
 for ds in ['hc2','vidstg']:
  m=p['source_provenance'][ds];assert sha(m['annotation'])==m['annotation_sha256'];rr=[r for r in p['source']if r['dataset']==ds];keys={r['annotation_key']for r in rr}
  with open(m['annotation'],'rb')as f:a={k:v for k,v in ijson.kvitems(f,'',use_float=True)if k in keys}
  assert len(a)==48
  for r in rr:
   v=a[r['annotation_key']];z=zs[r['alias']]
   if ds=='hc2':
    assert v['English'].lower()==r['input']['caption'].lower();s=int(v['st_frame'])-1;e=s+len(v['bbox']);dense={s+i:[x,y,min(x+w,r['input']['width']),min(y+h,r['input']['height'])]for i,(x,y,w,h)in enumerate(v['bbox'])}
   else:
    assert v['sentence']['description'].lower()==r['input']['caption'].lower();s=int(v['temp_gt']['begin_fid']);bb=v['target_bboxs'];e=s+len(bb);assert e-1==int(v['temp_gt']['end_fid'])
    dense={s+i:[b['xmin'],b['ymin'],b['xmax'],b['ymax']]for i,b in enumerate(bb)}
   labels.append(dict(alias=r['alias'],dataset=ds,split=r['split'],values=evaluate_payload(z,dense,[s,e]),source_GT=True))
 put(BASE/'SOURCE_LABELS.json',dict(status='complete',count=96,rows=labels,source_GT_only=True,target_GT_training=False,prediction_barrier_sha256=sha(BASE/'SOURCE_PREDICTION_BARRIER.json')))

def examples(z):
 from vg_tta.stvg_joint_event_critic_v1 import candidate_tensor
 return [candidate_tensor(f,I,z['input']['fps'])for f in z['features']for I in z['candidates']['intervals']]
def train():
 import torch,numpy as np
 from vg_tta.stvg_joint_event_critic_v1 import EventCritic,collate,ranking_loss,pair_accuracy
 torch.set_num_threads(4);p=verify();lab=read(BASE/'SOURCE_LABELS.json');assert lab['count']==96
 for ds in ['hc2','vidstg']:
  rows=[r for r in lab['rows']if r['dataset']==ds];data={r['alias']:examples(load(BASE/'source'/(r['alias']+'.pt')))for r in rows};ys={r['alias']:torch.tensor([a['v']for v in r['values']for a in v])for r in rows}
  tr=[r['alias']for r in rows if r['split']=='source_train'];va=[r['alias']for r in rows if r['split']=='source_validation'];assert len(tr)==36 and len(va)==12
  torch.manual_seed(p['critic']['seed']);np.random.seed(p['critic']['seed']);model=EventCritic();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  best=float('inf');selected=None;history=[];rng=np.random.default_rng(p['critic']['seed'])
  for epoch in range(80):
   verify();model.train();losses=[]
   order=list(rng.permutation(tr))
   for off in range(0,len(order),4):
    batch=order[off:off+4];x=collate([e for alias in batch for e in data[alias]]);scores=model(*x).reshape(len(batch),6);ls=[]
    for i,alias in enumerate(batch):ls.append(ranking_loss(scores[i],ys[alias])[0])
    loss=torch.stack(ls).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss))
   model.eval();vl=[];ac=[]
   with torch.no_grad():
    for alias in va:
     s=model(*collate(data[alias]));l,n=ranking_loss(s,ys[alias]);a,n=pair_accuracy(s.numpy(),ys[alias].numpy())
     if n:vl.append(float(l));ac.append((a,n))
   val=float(np.mean(vl))if vl else float('inf');acc=sum(a*n for a,n in ac)/sum(n for a,n in ac)if ac else None
   history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_pair_loss=val if np.isfinite(val)else None,validation_pair_accuracy=acc))
   if val<best:best=val;selected=dict(state={k:v.detach().clone()for k,v in model.state_dict().items()},epoch=epoch+1,validation_pair_accuracy=acc)
   status('source_critic_training_'+ds,epoch+1,80,target_GT_training=False)
  assert selected is not None,'No strict source validation pairs; cannot qualify critic'
  path=BASE/'critics'/(ds+'.pt');save(path,selected)
  put(BASE/'critics'/(ds+'.json'),dict(status='complete',source_dataset=ds,target_dataset='vidstg'if ds=='hc2'else'hc2',train=36,validation=12,parameters=sum(v.numel()for v in model.parameters()),selected_epoch=selected['epoch'],selected_validation_pair_accuracy=selected['validation_pair_accuracy'],history=history,source_GT_training=True,target_GT_training=False))
 put(BASE/'CRITIC_COMPLETION.json',dict(status='complete',count=2,source_domains=['hc2','vidstg'],target_GT_training=False))

def infer():
 import torch,numpy as np
 from vg_tta.stvg_joint_event_critic_v1 import EventCritic,collate
 torch.set_num_threads(4);p=verify();assert read(BASE/'CRITIC_COMPLETION.json')['count']==2
 models={};rows=[]
 for ds in ['hc2','vidstg']:
  sd='vidstg'if ds=='hc2'else'hc2';m=EventCritic();m.load_state_dict(load(BASE/'critics'/(sd+'.pt'))['state']);m.eval().requires_grad_(False);models[ds]=m
 for i,r in enumerate(p['target']):
  z=load(BASE/'target'/(r['alias']+'.pt'))
  with torch.no_grad():scores=models[z['dataset']](*collate(examples(z))).reshape(2,3).numpy()
  joint=list(np.unravel_index(scores.argmax(),scores.shape));temporal=int(scores[1].argmax())
  # Separate marginal maximizers can construct an incompatible joint choice.
  prob=torch.softmax(torch.tensor(scores).flatten(),0).reshape(2,3).numpy();ind=[int(prob.sum(1).argmax()),int(prob.sum(0).argmax())]
  rows.append(dict(alias=z['alias'],dataset=z['dataset'],scores=scores.tolist(),joint=[int(x)for x in joint],direct_conditional=[1,temporal],independent_marginals=ind,critic_checkpoint_sha256=sha(BASE/'critics'/(('vidstg'if z['dataset']=='hc2'else'hc2')+'.pt'))))
  status('frozen_target_critic',i+1,32,target_GT_read=False)
 put(BASE/'SELECTION_BARRIER.json',dict(status='complete',count=32,rows=rows,all_selection_sealed_before_target_evaluation=True,target_GT_read=False))

def score():
 import numpy as np
 from vg_tta.stvg_joint_event_critic_v1 import pair_accuracy
 from scipy.stats import t
 p=verify();b=read(BASE/'SELECTION_BARRIER.json');assert b['count']==32;labels=target_labels();sel={r['alias']:r for r in b['rows']};rows=[]
 for r in labels:
  z=sel[r['alias']];v=r['values'];scores=np.array(z['scores']);quality=np.array([[x['v']for x in row]for row in v]);a,n=pair_accuracy(scores.flatten(),quality.flatten())
  vals=dict(Frozen_native=v[0][0],Direct_native=v[1][0],Direct_conditional=v[1][z['direct_conditional'][1]],Joint=v[z['joint'][0]][z['joint'][1]],Independent_marginals=v[z['independent_marginals'][0]][z['independent_marginals'][1]],Oracle=v[np.unravel_index(quality.argmax(),quality.shape)[0]][np.unravel_index(quality.argmax(),quality.shape)[1]])
  payload=load(BASE/'target'/(r['alias']+'.pt'));from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
  import torch
  anchors=payload['expert']['anchors']['single4'];idx=[x['position']for x in anchors]
  agreement=[]
  if idx:
   refs=torch.tensor([x['box']for x in anchors]);agreement=[float(aligned_iou(box[idx],refs).mean())for box in payload['boxes']]
  bad=bool(agreement and agreement[1]>agreement[0]+1e-8 and v[1][0]['v']<v[0][0]['v']-1e-8)
  rows.append(dict(alias=r['alias'],dataset=r['dataset'],values=vals,all_hypothesis_metrics=v,critic_scores=z['scores'],joint=z['joint'],pair_accuracy=a,strict_pairs=n,DINO_agreement=agreement,DINO_better_vIoU_worse=bad,critic_rejects_bad_direct=bool(z['joint'][0]==0)if bad else None))
 def summary(rr):
  out={}
  for arm in rr[0]['values']:
   out[arm]={k:float(np.mean([r['values'][arm][k]for r in rr]))for k in ['v','t','s']}
  deltas={}
  for arm in ['Direct_conditional','Joint','Independent_marginals','Oracle']:
   x=np.array([r['values'][arm]['v']-r['values']['Direct_native']['v']for r in rr]);mu=float(x.mean());e=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
   deltas[arm]=dict(mean_pp=100*mu,t95_pp=[100*(mu-e),100*(mu+e)],positive=int((x>1e-8).sum()),negative=int((x< -1e-8).sum()),neutral=int((abs(x)<=1e-8).sum()),gross_gain_pp=float(np.maximum(x,0).mean()*100),gross_loss_pp=float(-np.minimum(x,0).mean()*100))
  n=sum(r['strict_pairs']for r in rr);ac=sum((r['pair_accuracy']or 0)*r['strict_pairs']for r in rr)/n if n else None
  bad=[r for r in rr if r['DINO_better_vIoU_worse']]
  return dict(count=len(rr),metrics=out,deltas_vs_Direct_native=deltas,strict_pair_accuracy=ac,strict_pairs=n,DINO_better_vIoU_worse_count=len(bad),critic_rejects_those=sum(r['critic_rejects_bad_direct']for r in bad))
 groups={d:summary([r for r in rows if r['dataset']==d])for d in ['hc2','vidstg']};source={d:read(BASE/'critics'/(d+'.json'))['selected_validation_pair_accuracy']for d in ['hc2','vidstg']}
 advance=all(g['strict_pair_accuracy']is not None and g['strict_pair_accuracy']>.5 and g['deltas_vs_Direct_native']['Joint']['mean_pp']>0 for g in groups.values())and all(x is not None and x>.5 for x in source.values())
 put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,groups=groups,rows=rows,source_validation_accuracy=source,advance_matched_OPD=advance,development_only=True,extra_source_supervision=True,no_target_GT_training=True))
 put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_read=True,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json')))
 put(BASE/'DECISION.json',dict(status='advance_matched_OPD'if advance else'stop_finite_pilot',scope='this fixed source-trained critic and exposed target32; not all conditional tube models',CURRENT_unchanged=True,no_automatic_full_stream=True,matched_OPD='required_next'if advance else'not_triggered'))
 status('finite_pilot_scored',32,32,GT_read=True)
 print(json.dumps(groups,indent=2),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','target','headroom','source','source_labels','train','infer','score']);x=a.parse_args().stage
 try:
  if x in ('source','target'):capture(x)
  elif x=='headroom':target_labels(True)
  else:globals()[x]()
 except BaseException as e:
  if BASE.exists():put(BASE/('FAILURE_'+x+'_'+str(time.time_ns())+'.json'),dict(status='failed',stage=x,error_type=type(e).__name__,error=str(e),partial_preserved=True))
  raise
