"""Bounded source-only joint posterior pilot; production registries untouched."""
from pathlib import Path
import argparse,collections,gc,hashlib,json,math,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
OLD=ROOT/'artifacts/stvg_joint_event_critic_v1';BASE=ROOT/'artifacts/stvg_joint_posterior_v1'
KINDS=['joint_diffusion','joint_deterministic','separated_diffusion']
def status(stage,done,total,**kw):put(BASE/'STATUS.json',dict(status='running',stage=stage,done=done,total=total,pid=os.getpid(),**kw),True)
def verify():
 p=read(BASE/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch'],'finite deadline reached'
 for f,h in read(BASE/('RUNTIME_LOCK002.json'if(BASE/'RUNTIME_LOCK002.json').exists()else'RUNTIME_LOCK.json'))['pins'].items():assert sha(ROOT/f)==h,f
 from main_method.runtime import resource_guard
 resource_guard();return p

def prepare():
 import torch
 old=read(OLD/'PLAN_LOCK.json');assert not(BASE/'PLAN_LOCK.json').exists()
 plan=dict(status='locked_before_new_model',source=old['source'],target=old['target'],source_provenance=old['source_provenance'],
  source_train_per_domain=36,source_validation_per_domain=12,target_per_domain=16,target_status='previously exposed development, including prior oracle diagnosis; never independent test',
  source_binding002_sha256=sha(OLD/'SOURCE_LABEL_BINDING002.json'),source_payload_barrier_sha256=sha(OLD/'SOURCE_PREDICTION_BARRIER.json'),target_payload_barrier_sha256=sha(OLD/'TARGET_PREDICTION_BARRIER.json'),
  kinds=KINDS,representation='2 legal interval logits +32 event-relative normalized cxcywh logit knots; spatial loss only inside original annotated source support',
  conditioning='fixed preupdate frozen Student appearance/motion maps pooled8x8/resampled32 full-context times; dynamic4-point ROI; all Query tokens',
  association='binary source geometric inlier IoU>=.5 vs nonmatch, outside-event labels unknown; other-instance and no-match marginalized together; NOT calibrated identity classifier',
  prior_training=dict(seed=20261018,epochs=60,batch_size=6,lr=.0003,weight_decay=.0001,clip_gradient=1.,sigma_min=.03,sigma_max=2.,selection='minimum fixed-noise source validation loss only',joint_and_separated_same_noise=True),
  deterministic_training='same architecture/features/train steps; supervised Source/Direct corrupted-state correction; state-noise law differs from GT-centered diffusion DSM, disclosed control limitation',
  posterior=dict(samples=4,levels=16,sigma_max=2.,sigma_min=.03,observation_step=.02,component_step_cap=.2,aggregation='mean latent state of four fixed-seed posterior samples; no target selection'),
  transfer_gate='joint diffusion must exceed matched all-anchor episodic Direct mean vIoU AND mean tIoU in EACH target domain; no CI threshold, no model/config selection on target',
  adaptation_if_gate_passes='new fixed implementation: actual frozen Decoder outputs; spatial1792 plus temporal512 residual interface; source-only engineering qualification before target model updates; no output-to-parameter regression masquerading as score guidance',
  budgets=dict(new_source_captures=96,new_target_captures=32,new_DINO_calls=0,new_Qwen_calls=0,maximum_prior_models=6,maximum_association_models=2,maximum_target_parameter_fits_if_gate=32),
  CURRENT_unchanged=True,external_actionness_projection=False,backbone_internal_action_gates_retained=True,created_epoch=time.time(),deadline_epoch=time.time()+8*3600)
 put(BASE/'PLAN_LOCK.json',plan)
 own=['scripts/run_stvg_joint_posterior_v1.py','vg_tta/stvg_joint_posterior_v1.py','protocols/stvg_joint_posterior_v1.md']
 pins={f:sha(ROOT/f)for f in own}
 for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','vg_tta/stvg_opd_frozen_mask_cache012.py','methods/decota_final_simplified_v1/replay.py','scripts/run_decota_paper_main_v1.py']:pins[f]=sha(ROOT/f)
 put(BASE/'RUNTIME_LOCK.json',dict(status='locked',pins=pins,plan_sha256=sha(BASE/'PLAN_LOCK.json')))
 status('prepared',0,128,target_GT_training=False);print('PREPARED',flush=True)

def labels():
 """CPU source GT only; exact original source binding002, never target truth."""
 import zipfile,numpy as np,torch
 from scripts.prepare_vidstg_wrong_domain_support import trajectory_for_target
 from vg_tta.stvg_joint_posterior_v1 import logit,encode_bounds
 p=verify();binding=read(OLD/'SOURCE_LABEL_BINDING002.json');assert sha(OLD/'SOURCE_LABEL_BINDING002.json')==p['source_binding002_sha256']
 assert sha(ROOT/binding['raw_annotations'])==binding['raw_annotations_sha256']and sha(ROOT/binding['VidOR_annotations'])==binding['VidOR_sha256']
 ann=read(ROOT/binding['raw_annotations']);br={r['alias']:r for r in binding['rows']};hc=read(p['source_provenance']['hc2']['annotation'])
 assert sha(p['source_provenance']['hc2']['annotation'])==p['source_provenance']['hc2']['annotation_sha256']
 archive=zipfile.ZipFile(ROOT/binding['VidOR_annotations']);names={Path(n).stem:n for n in archive.namelist()if n.endswith('.json')};files=[];summary=[]
 for r in p['source']:
  z=load(OLD/'source'/(r['alias']+'.pt'));q=r['input'];ds=r['dataset']
  if ds=='hc2':
   v=hc[r['annotation_key']];assert v['English'].lower()==q['caption'].lower();a=int(v['st_frame'])-1;b=a+len(v['bbox'])-1
   gt={a+i:[x,y,min(x+w,q['width']),min(y+h,q['height'])]for i,(x,y,w,h)in enumerate(v['bbox'])}
  else:
   bb=br[r['alias']];v=ann[bb['original_annotation_index']];c=v[bb['field']][bb['query_index']]
   assert v['vid']==r['parent']and c['description'].strip().lower()==q['caption'].strip().lower()and int(c['target_id'])==bb['target_id']
   assert int(v['used_segment']['begin_fid'])==q['start_frame']and int(v['used_segment']['end_fid'])==q['end_frame']
   raw=json.loads(archive.read(names[r['parent']]));a,b=int(v['temporal_gt']['begin_fid']),int(v['temporal_gt']['end_fid'])
   gt={int(fid):[t['bbox'][0],t['bbox'][1],t['bbox'][0]+t['bbox'][2],t['bbox'][1]+t['bbox'][3]]for fid,t in trajectory_for_target(raw,bb['target_id']).items()if a<=int(fid)<b}
  assert gt and b>a
  ids=z['frame_ids'];lo,hi=ids[0],ids[-1]+1;aa=max(a,lo);bb=min(b,hi);assert bb>aa,(r['alias'],'source event outside model support')
  # Source-only interpolation between actually annotated event-interior frames.
  keys=np.asarray(sorted(gt));t=np.linspace(aa,min(bb-1,keys[-1]),32);valid=(t>=keys[0])&(t<=keys[-1])
  xy=np.stack([np.interp(t,keys,np.array([gt[int(k)]for k in keys])[:,j])for j in range(4)],-1)/np.array([q['width'],q['height'],q['width'],q['height']])
  boxes=np.concatenate(((xy[:,:2]+xy[:,2:])/2,xy[:,2:]-xy[:,:2]),-1)
  state=torch.zeros(33,4);state[0,:2]=encode_bounds(torch.tensor((aa-lo)/(hi-lo)),torch.tensor((bb-lo)/(hi-lo)));state[1:]=logit(torch.tensor(boxes,dtype=torch.float32))
  mask=torch.zeros(33,4);mask[0,:2]=1;mask[1:]=torch.tensor(valid)[:,None]
  associations=[]
  for anchor in z['expert']['anchors']['single4']:
   fid=ids[anchor['position']]
   if fid not in gt:continue
   from vg_tta.tastvg_oracle_event5_v1 import box_iou
   bx=np.array(anchor['box']);px=np.concatenate((bx[:2]-bx[2:]/2,bx[:2]+bx[2:]/2))*np.array([q['width'],q['height'],q['width'],q['height']]);iou=float(box_iou(px,np.array(gt[fid])))
   associations.append(dict(position=anchor['position'],label=float(iou>=.5),box=anchor['box'],GT_box_iou=iou))
  files.append(save(BASE/'source_supervision'/(r['alias']+'.pt'),dict(state=state,mask=mask,associations=associations,source_GT=True,split=r['split'],span=[a,b],dense_source_GT=gt,source_interpolation_only_inside_annotation=True)))
  summary.append(dict(alias=r['alias'],dataset=ds,split=r['split'],spatial_supervised_knots=int(valid.sum()),event_clipped_to_input=(aa!=a or bb!=b),association_labels=len(associations)))
 put(BASE/'SOURCE_SUPERVISION_BARRIER.json',dict(status='complete',count=96,files=files,rows=summary,source_GT_only=True,target_GT_training=False))
 print('SOURCE_LABELS',len(files),flush=True)

def capture():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for,frames_for,read_row
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import state_hash
 from torch.nn import functional as F
 p=verify();lease=gpu();model=None;current=None;cache=collections.OrderedDict();files=[]
 try:
  for i,(which,r)in enumerate([('source',r)for r in p['source']]+[('target',r)for r in p['target']]):
   verify();dest=BASE/'features'/which/(r['alias']+'.pt')
   if dest.exists():load(dest);files.append(read(dest.with_suffix('.receipt.json')));continue
   ds=r['dataset'];source=(('vidstg'if ds=='hc2'else'hcstvg2')if which=='target'else('vidstg'if ds=='vidstg'else'hcstvg2'))
   if source!=current:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    model=model_for(source);current=source;cache.clear()
   row=read_row(ds,r['query_ordinal'])if which=='target'else dict(input=r['input'],frame_ids=r['frame_ids'],parses=r['parses'],key=r['key'],ordinal=i)
   frames,ids,_=frames_for(ds,row,cache);batch,records,base=frozen_forward(model,frames,row)
   native=prediction(base.zero['logits'],base.zero['boxes'],records,ids);old=load(OLD/which/(r['alias']+'.pt'))
   assert torch.equal(native['boxes'].cpu(),old['boxes'][0])and native['physical_interval']==old['native_interval']
   seq=[None]*len(ids);query=[]
   with torch.no_grad():
    for off,view in enumerate(base.views):
     H=base.norm(view['prefix']).float();h,w=view['info']['fea_map_size'];n=h*w;nf=H.shape[1]
     app=H[:n].permute(1,2,0).reshape(nf,256,h,w);motion=H[-n:].permute(1,2,0).reshape(nf,256,h,w)
     mm=F.adaptive_avg_pool2d(torch.cat((app,motion),1),(8,8)).cpu()
     for j in range(nf):seq[off+2*j]=mm[j]
     query.append(H[n:-n].mean(1).cpu())
    assert query[0].shape==query[1].shape
    full=torch.stack(seq);physical=np.array(ids);grid=np.linspace(ids[0],ids[-1]+1,32);coords=np.interp(grid,physical,np.arange(len(ids)))
    a=np.floor(coords).astype(int);b=np.minimum(a+1,len(ids)-1);w=torch.tensor(coords-a,dtype=torch.float32)[:,None,None,None]
    maps=(full[a]*(1-w)+full[b]*w).permute(1,0,2,3).contiguous()
   payload=dict(maps=maps,query=(query[0]+query[1])/2,alias=r['alias'],dataset=ds,frame_ids=ids,input=old['input'],native_interval=old['native_interval'],boxes=old['boxes'],expert=old['expert'],model_hash=model._fixed_full_state_hash,old_payload_sha256=sha(OLD/which/(r['alias']+'.pt')),source_boxes_exact=True,GT_input=False,pooling='fixed8x8 average, 32 physical-context times linear frozen-feature interpolation')
   files.append(save(dest,payload));assert state_hash(model.state_dict())==model._fixed_full_state_hash
   status('frozen_feature_capture',i+1,128,target_GT_training=False);print('CAPTURE',which,r['alias'],i+1,flush=True)
   del frames,batch,base,full,maps,payload,old,native,seq;gc.collect();torch.cuda.empty_cache()
  put(BASE/'FEATURE_BARRIER.json',dict(status='complete',count=len(files),files=files,new_DINO_calls=0,new_Qwen_calls=0,target_GT_training=False))
 finally:lease.close()

def condition(z,device='cuda'):
 import torch,numpy as np
 from vg_tta.stvg_joint_posterior_v1 import encode_bounds,logit
 ids=np.array(z['frame_ids']);lo,hi=ids[0],ids[-1]+1;a,b=z['native_interval'];tt=np.linspace(a,min(b-1,ids[-1]),32)
 box=np.stack([np.interp(tt,ids,z['boxes'][0][:,j].numpy())for j in range(4)],-1)
 x=torch.zeros(33,4);x[0,:2]=encode_bounds(torch.tensor((a-lo)/(hi-lo)),torch.tensor((b-lo)/(hi-lo)));x[1:]=logit(torch.tensor(box,dtype=torch.float32))
 c=dict(unit=torch.tensor([1/(hi-lo)],device=device,dtype=torch.float32),maps=z['maps'][None].to(device),query=z['query'][None].to(device),qmask=torch.zeros(1,len(z['query']),dtype=torch.bool,device=device),context=z['maps'].mean((-1,-2)).T[None].to(device),native=x[None].to(device))
 return c

def batch_conditions(zs,device='cuda'):
 import torch
 cc=[condition(z,device)for z in zs];m=max(c['query'].shape[1]for c in cc);qs=[];pm=[]
 for c in cc:
  n=c['query'].shape[1];qs.append(torch.nn.functional.pad(c['query'],(0,0,0,m-n)));pm.append(torch.arange(m,device=device)[None]>=n)
 return {k:torch.cat([c[k]for c in cc])for k in ['unit','maps','context','native']}|dict(query=torch.cat(qs),qmask=torch.cat(pm))

def train():
 import torch,numpy as np
 from vg_tta.stvg_joint_posterior_v1 import JointPrior,Association,roi
 from scripts.run_decota_paper_main_v1 import gpu
 p=verify();assert read(BASE/'FEATURE_BARRIER.json')['count']==128;lease=gpu();rows=[];cfg=p['prior_training'];beg=time.perf_counter()
 try:
  for ds in ['hc2','vidstg']:
   rr=[r for r in p['source']if r['dataset']==ds];zs=[load(BASE/'features/source'/(r['alias']+'.pt'))for r in rr];ys=[load(BASE/'source_supervision'/(r['alias']+'.pt'))for r in rr]
   train_idx=[i for i,r in enumerate(rr)if r['split']=='source_train'];val_idx=[i for i,r in enumerate(rr)if r['split']=='source_validation'];assert len(train_idx)==36 and len(val_idx)==12
   # Prior and association training see source-domain labels only.
   av=[];ay=[];residual=[]
   for i in train_idx:
    z=zs[i];c=condition(z);lo,hi=z['frame_ids'][0],z['frame_ids'][-1]+1
    for a in ys[i]['associations']:
     t=torch.tensor([[(z['frame_ids'][a['position']]-lo)/(hi-lo)]],device='cuda');box=torch.tensor([[a['box']]],device='cuda');v=roi(c['maps'],box,t)[0,0]
     av.append(torch.cat((v,c['query'].mean(1)[0])));ay.append(a['label'])
     if a['label']:
      gt=ys[i]['dense_source_GT'][z['frame_ids'][a['position']]];x,y,xx,yy=gt;w,h=z['input']['width'],z['input']['height'];target=np.array([(x+xx)/(2*w),(y+yy)/(2*h),(xx-x)/w,(yy-y)/h]);residual.extend((np.array(a['box'])-target).tolist())
   assert av and sum(ay)>0 and sum(ay)<len(ay)
   if not(BASE/'models'/(ds+'_association.pt')).exists():
    torch.manual_seed(cfg['seed']);assoc=Association().cuda();opt=torch.optim.AdamW(assoc.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);xx=torch.stack(av);yy=torch.tensor(ay,device='cuda')
    for k in range(120):
     opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(assoc.net(xx).squeeze(-1),yy);loss.backward();opt.step()
    variance=max(float(np.mean(np.square(residual))),.0025)
    assoc_file=save(BASE/'models'/(ds+'_association.pt'),dict(state=assoc.cpu().state_dict(),variance=variance,count=len(ay),inliers=int(sum(ay)),source_only=True,parameter_count=sum(v.numel()for v in assoc.parameters())))
    del assoc,opt,xx,yy,av
   else:
    load(BASE/'models'/(ds+'_association.pt'));del av
   for kind in KINDS:
    verify();dest=BASE/'models'/(ds+'_'+kind+'.pt')
    if dest.exists():
     saved=load(dest);rows.append({k:v for k,v in saved.items()if k!='state'});continue
    torch.manual_seed(cfg['seed']);net=JointPrior(kind=='separated_diffusion').cuda();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);history=[];best=float('inf');beststate=None;updates=0
    rng=np.random.default_rng(cfg['seed']);valgen=torch.Generator(device='cuda').manual_seed(cfg['seed']+99)
    vn=torch.randn((12,33,4),generator=valgen,device='cuda');vs=torch.logspace(math.log10(.03),math.log10(2.),12,device='cuda')
    vc=batch_conditions([zs[i]for i in val_idx]);vy=torch.stack([ys[i]['state']for i in val_idx]).cuda();vm=torch.stack([ys[i]['mask']for i in val_idx]).cuda()
    for epoch in range(cfg['epochs']):
     net.train();order=rng.permutation(train_idx);train_loss=[]
     for start in range(0,36,6):
      idx=order[start:start+6];c=batch_conditions([zs[i]for i in idx]);truth=torch.stack([ys[i]['state']for i in idx]).cuda();mask=torch.stack([ys[i]['mask']for i in idx]).cuda()
      sig=torch.exp(torch.empty(len(idx),device='cuda').uniform_(math.log(.03),math.log(2.)));noise=torch.randn_like(truth)
      if kind=='joint_deterministic':
       which=rng.integers(2,size=len(idx));initial=[]
       for i,w in zip(idx,which):
        zz=dict(zs[i]);zz['boxes']=[zs[i]['boxes'][int(w)]];initial.append(condition(zz)['native'][0])
       corrupted=torch.stack(initial)+sig[:,None,None]*noise
      else:corrupted=truth+sig[:,None,None]*noise
      corrupted[:,0,2:]=0;opt.zero_grad();pred=net(corrupted,sig,c)
      # Normalize temporal and spatial blocks equally, no invented exterior GT.
      err=(pred-truth).square()*mask;loss=(err[:,0,:2].mean()+err[:,1:].sum()/mask[:,1:].sum().clamp_min(1))/2
      assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();updates+=1;train_loss.append(float(loss.detach()))
     net.eval()
     with torch.no_grad():
      vx=(vc['native']if kind=='joint_deterministic'else vy)+vs[:,None,None]*vn;vx[:,0,2:]=0;vpred=net(vx,vs,vc);ve=(vpred-vy).square()*vm;vl=float((ve[:,0,:2].mean()+ve[:,1:].sum()/vm[:,1:].sum())/2)
     history.append(dict(epoch=epoch+1,train_loss=float(np.mean(train_loss)),source_validation_loss=vl))
     if vl<best:best=vl;beststate={k:v.detach().cpu().clone()for k,v in net.state_dict().items()};bestepoch=epoch+1
     status('source_prior_training',len(rows),6,dataset=ds,model=kind,epoch=epoch+1,target_GT_training=False)
    result=dict(state=beststate,selected_epoch=bestepoch,source_validation_loss=best,history=history,optimizer_steps=updates,parameter_count=sum(v.numel()for v in net.parameters()),source_domain=ds,kind=kind,target_GT_training=False)
    save(dest,result);rows.append({k:v for k,v in result.items()if k!='state'});print('TRAIN',ds,kind,bestepoch,best,flush=True)
    del net,opt,vc,vy,vm;gc.collect();torch.cuda.empty_cache()
   del zs,ys;gc.collect()
  put(BASE/'TRAINING_COMPLETION.json',dict(status='complete',count=6,rows=rows,source_train_count=72,source_validation_count=24,target_GT_training=False,seconds=time.perf_counter()-beg))
 finally:lease.close()

def decode_state(x,z):
 import numpy as np,torch
 from vg_tta.stvg_joint_posterior_v1 import bounds
 a,b=bounds(x[None])[0].detach().cpu().numpy();ids=np.array(z['frame_ids']);lo,hi=ids[0],ids[-1]+1;a=lo+a*(hi-lo);b=lo+b*(hi-lo);iv=[max(lo,min(int(round(a)),hi-2)),max(lo+1,min(int(round(b)),hi))];iv[1]=max(iv[1],iv[0]+1)
 tt=np.linspace(a,max(a+1e-4,b-1),32);bb=x[1:].sigmoid().detach().cpu().numpy();out=z['boxes'][0].numpy().copy();mask=(ids>=iv[0])&(ids<iv[1])
 for j in range(4):out[mask,j]=np.interp(ids[mask],tt,bb[:,j])
 return dict(boxes=out,interval=iv,event_exterior_output='unchanged Source boxes; no learned exterior trajectory claim')

def infer():
 import torch,numpy as np
 from vg_tta.stvg_joint_posterior_v1 import JointPrior,Association,roi,posterior,observation_loglik
 from scripts.run_decota_paper_main_v1 import gpu
 p=verify();assert read(BASE/'TRAINING_COMPLETION.json')['count']==6;lease=gpu();files=[];models={};start=time.perf_counter()
 try:
  for n,r in enumerate(p['target']):
   verify();z=load(BASE/'features/target'/(r['alias']+'.pt'));ds='vidstg'if r['dataset']=='hc2'else'hc2';c=condition(z);lo,hi=z['frame_ids'][0],z['frame_ids'][-1]+1;anchors=z['expert']['anchors']['single4']
   obs=dict(time=torch.tensor([(z['frame_ids'][a['position']]-lo)/(hi-lo)for a in anchors],device='cuda'),boxes=torch.tensor([a['box']for a in anchors],device='cuda').reshape(-1,4))
   ad=load(BASE/'models'/(ds+'_association.pt'));assoc=Association().cuda();assoc.load_state_dict(ad['state']);assoc.eval().requires_grad_(False)
   with torch.no_grad():
    vis=roi(c['maps'],obs['boxes'][None],obs['time'][None])[0]if anchors else torch.zeros(0,512,device='cuda');pp=assoc(vis,c['query'].mean(1).expand(len(anchors),-1)).sigmoid()
   arms=dict(Source=dict(boxes=z['boxes'][0].numpy(),interval=z['native_interval']),Direct=dict(boxes=z['boxes'][1].numpy(),interval=z['native_interval']));traces={};jac={}
   for kind in KINDS:
    if (ds,kind)not in models:
     net=JointPrior(kind=='separated_diffusion').cuda();net.load_state_dict(load(BASE/'models'/(ds+'_'+kind+'.pt'))['state']);models[ds,kind]=net.eval().requires_grad_(False)
    net=models[ds,kind];cc={k:v.expand((4,)+v.shape[1:])for k,v in c.items()}
    x,trace=posterior(net,cc,obs,pp,ad['variance'],'deterministic'if kind=='joint_deterministic'else'diffusion',20261018+n)
    arms[kind]=decode_state(x.mean(0),z);traces[kind]=trace
    # Evidence that implemented state coupling is real; not proof of benefit.
    xx=c['native'].clone().requires_grad_(True);pred=net(xx,torch.tensor([.3],device='cuda'),c);g=torch.autograd.grad(pred[0,0,0],xx,retain_graph=True)[0]
    h=torch.autograd.grad(pred[0,1:,0].sum(),xx)[0];jac[kind]=dict(time_output_spatial_state_gradient=float(g[:,1:].norm()),spatial_output_time_state_gradient=float(h[:,0,:2].norm()))
   files.append(save(BASE/'predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],arms=arms,traces=traces,association_prior=pp.detach().cpu(),association_variance=ad['variance'],coupling_gradient=jac,target_GT_input=False,new_Teacher_calls=0,source_model_domain=ds)))
   status('target_posterior_inference',n+1,32,target_GT_training=False);print('INFER',r['alias'],flush=True)
  put(BASE/'PREDICTION_BARRIER.json',dict(status='complete',count=32,arms=5,files=files,target_GT_training=False,new_DINO_calls=0,seconds=time.perf_counter()-start))
 finally:lease.close()

def score():
 import numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
 from scipy.stats import t as student_t
 p=verify();bar=read(BASE/'PREDICTION_BARRIER.json');assert bar['count']==32
 target={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['target']if r['dataset']==ds]))for ds in ['hc2','vidstg']};rows=[]
 for r in p['target']:
  z=load(BASE/'features/target'/(r['alias']+'.pt'));pr=load(BASE/'predictions'/(r['alias']+'.pt'));gt=target[r['dataset']][0][r['query_ordinal']];span=target[r['dataset']][1][r['query_ordinal']];row=dict(frame_ids=z['frame_ids'],input=z['input']);metrics={}
  for arm,a in pr['arms'].items():
   m=official(a['boxes'],row,gt,span,a['interval'],r['dataset']);d=DenseTube(a['boxes'],row,gt,span,clip=r['dataset']=='hc2');ind=d.score(a['interval']);assert max(abs(m[k]-ind[k])for k in m)<2e-9
   observed={z['frame_ids'][v['position']]for v in z['expert']['anchors']['single4']};native=set(z['frame_ids']);inside=(d.fids>=max(a['interval'][0],span[0]))&(d.fids<min(a['interval'][1],span[1]));native_mask=np.array([i in native for i in d.fids]);obs_mask=np.array([i in observed for i in d.fids])
   m.update(R03=float(m['v']>=.3),R05=float(m['v']>=.5),fixed_Source_time_v=d.score(z['native_interval'])['v'],observed_s=float(d.iou[obs_mask].mean())if obs_mask.any()else None,unobserved_native_s=float(d.iou[native_mask&~obs_mask].mean())if(native_mask&~obs_mask).any()else None)
   metrics[arm]=m
  rows.append(dict(alias=r['alias'],dataset=r['dataset'],metrics=metrics,association_count=len(pr['association_prior']),association_prior_mean=float(pr['association_prior'].mean())if len(pr['association_prior'])else None,coupling_gradient=pr['coupling_gradient']))
 means={};contrasts={};passes=[]
 pairs=[('joint_diffusion','Direct'),('joint_deterministic','Direct'),('separated_diffusion','Direct'),('joint_diffusion','joint_deterministic'),('joint_diffusion','separated_diffusion')]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];means[ds]={a:{k:float(np.mean([r['metrics'][a][k]for r in rr]))for k in ['v','t','s','R03','R05','fixed_Source_time_v']}for a in rows[0]['metrics']};contrasts[ds]={}
  for a,b in pairs:
   vals={}
   for k in ['v','t','s','fixed_Source_time_v']:
    d=np.array([r['metrics'][a][k]-r['metrics'][b][k]for r in rr]);se=float(d.std(ddof=1)/np.sqrt(len(d)));radius=float(student_t.ppf(.975,len(d)-1))*se;vals[k]=dict(mean=float(d.mean()),CI95=[float(d.mean()-radius),float(d.mean()+radius)],positive=int((d>1e-10).sum()),negative=int((d< -1e-10).sum()),neutral=int((abs(d)<=1e-10).sum()),values=d.tolist())
   contrasts[ds][a+' - '+b]=vals
  passes.append(all(means[ds]['joint_diffusion'][k]>means[ds]['Direct'][k]for k in ['v','t']))
 out=dict(status='complete',count=32,rows=rows,means=means,paired=contrasts,target_GT_used_only_after_all_predictions_sealed=True,gate_pass=all(passes),uncertainty='paired source-level t intervals; exposed development; no independent confirmation')
 put(BASE/'POSTSEAL_RESULTS.json',out);put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_read=True,prediction_barrier_sha256=sha(BASE/'PREDICTION_BARRIER.json')))
 put(BASE/'DECISION.json',dict(status='continue_parameter_adapter_qualification'if all(passes)else'stop_before_parameter_TTA',scope='finite joint posterior source96-target32',cross_domain_gate=all(passes),reason='prelocked per-domain vIoU AND tIoU versus all-anchor episodic Direct',remaining_third_control='conditional actual parameter adaptation, not executed if prerequisite fails',CURRENT_unchanged=True))
 print(json.dumps(dict(means=means,gate=all(passes)),indent=2),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','labels','capture','train','infer','score']);a=ap.parse_args();globals()[a.stage]()
