"""Fixed episodic temporal-LayerNorm entropy minimization, not original BN-Tent."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,hashlib,traceback,gc,csv,subprocess,math
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
from full import RUN,PROVIDER,read,sha,atomic,check,identity
D=R/'outputs/entropy_tta_v2';PUB=R/'results/entropy_tta_v2'
SEED=20261024

def validate():
 p=read(D/'plan.json');assert sha(Path(__file__))==p['code_sha256'];assert sha(D/'selection.json')==p['selection_sha256']
 assert hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()==p['fingerprint']
 check()
 for ds,h in p['seals'].items():assert sha(RUN/ds/'sealed.json')==h
 return p

def prepare():
 assert not (D/'plan.json').exists();check(True);p0=read(RUN/'plan.json');selection={}
 for ds,job in p0['jobs'].items():
  groups={}
  for row in map(json.loads,(PROVIDER/job['manifest']).read_text().splitlines()):groups.setdefault(str(row['metadata']['source']),[]).append(row)
  key=lambda s:hashlib.sha256(f'{SEED}:{ds}:{s}'.encode()).hexdigest()
  chosen=sorted(groups,key=key)[:128]
  selection[ds]=[min(groups[parent],key=lambda r:key(r['id']))for parent in chosen]
  assert len(selection[ds])==len({r['metadata']['source']for r in selection[ds]})==128
 atomic(D/'selection.json',selection)
 p=dict(schema='entropy_tta_v2',authorization='Latest user explicitly requests running real paired surrogate-optimization experiment; independent of frozen SimplePath and cancelled studies.',
  objective='Mean Shannon entropy of four native endpoint distributions (start/end x two offsets), natural logarithm, softmax along time. No labels or teacher.',
  optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=0,steps=5,batch='one query with all its native sampled frames/two offsets'),
  update_scope='Only existing nn.LayerNorm weight/bias in original ground_decoder.time_decoder; all other parameters frozen, dropout/every module remains eval; full reset of affine parameters and optimizer per query.',
  scope_note='Transformer-specific entropy-minimization baseline; NOT a reproduction of original BN-Tent, not a claim about all TTA implementations.',
  initialization='Exact original Source224 frozen inference; final second-pass native TimeDecoder inputs captured once, then reused while updating only downstream decoder LayerNorm. Spatial Source boxes fixed.',
  temporal_readout='Original final-layer FP32 two-offset legal-pair posterior, physical hull, original MBR; same readout before/after; no Global6 or Native2D or expert.',
  selection='128 parent videos/domain selected by SHA256(seed:dataset:parent), one query/parent by same fixed hash; no GT/quality-dependent selection; already exposed official datasets, not unseen confirmation.',
  cases_per_domain=128,total=256,selection_seed=SEED,inference_seed=20260920,
  outcome='Paired loss_before-loss_after and official complete vIoU_after-vIoU_before, all cases/last step. Primary P(delta_v<0|delta_loss>0), including ties in denominator. Numerical loss threshold1e-6 sensitivity. No favorable-outcome requirement or tuning.',
  controls='Actual frozen Source/logits/prior/interval/boxes bitwise sealed full; before/after gradient-context no-update checks; exact affine-state reset every query and post-reset native readout; only selected params can receive gradients.',
  budget=dict(maximum_wall_seconds=7200,new_expert=0,additional_backbone=0,new_observations=0,no_hyperparameter_sweep=True),
  phase_boundary='ALL256 predictions and loss paths seal before independent CPU labels; preserve every signed outcome, no online GT.',
  sources=['https://github.com/DequanWang/tent/blob/master/tent.py','https://github.com/DequanWang/tent/blob/master/conf.py','https://github.com/HengLan/TA-STVG/blob/904ad0c344eb12fe34513f97a33f93118f6d3be4/models/grounding_model/query_decoder.py'],
  preflight_v1='Prior exact same selection/protocol stopped before optimization on a cloned-input-stride no-op failure; preserve old fingerprint. V2 only restores original actual seven-input tensor layouts.',adaptation_difference='Tent official defaults Adam1e-3/one step and BN-affine; this predeclared experiment uses time-decoder LN-affine, FIVE steps, episodic reset, eval dropout.',
  code_sha256=sha(Path(__file__)),selection_sha256=sha(D/'selection.json'),production_plan_sha256=sha(RUN/'plan.json'),
  seals={ds:sha(RUN/ds/'sealed.json')for ds in selection},GT_in_inference=False,production_unchanged=True)
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(D/'plan.json',p);atomic(PUB/'protocol.json',p)
 dest=D/'source_snapshot/run.py';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(__file__).read_bytes())
 atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'],flush=True)

def restore_inputs(offset,device):
    import torch
    result={}
    for key,value in offset['inputs'].items():
        layout=offset['input_layouts'][key];shape=tuple(layout['shape']);stride=tuple(layout['stride']);start=int(layout.get('storage_offset',0))
        size=start+1+sum((s-1)*d for s,d in zip(shape,stride));back=torch.empty(size,device=device,dtype=value.dtype)
        collapsed=tuple(1 if d==0 else s for s,d in zip(shape,stride));view=torch.as_strided(back,collapsed,stride,start)
        source=value[tuple(slice(0,1)if d==0 else slice(None)for d in stride)].to(device)
        view.copy_(source);result[key]=torch.as_strided(back,shape,stride,start)
    return result


def entropy(logs):
 import torch
 return torch.stack([-(z[0].log_softmax(0).exp()*z[0].log_softmax(0)).sum(0).mean()for z in logs]).mean()

def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.native import frozen_native_forward
 from stvg.query import QuerySubjectParser
 from stvg.temporal_hidden import cache_temporal
 from stvg.temporal import probabilities,mass_table,final_mbr
 p=validate();assert not(D/'sealed.json').exists();setup();torch.set_num_threads(1);lock=lease();started=time.perf_counter();atomic(D/'predictor.json',identity());atomic(D/'state.json',dict(status='running',fingerprint=p['fingerprint'],**identity()));files={};selection=read(D/'selection.json');cfg=read(R/'configs/default.json');parser=QuerySubjectParser(R/'checkpoints/stanza');steps=p['optimizer']['steps']
 counts=dict(queries=0,Source_forwards=0,time_decoder_calls=0,optimizer_steps=0,new_expert=0)
 try:
  for ds,rows in selection.items():
   model=load_source(cfg['datasets'][ds]);dec=model.ground_decoder.time_decoder;head=model.temp_embed
   params=[];names=[]
   for name,module in dec.named_modules():
    if isinstance(module,torch.nn.LayerNorm):
     for nm,par in module.named_parameters(recurse=False):
      assert nm in ('weight','bias');params.append(par);names.append(name+'.'+nm)
   assert params and all(not par.requires_grad for par in model.parameters());initial=[par.detach().clone()for par in params]
   atomic(D/(ds+'_parameters.json'),dict(names=names,scalars=sum(x.numel()for x in params),frozen_others=True,model_eval=True))
   fullseal=read(RUN/ds/'sealed.json');files[ds]={}
   for i,row in enumerate(rows,1):
    tick=time.perf_counter();assert time.perf_counter()-started<p['budget']['maximum_wall_seconds'];key=row['id'];old=fullseal['files'][key];path=D/ds/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    assert sha(RUN/ds/old['capture_path'])==old['capture_sha256']and sha(RUN/ds/old['path'])==old['sha256']
    cap=torch.load(RUN/ds/old['capture_path'],map_location='cpu',weights_only=False)
    assert cap['GT_read']is False;frames,ids=decode(row['metadata'],ds);assert ids==list(cap['frame_ids'])
    torch.manual_seed(p['inference_seed']);np.random.seed(p['inference_seed'])
    subject=parser(row['metadata']['caption'])['subject']
    batch,recs,base=frozen_native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses=dict(subject=subject)));counts['Source_forwards']+=1
    assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source']),'Original Source boxes not bitwise'
    assert all(torch.equal(z.cpu(),zz)for z,zz in zip(base.zero['logits'],cap['native_logits'])),'Original Source logits not bitwise'
    U,q=mass_table(probabilities(recs,base.zero['logits']));iv0=final_mbr(U,q)['interval']
    assert np.array_equal(q,cap['native_prior'])and iv0==list(cap['native_interval']),'Native prior/interval mismatch'
    cache=cache_temporal(model,base,recs,ids);cf=D/ds/(key+'.native.pt');torch.save(cache,cf)
    kwargs=[restore_inputs(off,'cuda')for off in cache['offsets']];native_hidden=[x.clone()for x in base.temporal_inputs];native_logs=[x.clone()for x in base.zero['logits']]
    # All actual seven kwargs are retained, without reconstructing fused memory.
    del batch,base,frames;gc.collect();torch.cuda.empty_cache()
    with torch.no_grad():
     replay=[head(dec(**kw))[-1]for kw in kwargs];counts['time_decoder_calls']+=2
     assert all(torch.equal(z,z0)for z,z0 in zip(replay,native_logs)),'Native replay no-op failed'
    for par in params:par.requires_grad_(True)
    assert sum(x.requires_grad for x in model.parameters())==len(params)
    opt=torch.optim.Adam(params,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0);trace=[];after=None
    try:
     for step in range(steps+1):
      zz=[head(dec(**kw))[-1]for kw in kwargs];counts['time_decoder_calls']+=2;loss=entropy(zz);assert bool(torch.isfinite(loss))
      if step==0:
       err=max(float((z.detach()-z0).abs().max())for z,z0 in zip(zz,native_logs));assert err<1e-5,'Grad-mode no-op drift'
       assert final_mbr(U,mass_table(probabilities(recs,[z.detach()for z in zz]))[1])['interval']==iv0,'Grad-mode no-op readout drift'
      event=dict(step=step,loss=float(loss.detach()))
      if step<steps:
       opt.zero_grad(set_to_none=True);loss.backward();gn=math.sqrt(sum(float(par.grad.detach().square().sum())for par in params if par.grad is not None));assert np.isfinite(gn)and gn>0;event['gradient_norm']=gn;opt.step();counts['optimizer_steps']+=1
      else:after=[z.detach().cpu().clone()for z in zz]
      trace.append(event);del zz,loss
     assert all(x.grad is None for x in model.parameters()if not x.requires_grad)
     delta_norm=math.sqrt(sum(float((x.detach()-b).square().sum())for x,b in zip(params,initial)))
    finally:
     with torch.no_grad():
      for par,b in zip(params,initial):par.copy_(b);par.grad=None;par.requires_grad_(False)
     assert all(torch.equal(x,b)for x,b in zip(params,initial))
    with torch.no_grad():
     reset=[head(dec(**kw))[-1]for kw in kwargs];counts['time_decoder_calls']+=2
     assert all(torch.equal(z,z0)for z,z0 in zip(reset,native_logs)),'Reset not exact'
    _,qa=mass_table(probabilities(recs,after));iva=final_mbr(U,qa)['interval']
    with np.load(RUN/ds/old['path'],allow_pickle=False)as z:
     assert np.array_equal(z['Source'],cap['Source'])and list(z['Source_interval'])==iv0
     boxes=z['Source'].copy();pos=z['positions8'].copy()
    np.savez_compressed(path,Source=boxes,EntropyTTA=boxes,Source_interval=iv0,EntropyTTA_interval=iva,frame_ids=ids,positions8=pos,loss_before=trace[0]['loss'],loss_after=trace[-1]['loss'])
    tr=D/ds/(key+'.trace.pt');torch.save(dict(path=trace,logits_before=[z.cpu()for z in native_logs],logits_after=after,q_before=q,q_after=qa,updated_affine_L2=delta_norm,native_bitwise=True,reset_exact=True,grad_noop_max=err,GT_read=False),tr)
    files[ds][key]=dict(path=str(path.relative_to(D)),sha256=sha(path),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),native_path=str(cf.relative_to(D)),native_sha256=sha(cf),source_parent_SHA=old['sha256'],wall_seconds=time.perf_counter()-tick)
    counts['queries']+=1;atomic(D/'progress.json',dict(dataset=ds,completed=counts['queries'],total=256,domain_completed=i,domain_total=128,seconds=time.perf_counter()-started,counts=counts,**identity(),GT_read=False))
    print(json.dumps(dict(dataset=ds,completed=i,total=128,loss_before=trace[0]['loss'],loss_after=trace[-1]['loss'],seconds=time.perf_counter()-tick)),flush=True)
    del kwargs,native_hidden,native_logs,replay,reset,cap,cache,opt,qa,after;gc.collect();torch.cuda.empty_cache()
   assert not any(x.requires_grad or x.grad is not None for x in model.parameters());del model,dec,head,params,initial;gc.collect();torch.cuda.empty_cache()
  validate();assert counts['queries']==256 and counts['optimizer_steps']==1280
  atomic(D/'sealed.json',dict(status='all_predictions_sealed_before_labels',fingerprint=p['fingerprint'],files=files,counts=counts,GT_read=False,seconds=time.perf_counter()-started));atomic(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()

def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import numpy as np,torch
 from full_score import metric
 from scipy.stats import spearmanr
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 p=validate();seal=read(D/'sealed.json');assert seal['GT_read']is False and seal['counts']['queries']==256
 atomic(D/'scorer.json',identity());torch.set_num_threads(1);selection=read(D/'selection.json');jobs=read(RUN/'plan.json')['jobs'];allrows={};checks=0;maxerr=0
 for ds,rows in selection.items():
  f=PROVIDER/jobs[ds]['labels'];assert sha(f)==jobs[ds]['labels_sha256'];truth={r['id']:r for r in map(json.loads,f.read_text().splitlines())if r['dataset']==ds};saved={r['id']:r for r in read(RUN/ds/'metrics.json')['private_rows']};out=[]
  for r in rows:
   key=r['id'];e=seal['files'][ds][key];assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256'];g=truth[key];gt={int(k):v for k,v in g['boxes'].items()};vals={}
   with np.load(D/e['path'],allow_pickle=False)as z:
    for arm in ['Source','EntropyTTA']:
     vals[arm],err=metric(z[arm],r,gt,g['interval'],z[arm+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err)
    assert vals['Source']==saved[key]['values']['Source'],'Source official metrics drift'
    out.append(dict(query_id=key,parent=str(r['metadata']['source']),loss_before=float(z['loss_before']),loss_after=float(z['loss_after']),viou_before=vals['Source']['v'],viou_after=vals['EntropyTTA']['v'],tiou_before=vals['Source']['t'],tiou_after=vals['EntropyTTA']['t']))
  allrows[ds]=out;atomic(D/ds/'paired_private.json',out);public=PUB/ds;public.mkdir(parents=True,exist_ok=True)
  x=np.array([r['loss_before']-r['loss_after']for r in out]);y=100*np.array([r['viou_after']-r['viou_before']for r in out]);better=x>0;harm=y<0;n=int(better.sum());k=int((better&harm).sum());rng=np.random.default_rng(SEED);samples=rng.integers(0,len(out),size=(10000,len(out)));den=better[samples].sum(1);rates=np.divide((better&harm)[samples].sum(1),den,out=np.full(10000,np.nan),where=den>0)
  report=dict(dataset=ds,queries=len(out),parents=len({r['parent']for r in out}),source_mean_vIoU_percent=100*float(np.mean([r['viou_before']for r in out])),after_mean_vIoU_percent=100*float(np.mean([r['viou_after']for r in out])),mean_delta_vIoU_pp=float(y.mean()),mean_delta_tIoU_pp=100*float(np.mean([r['tiou_after']-r['tiou_before']for r in out])),surrogate_improves=n,grounding_degrades_given_surrogate_improves=k,conditional_rate=k/n if n else None,conditional_rate_CI95=np.nanquantile(rates,[.025,.975]).tolist(),loss_improves_gt1e6_negative_threshold=dict(denominator=int((x>1e-6).sum()),numerator=int(((x>1e-6)&harm).sum())),win=int((y>0).sum()),harm=int((y<0).sum()),tie=int((y==0).sum()),severe_v_drop_gt5pp=int((y < -5).sum()),spearman_delta_loss_vs_delta_v=float(spearmanr(x,y).statistic),selection=p['selection'],objective=p['objective'],update_scope=p['update_scope'],optimizer=p['optimizer'],no_outcome_selection=True)
  atomic(public/'report.json',report)
  with (public/'paired.csv').open('w')as f:
   w=csv.writer(f,lineterminator='\n');w.writerow(['query_id','loss_before','loss_after','viou_before','viou_after','delta_surrogate','delta_vIoU_pp'])
   for i,r in enumerate(out):w.writerow([f'Q{i+1:03d}',r['loss_before'],r['loss_after'],r['viou_before'],r['viou_after'],x[i],y[i]])
  plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
  fig,ax=plt.subplots(figsize=(7,5));xs=max(float(abs(x).max()),.01)*1.12;ys=max(float(abs(y).max()),1)*1.15
  ax.set(xlim=(-.1*xs,xs),ylim=(-ys,ys));ax.add_patch(Rectangle((0,-ys),xs,ys,facecolor='#F6E7E3',edgecolor='none',zorder=0));ax.axhline(0,color='#A3A7AA',lw=.8);ax.axvline(0,color='#A3A7AA',lw=.8)
  ax.scatter(x,y,c='#73777A',s=19,alpha=.65,linewidths=0,zorder=2)
  name='HC-STVG2.0'if ds=='hc2'else'VidSTG';ax.set_title('Does Optimizing a TTA Surrogate Improve STVG?',fontsize=12,pad=32);ax.text(.5,1.035,f'{name} · {len(out)} queries · Entropy minimization, temporal LN · 5 steps',ha='center',transform=ax.transAxes,fontsize=8.5,color='#555555')
  ax.set_xlabel(r'Surrogate loss reduction $\mathcal{L}_{\rm before}-\mathcal{L}_{\rm after}$ (nats)',fontsize=10);ax.set_ylabel(r'$\Delta$ vIoU (percentage points)',fontsize=10)
  for side in ['top','right']:ax.spines[side].set_visible(False)
  fig.text(.14,.045,f'Grounding degrades among surrogate-improving queries: {k}/{n} = {100*k/n:.1f}%',fontsize=9,color='#925644');fig.subplots_adjust(left=.14,right=.97,bottom=.18,top=.79)
  (D/'figures').mkdir(exist_ok=True)
  for ext in ['svg','pdf','png']:
   file=(public/'fig1.svg')if ext=='svg'else D/'figures'/f'fig1_{ds}.{ext}';fig.savefig(file,dpi=300,facecolor='white')
   if ext=='svg':file.write_text('\n'.join(s.rstrip()for s in file.read_text().splitlines())+'\n')
  plt.close(fig);print(json.dumps(report),flush=True)
 validate();atomic(PUB/'verification.json',dict(status='completed_scored',queries=256,official_dense_checks=checks,max_error=maxerr,Source_metrics_exact=True,ALL256_seal_before_labels=True,no_new_expert=True,production_unchanged=True,plan_fingerprint=p['fingerprint']));atomic(D/'score_receipt.json',dict(seal_sha256=sha(D/'sealed.json'),files={str(f.relative_to(R)):sha(f)for f in PUB.rglob('*')if f.is_file()}));atomic(D/'state.json',dict(status='completed_scored',fingerprint=p['fingerprint']))

def controller():
 p=validate();atomic(D/'controller.json',identity());children=[]
 for phase in ['predict','score']:
  with (D/(phase+'.log')).open('a')as log:
   proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),phase],cwd=R,stdout=log,stderr=subprocess.STDOUT);rec=dict(pid=proc.pid,startticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]),phase=phase);children.append(rec);atomic(D/'children.json',children);rc=proc.wait();atomic(D/(phase+'_exit.json'),dict(**rec,exit_code=rc,time=time.time()))
   if rc:raise RuntimeError(f'{phase} failed {rc}; preserve protocol/partials')
 atomic(D/'completion.json',dict(status='completed',fingerprint=p['fingerprint'],children=children))

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  if D.exists():atomic(D/f'failure_{time.time_ns()}.json',dict(error=traceback.format_exc(),identity=identity(),partial_preserved=True))
  raise
