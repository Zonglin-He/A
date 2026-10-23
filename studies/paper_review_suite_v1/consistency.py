"""Matched episodic temporal LN entropy / real horizontal-flip consistency TTA.

Actual clean/flip native final-pass contexts; this is not canonical Tent/CoTTA.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,hashlib,argparse,math,gc,subprocess,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
from full import RUN,PROVIDER,read,sha,atomic,check,identity
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1';ARMS=['Entropy','FlipConsistency']

def validate():
 p=read(D/'consistency_plan.json');assert sha(Path(__file__))==p['code_sha256']and sha(D/'selection.json')==p['selection_sha256'];assert sha(R/'studies/entropy_tta_v2/run.py')==p['input_restore_sha256'];check();return p

def js(clean,aug):
 import torch
 terms=[]
 for a,b in zip(clean,aug):
  aa=a[0].log_softmax(0);bb=b[0].log_softmax(0);middle=torch.logaddexp(aa,bb)-math.log(2);terms.append(.5*((aa.exp()*(aa-middle)).sum(0)+(bb.exp()*(bb-middle)).sum(0)).mean())
 return torch.stack(terms).mean()

def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.native import frozen_native_forward
 from stvg.query import QuerySubjectParser
 from stvg.temporal_hidden import cache_temporal
 from stvg.temporal import mass_table,probabilities,final_mbr
 from studies.entropy_tta_v2.run import restore_inputs,entropy
 p=validate();assert(D/'supports_sealed.json').exists();setup();gpu=lease();atomic(D/'consistency_predictor.json',identity());cfg=read(R/'configs/default.json');selection=read(D/'selection.json');support=read(D/'supports_sealed.json');parser=QuerySubjectParser(R/'checkpoints/stanza');files={};counts=dict(queries=0,flip_Source_forwards=0,steps=0,time_decoder_calls=0,new_expert=0);started=time.perf_counter()
 try:
  for ds,groups in selection.items():
   model=load_source(cfg['datasets'][ds]);dec=model.ground_decoder.time_decoder;head=model.temp_embed;params=[];names=[]
   for name,module in dec.named_modules():
    if isinstance(module,torch.nn.LayerNorm):
     for n,par in module.named_parameters(recurse=False):params.append(par);names.append(name+'.'+n)
   initial=[par.detach().clone()for par in params];files[ds]={};atomic(D/(ds+'_consistency_parameters.json'),dict(names=names,scalars=sum(x.numel()for x in params)))
   for stage,rows in groups.items():
    files[ds][stage]={}
    for i,row in enumerate(rows,1):
     assert time.perf_counter()-started<14400;key=row['id'];dest=D/'consistency'/ds;dest.mkdir(parents=True,exist_ok=True);target=dest/(key+'.npz');assert not target.exists();e=support['files'][ds][stage][key];assert sha(D/e['shared_path'])==e['shared_sha256'];shared=torch.load(D/e['shared_path'],map_location='cpu',weights_only=False);clean=shared['temporal'];cap=torch.load(shared['parent_capture'],map_location='cpu',weights_only=False);assert sha(shared['parent_capture'])==shared['parent_capture_sha256'];frames,ids=decode(row['metadata'],ds);assert ids==list(cap['frame_ids']);torch.manual_seed(20260920);np.random.seed(20260920);model.cuda();flip=np.ascontiguousarray(frames[:,:,::-1]);batch,recs,base=frozen_native_forward(model,flip,dict(input=row['metadata'],frame_ids=ids,parses=dict(subject=parser(row['metadata']['caption'])['subject'])));aug=cache_temporal(model,base,recs,ids);counts['flip_Source_forwards']+=1;nativeclean=[off['native_logits']for off in clean['offsets']];nativeflip=[off['native_logits']for off in aug['offsets']];assert all(torch.equal(z,zz)for z,zz in zip(nativeclean,cap['native_logits']));cf=dest/(key+'.flip.pt');torch.save(aug,cf);del batch,base,frames,flip;gc.collect();torch.cuda.empty_cache();ck=[restore_inputs(off,'cuda')for off in clean['offsets']];ak=[restore_inputs(off,'cuda')for off in aug['offsets']]
     with torch.no_grad():
      c0=[head(dec(**kw))[-1]for kw in ck];a0=[head(dec(**kw))[-1]for kw in ak];counts['time_decoder_calls']+=4;assert all(torch.equal(x.cpu(),y)for x,y in zip(c0,nativeclean));assert all(torch.equal(x.cpu(),y)for x,y in zip(a0,nativeflip))
     arrays={};traces={};q0=cap['native_prior']
     for arm in ARMS:
      for par in params:par.requires_grad_(True)
      opt=torch.optim.Adam(params,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0);trace=[]
      try:
       for step in range(6):
        zz=[head(dec(**kw))[-1]for kw in ck];counts['time_decoder_calls']+=2
        if step==0:assert max(float((x.detach().cpu()-y).abs().max())for x,y in zip(zz,nativeclean))<1e-5
        if arm=='FlipConsistency':aa=[head(dec(**kw))[-1]for kw in ak];loss=js(zz,aa);counts['time_decoder_calls']+=2
        else:loss=entropy(zz)
        assert torch.isfinite(loss);event=dict(step=step,loss=float(loss.detach()))
        if step<5:
         opt.zero_grad(set_to_none=True);loss.backward();gn=math.sqrt(sum(float(par.grad.square().sum())for par in params if par.grad is not None));assert math.isfinite(gn);event['gradient_norm']=gn;opt.step();counts['steps']+=1
        else:after=[z.detach().cpu().clone()for z in zz]
        trace.append(event);del zz,loss
        if arm=='FlipConsistency':del aa
       _,q=mass_table(probabilities(recs,after));arrays[arm+'_interval']=np.asarray(final_mbr(ids,q)['interval']);traces[arm]=trace
      finally:
       with torch.no_grad():
        for par,value in zip(params,initial):par.copy_(value);par.grad=None;par.requires_grad_(False)
      with torch.no_grad():
       reset=[head(dec(**kw))[-1]for kw in ck];counts['time_decoder_calls']+=2;assert all(torch.equal(x.cpu(),y)for x,y in zip(reset,nativeclean))
      assert all(torch.equal(x,y)for x,y in zip(params,initial));del opt
     arrays.update(Source=np.asarray(cap['Source']),Source_interval=np.asarray(cap['native_interval']),positions8=np.asarray(cap['positions8']),frame_ids=np.asarray(ids));np.savez_compressed(target,**arrays);tr=dest/(key+'.trace.json');atomic(tr,dict(traces=traces,native_bitwise=True,episodic_reset_exact=True,GT_read=False));files[ds][stage][key]=dict(path=str(target.relative_to(D)),sha256=sha(target),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),flip_path=str(cf.relative_to(D)),flip_sha256=sha(cf));counts['queries']+=1;atomic(D/'consistency_progress.json',dict(dataset=ds,stage=stage,domain_completed=i,counts=counts,seconds=time.perf_counter()-started,GT_read=False));print(json.dumps(dict(dataset=ds,stage=stage,completed=i,counts=counts)),flush=True);del shared,clean,aug,cap,ck,ak,c0,a0,reset,after;gc.collect();torch.cuda.empty_cache()
   assert not any(x.requires_grad or x.grad is not None for x in model.parameters());del model,params,initial;gc.collect();torch.cuda.empty_cache()
  assert counts['queries']==384;atomic(D/'consistency_sealed.json',dict(files=files,counts=counts,GT_read=False,fingerprint=p['fingerprint'],seconds=time.perf_counter()-started))
 finally:gpu.close()

def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import numpy as np,torch
 from full_score import metric,cluster
 torch.set_num_threads(1);p=validate();seal=read(D/'consistency_sealed.json');assert seal['counts']['queries']==384 and not seal['GT_read'];checks=0;maxerr=0.
 for ds,groups in read(D/'selection.json').items():
  job=read(RUN/'plan.json')['jobs'][ds];f=PROVIDER/job['labels'];assert sha(f)==job['labels_sha256'];truth={x['id']:x for x in map(json.loads,f.read_text().splitlines())if x['dataset']==ds};old={x['id']:x for x in read(RUN/ds/'metrics.json')['private_rows']}
  for stage,records in groups.items():
   rows=[];losschanges={a:[]for a in ARMS}
   for r in records:
    key=r['id'];g=truth[key];e=seal['files'][ds][stage][key];assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256'];trace=read(D/e['trace_path'])['traces'];vals={a:old[key]['values'][a]for a in ['Source','Full','Interpolation']}
    with np.load(D/e['path'])as z:
     for a in ARMS:vals[a],err=metric(z['Source'],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err);losschanges[a].append(trace[a][0]['loss']-trace[a][-1]['loss'])
    rows.append(dict(id=key,parent=str(g['parent']),values=vals))
   arms=list(rows[0]['values']);fields=['v','t'];pairs=[('Entropy','Source'),('FlipConsistency','Source'),('Full','Entropy'),('Full','FlipConsistency')];report=dict(dataset=ds,stage=stage,queries=len(rows),means={a:{f:float(np.mean([x['values'][a][f]for x in rows]))for f in fields}for a in arms},paired={a+'-'+b:cluster(rows,a,b)for a,b in pairs},objective_improved={a:int((np.asarray(v)>0).sum())for a,v in losschanges.items()},fingerprint=p['fingerprint'],scope=p['description'],not_canonical_Tent_or_CoTTA=True);pub=P/'consistency'/ds/stage;atomic(pub/'report.json',report);atomic(D/'consistency'/ds/(stage+'_metrics.json'),dict(report,private_rows=rows))
   with(pub/'signed.tsv').open('w')as f:
    w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query']+[a+'_'+k for a in arms for k in fields]);w.writerows([[f'Q{i:03d}']+[x['values'][a][k]for a in arms for k in fields]for i,x in enumerate(rows,1)])
 atomic(D/'consistency_score_receipt.json',dict(status='scored',checks=checks,max_error=maxerr,all384sealed_before_GT=True));print('Consistency scored',flush=True)

def controller():
 p=validate();atomic(D/'consistency_controller.json',identity());atomic(D/'consistency_state.json',dict(status='waiting_shared_grid_owner',controller=identity(),fingerprint=p['fingerprint']))
 while not(D/'supports_sealed.json').exists():
  who=read(D/'supports_launch.json')
  try:st=Path(f"/proc/{who['pid']}/stat").read_text().rsplit(')',1)[1].split();assert int(st[19])==who['startticks']and st[0]!='Z'
  except (OSError,AssertionError):raise RuntimeError('Shared grid owner failed; no model starts')
  time.sleep(10)
 # Wait until actual current GPU lease owner exits, not merely until a file appears.
 while True:
  import fcntl
  lock=(R/'runtime/gpu.lock').open('a')
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close();break
  except BlockingIOError:lock.close();time.sleep(5)
 for command in ['predict','score']:
  atomic(D/'consistency_state.json',dict(status=command,controller=identity(),fingerprint=p['fingerprint']));c=subprocess.Popen([sys.executable,'-B',__file__,command]);atomic(D/('consistency_'+command+'_child.json'),identity(c.pid));assert c.wait()==0,'Preserve failed consistency stage, no auto repeat'
 atomic(D/'consistency_state.json',dict(status='complete_scored_pending_publication',controller=identity(),fingerprint=p['fingerprint']))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['run','predict','score']);args=a.parse_args();{'run':controller,'predict':predict,'score':score}[args.command]()
