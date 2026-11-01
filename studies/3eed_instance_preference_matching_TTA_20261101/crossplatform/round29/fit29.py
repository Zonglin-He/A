"""All accepted runtime pseudo preferences; no task/identity labels in optimizer."""
import sys,time,copy,numpy as np,collections
from common29 import *
sys.path.insert(0,str(ROOT/'crossplatform/round24'))
import fit24
import torch
fit24.feature=feature
from fit24 import module,batch,scores,interact

def logodds_scores(m,b):
 q=F.normalize(m(b['h']),dim=-1);v=q@b['tokens'].transpose(-1,-2)/.07;pos=b['mask']&~b['padding'];neg=~b['mask']&~b['padding'];return torch.logsumexp(v.masked_fill(~pos[:,None],-torch.inf),-1)-torch.logsumexp(v.masked_fill(~neg[:,None],-torch.inf),-1)
F=torch.nn.functional

def group_readout(r,s):
 if r['stage']=='R':return int(np.argmax(s))
 z=feature(r);g=read(BASE/'groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'];used=set();options=[]
 for inds in g.values():
  used.update(inds);rep=max(inds,key=lambda i:(float(z['scores'][i]),-i));options.append((float(np.max(s[inds])),-rep,rep))
 for i in range(256):
  if i not in used:options.append((float(s[i]),-i,i))
 return int(max(options)[2])

def write(out,rr,preds):
 out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir()
 for r in rr:
  z=feature(r);p=preds[r['id']];q=np.array([p[n]['q'] for n in p]);np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(list(p)),queries=q,boxes=z['boxes'][q],**{n+'_scores':p[n]['scores'] for n in p})
 seal(out,'R29_frozen_box_predictions_before_A_B_labels',n=len(rr))

def main():
 verify(BASE/'sourceR64_attempt001');verify(BASE/'direct208_attempt001');use_logodds='--logodds' in sys.argv;out=BASE/('ranknet_logodds_attempt001' if use_logodds else 'ranknet_attempt001');out.mkdir(exist_ok=False);(out/'states_PRIVATE').mkdir();torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.perf_counter();rr=rows()[:208];bb=read(EXP.parent/'input/R64_PRIVATE.json');feedback=read(BASE/'direct208_attempt001/FEEDBACK_PRIVATE.json');predA={};predB={};receipts=[]
 for pl in ['drone','quad']:
  A=[r for r in rr if r['platform']==pl];B=[r for r in bb if r['platform']==pl];lookup={r['id']:i for i,r in enumerate(A)};fb=[x for x in feedback if x['platform']==pl];a=batch(A);b=batch(B);m=module('query_mlp');initial={n:p.detach().clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=0.)
  # Train only on rows with feedback; all other rows used for replay/readout, not loss.
  with torch.no_grad():sa0=scores(m,'query_mlp',a);sb0=scores(m,'query_mlp',b);err=max(float(abs(sa0-a['scores']).max()),float(abs(sb0-b['scores']).max()));assert err<2e-5
  def margins(s):return torch.stack([s[lookup[x['id_PRIVATE']],x['positive_group_PRIVATE']].max()-s[lookup[x['id_PRIVATE']],x['negative_group_PRIVATE']].max() for x in fb])
  trace=[]
  for step in range(1,17):
   opt.zero_grad();s=scores(m,'query_mlp',a);gap=margins(logodds_scores(m,a) if use_logodds else s);loss=F.softplus(-gap/.1).mean();assert torch.isfinite(loss);loss.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters())));opt.step()
   if step in [1,4,8,16]:trace.append({'loss_readout':'exact_logodds' if use_logodds else 'official_probability','step':step,'loss':float(loss.detach()),'gradient_L2':grad,'pair_satisfaction_before_step':float((gap>0).float().mean()),'mean_gap_before_step':float(gap.detach().mean())})
  with torch.no_grad():sa=scores(m,'query_mlp',a).cpu().numpy();sb=scores(m,'query_mlp',b).cpu().numpy();gap1=margins(torch.tensor(sa,device='cuda')).cpu().numpy();gap0=margins(sa0).cpu().numpy()
  np.savez_compressed(out/'states_PRIVATE'/(pl+'_query_mlp.npz'),**{n:p.detach().cpu().numpy() for n,p in m.state_dict().items()})
  # Same-A no-learned-parameter transfer control: minimum .1 pair margin correction.
  keys=np.stack([interact(feature(r)) for r in A]);keys/=np.maximum(np.linalg.norm(keys,axis=-1,keepdims=True),1e-12);values=np.zeros((len(A),256),np.float32);counts=np.zeros_like(values)
  for x in fb:
   j=lookup[x['id_PRIVATE']];sp=feature(A[j])['scores'];P=x['positive_group_PRIVATE'];N=x['negative_group_PRIVATE'];delta=max(.1-float(sp[P].max()-sp[N].max()),0.)/2
   values[j,P]+=delta;values[j,N]-=delta;counts[j,P]+=1;counts[j,N]+=1
  values/=np.maximum(counts,1);active=np.array(sorted({lookup[x['id_PRIVATE']] for x in fb}));K=torch.tensor(keys[active]);V=torch.tensor(values[active]);memory=[]
  for r in B:
   z=feature(r);q=interact(z);q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12);sim=torch.einsum('qc,anc->qan',torch.tensor(q),K);near,which=sim.max(-1);best,obs=near.topk(min(4,len(active)),dim=-1);pos=which.gather(1,obs);v=V[obs,pos];w=((best-1)/.1).exp();corr=((v*w).sum(-1)/w.sum(-1).clamp_min(1e-12)).numpy();memory.append(z['scores']+corr)
  for rs,ss,preds in [(A,sa,predA),(B,sb,predB)]:
   for i,r in enumerate(rs):
    z=feature(r);p={'source':{'q':int(z['scores'].argmax()),'scores':z['scores']},'ranknet_group':{'q':group_readout(r,ss[i]),'scores':ss[i]}}
    if r['stage']=='R':p['same_A_memory']={'q':int(memory[i].argmax()),'scores':memory[i]}
    else:
     zz=load_npz(BASE/'direct208_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));p['same_feedback_direct']={'q':int(zz['queries'][zz['arm_names'].tolist().index('full_pair_group')]),'scores':z['scores']}
    preds[r['id']]=p
  receipts.append({'platform':pl,'feedback_pairs':len(fb),'A_unique_observations':len({x['id_PRIVATE'] for x in fb}),'params':sum(p.numel() for p in m.parameters()),'steps':16,'initial_replay_maxerror':err,'parameter_delta_L2':float(torch.sqrt(sum((p-initial[n]).square().sum() for n,p in m.named_parameters()))),'pair_before_satisfied':int((gap0>0).sum()),'pair_after_satisfied':int((gap1>0).sum()),'pair_gaps_before':gap0.tolist(),'pair_gaps_after':gap1.tolist(),'trace':trace,'memory_cached_observations':len(active),'memory_no_optimizer':True});print('RANKNET29',pl,'pairs',len(fb),'satisfied',int((gap0>0).sum()),int((gap1>0).sum()),flush=True)
 write(out/'A208_predictions',rr,predA);write(out/'R64_predictions',bb,predB);dump(out/'FIT_RECEIPTS.json',receipts);seal(out,'R29_ranknet_matching_only_A_feedback_B_predictions_sealed',A=208,B=64,steps=32,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),loss_readout='exact_logodds' if use_logodds else 'official_probability',B_feedback_or_task_labels_used=False);print('RANKNET29_SEALED',flush=True)
if __name__=='__main__':main()
