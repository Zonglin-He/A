"""Frozen-box ranking-interface screen. Runtime never reads task annotations.
D208 G supervision only; V64 is predicted before independent labeled selection.
"""
import time,copy,argparse,numpy as np
from common24 import *
import torch
from torch import nn
F=torch.nn.functional
LAYERS=['rank_adapter','query_last','query_mlp','text_mlp']
def mlp():return nn.Sequential(nn.Linear(288,288),nn.ReLU(),nn.Linear(288,288),nn.ReLU(),nn.Linear(288,64))
def source_state():return load_npz(BASE/'capture336_attempt001/projection_source_PRIVATE.npz')
def module(layer):
 s=source_state()
 if layer=='rank_adapter':m=nn.Linear(64,1,bias=False);nn.init.zeros_(m.weight)
 elif layer=='query_last':m=nn.Linear(288,64);m.load_state_dict({k:torch.tensor(s['image.4.'+k]) for k in ('weight','bias')})
 else:
  m=mlp();prefix='text.' if layer=='text_mlp' else 'image.';m.load_state_dict({k:torch.tensor(s[prefix+k]) for k in m.state_dict()})
 return m.cuda().eval()
def bank(r):
 p=ROOT/'crossplatform/round23/bank272_attempt001/samples_PRIVATE'/(r['id']+'.npz')
 return load_npz(p)
def teacher(r,z):
 b=bank(r);return z['scores'].astype(float)+2*b['vision']+2*b['position']+2*b['relation']
def interact(z):
 t=z['tokens'][z['token_mask']].mean(0);t=t/max(float(np.linalg.norm(t)),1e-12)
 return z['query']*t

def batch(rr,with_teacher=False):
 zz=[feature(r) for r in rr];T=max(len(z['tokens']) for z in zz);N=len(rr);d={}
 for key,dim in [('tokens',64),('text_h',288)]:
  x=np.zeros((N,T,dim),np.float32)
  for i,z in enumerate(zz):x[i,:len(z[key])]=z[key]
  d[key]=torch.tensor(x,device='cuda')
 padding=np.ones((N,T),bool);mask=np.zeros((N,T),bool)
 for i,z in enumerate(zz):padding[i,:len(z['tokens'])]=False;mask[i,:len(z['tokens'])]=z['token_mask']
 d['padding']=torch.tensor(padding,device='cuda');d['mask']=torch.tensor(mask,device='cuda')
 for key in ('x','h','query','scores'):d[key]=torch.tensor(np.stack([z[key] for z in zz]),device='cuda')
 d['joint']=torch.tensor(np.stack([interact(z) for z in zz]),device='cuda')
 if with_teacher:
  target=np.stack([teacher(r,z) for r,z in zip(rr,zz)]).astype(np.float32);d['teacher']=torch.tensor(target,device='cuda');d['target']=d['teacher'].argmax(1);d['pT']=(d['teacher']/.25).softmax(-1);d['p0']=(d['scores']/.25).softmax(-1)
 return d

def scores(m,layer,b):
 if layer=='rank_adapter':return b['scores']+m(b['joint']).squeeze(-1)
 q=F.normalize(m(b['x'] if layer=='query_last' else b['h']),dim=-1) if layer in ['query_last','query_mlp'] else b['query']
 t=F.normalize(m(b['text_h']),dim=-1) if layer=='text_mlp' else b['tokens']
 logits=(q@t.transpose(-1,-2))/.07;logits=logits.masked_fill(b['padding'][:,None],-torch.inf)
 return (logits.softmax(-1)*b['mask'][:,None]).sum(-1)

def write_predictions(out,rr,qs,ss=None):
 out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir()
 for i,r in enumerate(rr):
  z=feature(r);names=list(qs);ix=np.array([qs[k][i] for k in names]);payload={'arm_names':np.array(names),'query_indices':ix,'boxes':z['boxes'][ix]}
  if ss is not None:payload['adapted_scores']=ss[i]
  np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),**payload)
 seal(out,'R24_predictions_before_current_subset_scoring',n=len(rr),frozen_boxes=True,no_current_GT=True)

def screen():
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 spec={'layers':LAYERS,'LR':{'rank_adapter':[.01,.05],'query_last':[.0001,.0005],'query_mlp':[.0001,.0005],'text_mlp':[.0001,.0005]},'checkpoints':[16,64],'optimizer':'AdamW weight_decay0','loss':'KL(softmax(G/.25)||softmax(student/.25)) + .2 KL(source||student) + .5 mean relu(.05 + max_other - selected_G_score) + sum mean_squared_parameter_delta','all256_candidates':True,'D208_target_labels_used_to_fit':False,'V64_current_G_or_labels_used_to_fit':False,'platform_specific_states_shared_hyperparameters':True,'frozen_boxes_decoder_backbones':True,'program_sha256':sha(__file__)}
 dump(EXP/'FIT_SPEC_LOCK.json',spec);out=BASE/'screen_attempt001';out.mkdir(exist_ok=False);(out/'states_PRIVATE').mkdir();start=time.perf_counter();receipts=[];allpred={};allD={}
 for platform in ['drone','quad']:
  A=[r for r in rows('D208') if r['platform']==platform];B=[r for r in rows('V64') if r['platform']==platform];a=batch(A,True);b=batch(B);qs={};dq={}
  for layer in LAYERS:
   for k,lr in enumerate(spec['LR'][layer]):
    m=module(layer);source={n:p.detach().clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.);trace=[]
    with torch.no_grad():base=scores(m,layer,a);err=float(abs(base-a['scores']).max());assert err<2e-5
    for step in range(1,65):
     opt.zero_grad();s=scores(m,layer,a);lp=(s/.25).log_softmax(-1);kl=F.kl_div(lp,a['pT'],reduction='batchmean');anchor=F.kl_div(lp,a['p0'],reduction='batchmean');chosen=s.gather(1,a['target'][:,None]).squeeze(1);other=s.clone().scatter(1,a['target'][:,None],-torch.inf).max(-1).values;pair=F.relu(.05+other-chosen).mean();reg=sum((p-source[n]).square().mean() for n,p in m.named_parameters());loss=kl+.2*anchor+.5*pair+reg;assert torch.isfinite(loss);loss.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters())));opt.step()
     if step in (1,16,64):trace.append({'step':step,'loss':float(loss.detach()),'teacher_KL':float(kl.detach()),'source_KL':float(anchor.detach()),'pair':float(pair.detach()),'gradient_L2':grad})
     if step in spec['checkpoints']:
      name=f'{layer}_lr{k}_s{step}';np.savez_compressed(out/'states_PRIVATE'/(platform+'_'+name+'.npz'),**{n:p.detach().cpu().numpy() for n,p in m.state_dict().items()})
      with torch.no_grad():sb=scores(m,layer,b).cpu().numpy();sa=scores(m,layer,a).cpu().numpy()
      qs[name]=sb.argmax(1);dq[name]=sa.argmax(1);receipts.append({'platform':platform,'name':name,'layer':layer,'lr':lr,'steps':step,'params':sum(p.numel() for p in m.parameters()),'parameter_L2_delta':float(torch.sqrt(sum((p-source[n]).square().sum() for n,p in m.named_parameters()))),'initial_replay_max_error':err,'D_G_top1_agreement_percent':float((sa.argmax(1)==a['target'].cpu().numpy()).mean()*100),'trace':copy.deepcopy(trace)})
    print('FIT24',platform,layer,lr,'elapsed',round(time.perf_counter()-start,1),flush=True)
  qs['source']=b['scores'].cpu().numpy().argmax(1);dq['source']=a['scores'].cpu().numpy().argmax(1)
  for i,r in enumerate(B):allpred[r['id']]={k:int(v[i]) for k,v in qs.items()}
  for i,r in enumerate(A):allD[r['id']]={k:int(v[i]) for k,v in dq.items()}
  del a,b,m
 for split,rr,pp in [('V64',rows('V64'),allpred),('D208',rows('D208'),allD)]:
  names=list(next(iter(pp.values())));write_predictions(out/(split+'_predictions'),rr,{k:np.array([pp[r['id']][k] for r in rr]) for k in names})
 dump(out/'FIT_RECEIPTS.json',receipts);seal(out,'R24_D208_G_fit_V64_predictions_no_labels',seconds=time.perf_counter()-start,n_D=208,n_V=64,optimizer_steps=sum(64*2*len(LAYERS) for _ in ['drone','quad']),peak_bytes=torch.cuda.max_memory_allocated(),states=32);print('SCREEN24_SEALED',flush=True)
if __name__=='__main__':screen()
