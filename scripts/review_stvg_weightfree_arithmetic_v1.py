"""Independent float64 policy/loss/Adam arithmetic from saved gradients, not backward replay."""
import sys,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_tuning_v1'
def iou(x,y):
 import numpy as np
 lo=np.maximum(x[...,:2]-x[...,2:]/2,y[...,:2]-y[...,2:]/2);hi=np.minimum(x[...,:2]+x[...,2:]/2,y[...,:2]+y[...,2:]/2);inter=np.maximum(hi-lo,0).prod(-1);return inter/(x[...,2:].prod(-1)+y[...,2:].prod(-1)-inter+1e-8)
def main():
 import numpy as np,torch
 stages=[('lr',B),('factors',B/'factors002'),('interactions',B/'interaction004')]
 if (B/'confirmation003/STUDENT_GLOBAL_BARRIER.json').exists():stages.append(('confirmation',B/'confirmation003'))
 out=[]
 for name,d in stages:
  counts=dict(fits=0,steps=0,adam_steps=0);maxima=dict(sample=0.,reward=0.,weight=0.,loss=0.,adam_parameter=0.,adam_moment=0.)
  for item in read(d/'STUDENT_GLOBAL_BARRIER.json')['files']:
   assert sha(ROOT/item['path'])==item['sha256'];z=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False)
   for variant,f in z['fits'].items():
    counts['fits']+=1;cfg=f['config'];aa=z['reference_record'][variant]['anchors'];e=np.asarray([a['box_cxcywh']for a in aa]);w=np.asarray([a['weight']for a in aa]);last=None
    for s in f['path']:
     if s.get('skipped'):continue
     counts['steps']+=1;r=s['rollout'];mu=r['mean'].numpy().astype('float64');sample=r['samples'].numpy().astype('float64');seed=int(hashlib.sha256(f"spatial-opd-v1|{f['seed_key']}|{s['step']}".encode()).hexdigest()[:15],16);assert seed==r['seed']
     noise=torch.randn(len(mu),16,4,generator=torch.Generator().manual_seed(seed),dtype=torch.float32);noise=torch.cat((noise,-noise),1).numpy();diff=np.max(abs(sample-(mu[:,None,:]+cfg['sigma']*noise)));maxima['sample']=max(maxima['sample'],float(diff));assert diff<2e-6
     box=1/(1+np.exp(-sample));reward=iou(box,e[:,None,:]);diff=np.max(abs(reward-r['rewards'].numpy()));maxima['reward']=max(maxima['reward'],float(diff));assert diff<3e-6
     scaled=r['used_rewards'].numpy().astype('float64')/cfg['tau'];ex=np.exp(scaled-scaled.max(-1,keepdims=True));weight=ex/ex.sum(-1,keepdims=True);diff=np.max(abs(weight-r['weights'].numpy()));maxima['weight']=max(maxima['weight'],float(diff));assert diff<3e-6
     logp=-(sample-mu[:,None,:])**2;logp=logp.sum(-1)/(2*cfg['sigma']**2)-2*math.log(2*math.pi*cfg['sigma']**2);per=-((r['weights'].numpy().astype('float64')-1/32)*logp).sum(-1);loss=np.mean(w*per);diff=abs(loss-s['loss']);maxima['loss']=max(maxima['loss'],float(diff));assert diff<2e-4
     if s['update']is None:continue
     counts['adam_steps']+=1;offset=0
     for j,(key,p)in enumerate(s['state'].items()):
      a=p.numpy().astype('float64');n=a.size;g=s['update']['gradient'][offset:offset+n].numpy().astype('float64').reshape(a.shape);offset+=n;old=s['optimizer_before']['state'].get(j,{});step=int(old['step'])+1 if old else 1;m0=old['exp_avg'].numpy().astype('float64')if old else np.zeros_like(a);v0=old['exp_avg_sq'].numpy().astype('float64')if old else np.zeros_like(a);m=.9*m0+.1*g;v=.999*v0+.001*g*g;pred=a-cfg['lr']*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8);actual=s['proposed_state'][key].numpy();diff=np.max(abs(pred-actual));maxima['adam_parameter']=max(maxima['adam_parameter'],float(diff));assert diff<2e-6
      opt=s['optimizer_proposed']['state'][j];assert int(opt['step'])==step
      for key2,val in [('exp_avg',m),('exp_avg_sq',v)]:
       err=np.max(abs(val-opt[key2].numpy()));maxima['adam_moment']=max(maxima['adam_moment'],float(err));assert np.allclose(val,opt[key2].numpy(),rtol=2e-5,atol=2e-6)
     assert offset==1792
  out.append(dict(phase=name,counts=counts,max_abs_errors=maxima))
 write(B/('ROOT_POLICY_ARITHMETIC_COMPLETE.json'if len(stages)==4 else'ROOT_POLICY_ARITHMETIC_DEV.json'),dict(status='pass',records=out,CPU_only=True,GT_read=False,scope='saved mean/noise/reward/softmax/weighted loss/Adam arithmetic; saved gradients only, no independent backward replay'))
 print(out)
if __name__=='__main__':main()
