"""Second independently expressed NumPy state/application/choice and loss root, no GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,hashlib,math,time
from scripts.stvg_opd_real_forward_task_common_v1 import *

def review():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();verify()
 from scripts.score_stvg_opd_real_forward_task_v1 import require_seal
 from scripts.audit_stvg_opd_forward_candidate_v1 import iou
 b=require_seal();counts=collections.Counter()
 for rel,h in b['files'].items():
  assert sha(ROOT/rel)==h;z=load(ROOT/rel);f=z['fit'];names=list(f['initial']);entering=f['before'].numpy();arm=z['arm']
  assert len(f['path'])==f['config']['steps']+1
  for k,hist in enumerate(f['path'][:-1]):
   nxt=f['path'][k+1];rnd=f['rounds'][k];u=hist.get('update')
   if f['empty'] or not rnd['updated']:
    assert all(np.array_equal(hist['state'][n].numpy(),nxt['state'][n].numpy()) for n in names);counts['no_information_rounds']+=1;continue
   rr=rnd['rollout'];mu=rnd['mean_before'].numpy().astype(float);center=rr['mean'].numpy().astype(float);samples=rr['samples'].numpy().astype(float);weights=rr['weights'].numpy().astype(float);sigma=f['config']['sigma']
   # Independently recompute exact float64 likelihood; compare original GPU32 via scale-aware reduction error.
   logp=-np.square(samples-mu[:,None,:]).sum(-1)/(2*sigma*sigma)-2*math.log(2*math.pi*sigma*sigma)
   loss=float((np.square(mu-center).sum(-1)/(2*sigma*sigma)-((weights-1/32)*logp).sum(-1)).mean())
   magnitude=float((np.abs((weights-1/32)*logp)).sum(-1).mean())
   assert abs(loss-rnd['loss'])<1e-5*(1+magnitude)
   counts['likelihood_scalars']+=1
   if arm=='A':continue
   row=f['geometry']['rounds'][k]['policy'];assert np.array_equal(row['current_boxes'].numpy(),hist['boxes'].numpy()) and np.array_equal(row['entering_boxes'].numpy(),entering)
   before=hist['state'];proposal=row['proposal_state'];want={}
   for n in names:
    x=before[n].numpy();y=proposal[n].numpy()
    choice=row['selected']
    expect=y if choice=='A' or (choice=='LN' and n!=prior.old.Q) or (choice=='Query' and n==prior.old.Q) else x if choice in ['LN','Query'] else x+np.float32(.5)*(y-x)
    assert np.array_equal(expect,nxt['state'][n].numpy());counts['state_coordinates']+=expect.size
   if arm=='HALF':assert row['selected']=='Half'
   if arm=='LN':assert row['selected']=='LN'
   if arm=='SEL':
    target=row['targets'].numpy().astype(float);pos=row['positions'];U=row['unvisited'];cur=hist['boxes'].numpy().astype(float);out=row['outputs'];G={a:float((iou(v.numpy()[pos],target)-iou(cur[pos],target)).mean()) for a,v in out.items()}
    D={a:float(np.square(v.numpy()[U].astype(float)-entering[U].astype(float)).sum()/len(U)) if U else None for a,v in out.items()}
    fe=[a for a in ['A','LN','Query','Half'] if G[a]>=.9*G['A']] if G['A']>0 and U else []
    selected='A' if not fe else sorted(fe,key=lambda a:(D[a],['A','LN','Query','Half'].index(a)))[0]
    assert selected==row['selected'] and fe==row['feasible']
    assert all(abs(G[a]-row['scores'][a]['G'])<2e-12 for a in G);counts['independent_selection_rounds']+=1
   counts['candidate_applied_rounds']+=1
  for v in z['audit']['native_head_VJP']:assert v['native_head_VJP_max_error']<2e-6 and not v['independent_full_decoder_Jacobian']
  counts['fits']+=1
  assert sha(ROOT/rel)==h
 for rel,h in read(BASE/'QUALIFICATION_BARRIER.json')['files'].items():
  assert sha(ROOT/rel)==h;q=load(ROOT/rel)
  for v in q['audit']['native_head_VJP']:
   assert v['native_head_VJP_max_error']<2e-6 and not v['independent_full_decoder_Jacobian'];counts['actual_qualification_native_head_VJP_rounds']+=1
 write(BASE/'ROOT_INDEPENDENT_POLICY_APPLICATION_READBACK.json',dict(status='pass',scope='independent NumPy applied proposal masks/current and entering reference/live choices and native float64 likelihood; actual qualification native head VJP trace retained; no independent full physical decoder Jacobian claim',counts=dict(counts),source_sha256=sha(ROOT/'scripts/review_stvg_opd_real_forward_policy_independent_v1.py'),CPU_only=True,GT_read=False,time=time.time()))
 print('INDEPENDENT_POLICY_APPLICATION_PASS',dict(counts))
if __name__=='__main__':review()
