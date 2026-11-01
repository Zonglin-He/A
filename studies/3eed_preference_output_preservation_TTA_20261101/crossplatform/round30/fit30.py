"""Single change: relative logodds keep on unlabelled/unaccepted A observations."""
import sys,time,numpy as np,collections
from common30 import *
import torch
import fit24
fit24.feature=feature
from fit24 import module,batch,scores
from fit29 import logodds_scores,group_readout
F=torch.nn.functional

def write(out,rr,ss,q):
 out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir()
 for r,s,ix in zip(rr,ss,q):np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),scores=s,query=int(ix),box=feature(r)['boxes'][ix])
 seal(out,'R30_model_predictions_before_current_taskscore',n=len(rr),boxes_frozen=True)

def main():
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;out=BASE/'screen_attempt001';out.mkdir(exist_ok=False);(out/'states_PRIVATE').mkdir();start=time.perf_counter();fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');receipts=[]
 for pl in ['drone','quad']:
  A=[r for r in rows() if r['platform']==pl];D=[r for r in rows('DEV64') if r['platform']==pl];a=batch(A);d=batch(D);lookup={r['id']:i for i,r in enumerate(A)};ff=[f for f in fb if f['platform']==pl];U=[i for i,r in enumerate(A) if r['id'] not in {f['id_PRIVATE'] for f in ff}];q0=a['scores'].argmax(1)
  for beta in read(EXP/'PLAN_LOCK.json')['finite_beta_grid']:
   m=module('query_mlp');initial={n:p.detach().clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=0.);trace=[]
   with torch.no_grad():z0=logodds_scores(m,a);d0=z0-z0.gather(1,q0[:,None]);base=scores(m,'query_mlp',a);err=float(abs(base-a['scores']).max());assert err<2e-5
   def gaps(z):return torch.stack([z[lookup[f['id_PRIVATE']],f['positive_group_PRIVATE']].max()-z[lookup[f['id_PRIVATE']],f['negative_group_PRIVATE']].max() for f in ff])
   for step in range(1,17):
    opt.zero_grad();z=logodds_scores(m,a);gap=gaps(z);pref=F.softplus(-gap/.1).mean();dist=z-z.gather(1,q0[:,None]);keep=(dist[U]-d0[U]).square().mean();loss=pref+beta*keep;assert torch.isfinite(loss);loss.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters())));opt.step()
    if step in [1,4,8,16]:trace.append({'step':step,'total':float(loss.detach()),'preference':float(pref.detach()),'keep':float(keep.detach()),'gradient_L2':grad})
   with torch.no_grad():sa=scores(m,'query_mlp',a).cpu().numpy();sd=scores(m,'query_mlp',d).cpu().numpy();za=logodds_scores(m,a);g1=gaps(za).cpu().numpy();keep_final=float(((za-za.gather(1,q0[:,None]))[U]-d0[U]).square().mean())
   name='b'+str(beta).replace('.','p');np.savez_compressed(out/'states_PRIVATE'/(pl+'_'+name+'.npz'),**{n:p.detach().cpu().numpy() for n,p in m.state_dict().items()});qa=[group_readout(r,s) for r,s in zip(A,sa)];qd=sd.argmax(1)
   write(out/(pl+'_'+name+'_A'),A,sa,qa);write(out/(pl+'_'+name+'_DEV'),D,sd,qd)
   old=load_npz(ROOT/'crossplatform/round29/ranknet_logodds_attempt001/states_PRIVATE'/(pl+'_query_mlp.npz'));stateerr=max(float(abs(v.detach().cpu().numpy()-old[n]).max()) for n,v in m.state_dict().items()) if beta==0 else None
   if beta==0:assert stateerr<1e-6,('beta0_reproduction',pl,stateerr)
   receipt={'platform':pl,'beta':beta,'name':name,'feedback_pairs':len(ff),'keep_observations':len(U),'steps':16,'params':sum(p.numel() for p in m.parameters()),'initial_replay_maxerror':err,'parameter_delta_L2':float(torch.sqrt(sum((p-initial[n]).square().sum() for n,p in m.named_parameters()))),'final_keep_MSE':keep_final,'pairs_satisfied':int((g1>0).sum()),'B_pairs_satisfied':sum(g1[j]>0 for j,f in enumerate(ff) if f['label']=='B'),'B_final_positive':sum(qa[lookup[f['id_PRIVATE']]] in f['positive_group_PRIVATE'] for f in ff if f['label']=='B'),'beta0_state_vs_R29_maxerror':stateerr,'trace':trace};receipts.append(receipt);print('FIT30',pl,beta,'B',receipt['B_pairs_satisfied'],receipt['B_final_positive'],'keep',keep_final,flush=True)
 dump(out/'FIT_RECEIPTS.json',receipts);seal(out,'R30_finite_keep_screen_DEVpredictions_before_selection',models=8,steps=128,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),no_DEV_or_Blabels_fit=True);print('SCREEN30_SEALED')
if __name__=='__main__':main()
