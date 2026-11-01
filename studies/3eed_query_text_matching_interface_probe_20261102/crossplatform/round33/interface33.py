"""Existing query/text matching projection 2x2 interface probe, no new labels."""
import sys,time,numpy as np
from common33 import *
import torch,fit24
from fit29 import logodds_scores as query_logodds
import scope32
fit24.feature=feature
from fit24 import module,batch,scores
F=torch.nn.functional

def logodds(m,layer,b):
 if layer=='query_mlp':return query_logodds(m,b)
 t=F.normalize(m(b['text_h']),dim=-1);v=b['query']@t.transpose(-1,-2)/.07;pos=b['mask']&~b['padding'];neg=~b['mask']&~b['padding'];return torch.logsumexp(v.masked_fill(~pos[:,None],-torch.inf),-1)-torch.logsumexp(v.masked_fill(~neg[:,None],-torch.inf),-1)

def main():
 out=BASE/'interface_screen_attempt001';out.mkdir(exist_ok=False);(out/'states_PRIVATE').mkdir();(out/'predictions_PRIVATE').mkdir();torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.perf_counter();feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');fids={f['id_PRIVATE'] for f in feedback};receipts=[]
 for pl in ['drone','quad']:
  A=[r for r in rows() if r['platform']==pl];D=[r for r in rows('DEV64') if r['platform']==pl];a=batch(A);d=batch(D);lookup={r['id']:i for i,r in enumerate(A)};ff=[f for f in feedback if f['platform']==pl];U=[i for i,r in enumerate(A) if r['id'] not in fids];q0=a['scores'].argmax(1)
  for layer in ['query_mlp','text_mlp']:
   for lr in [.0001,.0005]:
    m=module(layer);initial={n:p.detach().clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.)
    with torch.no_grad():z0=logodds(m,layer,a);d0=z0-z0.gather(1,q0[:,None]);p0=scores(m,layer,a);err=float(abs(p0-a['scores']).max());assert err<2e-5
    def margins(z):return torch.stack([z[lookup[f['id_PRIVATE']],f['positive_group_PRIVATE']].max()-z[lookup[f['id_PRIVATE']],f['negative_group_PRIVATE']].max() for f in ff])
    trace=[]
    for step in range(1,17):
     opt.zero_grad();z=logodds(m,layer,a);gap=margins(z);pref=F.softplus(-gap/.1).mean();rel=z-z.gather(1,q0[:,None]);keep=(rel[U]-d0[U]).square().mean();loss=pref+keep;assert torch.isfinite(loss);loss.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters())));opt.step()
     if step in [1,4,8,16]:trace.append({'step':step,'loss':float(loss.detach()),'preference':float(pref.detach()),'keep':float(keep.detach()),'gradient_L2':grad})
    with torch.no_grad():pa=scores(m,layer,a).cpu().numpy();pd=scores(m,layer,d).cpu().numpy();g=margins(logodds(m,layer,a)).cpu().numpy()
    qa=np.array([scope32.group_readout(r,s) for r,s in zip(A,pa)]);qd=pd.argmax(1);name=layer+'_lr'+str(lr).replace('.','p');state={n:p.detach().cpu().numpy() for n,p in m.state_dict().items()};np.savez_compressed(out/'states_PRIVATE'/(pl+'_'+name+'.npz'),**state)
    for split,rr,pp,qq in [('A208',A,pa,qa),('DEV64',D,pd,qd)]:np.savez_compressed(out/'predictions_PRIVATE'/(pl+'_'+name+'_'+split+'.npz'),row_ids_PRIVATE=np.array([r['id'] for r in rr]),queries=qq,scores=pp)
    repro=None
    if layer=='query_mlp' and lr==.0005:
     old=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));repro=max(float(abs(state[n]-old[n]).max()) for n in state);assert repro<1e-6
    receipt={'platform':pl,'layer':layer,'lr':lr,'name':name,'params':sum(p.numel() for p in m.parameters()),'steps':16,'keep_observations':len(U),'initial_replay_maxerror':err,'pairs_satisfied':int((g>0).sum()),'B_pairs_satisfied':sum(g[j]>0 for j,f in enumerate(ff) if f['label']=='B'),'B_final_positive':sum(qa[lookup[f['id_PRIVATE']]] in f['positive_group_PRIVATE'] for f in ff if f['label']=='B'),'parameter_delta_L2':float(torch.sqrt(sum((p-initial[n]).square().sum() for n,p in m.named_parameters()))),'R30_query_state_reproduction_maxerror':repro,'trace':trace};receipts.append(receipt);print('INTERFACE33',pl,layer,lr,'Bpair/final',receipt['B_pairs_satisfied'],receipt['B_final_positive'],flush=True)
 dump(out/'FIT_RECEIPTS.json',receipts);seal(out,'R33_existing_interface_finite_matrix_before_DEVselection',models=8,steps=128,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),no_target_tasklabels_fit=True);print('INTERFACE33_SEALED')
if __name__=='__main__':main()
