"""Four fixed rank-MLP states, native-score preference+keep, all outputs sealed."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time,numpy as np
from common36 import *
sys.path.insert(0,str(ROOT/'crossplatform/round32'))
from coordinate_projection32 import project_group_preferences

def main():
 import torch
 from torch import nn
 F=torch.nn.functional;check_plan();verify(BASE/'capture272_attempt001');out=BASE/'fit272_attempt001';out.mkdir(exist_ok=False)
 for n in ['states_PRIVATE','predictions_PRIVATE','libraries_PRIVATE']:(out/n).mkdir()
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.perf_counter();fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');assert len(fb)==22;receipts=[]
 for mode in MODES:
  source={k.removeprefix('rank.'):v for k,v in load_npz(source_path(mode)).items() if k.startswith('rank.')}
  for pl in ['drone','quad']:
   A=[r for r in rows() if r['platform']==pl];D=[r for r in rows('DEV64') if r['platform']==pl];rr=A+D;zz=[captured(r,mode) for r in rr];frozen=np.stack([z['native_scores'] for z in zz]);inputs=np.stack([z['rankinput'] for z in zz]);gate=np.stack([z['gate'] for z in zz]);U=[j for j,r in enumerate(A) if not any(f['id_PRIVATE']==r['id'] for f in fb)];assert len(U)=={'drone':96,'quad':93}[pl];ff=[f for f in fb if f['platform']==pl];assert len(ff)=={'drone':10,'quad':12}[pl];lookup={r['id']:j for j,r in enumerate(A)};q0=frozen[:len(A)].argmax(-1);m=nn.Sequential(nn.Linear(200,64),nn.GELU(),nn.Linear(64,1)).cuda().eval();m.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in source.items()});assert sum(p.numel() for p in m.parameters())==12929;original={k:v.detach().clone() for k,v in m.state_dict().items()};x=torch.tensor(inputs,device='cuda');g=torch.tensor(gate,device='cuda');s0=torch.tensor(frozen,device='cuda');q0t=torch.tensor(q0,device='cuda')
   with torch.no_grad():r0=m(x).squeeze(-1).detach();d0=s0[:len(A)]-s0[:len(A)].gather(1,q0t[:,None]);assert torch.equal(s0,s0+g*(m(x).squeeze(-1)-r0))
   opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=0.);trace=[]
   def ss():return s0+g*(m(x).squeeze(-1)-r0)
   def gaps(s):return torch.stack([s[lookup[f['id_PRIVATE']],f['positive_group_PRIVATE']].max()-s[lookup[f['id_PRIVATE']],f['negative_group_PRIVATE']].max() for f in ff])
   for step in range(1,17):
    opt.zero_grad();s=ss();gap=gaps(s);pref=F.softplus(-gap/.1).mean();rel=s[:len(A)]-s[:len(A)].gather(1,q0t[:,None]);keep=(rel[U]-d0[U]).square().mean();loss=pref+keep;assert torch.isfinite(loss);loss.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters())));opt.step()
    if step in [1,4,8,16]:trace.append({'step':step,'timing':'before step','pref':float(pref.detach()),'keep':float(keep.detach()),'gradnorm':grad})
   with torch.no_grad():adapted=ss().cpu().numpy();finalgaps=gaps(ss()).cpu().numpy()
   state={k:v.detach().cpu().numpy() for k,v in m.state_dict().items()};np.savez_compressed(out/'states_PRIVATE'/(pl+'_'+mode+'.npz'),**state);keys=norm(inputs[:len(A)]).astype(np.float32);raw=np.zeros((len(A),256),float);qps=[]
   for j,r in enumerate(A):
    fff=[f for f in ff if f['id_PRIVATE']==r['id']]
    if fff:
     p=project_group_preferences(frozen[j].astype(float),[(f['positive_group_PRIVATE'],f['negative_group_PRIVATE']) for f in fff],margin=.1,q0=int(q0[j]),representative_scores=frozen[j]);raw[j]=p['raw_query_correction'];qps.append({'alias':r['alias'],'objective':p['objective_half_group_l2'],'KKT':p['kkt_max_abs_error']})
   np.savez_compressed(out/'libraries_PRIVATE'/(pl+'_'+mode+'.npz'),keys=keys,raw=raw,A_ids_PRIVATE=np.array([r['id'] for r in A]));K=torch.tensor(keys,device='cuda');qa=[];legacy=[]
   for j,r in enumerate(rr):
    p0=frozen[j];qt0=int(p0.argmax());memory=None;payload={}
    if j<len(A):memory=p0.astype(float)+raw[j]
    else:
     Q=torch.tensor(norm(inputs[j]).astype(np.float32),device='cuda');sim=torch.einsum('qc,anc->qan',Q,K);near,which=sim.max(-1);best,obs=near.topk(4,dim=-1);pos=which.gather(1,obs);w=((best-1)/.1).exp().double();o=obs.cpu().numpy();po=pos.cpu().numpy();ww=w.cpu().numpy();retrieved=(raw[o[:,:1],po[:,:1]]*ww[:,:1]).sum(-1)/np.maximum(ww[:,:1].sum(-1),1e-12);memory=p0.astype(float)+retrieved-retrieved[qt0];selected=sim[torch.arange(256,device='cuda'),obs[:,0]];top2=selected.topk(2,-1).values
     payload={'GPU_obs':o,'GPU_pos':po,'GPU_similarity':best.cpu().numpy(),'GPU_weights':ww,'A_row_margin':(best[:,0]-best[:,1]).cpu().numpy(),'chosen_row_candidate_margin':(top2[:,0]-top2[:,1]).cpu().numpy(),'A_exact_tie_count':(near==best[:,0,None]).sum(-1).cpu().numpy(),'candidate_exact_tie_count':(selected==best[:,0,None]).sum(-1).cpu().numpy(),'retrieved_raw':retrieved}
    vals=np.stack([p0.astype(float),adapted[j].astype(float),memory]);query=np.array([grouped_readout(r,s,p0) for s in vals]);assert query[0]==qt0;qa.append(int(query[1]));z=feature(r);np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'_'+mode+'.npz'),arm_names=np.array(['fused_frozen','rank_TTA','sameinfo_direct_QP' if j<len(A) else 'sameinfo_rank200_memory']),native_scores=vals,queries=query,boxes=z['boxes'][query],source_rank0_used=r0[j].cpu().numpy(),source_q0=np.array(qt0),**payload)
   for k,f in enumerate(ff):legacy.append({'pair_alias':f['pair_alias'],'legacy_label':f['label'],'gap':float(finalgaps[k]),'positive_over_negative':bool(finalgaps[k]>0),'final_positive':qa[lookup[f['id_PRIVATE']]] in f['positive_group_PRIVATE'],'source_frozen_final_positive':int(q0[lookup[f['id_PRIVATE']]]) in f['positive_group_PRIVATE']})
   rec={'mode':mode,'platform':pl,'params':12929,'steps':16,'U':len(U),'pref_denominator':len(ff),'all_pairs_satisfied':int((finalgaps>0).sum()),'legacy_B_pairs_satisfied':sum(x['positive_over_negative'] for x in legacy if x['legacy_label']=='B'),'legacy_B_final_positive':sum(x['final_positive'] for x in legacy if x['legacy_label']=='B'),'legacy_B_not_adaptation_eligibility':True,'rank_delta_L2':float(torch.sqrt(sum((p-original[n]).square().sum() for n,p in m.state_dict().items()))),'nonrank_fusion_parameters_not_present_in_optimizer':True,'source_frozen_initialization_bitexact_by_delta_form':True,'native_formula':'s0+alpha*(rank_theta(rankinput)-rank_source(rankinput)) algebraically native base+alpha*rank','trace':trace,'pairs':legacy,'QP':qps};receipts.append(rec);print('FIT36',mode,pl,'pairs',rec['all_pairs_satisfied'],'seconds',round(time.perf_counter()-start,1),flush=True)
 dump(out/'FIT_RECEIPTS.json',receipts);seal(out,'R36_fixed_2x2_all_ADEV_native_score_predictions_before_labels',states=4,optimizer_steps=64,n=272,modes=MODES,all22feedback_retained=True,complete_scores_and_actual_GPU_neighbors_saved=True,seconds=time.perf_counter()-start,plan_sha256=sha(EXP/'PLAN_LOCK.json'));print('FIT36_SEALED')
if __name__=='__main__':main()
