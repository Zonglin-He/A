"""Independent CPU source/transport derivation, all states/readouts and source invariants."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round110';B=R/'crossplatform/round110';sys.path.insert(0,str(R/'research/round108'));from replay108 import ce_cpu108
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def pre(h,S,j):
 x=torch.from_numpy(h).T[None]
 if j==1:return F.conv1d(x,torch.from_numpy(S['net.0.weight']))
 assert j==5;x=F.conv1d(x,torch.from_numpy(S['net.0.weight']));x=F.batch_norm(x,torch.from_numpy(S['net.1.running_mean']),torch.from_numpy(S['net.1.running_var']),torch.from_numpy(S['net.1.weight']),torch.from_numpy(S['net.1.bias']),training=False,momentum=0.,eps=1e-5).relu();return F.conv1d(x,torch.from_numpy(S['net.4.weight']))
def moments(x):return x.double().mean((0,2)).numpy(),x.double().var((0,2),unbiased=False).numpy()
def gm(*xs):z=np.mean(np.log(np.stack(xs)),0);p=np.exp(z-z.max());return p/p.sum()
def main():
 torch.set_num_threads(4);matrix=rd(B/'MATRIX_SEALED.json');assert matrix['n']==768 and len(matrix['arms'])==58 and rd(H/'ENGINEERING.json')['n']==12;wrong=rd(H/'WRONG_ANCHOR_PRIVATE.json');bufferkeys=[f'net.{j}.{k}' for j in [1,5] for k in ['running_mean','running_var']];n=ends=alias=states=0;mx=mpmax=0.;details=[]
 for p,h in rd(B/'original_PRIVATE/SEALED.json')['files'].items():assert sha(B/'original_PRIVATE'/p)==h
 for name,hs in matrix['directions'].items():
  src,_,tgt=name.split('_');D=B/name;assert sha(D/'SEALED.json')==hs
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
  S=dict(np.load(B/(src+'_CE_SOURCE_PRIVATE.npz')));S0=dict(np.load(R/'crossplatform/round108'/(src+'_CE_WEIGHTS_PRIVATE.npz')));assert set(S)==set(S0) and all(np.array_equal(S[k],v) for k,v in S0.items())
  for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
   sid=r['id'];out=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(out['arm_names'],out['scores']));old=np.load(R/'crossplatform/round108'/name/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(old['arm_names'],old['scores']));rec=rd(D/'records_PRIVATE'/(sid+'.json'));assert np.array_equal(out['source_boxes'],old['source_boxes']) and np.array_equal(out['queries'],out['scores'].argmax(1));n+=1
   for k,p in po.items():assert np.array_equal(ps[k],p)
   def ck(a,b):
    nonlocal mx,ends
    e=float(abs(a-b).max());mx=max(mx,e);assert e<5e-6 and int(a.argmax())==int(b.argmax()),(name,sid,e);ends+=1
   if rec['protected_alias_fallback']:
    alias+=1;assert not rec['calibrations'] and not (B/'original_PRIVATE'/name/(sid+'.npz')).exists();assert all(np.array_equal(p,po['prepared_frozen']) for k,p in ps.items() if k not in po);continue
   orig=np.load(B/'original_PRIVATE'/name/(sid+'.npz'));qO,_=ce_cpu108(orig['h'],orig['mask'],S);ck(qO,orig['pce']);assert abs(orig['p']-po['original_source']).max()<1e-7 and np.array_equal(orig['source_boxes'],old['source_boxes']);ck(qO,ps['native_original_CE']);ck(orig['p'],ps['original_source_matchedcoverage']);ck(gm(orig['p'],qO),ps['original_native_pair_geometric']);ck((orig['p']+qO)/2,ps['original_native_pair_arithmetic']);wa=np.load(B/'original_PRIVATE'/name/(wrong[name][sid]+'.npz'));cache=np.load(R/'crossplatform/round106'/name/'features_PRIVATE'/(sid+'.npz'));P=cache['real_A_h'];ev=rec['events'];assert ev.index('calibrated_transport')<ev.index('first_C_and_fusedABC_read') and ev.index('calibrated_wrong_anchor')<ev.index('first_C_and_fusedABC_read')
   pA=cache['real_A_p'];pC=cache['real_C_p'];pF=cache['real_fusedABC_p'];qF=po['softtoken_fusedABC']
   for a,p in {'original_CE_with_prepared_geometric_A':gm(pA,qO),'original_CE_with_prepared_arithmetic_A':(pA+qO)/2,'original_CE_with_fused_geometric':gm(pF,qO),'original_CE_with_fused_arithmetic':(pF+qO)/2,'original_CE_with_C_geometric':gm(pC,qO),'direct_three_geometric':gm(pF,qO,qF),'direct_three_arithmetic':(pF+qO+qF)/3}.items():ck(p,ps[a])
   for label,O in [('transport',orig['h']),('wrong_anchor',wa['h'])]:
    st=dict(np.load(D/'states_PRIVATE'/(sid+'_'+label+'.npz')));stats=np.load(D/'transport_PRIVATE'/(sid+'_'+label+'.npz'));assert all(np.array_equal(st[k],v) for k,v in S.items() if k not in bufferkeys);partial={k:v.copy() for k,v in S.items()}
    for ix,j in enumerate([1,5]):
     muO=stats[f'{ix}_reference_mean'];vO=stats[f'{ix}_reference_var'];muP=stats[f'{ix}_input_mean'];vP=stats[f'{ix}_input_var'];eps=float(stats[f'{ix}_eps']);a=np.sqrt(vP/np.maximum(vO,eps));mu=muP+a*(S[f'net.{j}.running_mean'].astype(float)-muO);var=a*a*(S[f'net.{j}.running_var'].astype(float)+eps)-eps;ok=(vO>eps)&(vP>eps)&(var>=0)&np.isfinite(mu)&np.isfinite(var);assert np.array_equal(ok,stats[f'{ix}_valid']);expectMu=np.where(ok,mu,S[f'net.{j}.running_mean']).astype(np.float32);expectVar=np.where(ok,var,S[f'net.{j}.running_var']).astype(np.float32)
     assert np.allclose(st[f'net.{j}.running_mean'],expectMu,rtol=1e-6,atol=1e-6) and np.allclose(st[f'net.{j}.running_var'],expectVar,rtol=1e-6,atol=1e-6)
     m0,v0=moments(pre(O,S,j));m1,v1=moments(pre(P,partial,j))
     for x,y in [(m0,muO),(v0,vO),(m1,muP),(v1,vP)]:e=float(np.max(abs(x-y)/(1+abs(y))));mpmax=max(mpmax,e);assert e<3e-5,(name,sid,label,j,e)
     partial[f'net.{j}.running_mean']=st[f'net.{j}.running_mean'];partial[f'net.{j}.running_var']=st[f'net.{j}.running_var']
    info=rec['calibrations'][label];assert info['learned_parameters_changed']==0 and info['calibrated_running_stat_values']==1152;norm=float(np.sqrt(sum(np.square(st[k].astype(float)-S[k].astype(float)).sum() for k in bufferkeys)));assert abs(norm-info['buffer_L2'])<1e-10;states+=1;details.append({'source':src,'target':tgt,'calibration':label,**info,'actual_C_available':rec['actual_C_available']})
    for v,p in [('A',pA),('C',pC),('fusedABC',pF)]:
     q,_=ce_cpu108(cache['real_'+v+'_h'],cache['real_'+v+'_mask'],st)
     for pol,x in [('CE',q),('geometric',gm(p,q)),('arithmetic',(p+q)/2)]:ck(x,ps[label+'_'+pol+'_'+v])
  print('AUDIT110',name,flush=True)
 dump(H/'NATIVE_AUDIT.json',{'state':'PASS_SEALS_ORIGINAL_NATIVE_SOURCES_STATS_FORMULA_INDEPENDENT_CPU_MOMENTS_ALL_STATES_READOUTS_EVENTS_FALLBACKS','n':n,'arms':58,'endpoints':ends,'calibration_states':states,'protected_alias_fallback':alias,'max_CPU_native_probability_error':mx,'max_relative_independent_preBN_moment_error':mpmax,'full_native_engineering':12,'C_hidden_used_in_calibration':False,'GT_used':False,'FINAL_access':False});dump(H/'CALIBRATION_ROWS_PRIVATE.json',details);print('AUDIT110_PASS',n,ends,states)
if __name__=='__main__':main()
