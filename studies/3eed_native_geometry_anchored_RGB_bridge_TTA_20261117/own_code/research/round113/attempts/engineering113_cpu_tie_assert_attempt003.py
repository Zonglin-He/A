"""Actual real-input intervention and gradient/readout checks, no task GT."""
from case113 import *
def main():
 for p,h in jread(H113/'experiment/ENGINEERING_CODE_LOCK.json')['files'].items():assert sha(R113/p)==h,p
 checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src)
  assert sum(p.numel() for n,p in m.tta_parameters())==102498
  assert sum(p.numel() for n,p in m.tta_parameters() if scope113(n,'bridge'))==45313
  assert sum(p.numel() for n,p in m.tta_parameters() if scope113(n,'rgb'))==47441
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=[r for r in jread(R113/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json') if (B111/name/'prefix_PRIVATE'/(r['id']+'.npz')).exists()]
   for r in rows[:2]:
    st=time.perf_counter();c=fitcase113(base,m,S,src,r);ps,ff,mapping,ev=evaluate113(base,m,S,src,r,c,True);err=0.;cpu=0
    head={k:v for k,v in S.items() if k.startswith(('match.','interface.out.'))}
    for key in sorted({v['feature_key'] for v in mapping.values()}):
     a={k[len(key)+1:]:v for k,v in ff.items() if k.startswith(key+'_')};cp,diag=cpu113(a,head);assert max(diag.values())<5e-6
     for arm,desc in mapping.items():
      if desc['feature_key']!=key:continue
      if desc['no_late'] or desc['no_r']:cc,_=cpu113(a,head,desc['no_late'],desc['no_r'])
      else:cc=cp
      v=cc[desc['readout']];e=float(abs(v-ps[arm]).max());err=max(err,e);assert e<5e-6 and int(v.argmax())==int(ps[arm].argmax()), {'arm':arm,'CPU_error':e,'CPU_top':int(v.argmax()),'native_top':int(ps[arm].argmax()),'native_gap':float(ps[arm].max()-ps[arm][int(v.argmax())]),'CPU_gap':float(v.max()-v[int(ps[arm].argmax())])};cpu+=1
    # Source-only depth predictor bypasses bridge: prove its exact zero gradient.
    restore(m,c['starts']['anchor']);enable_theta91(m);pm,im=c['pm'][0],c['im'][0];lp,li,ld=m.local_losses(c['z'],c['d'],pm,im);pars=[p for n,p in m.tta_parameters() if scope113(n,'bridge')];dg=torch.autograd.grad(ld.sum(),pars,allow_unused=True);assert all(g is None or not torch.count_nonzero(g) for g in dg)
    restore(m,S);freeze91(m);assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());checks.append({'source':src,'target':tgt,'CPU_readouts':cpu,'independent_CPU_error':err,'legacy_R97_currentA_raw_BT_query_alpha_exact':ev['legacy_R97_raw_bt_query_and_alpha_exact'],'full_actual_versus_fast_foundation_probability_error':ev['fast_full_native_max_error'],'foundation_h_u_r_invariant':True,'depth_bridge_gradient_exact_zero':True,'C_neural_after_all_fits':True,'source_unchanged':True,'seconds':time.perf_counter()-st});print('ENGINEERING113',src,tgt,len(checks),round(time.perf_counter()-st,2),flush=True)
  del base,m;torch.cuda.empty_cache()
 jdump(H113/'ENGINEERING.json',{'state':'PASS_REAL_NATIVE_SCOPE_GRADIENT_LEGACY_SAMEINFO_CPU_C_BOUNDARY','n':len(checks),'checks':checks,'GT_used':False,'FINAL_access':False})
if __name__=='__main__':main()
