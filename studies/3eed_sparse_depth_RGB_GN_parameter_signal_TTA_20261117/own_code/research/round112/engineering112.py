"""Actual edits versus functional ablations and fully native positive controls."""
from core112 import *
def main():
 checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);head=state(m.match)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=[r for r in jread(R112/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json') if (B111/name/'prefix_PRIVATE'/(r['id']+'.npz')).exists()]
   for r in rows[:2]:
    sid=r['id'];z0,g,zz,proof=pair112(src,r);d=tensor_data(g,r);restore(m,S);rt,info=role97(base,m,r,z0,g,zz['A'],d);route=tensor_route89(rt);perm=permutation94(r,d,0);dw=bound94(d,perm);assert torch.equal(d['visible'],dw['visible']) and torch.equal(d['p2i'].ge(0).any(1),dw['p2i'].ge(0).any(1));old=load(B106/name/'features_PRIVATE'/(sid+'.npz'));native=load(B111/name/'features_PRIVATE'/(sid+'.npz'));orig=load(R112/'crossplatform/round110/original_PRIVATE'/name/(sid+'.npz'));mx=cpu=0.
    for view in VIEWS112:
     edit112(m,S,'full');ref=forward112(m,base,zz[view],d,route);assert float(abs(ref['p'].cpu().numpy()-old['real_'+view+'_p']).max())<1e-7
     for binding,dd in [('correct',d),('wrong',dw)]:
      for variant in VAR112:
       meta=edit112(m,S,variant);a=forward112(m,base,zz[view],dd,route);restore(m,S)
       with functional112(m,variant):b=forward112(m,base,zz[view],dd,route)
       e=float(abs(a['p']-b['p']).max());mx=max(mx,e);assert e<1e-12 and all(torch.equal(a[k],b[k]) for k in ['h','u','late','r'])
       cp=cpu112(a['h'].cpu().numpy(),a['u'].cpu().numpy(),a['mask'].cpu().numpy(),a['late'].cpu().numpy(),a['r'].cpu().numpy(),head);ee=float(abs(cp-a['p'].cpu().numpy()).max());cpu=max(cpu,ee);assert ee<5e-6 and int(cp.argmax())==int(a['p'].argmax())
       if variant=='neither':
        nh=orig['h'] if view=='A' else native['real_fusedABC_h'];nu=orig['tokens'] if view=='A' else native['real_fusedABC_tokens'];assert np.array_equal(a['h'].cpu().numpy(),nh) and np.array_equal(a['u'].cpu().numpy(),nu)
    restore(m,S);assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());checks.append({'source':src,'target':tgt,'full_R91_probability_parity':True,'both_bypassed_h_and_text_bitexact_native_A_and_fused':True,'actual_edit_functional_probability_error':mx,'actual_edit_functional_h_u_late_residual_bitexact':True,'independent_CPU_head_probability_error':cpu,'only_three_allowed_tensors_changed':True,'source_states_restored':True,'same_visibility_for_wrongbinding':True,'no_new_sensor_or_PointNet_forward':True});print('ENGINEERING112',src,tgt,len(checks),flush=True)
  del base,m;torch.cuda.empty_cache()
 jdump(H112/'ENGINEERING.json',{'state':'PASS_ACTUAL_FUNCTIONAL_NATIVE_POSITIVE_CONTROLS_FIXED_POSITIONS_FULL_SOURCE_PARITY_CPU_PROJECTIONS_RESTORE','n':len(checks),'checks':checks,'GT_used':False,'FINAL_access':False})
if __name__=='__main__':main()
