"""Reproduce registered R95 saved last-layer fits and every F/C/FC endpoint."""
from core98 import *
def main():
 setsource97('waymo');base=base97('waymo');m,S=model97(base,'waymo');rows=jread(R98/'research/round95/experiment/DEV336_PRIVATE.json');rows=[next(r for r in rows if r['platform']==p) for p in ['drone','quad']];checks=[]
 for r in rows:
  restore(m,S);Shead=patch94(m);f=feature_to92(load(B94/'prediction336_attempt001/features_PRIVATE'/(r['id']+'.npz')));ev=load(B94/'prediction336_attempt001/evidence_PRIVATE'/(r['id']+'.npz'));fc=feature_to92(load(B95/'prediction336_attempt001/C_features_PRIVATE'/(r['id']+'.npz')));r0=torch.as_tensor(ev['r0_PRIVATE'],device='cuda');qp=torch.as_tensor(ev['binding_target_PRIVATE'],device='cuda');p0=torch.as_tensor(ev['correct_base_PRIVATE'][0]+ev['r0_PRIVATE'],device='cuda').softmax(0);mix=(p0+qp)/2;shuffle=torch.as_tensor(ev['shuffle_target_PRIVATE'],device='cuda');old=load(B95/'prediction336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));oo=dict(zip(old['arm_names'],old['scores']));worst=0.;stateerr=0.;trials=0
  for name in FITS98:
   final,states,trace,rec=lm95(m,f,r0,(mix if name=='lm_mixture' else shuffle if name=='lm_shuffle' else qp).log(),Shead,name);sf=load(B95/'prediction336_attempt001/states_PRIVATE'/(r['id']+'_'+name+'_final.npz'));err=max(float(abs(final[k]-sf[k]).max()) for k in final);stateerr=max(stateerr,err);assert err<1e-7
   with torch.no_grad():
    put94(m,final);pF=read94(m,f,r0);pC=read94(m,fc,r0)
    for view,p in [('F',pF),('C',pC),('FC',(pF+pC)/2)]:
     e=float(abs(p.cpu().numpy()-oo[name+'_'+view]).max());worst=max(worst,e);assert e<1e-6 and int(p.argmax())==int(oo[name+'_'+view].argmax())
   oldrec=jread(B95/'prediction336_attempt001/records_PRIVATE'/(r['id']+'.json'))[name];assert len(trace)==len(oldrec['trials']) and rec['accepted_steps']==oldrec['accepted_steps'];trials+=len(trace)
  checks.append({'platform':r['platform'],'max_probability_error':worst,'max_parameter_error':stateerr,'Top1_mismatches':0,'trials':trials})
 jdump(H98/'LEGACY_R95_PARITY_AUDIT.json',{'state':'PASS_EXACT_REGISTERED_R95_FOUR_FITS_AND_TWELVE_ENDPOINTS','checks':checks,'new_task_scores':0,'no_method_math_changes':True});print(checks)
if __name__=='__main__':main()
