"""Existing R91 saved endpoints are the reference; no new task scores."""
from run97 import *
def main():
 setsource97('waymo');base=base97('waymo');m,S=model97(base,'waymo');rows=jread(R97/'research/round91/experiment/DEV336_PRIVATE.json');rows=[next(r for r in rows if r['platform']==p) for p in ['drone','quad']];checks=[]
 for r in rows:
  z0=prefix(r);g=load(casepath(r));rt=load(R97/'crossplatform/round89/target_evidence336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));scores,rec,cost=case97(base,m,S,r,z0,g,rt);old=load(R97/'crossplatform/round91/prediction336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));oldrows=dict(zip(old['arm_names'],old['scores']));errs=[]
  for j,a in enumerate(ARMS97):
   if a=='original_source':continue
   name={'prepared_frozen':'frozen','direct_mean3':'mean3','direct_feature3':'feature3'}.get(a,a);q=oldrows['R76_full_disc_'+name];error=float(abs(q-scores[j]).max());errs.append(error);assert error<2e-6,(a,error);assert q.argmax()==scores[j].argmax(),a
  checks.append({'platform':r['platform'],'max_probability_error':max(errs),'Top1_mismatches':0})
 jdump(H97/'ENGINEERING_R91_PARITY_AUDIT.json',{'state':'PASS_FIXED_R91_SEVEN_PREPARED_DIRECT_AND_UPDATE_READOUTS','checks':checks,'new_GT_scores':0,'method_math_changed':False});print(checks)
if __name__=='__main__':main()
