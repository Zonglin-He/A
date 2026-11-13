"""Immutable frozen R76 features, strict pools, and exactly original R80 masks."""
from core91 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args();base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=loadbase91(m,'R76');rows=source_rows('TRAIN512')+source_rows('VAL128');rows=rows[:2] if a.engineering else rows;O=B91/('engineering_source_cache_attempt001' if a.engineering else 'source_features640_attempt001');(O/'samples_PRIVATE').mkdir(parents=True,exist_ok=False);start=time.perf_counter();error=0
with torch.no_grad():
 for j,r in enumerate(rows):
  assert r['platform']=='waymo';z0,z,d,rt,route,pm,im,ws=data91(r,True);m.route=route;out={}
  for v,(p,i) in enumerate([(None,None)]+list(zip(pm,im))):
   f=native_feature91(m,base,z,d,p,i)
   for k,x in f.items():out[str(v)+'_'+k]=x.cpu().numpy()
  # Shared suffix representations are unchanged by the R89 matching-head continuation.
  loadbase91(m,'R89continue');f2=native_feature91(m,base,z,d);error=max(error,float(abs(f2['h'].cpu().numpy()-out['0_h']).max()));assert error==0;loadbase91(m,'R76');np.savez_compressed(O/'samples_PRIVATE'/(r['id']+'.npz'),**out)
  if (j+1)%128==0:print('CACHE91',j+1,flush=True)
assert fingerprint(base)==fp;dump(O/'COUNTS.json',{'n':len(rows),'native_suffix_calls':4*len(rows),'original_R76_or_3EED_parameters_changed':False,'target_inputs_or_labels_read':False,'source_labels_read':False,'strict_object_support':True,'native_hidden_difference_between_bases':error,'seconds':time.perf_counter()-start});seal(O,'R91_FROZEN_SOURCE_FEATURES_BOTH_BASES_ORIGINAL_PHYSICAL_VIEWS',n=len(rows));print(read(O/'COUNTS.json'))
