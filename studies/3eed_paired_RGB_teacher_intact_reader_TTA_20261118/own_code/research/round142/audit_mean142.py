"""CPU exact frozen-mean readback; no task labels."""
from common142 import *
import numpy as np
def main():
 check();root=B/'supplementary_mean';seal=rd(root/'SEALED.json');n=0
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;old=np.load(B/'image'/name/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(old['arm_names'],old['scores']));a=np.load(root/name/(sid+'.npz'));expect=np.stack([np.add(p['Frozen_F'],p['F_Bimage_only_teacher'])*.5,np.add(p['F_RGBB_only'],p['F_Bimage_only_teacher'])*.5]);assert np.array_equal(a['scores'],expect) and np.array_equal(a['queries'],expect.argmax(1)) and np.array_equal(a['source_boxes'],old['source_boxes'])
  if not e['has_B']:assert np.array_equal(expect[0],p['Frozen_A']) and np.array_equal(expect[1],p['Frozen_A'])
  n+=1
 assert n==seal['n']==1798;out={'status':'PASS','n':n,'task_GT_read':False,'fixed_A256_and_same_observations':True,'probability_arithmetic_bitexact':True,'FINAL_access':False};dump(H/'MEAN_AUDIT.json',out);print(out,flush=True)
if __name__=='__main__':main()
