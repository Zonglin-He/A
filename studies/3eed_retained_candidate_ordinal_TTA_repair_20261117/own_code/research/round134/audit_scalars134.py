from common134 import *
import numpy as np
def main():
 mat=rd(B134/'MATRIX_SEALED.json');Q=H134/'scores_attempt001';rows=[];maximum=0.;checks=0
 for name in mat['seals']:
  rr=rd(Q/(name+'_ALL_ROWS_PRIVATE.json'));rows+=rr
  for r in rr:
   a=np.load(B134/'formal'/name/'samples_PRIVATE'/(r['id_PRIVATE']+'.npz'));pol=np.load(B134/'policy_PRIVATE'/name/(r['id_PRIVATE']+'.npz'));allnames=np.r_[a['arm_names'],pol['arm_names']];allscores=np.r_[a['scores'],pol['scores']];y=np.load(R134/'research/round131/scores_attempt001'/(name+'_'+r['id_PRIVATE']+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious']
   for k,p in zip(allnames,allscores):assert abs(float(y[int(np.argmax(p))])-r['IoUs'][str(k)])<1e-14;checks+=1
 methods=rd(Q/'ALL_METHOD_ROWS.json')
 for table,sub in [('ALL_METHOD_ROWS',False),('ALL_SCENE_ROWS_PRIVATE',True)]:
  for m in rd(Q/(table+'.json')):
   rr=[r for r in rows if r['source']==m['source'] and r['target']==m['target'] and (not sub or r['scene_PRIVATE']==m['scene_PRIVATE'])];w=np.asarray([r['sampling_weight'] if m['aggregation']=='design_weighted' else 1. for r in rr]);y=np.array([r['IoUs'][m['arm']] for r in rr]);expected={'mIoU_percent':float(np.dot(w,y)/w.sum()*100),'Acc25_percent':float(np.dot(w,y>.25)/w.sum()*100),'Acc50_percent':float(np.dot(w,y>.5)/w.sum()*100)}
   assert len(rr)==m['n']
   for k,v in expected.items():maximum=max(maximum,abs(v-m[k]));assert abs(v-m[k])<1e-10
 for m in rd(Q/'AGGREGATE_ROWS.json'):
  rr=[r for r in methods if r['aggregation']==m['aggregation'] and r['arm']==m['arm'] and (m['group']=='all_six_equal' or r['target']==m['group'])]
  for k in ['mIoU_percent','Acc25_percent','Acc50_percent']:assert abs(sum(r[k] for r in rr)/len(rr)-m[k])<1e-12
 for m in rd(Q/'ALL_PAIRED_ROWS.json'):
  rr=[r for r in rows if r['source']==m['source'] and r['target']==m['target']];d=np.array([r['IoUs'][m['method']]-r['IoUs'][m['control']] for r in rr]);w=np.array([r['sampling_weight'] if m['aggregation']=='design_weighted' else 1. for r in rr]);assert abs(float(np.dot(w,d)/w.sum()*100)-m['gain_pp'])<1e-12;assert int((d>1e-12).sum())==m['better'] and int((d< -1e-12).sum())==m['worse'] and int((abs(d)<=1e-12).sum())==m['same']
 assert len(rows)==1798 and checks==1798*13;out={'status':'PASS_ALL23374_CACHED_IOUS_SIX_DIRECTION_SCENE_WEIGHTS_AND_PAIRED_SCALARS','n':len(rows),'task_values':checks,'max_scalar_error':maximum,'new_raw_GT_geometry':False,'FINAL_access':False};dump(H134/'SCALAR_AUDIT.json',out);print(out)
if __name__=='__main__':main()
