"""Postseal descriptive diagnostics, never a deployment selector."""
from inventory105 import *
from PIL import Image
import numpy as np
def stats(x):
 a=np.array(x,float);return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None,'max':float(a.max()) if len(a) else None}
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;pairs=rd(H/'FULL_PAIRS_PRIVATE.json');out=[]
 for pl in ['waymo','drone','quad']:
  trans=[];rot=[];pix=[];samepoint=0;n=0
  for pair in pairs:
   if pair['A']['platform']!=pl:continue
   f=R/'crossplatform/round105/registration_full_PRIVATE'/(pair['pair_PRIVATE']+'.json');reg=rd(f)
   if not reg['geometry_cycle_supported']:continue
   T=np.array(reg['forward']['T_raw_B_to_A']);trans.append(float(np.linalg.norm(T[:3,3])));rot.append(float(np.degrees(np.arccos(np.clip((np.trace(T[:3,:3])-1)/2,-1,1)))));samepoint+=reg['A_sensor']['point_sha256']==reg['B_sensor']['point_sha256'];A=np.asarray(Image.open(pair['A']['imagepath']).convert('RGB').resize((384,256)),float)/255;B=np.asarray(Image.open(pair['B']['imagepath']).convert('RGB').resize((384,256)),float)/255;pix.append(float(abs(A-B).mean()));n+=1
  out.append({'platform':pl,'supported_pairs':n,'same_pointcloud_bytes':samepoint,'estimated_translation_m':stats(trans),'estimated_rotation_degrees':stats(rot),'resized_RGB_mean_absolute_difference':stats(pix),'not_an_identity_or_camera_motion_certificate':True})
 dirs=[];subsets=[]
 for name in rd(R/'crossplatform/round105/MATRIX_SEALED.json')['directions']:
  rr=rd(H/'scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'));enabled=[r for r in rr if r['status'].get('TTA_output_enabled')];allB=[r for r in rr if r['status'].get('adaptation_attempted')];changes=[]
  for r in rr:
   a=r['arms']['prepared_frozen'];b=r['arms']['TTA_jointABC']
   if a['query_PRIVATE']!=b['query_PRIVATE']:changes.append({'row_PRIVATE':r['id_PRIVATE'],'before_IoU':a['IoU'],'after_IoU':b['IoU'],'enabled':r['status'].get('TTA_output_enabled',False),'B_mapped_queries':r['status'].get('B_visible_queries',0),'C_mapped_queries':r['status'].get('C_visible_queries',0)})
  dirs.append({'direction':name,'n':len(rr),'TTA_output_enabled':len(enabled),'query_switches_vs_prepared':len(changes),'positive_IoU_changes':sum(x['after_IoU']>x['before_IoU']+1e-12 for x in changes),'negative_IoU_changes':sum(x['after_IoU']<x['before_IoU']-1e-12 for x in changes),'B_mapped_queries_on_attempted':stats([r['status'].get('B_visible_queries',0) for r in allB]),'C_mapped_queries_on_enabled':stats([r['status'].get('C_visible_queries',0) for r in enabled])});dump(H/(name+'_QUERY_CHANGES_PRIVATE.json'),changes)
  for label,ss in [('all',rr),('output_enabled',enabled),('B_available',allB)]:
   if not ss:continue
   for arm in ['prepared_frozen','frozen_jointABC','frozen_ensemble3','TTA_jointABC','wrong_TTA_jointABC']:
    x=np.array([r['arms'][arm]['IoU'] for r in ss]);subsets.append({'direction':name,'subset':label,'arm':arm,'n':len(ss),'mIoU_percent':float(x.mean()*100),'Acc25_percent':float((x>.25).mean()*100),'Acc50_percent':float((x>.5).mean()*100)})
 dump(H/'OBSERVATION_DIAGNOSTICS.json',{'sensor':out,'directions':dirs,'descriptive_only_no_method_change':True});dump(H/'SUBSET_RESULTS.json',subsets)
 with (H/'REPORT.md').open('a') as f:f.write('\n\n## 事后观测与候选支持诊断\n\n'+json.dumps({'sensor':out,'directions':dirs},indent=2)+'\n\n以上仅解释已封存结果。真实图像内容不同不等于有独立主体信息；变换由传感器估计，不认证相机运动或物体身份。启用子集单列用于机制理解，仍以全部768条作为主分母，未据此重新选择方法或步长。\n')
 print('DIAGNOSE105_COMPLETE')
if __name__=='__main__':main()
