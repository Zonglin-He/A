"""Independent CPU geometry, reference binding, rollback and final-output readback."""
import sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_weightfree_tuning_v1/interaction004'
def main():
 import torch,numpy as np
 from scripts.review_stvg_instance_step_diagnostic_v1 import iou_cxcywh
 from scripts import stvg_target_teacher_student_v1 as core
 seal=read(BASE/'STUDENT_GLOBAL_BARRIER.json');assert seal['status']=='complete'and seal['count']==8 and not seal['GT_read'];checks=0;steps=0;err=0.;fitcount=0;refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']}
 for item in seal['files']:
  p=ROOT/item['path'];assert sha(p)==item['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];assert d['reference_record']==refs[cid]and not d['GT_read']and d['source_restored'];checks+=3
  for variant in ['combo1','combo2']:
   aa=refs[cid][variant]['anchors'];fit=d['fits'].get(variant)
   if not aa:
    assert np.array_equal(d['arms'][variant]['boxes'],d['arms']['F']['boxes']) and d['arms'][variant]['interval']==d['arms']['F']['interval'];assert np.array_equal(d['arms']['T_'+variant]['boxes'],d['arms']['T']['boxes']) and d['arms']['T_'+variant]['interval']==d['arms']['T']['interval'];checks+=2;continue
   fitcount+=1;assert torch.equal(d['readouts'][variant]['boxes'].cpu(),fit['final_cached_decoder_boxes']);checks+=1
   pos=[a['position']for a in aa];e=np.asarray([a['box_cxcywh']for a in aa]);w=np.asarray([a['weight']for a in aa]);stopped=False
   for s in fit['path']:
    if s.get('skipped'):continue
    assert not stopped
    q0=iou_cxcywh(s['boxes'].numpy()[pos],e);q1=iou_cxcywh(s['proposed_boxes'].numpy()[pos],e)
    err=max(err,float(np.max(abs(q0-s['reference_iou_before'].numpy()))),float(np.max(abs(q1-s['reference_iou_proposed'].numpy()))));assert err<3e-6
    if s['accepted']:assert float(w@(q1-q0)/w.sum())>=-3e-6
    if s['proposed']and not s['accepted']:
     assert core._tree_equal(s['optimizer_after'],s['optimizer_before'])and core._tree_equal(s['update_state'],s['state']);stopped=True
    steps+=1;checks+=3
   xy=np.asarray(d['arms'][variant]['boxes']);final=fit['final_cached_decoder_boxes'].numpy();ind=np.concatenate([final[:,:2]-final[:,2:]/2,final[:,:2]+final[:,2:]/2],axis=1);assert np.allclose(ind,xy,atol=1e-7,rtol=0);checks+=1
 write(BASE/'ROOT_READBACK.json',dict(status='pass',case_count=8,fit_count=fitcount,step_count=steps,check_count=checks,max_iou_abs_error=err,GT_read=False,CPU_only=True,scope='independent geometry, saved Adam/parameter rollback, full-vs-cached final decoder equality; not independent backward replay'))
 files=list(seal['files'])
 for p in [BASE/'REFERENCE_LOCK.json',BASE/'TEACHER_GLOBAL_BARRIER.json',BASE/'ROOT_READBACK.json']:files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=8,GT_read=False,files=files,scorer_sha256=sha(ROOT/'scripts/score_stvg_weightfree_interaction_v1.py')))
 print('root pass',fitcount,steps,checks,err)
if __name__=='__main__':main()
