"""CPU saved-output cross-frame diagnosis; no model, media or GT."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_cross_frame_common_v1 import BASE,PRIOR,OLD,read,write,sha,guard

def main():
 sys.addaudithook(guard)
 import torch,numpy as np
 from scripts.review_stvg_instance_step_diagnostic_v1 import iou_cxcywh
 refs=read(BASE/'REFERENCE_LOCK.json');rows=[]
 for r in refs['records']:
  cid=r['case_id'];p=PRIOR/'student'/(cid+'.pt')
  if sha(p)!=read(p.with_suffix('.receipt.json'))['sha256']:raise RuntimeError('prior receipt changed')
  d=torch.load(p,map_location='cpu',weights_only=False)
  op=OLD/'student'/(cid+'.pt')
  if sha(op)!=read(op.with_suffix('.receipt.json'))['sha256']:raise RuntimeError('original receipt changed')
  o=torch.load(op,map_location='cpu',weights_only=False)
  vals=r['validation_anchors'];positions=[a['position']for a in vals];boxes=np.asarray([a['box_cxcywh']for a in vals]);result=[]
  for name in ['raw_iou','validated_loss_weighted_iou']:
   fit=d['fits'].get(name)
   if fit is None and name=='raw_iou':fit=o['fits'].get('iou')
   if fit is None:result.append(dict(arm=name,status='no_fit',steps=[]));continue
   step_rows=[]
   for s in fit['path']:
    before=s.get('boxes');after=s.get('proposed_boxes',s.get('after_boxes'))
    if after is None and before is not None and 'actual_delta' in s:after=before+s['actual_delta']
    if before is None or after is None:continue
    q0=None if not positions else float(iou_cxcywh(before.numpy()[positions],boxes).mean())
    q1=None if not positions else float(iou_cxcywh(after.numpy()[positions],boxes).mean())
    step_rows.append(dict(step=int(s['step']),validation_mean_before=q0,validation_mean_proposed=q1,cross_frame_non_decrease=None if q0 is None else q1>=q0-1e-7,proposal=bool(s.get('update') is not None),accepted_actual=s.get('accepted')))
   result.append(dict(arm=name,status='saved_path_readback',steps=step_rows))
  rows.append(dict(case_id=cid,dataset=r['dataset'],validation_count=len(vals),records=result))
 if torch.cuda.is_initialized():raise RuntimeError('unexpected GPU init')
 write(BASE/'SAVED_UNLABELED_DIAGNOSTIC.json',dict(status='complete',scope='saved decoder outputs against fixed validation masks; no new GT',count=8,records=rows,reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),CPU_only=True,GT_read=False,time=time.time()))
 print('Saved path no-label diagnostic complete')
if __name__=='__main__':main()
