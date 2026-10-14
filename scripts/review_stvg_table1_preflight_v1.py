"""GT-free custody, frozen-arm and actual update/readout checks."""
from pathlib import Path
import sys,json,hashlib,argparse,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_final_table1_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def iou(x,y):
 import numpy as np
 x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 xa=x[:,:2]-x[:,2:]/2;xb=x[:,:2]+x[:,2:]/2;ya=y[:,:2]-y[:,2:]/2;yb=y[:,:2]+y[:,2:]/2
 inter=np.maximum(0,np.minimum(xb,yb)-np.maximum(xa,ya)).prod(-1)
 return inter/np.maximum(x[:,2:].prod(-1)+y[:,2:].prod(-1)-inter,1e-12)
def main():
 import numpy as np,torch
 from scripts import stvg_table1_temporal_adapter_v1 as temporal
 from scripts.stvg_native_instance_common_v1 import guard
 sys.addaudithook(guard)
 p=argparse.ArgumentParser();p.add_argument('--case-alias');p.add_argument('--output',required=True);a=p.parse_args()
 barrier=BASE/(f'STUDENT_{a.case_alias}_BARRIER.json'if a.case_alias else'STUDENT_GLOBAL_BARRIER.json');b=json.loads(barrier.read_text());assert b['status']=='sealed'and b['GT_read']is False
 checked=[];steps=0
 for item in b['files']:
  alias=item['alias'];path=ROOT/item['path'];assert sha(path)==item['sha256'];rp=ROOT/item['receipt']['path'];assert sha(rp)==item['receipt']['sha256']
  v=torch.load(path,map_location='cpu',weights_only=False);s=torch.load(BASE/'source_cases'/(alias+'.pt'),map_location='cpu',weights_only=False);tpath=BASE/'teacher_cases'/(alias+'.pt');t=torch.load(tpath,map_location='cpu',weights_only=False)
  assert v['source_file_sha256']==sha(BASE/'source_cases'/(alias+'.pt'))and v['reference_sha256']==sha(tpath)
  assert not v['GT_read']and not v['online_carry']
  arms=v['arms'];assert set(arms)=={'F','T','S','TS','TeacherGeometry'}
  assert np.array_equal(arms['F']['boxes'],s['native_boxes_xyxy_normalized'])and np.array_equal(arms['F']['boxes'],arms['T']['boxes'])and np.array_equal(arms['S']['boxes'],arms['TS']['boxes'])
  assert list(arms['F']['interval'])==list(arms['S']['interval'])==list(s['Native_interval'])
  assert list(arms['T']['interval'])==list(arms['TS']['interval'])==list(arms['TeacherGeometry']['interval'])==list(t['temporal']['final_interval'])
  for arm in arms.values():assert np.isfinite(arm['boxes']).all()
  assert np.array_equal(arms['TeacherGeometry']['boxes'],t['reference']['teacher_geometry']['boxes_xyxy_normalized'])
  if 'teacher_responses'in t['temporal']:
   replay=temporal.fit_query(s,s['temporal_candidates'],t['temporal']['teacher_responses']);assert replay['final_interval']==t['temporal']['final_interval']
  fits=v['fits'];anchors=t['reference']['anchors']
  if not anchors:assert not fits and np.array_equal(arms['F']['boxes'],arms['S']['boxes'])
  else:
   f=fits['SpatialA'];assert f['active_parameters']==1792 and f['declared_steps']==5
   assert f['seed_key']=='caption_query_fixed_space|'+s['dataset']+'_'+s['query_key_sha256']
   positions=[r['position']for r in anchors];evidence=np.asarray([r['box_cxcywh']for r in anchors],dtype=np.float32);w=np.asarray([r['weight']for r in anchors])
   for row in f['path']:
    if row['skipped']:continue
    before=iou(row['boxes'][positions],evidence);after=iou(row['proposed_boxes'][positions],evidence)
    assert np.max(np.abs(before-np.asarray(row['reference_iou_before'])))<2e-6
    assert np.max(np.abs(after-np.asarray(row['reference_iou_proposed'])))<2e-6
    expected=row['proposed']and float((w*(np.asarray(row['reference_iou_proposed'])-np.asarray(row['reference_iou_before']))).sum()/w.sum())>=-1e-7
    assert row['accepted']==expected
    if row['proposed']and not row['accepted']:
     for k in row['state']:assert torch.equal(row['state'][k],row['update_state'][k])
    steps+=1
   readout=v['readouts']['SpatialA']['boxes'];assert torch.equal(readout,f['final_cached_decoder_boxes'])
   xy=readout.double().numpy();xy=np.concatenate((xy[:,:2]-xy[:,2:]/2,xy[:,:2]+xy[:,2:]/2),axis=1);assert np.array_equal(xy,arms['S']['boxes'])
  checked.append({'alias':alias,'status':'pass','sha256':sha(path),'fit_count':len(fits),'anchor_count':len(anchors)})
 out={'status':'pass','scope':'actual source/teacher/student binding, arm composition, temporal replay, independent anchor IoU and accept rule, full/cached readout','count':len(checked),'checked_update_steps':steps,'records':checked,'barrier_sha256':sha(barrier),'GT_read':False,'CPU_only':True,'time':time.time()}
 q=ROOT/a.output
 with q.open('x')as stream:json.dump(out,stream,indent=2)
 print(json.dumps({k:v for k,v in out.items()if k!='records'}))
if __name__=='__main__':main()
