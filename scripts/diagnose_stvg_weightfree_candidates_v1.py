"""No-GT comparison to cached imperfect native references; not oracle truth."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_weightfree_spatial_v1 import BASE,read,write,sha,iou

def main():
 native=read(ROOT/'artifacts/stvg_instance_identity_update_v1/REFERENCE_LOCK.json')['records'];rows=[]
 for case in native:
  cid=case['case_id'];p=read(BASE/'proposals'/(cid+'.json'));lookup={a['physical_frame_id']:a for a in case['presence_supported']['anchors']}
  for w in p['windows']:
   ref=lookup.get(w['center_physical_frame_id']);tr=w['tracks'];n=len(tr)-1
   r=dict(case_id=cid,slot=w['slot'],new_count=n,small_region_count=sum((t['center_box'][2]-t['center_box'][0])*(t['center_box'][3]-t['center_box'][1])<.03 for t in tr[1:]),native_reference_available=ref is not None)
   if ref:
    r['Source_native_reference_IoU']=iou(tr[0]['center_box'],ref['box_xyxy_normalized']);r['pool_max_native_reference_IoU']=max(iou(t['center_box'],ref['box_xyxy_normalized'])for t in tr);r['new_pool_max_native_reference_IoU']=max([iou(t['center_box'],ref['box_xyxy_normalized'])for t in tr[1:]],default=None)
   rows.append(r)
 write(BASE/'CANDIDATE_NATIVE_REFERENCE_DIAGNOSTIC.json',dict(status='complete',GT_read=False,scope='imperfect cached native reference agreement; not GT coverage or method performance',records=rows))
 for cid in dict.fromkeys(r['case_id']for r in rows):
  rr=[r for r in rows if r['case_id']==cid];valid=[r for r in rr if r['native_reference_available']];print(cid,'regions',sum(r['new_count']for r in rr),'native_ref_positions',len(valid),'source',sum(r['Source_native_reference_IoU']for r in valid)/len(valid)if valid else None,'pool',sum(r['pool_max_native_reference_IoU']for r in valid)/len(valid)if valid else None)
if __name__=='__main__':main()
