"""Seal GT-free references from actual candidate selections; no GT selection."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_identity_common_v1 import read,write,sha,guard
BASE=ROOT/'artifacts/stvg_weightfree_spatial_v1'
def main():
 sys.addaudithook(guard);tb=read(BASE/'TEACHER_GLOBAL_BARRIER.json');assert tb['count']==64 and tb['GT_read'] is False
 for f in tb['files']:assert sha(ROOT/f['path'])==f['sha256']
 jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];rows=[];counts=dict(valid=0,invalid=0,abstain=0,source=0,non_source=0,low_presence=0)
 ids=list(dict.fromkeys(j['case_id']for j in jobs))
 for cid in ids:
  p=read(BASE/'proposals'/(cid+'.json'));r=dict(case_id=cid,dataset=p['dataset'],source_case_sha256=p['source_case_sha256'],GT_read=False)
  for variant in ['fast4','diverse8']:
   aa=[];decisions=[]
   for j in [x for x in jobs if x['case_id']==cid and x['variant']==variant]:
    out=read(BASE/'teacher'/(j['job_id']+'.json'));z=out['parsed'];sel=z['selected_candidate_id'];win=p['windows'][j['center_slot']];reason='invalid';counts['valid'if z['status']=='valid'else'invalid']+=1
    if z['status']=='valid':
     if sel is None:reason='abstain';counts['abstain']+=1
     elif sel==1:reason='source_no_new_evidence';counts['source']+=1
     else:
      tr=next(t for t in win['tracks']if t['id']==sel);assert sel in j['allowed_output_ids'];counts['non_source']+=1
      if tr['presence']<.6:reason='insufficient_local_presence';counts['low_presence']+=1
      else:
       a,b,c,d=tr['center_box'];aa.append(dict(position=win['source_grid_position'],physical_frame_id=win['center_physical_frame_id'],box_cxcywh=[(a+c)/2,(b+d)/2,c-a,d-b],weight=tr['presence'],candidate_id=sel,teacher_output_sha256=sha(BASE/'teacher'/(j['job_id']+'.json'))));reason='selected_appearance_reference'
    decisions.append(dict(slot=j['center_slot'],selected=sel,reason=reason))
   r[variant]=dict(anchors=aa,decisions=decisions)
  rows.append(r)
 write(BASE/'REFERENCE_LOCK.json',dict(status='locked',records=rows,counts=counts,GT_read=False,teacher_barrier_sha256=sha(BASE/'TEACHER_GLOBAL_BARRIER.json')))
 old=read(ROOT/'artifacts/stvg_cross_frame_validation_v1/STUDENT_RUNTIME.json');pins={p:h for p,h in old['pins'].items() if not p.startswith('artifacts/stvg_cross_frame_validation_v1/')}
 for p in ['scripts/run_stvg_weightfree_student_v1.py','scripts/launch_stvg_weightfree_v1.py','scripts/prepare_stvg_weightfree_references_v1.py','scripts/stvg_instance_validated_fit_v1.py','scripts/stvg_native_instance_spatial_fit_v1.py',str((BASE/'REFERENCE_LOCK.json').relative_to(ROOT))]:pins[p]=sha(ROOT/p)
 for cid in ids:
  p=ROOT/'artifacts/stvg_instance_identity_update_v1/student'/(cid+'.pt');pins[str(p.relative_to(ROOT))]=sha(p)
 write(BASE/'STUDENT_RUNTIME.json',dict(status='locked',GT_read=False,pins=pins))
 print(counts,{r['case_id']:{v:len(r[v]['anchors'])for v in ['fast4','diverse8']}for r in rows})
if __name__=='__main__':main()
