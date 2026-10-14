"""Majority consensus in canonical proposal identity; no labels read."""
import sys,collections,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_identity_common_v1 import read,write,sha,guard
OLD=ROOT/'artifacts/stvg_weightfree_spatial_v1';BASE=OLD/'consensus002'
def main():
 sys.addaudithook(guard);seal=read(BASE/'TEACHER_GLOBAL_BARRIER.json');assert seal['count']==64 and not seal['GT_read']
 for f in seal['files']:assert sha(ROOT/f['path'])==f['sha256']
 jobs=read(OLD/'TEACHER_JOB_LOCK.json')['records'];ids=list(dict.fromkeys(j['case_id']for j in jobs));rows=[];counts=collections.Counter()
 for cid in ids:
  p=read(OLD/'proposals'/(cid+'.json'));anchors=[];decisions=[]
  for win in p['windows']:
   slot=win['slot'];paths=[OLD/'teacher'/f'{cid}_diverse8_{slot}.json']+[BASE/'teacher'/f'{cid}_consensus8_{slot}_{kind}.json'for kind in ['rotate3','reverse']];vote=[]
   for path in paths:
    z=read(path)['parsed'];vote.append(z['selected_candidate_id']if z['status']=='valid'else None)
   counter=collections.Counter(vote);chosen,n=counter.most_common(1)[0];selected=chosen if n>=2 else None;reason='abstain_or_disagreement'
   if selected==1:reason='source_no_new_evidence'
   elif selected is not None:
    tr=next(t for t in win['tracks']if t['id']==selected)
    if tr['presence']<.6:reason='insufficient_local_presence'
    else:
     a,b,c,d=tr['center_box'];anchors.append(dict(position=win['source_grid_position'],physical_frame_id=win['center_physical_frame_id'],box_cxcywh=[(a+c)/2,(b+d)/2,c-a,d-b],weight=tr['presence'],candidate_id=selected,teacher_output_sha256=[sha(x)for x in paths]));reason='majority_non_source_reference'
   counts[reason]+=1;decisions.append(dict(slot=slot,votes=vote,selected=selected,reason=reason))
  rows.append(dict(case_id=cid,dataset=p['dataset'],source_case_sha256=p['source_case_sha256'],GT_read=False,consensus8=dict(anchors=anchors,decisions=decisions)))
 # A 2x2 diagnosis: original vs majority teacher selection, nested shrink guard off/on.
 original={r['case_id']:r for r in read(OLD/'REFERENCE_LOCK.json')['records']}
 for row in rows:
  props=read(OLD/'proposals'/(row['case_id']+'.json'))['windows'];lookup={w['source_grid_position']:w for w in props}
  for target,source in [('nested8',original[row['case_id']]['diverse8']),('consensus_nested8',row['consensus8'])]:
   rr=copy.deepcopy(source);kept=[];rejected=[]
   for a in rr['anchors']:
    win=lookup[a['position']];src=win['tracks'][0]['center_box'];x,y,w,h=a['box_cxcywh'];new=[x-w/2,y-h/2,x+w/2,y+h/2];area=w*h;sa=(src[2]-src[0])*(src[3]-src[1]);inter=max(0,min(src[2],new[2])-max(src[0],new[0]))*max(0,min(src[3],new[3])-max(src[1],new[1]));reject=inter/area>=.9 and area/sa<2/3
    if reject:rejected.append(dict(position=a['position'],candidate_id=a['candidate_id'],intersection_fraction=inter/area,area_ratio=area/sa))
    else:kept.append(a)
   rr['anchors']=kept;rr['nested_rejections']=rejected;row[target]=rr
 write(BASE/'REFERENCE_LOCK.json' ,dict(status='locked',GT_read=False,records=rows,counts=dict(counts),rule='unique >=2/3 canonical-candidate votes; else no anchor'))
 pins={p:h for p,h in read(OLD/'STUDENT_RUNTIME.json')['pins'].items()if p!=str((OLD/'REFERENCE_LOCK.json').relative_to(ROOT))}
 for p in [BASE/'REFERENCE_LOCK.json',ROOT/'scripts/run_stvg_weightfree_student002.py',ROOT/'scripts/launch_stvg_weightfree002.py',ROOT/'scripts/prepare_stvg_weightfree_consensus_references002.py',BASE/'GEOMETRY_GUARD_ADDENDUM.json']:pins[str(p.relative_to(ROOT))]=sha(p)
 write(BASE/'STUDENT_RUNTIME.json',dict(status='locked',GT_read=False,pins=pins));print(dict(counts));print({r['case_id']:{v:len(r[v]['anchors'])for v in ['consensus8','nested8','consensus_nested8']}for r in rows})
if __name__=='__main__':main()
