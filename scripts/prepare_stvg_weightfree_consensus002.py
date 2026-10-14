"""One matched teacher selection diagnostic: two ID/order permutations, same pixels/regions."""
import sys,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_weightfree_spatial_v1 as c
import numpy as np
OLD=c.BASE;BASE=OLD/'consensus002'
def main():
 sys.addaudithook(c.guard);BASE.mkdir(exist_ok=True)
 c.write(BASE/'DESIGN_LOCK.json',dict(status='locked',GT_read=False,development_after_initial_results=True,changes='same diverse8 regions; two ID-and-tile permutations plus original vote; accept unique at-least-2/3 candidate majority, Source/0/tie no update',regions_changed=False,OPD_changed=False,case_count=8,additional_calls=64,time=time.time()))
 jobs=[];basejobs=[j for j in c.read(OLD/'TEACHER_JOB_LOCK.json')['records']if j['variant']=='diverse8']
 for cid in dict.fromkeys(j['case_id']for j in basejobs):
  rec=next(r for r in c.read(c.OLD/'GLOBAL_TRACKLET_BARRIER.json')['case_records']if r['case_id']==cid);tc=c.read(ROOT/rec['path']);images,meta=c.old._load_rgb_npz(ROOT/rec['rgb_npz_path'],tc,rec['rgb_npz_sha256']);p=c.read(OLD/'proposals'/(cid+'.json'))
  for old in [j for j in basejobs if j['case_id']==cid]:
   slot=old['center_slot'];win=p['windows'][slot];tracks=win['tracks'];frames=[np.asarray(images[i])for i in win['physical_frame_ids']]
   for kind,ts in [('rotate3',tracks[3:]+tracks[:3]),('reverse',list(reversed(tracks)))]:
    mapping={str(t['id']):i+1 for i,t in enumerate(ts)};vid=c.render(frames,ts,[],mapping);vp=BASE/'teacher_inputs'/f'{cid}_{slot}_{kind}.npy';vp.parent.mkdir(exist_ok=True);np.save(vp,vid)
    j=copy.deepcopy(old);j.update(job_id=f'{cid}_consensus8_{slot}_{kind}',variant='consensus8',kind=kind,visible_id_to_candidate={str(v):int(k)for k,v in mapping.items()},candidate_bindings=ts)
    j['videos'][1]=dict(j['videos'][1],path=str(vp.relative_to(ROOT)),sha256=c.sha(vp),tensor_sha256=c.hashlib.sha256(vid.tobytes()).hexdigest());jobs.append(j)
 c.write(BASE/'TEACHER_JOB_LOCK.json',dict(status='locked',records=jobs,GT_read=False));c.write(BASE/'CANDIDATE_INPUT_BARRIER.json',dict(status='complete',count=8,window_count=32,teacher_job_count=64,GT_read=False,regions_exact_reused=True,old_barrier_sha256=c.sha(OLD/'CANDIDATE_INPUT_BARRIER.json')))
 pins=dict(c.read(OLD/'TEACHER_RUNTIME.json')['pins'])
 for p in [BASE/'DESIGN_LOCK.json',BASE/'TEACHER_JOB_LOCK.json',BASE/'CANDIDATE_INPUT_BARRIER.json',ROOT/'scripts/prepare_stvg_weightfree_consensus002.py',ROOT/'scripts/run_stvg_weightfree_teacher002.py',ROOT/'scripts/launch_stvg_weightfree002.py']:pins[str(p.relative_to(ROOT))]=c.sha(p)
 for j in jobs:
  for v in j['videos']:pins[v['path']]=v['sha256']
 c.write(BASE/'TEACHER_RUNTIME.json',dict(status='locked',GT_read=False,pins=pins,protected=c.read(OLD/'TEACHER_RUNTIME.json')['protected']))
 print('64 ID/order control requests locked, no new proposals, no GT')
if __name__=='__main__':main()
