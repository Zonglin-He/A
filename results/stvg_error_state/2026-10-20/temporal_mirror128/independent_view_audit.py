"""Independent actual-pixel view and all sealed reflection/no-op arithmetic."""
from pathlib import Path
import sys,re,hashlib,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
 from methods.stvg_referdino_opd_v1.runtime import decode
 D=R/'artifacts/stvg_temporal_mirror128_v1';HR=R/'artifacts/stvg_temporal_hires128_v1';p=read(D/'PLAN_LOCK.json');b=read(D/'GLOBAL_PREDICTION_BARRIER.json');raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];pattern=re.compile(r'\b(?:left|right|leftward|rightward|leftmost|rightmost|east|west|clockwise|counterclockwise|anticlockwise)\b',re.I);yes=no=0;first=True;maxerr=0.
 assert b['count']==128 and b['plan_sha256']==sha(D/'PLAN_LOCK.json')
 for r,f in zip(p['records'],b['files']):
  assert sha(R/f['path'])==f['sha256'];z=load(R/f['path']);h=load(HR/'predictions'/f"{r['alias']}.pt");caption=raw[r['query_ordinal']]['input']['caption'];subject=read(R/r['subject_path'])['parses']['subject'];ok=not pattern.search(caption+' '+subject);assert bool(ok)==z['mirror_eligible']==r['mirror_eligible'];m=z['outputs']['Mirror']['q'];hh=h['observation']['q'];p0=h['outputs']['Native']['q']
  expected=.5*p0+.25*hh+.25*m;err=float(abs(expected-z['outputs']['MirrorMix']['q']).max());assert err<1e-14;maxerr=max(maxerr,err)
  if ok:
   yes+=1;assert z['observation']is not None
   if first:
    frames,ids=decode(raw[r['query_ordinal']]['input']);assert ids==z['frame_ids'];ref=np.flip(frames,axis=2).copy();assert np.array_equal(np.flip(ref,axis=2),frames)
    assert hashlib.sha256(frames.tobytes()).hexdigest()==z['observation']['original_RGB_sha256']
    assert hashlib.sha256(ref.tobytes()).hexdigest()==z['observation']['RGB_sha256'];first=False
  else:
   no+=1;assert z['observation']is None and np.array_equal(m,hh)and np.array_equal(z['outputs']['MirrorMix']['q'],h['outputs']['HiResMix']['q'])
 assert (yes,no)==(119,9)and read(D/'COST.json')['mirror_Source_attempts']==119
 assert all(sha(R/f)==h for f,h in p['pins'].items())
 out=dict(status='pass',all128_view_fusion_independently_checked=True,mirror_eligible=yes,no_observation_veto=no,raw_first_actual_video_decoded_and_numpy_reflection_hash_exact=True,raw_first_physical_frame_ids_exact=True,max_probability_error=maxerr,new_GPU=0,new_models=0,extra_CPU_video_decode=1)
 put(R/'results/stvg_error_state/2026-10-20/temporal_mirror128/VIEW_AUDIT_PUBLIC.json',out);print(out)

if __name__=='__main__':main()
