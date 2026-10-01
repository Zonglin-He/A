"""Apply the explicit user-approved shared sampling revision; metadata only."""
import sys,time,copy,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.native_support_fig1_common_v1 import BASE,OUT,read,write,sha,set_status
from vg_tta.tastvg_deployment_corruption_v2 import burst_spec

def run():
 assert not (OUT/'RUNTIME_LOCK.json').exists()
 original=read(BASE/'ROSTER.json');roster=copy.deepcopy(original);mapping=[]
 for row,old in zip(roster['rows'],original['rows']):
  ids=old['frame_ids'];pos=np.rint(np.linspace(0,len(ids)-1,min(len(ids),64))).astype(int).tolist();chosen=[ids[i] for i in pos]
  assert chosen==sorted(set(chosen)) and chosen[0]==ids[0] and chosen[-1]==ids[-1]
  row['original_frame_ids']=ids;row['frame_ids']=chosen;row['input']['frame_ids']=chosen
  a=burst_spec(row['source'],row['input']['frame_count'],ids,5);b=burst_spec(row['source'],row['input']['frame_count'],chosen,5)
  assert all(a[k]==b[k] for k in ['physical_start','physical_end','length','uniform_draw','actual_physical_fraction'])
  mapping.append(dict(ordinal=row['ordinal'],original_frames=len(ids),selected_frames=len(chosen),positions=pos,original_observed_fraction=a['actual_observed_fraction'],selected_observed_fraction=b['actual_observed_fraction']))
 assert len(roster['rows'])==len({r['source'] for r in roster['rows']})==128
 roster['sampling_revision']='uniform64_v2 user-approved';roster['original_roster_sha256']=sha(BASE/'ROSTER.json')
 write(OUT/'ROSTER.json',roster);write(OUT/'SAMPLING_AUDIT.json',dict(status='pass',sources=128,changed_sources=sum(x['original_frames']!=x['selected_frames'] for x in mapping),physical_bursts_unchanged=True,mapping=mapping,GT_read=False))
 write(OUT/'USER_AUTHORIZATION.json',dict(date='2026-10-01',user_reply='同意',uniform_max64_approved=True,scope='three models same max64 equispaced original-frame subset, preserve128 sources/corruption/K, rerun no-GT smoke then full diagnostic',time=time.time()))
 shutil.copy2(BASE/'SUBJECTS.json',OUT/'SUBJECTS.json')
 p=read(BASE/'RUNTIME_LOCK.json');p['original_roster_sha256']=sha(BASE/'ROSTER.json');p['original_runtime_lock_sha256']=sha(BASE/'RUNTIME_LOCK.json');p['roster_sha256']=sha(OUT/'ROSTER.json');p['subjects_sha256']=sha(OUT/'SUBJECTS.json');p['sampling_protocol_sha256']=sha(ROOT/'protocols/stvg_native_support_uniform64_v2.md');p['time']=time.time();p['status']='approved_uniform64_runtime_locked_before_new_predictions';p['model_inputs']='same min(n,64) equispaced original-frame IDs for all models; identical raw corruption, native preprocessing'
 p['PTD']['geometry']='native grammar rejection makes missing tube; individual degenerate boxes give zero overlap per frame only; preserve all6 candidates'
 p['pins']={f:sha(ROOT/f) for f in p['pins']};p['pins']['scripts/prepare_native_support_uniform64_v2.py']=sha(Path(__file__));p['pins']['protocols/stvg_native_support_uniform64_v2.md']=p['sampling_protocol_sha256']
 write(OUT/'RUNTIME_LOCK.json',p);write(OUT/'CPU_TESTS.json',dict(passed=6,scope='product beam vs exhaustive / deterministic ties / native inclusion / temporal legality / ratio and paired masks',time=time.time()))
 set_status(dict(status='approved_ready_for_smoke',stage='smoke',model_predictions=0,GT_read=False,GPU_started=False,time=time.time()))
 print('Prepared128 unchanged sources; changed frame lists',sum(x['original_frames']!=x['selected_frames'] for x in mapping),flush=True)
if __name__=='__main__':run()
