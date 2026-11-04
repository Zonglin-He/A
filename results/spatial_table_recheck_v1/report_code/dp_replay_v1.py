"""Four fixed cached current-method temporal+DP replays, no method change."""
from pathlib import Path
import sys,time
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'studies/spatial_table_recheck_v1'))
from common import *
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval
from stvg.spatial import SpatialGeometry,evidence_dp
validate();old.infer_guard();torch.set_num_threads(1);cases=[];start=time.perf_counter()
for ds in COUNTS:
 for j,row in enumerate(rows(ds)[:2],1):
  c,proof=old.source(ds,row);e=fixed(TEMP/'temporal'/ds/'sealed.json')['files'][row['id']];assert sha(TEMP/e['path'])==e['sha256'];q,diag=one_step(c['features']['AuxE_ROI'],c['q0'],kappa=0,beta=1);iv=map_interval(c['ids'],q);g=SpatialGeometry.from_cached_geometry(c['geometries']['main']);new=evidence_dp(g.for_interval(iv))['final'].numpy()
  with np.load(TEMP/e['path'])as z:assert np.array_equal(np.asarray(iv),z['Ours_interval'])and np.array_equal(new,z['Ours'])and np.array_equal(diag['L'],z['evidence_CorrectedROI'])
  cases.append(dict(dataset=ds,case=f'fixed_first{j}',original_input_compact_sha256=proof['compact_sha256'],E_ROI_features_q0_update_MAP_finalDP_array_exact=True,old_prediction_sha256=e['sha256']))
atomic(P/'four_case_current_DP_replay.json',dict(status='four_fixed_actual_cached_current_temporal_and_DP_replays_exact',cases=cases,scope='four codepath cases, not an all-query new method rerun or hard-anchor repair',new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_GTscore=0,elapsed_wall_seconds=time.perf_counter()-start,worker=identity()));print('4cached_current_DP_replays_exact',flush=True)
