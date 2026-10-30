"""Pure synthetic CPU masks/metrics/gate fixtures, no taskdata or GT file."""
from pathlib import Path
import json,types,hashlib,tempfile
import numpy as np
from metrics import evaluate_case,summaries
from score_geometry import ARMS,ROLES,gate
rows=[];gt=np.array([0,0,0,2,2,2,0.],float);source=np.array([1,0,0,2,2,2.],np.float32)
for i,(xf,xn,xm) in enumerate([(.5,.7,1.2),(1.,1.,1.),(1.,.2,.8)]):
 boxes=np.tile(source,(4,1));boxes[1:,0]=[xf,xn,xm];axes=np.zeros((4,6),bool)
 if i!=1:axes[1:,:3]=True
 r,c=evaluate_case(sid='synthetic'+str(i),platform='drone',scene='s'+str(i%2),boxes_raw6=boxes,arm_names=ARMS,accepted_axes=axes,target7=gt,statuses=['source','ok','ok','ok']);rows.append(r)
s,c=summaries(rows,ROLES)
get=lambda arm,pop:next(x for x in s if x['platform']=='drone' and x['arm']==arm and x['population']==pop)
assert get('full_rgb','own_changed_descriptive')['n']==1 and get('no_rgb_coordinate','own_changed_descriptive')['n']==2
assert all(get(arm,'coordinate_pair_union_changed')['n']==2 for arm in ARMS)
assert get('source','own_changed_descriptive')['n']==0 and get('source','own_changed_descriptive')['mean_iou'] is None
assert all(x['scene_clusters']<=2 for x in s if x['platform']=='drone')
assert all(x['size_L2_m_mean'] in (0.,None) for x in s)
assert any(x['comparison']=='RGB_coordinate_increment' and x['IoU_gain']['negative_count']>0 for x in c if x['platform']=='drone')
# Strict equality atIoU=.25 failsAcc25.
b=np.tile(np.array([0,0,0,1,1,2],np.float32),(4,1));r,_=evaluate_case(sid='threshold',platform='drone',scene='synthetic',boxes_raw6=b,arm_names=ARMS,accepted_axes=np.zeros((4,6),bool),target7=gt,statuses=['source']*4);assert r['arms']['source']['iou']==.25
ss,_=summaries([r],ROLES);assert next(x for x in ss if x['platform']=='drone' and x['arm']=='source' and x['population']=='all')['Acc25']==0
# Fake release is refused before even requesting prediction/index arguments.
with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as td:
 p=Path(td)/'release.json';p.write_text('{"CPU_SCORE_RELEASE":false}')
 try:gate(types.SimpleNamespace(release=p,release_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 except AssertionError:pass
 else:raise AssertionError('Unreleased target evaluation admitted')
out={'state':'synthetic_R10_CPU_metric_matching_and_release_gate_pass','synthetic_only':True,'same_union_changed_denominator_allarms':True,'own_changed_not_same_population':True,'empty_population_None':True,'acceptance_doesnot_imply_actualchange':True,'negative_RGB_constraint_case_retained':True,'strict25_equality_not_success':True,'unreleased_gate_denied_before_index_or_GT':True,'no_model_GPU_targetdata_or_annotation_read':True}
f=Path(__file__).with_name('CPU_METRIC_FIXTURE.json');assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
