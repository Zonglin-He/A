"""Post-seal signed failure audit; annotations never feed predictions."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from run import R,D,P,read,put,verify
import numpy as np,torch
verify();assert read(D/'state.json')['status']=='completed_scored'
r=read(P/'report.json');old={x['query']:x for x in read(R/'results/event_query_hc64_v1/report.json')['rows']}
labels={x['id']:x for x in map(json.loads,(R/'data/evaluation/development_labels.jsonl').read_text().splitlines())if x['dataset']=='hc2'}
n=dict(time_harms=0,harm_with_shorter_interval=0,harm_with_longer_interval=0,harm_start_MAE_increased=0,harm_end_MAE_increased=0,moved_to_stronger_feedback=0)
cases=[];changes=[]
for x in r['rows']:
    k=x['query'];tr=torch.load(D/'traces'/f'{k}.pt',map_location='cpu',weights_only=False)
    with np.load(D/'predictions'/f'{k}.npz')as z:
        b=z['NativeG6_interval'];a=z['BoundaryCell_interval'];ids=z['frame_ids'];truth=labels[k]['interval'];idx={int(t):i for i,t in enumerate(ids)}
        def strength(v):return float(tr['curve'][0,idx[int(v[0])]]+tr['curve'][1,idx[int(v[1])-1]])
        dt=100*(x['values']['BoundaryCell']['t']-x['values']['NativeG6']['t']);dv=100*(x['values']['BoundaryCell']['v']-x['values']['NativeG6']['v'])
        c=dict(query=k,group=x['duration_group'],delta_t_pp=dt,delta_v_pp=dv,
               delta_duration_frames=int((a[1]-a[0])-(b[1]-b[0])),
               start_MAE_delta_frames=int(abs(a[0]-truth[0])-abs(b[0]-truth[0])),
               end_MAE_delta_frames=int(abs(a[1]-truth[1])-abs(b[1]-truth[1])),
               feedback_score_delta=strength(a)-strength(b),
               raw_contrast_min=float(min(rr['strengths'].min()for rr in tr['raw'])),
               raw_contrast_max=float(max(rr['strengths'].max()for rr in tr['raw'])),
               legacy_raw_query_delta_t_pp=100*(old[k]['values']['RawQuery']['t']-old[k]['values']['NativeG6']['t']))
    changes.append(c)
    if dt< -1e-10:
        n['time_harms']+=1;n['harm_with_shorter_interval']+=int(c['delta_duration_frames']<0);n['harm_with_longer_interval']+=int(c['delta_duration_frames']>0)
        n['harm_start_MAE_increased']+=int(c['start_MAE_delta_frames']>0);n['harm_end_MAE_increased']+=int(c['end_MAE_delta_frames']>0);n['moved_to_stronger_feedback']+=int(c['feedback_score_delta']>0)
    if k in ['RC009','RC025','RC029','RC030','RC039','RC042','RC016']:cases.append(c)
base=np.array([x['mechanism']['expected_t']['BoundaryCell']-x['mechanism']['expected_t']['ShiftCell']for x in r['rows']])*100
ix=np.random.default_rng(20261022).integers(0,64,(10000,64))
put(P/'failure_analysis.json',dict(scope='post-seal development failure analysis; no prediction modification',harms=n,cases=cases,all_signed_changes=changes,
    expected_t_true_minus_shift=dict(mean_pp=float(base.mean()),CI95_pp=np.quantile(base[ix].mean(1),[.025,.975]).tolist()),
    changed_intervals=sum(x['mechanism']['changed_interval']for x in r['rows']),max_class_mass_error=max(x['mechanism']['max_class_mass_error']for x in r['rows']),
    classes_distinguished=sum(x['mechanism']['classes_with_nonconstant_new_feedback']for x in r['rows']),multi_candidate_classes=sum(x['mechanism']['multi_candidate_classes']for x in r['rows'])))
