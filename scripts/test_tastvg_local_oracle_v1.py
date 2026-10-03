"""Small analytic controls for local capacity and tied-oracle geometry."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_local_oracle_math_v1 import geometry,derive,summarize

def record(intervals,values):
    return dict(dataset='test',split='search',source_id=0,condition='clean',order='order1',arrival=0,
        A_state_pre_sha256='x',A_state_post_sha256='y',choices={'A8':0},duration_seconds=10.,
        intervals_normalized=intervals,candidate_v=values,candidate_t=values)

def main():
    # Extent and placement endpoint L1 are max components, not additive.
    z=geometry([.2,.6],[.1,.8],10.)
    assert abs(z['distance']-.3)<1e-12 and z['category']=='expansion' and z['dominance']=='extent'
    z=geometry([.2,.6],[.4,.8],10.)
    assert abs(z['distance']-.4)<1e-12 and z['dominance']=='centre'
    # Tied maxima have distinct local availability; retain both extremes.
    xy=[[.2,.6]]*8+[[.19,.6],[.7,.9]]+[[0.,1.]]*22
    v=[.2]*8+[.5,.5]+[.1]*22;r=derive(record(xy,v))
    assert r['oracle_indices']==[8,9] and r['nearest_index']==8 and r['farthest_index']==9
    assert r['anchor_0p1_accessible_extra']==.3 and r['anchor_0p1_near_gain_mass']==.3 and r['anchor_0p1_far_gain_mass']==0
    # Local improvement over local Old8 is different from added capacity over full Old8.
    xy[1]=[.7,.9];v[1]=.45;v[8]=.4;r=derive(record(xy,v))
    assert abs(r['anchor_0p1_local_extra']-.2)<1e-12 and r['anchor_0p1_accessible_extra']==0
    assert abs(r['anchor_inf_accessible_extra']-.05)<1e-12
    # Ratios remain undefined if there is no added opportunity.
    r=derive(record([[.2,.6]]*32,[.2]*32));s=summarize([r])
    assert s['capacity_gain_shares']['anchor_inf_accessible_extra']['mean'] is None
    assert s['capacity_gain_shares']['anchor_inf_accessible_extra']['bootstrap_zero_denominator_draws']==10000
    json.dumps([r,s],allow_nan=False)
    print('LOCAL_ORACLE_ANALYTIC_CONTROLS_PASS: 6')

if __name__=='__main__':main()
