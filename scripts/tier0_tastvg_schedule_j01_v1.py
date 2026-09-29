"""Exact finite-roster subset and margin audit, no inference or gate selection."""
import sys,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_schedule_j01_v1';J0=ROOT/'artifacts/tastvg_joint_j0_v1'

def run():
    assert (OUT/'PREDICTION_BARRIER.json').exists()
    rows=read(J0/'TEMPORAL_REFERENCE_ROWS.json');parents=sorted({r['parent'] for r in rows});source=[]
    for parent in parents:
        rr=[r for r in rows if r['parent']==parent and r['condition']!='clean'];assert len(rr)==5
        source.append(dict(parent=parent,v_gain=float(np.mean([r['historical_full_rerank_gain_v'] for r in rr])),t_gain=float(np.mean([r['historical_full_rerank_gain_t'] for r in rr]))))
    combos=[]
    for indices in itertools.combinations(range(16),4):combos.append(dict(parents=[parents[i] for i in indices],subset_v_gain=float(np.mean([source[i]['v_gain'] for i in indices]))))
    assert len(combos)==1820
    gains=np.array([c['subset_v_gain'] for c in combos]);current=next(c['subset_v_gain'] for c in combos if c['parents']==[0,4,8,12])
    data=dict(subsets=1820,source_count=16,quantiles={str(q):float(np.quantile(gains,q)) for q in [.05,.25,.5,.75,.95]},current_subset_v_gain=current,current_whole_stream_v_gain=current*.25,current_percentile_strict=float(np.mean(gains<current)*100),current_percentile_inclusive=float(np.mean(gains<=current)*100),negative_count=int((gains<0).sum()),negative_fraction=float(np.mean(gains<0)),mean_subset_gain=float(gains.mean()),mean_whole_stream_gain=float(gains.mean()*.25))
    margins=[]
    for r in rows:
        ordered=sorted(r['candidate_scores'],reverse=True);margins.append(dict(parent=r['parent'],condition=r['condition'],expert_scheduled=r['expert_scheduled'],margin=ordered[0]-ordered[1],v_gain=r['historical_full_rerank_gain_v']))
    corr={}
    for group in ['all96','scheduled24']:
        rr=[r for r in margins if group=='all96' or r['expert_scheduled']];x=np.array([r['margin'] for r in rr]);y=np.array([r['v_gain'] for r in rr]);q=float(np.quantile(x,.75));keep=x>=q
        corr[group]=dict(cells=len(rr),sources=len({r['parent'] for r in rr}),pearson=float(np.corrcoef(x,y)[0,1]),upper_quartile_margin=q,upper_quartile_cells=int(keep.sum()),upper_quartile_v_gain=float(y[keep].mean()))
    write(OUT/'TIER0_SOURCE_ROWS.json',source);write(OUT/'TIER0_SUBSETS.json',combos);write(OUT/'TIER0_MARGIN_ROWS.json',margins);write(OUT/'TIER0.json',dict(status='completed_read_only',subset_distribution=data,margin_descriptive=corr,reference_sha256=sha(J0/'TEMPORAL_REFERENCE_ROWS.json'),new_inference=0,new_GT=0,gate_implemented=False,scope='Finite16exposed-source distribution; subsets overlap and are not independent trials. Margin pooled-cell correlations descriptive, not a reliability gate.'))
    print(data,corr)

if __name__=='__main__':run()
