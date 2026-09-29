import numpy as np
from vg_tta.tastvg_paper_readouts_v1 import standard_columns,arrival_kind,quartile,mixed_schedule,dense_official_metrics

def test_recall_thresholds_are_strict_per_query_not_on_mean():
 z=standard_columns([[0,.2,.3],[0,.4,.5]])
 np.testing.assert_array_equal(z[:,2:],[ [0,0],[1,0]])
 assert z[:,2].mean()==.5

def test_transfer_partition_and_quartile_edges():
 assert [arrival_kind(f,e) for f,e in [(True,True),(False,True),(True,False),(False,False)]]==['expert','expert','first_source_nonexpert','later_source_nonexpert']
 assert [quartile(i,8) for i in range(8)]==[0,0,1,1,2,2,3,3]

def test_mixed_stream_resets_only_once_and_sources_share_condition():
 rows=[dict(source=str(i//2)) for i in range(28)];z=mixed_schedule(rows,list(range(28)))
 assert sum(r['reset'] for r in z)==1
 assert len(set(x['condition'] for x in z))==6
 assert all(z[i]['condition']==z[i+1]['condition'] for i in range(0,28,2))

def test_dense_interpolation_differs_from_sampled_metric():
 # Middle prediction is a perfect linear interpolation; all 3 physical frames count.
 pred=[[0,0,1,1],[2,0,3,1]];gt={i:[i,0,i+1,1] for i in range(3)}
 z=dense_official_metrics(pred,[0,2],(0,3),gt,(0,3));assert z['m_vIoU']==1
 assert dense_official_metrics(pred,[0,2],(4,5),gt,(0,3))['m_vIoU']==0
