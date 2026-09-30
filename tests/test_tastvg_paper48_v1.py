import numpy as np
from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy,source_summary
from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics

def test_actual_official_evaluator_dense_interpolation_and_thresholds():
 f=DenseMetric();b=np.array([[0,0,100,100],[200,0,300,100]],float);ids=[0,2];gt={i:[100*i,0,100*(i+1),100] for i in range(3)}
 x=f(b,ids,[0,3],gt,[0,3]);y=dense_official_metrics(b,ids,[0,3],gt,[0,3]);assert abs(x['m_vIoU']-y['m_vIoU'])<1e-6 and x['vIoU@0.5']==1
 z=f(b,ids,[4,5],gt,[0,3]);assert z['m_vIoU']==z['m_tIoU']==0

def test_normalized_boxes_convert_to_original_pixel_space():
 np.testing.assert_allclose(xyxy([[.5,.5,.2,.4]],100,200),[[40,60,60,140]])

def test_source_macro_clusters_shared_source_orders():
 r=[dict(parent=0,order='a',delta_v=0.),dict(parent=0,order='b',delta_v=1.),dict(parent=1,order='a',delta_v=0.)]
 x=source_summary(r,['delta_v']);assert x['sources']==2 and x['metrics']['delta_v']['mean']==.25
