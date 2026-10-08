"""Synthetic CPU metric/aggregation contracts, using no research data or model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.audit_decota_paper_baseline_scoring_v1 import metric,independent_statistics
from scripts.score_decota_paper_baseline_scoring_v1 import statistics,FIELDS
from vg_tta.tastvg_oracle_event5_v1 import DenseTube
from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy
from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric

def run():
    checks=0;rng=np.random.default_rng(81)
    for clip,official in [(False,DenseMetric()),(True,HC2DenseMetric())]:
        for variant in range(8):
            ids=[2,5,8,11];row={'frame_ids':ids,'input':{'width':200,'height':100}}
            boxes=rng.uniform(.1,.9,(4,4));boxes[:,2:]*=.6
            if variant==1:boxes[:,:2]=-.05
            gt={f:[25.,15.,125.,75.] for f in range(0,14)}
            span=[1,13];iv=[[0,14],[2,10],[1,2],[13,15],[3,4],[0,1],[7,13],[1,13]][variant]
            a=DenseTube(boxes,row,gt,span,clip).score(iv);b=metric(boxes,row,gt,span,iv,clip)
            corners=xyxy(boxes,200,100)
            if clip:corners=np.maximum(corners,0)
            c=official(corners,ids,iv,gt,span)
            for k,f in [('v','m_vIoU'),('t','m_tIoU'),('s','sIoU_dense_GT')]:
                assert abs(a[k]-b[k])<2e-12 and abs(b[k]-c[f])<2e-12;checks+=2
    # Exact endpoint and no-extrapolation behavior on perfect sampled boxes.
    row={'frame_ids':[2,5],'input':{'width':100,'height':100}}
    boxes=np.array([[.5,.5,1.,1.],[.5,.5,1.,1.]])
    z=metric(boxes,row,{f:[0.,0.,100.,100.] for f in range(7)},[0,6],[0,6],False)
    assert z=={'v':4/6,'t':1.,'s':4/7};checks+=3
    assert float(.3>.3)==0 and float(.5>.5)==0;checks+=2
    rows=[]
    for query,source,base in [(0,0,.1),(1,0,.3),(2,1,.8)]:
        for order in range(1,4):
            r=dict(query_id=query,source_id=source,order=f'order{order}')
            r.update({f:base+order*.001 for f in FIELDS});rows.append(r)
    a=statistics(rows);b=independent_statistics(rows,FIELDS)
    assert abs(a['metrics']['After_v']['source_macro']-.502)<1e-12
    assert abs(a['metrics']['After_v']['query_macro']-.402)<1e-12
    for j,f in enumerate(FIELDS):
        s=a['metrics'][f]
        assert np.allclose([s['source_macro'],s['query_macro']],[b['source_macro'][j],b['query_macro'][j]],rtol=0,atol=1e-12)
        assert np.allclose(s['ci95_source'],b['ci'][:,j],rtol=0,atol=1e-12)
        assert np.allclose(s['ci95_query_clustered'],b['query_ci'][:,j],rtol=0,atol=1e-12);checks+=6
    print(json.dumps(dict(status='pass',checks=checks,synthetic_only=True,GT_read=False,model_or_GPU_run=False)))

if __name__=='__main__':run()
