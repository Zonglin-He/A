import torch
from scripts.score_desta3d_v3_source_fit import summarize,compare,choose_epoch,score_tube_independently,_tube_metric

def test_parent_domain_macro_and_paired_identical_CI():
    rows=[]
    for domain,parent,count,value in [('Vid','x',9,.2),('Vid','y',1,.8),('HC1','h',1,1.)]:
        rows.extend(dict(domain=domain,parent=parent,metrics={'a':dict(vIoU=value,sIoU=value,tIoU=value)}) for _ in range(count))
    s=summarize(rows,'a');assert s['equal_domain_parent_macro']['vIoU']==.75
    c=compare(s,s,100);assert c['vIoU']['delta_pp']==0 and c['vIoU']['CI95_pp']==[0.,0.]

def test_patience_ties_and_no_premature_stop():
    def report(x):return dict(summary={'trained':{'equal_domain_parent_macro':{'vIoU':x}},'Frozen':{'equal_domain_parent_macro':{'vIoU':.5}}})
    assert choose_epoch([report(.7),report(.7)])['stop'] is False
    d=choose_epoch([report(.7),report(.7),report(.6)]);assert d['stop'] and d['best_epoch']==0 and d['mean_gate_vs_Frozen']
    assert choose_epoch([report(.4),report(.5),report(.6)])['stop'] is False

def test_scalar_tensor_physical_sparse_invalid_support():
    label=dict(frame_ids=[10,20,30],boxes_xyxy=[[0.,0.,1.,1.]]*3,box_valid=[True]*3,event_active=[True]*3,event_interval={'begin_fid':10,'end_fid':31})
    pred=dict(frame_ids=[10,20,30],boxes_cxcywh=torch.tensor([[.5,.5,1.,1.],[.5,.5,1.,1.]]),geometry_valid=torch.tensor([True,False]),positions=[0,2],interval=[0,2],format_ok=True)
    a=score_tube_independently(pred,label);b=_tube_metric(pred,label)
    assert abs(a['vIoU']-1/3)<1e-9
    assert max(abs(a[k]-b[k]) for k in ['vIoU','sIoU','tIoU'])<1e-6
    pred['format_ok']=False;assert score_tube_independently(pred,label)['vIoU']==0
