"""Postseal spatial degradation decomposition; no fitting or parameter selection.

Only the already scored confirmation cohort is inspected. Actual GT is used
offline at exactly observed anchor frames, never interpolated into a reference.
"""
from pathlib import Path
import sys,json,math,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_lite_event_support_v1/confirmation'
OUT=ROOT/'artifacts/stvg_temporal_evidence_budget_v1/spatial_diagnosis'

def iou(a,b):
    import numpy as np
    a=np.asarray(a,float);b=np.asarray(b,float);lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);inter=float(np.maximum(hi-lo,0).prod())
    union=float(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter)
    return inter/union if union>0 else 0.

def main():
    import numpy as np,torch,statistics
    from scripts.stvg_evidence_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    cpu=read(BASE/'complete/CPU_COMPLETION.json');assert cpu['status']=='complete'and cpu['count']==64 and cpu['GT_read']is True
    for name,h in cpu['output_sha256'].items():assert sha(BASE/'complete'/name)==h
    pred=read(BASE/'complete/GLOBAL_PREDICTION_BARRIER.json');assert pred['GT_read']is False
    for rel,h in pred['bindings'].items():assert sha(ROOT/rel)==h
    roster=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in roster}
    # Prediction seal and independent ROOT have been verified before this read.
    targets,gtbindings=truth._truths_after_seal(cases)
    raw=read(BASE/'complete/POSTSEAL_ROWS.json')['rows'];idx={(r['case_id'],r['arm']):r for r in raw}
    b=BASE/'trials/C_FULL';bar=read(b/'STUDENT_GLOBAL_BARRIER.json');rows=[]
    for ds in ('hc2','vidstg'):
        rr=sorted([r for r in roster if r['dataset']==ds],key=lambda r:r['probe_alias'])
        for j,rec in enumerate(rr,1):
            cid=rec['probe_alias'];case=cases[cid];entry=next(f for f in bar['files']if f['case_id']==cid);assert sha(ROOT/entry['path'])==entry['sha256']
            d=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=False)
            assert d['source_case_sha256']==case['source_case_sha256']and not d['GT_read']
            assert d['arms']['T']['interval']==d['arms']['TS_M2']['interval']
            assert np.array_equal(d['arms']['T']['boxes'],case['native_boxes_xyxy_normalized'])
            before=np.asarray(d['arms']['T']['boxes']);after=np.asarray(d['arms']['TS_M2']['boxes']);anchors=d['reference_records']['anchors']
            target=metric._truth_map(targets[cid]);width,height=case['input']['width'],case['input']['height'];known=[];agreement=[]
            for a in anchors:
                pos=a['position'];fid=int(case['frame_ids'][pos]);assert fid==a['physical_frame_id']
                cx,cy,w,h=a['box_cxcywh'];ref=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
                agreement.append(dict(before=iou(before[pos],ref),after=iou(after[pos],ref)))
                if fid not in target:continue
                gt=np.asarray(target[fid])/np.array([width,height,width,height])
                known.append(dict(source_iou=iou(before[pos],gt),reference_iou=iou(ref,gt),adapted_iou=iou(after[pos],gt)))
            t=idx[cid,'Temporal'];f=idx[cid,'Full'];m=idx[cid,'Matched_original_OPD'];assert t['metrics']['m_tIoU']==f['metrics']['m_tIoU']
            def avg(v,k):return math.fsum(x[k]for x in v)/len(v)if v else None
            sourceq=avg(known,'source_iou');refq=avg(known,'reference_iou');adaptq=avg(known,'adapted_iou')
            val=dict(case_id=('CHC'if ds=='hc2'else'CVID')+f'{j:03}',dataset=ds,query_type=rec['query_type'],
                metric_delta_pp={k:100*(f['metrics'][k]-t['metrics'][k])if f['metrics'][k]is not None and t['metrics'][k]is not None else None for k in f['metrics']},
                full_minus_original_opd_vIoU_pp=100*(f['metrics']['m_vIoU']-m['metrics']['m_vIoU']),
                exact_same_output_interval=True,anchor_count=len(anchors),known_GT_anchor_count=len(known),unknown_GT_anchor_count=len(anchors)-len(known),
                known_anchor_source_iou=sourceq,known_anchor_reference_iou=refq,known_anchor_adapted_iou=adaptq,
                reference_minus_source_iou_pp=100*(refq-sourceq)if known else None,
                anchor_adapted_minus_source_iou_pp=100*(adaptq-sourceq)if known else None,
                all_anchor_teacher_agreement_change_pp=100*(avg(agreement,'after')-avg(agreement,'before'))if agreement else None,
                reference_quality_status='unknown_no_exact_GT_anchor'if anchors and not known else'no_reference'if not anchors else'mixed_partial_exact_GT_anchor_support'if len(known)<len(anchors)else'exact_GT_anchor_support',
                per_known_anchor=known,optimization_steps=d['fits']['M2']['optimizer_steps']if d['fits']['M2']else 0)
            rows.append(val)
    summary=[]
    for ds in ('hc2','vidstg'):
        for kind in ('all','question','statement'):
            sub=[r for r in rows if r['dataset']==ds and (kind=='all'or ('question'in r['query_type']or'interrog'in r['query_type'])==(kind=='question'))]
            if not sub:continue
            vals=[r['metric_delta_pp']['m_vIoU']for r in sub];negative=[r for r in sub if r['metric_delta_pp']['m_vIoU']< -1e-6]
            summary.append(dict(dataset=ds,query_group=kind,count=len(sub),full_minus_temporal_vIoU_pp=math.fsum(vals)/len(vals),
                temporal_delta_all_exact_zero=all(r['metric_delta_pp']['m_tIoU']==0 for r in sub),positive=sum(v>1e-6 for v in vals),negative=len(negative),neutral=sum(abs(v)<=1e-6 for v in vals),
                negative_with_worse_known_reference=sum(r['reference_minus_source_iou_pp']is not None and r['reference_minus_source_iou_pp']<0 for r in negative),
                negative_with_better_known_reference=sum(r['reference_minus_source_iou_pp']is not None and r['reference_minus_source_iou_pp']>0 for r in negative),
                negative_without_exact_GT_anchor=sum(r['known_GT_anchor_count']==0 for r in negative),
                negative_while_teacher_agreement_improves=sum(r['all_anchor_teacher_agreement_change_pp']is not None and r['all_anchor_teacher_agreement_change_pp']>0 for r in negative),
                dense_sIoU_delta_pp=math.fsum(r['metric_delta_pp']['m_sIoU_dense_GT']for r in sub)/len(sub)))
    result=dict(status='complete',count=64,rows=rows,summary=summary,GT_read=True,new_fit=False,new_model=False,no_hyperparameter_selection=True,
        scope='exact same output intervals and exact physical GT anchors only; unknown anchors not interpolated or imputed; reference IoU is geometric quality, not instance identity proof',
        prediction_seal_sha256=sha(BASE/'complete/GLOBAL_PREDICTION_BARRIER.json'),source_score_sha256=sha(BASE/'complete/POSTSEAL_ROWS.json'),GT_bindings=gtbindings)
    write(OUT/'POSTSEAL_DIAGNOSIS.json',result)
    print(json.dumps({'status':'complete','summary':summary},ensure_ascii=False))

if __name__=='__main__':main()
