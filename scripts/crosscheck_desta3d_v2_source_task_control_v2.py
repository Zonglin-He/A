"""Second tensor geometry, pairwise event ranks and parent bootstrap calculation."""
import sys,time,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_source_fit_v1 import _tube_metric
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
RUN=ROOT/'artifacts/desta3d_v2/tta_v2/source_task_control_v2'
OUT=RUN/'independent_readback_v1'
ARMS=['no_update','unlabeled','supervised'];MS=['vIoU','sIoU','tIoU']

def main():
    torch.set_num_threads(2);done=read(OUT/'COMPLETE.json')
    assert sha(OUT/'REPORT.json')==done['report_sha']
    report=read(OUT/'REPORT.json');assert report['audit_sha']==sha(OUT/'PRE_SCORE_AUDIT.json')
    for p,h in read(RUN/'ALL_PREDICTIONS_SEAL.json')['pins'].items():assert sha(Path(p))==h
    lp=RUN/'SOURCE_RECORDS.json';assert sha(lp)==report['source_label_sha']
    labels={r['key']:r for r in read(lp)};rows=read(RUN/'INPUTS.json');errors=[];aucs=[];parents={};ret={}
    for a in ARMS:
        values={};metric_by_key={}
        for i,row in enumerate(rows):
            p=torch.load(RUN/'episodes'/f'{i:02}'/(a+'.pt'),map_location='cpu',weights_only=False)
            label=labels[row['key']];m=_tube_metric(p,label)
            expected=next(r for r in report['arms'][a]['query_rows'] if r['key']==row['key'])
            for k in MS:
                e=abs(m[k]-expected['metrics'][k]);errors.append(e);assert e<2e-6
            assert m['format_ok']==expected['metrics']['format_ok']
            y=np.asarray(label['event_active'],bool);x=p['event_logits'].numpy().ravel()
            auc=None
            if y.any() and (~y).any():
                diff=x[y,None]-x[None,~y];auc=float(np.mean((diff>0)+.5*(diff==0)))
            assert (auc is None)==(expected['event_AUROC'] is None)
            if auc is not None:aucs.append(abs(auc-expected['event_AUROC']));assert aucs[-1]<1e-12
            values[str(row['source'])]=np.array([m[k] for k in MS]);metric_by_key[row['key']]=m
        parents[a]=values;ret[a]=metric_by_key
        macro=np.mean(list(values.values()),axis=0)
        assert np.max(np.abs(macro-np.array([report['arms'][a]['summary']['parent_macro'][k] for k in MS])))<2e-6
    cierr=0.;tailchecks=0
    for name,c in report['comparisons'].items():
        a,b=name.split('_minus_');keys=sorted(parents[a]);d=np.array([parents[a][k]-parents[b][k] for k in keys])
        rng=np.random.default_rng(20260927);idx=rng.integers(0,len(keys),(10000,len(keys)))
        ci=np.quantile(d[idx].mean(axis=1),[.025,.975],axis=0)*100
        for j,k in enumerate(MS):
            e=float(np.max(np.abs(ci[:,j]-c[k]['bootstrap_ci95_pp'])));cierr=max(cierr,e);assert e<2e-4
            assert int((d[:,j]<-.05).sum())==c[k]['severe_loss_below_minus5pp'];tailchecks+=1
            assert abs(float(d[:,j].mean()*100)-c[k]['mean_delta_pp'])<2e-4
    for j,k in enumerate(['vIoU','tIoU']):
        good={q for q,m in ret['no_update'].items() if m[k]>.5};assert len(good)==report['retention'][k]['eligible']
        for a in ARMS:
            lost=sorted(q for q in good if ret[a][q][k]<=.5)
            assert lost==sorted(report['retention'][k]['arms'][a]['lost'])
    result={'time':time.time(),'status':'passed','report_sha':sha(OUT/'REPORT.json'),
        'tensor_geometry_comparisons':len(errors),'max_geometry_error':max(errors),'max_event_AUROC_error':max(aucs,default=0.),
        'max_parent_CI_error_pp':cierr,'tail_comparisons':tailchecks,'native_good_v_t_retention_recomputed':True,
        'source_training_diagnostic_only':True,'target_inputs_or_GT_read':False,'script_sha':sha(Path(__file__))}
    save_once(RUN/'ROOT_SCORE_CROSSCHECK.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
