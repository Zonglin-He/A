"""Independent CPU scoring of the fixed intervention and frozen domain rechecks."""
import argparse,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.interval_pool import intervention,confirmation
from scripts.tube_temporal.run import read,sha,atomic
from stvg.metrics import official,DenseTube,box_iou
from stvg.metrics_base import xyxy
from stvg.evaluation import add_reporting_metrics,finite_mean,main_table
from scripts.tube_temporal.score import interval_metrics,FIELDS

def score_boxes(boxes,record,truth,interval,span,positions):
    ds=record['dataset'];ids=record['metadata']['frame_ids'];meta=record['metadata'];row=dict(input=meta,frame_ids=ids)
    values=official(boxes,row,truth,span,interval,ds);other=DenseTube(boxes,row,truth,span,clip=ds=='hc2').score(interval)
    error=max(abs(values[k]-other[k])for k in('v','t','s'));assert error<2e-10
    pixels=xyxy(boxes,meta['width'],meta['height']);pixels=np.maximum(pixels,0)if ds=='hc2'else pixels
    observed={ids[int(p)]for p in positions};lookup={f:i for i,f in enumerate(ids)};fixed=[f for f in ids if f in truth]
    for name,frames in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',[f for f in fixed if f in observed]),('unobserved_actual_sIoU',[f for f in fixed if f not in observed])]:
        values[name]=float(np.mean([box_iou(pixels[lookup[f]],truth[f])for f in frames]))if frames else None
    values['boundary_MAE']=(abs(interval[0]-span[0])+abs(interval[1]-span[1]))/2
    values=add_reporting_metrics(values,meta);values.update(interval_metrics(interval,span,meta['fps']));return values,error

def summarize(rows,names,key):
    means={a:{f:finite_mean([r[key][a][f]for r in rows])for f in rows[0][key][a]}for a in names};main=main_table(means)
    for name in names:
        for f in('start_error_seconds','end_error_seconds','interval_length_seconds','length_error_seconds','length_ratio_to_GT'):main[name][f]=means[name][f]
    return main

def paired(rows,pairs,key):
    assert len(rows)==len({r['parent']for r in rows})
    out={}
    for a,b in pairs:
        out[a+'-'+b]={}
        for f,(unit,scale,higher)in FIELDS.items():
            d=np.asarray([scale*(r[key][a][f]-r[key][b][f])for r in rows if r[key][a][f]is not None and r[key][b][f]is not None]);n=len(d)
            if not n:continue
            ix=np.random.default_rng(20261021).integers(0,n,(10000,n));sign=d if higher else-d
            v=dict(queries=n,mean=float(d.mean()),CI95=np.quantile(d[ix].mean(1),[.025,.975]).tolist(),unit=unit,
                increased=int((d>1e-10).sum()),decreased=int((d < -1e-10).sum()),unchanged=int((abs(d)<=1e-10).sum()))
            if higher is not None:v.update(improved=int((sign>1e-10).sum()),harmed=int((sign < -1e-10).sum()))
            if higher and unit=='pp':v['declined_over5pp']=int((d < -5).sum())
            out[a+'-'+b][f]=v
    return out

def signed(rows,pairs,prefix):
    out=[]
    for i,r in enumerate(rows):
        for key in('values','fixed_spatial'):
            for a,b in pairs:
                x,y=r[key][a],r[key][b];dv=100*(x['v']-y['v']);dt=100*(x['t']-y['t'])
                out.append(dict(query=f'{prefix}{i+1:03d}',analysis=key,method=a,baseline=b,vIoU_delta_pp=dv,tIoU_delta_pp=dt,
                    boundary_MAE_delta_seconds=x['boundary_MAE_seconds']-y['boundary_MAE_seconds'],
                    severe_vIoU_decline_over5pp=dv < -5,severe_tIoU_decline_over5pp=dt < -5,
                    lost_tIoU_half=y['t']>=.5 and x['t']<.5,recovered_tIoU_half=y['t']<.5 and x['t']>=.5,
                    newly_missed_event=y['t']>0 and x['t']==0))
    return out

def costs(receipts):
    def aggregate(items):
        keys=sorted({k for x in items for k,v in x.items()if isinstance(v,(int,float))and not isinstance(v,bool)})
        return {k:dict(mean=float(np.mean([x[k]for x in items if k in x])),median=float(np.median([x[k]for x in items if k in x])),total=float(sum(x[k]for x in items if k in x)))for k in keys}
    rr=list(receipts.values());out=dict(top=aggregate(rr))
    for field in('capture_cost','head_cost','posterior_cost'):
        if field in rr[0]:out[field]=aggregate([r[field]for r in rr])
    if'branch_cost'in rr[0]:out['branches']={m:aggregate([r['branch_cost'][m]for r in rr])for m in('Hull_MBR','Mix_MBR')}
    elif'cost'in rr[0]:out['branches']={m:aggregate([r['cost'][m]for r in rr])for m in rr[0]['cost']}
    out['scope']='Stage times may be nested. Posterior construction is compared on shared logits; no end-to-end multiplier inferred. Recheck has one shared224/eight-frame expert per query; cached intervention has no new network. Head/IO/MBR/DP and full capture are separate.'
    return out

def render(p):
    lines=[f'# {p["scope"]}: fixed physical interval pooling','',p['exposure'],'',
        '| Method | Full vIoU % | tIoU % | Boundary MAE s | Unobserved sIoU % |','|---|---:|---:|---:|---:|']
    for name,m in p['means'].items():
        si=m['m_sIoU_unobserved_actual'];s='—'if si is None else f'{100*si:.4f}'
        lines.append(f'| {name} | {100*m["m_vIoU"]:.4f} | {100*m["m_tIoU"]:.4f} | {m["boundary_MAE_seconds"]:.4f} | {s} |')
    lines+=['','| Comparison | vIoU pp [95% CI] | tIoU pp [95% CI] | v improve/harm/same | v decline >5pp |','|---|---:|---:|---:|---:|']
    for name,row in p['paired'].items():
        v,t=row['v'],row['t'];fmt=lambda x:f'{x["mean"]:+.4f} [{x["CI95"][0]:.4f},{x["CI95"][1]:.4f}]'
        lines.append(f'| {name} | {fmt(v)} | {fmt(t)} | {v["improved"]}/{v["harmed"]}/{v["unchanged"]} | {v["declined_over5pp"]} |')
    lines+=['','## Same frozen spatial trajectory','',
        '| Interval source | Fixed-space vIoU % | tIoU % |','|---|---:|---:|']
    for name,m in p['fixed_spatial_means'].items():lines.append(f'| {name} | {100*m["m_vIoU"]:.4f} | {100*m["m_tIoU"]:.4f} |')
    lines+=['','Source_Native224 above is the raw Source overall baseline. Its fixed-space row uses the common BoxFixed boxes and is only a temporal analysis. MAP is diagnostic only; main comparison is Mix_MBR−Hull_MBR. Strong is historical reference, not a gate. All signed/severe changes and costs remain in JSON/TSV.','',p['costs']['scope']]
    return'\n'.join(lines)+'\n'

def evaluate(scope):
    if scope=='development':
        tr=intervention.TRIAL;receipts=intervention.validate();lock=read(tr/'lock.json');seal=read(tr/'sealed.json');ids=lock['ids'];names=intervention.METHODS
        manifest=ROOT/'data/evaluation/development.jsonl';label=ROOT/'data/evaluation/development_labels.jsonl';expected=read(ROOT/'results/tube_fusion_hc64/report.json')['proof']['labels_sha256'];ds='hc2';prefix='H';n=64
        pairs=[('Mix_MBR','Hull_MBR'),('Mix_MAP','Hull_MAP'),('Hull_MBR','Hull_MAP'),('Mix_MBR','Mix_MAP'),('Mix_MBR','Source_Native224'),('Hull_MBR','Source_Native224')]
        exposure='HC64 repeatedly exposed development panel. Fixed 2x2 readout; MAP is diagnostic. All predictions sealed before labels.'
    else:
        ds=scope;tr=confirmation.BASE/('recheck_'+ds);receipts=confirmation.validate(ds);lock=read(confirmation.BASE/'confirmation_lock.json');seal=read(tr/'sealed.json')
        select=read(confirmation.BASE/'confirmation_selection.json')['development_recheck_selection'][ds];ids=[r['query_id']for r in select];names=confirmation.METHODS
        manifest=ROOT/select[0]['manifest_path'];job=next(j for j in read(ROOT/'outputs/formal/plan.json')['jobs']if j['id']=='tastvg-'+ds);label=ROOT/job['labels'];expected=job['labels_sha256'];prefix='C'if ds=='hc2'else'V';n=32
        pairs=[('Mix_MBR','Hull_MBR'),('Mix_MBR','Source_Native224'),('Hull_MBR','Source_Native224')]
        exposure='Frozen parent/video-isolated DEVELOPMENT RECHECK, not historically unexposed independent confirmation. Both domains have documented full-domain historical exposure; selected parents exclude the current128 panel, not all retired older panels.'
    assert len(receipts)==n and seal['GT_read']is False and not torch.cuda.is_initialized();sealsha=sha(tr/'sealed.json')
    records={r['id']:r for r in [json.loads(s)for s in manifest.read_text().splitlines()if s.strip()]if r['dataset']==ds and r['id']in ids};assert set(records)==set(ids)
    labelsha=sha(label);assert labelsha==expected
    truth={g['id']:g for g in [json.loads(s)for s in label.read_text().splitlines()if s.strip()]if g['id']in ids and g['dataset']==ds};assert set(truth)==set(ids)
    rows=[];maxerror=0.;checks=0
    for key in ids:
        record=records[key];g=truth[key];gt={int(k):v for k,v in g['boxes'].items()};v={};fixed={}
        with np.load(tr/seal['files'][key]['path'],allow_pickle=False)as p:
            for name in names:
                interval=p[name+'_interval'].tolist()
                for collection,boxes in((v,p[name]),(fixed,p['FixedSpatial'])):
                    collection[name],e=score_boxes(boxes,record,gt,interval,g['interval'],p['positions8'].tolist());maxerror=max(maxerror,e);checks+=1
                for f in('fixed_actual_sIoU','observed_actual_sIoU','unobserved_actual_sIoU'):assert fixed[name][f]==fixed['Source_Native224'][f]
        rows.append(dict(id=key,parent=g['parent'],values=v,fixed_spatial=fixed))
    assert len({r['parent']for r in rows})==n and sha(label)==labelsha and sha(tr/'sealed.json')==sealsha
    intervention.validate()if scope=='development'else confirmation.validate(ds)
    proof=dict(all_predictions_sealed_before_GT=True,seal_sha256=sealsha,labels_sha256=labelsha,official_dense_checks=checks,official_dense_max_absolute_error=maxerror,CUDA_initialized=False)
    public=dict(status='completed',scope=scope,dataset=ds,queries=n,parents=n,exposure=exposure,proof=proof,
        means=summarize(rows,names,'values'),paired=paired(rows,pairs,'values'),fixed_spatial_means=summarize(rows,names,'fixed_spatial'),fixed_spatial_paired=paired(rows,pairs,'fixed_spatial'),
        costs=costs(receipts),signed_changes=signed(rows,pairs,prefix),queries_anonymous=[dict(query=f'{prefix}{i+1:03d}',values=r['values'],fixed_spatial=r['fixed_spatial'])for i,r in enumerate(rows)])
    dest=ROOT/'results/interval_pool'/('intervention_hc64'if scope=='development'else'recheck_'+scope);dest.mkdir(parents=True,exist_ok=True);assert not(dest/'report.json').exists(),'No overwrite'
    atomic(tr/'metrics.json',dict(public,private_rows=rows));atomic(dest/'report.json',public);(dest/'report.md').write_text(render(public));fields=list(public['signed_changes'][0])
    (dest/'signed_changes.tsv').write_text('\n'.join(['\t'.join(fields)]+['\t'.join(str(x[k])for k in fields)for x in public['signed_changes']])+'\n')
    atomic(tr/'score_receipt.json',dict(status='scored',proof=proof,public_report_sha256=sha(dest/'report.json')));return public

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scope',choices=['development','hc2','vidstg'],required=True);a=p.parse_args();r=evaluate(a.scope);print(json.dumps(dict(scope=a.scope,means=r['means'],paired=r['paired']['Mix_MBR-Hull_MBR']),indent=2))
