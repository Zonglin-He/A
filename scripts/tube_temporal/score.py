"""Independent CPU evaluation of the fully sealed four-mode temporal trial.

Every mode uses its own interval and final BoxFixed trajectory. A second,
explicit analysis holds A's spatial trajectory fixed across all intervals.
No input, prediction, old scored result or label is read at module import.
"""
from __future__ import annotations
import argparse,itertools,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.tube_temporal.run import TRIAL,METHODS,REFERENCE,read,sha,atomic,validate
from scripts.state_opd.evaluate import score_boxes,mean_table

PAIRS=(('C_TubeReplay','B_Native224'),('C_TubeReplay','D_Permuted'),('B_Native224','A_Strong'),
       ('C_TubeReplay','A_Strong'),('D_Permuted','A_Strong'),('D_Permuted','B_Native224'))
FIELDS={'v':('pp',100.,True),'t':('pp',100.,True),'observed_actual_sIoU':('pp',100.,True),
    'unobserved_actual_sIoU':('pp',100.,True),'tIoU_ge_05':('pp',100.,True),'tIoU_eq_0':('pp',100.,False),
    'boundary_MAE_seconds':('seconds',1.,False),'start_error_seconds':('seconds',1.,False),
    'end_error_seconds':('seconds',1.,False),'interval_length_seconds':('seconds',1.,None),
    'length_error_seconds':('seconds',1.,False)}

def interval_metrics(interval,truth,fps):
    assert isinstance(fps,(float,int))and not isinstance(fps,bool)and np.isfinite(fps)and fps>0
    a,b=map(float,interval);s,e=map(float,truth);assert b>a and e>s
    return dict(start_error_seconds=abs(a-s)/fps,end_error_seconds=abs(b-e)/fps,
        boundary_MAE_seconds=(abs(a-s)+abs(b-e))/(2*fps),interval_length_seconds=(b-a)/fps,
        GT_length_seconds=(e-s)/fps,length_error_seconds=abs((b-a)-(e-s))/fps,
        length_ratio_to_GT=(b-a)/(e-s))

def paired(rows,kind='values'):
    assert len(rows)==len({r['parent']for r in rows})==64
    rng=np.random.default_rng(20261021);draws={};result={}
    for first,second in PAIRS:
        fields={}
        for field,(unit,scale,higher)in FIELDS.items():
            valid=[r for r in rows if r[kind][first][field]is not None and r[kind][second][field]is not None]
            n=len(valid)
            if not n:fields[field]=dict(queries=0,mean=None,CI95=None,unit=unit);continue
            delta=scale*np.asarray([r[kind][first][field]-r[kind][second][field]for r in valid],float)
            if n not in draws:draws[n]=rng.integers(0,n,(10000,n))
            row=dict(queries=n,mean=float(delta.mean()),CI95=np.quantile(delta[draws[n]].mean(-1),[.025,.975]).tolist(),unit=unit,
                increased=int((delta>1e-10).sum()),decreased=int((delta < -1e-10).sum()),unchanged=int((abs(delta)<=1e-10).sum()))
            if higher is not None:
                beneficial=delta if higher else-delta
                row.update(improved=int((beneficial>1e-10).sum()),harmed=int((beneficial < -1e-10).sum()),higher_is_better=higher)
            if unit=='pp'and higher:row['declined_over5pp']=int((delta < -5).sum())
            fields[field]=row
        result[first+'-'+second]=fields
    return result

def means(rows,kind='values'):
    standard,_=mean_table([dict(row,values=row[kind])for row in rows],METHODS)
    for name in METHODS:
        for field in ('interval_length_seconds','GT_length_seconds','length_error_seconds','length_ratio_to_GT','start_error_seconds','end_error_seconds'):
            valid=[row[kind][name][field]for row in rows if row[kind][name][field]is not None]
            standard[name][field]=float(np.mean(valid))if valid else None
    return standard

def signed(rows):
    result=[]
    for i,row in enumerate(rows):
        for first,second in PAIRS:
            for analysis,key in [('final_spatial_and_time','values'),('fixed_A_spatial_time_only','fixed_spatial')]:
                a,b=row[key][first],row[key][second];dv=100*(a['v']-b['v']);dt=100*(a['t']-b['t'])
                result.append(dict(query=f'H{i+1:03d}',analysis=analysis,method=first,baseline=second,
                    vIoU_delta_pp=dv,tIoU_delta_pp=dt,boundary_MAE_delta_seconds=a['boundary_MAE_seconds']-b['boundary_MAE_seconds'],
                    interval_length_delta_seconds=a['interval_length_seconds']-b['interval_length_seconds'],
                    severe_vIoU_decline_over5pp=dv < -5,severe_tIoU_decline_over5pp=dt < -5,
                    lost_tIoU_half=b['t']>=.5 and a['t']<.5,recovered_tIoU_half=b['t']<.5 and a['t']>=.5,
                    newly_missed_event=b['t']>0 and a['t']==0))
    return result

def _numbers(rows):
    fields=sorted({key for row in rows for key,value in row.items()if isinstance(value,(int,float))and not isinstance(value,bool)})
    result={}
    for field in fields:
        values=[float(row[field])for row in rows if field in row and isinstance(row[field],(int,float))and not isinstance(row[field],bool)]
        assert np.isfinite(values).all()
        result[field]=dict(queries=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p90=float(np.quantile(values,.9)),total=float(np.sum(values)))
    return result

def cost_report(receipts):
    rows=list(receipts.values());branches={method:_numbers([r['costs'][method]for r in rows])for method in METHODS[1:]}
    shared=_numbers([dict(r['capture_cost'],**r['timings'])for r in rows]);components={}
    for method in METHODS[1:]:
        values=[dict(component_total_excluding_output_IO_seconds=r['timings']['input_cache_IO_seconds']+r['timings']['RGB_decode_seconds']+
            r['capture_cost']['capture_wall_seconds']+r['costs'][method]['finish_wall_seconds'])for r in rows]
        components[method]=_numbers(values)
    return dict(shared_current_capture=shared,branch_current_finish=branches,component_totals=components,
        actual_multi_branch_trial=_numbers([dict(total_seconds=r['total_seconds'],peak_GPU_bytes=r['peak_GPU_bytes'])for r in rows]),
        cache_hits={key:dict(hits=sum(bool(r['cache_hits'][key])for r in rows),misses=sum(not bool(r['cache_hits'][key])for r in rows))for key in rows[0]['cache_hits']},
        A_reference='Predictions are reused; this run does not remeasure a64-query Strong end-to-end latency. Historical A timings are not treated as matched current costs.',
        scope='Shared224 capture is measured once per query; component totals add that measured shared cost to each measured branch finish, excluding shared output-cache IO. Multi-branch trial wall includes all B/C/D work and IO; it is not one deployed method latency. Expert evidence hits and new native224 misses are reported separately. No predeclared speedup multiplier is assumed.')

def validate_before_labels(trial=TRIAL):
    trial=Path(trial).resolve();assert trial==TRIAL.resolve();lock=read(trial/'lock.json');receipts=validate(trial,lock)
    seal=read(trial/'sealed.json');assert seal['queries']==64 and seal['GT_read']is False
    manifest=ROOT/'data/evaluation/development.jsonl';assert sha(manifest)==lock['manifest_sha256']
    records=[json.loads(line)for line in manifest.read_text().splitlines()if line.strip()];records=[r for r in records if r['dataset']=='hc2'][:64]
    assert [r['id']for r in records]==lock['ids']
    for record in records:
        fps=record['metadata'].get('fps');assert isinstance(fps,(int,float))and not isinstance(fps,bool)and np.isfinite(fps)and fps>0
    assert not torch.cuda.is_initialized()
    proof=dict(fingerprint=lock['fingerprint'],lock_sha256=sha(trial/'lock.json'),seal_sha256=sha(trial/'sealed.json'),
        manifest_sha256=sha(manifest),legacy=lock['legacy'],all64_predictions_sources_and_inputs_before_labels=True,
        GT_used_for_intervals_or_spatial_choices=False,A_reused_exact=True,metric_sources=lock['metric_sources'],metric_helpers=lock['metric_helpers'],
        new_448_forwards=0,new_expert_forwards=0,parameter_updates=0,CUDA_initialized=False)
    return trial,lock,seal,receipts,records,proof

def _render(public):
    lines=['# HC64: BoxFixed with native and tube-guided temporal decoding','',
        'A is the exact previously sealed BoxFixed+Strong reference. B, C and D share one fresh224 native capture and the same cached '
        'eight selected ReferDINO boxes/masks. B uses native224 endpoints; C replays the cached temporal decoder once using the '
        'full-timeline auxiliary tube; D uses the fixed permuted spatial prior. There is no448 call, new expert call, OPD or optimizer '
        'in the new paths. All64 predictions are sealed before labels.','',
        '| Method | Full vIoU % | tIoU % | Boundary MAE s ↓ | Predicted length s | Unobserved sIoU % |',
        '|---|---:|---:|---:|---:|---:|']
    for name in METHODS:
        m=public['means'][name]
        lines.append(f'| {name} | {100*m["m_vIoU"]:.4f} | {100*m["m_tIoU"]:.4f} | {m["boundary_MAE_seconds"]:.4f} | '
            f'{m["interval_length_seconds"]:.4f} | {100*m["m_sIoU_unobserved_actual"]:.4f} |')
    lines+=['','| Comparison | vIoU delta pp [95% CI] | tIoU delta pp [95% CI] | Boundary MAE delta s [95% CI] | vIoU improve / harm / same | vIoU drops >5 pp |',
        '|---|---:|---:|---:|---:|---:|']
    def ci(value):return f'{value["mean"]:+.4f} [{value["CI95"][0]:.4f}, {value["CI95"][1]:.4f}]'
    for first,second in PAIRS:
        row=public['paired'][first+'-'+second];v=row['v']
        lines.append(f'| {first} − {second} | {ci(v)} | {ci(row["t"])} | {ci(row["boundary_MAE_seconds"])} | '
            f'{v["improved"]} / {v["harmed"]} / {v["unchanged"]} | {v["declined_over5pp"]} |')
    lines+=['','Negative boundary-MAE differences are improvements. All intervals retain their original physical frame/offset '
        'mapping, with boundary errors and lengths divided by the explicit input FPS. Paired intervals use10000 parent-query draws. '
        'Full signed positive/negative/unchanged and severe outcomes remain in JSON/TSV. A confidence interval spanning zero does '
        'not establish equal accuracy.','',
        '## Pure temporal contribution with A spatial trajectory fixed','',
        '| Interval source | Fixed-A full vIoU % | tIoU % | Boundary MAE s |',
        '|---|---:|---:|---:|']
    for name in METHODS:
        m=public['fixed_spatial_means'][name];lines.append(f'| {name} | {100*m["m_vIoU"]:.4f} | {100*m["m_tIoU"]:.4f} | {m["boundary_MAE_seconds"]:.4f} |')
    lines+=['','This second analysis uses exactly the same saved A box trajectory for every interval, including its existing '
        'fallback boxes outside the old interval. It generates no additional prediction and separates interval effects from the '
        'final interval-conditioned BoxFixed re-selection. All fixed-node spatial IoUs must remain identical in this analysis.','',
        '## Actual cost and reuse boundaries','',public['costs']['scope'],'',public['costs']['A_reference'],'',
        'Per-stage224/expert/evidence/auxiliary DP/extra decoder/postprocess/final DP and IO statistics are retained in report.json. '
        'Source model loading and transfers remain measured components. The existing-query raw-input production smoke, including '
        'its expert cache miss and Strong reference, is separate acceptance evidence; it is not silently averaged into this cache-hit panel.','',
        'Tube replay remains experimental. C−B tests whether tube feedback helps; C−D tests sensitivity to spatial placement. '
        'This exposed-development comparison does not validate other backbone adapters.']
    return '\n'.join(lines)+'\n'

def evaluate(trial=TRIAL):
    trial,lock,seal,receipts,records,proof=validate_before_labels(trial)
    labels_path=ROOT/'data/evaluation/development_labels.jsonl';labelsha=sha(labels_path)
    assert labelsha==read(ROOT/'outputs/layout_reward_dev/scope_hc2/metrics.json')['labels_sha256']
    labels={r['id']:r for r in [json.loads(line)for line in labels_path.read_text().splitlines()if line.strip()]if r['dataset']=='hc2'and r['id']in lock['ids']}
    assert set(labels)==set(lock['ids'])
    previous=read(ROOT/'results/box_propagation_hc64/report.json')['queries_anonymous'];rows=[];maxerror=baselineerror=0.;checks=0
    for i,record in enumerate(records):
        key=record['id'];g=labels[key];truth={int(k):v for k,v in g['boxes'].items()};fps=record['metadata']['fps']
        with np.load(trial/seal['files'][key]['path'],allow_pickle=False)as p:
            values={};fixed={};positions=p['positions8'].tolist();intervals={m:p[m+'_interval'].tolist()for m in METHODS}
            for name in METHODS:
                for collection,boxes in [(values,p[name]),(fixed,p['A_Strong'])]:
                    collection[name],error,support=score_boxes(boxes,record,truth,intervals[name],g['interval'],positions)
                    collection[name].update(interval_metrics(intervals[name],g['interval'],fps));maxerror=max(maxerror,error);checks+=1
                old=previous[i]['values']['BoxFixed']
                if name=='A_Strong':
                    for field in ('v','t','s','observed_actual_sIoU','unobserved_actual_sIoU'):
                        if old[field]is None:assert values[name][field]is None
                        else:baselineerror=max(baselineerror,abs(values[name][field]-old[field]))
            for name in METHODS:
                for field in ('fixed_actual_sIoU','observed_actual_sIoU','unobserved_actual_sIoU'):
                    assert fixed[name][field]==fixed['A_Strong'][field]
                assert values[name]['t']==fixed[name]['t']
        rows.append(dict(id=key,parent=g['parent'],values=values,fixed_spatial=fixed,intervals=intervals,support=support))
    assert len(rows)==len({r['parent']for r in rows})==64 and maxerror<2e-10 and baselineerror<2e-10
    assert sha(labels_path)==labelsha;validate(trial,lock)
    assert sha(trial/'lock.json')==proof['lock_sha256']and sha(trial/'sealed.json')==proof['seal_sha256']and not torch.cuda.is_initialized()
    proof.update(labels_sha256=labelsha,official_dense_checks=checks,official_dense_max_absolute_error=maxerror,
        legacy_A_metric_max_absolute_error=baselineerror,all_hashes_rechecked_after_scoring=True,
        pure_time_analysis_uses_identical_spatial_trajectory=True)
    changes=signed(rows);public=dict(status='completed_HC64_exposed_development_tube_temporal',queries=64,parents=64,
        method_order=list(METHODS),means=means(rows),paired=paired(rows),fixed_spatial_means=means(rows,'fixed_spatial'),
        fixed_spatial_paired=paired(rows,'fixed_spatial'),costs=cost_report(receipts),signed_changes=changes,proof=proof,config=lock['config'],
        queries_anonymous=[dict(query=f'H{i+1:03d}',values=r['values'],fixed_spatial=r['fixed_spatial'])for i,r in enumerate(rows)])
    dest=ROOT/'results/tube_temporal_hc64';dest.mkdir(parents=True,exist_ok=True);assert not(dest/'report.json').exists(),'No overwrite'
    atomic(trial/'metrics.json',dict(public,private_rows=rows));atomic(dest/'report.json',public);(dest/'report.md').write_text(_render(public))
    fields=list(changes[0]);(dest/'signed_changes.tsv').write_text('\n'.join(['\t'.join(fields)]+['\t'.join(str(r[f])for f in fields)for r in changes])+'\n')
    atomic(trial/'score_receipt.json',dict(status='scored',proof=proof,metrics_sha256=sha(trial/'metrics.json'),public_json_sha256=sha(dest/'report.json'),
        public_md_sha256=sha(dest/'report.md'),signed_changes_sha256=sha(dest/'signed_changes.tsv')))
    return public

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-after-seal',action='store_true');a=p.parse_args()
    if not a.run_after_seal:p.error('Explicit --run-after-seal and complete64 seal required')
    result=evaluate();print(json.dumps(dict(status=result['status'],means=result['means']),indent=2))
