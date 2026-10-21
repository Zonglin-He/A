"""CPU-only evaluation, strictly after all64 cached fusion outputs seal."""
import argparse,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.tube_fusion.run import TRIAL,METHODS,read,sha,atomic,validate
from scripts.tube_temporal import run as old
from scripts.tube_temporal.score import FIELDS,interval_metrics,_numbers
from scripts.state_opd.evaluate import score_boxes,mean_table
PAIRS=(('GlobalTube6','Source_Native224'),('Global6','Source_Native224'),('GlobalPermuted6','Source_Native224'),
       ('GlobalTube6','Global6'),('GlobalTube6','GlobalPermuted6'),('GlobalTube6','Strong_Reference'),('Global6','Strong_Reference'))

def paired(rows,key):
    assert len(rows)==len({r['parent']for r in rows})==64
    draws=np.random.default_rng(20261021).integers(0,64,(10000,64));out={}
    for a,b in PAIRS:
        out[a+'-'+b]={}
        for field,(unit,scale,higher)in FIELDS.items():
            good=[r for r in rows if r[key][a][field]is not None and r[key][b][field]is not None]
            if not good:continue
            d=scale*np.asarray([r[key][a][field]-r[key][b][field]for r in good]);n=len(d)
            ix=draws if n==64 else np.random.default_rng(20261021).integers(0,n,(10000,n))
            x=dict(mean=float(d.mean()),CI95=np.quantile(d[ix].mean(1),[.025,.975]).tolist(),queries=n,unit=unit,
                increased=int((d>1e-10).sum()),decreased=int((d < -1e-10).sum()),unchanged=int((abs(d)<=1e-10).sum()))
            if higher is not None:
                sign=d if higher else-d;x.update(improved=int((sign>1e-10).sum()),harmed=int((sign < -1e-10).sum()),higher_is_better=higher)
            if higher and unit=='pp':x['declined_over5pp']=int((d < -5).sum())
            out[a+'-'+b][field]=x
    return out

def means(rows,key):
    m,_=mean_table([dict(r,values=r[key])for r in rows],METHODS)
    for name in METHODS:
        for field in('start_error_seconds','end_error_seconds','interval_length_seconds','GT_length_seconds','length_error_seconds','length_ratio_to_GT'):
            m[name][field]=float(np.mean([r[key][name][field]for r in rows]))
    return m

def signed(rows):
    result=[]
    for i,row in enumerate(rows):
        for key in('values','fixed_spatial'):
            for a,b in PAIRS:
                first,second=row[key][a],row[key][b];dv=100*(first['v']-second['v']);dt=100*(first['t']-second['t'])
                result.append(dict(query=f'H{i+1:03d}',analysis=key,method=a,baseline=b,vIoU_delta_pp=dv,tIoU_delta_pp=dt,
                    boundary_MAE_delta_seconds=first['boundary_MAE_seconds']-second['boundary_MAE_seconds'],
                    interval_length_delta_seconds=first['interval_length_seconds']-second['interval_length_seconds'],
                    severe_vIoU_decline_over5pp=dv < -5,severe_tIoU_decline_over5pp=dt < -5,
                    lost_tIoU_half=second['t']>=.5 and first['t']<.5,recovered_tIoU_half=second['t']<.5 and first['t']>=.5,
                    newly_missed_event=second['t']>0 and first['t']==0))
    return result

def render(p):
    lines=['# HC64: global and tube-conditioned temporal posterior fusion','',
        'All64 predictions seal before GT. Source_Native224 is the unadapted Source spatial output and its native final-layer temporal MBR. '
        'Global6, GlobalTube6 and GlobalPermuted6 share unchanged original BoxFixed geometry/score/DP. Strong is a historical reference, not an acceptance gate. '
        'Each of six layer logits is normalized and mapped to the original physical interval axis separately before averaging. Fusion is fixed .5/.5.','',
        '| Method | Full vIoU % | tIoU % | Boundary MAE s | Predicted length s | Unobserved sIoU % |','|---|---:|---:|---:|---:|---:|']
    for name in METHODS:
        m=p['means'][name];lines.append(f'| {name} | {100*m["m_vIoU"]:.4f} | {100*m["m_tIoU"]:.4f} | {m["boundary_MAE_seconds"]:.4f} | {m["interval_length_seconds"]:.4f} | {100*m["m_sIoU_unobserved_actual"]:.4f} |')
    lines+=['','| Comparison | vIoU pp [95% CI] | tIoU pp [95% CI] | Boundary MAE seconds [95% CI] | v improve / harm / same | v decline >5pp |','|---|---:|---:|---:|---:|---:|']
    def ci(v):return f'{v["mean"]:+.4f} [{v["CI95"][0]:.4f},{v["CI95"][1]:.4f}]'
    for a,b in PAIRS:
        x=p['paired'][a+'-'+b];v=x['v'];lines.append(f'| {a} - {b} | {ci(v)} | {ci(x["t"])} | {ci(x["boundary_MAE_seconds"])} | {v["improved"]}/{v["harmed"]}/{v["unchanged"]} | {v["declined_over5pp"]} |')
    lines+=['','## Identical spatial trajectory, different temporal intervals','',
        'Every row below uses the same historical A BoxFixed boxes. These are pure temporal results, including the Source interval row; they are not the overall unadapted Source baseline.','',
        '| Interval source | Fixed-space vIoU % | tIoU % | Boundary MAE s |','|---|---:|---:|---:|']
    for name in METHODS:
        m=p['fixed_spatial_means'][name];lines.append(f'| {name} | {100*m["m_vIoU"]:.4f} | {100*m["m_tIoU"]:.4f} | {m["boundary_MAE_seconds"]:.4f} |')
    lines+=['','## Cost boundary','',p['costs']['scope'],'',
        'No new RGB/encoder/decoder/ReferDINO calls or updates. Six original head batches per query read G/T/P at two offsets; all final-layer logits/posteriors/intervals recover previous results bitwise. '
        'The real deployment of a conditioned branch still requires two offset temporal decoder calls, plus native224, expert evidence, propagation, IO and final DP. '
        'This cached validation wall is not deployed latency. Strong remains a reference only. All signed and severe outcomes, parent paired intervals and costs remain in JSON/TSV. '
        'HC64 is exposed development data. No fusion-weight scan is authorized; a nonpositive real spatial contribution ends this feedback trial.']
    return '\n'.join(lines)+'\n'

def evaluate():
    lock=read(TRIAL/'lock.json');receipts=validate(lock);seal=read(TRIAL/'sealed.json');sealsha=sha(TRIAL/'sealed.json')
    assert not torch.cuda.is_initialized()
    manifest=ROOT/'data/evaluation/development.jsonl';assert sha(manifest)==lock['manifest_sha256']
    records=[json.loads(x)for x in manifest.read_text().splitlines()if x.strip()];records=[r for r in records if r['id']in lock['ids']and r['dataset']=='hc2']
    assert [r['id']for r in records]==lock['ids']
    labels=ROOT/'data/evaluation/development_labels.jsonl';labelsha=sha(labels)
    assert labelsha==read(ROOT/'results/tube_temporal_hc64/report.json')['proof']['labels_sha256']
    truth={g['id']:g for g in [json.loads(x)for x in labels.read_text().splitlines()if x.strip()]if g['id']in lock['ids']and g['dataset']=='hc2'}
    assert set(truth)==set(lock['ids']);rows=[];error=baselineerror=0.;checks=0
    previous=read(ROOT/'results/tube_temporal_hc64/report.json')['queries_anonymous']
    for i,record in enumerate(records):
        key=record['id'];g=truth[key];boxes_gt={int(k):v for k,v in g['boxes'].items()};fps=record['metadata']['fps']
        with np.load(TRIAL/seal['files'][key]['path'],allow_pickle=False)as p:
            values={};fixed={};intervals={name:p[name+'_interval'].tolist()for name in METHODS}
            for name in METHODS:
                for collection,boxes in((values,p[name]),(fixed,p['Strong_Reference'])):
                    v,e,support=score_boxes(boxes,record,boxes_gt,intervals[name],g['interval'],p['positions8'].tolist());v.update(interval_metrics(intervals[name],g['interval'],fps));collection[name]=v;error=max(error,e);checks+=1
                if name=='Strong_Reference':
                    for field in('v','t','s','observed_actual_sIoU','unobserved_actual_sIoU'):
                        a=values[name][field];b=previous[i]['values']['A_Strong'][field]
                        assert(a is None)==(b is None)
                        if a is not None:baselineerror=max(baselineerror,abs(a-b))
                for field in('fixed_actual_sIoU','observed_actual_sIoU','unobserved_actual_sIoU'):
                    assert fixed[name][field]==fixed['Source_Native224'][field]
        rows.append(dict(id=key,parent=g['parent'],values=values,fixed_spatial=fixed,intervals=intervals,support=support))
    assert error<2e-10 and baselineerror==0 and len(rows)==len({r['parent']for r in rows})==64
    validate(lock);assert sha(labels)==labelsha and sha(TRIAL/'sealed.json')==sealsha and not torch.cuda.is_initialized()
    rc=list(receipts.values());run=read(TRIAL/'run.json')
    costs=dict(scope='Head-only cached validation. Input IO, original frozen head projection, posterior construction, three final BoxFixed DPs and engineering recovery checks are measured. Historical native224/expert/decoder work is not counted as new computation. Do not report this as deployment time.',
        current=_numbers(rc),head_by_branch={name:_numbers([r['head_costs'][name]for r in rc])for name in('G','T','P')},
        entire_run_seconds=run['seconds'],head_load_seconds=seal['head_load_seconds'],head=seal['head'],cache_hits=64,
        new_RGB_encoder_decoder_expert_updates=0,head_batches=384,
        deployment_reference='C previous measured native224 capture plus aux DP/two offset decoder replay/final DP; six-layer probability averaging is new lightweight postprocessing, not measured complete deployment here.')
    proof=dict(fingerprint=lock['fingerprint'],lock_sha256=sha(TRIAL/'lock.json'),seal_sha256=sealsha,labels_sha256=labelsha,
        old_lock_sha256=lock['old_lock_sha256'],old_seal_sha256=lock['old_seal_sha256'],all64_predictions_before_GT=True,
        last_layer_logits_posteriors_intervals_bit_exact=True,old_BoxFixed_actions_boxes_scores_bit_exact=True,official_dense_checks=checks,
        official_dense_max_absolute_error=error,Strong_reference_metric_error=baselineerror,source_native_unadapted=True,
        head_weight_updates=0,RGB_encoder_decoder_expert_calls=0,all_input_code_hashes_rechecked=True,CUDA_initialized=False)
    public=dict(status='completed_HC64_exposed_development_posterior_fusion',queries=64,parents=64,config=lock['config'],proof=proof,
        means=means(rows,'values'),paired=paired(rows,'values'),fixed_spatial_means=means(rows,'fixed_spatial'),fixed_spatial_paired=paired(rows,'fixed_spatial'),
        costs=costs,signed_changes=signed(rows),queries_anonymous=[dict(query=f'H{i+1:03d}',values=r['values'],fixed_spatial=r['fixed_spatial'])for i,r in enumerate(rows)])
    dest=ROOT/'results/tube_fusion_hc64';dest.mkdir(parents=True,exist_ok=True);assert not(dest/'report.json').exists(),'No overwrite'
    atomic(TRIAL/'metrics.json',dict(public,private_rows=rows));atomic(dest/'report.json',public);(dest/'report.md').write_text(render(public))
    fields=list(public['signed_changes'][0]);(dest/'signed_changes.tsv').write_text('\n'.join(['\t'.join(fields)]+['\t'.join(str(x[k])for k in fields)for x in public['signed_changes']])+'\n')
    atomic(TRIAL/'score_receipt.json',dict(status='scored',proof=proof,public_json_sha256=sha(dest/'report.json'),signed_sha256=sha(dest/'signed_changes.tsv')))
    return public

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-after-seal',action='store_true');a=p.parse_args()
    if not a.run_after_seal:p.error('Explicit --run-after-seal required')
    out=evaluate();print(json.dumps(dict(status=out['status'],means=out['means'],paired=out['paired']),indent=2))
