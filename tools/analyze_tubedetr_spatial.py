"""Offline spatial diagnosis of complete development candidates; no GPU or tuning.

Labels are opened only after all selected prediction seals/score receipts are
verified. Never reads partial full_selected runs or chooses samples by outcome.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write
from stvg.metrics import DenseTube,box_iou
from stvg.metrics_base import xyxy


def mean(values):
    data=[float(v)for v in values if v is not None]
    return float(np.mean(data)) if data else None


def signed_changes(rows,a,b,key):
    # Pair query denominators; never fill an undefined observed metric with zero.
    values=[100*(r['values'][a][key]-r['values'][b][key])for r in rows
            if r['values'][a][key] is not None and r['values'][b][key] is not None]
    return dict(queries=len(values),mean_pp=mean(values),positive=sum(v>1e-10 for v in values),
                negative=sum(v< -1e-10 for v in values),neutral=sum(abs(v)<=1e-10 for v in values),
                below_minus5=sum(v< -5 for v in values))


def joint_outcomes(rows):
    counters=dict(observed_up_unobserved_down=0,observed_down_unobserved_up=0,
                  spatial_down_volume_up=0,spatial_up_volume_down=0,spatial_up_volume_up=0)
    valid=0
    for r in rows:
        v=r['values'];s=v['Full']['m_sIoU']-v['Temporal']['m_sIoU']
        volume=v['Full']['m_vIoU']-v['Temporal']['m_vIoU']
        counters['spatial_down_volume_up']+=int(s< -1e-12 and volume>1e-12)
        counters['spatial_up_volume_down']+=int(s>1e-12 and volume< -1e-12)
        counters['spatial_up_volume_up']+=int(s>1e-12 and volume>1e-12)
        obs=[v[a]['observed_actual_sIoU']for a in ['Full','Temporal']]
        unobs=[v[a]['unobserved_actual_sIoU']for a in ['Full','Temporal']]
        if any(x is None for x in obs+unobs):continue
        valid+=1;o=obs[0]-obs[1];u=unobs[0]-unobs[1]
        counters['observed_up_unobserved_down']+=int(o>1e-12 and u< -1e-12)
        counters['observed_down_unobserved_up']+=int(o< -1e-12 and u>1e-12)
    return dict(queries=len(rows),paired_observed_unobserved_queries=valid,**counters)


def pareto_front(profiles):
    # Descriptive only: frozen selection remains full128 m_vIoU, not this frontier.
    keys=['m_vIoU','m_sIoU','unobserved_actual_sIoU']
    front=[]
    for name,p in profiles.items():
        x=p['means']['Full']
        dominated=False
        for other,q in profiles.items():
            if name==other:continue
            y=q['means']['Full']
            if all(y[k]>=x[k] for k in keys)and any(y[k]>x[k]for k in keys):dominated=True;break
        if not dominated:front.append(name)
    return sorted(front)


def candidate_inputs(dataset):
    base=ROOT/f'outputs/tubedetr_{"vid" if dataset=="vidstg" else "hc2"}_development'
    plan=ROOT/f'results/tubedetr_{"vid" if dataset=="vidstg" else "hc2"}_development/plan.json'
    entries=[]
    for folder in sorted((base/'candidates').glob('p_*')):
        metrics=folder/'metrics.json';receipt=folder/'score_receipt.json';seal=folder/'sealed.json'
        if not all(p.exists()for p in [metrics,receipt,seal]):continue
        score=json.loads(receipt.read_text());sealed=json.loads(seal.read_text())
        if score['metrics_sha256']!=sha(metrics)or score['seal_sha256']!=sha(seal):
            raise ValueError('Candidate score receipt changed')
        if sealed['queries']!=128 or len(sealed['files'])!=128:raise ValueError('Not a complete dev128')
        for e in sealed['files'].values():
            if sha(folder/e['path'])!=e['sha256']or sha(folder/e['receipt_path'])!=e['receipt_sha256']:
                raise ValueError('Candidate prediction bytes changed')
        result=json.loads(metrics.read_text())
        if result['manifest_sha256']!=json.loads(plan.read_text())['manifest_sha256']:
            raise ValueError('Analysis would mix development panels')
        if set(sealed['files'])!={row['id']for row in result['rows']}:raise ValueError('Unpaired candidate rows')
        entries.append((folder,sealed,result))
    return base,plan,entries


def analyze(dataset):
    base,plan,entries=candidate_inputs(dataset)
    # No full_selected paths enter this analysis. Open labels after seal checks.
    label_path=ROOT/f'data/evaluation/tubedetr_{"vid" if dataset=="vidstg" else "hc2"}_dev128_labels.jsonl'
    meta_path=label_path.with_name(label_path.name.replace('_labels',''))
    if sha(label_path)!=json.loads(plan.read_text())['labels_sha256']:raise ValueError('Labels changed')
    labels={r['id']:r for r in map(json.loads,label_path.read_text().splitlines())}
    inputs={r['id']:r for r in map(json.loads,meta_path.read_text().splitlines())}
    profiles={};private=[]
    for folder,sealed,metrics in entries:
        profiles[folder.name]=dict(configuration=sealed['identity']['configuration'],
            metrics_sha256=sha(folder/'metrics.json'),prediction_seal_sha256=sha(folder/'sealed.json'),
            means=metrics['datasets'][dataset]['query_means'],joint=joint_outcomes(metrics['rows']),
            changes={a+'-'+b:{key:signed_changes(metrics['rows'],a,b,key)for key in
                ['m_vIoU','m_sIoU','observed_actual_sIoU','unobserved_actual_sIoU']}for a,b in
                [('WithoutOPD','Temporal'),('Full','WithoutOPD'),('Full','Temporal')]})
        distance={name:{arm:[]for arm in ['Source','E_Ref','Live']}for name in ['observed','within_0_5s','0_5_to_1s','beyond_1s']}
        teachers=[];teacher_groups={'teacher_better':[],'teacher_not_better':[]};details=[]
        for row in metrics['rows']:
            key=row['id'];entry=sealed['files'][key];truth=labels[key];meta=inputs[key]['metadata']
            gt={int(f):b for f,b in truth['boxes'].items()};span=truth['interval']
            with np.load(folder/entry['path'],allow_pickle=False)as z:
                ids=z['frame_ids'];positions=z['positions8'].astype(int);observed=ids[positions]
                dense={a:DenseTube(z[a],dict(input=meta,frame_ids=ids),gt,span,clip=dataset=='hc2')for a in ['Source','E_Ref','Live']}
                d=np.min(np.abs(dense['Source'].fids[:,None]-observed[None,:]),axis=1)/meta['fps']
                masks={'observed':d==0,'within_0_5s':(d>0)&(d<=.5),'0_5_to_1s':(d>.5)&(d<=1),'beyond_1s':d>1}
                per_dist={}
                for name,mask in masks.items():
                    if not mask.any():continue
                    per_dist[name]={a:float(value.iou[mask].mean())for a,value in dense.items()}
                    for a,v in per_dist[name].items():distance[name][a].append(v)
                cache=base/'teacher_cache'/f'{key}.npz';teacher=None
                if cache.exists():
                    m=json.loads(cache.with_suffix('.json').read_text())
                    if sha(cache)!=m['sha256']:raise ValueError('Expert cache changed')
                    with np.load(cache,allow_pickle=False)as t:boxes=t['boxes'].copy()
                    expected=json.loads((folder/entry['receipt_path']).read_text())['initialization']['evidence_sha256']
                    actual=hashlib.sha256(np.asarray(boxes,dtype=np.float64).tobytes()).hexdigest()
                    if actual!=expected:raise ValueError('Teacher evidence does not match candidate')
                    valid=[i for i,f in enumerate(observed)if int(f)in gt]
                    if valid:
                        pix=xyxy(boxes,meta['width'],meta['height']);src=xyxy(z['Source'][positions],meta['width'],meta['height'])
                        if dataset=='hc2':pix=np.maximum(pix,0);src=np.maximum(src,0)
                        ground=np.array([gt[int(observed[i])]for i in valid])
                        teacher=dict(valid_GT_observations=len(valid),teacher_IoU=float(box_iou(pix[valid],ground).mean()),
                            source_IoU=float(box_iou(src[valid],ground).mean()))
                        teachers.append(teacher)
                        group='teacher_better'if teacher['teacher_IoU']>teacher['source_IoU']else'teacher_not_better'
                        teacher_groups[group].append(row)
                details.append(dict(id=key,teacher=teacher,distance=per_dist))
        profile=profiles[folder.name]
        profile['distance_to_nearest_observation_seconds']= {name:dict(queries=len(values['Source']),
            means={arm:mean(v)for arm,v in values.items()},
            OPD_sIoU_pp=mean([100*(a-b)for a,b in zip(values['Live'],values['E_Ref'])]),
            complete_sIoU_pp=mean([100*(a-b)for a,b in zip(values['Live'],values['Source'])]))for name,values in distance.items()}
        profile['no_observed_GT_queries']=dict(queries=sum(r['values']['Full']['observed_actual_sIoU']is None for r in metrics['rows']),
            OPD_dense=signed_changes([r for r in metrics['rows']if r['values']['Full']['observed_actual_sIoU']is None],'Full','WithoutOPD','m_sIoU'),
            OPD_volume=signed_changes([r for r in metrics['rows']if r['values']['Full']['observed_actual_sIoU']is None],'Full','WithoutOPD','m_vIoU'))
        profile['teacher_at_observed_GT']=dict(queries=len(teachers),teacher_IoU=mean([t['teacher_IoU']for t in teachers]),
            source_IoU=mean([t['source_IoU']for t in teachers]),
            posthoc_groups={g:dict(queries=len(rows),joint=joint_outcomes(rows),
                OPD_dense=signed_changes(rows,'Full','WithoutOPD','m_sIoU'),
                OPD_observed=signed_changes(rows,'Full','WithoutOPD','observed_actual_sIoU'),
                OPD_unobserved=signed_changes(rows,'Full','WithoutOPD','unobserved_actual_sIoU'))for g,rows in teacher_groups.items()})
        private.append(dict(profile=folder.name,rows=details))
    result=dict(status='complete_offline_analysis_of_scored_candidates'if entries else'waiting_for_scored_candidates',
        dataset=dataset,development_queries=128,profiles=profiles,descriptive_pareto_profiles=pareto_front(profiles),
        selected_profile_unchanged=True,new_parameter_evaluation=False,no_new_GPU_work=True,
        not_independent_confirmation=True,GT_groups_not_used_for_query_gating=True,
        dense_distance_policy='all annotated dense frames, original no-extrapolation; per-query mean then equal-query mean within each bin',
        observed_policy='actual eight positions intersecting GT; null excluded with explicit query denominator',
        formal_selection_objective_unchanged='Full equal-query m_vIoU; spatial Pareto front is descriptive only')
    dest=ROOT/f'results/tubedetr_spatial_analysis/{dataset}.json';write(dest,result)
    write(ROOT/f'outputs/tubedetr_spatial_analysis/{dataset}_private.json',private)
    lines=[f'# TubeDETR {dataset} 空间诊断','',
        '仅完整封存且已评分的128开发候选；不读取新全量的部分输出，不更改选定配置，不作query级GT门控。','',
        '| profile | gamma | lr | tau | steps | m_vIoU | m_sIoU | 观测sIoU | 未观测sIoU |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,p in profiles.items():
        c=p['configuration'];v=p['means']['Full']
        lines.append('| '+name+' | '+' | '.join(str(c[k])for k in ['initialization_gamma','lr','tau','steps'])+' | '+
            ' | '.join(f'{100*v[k]:.5f}'if v[k]is not None else'null'for k in ['m_vIoU','m_sIoU','observed_actual_sIoU','unobserved_actual_sIoU'])+' |')
    lines+=['','空间—完整定位Pareto前沿仅作诊断：'+', '.join(result['descriptive_pareto_profiles']),
        '分距离、教师与Source观测质量及正负query统计见同名JSON。128已用于选参，不能将微小优势解释为独立验证。','']
    dest.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps(dict(dataset=dataset,profiles=len(profiles),descriptive_pareto_profiles=result['descriptive_pareto_profiles'])))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['hc2','vidstg'],required=True)
    analyze(p.parse_args().dataset)
