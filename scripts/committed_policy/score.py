"""Independent CPU scoring after the complete committed-policy HC64 seal.

No labels or cached assets are read at import. All64 actual fits must be sealed
and source/input hashes validated before the offline score entry reads GT.
There is no diagnostic gate before adaptation or outcome-selected query scope.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch

ROOT=Path(os.environ.get('STVG_PROJECT_ROOT',Path(__file__).resolve().parents[2])).expanduser().resolve()
sys.path.insert(0,str(ROOT))
from scripts.state_opd.evaluate import read,sha,write_json,score_boxes,mean_table,paired_stats,cost_summary
from scripts.evidence_policy.score import final_node_changes,_aggregate_nodes
from scripts.box_propagation.score import _shared_cost_summary

REUSED=['ERefStrong','GeometryOPD','BoxFixed','B_OPD']
CONTROLS=['CommitInitial','CommitDP']
ARMS=['CommitOPD','CommitFixedKD','CommitDirect']
NAMES=REUSED+CONTROLS+ARMS
TRIAL=ROOT/'outputs/committed_policy_hc64'


def focus_pairs():
    pairs=[(name,baseline)for name in ARMS for baseline in ('ERefStrong','GeometryOPD','BoxFixed','B_OPD','CommitInitial','CommitDP')]
    pairs.extend([('CommitOPD','CommitDirect'),('CommitOPD','CommitFixedKD'),('CommitDP','CommitInitial')])
    return list(dict.fromkeys(pairs))


def replacement_counts(final,E,active,observed):
    final=np.asarray(final);E=np.asarray(E);active=np.asarray(active,bool);observed=np.asarray(observed,bool)
    assert final.shape==E.shape==(len(active),4)and observed.shape==active.shape
    replaced=(final!=E).any(-1)
    assert not(replaced&~active).any()
    result={}
    for name,mask in [('all',np.ones(len(active),bool)),('observed',observed),('unobserved',~observed),
                      ('active_unobserved',active&~observed),('inactive',~active)]:
        result[name]=dict(nodes=int(mask.sum()),replaced_nodes=int((mask&replaced).sum()))
    return result


def public_statistics(value):
    """Avoid publishing private node vectors or serializing raw Torch tensors."""
    if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
    if isinstance(value,np.ndarray)and value.ndim>0:
        numbers=np.asarray(value,float).reshape(-1);assert np.isfinite(numbers).all()
        return dict(count=len(numbers),mean=float(numbers.mean())if len(numbers)else None,
            minimum=float(numbers.min())if len(numbers)else None,maximum=float(numbers.max())if len(numbers)else None,
            p10=float(np.quantile(numbers,.1))if len(numbers)else None,p50=float(np.quantile(numbers,.5))if len(numbers)else None,
            p90=float(np.quantile(numbers,.9))if len(numbers)else None,
            positive=int((numbers>0).sum()),zero=int((numbers==0).sum()),negative=int((numbers<0).sum()))
    if isinstance(value,(np.generic,np.ndarray)):value=value.item()
    if isinstance(value,dict):return {str(k):public_statistics(v)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [public_statistics(v)for v in value]
    assert value is None or isinstance(value,(str,int,float,bool))
    assert not isinstance(value,float)or np.isfinite(value)
    return value


def signed_changes(rows):
    result=[]
    for index,row in enumerate(rows):
        for method,baseline in focus_pairs():
            first,second=row['values'][method],row['values'][baseline]
            v=100*(first['v']-second['v'])
            s=None if first['unobserved_actual_sIoU']is None or second['unobserved_actual_sIoU']is None else 100*(first['unobserved_actual_sIoU']-second['unobserved_actual_sIoU'])
            result.append(dict(query=f'H{index+1:03d}',method=method,baseline=baseline,vIoU_delta_pp=v,
                unobserved_sIoU_delta_pp=s,severe_vIoU_drop_over5pp=v < -5,
                severe_unobserved_sIoU_drop_over5pp=s is not None and s < -5))
    return result


def route_counts(path):
    active=np.asarray(path['active'],bool);observed=np.asarray(path['observed'],bool)
    result={}
    events={name:np.asarray(path[name],bool)for name in ('commit_event','switch_event','reset_event')}
    assert all(value.shape==active.shape for value in events.values())
    mass=np.asarray(path['commit_mass'],float);assert mass.shape==active.shape and np.isfinite(mass).all()
    for group,mask in [('all',np.ones(len(active),bool)),('observed',observed),('unobserved',~observed),('active_unobserved',active&~observed)]:
        result[group]=dict(nodes=int(mask.sum()),**{name:int((event&mask).sum())for name,event in events.items()},
            commit_mass_mean=float(mass[mask].mean())if mask.any()else None)
    return result


def teacher_suggestions(teacher_iou,student_iou,E_iou,teacher_actions,student_actions,active,observed):
    teacher=np.asarray(teacher_iou,float);student=np.asarray(student_iou,float);E=np.asarray(E_iou,float)
    teacher_actions=np.asarray(teacher_actions);student_actions=np.asarray(student_actions)
    active=np.asarray(active,bool);observed=np.asarray(observed,bool)
    finite=np.isfinite(teacher);assert all(np.array_equal(np.isfinite(value),finite)for value in (student,E))
    assert all(value.shape==active.shape for value in (teacher,student,E,teacher_actions,student_actions,observed))
    result={}
    for group,mask in [('all',finite),('active',finite&active),('observed',finite&active&observed),('unobserved',finite&active&~observed)]:
        proposed=mask&(teacher_actions!=student_actions);delta=teacher-student
        result[group]=dict(GT_nodes=int(mask.sum()),suggested_replacement_nodes=int(proposed.sum()),
            teacher_minus_student_IoU_mean=float(delta[mask].mean())if mask.any()else None,
            teacher_minus_E_IoU_mean=float((teacher-E)[mask].mean())if mask.any()else None,
            replacement_IoU_delta_mean=float(delta[proposed].mean())if proposed.any()else None,
            harmed_suggested_replacements=int((proposed&(delta < -1e-12)).sum()),
            severe_harm_over5pp_suggested_replacements=int((proposed&(delta < -.05)).sum()),
            improved_suggested_replacements=int((proposed&(delta>1e-12)).sum()),
            harmful_replacement_fraction=float((proposed&(delta < -1e-12)).sum()/proposed.sum())if proposed.any()else None)
    return result


def _validate_fit(output):
    costs=output['costs'];rounds=output['rounds']
    assert np.asarray(output['initialtheta']).shape==(4,)and(np.asarray(output['initialtheta'])==0).all()
    assert len(rounds)==3 and costs['updates']==costs['backward']==30
    assert costs['state_only_backwards']==30
    assert costs['accepted_outer']+costs['rejected_outer']==3
    assert all(costs[name]==0 for name in ('heavy_forwards','decoder_calls','decoder_backwards','encoder_calls','expert_calls'))
    for row in rounds:
        assert len(row['inner'])==10
        assert all(np.isfinite(np.asarray(inner['gradient'])).all()for inner in row['inner'])
        assert row['after_score']>=row['before_score']-1e-12
        if row['rolled_back']:
            assert np.array_equal(np.asarray(row['before']),np.asarray(row['after']))
            assert row['optimizer_before_sha256']==row['optimizer_after_sha256']
    path=output['trace']['final_rollout']
    assert path['teacherq']is not None
    assert len(output['trace']['training_paths'])==3 and all(len(outer)==10 and all(len(paths)==2 for paths in outer)for outer in output['trace']['training_paths'])
    return dict(route=route_counts(path),rounds=public_statistics(rounds),
        final_raw_weights=public_statistics(output['finaltheta']),final_effective_weights=public_statistics(output['finalweights']))


def validate_before_labels(trial=TRIAL):
    from scripts.committed_policy.run import validate
    trial=Path(trial).resolve();assert trial==TRIAL.resolve()
    lock=read(trial/'lock.json');receipts=validate(trial,lock);seal=read(trial/'sealed.json')
    assert seal['queries']==64 and seal['GT_read']is False and seal['fingerprint']==lock['fingerprint']
    assert seal['lock_sha256']==sha(trial/'lock.json')and len(receipts)==len(seal['files'])==64
    manifest=ROOT/'data/evaluation/development.jsonl';assert sha(manifest)==lock['manifest_sha256']
    records=[json.loads(line)for line in manifest.read_text().splitlines()if line.strip()]
    records=[row for row in records if row['dataset']=='hc2'];assert len(records)==128
    records=records[:64];assert [row['id']for row in records]==lock['ids']
    statistics=[];shared=[]
    previous=ROOT/'outputs/evidence_policy_hc64';previous_seal=read(previous/'sealed.json')
    for record in records:
        key=record['id'];r=receipts[key];entry=seal['files'][key]
        for field,hashfield in [('path','sha256'),('receipt_path','receipt_sha256')]:
            path=(trial/entry[field]).resolve();assert path.is_relative_to(trial)and sha(path)==entry[hashfield]
        assert r['GT_read']is False
        path=(trial/r['trace_path']).resolve();assert path.is_relative_to(trial)and sha(path)==r['trace_sha256']
        trace=torch.load(path,map_location='cpu',weights_only=False)
        extra_path=(trial/r['extra_path']).resolve();assert extra_path.is_relative_to(trial)and sha(extra_path)==r['extra_sha256']
        assert torch.load(extra_path,map_location='cpu',weights_only=False)['GT_read']is False
        with np.load(trial/entry['path'],allow_pickle=False)as p,np.load(previous/previous_seal['files'][key]['path'],allow_pickle=False)as old:
            assert set(p.files)==set(NAMES)|{'frame_ids','positions8','Strong_interval'}
            assert p['frame_ids'].tolist()==record['metadata']['frame_ids']
            for field in (*REUSED,'frame_ids','positions8','Strong_interval'):assert np.array_equal(p[field],old[field])
            for name in NAMES:
                assert p[name].shape==p['ERefStrong'].shape and np.isfinite(p[name]).all()and(p[name][:,2:]>0).all()
            for name in CONTROLS:assert np.array_equal(p[name],np.asarray(trace[name]['final']))
            current={}
            for name in ARMS:
                output=trace[name];assert np.array_equal(p[name],np.asarray(output['final']))and output['costs']==r['costs'][name]
                current[name]=_validate_fit(output)
                path=output['trace']['final_rollout']
                current[name]['replacements']=replacement_counts(p[name],p['ERefStrong'],path['active'],path['observed'])
                assert np.array_equal(np.asarray(output['trace']['initial_rollout']['boxes']),p['CommitInitial'])
            statistics.append(current)
        shared.append(r.get('timings',{})|({'total_seconds':r['total_seconds']}if'total_seconds'in r else{}))
    assert not torch.cuda.is_initialized()
    proof=dict(fingerprint=lock['fingerprint'],lock_sha256=sha(trial/'lock.json'),seal_sha256=sha(trial/'sealed.json'),
        manifest_sha256=sha(manifest),all64_sources_inputs_fit_outputs_and_receipts_validated_before_labels=True,
        all64_three_fits_completed_before_GT=True,no_diagnostic_GO_gate_before_fit=True,reused_baselines_exact=True,
        GT_used_for_online_update_or_query_selection=False,model_calls=0,decoder_calls=0,expert_calls=0,RGB_reads=0,CUDA_initialized=False)
    return trial,lock,seal,receipts,records,statistics,shared,proof


def _node_iou(boxes,record,truth):
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    meta=record['metadata'];pixels=np.maximum(xyxy(np.asarray(boxes),meta['width'],meta['height']),0)
    return np.asarray([box_iou(box,truth[frame])if frame in truth else np.nan for frame,box in zip(meta['frame_ids'],pixels)])


def _optimization_summary(statistics):
    result={}
    for name in ARMS:
        rows=[query[name]for query in statistics];groups={};replacements={}
        for group in ('all','observed','unobserved','active_unobserved'):
            values=[row['route'][group]for row in rows];nodes=sum(row['nodes']for row in values)
            groups[group]=dict(nodes=nodes,**{field:sum(row[field]for row in values)for field in ('commit_event','switch_event','reset_event')},
                commit_mass_mean=sum(row['commit_mass_mean']*row['nodes']for row in values if row['commit_mass_mean']is not None)/nodes if nodes else None)
        for group in ('all','observed','unobserved','active_unobserved','inactive'):
            replacements[group]={field:sum(row['replacements'][group][field]for row in rows)for field in ('nodes','replaced_nodes')}
        result[name]=dict(route_events=groups,replacements=replacements)
    return result


def _teacher_summary(queries):
    result={}
    for name in ARMS:
        result[name]={}
        for group in ('all','active','observed','unobserved'):
            rows=[query[name][group]for query in queries]
            fields=('GT_nodes','suggested_replacement_nodes','harmed_suggested_replacements',
                'severe_harm_over5pp_suggested_replacements','improved_suggested_replacements')
            out={field:sum(row[field]for row in rows)for field in fields}
            for field in ('teacher_minus_student_IoU_mean','teacher_minus_E_IoU_mean','replacement_IoU_delta_mean'):
                values=[row[field]for row in rows if row[field]is not None]
                out['query_macro_'+field]=float(np.mean(values))if values else None
            out['harmful_replacement_fraction']=out['harmed_suggested_replacements']/out['suggested_replacement_nodes']if out['suggested_replacement_nodes']else None
            result[name][group]=out
    return result


def _quality_summary(extras):
    fields=('verified_anchors','valid_active_alternative_slots','committed_alternative_slots','quality_or_witness_suppressed_slots')
    summary={field:sum(extra['metadata'][field]for extra in extras)for field in fields}
    summary['anchor_quality_mean']=float(np.mean([extra['metadata']['anchor_quality_mean']for extra in extras]))
    summary['total_anchors']=8*len(extras)
    summary['commit_permission_fraction']=summary['committed_alternative_slots']/summary['valid_active_alternative_slots']if summary['valid_active_alternative_slots']else None
    summary['definition']='q is leave-self-out cached consistency, not GT quality or semantic-identity correctness. Commit permission additionally requires two other witnesses, positive external support delta and unchanged bank validity/activity.'
    return summary


def _render(public):
    lines=['# HC64: committed-source policy adaptation','',
        'The same64 development parents and existing full-box candidate bank are retained. Nine route states mean E plus eight '
        'observation sources; the box bank still has three candidate slots per sampled node. All three optimizers execute on the '
        'complete panel before GT. No diagnostic gate selects whether these fits run.','',
        '| Method | Full vIoU % | tIoU % | Observed sIoU % | Unobserved sIoU % |',
        '|---|---:|---:|---:|---:|---:|']
    for name in NAMES:
        mean=public['means'][name];values=['NA'if mean[f]is None else f'{100*mean[f]:.4f}'for f in ('m_vIoU','m_tIoU','m_sIoU_observed_actual','m_sIoU_unobserved_actual')]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines+=['','| Comparison | vIoU delta pp [95% CI] | Improve / decline / same | Drops >5 pp | Unobserved sIoU delta pp [95% CI] |',
        '|---|---:|---:|---:|---:|']
    def ci(value):return 'NA'if value['mean_pp']is None else f'{value["mean_pp"]:+.4f} [{value["CI95_pp"][0]:.4f}, {value["CI95_pp"][1]:.4f}]'
    for first,second in focus_pairs():
        row=public['paired'][first+'-'+second];v=row['v'];s=row['unobserved_actual_sIoU']
        lines.append(f'| {first} − {second} | {ci(v)} | {v["improved"]} / {v["declined"]} / {v["unchanged"]} | {v["declined_over5pp"]} | {ci(s)} |')
    lines+=['','Intervals use10000 paired parent-query draws. All signed positive, negative, unchanged and severe changes remain '
        'in report.json and signed_changes.tsv. Spatial IoU uses actual sampled nodes with GT; all eight observation positions '
        'are excluded from unobserved IoU, independently of temporal truncation. All methods use the exact saved Strong interval.','',
        '## Actual commitment, optimization and replacements','',
        '| Method | Backwards/query | Accepted outer/query | Rejected outer/query | Unobserved commit / switch / reset events | Unobserved replacements / nodes |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ARMS:
        cost=public['costs']['arms'][name];r=public['optimization'][name]['route_events']['unobserved'];b=public['optimization'][name]['replacements']['unobserved']
        lines.append(f'| {name} | {cost["backward"]["mean"]:.1f} | {cost["accepted_outer"]["mean"]:.3f} | {cost["rejected_outer"]["mean"]:.3f} | '
            f'{r["commit_event"]} / {r["switch_event"]} / {r["reset_event"]} | {b["replaced_nodes"]} / {b["nodes"]} |')
    quality=public['source_quality']
    lines+=['',f'Cached leave-self-out meanq={quality["anchor_quality_mean"]:.6f}; {quality["verified_anchors"]}/{quality["total_anchors"]} '
        f'anchors pass the quality-and-witness condition. Commit permission covers {quality["committed_alternative_slots"]}/'
        f'{quality["valid_active_alternative_slots"]} active valid alternative slots; '
        f'{quality["quality_or_witness_suppressed_slots"]} slots are suppressed by source quality or witnesses. '+quality['definition'],'',
        'CommitInitial uses four effective weights initialized to1, so it is a separate no-adaptation control and is not assumed equal to E. '
        'Each optimizer receives3 outer rounds×2 histories×10 inner backwards. Rejected outer proposals restore the raw weights and Adam, '
        'then the next predeclared outer round still runs. All quantities above come from saved real paths and update receipts.','',
        '## Teacher suggestions at the final visited state','',
        '| Method | Active unobserved suggestion−student mean IoU pp | Suggestion−E mean IoU pp | Suggested replacements | Harmful / suggested |',
        '|---|---:|---:|---:|---:|']
    for name in ARMS:
        row=public['final_teacher_suggestions'][name]['unobserved']
        def pp(field):return 'NA'if row[field]is None else f'{100*row[field]:+.4f}'
        lines.append(f'| {name} | {pp("query_macro_teacher_minus_student_IoU_mean")} | {pp("query_macro_teacher_minus_E_IoU_mean")} | '
            f'{row["suggested_replacement_nodes"]} | {row["harmed_suggested_replacements"]} / {row["suggested_replacement_nodes"]} |')
    lines+=['','These teacher proposals are evaluated only at the already saved final student histories. Their post-hoc GT IoU '
        'does not select any online action, configuration, stopping round or query. Different histories versus the previous B_OPD '
        'run prevent interpreting a teacher-quality difference as an isolated causal reliability gain. Actual net benefit is judged '
        'by the paired complete-query outcomes above.','',public['costs']['scope'],
        'No RGB, Source, ReferDINO, decoder or neural-weight update occurs in this cached-policy trial.']
    return '\n'.join(lines)+'\n'


def evaluate(trial=TRIAL):
    from scripts.committed_policy.run import validate
    trial,lock,seal,receipts,records,statistics,shared,proof=validate_before_labels(trial)
    labels_path=ROOT/'data/evaluation/development_labels.jsonl';labelsha=sha(labels_path)
    assert labelsha==read(ROOT/'outputs/layout_reward_dev/scope_hc2/metrics.json')['labels_sha256']
    labels={row['id']:row for row in [json.loads(line)for line in labels_path.read_text().splitlines()if line.strip()]
        if row['dataset']=='hc2'and row['id']in lock['ids']}
    assert set(labels)==set(lock['ids'])
    previous=read(ROOT/'results/evidence_policy_hc64/report.json')['queries_anonymous']
    rows=[];nodes={name:[]for name in ARMS};teacher_rows=[];extras=[];maxerror=baselineerror=0.;checks=0
    for index,record in enumerate(records):
        key=record['id'];g=labels[key];truth={int(k):v for k,v in g['boxes'].items()};receipt=receipts[key]
        trace=torch.load(trial/receipt['trace_path'],map_location='cpu',weights_only=False)
        bank=torch.load(ROOT/'outputs/box_propagation_hc64'/receipt['bank_path'],map_location='cpu',weights_only=False)
        extra=torch.load(trial/receipt['extra_path'],map_location='cpu',weights_only=False)
        assert extra['GT_read']is False;extras.append(extra);suggestions={}
        with np.load(trial/seal['files'][key]['path'],allow_pickle=False)as p:
            positions=p['positions8'].tolist();interval=p['Strong_interval'].tolist();values={}
            observed=np.isin(np.arange(len(p['frame_ids'])),positions);E_iou=_node_iou(p['ERefStrong'],record,truth)
            for name in NAMES:
                values[name],error,support=score_boxes(p[name],record,truth,interval,g['interval'],positions)
                maxerror=max(maxerror,error);checks+=1
                if name in REUSED:
                    old=previous[index]['values'][name]
                    for field in ('v','t','s','observed_actual_sIoU','unobserved_actual_sIoU'):
                        if old[field]is None:assert values[name][field]is None
                        else:baselineerror=max(baselineerror,abs(values[name][field]-old[field]))
                elif name in ARMS:
                    student_iou=_node_iou(p[name],record,truth)
                    nodes[name].append(final_node_changes(E_iou,student_iou,observed,np.any(p[name]!=p['ERefStrong'],axis=-1)))
                    path=trace[name]['trace']['final_rollout']
                    q=np.asarray(path['teacherq']);valid=np.asarray(path['effective_valid'],bool)
                    assert q.shape==valid.shape==(len(observed),3)and np.isfinite(q).all()
                    teacher_actions=np.where(valid,q,-np.inf).argmax(-1)
                    teacher_boxes=np.asarray(bank['boxes'])[np.arange(len(observed)),teacher_actions]
                    suggestions[name]=teacher_suggestions(_node_iou(teacher_boxes,record,truth),student_iou,E_iou,
                        teacher_actions,path['actions'],path['active'],path['observed'])
            for name in ARMS:
                assert np.array_equal(np.asarray(trace[name]['trace']['initial_rollout']['boxes']),p['CommitInitial'])
        teacher_rows.append(suggestions);rows.append(dict(id=key,parent=g['parent'],values=values,support=support))
        del trace,bank
    assert len(rows)==len({row['parent']for row in rows})==64
    assert maxerror<2e-10 and baselineerror<2e-10 and sha(labels_path)==labelsha
    validate(trial,lock)
    assert sha(trial/'sealed.json')==proof['seal_sha256']and sha(trial/'lock.json')==proof['lock_sha256']and not torch.cuda.is_initialized()
    means,counts=mean_table(rows,NAMES);signed=signed_changes(rows)
    proof.update(labels_sha256=labelsha,official_dense_checks=checks,official_dense_max_absolute_error=maxerror,
        reused_baseline_metric_max_absolute_error=baselineerror,all_hashes_rechecked_after_scoring=True,
        final_teacher_evaluation_uses_only_saved_final_feedback=True)
    public=dict(status='completed_HC64_exposed_development_committed_policy',queries=64,parents=64,
        arm_order=NAMES,new_arms=ARMS,no_adaptation_controls=CONTROLS,reused_methods=REUSED,
        means=means,valid_query_counts=counts,paired=paired_stats(rows,NAMES),costs=cost_summary(receipts,ARMS),
        shared_CPU_costs=_shared_cost_summary(shared),optimization=_optimization_summary(statistics),source_quality=_quality_summary(extras),
        final_node_changes={name:_aggregate_nodes(nodes[name])for name in ARMS},final_teacher_suggestions=_teacher_summary(teacher_rows),
        proof=proof,config=lock['config'],signed_changes=signed,
        queries_anonymous=[dict(query=f'H{i+1:03d}',values=row['values'],optimization=statistics[i],
            final_node_changes={name:nodes[name][i]for name in ARMS},final_teacher_suggestions=teacher_rows[i],
            source_quality={field:extras[i]['metadata'][field]for field in ('anchor_quality_mean','verified_anchors',
                'valid_active_alternative_slots','committed_alternative_slots','quality_or_witness_suppressed_slots')})for i,row in enumerate(rows)])
    dest=ROOT/'results/committed_policy_hc64';dest.mkdir(parents=True,exist_ok=True)
    assert not(dest/'report.json').exists(),'Refuse to overwrite earlier committed-policy results'
    write_json(trial/'metrics.json',dict(public,rows=rows));write_json(dest/'report.json',public)
    (dest/'report.md').write_text(_render(public));fields=list(signed[0]);lines=['\t'.join(fields)]
    lines+=['\t'.join('NA'if row[field]is None else str(row[field])for field in fields)for row in signed]
    (dest/'signed_changes.tsv').write_text('\n'.join(lines)+'\n')
    write_json(trial/'score_receipt.json',dict(status='scored',proof=proof,metrics_sha256=sha(trial/'metrics.json'),
        public_json_sha256=sha(dest/'report.json'),public_md_sha256=sha(dest/'report.md'),signed_changes_sha256=sha(dest/'signed_changes.tsv')))
    return public


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-after-seal',action='store_true');args=parser.parse_args()
    if not args.run_after_seal:parser.error('Explicit --run-after-seal and all64 sealed fits are required')
    result=evaluate();print(json.dumps(dict(status=result['status'],means=result['means']),indent=2))
