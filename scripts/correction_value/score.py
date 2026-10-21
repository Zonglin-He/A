"""Independent CPU assessment of sealed single-action correction rollouts.

Import is asset-free. GT is used only after all64 saved baseline/intervention
artifacts, sources and input hashes have passed the inference validator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np

ROOT=Path(os.environ.get('STVG_PROJECT_ROOT',Path(__file__).resolve().parents[2])).expanduser().resolve()
sys.path.insert(0,str(ROOT))
from scripts.state_opd.evaluate import read,sha,write_json,score_boxes

BRANCHES=('Student','Teacher','Random')
PAIRS=(('Teacher','Student'),('Random','Student'),('Teacher','Random'))
GROUPS=('active_unobserved_tail','all_unobserved_tail','near3_active_unobserved','far_active_unobserved')
BOOTSTRAP_SEED=20261021
BOOTSTRAP_DRAWS=10000
TRIAL=ROOT/'outputs/correction_value_hc64'


def intervention_metrics(ious,actions,intervention,active,observed):
    """The forced node is excluded from every tail metric and action count."""
    active=np.asarray(active,bool);observed=np.asarray(observed,bool)
    assert active.shape==observed.shape and active.ndim==1
    T=len(active);t=int(intervention);assert 0<=t<T
    ious={name:np.asarray(ious[name],float)for name in BRANCHES}
    actions={name:np.asarray(actions[name])for name in BRANCHES}
    assert all(value.shape==(T,)for value in (*ious.values(),*actions.values()))
    finite=np.isfinite(ious['Student'])
    assert all(np.array_equal(np.isfinite(value),finite)for value in ious.values())
    assert all(((value[finite]>=0)&(value[finite]<=1)).all()for value in ious.values())
    nodes=np.arange(T);later=nodes>t
    primary=later&active&~observed
    masks=dict(active_unobserved_tail=primary&finite,
        all_unobserved_tail=later&~observed&finite,
        near3_active_unobserved=primary&(nodes<=t+3)&finite,
        far_active_unobserved=primary&(nodes>t+3)&finite)
    result={}
    for group,mask in masks.items():
        count=int(mask.sum())
        means={name:float(value[mask].mean())if count else None for name,value in ious.items()}
        differences={first+'-'+second:float((ious[first][mask]-ious[second][mask]).mean())if count else 0.
            for first,second in PAIRS}
        student=ious['Student'];good=mask&(student>=.5);bad=mask&(student<.5)
        consequences={}
        for name in ('Teacher','Random'):
            delta=ious[name]-student;recovered=bad&(ious[name]>=.5);damaged=good&(ious[name]<.5)
            consequences[name]=dict(improved_nodes=int((mask&(delta>1e-12)).sum()),
                harmed_nodes=int((mask&(delta < -1e-12)).sum()),
                severe_harm_over5pp_nodes=int((mask&(delta < -.05)).sum()),
                initially_below_half_nodes=int(bad.sum()),recovered_to_half_nodes=int(recovered.sum()),
                initially_at_least_half_nodes=int(good.sum()),damaged_below_half_nodes=int(damaged.sum()),
                recovery_fraction=float(recovered.sum()/bad.sum())if bad.any()else None,
                damage_fraction=float(damaged.sum()/good.sum())if good.any()else None)
        result[group]=dict(nodes=count,absolute_sIoU=means,differences=differences,consequences=consequences)
    action_groups=dict(all_later=later,active_unobserved_tail=primary,
        all_unobserved_tail=later&~observed,near3_active_unobserved=primary&(nodes<=t+3),
        far_active_unobserved=primary&(nodes>t+3))
    action_counts={group:dict(nodes=int(mask.sum()),**{first+'-'+second:int(((actions[first]!=actions[second])&mask).sum())
        for first,second in PAIRS})for group,mask in action_groups.items()}
    return dict(groups=result,action_changes=action_counts,
        forced_node_IoU={name:float(value[t])if finite[t]else None for name,value in ious.items()},
        forced_node_excluded_from_tail=True)


def query_summary(points):
    """Average every preselected point equally, including zero-GT-point deltas."""
    out=dict(interventions=len(points),groups={},action_changes={})
    for group in GROUPS:
        values=[point['groups'][group]for point in points]
        out['groups'][group]=dict(scored_interventions=sum(value['nodes']>0 for value in values),
            summed_node_evaluations=sum(value['nodes']for value in values),
            eligible=any(value['nodes']>0 for value in values),
            differences={first+'-'+second:float(np.mean([value['differences'][first+'-'+second]for value in values]))if values else 0.
                for first,second in PAIRS},
            absolute_sIoU={name:float(np.mean([value['absolute_sIoU'][name]for value in values if value['absolute_sIoU'][name]is not None]))
                if any(value['absolute_sIoU'][name]is not None for value in values)else None for name in BRANCHES})
    for group in ('all_later','active_unobserved_tail','all_unobserved_tail','near3_active_unobserved','far_active_unobserved'):
        values=[point['action_changes'][group]for point in points]
        out['action_changes'][group]={field:sum(value[field]for value in values)for field in ('nodes',*(a+'-'+b for a,b in PAIRS))}
    return out


def paired_bootstrap(values,eligible=None):
    values=np.asarray(values,float);assert values.ndim==1 and np.isfinite(values).all()
    if eligible is not None:
        eligible=np.asarray(eligible,bool);assert eligible.shape==values.shape
        values=values[eligible]
    if not len(values):return dict(queries=0,mean_pp=None,CI95_pp=None,improved=0,declined=0,unchanged=0)
    rng=np.random.default_rng(BOOTSTRAP_SEED);draws=rng.integers(0,len(values),(BOOTSTRAP_DRAWS,len(values)))
    pp=100*values
    return dict(queries=len(values),mean_pp=float(pp.mean()),CI95_pp=np.quantile(pp[draws].mean(-1),[.025,.975]).tolist(),
        improved=int((pp>1e-8).sum()),declined=int((pp < -1e-8).sum()),unchanged=int((abs(pp)<=1e-8).sum()),
        declined_over5pp=int((pp < -5).sum()),
        positive_contribution_pp=float(pp[pp>0].sum()/len(pp)),negative_contribution_pp=float(pp[pp<0].sum()/len(pp)))


def decision_gate(primary):
    required=('Teacher-Student','Teacher-Random')
    passed=all(primary[pair]['CI95_pp']is not None and primary[pair]['CI95_pp'][0]>0 for pair in required)
    negative=any(primary[pair]['CI95_pp']is not None and primary[pair]['CI95_pp'][1]<0 for pair in required)
    return dict(passed=passed,status='go'if passed else'negative'if negative else'inconclusive',
        required_comparisons=list(required),rule='Both all64 primary active-unobserved-tail sIoU95% query-bootstrap CI lower bounds must be strictly above0.',
        interpretation='This is a predeclared resource-allocation gate for entering a new distillation experiment, not a universal theoretical claim about OPD.')


def _stratum(point,name):
    if name=='all':return True
    if name=='two_actions':return point['legal_action_count']==2
    if name=='three_actions':return point['legal_action_count']==3
    if name in ('policy_disagreement','noise_only_disagreement'):return point['disagreement_type']==name
    raise ValueError(name)


def aggregate_cohort(queries):
    """All64 means include zero-valued absent/empty points, never GT filtering."""
    strata=('all','two_actions','three_actions','policy_disagreement','noise_only_disagreement')
    result={};signed=[]
    for stratum in strata:
        summaries=[query_summary([p for p in query['points']if _stratum(p,stratum)])for query in queries]
        groups={}
        for group in GROUPS:
            eligible=[row['groups'][group]['eligible']for row in summaries]
            pairs={}
            for first,second in PAIRS:
                pair=first+'-'+second;deltas=[row['groups'][group]['differences'][pair]for row in summaries]
                pairs[pair]=dict(all_queries=paired_bootstrap(deltas),eligible_queries=paired_bootstrap(deltas,eligible))
                for index,(delta,row)in enumerate(zip(deltas,summaries)):
                    if stratum=='all'and group=='active_unobserved_tail':
                        signed.append(dict(query=f'H{index+1:03d}',stratum=stratum,group=group,comparison=pair,
                            interventions=row['interventions'],eligible=row['groups'][group]['eligible'],
                            scored_interventions=row['groups'][group]['scored_interventions'],tail_sIoU_delta_pp=100*delta,
                            severe_decline_over5pp=delta < -.05))
            absolutes={}
            for name in BRANCHES:
                values=[row['groups'][group]['absolute_sIoU'][name]for row in summaries]
                finite=[value for value in values if value is not None]
                absolutes[name]=dict(queries=len(finite),mean=float(np.mean(finite))if finite else None)
            groups[group]=dict(queries=len(queries),queries_with_interventions=sum(row['interventions']>0 for row in summaries),
                eligible_queries=sum(eligible),scored_interventions=sum(row['groups'][group]['scored_interventions']for row in summaries),
                summed_node_evaluations=sum(row['groups'][group]['summed_node_evaluations']for row in summaries),
                absolute_sIoU=absolutes,paired=pairs)
        result[stratum]=dict(groups=groups,interventions=sum(row['interventions']for row in summaries),
            query_summaries=summaries)
    result['decision']=decision_gate({pair:result['all']['groups']['active_unobserved_tail']['paired'][pair]['all_queries']
        for pair in ('Teacher-Student','Teacher-Random')})
    result['signed_changes']=signed
    return result


def consequence_summary(queries):
    result={}
    for group in GROUPS:
        result[group]={}
        for name in ('Teacher','Random'):
            records=[point['groups'][group]['consequences'][name]for query in queries for point in query['points']]
            integer_fields=('improved_nodes','harmed_nodes','severe_harm_over5pp_nodes',
                'initially_below_half_nodes','recovered_to_half_nodes','initially_at_least_half_nodes','damaged_below_half_nodes')
            summary={field:sum(row[field]for row in records)for field in integer_fields}
            for field,numerator,denominator in [('recovery_fraction','recovered_to_half_nodes','initially_below_half_nodes'),
                                                ('damage_fraction','damaged_below_half_nodes','initially_at_least_half_nodes')]:
                summary[field]=summary[numerator]/summary[denominator]if summary[denominator]else None
            result[group][name]=summary
    return result


def action_summary(queries):
    result={}
    for group in ('all_later','active_unobserved_tail','all_unobserved_tail','near3_active_unobserved','far_active_unobserved'):
        records=[point['action_changes'][group]for query in queries for point in query['points']]
        count=sum(row['nodes']for row in records)
        result[group]=dict(node_evaluations=count,comparisons={first+'-'+second:dict(
            changed_actions=sum(row[first+'-'+second]for row in records),
            changed_action_fraction=sum(row[first+'-'+second]for row in records)/count if count else None)
            for first,second in PAIRS})
    return result


def validate_before_labels(trial=TRIAL):
    from scripts.correction_value.run import validate
    import torch
    trial=Path(trial).resolve();assert trial==TRIAL.resolve()
    lock=read(trial/'lock.json');receipts=validate(trial,lock);seal=read(trial/'sealed.json')
    assert seal['queries']==64 and seal['GT_read']is False and seal['fingerprint']==lock['fingerprint']
    assert seal['lock_sha256']==sha(trial/'lock.json')and len(receipts)==len(seal['files'])==64
    manifest=ROOT/'data/evaluation/development.jsonl';assert sha(manifest)==lock['manifest_sha256']
    records=[json.loads(line)for line in manifest.read_text().splitlines()if line.strip()]
    records=[row for row in records if row['dataset']=='hc2'];assert len(records)==128
    records=records[:64];assert [row['id']for row in records]==lock['ids']
    previous=ROOT/'outputs/evidence_policy_hc64'
    for item in lock['inputs']:
        key=item['id'];r=receipts[key];entry=seal['files'][key]
        assert r['GT_read']is False and r['parameter_updates']==0
        assert all(r[field]==0 for field in ('model_calls','decoder_calls','expert_calls','RGB_reads'))
        for field,hashfield in [('path','sha256'),('receipt_path','receipt_sha256')]:
            path=(trial/entry[field]).resolve();assert path.is_relative_to(trial)and sha(path)==entry[hashfield]
        assert r['prediction_path']==entry['path']and r['prediction_sha256']==entry['sha256']
        tracepath=(trial/r['trace_path']).resolve();assert tracepath.is_relative_to(trial)and sha(tracepath)==r['trace_sha256']
        trace=torch.load(tracepath,map_location='cpu',weights_only=False)
        old=torch.load(previous/item['previous_trace_path'],map_location='cpu',weights_only=False)
        expected=hashlib.sha256(old['B_OPD']['finaltheta'].numpy().tobytes()).hexdigest()
        assert expected==r['phi_sha256']==trace['phi_sha256']
        assert torch.equal(trace['base']['phi'],old['B_OPD']['finaltheta'])
        assert r['points']==[branch['point']for branch in trace['branches']]
        assert len(r['points'])==r['interventions']<=4
        for branch in trace['branches']:
            point=branch['point'];node=point['node'];legal=point['valid_actions']
            assert len(legal)in (2,3)and len(set(legal))==len(legal)
            assert point['student_action']!=point['teacher_action']and point['random_action']!=point['student_action']
            assert all(point[field]in legal for field in ('student_action','teacher_action','random_action'))
            expected_type='policy_disagreement'if point['teacher_action']!=point['student_argmax']else'noise_only_disagreement'
            assert point['disagreement_type']==expected_type
            if len(legal)==2:assert point['teacher_action']==point['random_action']
            for name in BRANCHES:
                path=branch['paths'][name]
                assert torch.equal(path['phi'],trace['base']['phi'])and torch.equal(path['noise'],trace['base']['noise'])
                assert torch.equal(path['boxes'][:node],trace['base']['boxes'][:node])
        del trace,old
    assert not torch.cuda.is_initialized()
    proof=dict(fingerprint=lock['fingerprint'],lock_sha256=sha(trial/'lock.json'),seal_sha256=sha(trial/'sealed.json'),
        previous_lock_sha256=lock['previous_lock_sha256'],previous_seal_sha256=lock['previous_seal_sha256'],
        bank_seal_sha256=lock['bank_seal_sha256'],manifest_sha256=sha(manifest),
        all64_sources_inputs_baselines_interventions_receipts_validated_before_labels=True,
        same_frozen_final_B_OPD_weights=True,common_random_numbers=True,GT_used_to_select_interventions=False,
        parameter_updates=0,model_calls=0,decoder_calls=0,expert_calls=0,RGB_reads=0,CUDA_initialized=False)
    return trial,lock,seal,receipts,records,proof


def _node_iou(boxes,record,truth):
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    metadata=record['metadata'];pixels=np.maximum(xyxy(np.asarray(boxes),metadata['width'],metadata['height']),0)
    return np.asarray([box_iou(box,truth[frame])if frame in truth else np.nan for frame,box in zip(metadata['frame_ids'],pixels)])


def _auxiliary(queries):
    out={}
    for first,second in PAIRS:
        name=first+'-'+second
        values=[float(np.mean([point['full_video_values'][first]['v']-point['full_video_values'][second]['v']for point in query['points']]))
            if query['points']else 0. for query in queries]
        out[name]=paired_bootstrap(values)
    return out


def _execution_summary(receipts):
    seconds=np.asarray([r['seconds']for r in receipts.values()],float)
    assert np.isfinite(seconds).all()and(seconds>=0).all()
    points=[point for receipt in receipts.values()for point in receipt['points']]
    counts={name:sum(point['disagreement_type']==name for point in points)for name in ('policy_disagreement','noise_only_disagreement')}
    return dict(queries=len(receipts),queries_with_interventions=sum(r['interventions']>0 for r in receipts.values()),
        selected_interventions=len(points),eligible_disagreement_nodes=sum(r['eligible_disagreements']for r in receipts.values()),
        two_action_interventions=sum(len(p['valid_actions'])==2 for p in points),three_action_interventions=sum(len(p['valid_actions'])==3 for p in points),
        Teacher_equals_Random_interventions=sum(p['teacher_action']==p['random_action']for p in points),
        disagreement_types=counts,seconds=dict(total=float(seconds.sum()),mean=float(seconds.mean()),median=float(np.median(seconds)),p90=float(np.quantile(seconds,.9))),
        parameter_updates=0,model_calls=0,decoder_calls=0,expert_calls=0,RGB_reads=0,
        scope='Cached lightweight-policy intervention cost only; existing Source/ReferDINO evidence is reused. Not full deployed inference latency.')


def _render(public):
    lines=['# HC64: value of one teacher-directed correction','',
        'The student is frozen at each query’s previously sealed final B_OPD weights. An independently seeded sampled student path '
        'defines at most four preselected disagreement points. Each point branches independently into the original Student action, '
        'the Teacher-Q4 argmax, or a uniformly sampled legal non-Student Random action. All later decisions use the same frozen student '
        'and the same Gumbel numbers, with each branch updating its own memory. No optimization or model call is performed.','',
        '**Primary support:** strictly later than the forced point, active, unobserved sampled nodes with GT. '
        'The forced point is excluded. Each query first averages all its preselected interventions equally. A point with no scored tail '
        'contributes zero delta; a query with no point or scored tail contributes zero to the all64 mean. Eligible-only results retain '
        'these same query deltas and only remove queries with no scored point. Absolute IoU can be undefined.','',
        '| Tail support | Comparison | All64 delta pp [95% CI] | Eligible queries | Eligible-only delta pp [95% CI] | Improve / decline / same |',
        '|---|---|---:|---:|---:|---:|']
    def ci(row):return 'NA'if row['mean_pp']is None else f'{row["mean_pp"]:+.4f} [{row["CI95_pp"][0]:.4f}, {row["CI95_pp"][1]:.4f}]'
    for group in GROUPS:
        for first,second in PAIRS:
            row=public['aggregate']['all']['groups'][group]['paired'][first+'-'+second]
            allrow=row['all_queries'];eligible=row['eligible_queries']
            lines.append(f'| {group} | {first} − {second} | {ci(allrow)} | {eligible["queries"]} | {ci(eligible)} | '
                f'{allrow["improved"]} / {allrow["declined"]} / {allrow["unchanged"]} |')
    lines+=['','Near3 means sampled nodes t+1 through t+3; far means t+4 onward. Both retain the primary active/unobserved/GT mask. '
        'All-unobserved-tail additionally includes inactive later nodes. Intervals use10000 paired query draws, one query per parent.','',
        '## Disagreement and action-support strata','',
        '| Stratum | Selected interventions | Teacher − Student all64 pp [95% CI] | Teacher − Random all64 pp [95% CI] |',
        '|---|---:|---:|---:|']
    for stratum in ('two_actions','three_actions','policy_disagreement','noise_only_disagreement'):
        row=public['aggregate'][stratum];pairs=row['groups']['active_unobserved_tail']['paired']
        lines.append(f'| {stratum} | {row["interventions"]} | {ci(pairs["Teacher-Student"]["all_queries"])} | '
            f'{ci(pairs["Teacher-Random"]["all_queries"])} |')
    e=public['execution'];decision=public['decision']
    lines+=['',f'{e["selected_interventions"]} interventions across {e["queries_with_interventions"]}/64 queries; '
        f'{e["Teacher_equals_Random_interventions"]} have the same Teacher and Random action. With exactly two legal actions this equality is required, '
        'so that stratum cannot independently distinguish Teacher from Random. Policy-disagreement means Teacher differs from the student’s '
        'noise-free argmax; noise-only disagreement means Teacher agrees with that argmax but differs from the sampled action.','',
        '## Damage, recovery and gate','',
        '| Branch | Initially Student-IoU≥0.5 node-evaluations | Damaged below0.5 | Initially below0.5 | Recovered to≥0.5 | Tail harm >5 pp |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('Teacher','Random'):
        row=public['consequences']['active_unobserved_tail'][name]
        lines.append(f'| {name} | {row["initially_at_least_half_nodes"]} | {row["damaged_below_half_nodes"]} | '
            f'{row["initially_below_half_nodes"]} | {row["recovered_to_half_nodes"]} | {row["severe_harm_over5pp_nodes"]} |')
    lines+=['','Counts are node-evaluations across independent interventions and can revisit the same physical frame. '
        'Damage/recovery labels describe offline final branch boxes; GT never selects points or branch actions.','',
        'Across all later active unobserved nodes, including nodes without GT, Teacher versus Student changes '
        f'{public["actual_action_changes"]["active_unobserved_tail"]["comparisons"]["Teacher-Student"]["changed_actions"]} / '
        f'{public["actual_action_changes"]["active_unobserved_tail"]["node_evaluations"]} actions; '
        'the directly forced node is excluded.','',
        f'**Decision: {decision["status"]}; passed={decision["passed"]}.** '+decision['rule']+' '+decision['interpretation'],'',
        'Complete-video vIoU is auxiliary because it includes the forced point. Its gains cannot establish a tail-propagation benefit. '
        'All positive, negative, unchanged and empty-query primary changes are retained in report.json and signed_changes.tsv. '
        'All64 remains the primary exposed-development panel; no query is removed by outcome.','',e['scope']]
    return '\n'.join(lines)+'\n'


def evaluate(trial=TRIAL):
    from scripts.correction_value.run import validate
    import torch
    trial,lock,seal,receipts,records,proof=validate_before_labels(trial)
    labels_path=ROOT/'data/evaluation/development_labels.jsonl';labelsha=sha(labels_path)
    assert labelsha==read(ROOT/'outputs/layout_reward_dev/scope_hc2/metrics.json')['labels_sha256']
    labels={row['id']:row for row in [json.loads(line)for line in labels_path.read_text().splitlines()if line.strip()]
        if row['dataset']=='hc2'and row['id']in lock['ids']}
    assert set(labels)==set(lock['ids'])
    previous=read(ROOT/'results/evidence_policy_hc64/report.json')['queries_anonymous']
    queries=[];checks=0;maxerror=baselineerror=0.
    for index,record in enumerate(records):
        key=record['id'];g=labels[key];truth={int(k):v for k,v in g['boxes'].items()};r=receipts[key]
        with np.load(trial/seal['files'][key]['path'],allow_pickle=False)as p:
            interval=p['Strong_interval'].tolist();positions=p['positions8'].tolist()
            baselines={}
            for name in ('CurrentGreedy','BaseStudent'):
                baselines[name],error,_=score_boxes(p[name],record,truth,interval,g['interval'],positions)
                checks+=1;maxerror=max(maxerror,error)
            old=previous[index]['values']['B_OPD']
            for field in ('v','t','s','observed_actual_sIoU','unobserved_actual_sIoU'):
                if old[field]is None:assert baselines['CurrentGreedy'][field]is None
                else:baselineerror=max(baselineerror,abs(baselines['CurrentGreedy'][field]-old[field]))
            points=[]
            for i,point in enumerate(r['points']):
                ious={name:_node_iou(p[name][i],record,truth)for name in BRANCHES}
                actions={name:p[name+'_actions'][i]for name in BRANCHES}
                row=intervention_metrics(ious,actions,point['node'],p['active'],p['observed'])
                full={}
                for name in BRANCHES:
                    full[name],error,_=score_boxes(p[name][i],record,truth,interval,g['interval'],positions)
                    checks+=1;maxerror=max(maxerror,error)
                assert full['Student']==baselines['BaseStudent']
                row.update(intervention=i,time_bin=point['time_bin'],legal_action_count=len(point['valid_actions']),
                    disagreement_type=point['disagreement_type'],Teacher_equals_Random=point['teacher_action']==point['random_action'],
                    full_video_values=full)
                points.append(row)
            queries.append(dict(query=f'H{index+1:03d}',parent=g['parent'],points=points,baselines=baselines))
    assert len(queries)==len({query['parent']for query in queries})==64
    assert maxerror<2e-10 and baselineerror<2e-10 and sha(labels_path)==labelsha
    validate(trial,lock)
    assert sha(trial/'sealed.json')==proof['seal_sha256']and sha(trial/'lock.json')==proof['lock_sha256']and not torch.cuda.is_initialized()
    aggregate=aggregate_cohort(queries);decision=aggregate.pop('decision');signed=aggregate.pop('signed_changes')
    proof.update(labels_sha256=labelsha,official_dense_checks=checks,official_dense_max_absolute_error=maxerror,
        current_greedy_metric_max_absolute_error=baselineerror,all_hashes_rechecked_after_scoring=True)
    public=dict(status='completed_HC64_exposed_development_correction_value',queries=64,parents=64,
        decision=decision,aggregate=aggregate,consequences=consequence_summary(queries),actual_action_changes=action_summary(queries),
        auxiliary_full_video_vIoU=_auxiliary(queries),execution=_execution_summary(receipts),
        proof=proof,config=lock['config'],signed_changes=signed,
        queries_anonymous=[{key:value for key,value in query.items()if key!='parent'}for query in queries])
    dest=ROOT/'results/correction_value_hc64';dest.mkdir(parents=True,exist_ok=True)
    assert not(dest/'report.json').exists(),'Refuse to overwrite earlier correction-value results'
    write_json(trial/'metrics.json',dict(public,private_parents=[query['parent']for query in queries]))
    write_json(dest/'report.json',public);(dest/'report.md').write_text(_render(public))
    fields=list(signed[0]);lines=['\t'.join(fields)]+['\t'.join(str(row[field])for field in fields)for row in signed]
    (dest/'signed_changes.tsv').write_text('\n'.join(lines)+'\n')
    write_json(trial/'gate.json',dict(decision,fingerprint=lock['fingerprint'],seal_sha256=proof['seal_sha256'],
        labels_sha256=labelsha,diagnostics_sha256=sha(dest/'report.json')))
    write_json(trial/'score_receipt.json',dict(status='scored',proof=proof,metrics_sha256=sha(trial/'metrics.json'),
        public_json_sha256=sha(dest/'report.json'),public_md_sha256=sha(dest/'report.md'),
        signed_changes_sha256=sha(dest/'signed_changes.tsv'),gate_sha256=sha(trial/'gate.json')))
    return public


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-after-seal',action='store_true');args=parser.parse_args()
    if not args.run_after_seal:parser.error('Explicit --run-after-seal and all64 sealed predictions are required')
    result=evaluate();print(json.dumps(dict(status=result['status'],decision=result['decision']),indent=2))
