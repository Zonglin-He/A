"""CPU-only analysis of the fully sealed HC64 A/B evidence-policy run.

Importing this module reads no assets or labels. Prediction validation and all
cohort hashes must pass before the independent offline scoring entry reads GT.
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
from scripts.box_propagation.score import _JS,_shared_cost_summary

REUSED=['ERefStrong','GeometryOPD','BoxFixed','BoxOPD','BoxFixedKD','BoxDirect']
ARMS=['OPD','FixedKD','Direct']
NEW=[variant+'_'+arm for variant in ('A','B')for arm in ARMS]
NAMES=REUSED+NEW
TRIAL=ROOT/'outputs/evidence_policy_hc64'


def logit_diagnostics(logits,prior_log,valid,active,observed):
    """Margins use the actual hard action support, including fallback counts."""
    logits=np.asarray(logits,float);prior=np.asarray(prior_log,float)
    valid=np.asarray(valid,bool);active=np.asarray(active,bool);observed=np.asarray(observed,bool)
    assert logits.ndim==2 and logits.shape==prior.shape==valid.shape
    assert active.shape==observed.shape==logits.shape[:1]and valid[:,0].all()
    assert np.isfinite(logits[valid]).all()and np.isfinite(prior[valid]).all()
    eligible=valid.copy();eligible[~active,1:]=False
    alternatives=eligible.copy();alternatives[:,0]=False
    has_alt=alternatives.any(-1)
    best_alt=np.where(alternatives,logits,-np.inf).max(-1)
    best_prior_alt=np.where(alternatives,prior,-np.inf).max(-1)
    margin=np.where(has_alt,best_alt-logits[:,0],np.nan)
    barrier=np.where(has_alt,prior[:,0]-best_prior_alt,np.nan)
    result={}
    for group,mask in [('all',np.ones(len(active),bool)),('observed',observed),('unobserved',~observed),
                       ('active_unobserved',active&~observed)]:
        nodes=mask&has_alt
        x=margin[nodes];p=barrier[nodes]
        result[group]=dict(nodes=int(mask.sum()),eligible_alternative_nodes=int(nodes.sum()),
            E_only_nodes=int((mask&~has_alt).sum()),positive_margin_nodes=int((x>0).sum()),
            zero_margin_nodes=int((x==0).sum()),margin_mean=float(x.mean())if len(x)else None,
            margin_median=float(np.median(x))if len(x)else None,
            margin_minimum=float(x.min())if len(x)else None,margin_maximum=float(x.max())if len(x)else None,
            prior_barrier_mean=float(p.mean())if len(p)else None)
    return result


def target_change(scores,immediate,valid,active,observed):
    """Compare Q and immediate-r preferences at exactly the same saved states."""
    scores=np.asarray(scores,float);immediate=np.asarray(immediate,float)
    valid=np.asarray(valid,bool).copy();active=np.asarray(active,bool);observed=np.asarray(observed,bool)
    assert scores.shape==immediate.shape==valid.shape and scores.ndim==2
    valid[~active,1:]=False
    assert valid[:,0].all()and np.isfinite(scores[valid]).all()and np.isfinite(immediate[valid]).all()
    q=np.where(valid,scores,-np.inf).argmax(-1);r=np.where(valid,immediate,-np.inf).argmax(-1)
    result={}
    for name,mask in [('all',np.ones(len(active),bool)),('active',active),('active_unobserved',active&~observed)]:
        count=int(mask.sum());different=int(((q!=r)&mask).sum())
        result[name]=dict(nodes=count,argmax_differences=different,
            argmax_difference_fraction=different/count if count else None,
            Q_nonE_preference_nodes=int(((q!=0)&mask).sum()),
            immediate_nonE_preference_nodes=int(((r!=0)&mask).sum()))
    return result


def final_node_changes(E_iou,final_iou,observed,replaced):
    """GT is used only after sealing, on final boxes; never for online choices."""
    e=np.asarray(E_iou,float);f=np.asarray(final_iou,float)
    observed=np.asarray(observed,bool);replaced=np.asarray(replaced,bool)
    assert e.shape==f.shape==observed.shape==replaced.shape
    has=np.isfinite(e)&np.isfinite(f)
    result={}
    for group,subset in [('all',np.ones(len(e),bool)),('observed',observed),('unobserved',~observed)]:
        mask=has&subset;changed=mask&replaced;delta=f-e
        initially_bad=mask&(e<.5);initially_good=mask&(e>=.5)
        recovery=initially_bad&(f>=.5);damage=initially_good&(f<.5)
        result[group]=dict(nodes=int(mask.sum()),replaced_nodes=int(changed.sum()),
            improved_nodes=int((mask&(delta>1e-12)).sum()),harmed_nodes=int((mask&(delta < -1e-12)).sum()),
            unchanged_nodes=int((mask&(abs(delta)<=1e-12)).sum()),
            severe_harm_over5pp_nodes=int((mask&(delta < -.05)).sum()),
            mean_delta=float(delta[mask].mean())if mask.any()else None,
            mean_replaced_delta=float(delta[changed].mean())if changed.any()else None,
            harmful_replaced_nodes=int((changed&(delta < -1e-12)).sum()),
            initially_below_half_nodes=int(initially_bad.sum()),recovered_to_half_nodes=int(recovery.sum()),
            recovery_fraction=float(recovery.sum()/initially_bad.sum())if initially_bad.any()else None,
            initially_at_least_half_nodes=int(initially_good.sum()),damaged_below_half_nodes=int(damage.sum()),
            damage_fraction=float(damage.sum()/initially_good.sum())if initially_good.any()else None)
    return result


def _aggregate_nodes(rows):
    result={}
    for group in ('all','observed','unobserved'):
        values=[row[group]for row in rows]
        total={field:sum(row[field]for row in values)for field in values[0]if field.endswith('_nodes')or field=='nodes'}
        for output,numerator,denominator in [('recovery_fraction','recovered_to_half_nodes','initially_below_half_nodes'),
                                             ('damage_fraction','damaged_below_half_nodes','initially_at_least_half_nodes'),
                                             ('harmful_replacement_fraction','harmful_replaced_nodes','replaced_nodes')]:
            total[output]=total[numerator]/total[denominator]if total[denominator]else None
        total['macro_mean_delta']=float(np.mean([row['mean_delta']for row in values if row['mean_delta']is not None]))if any(row['mean_delta']is not None for row in values)else None
        result[group]=total
    return result


def _focus_pairs():
    pairs=[]
    for name in NEW:
        pairs.extend((name,base)for base in ('ERefStrong','GeometryOPD','BoxFixed','Box'+name.split('_')[1]))
    pairs.extend(('B_'+arm,'A_'+arm)for arm in ARMS)
    for variant in ('A','B'):
        pairs.extend((variant+'_OPD',variant+'_'+arm)for arm in ('FixedKD','Direct'))
    return list(dict.fromkeys(pairs))


def _signed(rows):
    result=[]
    for index,row in enumerate(rows):
        for method,baseline in _focus_pairs():
            first,second=row['values'][method],row['values'][baseline]
            dv=100*(first['v']-second['v'])
            ds=None if first['unobserved_actual_sIoU']is None or second['unobserved_actual_sIoU']is None else 100*(first['unobserved_actual_sIoU']-second['unobserved_actual_sIoU'])
            result.append(dict(query=f'H{index+1:03d}',method=method,baseline=baseline,vIoU_delta_pp=dv,
                unobserved_sIoU_delta_pp=ds,severe_vIoU_drop_over5pp=dv < -5,
                severe_unobserved_sIoU_drop_over5pp=ds is not None and ds < -5))
    return result


def _path_logits(path):
    if 'student_logits'in path:return path['student_logits']
    return path['student_log_probs']


def _rollout_statistics(path):
    logits=_path_logits(path)
    return logit_diagnostics(logits,path['prior_log_probs'],path['effective_valid'],path['active'],path['observed'])


def _training_statistics(output):
    rows=[]
    outer_paths=output['trace'].get('outer_acquisitions',[])
    if not outer_paths or any(len(paths)!=2 for paths in outer_paths):
        seen=set();outer_paths=[]
        for row,paths in zip(output['rounds'],output['trace']['training_paths']):
            if row['outer']not in seen:outer_paths.append(paths);seen.add(row['outer'])
    for outer,paths in enumerate(outer_paths):
        assert len(paths)==2
        for p,path in enumerate(paths):
            rows.append(dict(outer=outer,path=p,Q_versus_immediate=target_change(path['teacher_scores'],
                path['r'],path['effective_valid'],path['active'],path['observed'])))
    return rows


def _history_comparison(opd,kd):
    """Different saved histories at matched nodes; no new feedback is queried."""
    first={};second={}
    for output,lookup in ((opd,first),(kd,second)):
        for i,row in enumerate(output['rounds']):lookup.setdefault(row['outer'],i)
    rows=[]
    for outer in sorted(set(first)&set(second)):
        for p,(a,b)in enumerate(zip(opd['trace']['training_paths'][first[outer]],kd['trace']['training_paths'][second[outer]])):
            active=np.asarray(a['active'],bool);observed=np.asarray(a['observed'],bool)
            assert np.array_equal(active,np.asarray(b['active'],bool))
            memory_a=np.asarray(a['memory_before']);memory_b=np.asarray(b['memory_before'])
            prev_a=np.asarray(a['previous_action_probs']);prev_b=np.asarray(b['previous_action_probs'])
            feature_a=np.asarray(a['previous_features']);feature_b=np.asarray(b['previous_features'])
            different=~((memory_a==memory_b).all(-1)&(prev_a==prev_b).all(-1)&(feature_a==feature_b).all(-1))
            js=_JS(a['teacherq'],b['teacherq'])
            qarg=np.asarray(a['teacherq']).argmax(-1)!=np.asarray(b['teacherq']).argmax(-1)
            mask=active&~observed;changed=mask&different
            same=~different
            same_error=float(np.abs(np.asarray(a['teacherq'])[same]-np.asarray(b['teacherq'])[same]).max())if same.any()else None
            assert same_error is None or same_error<1e-12
            rows.append(dict(outer=int(outer),path=p,active_unobserved_nodes=int(mask.sum()),
                different_saved_history_unobserved_nodes=int(changed.sum()),
                identical_saved_history_nodes=int(same.sum()),identical_history_target_max_abs_error=same_error,
                target_argmax_differences=int((mask&qarg).sum()),
                different_history_target_argmax_differences=int((changed&qarg).sum()),
                target_JS_mean=float(js[mask].mean())if mask.any()else None,
                different_history_target_JS_mean=float(js[changed].mean())if changed.any()else None,
                memory_L2_mean=float(np.linalg.norm(memory_a-memory_b,axis=-1)[mask].mean())if mask.any()else None))
    return dict(rows=rows,matched_outer_rounds=len(set(first)&set(second)),
        interpretation='Targets use the same frozen immediate-teacher reference continuation rule conditioned on different incoming visited histories. Identical incoming memory, previous action distribution and previous selected feature must give the same target. This comparison is not an isolated causal memory ablation. No new rollout or teacher feedback is generated.')


def _scalar_summary(values):
    values=[float(value)for value in values if value is not None]
    return dict(count=len(values),mean=float(np.mean(values))if values else None,
        minimum=min(values)if values else None,maximum=max(values)if values else None)


def _public_margins(margins):
    """Keep per-node vectors private while exposing finite anonymous summaries."""
    result={}
    for key,value in margins.items():
        if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
        if isinstance(value,np.ndarray)and value.ndim>0:
            values=np.asarray(value,dtype=float).reshape(-1)
            assert np.isfinite(values).all()
            result[key]=dict(count=len(values),mean=float(values.mean())if len(values)else None,
                minimum=float(values.min())if len(values)else None,maximum=float(values.max())if len(values)else None,
                p10=float(np.quantile(values,.1))if len(values)else None,p50=float(np.quantile(values,.5))if len(values)else None,
                p90=float(np.quantile(values,.9))if len(values)else None,positive_nodes=int((values>0).sum()),
                zero_nodes=int((values==0).sum()),negative_nodes=int((values<0).sum()))
        else:
            if isinstance(value,(np.ndarray,np.generic)):value=value.item()
            assert value is None or isinstance(value,(str,int,float,bool))
            assert not isinstance(value,float)or np.isfinite(value)
            result[key]=value
    return result


def _validate_output(output):
    costs=output['costs'];updates=costs['updates']
    assert isinstance(updates,int)and 1<=updates<=30
    assert costs['backward']==costs['state_only_backwards']==updates
    assert costs['accepted_updates']+costs['rejected_updates']==updates
    assert all(costs[field]==0 for field in ('heavy_forwards','decoder_calls','decoder_backwards','encoder_calls','expert_calls'))
    assert len(output['rounds'])==updates
    assert np.asarray(output['initialtheta']).shape==(8,)and(np.asarray(output['initialtheta'])==0).all()
    assert len(output['trace']['training_paths'])==updates and all(len(p)==2 for p in output['trace']['training_paths'])
    for row in output['rounds']:
        assert np.isfinite(np.asarray(row['gradient'])).all()
        if row['rolled_back']:
            assert np.array_equal(np.asarray(row['before']),np.asarray(row['after']))
            assert row['optimizer_before_sha256']==row['optimizer_after_sha256']
    return dict(initial=_rollout_statistics(output['trace']['initial_rollout']),
        final=_rollout_statistics(output['trace']['final_rollout']),
        target_feedback=_training_statistics(output),
        updates=updates,accepted=costs['accepted_updates'],rejected=costs['rejected_updates'],
        loss_curve=[dict(update=i,loss=float(row['loss']),accepted=bool(row['accepted']),
            weight_delta_L2=float(np.linalg.norm(np.asarray(row['after'])-np.asarray(row['before']))),
            logit_margins=_public_margins(row['logit_margins']),
            before_score=float(row['before_score']),after_score=float(row['after_score']))for i,row in enumerate(output['rounds'])])


def validate_before_labels(trial=TRIAL):
    from scripts.evidence_policy.run import validate
    trial=Path(trial).resolve();assert trial==TRIAL.resolve()
    lock=read(trial/'lock.json');receipts=validate(trial,lock);seal=read(trial/'sealed.json')
    assert seal['queries']==64 and seal['GT_read']is False and seal['fingerprint']==lock['fingerprint']
    assert seal['lock_sha256']==sha(trial/'lock.json')
    manifest=ROOT/'data/evaluation/development.jsonl';assert sha(manifest)==lock['manifest_sha256']
    records=[json.loads(line)for line in manifest.read_text().splitlines()if line.strip()]
    records=[row for row in records if row['dataset']=='hc2'];assert len(records)==128
    records=records[:64];assert [row['id']for row in records]==lock['ids']
    assert len(receipts)==len(seal['files'])==64
    old=ROOT/'outputs/box_propagation_hc64';oldseal=read(old/'sealed.json')
    assert sha(old/'sealed.json')==lock['bank_source_seal_sha256']
    statistics=[];shared=[]
    for record,item in zip(records,lock['inputs']):
        key=record['id'];receipt=receipts[key];entry=seal['files'][key]
        assert receipt['GT_read']is False
        for field in ('path','receipt_path'):
            path=(trial/entry[field]).resolve();assert path.is_relative_to(trial)
            assert sha(path)==entry['sha256'if field=='path'else'receipt_sha256']
        tracepath=(trial/receipt['trace_path']).resolve();assert tracepath.is_relative_to(trial)
        assert sha(tracepath)==receipt['trace_sha256']
        trace=torch.load(tracepath,map_location='cpu',weights_only=False)
        with np.load(trial/entry['path'],allow_pickle=False)as p,np.load(old/oldseal['files'][key]['path'],allow_pickle=False)as previous:
            assert set(p.files)==set(NAMES)|{'frame_ids','positions8','Strong_interval'}
            for field in ('frame_ids','positions8','Strong_interval',*REUSED):assert np.array_equal(p[field],previous[field])
            assert p['frame_ids'].tolist()==record['metadata']['frame_ids']
            row={}
            for name in NEW:
                assert p[name].shape==p['ERefStrong'].shape and np.isfinite(p[name]).all()and(p[name][:,2:]>0).all()
                assert np.array_equal(p[name],np.asarray(trace[name]['final']))
                assert trace[name]['costs']==receipt['costs'][name]
                row[name]=_validate_output(trace[name])
                observed=np.isin(np.arange(len(p['frame_ids'])),p['positions8'])
                replaced=np.any(p[name]!=p['ERefStrong'],axis=-1)
                active=np.asarray(trace[name]['trace']['final_rollout']['active'],bool)
                assert not(replaced&~active).any()
                row[name]['output_replacements']={group:dict(nodes=int(mask.sum()),replaced_nodes=int((replaced&mask).sum()))
                    for group,mask in [('all',np.ones(len(observed),bool)),('observed',observed),
                                       ('unobserved',~observed),('active_unobserved',active&~observed),('inactive',~active)]}
            row['B_history_comparison']=_history_comparison(trace['B_OPD'],trace['B_FixedKD'])
            statistics.append(row)
        shared.append(dict(receipt.get('timings',{}),**({'total_seconds':receipt['total_seconds']}if'total_seconds'in receipt else{})))
    assert not torch.cuda.is_initialized()
    proof=dict(fingerprint=lock['fingerprint'],lock_sha256=sha(trial/'lock.json'),seal_sha256=sha(trial/'sealed.json'),
        bank_source_seal_sha256=lock['bank_source_seal_sha256'],manifest_sha256=sha(manifest),
        all64_predictions_inputs_sources_and_traces_validated_before_labels=True,GT_used_for_prediction_or_parameter_choice=False,
        reused_methods_exact=True,model_calls=0,decoder_calls=0,expert_calls=0,RGB_reads=0,CUDA_initialized=False)
    return trial,lock,seal,receipts,records,statistics,shared,proof


def _node_iou(boxes,record,truth):
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    meta=record['metadata'];pixels=np.maximum(xyxy(np.asarray(boxes),meta['width'],meta['height']),0)
    return np.array([box_iou(box,truth[fid])if fid in truth else np.nan for fid,box in zip(meta['frame_ids'],pixels)])


def _summarize_statistics(queries):
    summary={}
    for name in NEW:
        cases=[row[name]for row in queries]
        summary[name]=dict(queries=len(cases),updates=_scalar_summary([row['updates']for row in cases]),
            accepted=_scalar_summary([row['accepted']for row in cases]),rejected=_scalar_summary([row['rejected']for row in cases]),
            stages={},Q_versus_immediate={},output_replacements={})
        for group in ('all','observed','unobserved','active_unobserved','inactive'):
            summary[name]['output_replacements'][group]={field:sum(row['output_replacements'][group][field]for row in cases)
                for field in ('nodes','replaced_nodes')}
        summary[name]['loss_by_attempt']=[dict(attempt=i,loss=_scalar_summary([row['loss_curve'][i]['loss']for row in cases if len(row['loss_curve'])>i]),
            weight_delta_L2=_scalar_summary([row['loss_curve'][i]['weight_delta_L2']for row in cases if len(row['loss_curve'])>i]))
            for i in range(max(row['updates']for row in cases))]
        targets=[path['Q_versus_immediate']for row in cases for path in row['target_feedback']]
        for group in ('all','active','active_unobserved'):
            count=sum(row[group]['nodes']for row in targets);changed=sum(row[group]['argmax_differences']for row in targets)
            summary[name]['Q_versus_immediate'][group]=dict(saved_path_evaluations=len(targets),nodes=count,
                argmax_differences=changed,argmax_difference_fraction=changed/count if count else None)
        for stage in ('initial','final'):
            summary[name]['stages'][stage]={}
            for group in ('all','observed','unobserved','active_unobserved'):
                values=[row[stage][group]for row in cases]
                summary[name]['stages'][stage][group]=dict(nodes=sum(row['nodes']for row in values),
                    eligible_alternative_nodes=sum(row['eligible_alternative_nodes']for row in values),
                    E_only_nodes=sum(row['E_only_nodes']for row in values),positive_margin_nodes=sum(row['positive_margin_nodes']for row in values),
                    macro_margin_mean=_scalar_summary([row['margin_mean']for row in values]),
                    macro_prior_barrier_mean=_scalar_summary([row['prior_barrier_mean']for row in values]))
    return summary


def _summarize_histories(queries):
    rows=[path for query in queries for path in query['B_history_comparison']['rows']]
    counts={field:sum(row[field]for row in rows)for field in ('active_unobserved_nodes',
        'different_saved_history_unobserved_nodes','target_argmax_differences','different_history_target_argmax_differences')}
    return dict(saved_matched_paths=len(rows),**counts,
        target_JS=_scalar_summary([row['target_JS_mean']for row in rows]),
        different_history_target_JS=_scalar_summary([row['different_history_target_JS_mean']for row in rows]),
        identical_history_target_max_abs_error=max((row['identical_history_target_max_abs_error']for row in rows if row['identical_history_target_max_abs_error']is not None),default=None),
        interpretation='Actual saved OPD and fixed-KD targets use the same frozen immediate-teacher reference continuation rule conditioned on different incoming histories. The rule does not use student-weight continuation. This is not an isolated causal memory ablation; identical complete incoming states must have identical targets.')


def _render(public):
    lines=['# HC64: evidence-conditioned A/B policy adaptation','',
        'Fixed first64 development parents, one query per parent, seed41. All methods share the exact sealed '
        'Strong time interval and the same complete-video metric. Six previous outputs are reused byte-for-byte; '
        'the six A/B outputs are independently executed and sealed before GT. No network, decoder, expert or RGB call is made.','',
        '| Method | Full vIoU % | tIoU % | Observed sIoU % | Unobserved sIoU % |',
        '|---|---:|---:|---:|---:|']
    for name in NAMES:
        m=public['means'][name];values=['NA'if m[f]is None else f'{100*m[f]:.4f}'for f in ('m_vIoU','m_tIoU','m_sIoU_observed_actual','m_sIoU_unobserved_actual')]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines+=['','| Comparison | vIoU delta pp [95% CI] | Improve / decline / same | vIoU drops >5 pp | Unobserved sIoU delta pp [95% CI] |',
        '|---|---:|---:|---:|---:|']
    def ci(v):return 'NA'if v['mean_pp']is None else f'{v["mean_pp"]:+.4f} [{v["CI95_pp"][0]:.4f}, {v["CI95_pp"][1]:.4f}]'
    for first,second in _focus_pairs():
        pair=public['paired'][first+'-'+second];v=pair['v'];s=pair['unobserved_actual_sIoU']
        lines.append(f'| {first} − {second} | {ci(v)} | {v["improved"]} / {v["declined"]} / {v["unchanged"]} | {v["declined_over5pp"]} | {ci(s)} |')
    lines+=['','Intervals use10000 paired parent draws. All signed per-query changes and severe declines are retained. '
        'Observed/unobserved IoU uses fixed actual sampled nodes with GT, independently of temporal truncation; '
        'the eight expert positions are excluded from unobserved IoU.','',
        '## Actual optimization and final action margins','',
        '| Method | Seconds/query | Updates/query | Accepted/query | Active unobserved positive-margin nodes | All unobserved replacements / nodes | Recovery / initially below0.5 | Damage / initially at least0.5 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in NEW:
        c=public['costs']['arms'][name];s=public['optimization'][name]['stages']['final']['active_unobserved'];n=public['final_node_changes'][name]['unobserved'];r=public['optimization'][name]['output_replacements']['unobserved']
        lines.append(f'| {name} | {c["seconds"]["mean"]:.5f} | {c["updates"]["mean"]:.3f} | {c["accepted_updates"]["mean"]:.3f} | '
            f'{s["positive_margin_nodes"]} / {s["eligible_alternative_nodes"]} | {r["replaced_nodes"]} / {r["nodes"]} | '
            f'{n["recovered_to_half_nodes"]} / {n["initially_below_half_nodes"]} | {n["damaged_below_half_nodes"]} / {n["initially_at_least_half_nodes"]} |')
    lines+=['','The8 matching weights are the only updated state. All arms receive the same allowance of3 outer rounds×2 sampled histories×10 inner updates. '
        'A rejected hard-readout proposal restores weights and Adam and stops; actual accepted updates can differ. '
        'Margins compare the best eligible alternative with E on the actual action support; inactive and E-only nodes remain in replacement denominators. '
        'Recovery/damage thresholds are offline descriptions of final-node GT IoU, not online acceptance rules.','',
        public['costs']['scope'],'',
        '## Consequence-aware target and saved-history changes','',
        '| Method | Active unobserved saved-state evaluations | Q versus immediate-r argmax differences |',
        '|---|---:|---:|']
    for name in NEW:
        item=public['optimization'][name]['Q_versus_immediate']['active_unobserved']
        lines.append(f'| {name} | {item["nodes"]} | {item["argmax_differences"]} |')
    h=public['B_history_comparison']
    lines+=['',f'B OPD versus fixed-KD: {h["different_saved_history_unobserved_nodes"]} matched-node evaluations have different saved histories; '
        f'{h["different_history_target_argmax_differences"]} of these also change the saved teacher-target argmax. '
        +h['interpretation'],'',
        'Initial/final margins, prior barriers, every saved loss/update curve, target-selection changes and all anonymous final-node corrections are in report.json. '
        'All results are from an exposed development panel; no per-query GT selection or further configuration choice is performed by this scorer.']
    return '\n'.join(lines)+'\n'


def evaluate(trial=TRIAL):
    from scripts.evidence_policy.run import validate
    trial,lock,seal,receipts,records,statistics,shared,proof=validate_before_labels(trial)
    labels_path=ROOT/'data/evaluation/development_labels.jsonl';labelsha=sha(labels_path)
    assert labelsha==read(ROOT/'outputs/layout_reward_dev/scope_hc2/metrics.json')['labels_sha256']
    labels={row['id']:row for row in [json.loads(line)for line in labels_path.read_text().splitlines()if line.strip()]
        if row['dataset']=='hc2'and row['id']in lock['ids']}
    assert set(labels)==set(lock['ids'])
    previous=read(ROOT/'results/box_propagation_hc64/report.json')['queries_anonymous']
    rows=[];nodes={name:[]for name in NEW};maxerror=baselineerror=0.;checks=0
    for index,record in enumerate(records):
        key=record['id'];g=labels[key];truth={int(k):v for k,v in g['boxes'].items()}
        with np.load(trial/seal['files'][key]['path'],allow_pickle=False)as p:
            interval=p['Strong_interval'].tolist();positions=p['positions8'].tolist();values={}
            observed=np.isin(np.arange(len(p['frame_ids'])),positions)
            E_iou=_node_iou(p['ERefStrong'],record,truth)
            for name in NAMES:
                values[name],error,support=score_boxes(p[name],record,truth,interval,g['interval'],positions)
                maxerror=max(maxerror,error);checks+=1
                if name in REUSED:
                    old=previous[index]['values'][name]
                    for field in ('v','t','s','observed_actual_sIoU','unobserved_actual_sIoU'):
                        if old[field]is None:assert values[name][field]is None
                        else:baselineerror=max(baselineerror,abs(values[name][field]-old[field]))
                else:
                    final_iou=_node_iou(p[name],record,truth)
                    replaced=np.any(p[name]!=p['ERefStrong'],axis=-1)
                    nodes[name].append(final_node_changes(E_iou,final_iou,observed,replaced))
        rows.append(dict(id=key,parent=g['parent'],values=values,support=support))
    assert len(rows)==len({row['parent']for row in rows})==64
    assert maxerror<2e-10 and baselineerror<2e-10 and sha(labels_path)==labelsha
    validate(trial,lock)
    assert sha(trial/'sealed.json')==proof['seal_sha256']and sha(trial/'lock.json')==proof['lock_sha256']
    assert not torch.cuda.is_initialized()
    means,counts=mean_table(rows,NAMES);signed=_signed(rows)
    proof.update(labels_sha256=labelsha,official_dense_checks=checks,official_dense_max_absolute_error=maxerror,
        reused_baseline_metric_max_absolute_error=baselineerror,all_hashes_rechecked_after_scoring=True)
    public=dict(status='completed_HC64_exposed_development_evidence_policy_AB',queries=64,parents=64,
        arm_order=NAMES,new_arms=NEW,reused_methods=REUSED,means=means,valid_query_counts=counts,
        paired=paired_stats(rows,NAMES),costs=cost_summary(receipts,NEW),shared_CPU_costs=_shared_cost_summary(shared),
        optimization=_summarize_statistics(statistics),final_node_changes={name:_aggregate_nodes(nodes[name])for name in NEW},
        B_history_comparison=_summarize_histories(statistics),
        proof=proof,config=lock['config'],signed_changes=signed,
        queries_anonymous=[dict(query=f'H{i+1:03d}',values=row['values'],optimization=statistics[i],
            final_node_changes={name:nodes[name][i]for name in NEW})for i,row in enumerate(rows)])
    dest=ROOT/'results/evidence_policy_hc64';dest.mkdir(parents=True,exist_ok=True)
    assert not(dest/'report.json').exists(),'Refuse to overwrite an earlier result'
    write_json(trial/'metrics.json',dict(public,rows=rows));write_json(dest/'report.json',public)
    (dest/'report.md').write_text(_render(public));fields=list(signed[0])
    lines=['\t'.join(fields)]+['\t'.join('NA'if row[field]is None else str(row[field])for field in fields)for row in signed]
    (dest/'signed_changes.tsv').write_text('\n'.join(lines)+'\n')
    write_json(trial/'score_receipt.json',dict(status='scored',proof=proof,metrics_sha256=sha(trial/'metrics.json'),
        public_json_sha256=sha(dest/'report.json'),public_md_sha256=sha(dest/'report.md'),signed_changes_sha256=sha(dest/'signed_changes.tsv')))
    return public


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-after-seal',action='store_true');args=parser.parse_args()
    if not args.run_after_seal:parser.error('Explicit --run-after-seal and all64 sealed predictions are required')
    result=evaluate();print(json.dumps(dict(status=result['status'],means=result['means']),indent=2))
