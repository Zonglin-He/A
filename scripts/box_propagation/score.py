"""Independent CPU scoring for the fully sealed matched HC64 box-bank run.

Importing this module reads no inputs or labels. OldOPD/OldFixedKD/OldDirect are
new executions in this run; the earlier v0 gate did not execute those arms.
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.state_opd.evaluate import read,sha,write_json,score_boxes,mean_table,paired_stats,cost_summary

NAMES=['ERefStrong','GeometryOPD','OldFixed','OldOPD','OldFixedKD','OldDirect',
       'BoxFixed','BoxOPD','BoxFixedKD','BoxDirect']
ADAPTED=['OldOPD','OldFixedKD','OldDirect','BoxOPD','BoxFixedKD','BoxDirect']


def _probabilities(features, prior_log, valid, weights):
    feature=torch.as_tensor(features,dtype=torch.float64)
    prior=torch.as_tensor(prior_log,dtype=torch.float64)
    mask=torch.as_tensor(valid,dtype=torch.bool)
    phi=torch.as_tensor(weights,dtype=torch.float64)
    return (prior+feature@phi).masked_fill(~mask,-torch.inf).softmax(-1).numpy()


def _JS(first,second):
    a,b=np.asarray(first,float),np.asarray(second,float)
    midpoint=(a+b)/2
    def term(x):
        valid=x>0
        result=np.zeros_like(x)
        result[valid]=x[valid]*np.log(x[valid]/midpoint[valid])
        return result.sum(-1)
    return .5*(term(a)+term(b))


def saved_path_comparison(opd,kd):
    """Compare saved histories, then both policies on the identical OPD state.

    KD training states are reused teacher histories. Their saved studentq can
    reflect initialization, so KD probabilities must be rebuilt from the saved
    features and the actual corresponding round's before-weights.
    """
    paired_rounds=min(len(opd['rounds']),len(kd['rounds']))
    rows=[]
    for r in range(paired_rounds):
        paths_a=opd['trace']['training_paths'][r]
        paths_b=kd['trace']['training_paths'][r]
        assert len(paths_a)==len(paths_b)==2
        theta_a=opd['rounds'][r]['before'];theta_b=kd['rounds'][r]['before']
        for path_index,(a,b)in enumerate(zip(paths_a,paths_b)):
            actions_a,actions_b=np.asarray(a['actions']),np.asarray(b['actions'])
            active=np.asarray(a['active'],bool)
            assert np.array_equal(active,np.asarray(b['active'],bool))
            memory_a,memory_b=np.asarray(a['memory_before']),np.asarray(b['memory_before'])
            previous_a,previous_b=np.asarray(a['previous_action_probs']),np.asarray(b['previous_action_probs'])
            same=(memory_a==memory_b).all(-1)&(previous_a==previous_b).all(-1)
            q_a,q_b=np.asarray(a['teacherq']),np.asarray(b['teacherq'])
            same_error=float(np.abs(q_a[same]-q_b[same]).max())if same.any()else None
            assert same_error is None or same_error<1e-12
            pa=_probabilities(a['features'],a['prior_log_probs'],a['effective_valid'],theta_a)
            pb=_probabilities(a['features'],a['prior_log_probs'],a['effective_valid'],theta_b)
            reconstruction=float(np.abs(pa-np.asarray(a['studentq'])).max())
            assert reconstruction<1e-12
            common_argmax=(pa.argmax(-1)!=pb.argmax(-1))&active
            rows.append(dict(round=r,path=path_index,nodes=len(active),active_nodes=int(active.sum()),
                own_history_action_differences=int((actions_a!=actions_b).sum()),
                own_history_active_action_differences=int(((actions_a!=actions_b)&active).sum()),
                own_history_memory_L2_mean=float(np.linalg.norm(memory_a-memory_b,axis=-1).mean()),
                own_history_teacher_JS_mean=float(_JS(q_a,q_b).mean()),
                identical_saved_history_nodes=int(same.sum()),same_history_teacher_max_abs_error=same_error,
                same_OPD_history_student_JS_mean=float(_JS(pa,pb)[active].mean())if active.any()else None,
                same_OPD_history_student_argmax_differences=int(common_argmax.sum()),
                OPD_student_probability_reconstruction_error=reconstruction,
                saved_noise_equal=np.array_equal(np.asarray(a['noise']),np.asarray(b['noise']))))
    final_a=opd['trace']['final_rollout'];final_b=kd['trace']['final_rollout']
    a,b=np.asarray(final_a['actions']),np.asarray(final_b['actions'])
    observed=np.asarray(final_a['observed'],bool)
    return dict(paired_rounds=paired_rounds,OPD_rounds=len(opd['rounds']),KD_rounds=len(kd['rounds']),
        paired_paths=len(rows),path_rows=rows,
        final_action_differences=int((a!=b).sum()),
        final_unobserved_action_differences=int(((a!=b)&~observed).sum()),
        final_box_max_abs_difference=float(np.abs(np.asarray(opd['final'])-np.asarray(kd['final'])).max()),
        final_weight_L2_difference=float(np.linalg.norm(np.asarray(opd['finaltheta'])-np.asarray(kd['finaltheta']))),
        interpretation='Own-history differences compare actual visited states. Same-OPD-history probabilities evaluate both saved round weights on exactly the same fixed features/prior/valid support; no new rollout is generated. Identical saved memories and previous actions must produce the same teacher feedback. Cached KD studentq is not mistaken for its updated policy.')


def _signed(rows):
    comparisons=[]
    for name in NAMES[2:]:
        comparisons.extend((name,base)for base in ('ERefStrong','GeometryOPD')if name!=base)
    comparisons += [('BoxFixed','OldFixed'),('BoxOPD','OldOPD'),('BoxFixedKD','OldFixedKD'),('BoxDirect','OldDirect')]
    for prefix in ('Old','Box'):
        comparisons += [(prefix+'OPD',prefix+other)for other in ('Fixed','FixedKD','Direct')]
    out=[]
    for index,row in enumerate(rows):
        for method,baseline in comparisons:
            a,b=row['values'][method],row['values'][baseline]
            v=100*(a['v']-b['v'])
            s=None if a['unobserved_actual_sIoU']is None or b['unobserved_actual_sIoU']is None else 100*(a['unobserved_actual_sIoU']-b['unobserved_actual_sIoU'])
            out.append(dict(query=f'H{index+1:03d}',method=method,baseline=baseline,vIoU_delta_pp=v,
                unobserved_sIoU_delta_pp=s,severe_vIoU_drop_over5pp=v < -5,
                severe_unobserved_sIoU_drop_over5pp=s is not None and s < -5))
    return out


def _history_summary(queries):
    result={}
    for prefix in ('Old','Box'):
        cases=[row[prefix]for row in queries]
        path_rows=[path for row in cases for path in row['path_rows']]
        finite=lambda field:[row[field]for row in path_rows if row[field]is not None]
        result[prefix]=dict(queries=len(cases),queries_with_different_final_actions=sum(row['final_action_differences']>0 for row in cases),
            final_action_differences=sum(row['final_action_differences']for row in cases),
            final_unobserved_action_differences=sum(row['final_unobserved_action_differences']for row in cases),
            paired_paths=len(path_rows),own_history_nodes=sum(row['nodes']for row in path_rows),
            own_history_action_differences=sum(row['own_history_action_differences']for row in path_rows),
            identical_saved_history_nodes=sum(row['identical_saved_history_nodes']for row in path_rows),
            same_history_teacher_max_abs_error=max(finite('same_history_teacher_max_abs_error'),default=None),
            same_OPD_history_student_JS_mean=float(np.mean(finite('same_OPD_history_student_JS_mean')))if finite('same_OPD_history_student_JS_mean')else None,
            same_OPD_history_student_argmax_differences=sum(row['same_OPD_history_student_argmax_differences']for row in path_rows),
            OPD_student_probability_reconstruction_error=max(finite('OPD_student_probability_reconstruction_error'),default=None))
    return result


def _render(public):
    lines=['# HC64: matched component-bank and native-box propagation','',
        'The fixed first64 development parents are retained, one query per parent, seed41. '
        'All ten predictions share the saved Strong time interval and identical metric implementation. '
        '**OldOPD, OldFixedKD and OldDirect are newly executed in this run on the immutable old component bank.** '
        'The earlier v0 trial stopped before adaptation; it did not already produce those results. '
        'OldFixed alone reuses its sealed EvidenceDP output. The previous failed gate and all old artifacts remain unchanged.','',
        '| Method | Full vIoU % | tIoU % | Observed sIoU % | Unobserved sIoU % |',
        '|---|---:|---:|---:|---:|---:|']
    for name in NAMES:
        m=public['means'][name]
        values=['NA'if m[field]is None else f'{100*m[field]:.4f}'for field in ('m_vIoU','m_tIoU','m_sIoU_observed_actual','m_sIoU_unobserved_actual')]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', '| Comparison | vIoU delta pp [95% CI] | Improve / decline / unchanged | Drops >5 pp | Unobserved sIoU delta pp [95% CI] |',
        '|---|---:|---:|---:|---:|']
    pairs=[('BoxFixed','OldFixed'),('BoxOPD','OldOPD'),('BoxFixedKD','OldFixedKD'),('BoxDirect','OldDirect'),
           ('BoxOPD','BoxFixed'),('BoxOPD','BoxFixedKD'),('BoxOPD','BoxDirect'),('BoxOPD','ERefStrong'),('OldOPD','ERefStrong')]
    def ci(v):return 'NA'if v['mean_pp']is None else f'{v["mean_pp"]:+.4f} [{v["CI95_pp"][0]:.4f}, {v["CI95_pp"][1]:.4f}]'
    for a,b in pairs:
        row=public['paired'][a+'-'+b];v,s=row['v'],row['unobserved_actual_sIoU']
        lines.append(f'| {a} − {b} | {ci(v)} | {v["improved"]} / {v["declined"]} / {v["unchanged"]} | '
            f'{v["declined_over5pp"]} | {ci(s)} |')
    lines += ['', 'All parent intervals use10000 paired draws. Observed and unobserved spatial IoU use the fixed actual '
        'sampled nodes with GT; all eight expert positions are excluded from unobserved IoU, independently of temporal truncation. '
        'Every positive, negative, unchanged and severe decline is retained in report.json and signed_changes.tsv.','',
        '## Actual updates and cached incremental costs','',public['costs']['scope'],'',
        '| Method | Seconds/query | Attempted updates | Accepted | Rejected | Backwards | Heavy forwards |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ADAPTED:
        fields=public['costs']['arms'][arm]
        values=['NA'if field not in fields else f'{fields[field]["mean"]:.6f}'for field in
                ('seconds','updates','accepted_updates','rejected_updates','backward','heavy_forwards')]
        lines.append('| '+arm+' | '+' | '.join(values)+' |')
    lines += ['', 'Each adapted arm has the same locked v0 policy, optimizer, seed and allowance of at most3 rounds×2 paths. '
        'Fixed-score rejection restores weights/Adam and stops, so actual update counts can differ. '
        'The experiment uses existing Source/ReferDINO caches; these costs are not full deployed inference latency.','',
        '## OPD versus fixed-KD histories','',
        'Actual visited-history differences and policy differences on exactly the same saved OPD history are reported separately. '
        'Identical history nodes must give identical teacher feedback. Same-history probability evaluation only multiplies '
        'saved features by saved weights; it generates no additional candidate path. Cached fixed-KD studentq is not mistaken '
        'for its updated policy.','',
        '| Bank | Queries with different final actions | Different unobserved final actions | Same-state policy argmax differences |',
        '|---|---:|---:|---:|']
    for bank,row in public['history_comparison'].items():
        lines.append(f'| {bank} | {row["queries_with_different_final_actions"]} | '
            f'{row["final_unobserved_action_differences"]} | {row["same_OPD_history_student_argmax_differences"]} |')
    lines += ['', 'These are exposed-development results. No GT chooses per-query predictions, stopping steps, '
        'candidate slots or parameters. Fixed-score acceptance is not a guarantee of GT improvement; Direct uses '
        'the pre-existing fully differentiable soft-history relaxation and the same hard final decode.']
    if 'candidate_oracle' in public:
        lines += ['', '## Offline candidate coverage', '',public['candidate_oracle']['definition'],'',
            '| Bank | Fixed unobserved GT nodes | E-only nodes | Mean oracle node-IoU gain pp | GT nodewise full-vIoU readout % |',
            '|---|---:|---:|---:|---:|']
        for bank,row in public['candidate_oracle']['banks'].items():
            gain=row['macro_unobserved_node_IoU_gain']['mean']
            value=row['GT_nodewise_full_vIoU_readout_mean']
            lines.append(f'| {bank} | {row["nodes"]} | {row["E_only_nodes"]} | '
                f'{"NA"if gain is None else f"{100*gain:.4f}"} | {"NA"if value is None else f"{100*value:.4f}"} |')
    return '\n'.join(lines)+'\n'


def _validate_adaptation(output):
    costs=output['costs'];rounds=output['rounds']
    assert 1<=len(rounds)<=3
    assert costs['updates']==costs['backward']==costs['state_only_backwards']==len(rounds)
    assert costs['accepted_updates']+costs['rejected_updates']==len(rounds)
    assert all(costs[name]==0 for name in ('heavy_forwards','decoder_calls','decoder_backwards','encoder_calls','expert_calls'))
    assert len(output['trace']['training_paths'])==len(rounds)
    assert all(len(paths)==2 for paths in output['trace']['training_paths'])
    assert torch.as_tensor(output['initialtheta']).eq(0).all()
    for row in rounds:
        assert np.isfinite(np.asarray(row['gradient'])).all()
        if row['rolled_back']:
            assert np.array_equal(np.asarray(row['before']),np.asarray(row['after']))
            assert row['optimizer_before_sha256']==row['optimizer_after_sha256']
    assert not rounds[-1]['rolled_back'] or costs['rejected_updates']==1


def validate_before_labels(trial):
    from scripts.box_propagation.run import validate
    trial=Path(trial).resolve();assert trial==(ROOT/'outputs/box_propagation_hc64').resolve()
    lock=read(trial/'lock.json');receipts=validate(trial,lock)
    seal=read(trial/'sealed.json')
    assert seal['queries']==64 and seal['GT_read']is False and seal['fingerprint']==lock['fingerprint']
    assert seal['lock_sha256']==sha(trial/'lock.json')
    manifest=ROOT/'data/evaluation/development.jsonl'
    assert sha(manifest)==lock['manifest_sha256']
    records=[json.loads(line)for line in manifest.read_text().splitlines()if line.strip()]
    records=[record for record in records if record['dataset']=='hc2']
    assert len(records)==128
    records=records[:64];assert [record['id']for record in records]==lock['ids']
    assert len(receipts)==len(seal['files'])==64
    old=ROOT/'outputs/propagation_v0_hc64';oldseal=read(old/'stage1/sealed.json')
    assert sha(old/'lock.json')==lock['old_lock_sha256']
    assert sha(old/'stage1/sealed.json')==lock['old_seal_sha256']
    histories=[];cost_receipts={};shared=[]
    for record,item in zip(records,lock['inputs']):
        key=record['id'];receipt=receipts[key]
        path=trial/seal['files'][key]['path']
        assert path.resolve().is_relative_to(trial)
        assert sha(path)==seal['files'][key]['sha256']
        rpath=trial/seal['files'][key]['receipt_path']
        assert sha(rpath)==seal['files'][key]['receipt_sha256']
        assert receipt['GT_read']is False
        for stem in ('bank','trace'):
            p=(trial/receipt[stem+'_path']).resolve();assert p.is_relative_to(trial)
            assert sha(p)==receipt[stem+'_sha256']
        oldreceipt=read(old/'stage1/predictions'/f'{key}.json')
        oldbank=old/'stage1'/oldreceipt['bank_path']
        assert sha(oldbank)==receipt['old_bank_sha256']==oldreceipt['bank_sha256']
        bank=torch.load(trial/receipt['bank_path'],map_location='cpu',weights_only=False)
        trace=torch.load(trial/receipt['trace_path'],map_location='cpu',weights_only=False)
        assert bank['GT_read']is False
        with np.load(path,allow_pickle=False)as p,np.load(old/'stage1'/oldseal['files'][key]['path'],allow_pickle=False)as previous:
            assert p['frame_ids'].tolist()==record['metadata']['frame_ids']
            for field in ('frame_ids','positions8','Strong_interval','ERefStrong','GeometryOPD'):
                assert np.array_equal(p[field],previous[field])
            assert np.array_equal(p['OldFixed'],previous['EvidenceDP'])
            for name in NAMES:
                assert p[name].shape==p['ERefStrong'].shape and np.isfinite(p[name]).all()and(p[name][:,2:]>0).all()
            for name in ADAPTED:
                _validate_adaptation(trace[name]);assert np.array_equal(p[name],np.asarray(trace[name]['final']))
                assert trace[name]['costs']==receipt['costs'][name]
            assert np.array_equal(p['BoxFixed'],np.asarray(trace['BoxFixed']['final']))
        histories.append({prefix:saved_path_comparison(trace[prefix+'OPD'],trace[prefix+'FixedKD'])for prefix in ('Old','Box')})
        cost_receipts[key]=dict(costs={name:trace[name]['costs']for name in ADAPTED+['BoxFixed']})
        shared.append(dict(receipt.get('timings',{}),**({'total_seconds':receipt['total_seconds']}if'total_seconds'in receipt else{})))
    assert not torch.cuda.is_initialized()
    proof=dict(fingerprint=lock['fingerprint'],seal_sha256=sha(trial/'sealed.json'),lock_sha256=sha(trial/'lock.json'),
        manifest_sha256=sha(manifest),old_lock_sha256=lock['old_lock_sha256'],old_seal_sha256=lock['old_seal_sha256'],
        old_gate_sha256=sha(old/'gate.json'),all64_predictions_sources_and_traces_validated_before_labels=True,
        all_old_adaptation_arms_are_new_executions=True,old_failed_gate_unchanged=True,
        user_new_authorization_is_separate_from_old_failed_gate=True,GT_used_for_prediction_or_parameter_choice=False,
        model_calls=0,decoder_calls=0,expert_calls=0,RGB_reads=0,CUDA_initialized=False)
    return trial,lock,seal,receipts,records,histories,cost_receipts,shared,proof


def _node_ious(boxes,record,truth):
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    boxes=np.asarray(boxes,float);meta=record['metadata']
    pixels=np.maximum(xyxy(boxes,meta['width'],meta['height']),0)
    result=np.full(boxes.shape[:-1],np.nan)
    for t,fid in enumerate(meta['frame_ids']):
        if fid in truth:result[t]=[box_iou(box,truth[fid])for box in pixels[t]]
    return result


def _eligible_valid(bank):
    """Stored proposals outside the bank's hard action support remain fallback."""
    valid=np.array(bank['valid'],dtype=bool,copy=True)
    active=np.asarray(bank['active'],dtype=bool)
    assert valid.ndim==2 and active.shape==valid.shape[:1] and valid[:,0].all()
    valid[~active,1:]=False
    return valid


def _oracle(bank,record,truth,interval,span,positions):
    from scripts.propagation_v0.diagnose import oracle_unobserved
    candidate_iou=_node_ious(bank['boxes'],record,truth)
    has_GT=np.isfinite(candidate_iou[:,0])
    o=oracle_unobserved(candidate_iou,_eligible_valid(bank),bank['observed'],has_GT)
    chosen=np.asarray(bank['boxes'])[np.arange(len(has_GT)),o['selected_slots']]
    values,error,_=score_boxes(chosen,record,truth,interval,span,positions)
    result={key:value for key,value in o.items()if key not in ('selected_slots','node_mask')}
    result['GT_nodewise_full_vIoU_readout']=values['v']
    return result,error


def _shared_cost_summary(rows):
    result={}
    fields=sorted({key for row in rows for key,value in row.items()if isinstance(value,(int,float))})
    for field in fields:
        values=[float(row[field])for row in rows if field in row and isinstance(row[field],(int,float))]
        assert np.isfinite(values).all()and all(value>=0 for value in values)
        result[field]=dict(valid_queries=len(values),mean=float(np.mean(values)),median=float(np.median(values)),
            total=float(np.sum(values)),p90=float(np.quantile(values,.9)),maximum=max(values))
    return result


def evaluate(trial=ROOT/'outputs/box_propagation_hc64'):
    from scripts.box_propagation.run import validate
    from scripts.propagation_v0.diagnose import bootstrap_mean
    trial,lock,seal,receipts,records,histories,cost_receipts,shared,proof=validate_before_labels(trial)
    labels_path=ROOT/'data/evaluation/development_labels.jsonl';labelsha=sha(labels_path)
    assert labelsha==read(ROOT/'outputs/layout_reward_dev/scope_hc2/metrics.json')['labels_sha256']
    labels={row['id']:row for row in [json.loads(line)for line in labels_path.read_text().splitlines()if line.strip()]
            if row['dataset']=='hc2'and row['id']in lock['ids']}
    assert set(labels)==set(lock['ids'])
    old=ROOT/'outputs/propagation_v0_hc64';oldrows=read(ROOT/'results/propagation_v0_hc64/diagnostics.json')['queries_anonymous']
    rows=[];maxerror=baselineerror=0.;checks=0;oracle_rows={'Old':[],'Box':[]}
    for index,record in enumerate(records):
        key=record['id'];g=labels[key];truth={int(k):v for k,v in g['boxes'].items()}
        with np.load(trial/seal['files'][key]['path'],allow_pickle=False)as p:
            positions=p['positions8'].tolist();interval=p['Strong_interval'].tolist();values={}
            for name in NAMES:
                values[name],error,support=score_boxes(p[name],record,truth,interval,g['interval'],positions)
                maxerror=max(maxerror,error);checks+=1
            for name,priorname in [('ERefStrong','ERefStrong'),('GeometryOPD','GeometryOPD'),('OldFixed','EvidenceDP')]:
                prior=oldrows[index]['values'][priorname]
                for field in ('v','t','s','observed_actual_sIoU','unobserved_actual_sIoU'):
                    if prior[field]is None:assert values[name][field]is None
                    else:baselineerror=max(baselineerror,abs(values[name][field]-prior[field]))
            newbank=torch.load(trial/receipts[key]['bank_path'],map_location='cpu',weights_only=False)
            oldreceipt=read(old/'stage1/predictions'/f'{key}.json')
            oldbank=torch.load(old/'stage1'/oldreceipt['bank_path'],map_location='cpu',weights_only=False)
            for prefix,bank in [('Old',oldbank),('Box',newbank)]:
                oracle,error=_oracle(bank,record,truth,interval,g['interval'],positions)
                oracle_rows[prefix].append(oracle);maxerror=max(maxerror,error);checks+=1
        rows.append(dict(id=key,parent=g['parent'],values=values,support=support))
    assert len(rows)==len({row['parent']for row in rows})==64
    assert maxerror<2e-10 and baselineerror<2e-10 and sha(labels_path)==labelsha
    validate(trial,lock)
    assert sha(trial/'sealed.json')==proof['seal_sha256']and sha(trial/'lock.json')==proof['lock_sha256']
    assert sha(old/'gate.json')==proof['old_gate_sha256']and not torch.cuda.is_initialized()
    means,counts=mean_table(rows,NAMES);signed=_signed(rows)
    oracle=dict(definition='GT independently selects a candidate from the actual eligible action support only at fixed unobserved sampled nodes with GT, '
        'including all E-only fallback nodes. At inactive nodes every stored non-E proposal is excluded, so outside-event '
        'and confidence-rejected proposals do not count as actionable coverage. All other nodes retain E. This is an offline diagnostic, not an online method '
        'and not the exact global maximum of interpolated complete-video vIoU.',banks={})
    for prefix,values in oracle_rows.items():
        oracle['banks'][prefix]=dict(nodes=sum(row['nodes']for row in values),E_only_nodes=sum(row['E_only_nodes']for row in values),
            over5pp_nodes=sum(row['over5pp_nodes']for row in values),
            macro_unobserved_node_IoU_gain=bootstrap_mean([row['mean_improvement']for row in values]),
            GT_nodewise_full_vIoU_readout_mean=float(np.mean([row['GT_nodewise_full_vIoU_readout']for row in values])))
    oracle['Box_minus_Old_node_oracle_gain']=bootstrap_mean([None if a['mean_improvement']is None or b['mean_improvement']is None else a['mean_improvement']-b['mean_improvement']for a,b in zip(oracle_rows['Box'],oracle_rows['Old'])])
    proof.update(labels_sha256=labelsha,official_dense_checks=checks,official_dense_max_absolute_error=maxerror,
        old_baseline_metric_max_absolute_error=baselineerror,all_hashes_rechecked_after_scoring=True)
    public=dict(status='completed_HC64_exposed_development_matched_box_bank',queries=64,parents=64,
        arm_order=NAMES,newly_executed_adaptation_arms=ADAPTED,
        reused_methods=['ERefStrong','GeometryOPD','OldFixed'],
        means=means,valid_query_counts=counts,paired=paired_stats(rows,NAMES),costs=cost_summary(cost_receipts,ADAPTED+['BoxFixed']),
        shared_CPU_costs=_shared_cost_summary(shared),history_comparison=_history_summary(histories),candidate_oracle=oracle,
        proof=proof,config=lock['config'],signed_changes=signed,
        queries_anonymous=[dict(query=f'H{i+1:03d}',values=row['values'],history_comparison=histories[i],
            candidate_oracle={prefix:oracle_rows[prefix][i]for prefix in ('Old','Box')})for i,row in enumerate(rows)])
    dest=ROOT/'results/box_propagation_hc64';dest.mkdir(parents=True,exist_ok=True)
    assert not(dest/'report.json').exists(),'Refuse to overwrite an earlier result'
    write_json(trial/'metrics.json',dict(public,rows=rows))
    write_json(dest/'report.json',public);(dest/'report.md').write_text(_render(public))
    fields=list(signed[0]);lines=['\t'.join(fields)]
    lines += ['\t'.join('NA'if row[field]is None else str(row[field])for field in fields)for row in signed]
    (dest/'signed_changes.tsv').write_text('\n'.join(lines)+'\n')
    write_json(trial/'score_receipt.json',dict(status='scored',proof=proof,metrics_sha256=sha(trial/'metrics.json'),
        public_json_sha256=sha(dest/'report.json'),public_md_sha256=sha(dest/'report.md'),signed_changes_sha256=sha(dest/'signed_changes.tsv')))
    return public


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-after-seal',action='store_true');args=parser.parse_args()
    if not args.run_after_seal:parser.error('Scoring requires explicit --run-after-seal and all64 sealed predictions')
    result=evaluate();print(json.dumps(dict(status=result['status'],means=result['means'],history_comparison=result['history_comparison']),indent=2))
