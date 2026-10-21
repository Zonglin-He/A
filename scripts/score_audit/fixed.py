"""Read-only fixed-v0 score audit on all64 sealed development queries.

No network, RGB, expert, decoder, adaptation, replacement candidate bank or
policy rollout is executed. Saved choices are evaluated; mask components are
recovered only for the explicitly marked observation-geometry diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.state_opd.evaluate import read, sha, write_json, score_boxes
from scripts.propagation_v0.run import validate_phase
from scripts.propagation_v0.bank import component_boxes, iou as normalized_iou
from scripts.propagation_v0.diagnose import distance_bins, bootstrap_mean, bootstrap_ratio

CONTROLS = ('MemoryGreedy', 'NoMemoryGreedy', 'EvidenceDP')
TERMS = ('support', 'incoming_link', 'memory', 'negative_guard')
EPS = 1e-12


def path_score(unary, pair, actions):
    unary, pair, actions = np.asarray(unary), np.asarray(pair), np.asarray(actions, np.int64)
    nodes = np.arange(len(actions))
    return float(unary[nodes, actions].sum() + pair[nodes[:-1], actions[:-1], actions[1:]].sum())


def single_replacement_delta(unary, pair, context, node, candidate):
    """S(context with a_t=candidate) - S(context with a_t=E), both edges."""
    unary, pair = np.asarray(unary), np.asarray(pair)
    context = np.asarray(context, np.int64)
    own = float(unary[node, candidate] - unary[node, 0])
    incoming = float(pair[node-1, context[node-1], candidate] - pair[node-1, context[node-1], 0]) if node else 0.
    outgoing = float(pair[node, candidate, context[node+1]] - pair[node, 0, context[node+1]]) if node+1 < len(context) else 0.
    changed, original = context.copy(), context.copy()
    changed[node], original[node] = candidate, 0
    expected = path_score(unary, pair, changed) - path_score(unary, pair, original)
    result = own + incoming + outgoing
    assert abs(result - expected) < 1e-12
    return dict(unary=own, incoming=incoming, outgoing=outgoing, total=result,
                full_score_difference=expected, reconstruction_error=abs(result-expected))


def relative_feedback(bank, path, node, candidate):
    """Same saved state; no alternate history or teacher target is rolled out."""
    features = F.normalize(torch.as_tensor(bank['features']), dim=-1, eps=1e-12)
    memory = torch.as_tensor(path['memory_before'])[node]
    previous = torch.as_tensor(path['previous_action_probs'])[node]
    link = previous @ torch.as_tensor(bank['link'])[node-1] if node else features.new_zeros(3)
    u, guard = torch.as_tensor(bank['projection'])[node], torch.as_tensor(bank['guard'])[node]
    cosine = features[node] @ memory
    terms = dict(support=float(u[candidate]-u[0]), incoming_link=.25*float(link[candidate]-link[0]),
                 memory=.1*float(cosine[candidate]-cosine[0]), negative_guard=-.1*float(guard[candidate]-guard[0]))
    delta = sum(terms.values())
    saved = torch.as_tensor(path['r'])[node]
    expected = float(saved[candidate]-saved[0])
    assert abs(delta-expected) < 1e-12
    positive = {key:value for key,value in terms.items() if value > 0}
    return dict(terms=terms, margin=delta, saved_margin=expected,
                reconstruction_error=abs(delta-expected),
                largest_positive_term=max(positive,key=positive.get) if positive else 'none')


def outcome_summary(rows, query_count=64):
    """Denominator is actual non-E replacements with GT, not all candidate pairs."""
    delta = np.asarray([row['delta_IoU_pp'] for row in rows], float)
    harmed = delta < -1e-8
    correct, total = [], []
    for index in range(query_count):
        values = [row for row in rows if row['query_index'] == index]
        correct.append(sum(row['delta_IoU_pp'] < -1e-8 for row in values)); total.append(len(values))
    return dict(replacements_with_GT=len(rows), harmed=int(harmed.sum()),
        harm_fraction=float(harmed.mean()) if len(delta) else None,
        harm_fraction_parent_CI=bootstrap_ratio(correct,total),
        improved=int((delta > 1e-8).sum()), unchanged=int((np.abs(delta) <= 1e-8).sum()),
        severe_drop_over5pp=int((delta < -5).sum()), severe_drop_over10pp=int((delta < -10).sum()),
        severe_drop_over20pp=int((delta < -20).sum()),
        mean_delta_IoU_pp=float(delta.mean()) if len(delta) else None,
        median_delta_IoU_pp=float(np.median(delta)) if len(delta) else None,
        p10_delta_IoU_pp=float(np.quantile(delta,.1)) if len(delta) else None,
        worst_delta_IoU_pp=float(delta.min()) if len(delta) else None,
        mean_loss_among_harmed_pp=float(delta[harmed].mean()) if harmed.any() else None)


def _iou(boxes, record, truth):
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    boxes = np.asarray(boxes,float)
    meta = record['metadata']
    pixels = np.maximum(xyxy(boxes,meta['width'],meta['height']),0)
    result = np.full(boxes.shape[:-1],np.nan)
    for t,fid in enumerate(meta['frame_ids']):
        if fid in truth:
            if boxes.ndim == 2: result[t] = box_iou(pixels[t],truth[fid])
            else: result[t] = [box_iou(box,truth[fid]) for box in pixels[t]]
    return result


def _one_box_iou(box, gt, record):
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    meta=record['metadata']
    return float(box_iou(np.maximum(xyxy(np.asarray(box),meta['width'],meta['height']),0),gt))


def _path(control):
    return control if 'actions' in control else control['trace']


def _group(node, observed, bins):
    if observed[node]:return 'observed'
    return next(name for name in ('near','middle','far') if bins[name][node])


def _margin_summary(rows):
    selected = [row for row in rows if row['teacher_prefers_over_E']]
    return dict(comparisons=len(rows), teacher_prefers_alternative=len(selected),
        GT_harmed_among_preferred=sum(row['delta_IoU_pp'] < -1e-8 for row in selected),
        GT_improved_among_preferred=sum(row['delta_IoU_pp'] > 1e-8 for row in selected),
        harm_fraction_among_preferred=(float(np.mean([row['delta_IoU_pp'] < -1e-8 for row in selected])) if selected else None),
        mean_delta_IoU_pp_among_preferred=(float(np.mean([row['delta_IoU_pp'] for row in selected])) if selected else None),
        mean_components_among_preferred={term:float(np.mean([row['feedback']['terms'][term] for row in selected])) if selected else None for term in TERMS},
        mean_components_among_harmed_preferred={term:float(np.mean([row['feedback']['terms'][term] for row in selected if row['delta_IoU_pp'] < -1e-8])) if any(row['delta_IoU_pp'] < -1e-8 for row in selected) else None for term in TERMS},
        largest_positive_term_among_harmed_preferred=dict(Counter(row['feedback']['largest_positive_term'] for row in selected if row['delta_IoU_pp'] < -1e-8)))


def _observation_summary(rows):
    def stats(name):
        values=[row[name] for row in rows if row.get(name) is not None]
        return dict(valid_nodes=len(values),mean=float(np.mean(values)) if values else None,
                    median=float(np.median(values)) if values else None)
    eligible=[row for row in rows if row.get('native_IoU') is not None]
    d=[100*(row['native_IoU']-row['E_IoU']) for row in eligible]
    return dict(observation_positions=len(rows),with_GT=len(eligible),
        mean_quality={name:stats(name) for name in ['Source_IoU','E_IoU','native_IoU','top_mass_component_IoU','best_component_GT_IoU']},
        native_minus_E_mean_pp=float(np.mean(d)) if d else None,
        native_worse_than_E=sum(value < -1e-8 for value in d),
        native_better_than_E=sum(value > 1e-8 for value in d),
        native_over5pp_worse_than_E=sum(value < -5 for value in d),
        native_status_counts=dict(Counter(row['native_bank_status'] for row in rows)))


def _render(public):
    lines=['# Fixed v0: accepted replacement and feedback audit','',
        'All64 sealed stage-one queries are evaluated without changing candidates, histories, parameters, '
        'predictions or the failed development gate. No OPD/KD/Direct adaptation is run. The primary denominator '
        'below is actual saved non-E replacements at nodes with GT; it is not the full candidate pair pool.','',
        '| Saved control | Group | Replacements with GT | Harmed | Harm % | Drop >5 pp | Mean ΔIoU pp | Mean harmed ΔIoU pp |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda value:'NA' if value is None else f'{value:.4f}'
    for name,groups in public['accepted_replacements'].items():
        for group,row in groups.items():
            lines.append(f'| {name} | {group} | {row["replacements_with_GT"]} | {row["harmed"]} | '
                f'{fmt(None if row["harm_fraction"] is None else 100*row["harm_fraction"])} | {row["severe_drop_over5pp"]} | '
                f'{fmt(row["mean_delta_IoU_pp"])} | {fmt(row["mean_loss_among_harmed_pp"])} |')
    lines += ['', 'Observation/near/middle/far are mutually exclusive. The latter bins are fixed nearest-anchor '
        'physical distance divided by the median anchor gap: (0,.25], (.25,.5], and >.5. All replacement cases '
        'without GT remain counted separately, not treated as correct or incorrect.','',
        '## Same-state feedback and the fixed path score','',
        '`r(a)−r(E) = Δu + .25 Δincoming_link + .1 Δmemory_cos − .1 ΔG` is reconstructed at the '
        'saved memory and previous action. No alternate history is executed. The single-replacement fixed-score '
        'difference includes its unary, incoming edge and outgoing edge, and is independently checked against '
        'two complete fixed-table score evaluations. Both all-E-neighbor and saved-path-neighbor contexts are retained. '
        'Single-node differences must not be summed to infer the full path difference because edges overlap.','',
        '| Saved state subset | Comparisons | Teacher prefers alternative | Harmed preferred | Harm % among preferred |',
        '|---|---:|---:|---:|---:|']
    for name,row in public['initial_feedback_margins'].items():
        lines.append(f'| {name} | {row["comparisons"]} | {row["teacher_prefers_alternative"]} | '
            f'{row["GT_harmed_among_preferred"]} | {fmt(None if row["harm_fraction_among_preferred"] is None else 100*row["harm_fraction_among_preferred"])} |')
    lines += ['', '## Observation-frame geometry','',
        'Native ReferDINO boxes are evaluated at all eight saved positions even when E-NMS removed them or the '
        'node was inactive. Components are recovered from the saved maps using the exact strict >.5-of-own-peak '
        'four-connected rule, deduplicating identical two-direction observation components. They are audit geometry, '
        'not new method outputs. The best component uses GT and is explicitly an oracle; top-mass selection uses no GT.','',
        '| Observation subset | GT nodes | Source IoU % | E IoU % | Native IoU % | Top-mass component IoU % | Best component GT IoU % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group,row in public['observation_geometry'].items():
        fields=['Source_IoU','E_IoU','native_IoU','top_mass_component_IoU','best_component_GT_IoU']
        values=[fmt(None if row['mean_quality'][field]['mean'] is None else 100*row['mean_quality'][field]['mean']) for field in fields]
        lines.append('| '+group+' | '+str(row['with_GT'])+' | '+' | '.join(values)+' |')
    lines += ['', 'All64 anonymous per-query signed results and the largest observed harms are retained in JSON. '
        'Node-level private details remain in outputs/score_audit; no GT coordinates, raw masks/features or private '
        'dataset identifiers are exported. A positive proxy score is not a claim of GT improvement.']
    return '\n'.join(lines)+'\n'


def run():
    torch.set_num_threads(1);begin=__import__('time').perf_counter()
    trial=ROOT/'outputs/propagation_v0_hc64';phase=trial/'stage1'
    lock=read(trial/'lock.json');receipts=validate_phase(trial,'stage1',lock)
    assert len(receipts)==64
    seal=read(phase/'sealed.json');seal_sha=sha(phase/'sealed.json')
    gate_sha=sha(trial/'gate.json');formal_sha=sha(ROOT/'outputs/formal/state.json')
    config_sha=sha(ROOT/'configs/default.json')
    source_sha=sha(Path(__file__))
    manifest=ROOT/'data/evaluation/development.jsonl'
    assert sha(manifest)==lock['manifest_sha256']
    records=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    records=[row for row in records if row['dataset']=='hc2'][:64]
    assert [row['id'] for row in records]==lock['ids']
    # All input, trace and prediction hashes passed before this first label read.
    labels_path=ROOT/'data/evaluation/development_labels.jsonl';labels_sha=sha(labels_path)
    old=ROOT/'outputs/layout_reward_dev'
    assert labels_sha==read(old/'scope_hc2/metrics.json')['labels_sha256']
    labels={row['id']:row for row in [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()] if row['dataset']=='hc2'}
    old_diag=read(ROOT/'results/propagation_v0_hc64/diagnostics.json')
    assert old_diag['proof']['fingerprint']==lock['fingerprint'] and old_diag['proof']['stage1_seal_sha256']==seal_sha
    replacement_rows=[];feedback_rows=[];observation_rows=[];queries=[]
    maximum_r_error=maximum_S_error=maximum_metric_error=maximum_dense_error=0.
    methods=('ERefStrong','GeometryOPD',*CONTROLS)
    for index,(record,item) in enumerate(zip(records,lock['inputs'])):
        key=record['id'];anon=f'H{index+1:03d}';rr=receipts[key]
        bank=torch.load(phase/rr['bank_path'],map_location='cpu',weights_only=False)
        trace=torch.load(phase/rr['trace_path'],map_location='cpu',weights_only=False)
        evidence=torch.load(phase/rr['evidence_path'],map_location='cpu',weights_only=False)
        cache=torch.load(old/'student_cache'/f'{key}.pt',map_location='cpu',weights_only=False)
        assert sha(old/'student_cache'/f'{key}.pt')==item['cache_sha256']
        assert bank['GT_read'] is cache['GT_read'] is False
        with np.load(phase/seal['files'][key]['path'],allow_pickle=False) as f:pred={name:f[name].copy() for name in f.files}
        label=labels[key];truth={int(fid):box for fid,box in label['boxes'].items()}
        ids=pred['frame_ids'];positions=pred['positions8'].tolist();observed=np.asarray(bank['observed'],bool)
        bins=distance_bins(ids,positions);candidate_iou=_iou(bank['boxes'],record,truth)
        has_GT=np.isfinite(candidate_iou[:,0]);unary,pair=np.asarray(bank['score_unary']),np.asarray(bank['score_pair'])
        zeros=np.zeros(len(ids),np.int64);E_score=path_score(unary,pair,zeros)
        values={}
        for name in methods:
            values[name],error,_=score_boxes(pred[name],record,truth,pred['Strong_interval'].tolist(),label['interval'],positions)
            maximum_dense_error=max(maximum_dense_error,error)
            prior=old_diag['queries_anonymous'][index]['values'][name]
            for field in ('v','t','s','observed_actual_sIoU','unobserved_actual_sIoU'):
                if prior[field] is None:assert values[name][field] is None
                else:maximum_metric_error=max(maximum_metric_error,abs(values[name][field]-prior[field]))
        per_query={}
        for name in CONTROLS:
            saved=_path(trace['controls'][name]);actions=np.asarray(saved['actions'],np.int64)
            assert np.array_equal(pred[name],np.asarray(bank['boxes'])[np.arange(len(ids)),actions])
            count_noGT=0;new_rows=[]
            for node in np.flatnonzero(actions!=0):
                slot=int(actions[node])
                if not has_GT[node]:count_noGT+=1;continue
                contexts={context:single_replacement_delta(unary,pair,choices,node,slot)
                          for context,choices in [('all_E',zeros),('saved_path',actions)]}
                maximum_S_error=max(maximum_S_error,*[x['reconstruction_error'] for x in contexts.values()])
                row=dict(query=anon,query_index=index,node_index=int(node),slot=slot,group=_group(node,observed,bins),
                    E_IoU=float(candidate_iou[node,0]),chosen_IoU=float(candidate_iou[node,slot]),
                    delta_IoU_pp=float(100*(candidate_iou[node,slot]-candidate_iou[node,0])),
                    normalized_nearest_anchor_distance=float(bins['normalized_distance'][node]),method=name,
                    fixed_S_delta=contexts)
                if 'r' in saved:
                    row['feedback']=relative_feedback(bank,saved,node,slot)
                    maximum_r_error=max(maximum_r_error,row['feedback']['reconstruction_error'])
                new_rows.append(row);replacement_rows.append(row)
            per_query[name]=dict(total_nonE_nodes=int((actions!=0).sum()),replacement_without_GT=count_noGT,
                replacement_with_GT=len(new_rows),harmed=sum(row['delta_IoU_pp'] < -1e-8 for row in new_rows),
                mean_delta_IoU_pp=float(np.mean([row['delta_IoU_pp'] for row in new_rows])) if new_rows else None,
                fixed_S_gain=path_score(unary,pair,actions)-E_score,
                vIoU_delta_vs_E_pp=100*(values[name]['v']-values['ERefStrong']['v']),
                unobserved_sIoU_delta_vs_E_pp=(None if values[name]['unobserved_actual_sIoU'] is None else
                    100*(values[name]['unobserved_actual_sIoU']-values['ERefStrong']['unobserved_actual_sIoU'])))
        for path_index,path in enumerate(trace['diagnostic_paths']):
            actions=np.asarray(path['actions'],np.int64);valid=np.asarray(path['effective_valid'],bool)
            for node in np.flatnonzero(has_GT):
                for slot in np.flatnonzero(valid[node,1:])+1:
                    components=relative_feedback(bank,path,node,int(slot))
                    maximum_r_error=max(maximum_r_error,components['reconstruction_error'])
                    contexts={context:single_replacement_delta(unary,pair,choices,node,int(slot))
                              for context,choices in [('all_E',zeros),('saved_path',actions)]}
                    maximum_S_error=max(maximum_S_error,*[x['reconstruction_error'] for x in contexts.values()])
                    feedback_rows.append(dict(query=anon,query_index=index,path_index=path_index,node_index=int(node),slot=int(slot),
                        group=_group(node,observed,bins),feedback=components,
                        teacher_prefers_over_E=components['margin']>EPS,selected_by_initial_student=bool(actions[node]==slot),
                        delta_IoU_pp=float(100*(candidate_iou[node,slot]-candidate_iou[node,0])),fixed_S_delta=contexts))
        teacher=torch.as_tensor(cache['teacher']['boxes']).double()
        assert teacher.shape==(8,4)
        for j,node in enumerate(positions):
            native=teacher[j];available=bank['valid'][node]
            retained=any(torch.equal(native,bank['boxes'][node,k]) for k in range(1,3) if available[k])
            if retained:status='retained'
            elif not bool(bank['active'][node]):status='inactive'
            elif float(normalized_iou(native,bank['E'][node]))>=.7:status='removed_by_E_NMS'
            else:status='unexpected_missing_native'
            assert status!='unexpected_missing_native', (anon,node)
            maps=bank['side_maps'][node];valid_grid=evidence['valid'][node].reshape(maps.shape[-2:])
            components=[]
            for side in (0,1):
                if not bank['side_available'][node,side]:continue
                for component in component_boxes(maps[side],valid_grid):
                    if not any(torch.equal(component['box'],old_component['box']) for old_component in components):components.append(component)
            components.sort(key=lambda x:(-x['mass'],tuple(x['box'].tolist())))
            row=dict(query=anon,query_index=index,node_index=int(node),observation_index=j,native_bank_status=status,
                component_count=len(components),native_retained=retained,has_GT=int(ids[node]) in truth)
            if row['has_GT']:
                gt=truth[int(ids[node])]
                component_ious=[_one_box_iou(c['box'],gt,record) for c in components]
                row.update(Source_IoU=_one_box_iou(cache['Source'][node],gt,record),
                    E_IoU=_one_box_iou(bank['E'][node],gt,record),native_IoU=_one_box_iou(native,gt,record),
                    top_mass_component_IoU=component_ious[0] if component_ious else None,
                    best_component_GT_IoU=max(component_ious) if component_ious else None,
                    all_component_IoUs=component_ious)
            observation_rows.append(row)
        queries.append(dict(query=anon,values=values,accepted_replacements=per_query,
            observation_status_counts=dict(Counter(row['native_bank_status'] for row in observation_rows if row['query_index']==index))))
    assert maximum_metric_error < 2e-10 and maximum_dense_error < 2e-10
    groups=('all','observed','unobserved','near','middle','far')
    def group_rows(rows,group):
        return rows if group=='all' else [row for row in rows if (row['group']!='observed' if group=='unobserved' else row['group']==group)]
    replacement_summary={name:{group:outcome_summary(group_rows([row for row in replacement_rows if row['method']==name],group))
                               for group in groups} for name in CONTROLS}
    margins={name:_margin_summary(rows) for name,rows in dict(
        all_valid_alternatives=feedback_rows,
        selected_initial_student=[row for row in feedback_rows if row['selected_by_initial_student']],
        unobserved_valid_alternatives=[row for row in feedback_rows if row['group']!='observed']).items()}
    observation_summary={name:_observation_summary(rows) for name,rows in dict(
        all=observation_rows, retained=[r for r in observation_rows if r['native_bank_status']=='retained'],
        removed_by_E_NMS=[r for r in observation_rows if r['native_bank_status']=='removed_by_E_NMS'],
        inactive=[r for r in observation_rows if r['native_bank_status']=='inactive']).items()}
    actual_terms={name:_margin_summary([dict(row,teacher_prefers_over_E=row['feedback']['margin']>EPS)
        for row in replacement_rows if row['method']==name and 'feedback' in row]) for name in ('MemoryGreedy','NoMemoryGreedy')}
    validate_phase(trial,'stage1',lock)
    assert sha(labels_path)==labels_sha and sha(phase/'sealed.json')==seal_sha and sha(trial/'gate.json')==gate_sha
    assert sha(ROOT/'outputs/formal/state.json')==formal_sha and sha(ROOT/'configs/default.json')==config_sha
    assert sha(Path(__file__))==source_sha and not torch.cuda.is_initialized()
    proof=dict(fingerprint=lock['fingerprint'],stage1_seal_sha256=seal_sha,lock_sha256=sha(trial/'lock.json'),
        labels_sha256=labels_sha,fixed_audit_source_sha256=source_sha,old_gate_unchanged_sha256=gate_sha,
        all64_validated_before_labels=True,all_source_input_output_hashes_rechecked=True,
        saved_feedback_delta_max_absolute_error=maximum_r_error,single_replacement_score_max_absolute_error=maximum_S_error,
        saved_metric_regression_max_absolute_error=maximum_metric_error,official_dense_max_absolute_error=maximum_dense_error,
        model_calls=0,decoder_calls=0,expert_calls=0,RGB_reads=0,adaptation_updates=0,new_policy_rollouts=0,CUDA_initialized=False)
    harms=sorted(replacement_rows,key=lambda row:row['delta_IoU_pp'])[:20]
    safe_harms=[{k:v for k,v in row.items() if k!='query_index'} for row in harms]
    public=dict(status='complete_fixed_HC64_read_only_audit',queries=64,parents=64,
        accepted_replacement_definition='Saved action slot!=0; harm/severity evaluated only where that fixed node has GT; no per-query/action selection',
        accepted_replacements=replacement_summary,actual_selected_feedback_components=actual_terms,
        initial_feedback_margins=margins,observation_geometry=observation_summary,
        single_replacement_contexts=['all_E','saved_path'],
        single_replacement_warning='Two local fixed S differences include both adjacent edges. They hold all other choices fixed and do not simulate changed future memory. Overlapping local deltas are not additive.',
        largest_anonymous_harms=safe_harms,queries_anonymous=queries,proof=proof,
        elapsed_CPU_seconds=time.perf_counter()-begin,
        geometry_component_definition='Recovered from saved side maps: strict >.5 own-map peak, four-connected components, identical side components deduplicated; no new method predictions',
        GT_component_best_is_oracle=True,changes_to_registered_method=False)
    output=ROOT/'results/score_audit';private=ROOT/'outputs/score_audit'
    assert not (output/'fixed.json').exists(),'Do not overwrite an older fixed audit'
    write_json(private/'fixed_nodes.json',dict(proof=proof,replacements=replacement_rows,
        initial_feedback=feedback_rows,observations=observation_rows))
    write_json(output/'fixed.json',public);(output/'fixed.md').write_text(_render(public))
    write_json(private/'fixed_receipt.json',dict(status='complete',proof=proof,public_json_sha256=sha(output/'fixed.json'),
        public_md_sha256=sha(output/'fixed.md'),private_nodes_sha256=sha(private/'fixed_nodes.json')))
    return public


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.error('Read-only real audit requires explicit --run')
    result=run();print(json.dumps(dict(status=result['status'],accepted_replacements=result['accepted_replacements'],
        observation_geometry=result['observation_geometry'],proof=result['proof']),indent=2))
