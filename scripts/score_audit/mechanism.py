"""Small offline mechanism stratification of the unchanged fixed-v0 audit.

Uses saved replacements, native boxes and fixed nearest-anchor indices. No new
candidate, score-based readout, model, policy history or adaptation is run.
Geometric IoU/area evidence does not determine semantic identity correctness.
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.state_opd.evaluate import read,sha,write_json
from scripts.propagation_v0.run import validate_phase

METHODS=('MemoryGreedy','NoMemoryGreedy','EvidenceDP')
EPS_PP=1e-8


def direction(delta):
    if delta is None:return 'unknown'
    return 'better' if delta>EPS_PP else 'worse' if delta < -EPS_PP else 'tie'


def area_relation(ratio):
    return 'larger' if ratio>1+1e-12 else 'smaller' if ratio<1-1e-12 else 'equal'


def _summary(rows):
    delta=np.asarray([row['delta_IoU_pp']for row in rows],float)
    hurt=delta < -EPS_PP
    return dict(replacements=len(rows),harmed=int(hurt.sum()),
        harm_fraction=float(hurt.mean())if len(rows)else None,
        severe_over5pp=int((delta < -5).sum()),severe_over20pp=int((delta < -20).sum()),
        mean_delta_IoU_pp=float(delta.mean())if len(rows)else None,
        mean_loss_among_harmed_pp=float(delta[hurt].mean())if hurt.any()else None)


def _area_summary(rows):
    result=dict(selected_component_replacements=len(rows))
    for reference in ('native','GT'):
        values=np.asarray([row['area_ratio_to_'+reference]for row in rows],float)
        result[reference]=dict(counts=dict(Counter(area_relation(v)for v in values)),
            median_ratio=float(np.median(values))if len(values)else None,
            mean_ratio=float(values.mean())if len(values)else None,
            p25_ratio=float(np.quantile(values,.25))if len(values)else None,
            p75_ratio=float(np.quantile(values,.75))if len(values)else None)
    result['native_component_IoU_delta_directions']=dict(Counter(
        row['native_direction']+'__component_'+row['chosen_direction']for row in rows))
    return result


def _render(j):
    lines=['# Fixed v0: geometric mechanism stratification','',
        'This is an offline stratification of the unchanged saved replacements, not a new method or score experiment. '
        'Native quality means box IoU relative to E at the same labeled observation. These measurements cannot establish '
        'whether the teacher mask, propagated region or selected object has the correct semantic identity.','',
        '| Control | Harmed observation replacements | Native better than E | Native worse than E | Native tied with E |',
        '|---|---:|---:|---:|---:|']
    for method,row in j['observed_harm_strata'].items():
        lines.append(f'| {method} | {row["harmed_observation_replacements"]} | {row["native_better_than_E"]} | '
            f'{row["native_worse_than_E"]} | {row["native_tied_with_E"]} |')
    lines += ['', 'A chosen box that is worse than E while the native box is better demonstrates a lost available '
        'geometric correction; it does not isolate whether candidate suppression, mask-to-box conversion, scoring or '
        'history caused that loss. A native box already worse than E is compatible with teacher error, without proving '
        'that every related component is also wrong. Native E-NMS removal is retained as a separate saved status.','',
        '| Control | Harmed unobserved replacements | Nearest anchor better | Worse | Tied | No anchor GT |',
        '|---|---:|---:|---:|---:|---:|']
    for method,row in j['unobserved_harm_anchor_strata'].items():
        counts=row['nearest_native_quality_counts']
        lines.append(f'| {method} | {row["harmed_unobserved_replacements"]} | {counts.get("better",0)} | '
            f'{counts.get("worse",0)} | {counts.get("tie",0)} | {counts.get("unknown",0)} |')
    lines += ['', 'The nearest anchor is the saved label-free physical nearest-anchor index. If that exact anchor '
        'has no GT, quality is unknown; another anchor is never substituted using GT. Near/middle/far cells and every '
        'quality stratum, including zero-count groups, are retained in JSON.','',
        '## Component size and native agreement','',
        'Selected component areas are compared separately with the native box and with the saved GT box. Larger/smaller '
        'means an area ratio strictly above/below one, not a semantic or causal classification. Aspect ratio, translation, '
        'occlusion and annotation convention can each reduce IoU even at a similar area.','',
        '| Control | Harmed component replacements at observations | Smaller than GT | Larger than GT | Smaller than native | Larger than native |',
        '|---|---:|---:|---:|---:|---:|']
    for method,row in j['observed_selected_components_harmed'].items():
        gt,native=row['GT']['counts'],row['native']['counts']
        lines.append(f'| {method} | {row["selected_component_replacements"]} | {gt.get("smaller",0)} | '
            f'{gt.get("larger",0)} | {native.get("smaller",0)} | {native.get("larger",0)} |')
    cross=j['existing_cross_anchor']
    if cross['available']:
        lines += ['', '## Existing cross-anchor result','',
            f'CrossAnchor − Projection full vIoU: {cross["CrossAnchor_minus_Projection_vIoU_pp"]:+.4f} pp; '
            f'unobserved sIoU: {cross["CrossAnchor_minus_Projection_unobserved_sIoU_pp"]:+.4f} pp.',
            cross['scope_boundary'],cross['interpretation_boundary']]
    lines += ['', 'The fixed audit JSON, its private node table, the stage-one seal, the failed gate, original '
        'sources, formal pause and registered configuration are unchanged. No new predictions or adaptations were run.']
    return '\n'.join(lines)+'\n'


def run():
    torch.set_num_threads(1);begin=time.perf_counter()
    trial=ROOT/'outputs/propagation_v0_hc64';phase=trial/'stage1';lock=read(trial/'lock.json')
    receipts=validate_phase(trial,'stage1',lock)
    fixed_path=ROOT/'results/score_audit/fixed.json';nodes_path=ROOT/'outputs/score_audit/fixed_nodes.json'
    fixed_receipt=read(ROOT/'outputs/score_audit/fixed_receipt.json')
    fixed=read(fixed_path);nodes=read(nodes_path)
    assert sha(fixed_path)==fixed_receipt['public_json_sha256'] and sha(nodes_path)==fixed_receipt['private_nodes_sha256']
    assert fixed['proof']==nodes['proof']==fixed_receipt['proof']
    assert sha(ROOT/'scripts/score_audit/fixed.py')==fixed['proof']['fixed_audit_source_sha256']
    assert fixed['proof']['fingerprint']==lock['fingerprint']
    immutable={str(path.relative_to(ROOT)):sha(path)for path in (fixed_path,nodes_path,
        ROOT/'results/score_audit/fixed.md',ROOT/'scripts/score_audit/fixed.py',
        phase/'sealed.json',trial/'gate.json',ROOT/'outputs/formal/state.json',ROOT/'configs/default.json')}
    labels_path=ROOT/'data/evaluation/development_labels.jsonl'
    assert sha(labels_path)==fixed['proof']['labels_sha256']
    labels={row['id']:row for row in [json.loads(line)for line in labels_path.read_text().splitlines()if line.strip()]if row['dataset']=='hc2'}
    observation={(row['query_index'],row['node_index']):row for row in nodes['observations']}
    banks={};teachers={};selected_components=[];unobserved=[];observed=[]
    old=ROOT/'outputs/layout_reward_dev'
    records=[json.loads(line)for line in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if line.strip()]
    records=[row for row in records if row['dataset']=='hc2'][:64]
    assert [row['id']for row in records]==lock['ids']
    for index,(record,item)in enumerate(zip(records,lock['inputs'])):
        key=record['id'];rr=receipts[key]
        banks[index]=torch.load(phase/rr['bank_path'],map_location='cpu',weights_only=False)
        cache_path=old/'student_cache'/f'{key}.pt';assert sha(cache_path)==item['cache_sha256']
        cache=torch.load(cache_path,map_location='cpu',weights_only=False)
        teachers[index]=np.asarray(cache['teacher']['boxes'],float)
    for saved in nodes['replacements']:
        index,node=saved['query_index'],saved['node_index'];bank=banks[index]
        row=dict(saved);harmed=row['delta_IoU_pp'] < -EPS_PP
        if saved['group']=='observed':
            ob=observation[(index,node)]
            assert ob['has_GT']
            delta=100*(ob['native_IoU']-ob['E_IoU'])
            row.update(native_delta_IoU_pp=delta,native_direction=direction(delta),
                       native_bank_status=ob['native_bank_status'],
                       chosen_origin=int(bank['candidate_origin'][node,saved['slot']]))
            observed.append(row)
            if row['chosen_origin'] in (0,1):
                j=list(bank['positions8']).index(node);native=teachers[index][j]
                chosen=np.asarray(bank['boxes'][node,saved['slot']],float)
                gt=labels[records[index]['id']]['boxes'][str(int(bank['frame_ids'][node]))]
                gt_area=max(0.,gt[2]-gt[0])*max(0.,gt[3]-gt[1])
                meta=records[index]['metadata'];assert gt_area>0
                chosen_area=chosen[2]*chosen[3]*meta['width']*meta['height']
                row.update(chosen_direction=direction(row['delta_IoU_pp']),
                    area_ratio_to_native=float(chosen[2]*chosen[3]/(native[2]*native[3])),
                    area_ratio_to_GT=float(chosen_area/gt_area),harmed=harmed)
                selected_components.append(row)
        else:
            anchor_index=int(bank['nearest_anchor'][node]);anchor_node=int(bank['positions8'][anchor_index])
            anchor=observation[(index,anchor_node)]
            quality=100*(anchor['native_IoU']-anchor['E_IoU'])if anchor['has_GT']else None
            row.update(nearest_anchor_quality=direction(quality),nearest_anchor_delta_IoU_pp=quality,
                nearest_anchor_has_GT=anchor['has_GT'],harmed=harmed)
            unobserved.append(row)
    obs_strata={};unobs_strata={};area_all={};area_harmed={};distance_quality={}
    for method in METHODS:
        obs=[row for row in observed if row['method']==method and row['delta_IoU_pp'] < -EPS_PP]
        count=Counter(row['native_direction']for row in obs)
        obs_strata[method]=dict(harmed_observation_replacements=len(obs),native_better_than_E=count['better'],
            native_worse_than_E=count['worse'],native_tied_with_E=count['tie'],
            native_status_by_direction={name:dict(Counter(row['native_bank_status']for row in obs if row['native_direction']==name))for name in ('better','worse','tie')},
            chosen_origin_by_direction={name:dict(Counter(str(row['chosen_origin'])for row in obs if row['native_direction']==name))for name in ('better','worse','tie')},
            mean_chosen_loss_when_native_better_pp=(float(np.mean([row['delta_IoU_pp']for row in obs if row['native_direction']=='better']))if count['better']else None))
        unobs=[row for row in unobserved if row['method']==method and row['harmed']]
        unobs_strata[method]=dict(harmed_unobserved_replacements=len(unobs),
            nearest_native_quality_counts=dict(Counter(row['nearest_anchor_quality']for row in unobs)))
        area_all[method]=_area_summary([row for row in selected_components if row['method']==method])
        area_harmed[method]=_area_summary([row for row in selected_components if row['method']==method and row['harmed']])
        distance_quality[method]={group:{quality:_summary([row for row in unobserved if row['method']==method and
            row['group']==group and row['nearest_anchor_quality']==quality])for quality in ('better','worse','tie','unknown')}
            for group in ('near','middle','far')}
    topmass=[]
    for row in nodes['observations']:
        if row['has_GT']and row['top_mass_component_IoU']is not None:
            topmass.append(direction(100*(row['native_IoU']-row['E_IoU']))+'__component_'+
                           direction(100*(row['top_mass_component_IoU']-row['E_IoU'])))
    cross_path=ROOT/'results/score_audit/cross_anchor.json'
    cross=dict(available=cross_path.exists())
    if cross_path.exists():
        data=read(cross_path);means=data['means'];crosssha=sha(cross_path)
        cross.update(result_sha256=crosssha,status=data['status'],queries=data['queries'],folds=data['folds'],
            CrossAnchor_minus_Projection_vIoU_pp=100*(means['CrossAnchor']['m_vIoU']-means['Projection']['m_vIoU']),
            CrossAnchor_minus_Projection_unobserved_sIoU_pp=100*(means['CrossAnchor']['m_sIoU_unobserved_actual']-means['Projection']['m_sIoU_unobserved_actual']),
            harm_fraction_Projection=data['replacements']['Projection']['all']['harm_fraction'],
            harm_fraction_CrossAnchor=data['replacements']['CrossAnchor']['all']['harm_fraction'],
            scope_boundary='This separate probe starts from raw Source with seven-mask proposal evidence per held-out fold, '
                'not the fixed E-initialized v0 bank. Eight folds are averaged within parent; fold-nodes are repeated observations.',
            interpretation_boundary='Held-anchor readout has a small full-vIoU difference from projection but worse unobserved '
                'sIoU and does not lower the overall accepted harm fraction. It does not establish semantic identity recovery '
                'or isolate the causal mechanism in fixed v0. Cached teacher outputs were temporally coupled across eight frames.')
    validate_phase(trial,'stage1',lock)
    for path,digest in immutable.items():assert sha(ROOT/path)==digest
    assert sha(labels_path)==fixed['proof']['labels_sha256']and not torch.cuda.is_initialized()
    if cross['available']:assert sha(cross_path)==crosssha
    result=dict(status='complete_read_only_fixed_mechanism_stratification',queries=64,
        observed_harm_strata=obs_strata,unobserved_harm_anchor_strata=unobs_strata,
        observed_selected_components_all=area_all,observed_selected_components_harmed=area_harmed,
        all_observations_topmass_native_component_direction_counts=dict(Counter(topmass)),
        unobserved_distance_by_nearest_anchor_quality=distance_quality,existing_cross_anchor=cross,
        causal_boundaries=[
            'Chosen worse than E with native better than E establishes lost available geometric improvement, not the unique cause.',
            'Native worse than E is compatible with teacher error; it does not establish wrong semantic identity.',
            'Nearest-anchor quality is evaluated only at the pre-existing physical nearest anchor; missing GT remains unknown.',
            'Component size is reported relative to native and GT separately; area alone cannot establish identity, full extent or causation.',
            'All counts condition on saved replacements and cannot be treated as independent samples across the three controls.'],
        proof=dict(fingerprint=lock['fingerprint'],fixed_json_sha256=sha(fixed_path),fixed_nodes_sha256=sha(nodes_path),
            mechanism_source_sha256=sha(Path(__file__)),labels_sha256=sha(labels_path),all64_validated_before_labels=True,
            all_fixed_sources_results_and_gate_unchanged=immutable,model_calls=0,decoder_calls=0,expert_calls=0,RGB_reads=0,
            new_candidates=0,new_policy_rollouts=0,adaptation_updates=0,CUDA_initialized=False),
        elapsed_CPU_seconds=time.perf_counter()-begin)
    output=ROOT/'results/score_audit';assert not(output/'mechanism.json').exists(),'Refuse overwrite'
    write_json(output/'mechanism.json',result);(output/'mechanism.md').write_text(_render(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.error('Read-only mechanism stratification requires explicit --run')
    result=run();print(json.dumps({key:result[key]for key in ['status','observed_harm_strata','unobserved_harm_anchor_strata',
        'observed_selected_components_harmed','existing_cross_anchor']},indent=2))
