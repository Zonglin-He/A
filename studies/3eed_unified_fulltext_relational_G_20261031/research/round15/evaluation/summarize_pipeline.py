"""No-GT, post-complete-seal stage counts. Not an identity-quality audit."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(folder, expected):
    if sha(folder / 'SEALED.json') != expected:
        raise ValueError('Sealed stage-count input changed')
    seal = read(folder / 'SEALED.json')
    for rel, digest in seal['files'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or sha(folder / rel) != digest:
            raise ValueError('Unsafe or changed stage-count member')
    return seal


def main(args):
    semantic = check(args.semantic, args.semantic_sha256)
    direct = check(args.direct, args.direct_sha256)
    if (semantic['n_cases'] != 204 or semantic['n_model_calls'] != 612
            or direct['n_samples'] != 204 or direct['n_arms'] != 8
            or direct['semantic_seal_sha256'] != args.semantic_sha256):
        raise ValueError('Actual formal204 producer chain required')
    if args.output.exists():
        raise FileExistsError('New stage-count output required')
    cases = read(ROOT / 'research/round15/input/FORMAL_CASE204_PRIVATE.json')
    if sha(ROOT / 'research/round15/input/FORMAL_CASE204_PRIVATE.json') != direct['case_index_sha256']:
        raise ValueError('Actual case registry required')
    counters = collections.defaultdict(collections.Counter)
    statuses = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for case in cases:
        role = 'fresh' if case['role'] == 'fresh64_recent_content_relative_only' else 'reg'
        key = role + '_' + case['platform']; counts = counters[key]; counts['n'] += 1
        alias = case['case_alias']
        value = read(args.semantic / 'cases_PRIVATE' / (alias + '.json'))['semantics']
        text_ok = value['text_graph']['status'] == 'PARSED_text_graph'
        visual_ok = [v['status'] == 'PARSED_visual_graph' for v in value['visual_parsed']]
        counts['text_parsed_n'] += int(text_ok)
        counts['visual_calls_parsed_n'] += sum(visual_ok)
        counts['complete_text_two_visual_chain_n'] += int(text_ok and all(visual_ok))
        counts['graph_unique_supported_hypothesis_n'] += int(value['graph_match']['status'] == 'unique_supported')
        counts['proposal_unique_consensus_hypothesis_n'] += int(value['proposal_consensus']['status'] == 'proposal_control_unique')
        counts['disabled_unique_hypothesis_n'] += int(value['graph_relation_disabled']['status'] == 'unique_supported')
        for name in ('graph_match', 'proposal_consensus', 'graph_relation_disabled'):
            statuses[key][name][value[name]['status']] += 1
        for kind in ('text', 'visual_order0', 'visual_order1'):
            result = read(args.semantic / 'calls_PRIVATE' / (alias + '__' + kind + '.json'))['result']
            counts['complete_EOS_calls_n'] += int(result['ended_eos'])
            counts['cap_nonEOS_calls_n'] += int(not result['ended_eos'])
        with np.load(args.direct / 'cases_PRIVATE' / (alias + '.npz'), allow_pickle=False) as z:
            for index, branch in ((1, 'proposal'), (3, 'graph'), (6, 'relation_disabled')):
                counts[branch + '_mapping_accepted_n'] += int(z['arm_association_accepted'][index])
            counts['graph_query_changed_n'] += int(z['arm_query_index'][3] != z['q_source'])
            counts['graph_geometry_increment_changed_n'] += int(z['geometry_increment_axes'][4].any())
            counts['graph_output_changed_vs_source_n'] += int(z['actual_changed_axes'][4].any())
            counts['proposal_output_changed_vs_source_n'] += int(z['actual_changed_axes'][2].any())
            for index, name in enumerate(z['arm_names'].tolist()):
                statuses[key]['direct_' + name][str(z['arm_status'][index])] += 1
    expected = {'fresh_quad': 32, 'fresh_drone': 32, 'reg_quad': 76, 'reg_drone': 64}
    if {k: v['n'] for k, v in counters.items()} != expected:
        raise ValueError('Full four-cohort denominators required')
    output = {'state': 'R15_POSTSEAL_PIPELINE_COUNTS_NO_GT_NOT_IDENTITY_CERTIFICATION',
              'counts': {k: dict(v) for k, v in counters.items()},
              'status_histograms': {k: {a: dict(v) for a, v in arms.items()} for k, arms in statuses.items()},
              'source_semantic_seal_sha256': args.semantic_sha256,
              'source_direct_seal_sha256': args.direct_sha256, 'program_sha256': sha(__file__),
              'complete_counted_hypotheses_not_groundtruth': True, 'GT_or_task_IoU_read': False}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(output['counts']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--semantic', type=Path, required=True)
    parser.add_argument('--semantic-sha256', required=True); parser.add_argument('--direct', type=Path, required=True)
    parser.add_argument('--direct-sha256', required=True); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
