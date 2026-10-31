"""Root-only, complete-seal CPU release composer. No target payload/model IO.

Hashing the target-index bytes is integrity verification; its JSON is not parsed.
This helper never performs task scoring and cannot resume a partial model run.
"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'research/round15/experiment'
SOURCE_SHA = '0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
INDEX_SHA = 'e8eeb12b3f68232e6729cdf33d2bf49beada7fc662c83f98115be20b7ed6d562'
SCORER_SHA = 'a5f0c0a2362427c4dab94ea3c4a833a245ef07782737c00c1a5bfef8044a57b5'
DEPS = {
    'research/round12/evaluation/metrics.py': '4f696eecb3a7b46e237dc13cb5354a512e34f10799590b41bce579f8c04ca81f',
    'research/round5/audit/independent_core_v2.py': 'b4937fdd89f719fe1d9560f770c01597ff6807c74a75013720b470c43393379a',
    'tools/obb_iou.py': '45896d29d1f95167b95a27b044fcc06b4e6fe5f9cb98749096c2f707292d4083',
}
ARMS = ['source', 'proposal_assoc', 'proposal_geo', 'graph_assoc', 'graph_geo',
        'graph_geo_noRGB', 'graph_geo_relation_disabled', 'graph_geo_relation_flipped_diagnostic']
ROLES = {'fresh64_recent_content_relative_only': 64, 'R14_regression140_exposed': 140}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check_seal(directory, digest=None):
    directory = Path(directory).resolve()
    seal_path = directory / 'SEALED.json'
    if digest is not None and sha(seal_path) != digest:
        raise ValueError('Seal hash changed')
    seal = read(seal_path)
    if not isinstance(seal.get('files'), dict) or not seal['files']:
        raise ValueError('Complete nonempty sealed file inventory required')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and p.name != 'SEALED.json'}
    if actual != set(seal['files']):
        raise ValueError('Unlisted or missing artifact members')
    for relative, expected in seal['files'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or sha(directory / path) != expected:
            raise ValueError('Unsafe or changed sealed member')
    return seal


def merge_bindings(result, incoming):
    for path, digest in incoming.items():
        full = Path(path)
        full = full if full.is_absolute() else ROOT / full
        if sha(full) != digest or path in result and result[path] != digest:
            raise ValueError('Program/input byte binding mismatch')
        result[path] = digest


def main(args):
    # These actual producer seals, rather than a successful engineering probe,
    # must precede any release. Missing formal files fail before index access.
    semantic = args.semantic.resolve()
    direct = args.predictions.resolve()
    semseal = check_seal(semantic, args.semantic_seal_sha256)
    directseal = check_seal(direct, args.prediction_seal_sha256)
    if (semseal['state'] != 'sealed_R15_semantic204_three_calls_before_GT'
            or semseal['n_cases'] != 204 or semseal['n_model_calls'] != 612
            or directseal['state'] != 'sealed_R15_eight_direct_arms_before_GT'
            or directseal['n_samples'] != 204 or directseal['n_arms'] != 8):
        raise ValueError('Full formal204/612-call/eight-arm producers required')
    manifest = ROOT / 'research/round15/input/FORMAL204_GPU_INPUTS.json'
    cases = ROOT / 'research/round15/input/FORMAL_CASE204_PRIVATE.json'
    frames = ROOT / 'research/round15/input/G_FRAME_INDEX_PRIVATE.json'
    input_plan = ROOT / 'research/round15/input/INPUT_PLAN_LOCK.json'
    semantic_plan = EXP / 'SEMANTIC_PLAN_V4_LOCK.json'
    direct_plan = EXP / 'ASSEMBLY_FORMAL204_PLAN_LOCK.json'
    if (semseal['plan_sha256'] != sha(semantic_plan)
            or directseal['plan_sha256'] != sha(direct_plan)
            or read(semantic / 'registered_plan.json') != read(semantic_plan)
            or read(direct / 'registered_plan.json') != read(direct_plan)
            or directseal['semantic_seal_sha256'] != sha(semantic / 'SEALED.json')
            or directseal['arm_names'] != ARMS
            or semseal['role_counts'] != ROLES or directseal['role_counts'] != ROLES):
        raise ValueError('Actual formal plans, arms, roles and producer chain mismatch')
    for plan in (read(semantic_plan), read(direct_plan)):
        for key, path in [('manifest', manifest), ('case_index', cases), ('frame_index', frames)]:
            if plan[key + '_sha256'] != sha(path):
                raise ValueError('Current formal registry mismatch')
    if (semseal['manifest_sha256'] != sha(manifest)
            or directseal['manifest_sha256'] != sha(manifest)
            or directseal['case_index_sha256'] != sha(cases)
            or directseal['frame_index_sha256'] != sha(frames)):
        raise ValueError('Producer registry mismatch')
    audits = [
        ROOT / 'research/round15/evaluation/CAPTURE96_64_INDEPENDENT_AUDIT.json',
        args.semantic_audit.resolve(), args.direct_audit.resolve(),
    ]
    for path in audits:
        if not read(path)['state'].startswith('PASS'):
            raise ValueError('Actual prelabel audits must pass')
    if sha(audits[0]) != 'c4af65ec18348c1df685e62e8be50ebda42e9c8afce2de7b0a8c4716f5db2bf0':
        raise ValueError('Original completed source/capture audit changed')
    if read(audits[1])['prediction_seal_sha256'] != sha(semantic / 'SEALED.json'):
        raise ValueError('Semantic audit is not the current formal producer')
    direct_audit = read(audits[2])
    if (direct_audit['prediction_seal_sha256'] != sha(direct / 'SEALED.json')
            or direct_audit['semantic_seal_sha256'] != sha(semantic / 'SEALED.json')):
        raise ValueError('Direct audit is not the current formal composition')
    if (read(audits[1])['plan_sha256'] != semseal['plan_sha256']
            or direct_audit['plan_sha256'] != directseal['plan_sha256']
            or direct_audit['semantic_plan_sha256'] != semseal['plan_sha256']
            or direct_audit['manifest_sha256'] != sha(manifest)
            or direct_audit['case_index_sha256'] != sha(cases)
            or direct_audit['input_plan_sha256'] != sha(input_plan)):
        raise ValueError('Current audit plan/input chain mismatch')
    capture_audit = read(audits[0])
    for key, folder in [('source', ROOT / 'crossplatform/round15/source96_attempt001'),
                        ('mask', ROOT / 'crossplatform/round15/masks64_attempt001')]:
        if (capture_audit[key + '_seal_sha256'] != sha(folder / 'SEALED.json')
                or capture_audit[key + '_plan_sha256'] != sha(folder / 'registered_plan.json')):
            raise ValueError('Previously audited new capture changed')
    if capture_audit['input_plan_sha256'] != sha(input_plan):
        raise ValueError('Capture input plan changed')
    case_rows = read(cases)
    captures = sorted({(ROOT / c[key]).resolve() for c in case_rows
                       for key in ('source_dir', 'mask_dir')})
    if len(captures) != 8:
        raise ValueError('Exactly eight capture producers')
    normalized_maps = []
    for seal in (semseal, directseal):
        normalized_maps.append({str((ROOT / Path(p)).resolve()): h
                                for p, h in seal['cache_seals_sha256'].items()})
    expected_capture_seals = {str(p / 'SEALED.json') for p in captures}
    if normalized_maps[0] != normalized_maps[1] or set(normalized_maps[0]) != expected_capture_seals:
        raise ValueError('Captures do not match both actual producer closures')
    programs = dict(DEPS)
    programs['research/round15/evaluation/score_direct.py'] = SCORER_SHA
    programs['crossplatform/round12/capture_masks.py'] = '27a978ff72cc2fc162dcf61aedcea3cbaff03c4307b1c9db9c52ae32cf3e788a'
    artifacts = []
    for i, folder in enumerate(captures):
        seal = check_seal(folder, normalized_maps[0][str(folder / 'SEALED.json')])
        producer = seal.get('program_sha256')
        if isinstance(producer, dict):
            merge_bindings(programs, producer)
        plan = read(folder / 'registered_plan.json')
        merge_bindings(programs, plan.get('capture_program_sha256', {}))
        artifacts.append({'role': 'capture_' + str(i), 'directory': str(folder),
                          'seal_sha256': sha(folder / 'SEALED.json')})
    for role, folder, seal in [('semantic', semantic, semseal), ('direct', direct, directseal)]:
        merge_bindings(programs, seal['program_sha256'])
        artifacts.append({'role': role, 'directory': str(folder),
                          'seal_sha256': sha(folder / 'SEALED.json')})
    merge_bindings(programs, {str(Path(__file__).relative_to(ROOT)): sha(__file__)})
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [manifest, cases, *audits,
              ROOT / 'research/round15/input/G_FRAME_INDEX_PRIVATE.json',
              EXP / 'AUTHORIZATION.json', EXP / 'EXPERIMENT_CARD_LOCK.json',
              EXP / 'SEMANTIC_PLAN_V4_LOCK.json', EXP / 'FORMAL204_ENTRY_RELEASE.json',
              EXP / 'RUNTIME_ADAPTER_LOCK.json', EXP / 'QUOTE_BINDING_BUGFIX_ADDENDUM.json',
              EXP / 'DECODE_BUDGET_V4_ADDENDUM.json']}
    prompts = {p: programs[p] for p in ('crossplatform/round15/semantic_runner.py',
               'crossplatform/round15/semantic_runner_v2.py', 'crossplatform/round15/semantic_runner_v3.py')}
    merge_bindings({}, programs)
    index = ROOT / 'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json'
    checkpoint = ROOT / 'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth'
    if sha(index) != INDEX_SHA or sha(checkpoint) != SOURCE_SHA:
        raise ValueError('Original target index/source checkpoint byte mismatch')
    result = {
        'stage': 'R15_unified_full_description_direct8', 'CPU_SCORE_RELEASE': True,
        'all_direct_predictions_and_helpers_sealed_before_GT': True,
        'prediction_seal_sha256': sha(direct / 'SEALED.json'),
        'manifest_sha256': sha(manifest), 'cases_sha256': sha(cases),
        'scoring_index_sha256': INDEX_SHA, 'scorer_sha256': SCORER_SHA,
        'source_checkpoint_path': str(checkpoint), 'source_checkpoint_sha256': SOURCE_SHA,
        'geometry_dependency_sha256': DEPS, 'program_bindings': programs,
        'input_bindings': inputs, 'prompt_bindings': prompts,
        'sealed_artifacts': artifacts, 'required_seal_roles': [a['role'] for a in artifacts],
        'expected_case_role_counts': {'fresh64_recent_content_relative_only': 64,
                                      'R14_regression140_exposed': 140},
        'scope': 'Current-round CPU scoring only; historical target-label exposure remains. No TTA release.',
    }
    if args.output.exists():
        raise FileExistsError('New release path required')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'state': 'ROOT_COMPLETE_SEAL_CPU_RELEASE_WRITTEN_NO_SCORE_EXECUTED',
                      'sha256': sha(args.output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--semantic', type=Path, required=True)
    parser.add_argument('--semantic-seal-sha256', required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--prediction-seal-sha256', required=True)
    parser.add_argument('--semantic-audit', type=Path, required=True)
    parser.add_argument('--direct-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
