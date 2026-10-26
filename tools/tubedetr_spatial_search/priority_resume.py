"""Explicit user-authorized TubeDETR-first execution; frozen inference is reused."""
import argparse
import ast
import copy
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import types

from . import full

AMENDMENT = full.PUBLIC / 'priority_resume_plan.json'
ORDER_ASSERT = "assert json.loads((ROOT/'outputs/simplepath_llava/status.json').read_text())['status']=='complete_both_fixed128_and_full'"


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def active_llava():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = [x.decode() for x in (path / 'cmdline').read_bytes().split(b'\0') if x]
            if '-m' in args and args[args.index('-m') + 1] == 'tools.simplepath_llava.run':
                found.append(int(path.name))
        except (OSError, UnicodeError, IndexError):
            continue
    return found


def authorized_gpu_tree(source=None):
    """Replace exactly the old queue-order assertion; retain every numerical AST node."""
    source = inspect.getsource(full.gpu) if source is None else source
    tree = ast.parse(source)
    expected = ast.dump(ast.parse(ORDER_ASSERT).body[0], include_attributes=False)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)
               and ast.dump(node, include_attributes=False) == expected]
    if len(matches) != 1:
        raise ValueError('Expected exactly one frozen LLaVA queue-order assertion')
    original = copy.deepcopy(tree)
    class ReplaceOrder(ast.NodeTransformer):
        def visit_Assert(self, node):
            if ast.dump(node, include_attributes=False) == expected:
                return ast.copy_location(ast.Expr(ast.Call(ast.Name('_authorize', ast.Load()), [], [])), node)
            return node
    transformed = ast.fix_missing_locations(ReplaceOrder().visit(tree))
    # Restore only the authorization call and prove all other syntax is unchanged.
    class RestoreOrder(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == '_authorize'):
                return ast.copy_location(ast.parse(ORDER_ASSERT).body[0], node)
            return node
    restored = RestoreOrder().visit(copy.deepcopy(transformed))
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)
    return transformed


def checked():
    p, f = full.full_checked()
    a = json.loads(AMENDMENT.read_text())
    assert a['status'] == 'frozen_before_TubeDETR_first_GPU'
    assert a['full_plan_sha256'] == full.sha(full.FULL_PLAN)
    assert a['search_plan_sha256'] == full.sha(full.PLAN)
    for path, h in a['code_sha256'].items():
        assert full.sha(full.ROOT / path) == h, path
    tree = authorized_gpu_tree()
    assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == a['authorized_gpu_ast_sha256']
    return p, f, a


def _authorize():
    checked()
    assert not active_llava(), 'LLaVA must remain paused while TubeDETR has priority'


def prepare():
    assert not AMENDMENT.exists()
    full.full_checked()
    assert not active_llava()
    for ds in ['vidstg', 'hc2']:
        target = full.OUT / ds / 'full'
        assert not target.exists() or not any(target.iterdir()), 'Existing full evidence needs explicit audit'
    a = dict(status='frozen_before_TubeDETR_first_GPU', frozen_at=now(),
             user_request='先在继续，先做tubedert 巡检也开启',
             execution_order=['TubeDETR VidSTG 10303 qualification/GPU/seal/CPU score',
                              'TubeDETR HC2 1902 qualification/GPU/seal/CPU score'],
             LLaVA_remains_paused=True, LLaVA_last_committed=3949,
             full_plan_sha256=full.sha(full.FULL_PLAN), search_plan_sha256=full.sha(full.PLAN),
             code_sha256={str(Path(__file__).relative_to(full.ROOT)): full.sha(Path(__file__))},
             authorized_gpu_ast_sha256=hashlib.sha256(ast.dump(authorized_gpu_tree(), include_attributes=False).encode()).hexdigest(),
             change_scope='Only replace LLaVA-completion scheduling assertion with explicit priority authorization and no-active-LLaVA check; preserve all numerical AST nodes and original bound source bytes',
             Source_native_MAP_unchanged=True, selected_parameters_unchanged=True,
             labels_only_after_complete_prediction_seal=True, shared_gpu_lock=True,
             cache_only_teacher=True, free_space_floor_GiB=25,
             no_automatic_LLaVA_resume=True)
    full.write(AMENDMENT, a)


def bound_stage(stage):
    checked()
    namespace = dict(vars(full))
    namespace['_authorize'] = _authorize
    def write_with_amendment(path, value):
        full.write(path, {**value, 'execution_amendment_path': str(AMENDMENT.relative_to(full.ROOT)),
                          'execution_amendment_sha256': full.sha(AMENDMENT)})
    namespace['write'] = write_with_amendment
    if stage == 'gpu':
        exec(compile(authorized_gpu_tree(), str(full.ROOT / 'tools/tubedetr_spatial_search/full.py'), 'exec'), namespace)
        return namespace['gpu']
    assert stage == 'score'
    return types.FunctionType(full.score.__code__, namespace, 'score')


def run():
    p, f, a = checked()
    _authorize()
    for ds in p['dataset_order']:
        for stage in ['gpu', 'score']:
            _authorize()
            log = full.ROOT / 'logs' / f'tubedetr_spatial_search_{ds}_full_{stage}_priority_v1.log'
            cmd = [sys.executable, '-B', '-u', '-m', 'tools.tubedetr_spatial_search.priority_resume', stage, '--dataset', ds]
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            if stage == 'score':
                env['CUDA_VISIBLE_DEVICES'] = ''
            with log.open('a') as stream:
                proc = subprocess.Popen(cmd, cwd=full.ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
                state = dict(controller_pid=os.getpid(), worker_pid=proc.pid, command=cmd,
                             log=str(log.relative_to(full.ROOT)), dataset=ds, stage=stage,
                             execution_amendment_sha256=full.sha(AMENDMENT), LLaVA_remains_paused=True)
                full.write(full.ROOT / 'runtime/tubedetr_spatial_search_full.json', state)
                full.write(full.OUT / 'full_status.json', dict(status=stage, **state))
                if proc.wait():
                    raise RuntimeError(f'{ds} priority full {stage} failed')
    files = [full.PLAN, full.FULL_PLAN, AMENDMENT, full.PUBLIC / 'selected_profiles.json',
             *[full.PUBLIC / ds / 'final_seal.json' for ds in p['dataset_order']]]
    full.write(full.PUBLIC / 'final_seal.json', dict(status='complete',
               files={str(path.relative_to(full.ROOT)): full.sha(path) for path in files},
               execution_amendment_sha256=full.sha(AMENDMENT), publication_pending=True))
    full.write(full.OUT / 'full_status.json', dict(status='both_complete_populations_scored',
               execution_amendment_sha256=full.sha(AMENDMENT), Root_verification_and_publication_pending=True,
               LLaVA_remains_paused=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'gpu', 'score'])
    parser.add_argument('--dataset', choices=['vidstg', 'hc2'])
    args = parser.parse_args()
    try:
        if args.stage in ['prepare', 'run']:
            globals()[args.stage]()
        else:
            bound_stage(args.stage)(args.dataset)
    except Exception as exc:
        full.write(full.OUT / 'priority_resume_failure.json', dict(stage=args.stage, dataset=args.dataset,
                   error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                   execution_amendment_sha256=full.sha(AMENDMENT) if AMENDMENT.exists() else None))
        raise
