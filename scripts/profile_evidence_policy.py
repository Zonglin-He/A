"""Exclusive CPU wall profiling of the immutable saved B-policy experiment.

Preparation only reads saved receipts. Explicit --run-replay replays precisely
the original first sixteen queries and three B arms, using sealed candidate
banks and the frozen fit callable. No source in scripts/evidence_policy changes.
There is no model, decoder, RGB, GT, score module, new reward or parameter search.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

ARMS=('OPD','FixedKD','Direct')
STAGES=('bank_preparation','history_construction','feedback_Q','feedback_targets_and_loss',
        'backward','optimizer_and_clip','greedy_readout_and_score','rollback',
        'bookkeeping_and_acceptance')
TRIAL=ROOT/'outputs/evidence_policy_hc64'
DEST=ROOT/'outputs/correction_value_hc64'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temporary,path)


class ExclusiveTimer:
    """Nested scopes charge every nanosecond once to the innermost category.

    Wrapper overhead within the root scope remains in the measured total.
    This is wall-time attribution, not a claim of uninstrumented latency.
    """
    def __init__(self,clock=time.perf_counter_ns):
        self.clock=clock;self.stack=[];self.last=None;self.start=None;self.end=None
        self.nanoseconds=defaultdict(int);self.calls=defaultdict(int)

    @contextmanager
    def scope(self,stage):
        now=self.clock()
        if self.stack:self.nanoseconds[self.stack[-1]]+=now-self.last
        else:
            if self.start is not None:raise RuntimeError('Timer root scope cannot restart')
            self.start=now
        self.stack.append(stage);self.calls[stage]+=1;self.last=now
        try:yield
        finally:
            now=self.clock()
            if not self.stack or self.stack[-1]!=stage:raise RuntimeError('Unbalanced timer scopes')
            self.nanoseconds[stage]+=now-self.last;self.stack.pop();self.last=now
            if not self.stack:self.end=now

    def report(self):
        if self.stack or self.start is None or self.end is None:raise RuntimeError('Timer is not complete')
        wall=self.end-self.start;total=sum(self.nanoseconds.values())
        if wall!=total:raise RuntimeError('Exclusive wall-time accounting did not conserve time')
        return dict(wall_nanoseconds=wall,exclusive_nanoseconds={s:self.nanoseconds[s]for s in STAGES},
            exclusive_seconds={s:self.nanoseconds[s]/1e9 for s in STAGES},
            exclusive_scope_calls={s:self.calls[s]for s in STAGES},
            exclusive_sum_nanoseconds=total,conservation_residual_nanoseconds=wall-total,
            callable_seconds=wall/1e9)


class FunctionProfiler:
    """Temporarily wrap the frozen callable; restore all objects even on error."""
    def __init__(self,policy,timer):self.policy=policy;self.timer=timer;self.restore=[]

    def _patch(self,owner,name,stage):
        original=getattr(owner,name);owned=name in vars(owner)
        @wraps(original)
        def wrapped(*args,**kwargs):
            category=stage(args,kwargs) if callable(stage) else stage
            with self.timer.scope(category):return original(*args,**kwargs)
        self.restore.append((owner,name,original,owned));setattr(owner,name,wrapped)

    def __enter__(self):
        import torch
        p=self.policy
        def rollout_stage(args,kwargs):
            mode=kwargs.get('mode',args[3] if len(args)>3 else 'student_greedy')
            return 'greedy_readout_and_score' if mode=='student_greedy' else 'history_construction'
        try:
            self._patch(p,'_prepare','bank_preparation')
            self._patch(p,'rollout',rollout_stage)
            self._patch(p,'_query','feedback_Q')
            for name in ('_finish_feedback','_attach','_fixed_loss','_batch_loss'):
                self._patch(p,name,'feedback_targets_and_loss')
            self._patch(p,'_score','greedy_readout_and_score')
            self._patch(torch.Tensor,'backward','backward')
            self._patch(torch.optim.Adam,'step','optimizer_and_clip')
            self._patch(torch.optim.Optimizer,'zero_grad','optimizer_and_clip')
            self._patch(torch.nn.utils,'clip_grad_norm_','optimizer_and_clip')
            self._patch(torch.optim.Adam,'load_state_dict','rollback')
        except BaseException:
            self.__exit__(None,None,None);raise
        return self

    def __exit__(self,*_):
        for owner,name,original,owned in reversed(self.restore):
            if owned:setattr(owner,name,original)
            else:delattr(owner,name)
        self.restore=[]


def profile_fit(bank,key,arm,cfg):
    from scripts.evidence_policy import policy
    timer=ExclusiveTimer()
    with FunctionProfiler(policy,timer):
        with timer.scope('bookkeeping_and_acceptance'):
            result=policy.fit(bank,key,arm,cfg,variant='B')
    return result,timer.report()


def assert_replay_match(actual,expected):
    """Strictly compare outputs, every real update/acceptance, losses and counts."""
    import torch
    def same(a,b,path):
        if torch.is_tensor(a) or torch.is_tensor(b):
            if not(torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and
                   a.shape==b.shape and torch.equal(a,b)):raise AssertionError(path+' tensor differs')
        elif isinstance(a,dict) or isinstance(b,dict):
            if not(isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b)):
                raise AssertionError(path+' dictionary keys differ')
            for key in a:same(a[key],b[key],path+'.'+key)
        elif isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
            if not(isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)) and len(a)==len(b)):
                raise AssertionError(path+' sequence length differs')
            for i,(x,y) in enumerate(zip(a,b)):same(x,y,f'{path}[{i}]')
        elif a!=b:raise AssertionError(path+' differs')
    for name in ('final','initialtheta','finaltheta','initial_score','final_score','rounds'):
        same(actual[name],expected[name],name)
    skip={'seconds','teacher_setup_seconds'}
    ac={k:v for k,v in actual['costs'].items()if k not in skip}
    ex={k:v for k,v in expected['costs'].items()if k not in skip}
    same(ac,ex,'actual_cost_counts')
    return dict(final_boxes_bitwise=True,initial_weights_bitwise=True,final_weights_bitwise=True,
        all_round_weights_gradients_losses_acceptance_and_Adam_hashes_identical=True,
        actual_counts_identical=True,updates=ac['updates'],accepted_updates=ac['accepted_updates'])


def audit_saved_costs(trial=TRIAL):
    lock=read(trial/'lock.json');rows=[]
    for index,item in enumerate(lock['inputs']):
        receipt=read(trial/'receipts'/(item['id']+'.json'))
        if receipt['fingerprint']!=lock['fingerprint'] or receipt['GT_read'] is not False:
            raise AssertionError('Saved receipt provenance mismatch')
        for arm in ARMS:
            c=receipt['costs']['B_'+arm]
            if any(c.get(k,0) for k in ('heavy_forwards','decoder_calls','encoder_calls','expert_calls')):
                raise AssertionError('The saved B arms unexpectedly include heavy calls')
            rows.append(dict(query_ordinal=index+1,anonymous_query=item['id'],arm=arm,
                saved_fit_seconds=c['seconds'],saved_teacher_setup_seconds=c['teacher_setup_seconds'],
                updates=c['updates'],accepted_updates=c['accepted_updates'],q_calls=c['q_calls'],
                q_input_states=c['q_input_states'],q_branch_states=c['lookahead_branch_state_evaluations']))
    summaries={}
    for n in (16,64):
        summaries[str(n)]={}
        for arm in ARMS:
            rs=[r for r in rows if r['arm']==arm and r['query_ordinal']<=n]
            summaries[str(n)][arm]={key:dict(total=sum(r[key]for r in rs),mean=statistics.mean(r[key]for r in rs))
                for key in ('saved_fit_seconds','saved_teacher_setup_seconds','updates','accepted_updates',
                            'q_calls','q_input_states','q_branch_states')}
    return dict(status='saved_receipts_only_no_replay',source_fingerprint=lock['fingerprint'],
        available_timing_fields=['seconds','teacher_setup_seconds'],
        missing_timing=['exclusive_history','Q_forward','loss','backward','optimizer','greedy_and_score','rollback'],
        timing_overlap='teacher_setup_seconds is included in seconds; do not add them',
        summaries=summaries,rows=rows,GT_read=False,new_rollouts=0)


def make_plan(trial=TRIAL,dest=DEST):
    lock=read(trial/'lock.json');items=lock['inputs'][:16]
    if len(items)!=16:raise AssertionError('Exactly the fixed first sixteen queries are required')
    files={}
    for path in ('lock.json','sealed.json'):
        files[str((trial/path).relative_to(ROOT))]=sha(trial/path)
    for item in items:
        path=trial/'receipts'/(item['id']+'.json');r=read(path)
        files[str(path.relative_to(ROOT))]=sha(path)
        files[str((trial/r['trace_path']).relative_to(ROOT))]=r['trace_sha256']
        files[str((ROOT/r['bank_source']/r['bank_path']).relative_to(ROOT))]=r['bank_sha256']
    identity=dict(schema='exclusive_B_first16_CPU_wrapper_profile_v1',queries=16,variant='B',
        arms=list(ARMS),thread_count=1,source_fingerprint=lock['fingerprint'],
        source_files=lock['sources'],saved_files=files,profiler_sha256=sha(__file__),
        selection='immutable_original_input_order_first16_no_GT_selection',
        timing_excludes=['JSON/PT reading','result comparison','writing','imports','wrapper installation'],
        timing_scope='complete original fit callable including bank preparation and trace bookkeeping',
        backward_scope='whole objective autograd; Q/history backward not separately attributable by wrappers',
        residual_scope='Python acceptance comparison, state capture, margins, trace assembly and wrapper overhead',
        timing_is_instrumented=True,cache_only=True,end_to_end=False,GT_read=False,
        heavy_forwards=0,decoder_calls=0,RGB_reads=0,expert_calls=0)
    identity['fingerprint']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/'profile_plan.json'
    if path.exists() and read(path)!=identity:raise AssertionError('Existing profile plan differs; do not overwrite')
    if not path.exists():atomic(path,identity)
    atomic(dest/'profile_saved_costs.json',audit_saved_costs(trial))
    return identity


def _validate_plan(plan):
    if sha(__file__)!=plan['profiler_sha256']:raise AssertionError('Profiler changed after lock')
    for group in ('source_files','saved_files'):
        for rel,digest in plan[group].items():
            if sha(ROOT/rel)!=digest:raise AssertionError('Locked file changed: '+rel)


def _aggregate(rows):
    summary={}
    for arm in ARMS:
        rs=[r for r in rows if r['arm']==arm]
        summary[arm]=dict(queries=len(rs),callable_seconds_mean=statistics.mean(r['timing']['callable_seconds']for r in rs),
            callable_seconds_total=sum(r['timing']['callable_seconds']for r in rs),
            exclusive_seconds_mean={s:statistics.mean(r['timing']['exclusive_seconds'][s]for r in rs)for s in STAGES},
            exclusive_seconds_total={s:sum(r['timing']['exclusive_seconds'][s]for r in rs)for s in STAGES},
            saved_fit_seconds_mean=statistics.mean(r['saved_fit_seconds']for r in rs),
            replay_fit_reported_seconds_mean=statistics.mean(r['replay_fit_reported_seconds']for r in rs),
            actual_updates_total=sum(r['regression']['updates']for r in rs),
            accepted_updates_total=sum(r['regression']['accepted_updates']for r in rs))
    return summary


def run_replay(trial=TRIAL,dest=DEST):
    import torch
    from scripts.box_propagation.bank import CandidateBank
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise AssertionError('CUDA must remain uninitialized')
    plan=read(dest/'profile_plan.json');_validate_plan(plan)
    if any((dest/name).exists()for name in ('profile_run.json','profile_report.json','profile_failure.json')):
        raise AssertionError('No overwrite or implicit restart of the fixed replay')
    lock=read(trial/'lock.json');rows=[];begin=time.perf_counter()
    atomic(dest/'profile_run.json',dict(status='running',PID=os.getpid(),profile_fingerprint=plan['fingerprint'],GT_read=False))
    try:
        for ordinal,item in enumerate(lock['inputs'][:16],1):
            receipt=read(trial/'receipts'/(item['id']+'.json'))
            bank_state=torch.load(ROOT/receipt['bank_source']/receipt['bank_path'],map_location='cpu',weights_only=False)
            if bank_state['GT_read'] is not False:raise AssertionError('Saved bank is not GT-free')
            bank=CandidateBank.from_state(bank_state)
            saved=torch.load(trial/receipt['trace_path'],map_location='cpu',weights_only=False)
            for arm in ARMS:
                actual,timing=profile_fit(bank,item['key'],arm,lock['config'])
                # Saved after the timed callable and before comparison, so any
                # numerical mismatch retains its complete private failed trace.
                trace_path=dest/'profile_traces'/(item['id']+'_B_'+arm+'.pt')
                trace_path.parent.mkdir(parents=True,exist_ok=True)
                if trace_path.exists():raise AssertionError('Do not overwrite a replay trace')
                temporary=trace_path.with_suffix('.tmp');torch.save(actual,temporary);os.replace(temporary,trace_path)
                regression=assert_replay_match(actual,saved['B_'+arm])
                row=dict(query_ordinal=ordinal,anonymous_query=item['id'],arm=arm,timing=timing,
                    regression=regression,saved_fit_seconds=saved['B_'+arm]['costs']['seconds'],
                    replay_fit_reported_seconds=actual['costs']['seconds'],
                    replay_trace_path=str(trace_path.relative_to(dest)),replay_trace_sha256=sha(trace_path),GT_read=False)
                rows.append(row)
                with (dest/'profile_rows.jsonl').open('a')as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            print(json.dumps(dict(profile_queries_complete=ordinal,total=16,seconds=time.perf_counter()-begin)),flush=True)
        _validate_plan(plan)
        if torch.cuda.is_initialized():raise AssertionError('CUDA unexpectedly initialized')
        report=dict(status='complete',profile_fingerprint=plan['fingerprint'],queries=16,arm_queries=len(rows),
            all_bitwise_regressions_passed=True,all_exclusive_timers_conserved=True,
            summaries=_aggregate(rows),rows=rows,whole_script_seconds_including_IO=time.perf_counter()-begin,
            limitations=['Fixed first16 development panel; not all64 timing or independent generalization evidence.',
                'Instrumented callable time; wrapper and cold library overhead may affect timing.',
                'Exclusive measured fit categories include bookkeeping; this is not full RGB/expert/model end-to-end latency.',
                'No inference that OPD is inherently efficient; actual update counts and different computational objectives remain explicit.',
                'Greedy readout and scoring are timed; small acceptance comparisons and trace/Adam capture belong to bookkeeping.',
                'Autograd time is whole backward, not separately Q/history backward.'],
            GT_read=False,new_scientific_results=False,heavy_forwards=0,decoder_calls=0,RGB_reads=0,expert_calls=0)
        atomic(dest/'profile_report.json',report)
        atomic(dest/'profile_run.json',dict(status='complete',profile_fingerprint=plan['fingerprint'],arm_queries=len(rows),GT_read=False))
        return report
    except BaseException as error:
        atomic(dest/'profile_failure.json',dict(status='failed_stop_no_continue',profile_fingerprint=plan['fingerprint'],
            completed_arm_queries=len(rows),error=repr(error),traceback=traceback.format_exc(),GT_read=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-replay',action='store_true',help='Explicitly execute the locked fixed16 cached CPU replay')
    args=parser.parse_args()
    if args.run_replay:
        report=run_replay();print(json.dumps(dict(status=report['status'],queries=report['queries'],arm_queries=report['arm_queries'])))
    else:
        plan=make_plan();print(json.dumps(dict(status='prepared_no_replay',profile_fingerprint=plan['fingerprint'])))
