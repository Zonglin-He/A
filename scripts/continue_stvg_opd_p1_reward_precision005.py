"""Continue original missing P1 suffix; all directions seal before CPU labels."""
import os, subprocess, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_stvg_opd_p1_reward_precision005 import REC, verify, ALLOCATOR
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, PYTHON, read, write, sha, status


def run():
    verify(); assert read(REC/'ROOT_QUALIFICATION_READBACK.json')['status'] == 'pass'
    import scripts.continue_stvg_opd_paper_hc2_revision_v2 as original
    def child(job, *args, cpu=False):
        verify()
        if job == 'GPU' and args == ('P1_hc2',):
            p = BASE/'stages/P1_hc2/PREDICTION_BARRIER.json'
            assert read(p)['status'] == 'sealed' and read(p)['adapted_arrivals'] == 10446
            return
        command = [str(PYTHON), '-B', 'scripts/run_stvg_opd_p1_reward_precision005.py', job, *args]
        log = REC/('_'.join(['continuation', job, *args])+'.log')
        env = {**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': ALLOCATOR}
        if cpu: env['CUDA_VISIBLE_DEVICES'] = ''
        with log.open('a') as output:
            worker = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, env=env)
            status(BASE/'STAGE.json', dict(status='running_reward_precision005', controller_pid=os.getpid(),
                worker_pid=worker.pid, job=job, args=list(args), command=command, allocator=ALLOCATOR,
                log=str(log.relative_to(ROOT)), CPU=cpu, engineering_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'), time=time.time()))
            assert worker.wait() == 0, 'Preserve new failure without changing science or sampling'
    original.child = child; original.run('P1')
    write(REC/'COMPLETION.json', dict(status='reward_precision005_P1_CPU_complete_pending_actual_root',
        original_global_GT_barrier_preserved=True, old_prefix_and_HC2_not_rewritten=True,
        paper_suite_complete=False, time=time.time()))


if __name__ == '__main__':
    try: run()
    except BaseException:
        folder = REC/'controller_failures'/str(time.time_ns()); folder.mkdir(parents=True, exist_ok=True)
        (folder/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json', dict(status='reward_precision005_failure_preserved', evidence=str(folder.relative_to(ROOT)),
            paper_suite_complete=False, GT_read=False, time=time.time()))
        raise
