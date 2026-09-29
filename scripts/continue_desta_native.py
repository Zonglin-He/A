"""Wait for one measured worker wrapper, then continue the locked serial run.

This process allocates no GPU while waiting. A failed worker is never retried.
It waits on an OS process handle, not a stale PID or STARTED file.
"""
import argparse
import os
from pathlib import Path
import select
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta_native_common import D, OUT, read, write, sha


def main(name):
    started = D / 'CONTINUATION_STARTED_V3.json'
    try:
        active = read(D / 'ACTIVE.json')
        assert active['name'] == name
        wrapper_pid = active.get('wrapper_pid')
        if active['status'] == 'running':
            fd = os.pidfd_open(wrapper_pid)
            try:
                cmd = Path(f'/proc/{wrapper_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
                assert 'launch_desta_native_v3.py' in cmd and f'--run {name}' in cmd
                write(started, dict(pid=os.getpid(), time=time.time(), waiting_for_run=name,
                    wrapper_pid=wrapper_pid, cmdline=cmd, script_sha=sha(Path(__file__))))
                select.select([fd], [], [])
            finally:
                os.close(fd)
        else:
            write(started, dict(pid=os.getpid(), time=time.time(), waiting_for_run=name,
                wrapper_already_exited=True, script_sha=sha(Path(__file__))))
        run = OUT / ('desta_native_' + name)
        assert read(run / 'EXIT.json')['code'] == 0, 'Failed worker requires root repair'
        assert read(run / 'RECEIPT.json')['status'] == 'completed'
        assert read(run / 'COMPLETE.json')['status'] in ('safe_pause', 'complete')
        write(D / 'CONTINUATION_READY_V3.json', dict(time=time.time(), previous_run=name))
        from scripts.supervise_desta_native_v3 import main as supervise
        supervise()
    except BaseException as exc:
        write(D / 'CONTINUATION_FAILURE_V3.json', dict(time=time.time(), error=repr(exc),
            traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--after-run', required=True)
    main(p.parse_args().after_run)
