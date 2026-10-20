"""Finish the four authorized downloads, with bounded resume of dead workers."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write


JOBS = {
    'hc': ('--drive-id', '1mkY1vv1U_pSVRj4JRY6uYTABKONQEh_w', 'checkpoints/tubedetr/hc2_k2_res224.pth',
           None, 'c646a8b7131bb28806c350e16479cd8bb31d4425ccb15129b26a6e4d38dd4222', 24),
    'vid': ('--drive-id', '1FaKT54s7LWcTgW562OXpVWHPXxNG7INu', 'checkpoints/tubedetr/vidstg_k2_res224.pth',
            None, '2802c66049b2e7986a826b4bbd291eb8521446cfba6fee03b56729438bf44213', 24),
    'refer': ('--url', 'https://huggingface.co/liangtm/referdino/resolve/main/ryt_mevis_swinb.pth',
              'checkpoints/referdino/ryt_mevis_swinb.pth', '87e5356c6c463efec4f736c3348d1fbfba9bfaaa8181d51d15a8404295a5a9e3',
              '87e5356c6c463efec4f736c3348d1fbfba9bfaaa8181d51d15a8404295a5a9e3', 16),
    'vidor': ('--url', 'https://huggingface.co/datasets/shangxd/vidor/resolve/main/validation-video.zip',
              'data/vidor/validation-video.zip', '40dd102b43ecd2ed70f2dc38ee08258315e25919d2f1fb8f43bb235740a83a25',
              '40dd102b43ecd2ed70f2dc38ee08258315e25919d2f1fb8f43bb235740a83a25', 16),
}


def alive(pid):
    if not pid:
        return False
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        return b'tools/fetch_assets.py' in command and b'python' in command
    except FileNotFoundError:
        return False


def main():
    registry = ROOT/'runtime/download_processes.json'
    processes = json.loads(registry.read_text()) if registry.exists() else {}
    attempts = {k: 0 for k in JOBS}
    children = {}
    verified = set()
    while len(verified) != len(JOBS):
        for name, (kind, url, rel, resume_sha, expected, workers) in JOBS.items():
            if name in verified:
                continue
            path = ROOT/rel
            if path.exists():
                if sha(path) != expected:
                    raise RuntimeError('Complete asset differs from its expected official fingerprint: '+name)
                verified.add(name)
                print('COMPLETE_AND_VERIFIED', name, flush=True)
                continue
            if alive(processes.get(name)):
                continue
            if attempts[name] >= 3:
                raise RuntimeError('Bounded resume attempts exhausted for '+name)
            attempts[name] += 1
            command = [sys.executable, '-u', 'tools/fetch_assets.py', kind, url, '--output', rel,
                       '--workers', str(workers)]
            if resume_sha:
                command += ['--sha256', resume_sha]
            with (ROOT/f'logs/download_{name}_resume{attempts[name]}.log').open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes[name] = child.pid
            children[name] = child
            write(registry, processes)
            print('RESUME', name, attempts[name], child.pid, flush=True)
        for name, child in list(children.items()):
            if child.poll() is not None:
                child.wait()
                del children[name]
        write(ROOT/'runtime/acquisition_status.json', dict(status='complete' if len(verified) == len(JOBS) else 'running',
             verified=sorted(verified), resume_attempts=attempts))
        if len(verified) != len(JOBS):
            time.sleep(10)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'runtime/acquisition_status.json', dict(status='failed', error=str(error)))
        raise
