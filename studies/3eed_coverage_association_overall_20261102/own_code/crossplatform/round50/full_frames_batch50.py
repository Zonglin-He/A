"""Sequential registered full-mask chunks, one externally guarded GPU owner."""
from pathlib import Path
import subprocess,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    exp=ROOT/'research/round50/experiment'
    grant=json.loads((exp/'FULL_FRAMES_BATCH_AUTHORIZATION.json').read_text())
    assert grant['state']=='ROOT_AUTHORIZED_R50_REMAINING_FRAME_CHUNKS_SEQUENTIAL'
    assert grant['program_sha256']==sha(__file__)
    assert grant['plan_sha256']==sha(exp/'FULL_FRAMES_PLAN_LOCK.json')
    driver=ROOT/'crossplatform/round50/full_frames50.py'
    assert grant['driver_sha256']==sha(driver)
    base=[sys.executable,str(driver),'--plan',str(exp/'FULL_FRAMES_PLAN_LOCK.json'),
          '--authorization',str(exp/'FULL_FRAMES_AUTHORIZATION.json')]
    for n in range(grant['first_chunk'],grant['last_chunk']+1):
        log=exp/f'FULL_FRAMES_CHUNK{n:04d}_ATTEMPT001.log'
        assert not log.exists()
        with log.open('w') as stream:
            subprocess.run(base+['--chunk',str(n)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('FRAME50_BATCH_COMPLETE',n,flush=True)
    subprocess.run(base+['--finalize'],cwd=ROOT,check=True)
if __name__=='__main__':main()
