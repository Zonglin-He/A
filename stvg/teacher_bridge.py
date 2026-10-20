"""One native ReferDINO process; its returned evidence is used twice."""
import os,subprocess,tempfile,time
from pathlib import Path
import torch
from stvg.runtime import ROOT
def refer_native(frames,caption):
    start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='single-ref-')as d:
        inp,out=Path(d)/'request.pt',Path(d)/'reply.pt';torch.save(dict(frames=frames,caption=caption),inp)
        p=subprocess.run([str(Path(__import__('sys').executable)),'-B','-m','stvg.teacher_worker',str(inp),str(out)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        if p.returncode:raise RuntimeError(p.stdout.decode(errors='replace')[-5000:])
        z=torch.load(out,map_location='cpu',weights_only=False)
    z['process_wall_seconds']=time.perf_counter()-start;return z


def evidence_key(frames, caption, *, positions, frame_ids, fingerprint, dataset):
    """Key actual RGB and complete query context; never key evidence by an alias."""
    import hashlib,json,numpy as np
    a=np.ascontiguousarray(frames)
    context=dict(schema=1,RGB_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                 shape=list(a.shape),dtype=str(a.dtype),caption=caption,
                 positions=list(positions),frame_ids=list(frame_ids),
                 fingerprint=fingerprint,dataset=dataset)
    payload=json.dumps(context,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    return hashlib.sha256(payload).hexdigest(),context


class ResidentTeacher:
    """Private long-lived process; only its own process is ever terminated."""
    def __init__(self, cache_dir=None, fingerprint=None):
        self.cache_dir=Path(cache_dir)if cache_dir else None
        self.fingerprint=fingerprint;self.process=None
        self.directory=tempfile.TemporaryDirectory(prefix='stvg-resident-ref-')
        self.log=None
        if self.cache_dir:
            if not fingerprint:raise ValueError('Evidence cache requires a complete runtime fingerprint')
            self.cache_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
    def _start(self):
        if self.process is None:
            self.log=(Path(self.directory.name)/'teacher.log').open('wb')
            self.process=subprocess.Popen([__import__('sys').executable,'-B','-m','stvg.teacher_worker','--serve'],
                cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                text=True,bufsize=1,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    def _call(self,frames,caption):
        import json,select
        start=time.perf_counter();self._start();directory=Path(self.directory.name)
        inp,out=directory/'request.pt',directory/'reply.pt'
        torch.save(dict(frames=frames,caption=caption),inp)
        if out.exists():out.unlink()
        self.process.stdin.write(json.dumps(dict(input=str(inp),output=str(out)))+'\n');self.process.stdin.flush()
        ready,_,_=select.select([self.process.stdout],[],[],180)
        if not ready:raise TimeoutError('Resident ReferDINO request exceeded 180 seconds')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Resident ReferDINO exited; see '+str(directory/'teacher.log'))
        ack=json.loads(line)
        if not ack.get('ok'):raise RuntimeError('Resident ReferDINO failed: '+ack.get('error','unknown'))
        z=torch.load(out,map_location='cpu',weights_only=False)
        z['process_wall_seconds']=time.perf_counter()-start;return z
    def __call__(self,frames,caption,*,positions,frame_ids,dataset):
        import hashlib,json,copy
        key,context=evidence_key(frames,caption,positions=positions,frame_ids=frame_ids,
                                fingerprint=self.fingerprint,dataset=dataset)
        if self.cache_dir:
            payload=self.cache_dir/(key+'.pt');receipt=self.cache_dir/(key+'.json')
            if receipt.exists():
                meta=json.loads(receipt.read_text())
                if meta.get('key')!=key or meta.get('context')!=context or not payload.exists() or hashlib.sha256(payload.read_bytes()).hexdigest()!=meta.get('sha256'):
                    raise ValueError('Teacher cache binding/hash mismatch: '+key)
                stored=torch.load(payload,map_location='cpu',weights_only=False)
                z=copy.deepcopy(stored);z['evidence_original_cost']=stored['cost']
                z['cost']={k:0. for k in stored['cost']};z.update(cache_hit=True,cache_key=key,model_load_seconds=0.,process_wall_seconds=0.,query_forwards=0,peak_GPU_bytes=0)
                return z
        z=self._call(frames,caption);z.update(cache_hit=False,cache_key=key,query_forwards=1)
        if self.cache_dir:
            tmp=payload.with_suffix('.pt.tmp');torch.save(z,tmp);os.chmod(tmp,0o600)
            proof=dict(key=key,context=context,sha256=hashlib.sha256(tmp.read_bytes()).hexdigest())
            os.replace(tmp,payload);mtmp=receipt.with_suffix('.json.tmp');mtmp.write_text(json.dumps(proof,sort_keys=True,ensure_ascii=False));os.chmod(mtmp,0o600);os.replace(mtmp,receipt)
        return z
    def close(self):
        if self.process is not None:
            if self.process.poll()is None:
                try:self.process.stdin.write('{"stop":true}\n');self.process.stdin.flush();self.process.wait(timeout=30)
                except (BrokenPipeError,subprocess.TimeoutExpired):self.process.terminate();self.process.wait(timeout=15)
            self.process=None
        if self.log:self.log.close();self.log=None
        self.directory.cleanup()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
