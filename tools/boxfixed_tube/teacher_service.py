"""Content-bound native evidence cache and one process, never an extra expert."""
import hashlib,json,os,select,subprocess,sys,tempfile,time
from pathlib import Path
import torch
from stvg.runtime import ROOT,sha


class Teacher:
    def __init__(self,root,fingerprint):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.fingerprint=fingerprint;self.process=None;self.temp=tempfile.TemporaryDirectory(prefix='boxfixed-ref-');self.log=None
    def __call__(self,frames,caption):
        context=dict(RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),shape=list(frames.shape),caption=caption,teacher_fingerprint=self.fingerprint)
        key=hashlib.sha256(json.dumps(context,sort_keys=True,ensure_ascii=False).encode()).hexdigest();path=self.root/(key+'.pt');receipt=self.root/(key+'.json')
        if receipt.exists():
            p=json.loads(receipt.read_text());assert p['context']==context and sha(path)==p['sha256'];z=torch.load(path,map_location='cpu',weights_only=False);return z,dict(cache_hit=True,seconds=0.,cache_key=key)
        if self.process is None:
            self.log=open(Path(self.temp.name)/'worker.log','wb');self.process=subprocess.Popen([sys.executable,'-B','-m','tools.boxfixed_tube.teacher_worker'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)
        inp=Path(self.temp.name)/'request.pt';out=Path(self.temp.name)/'reply.pt';torch.save(dict(frames=frames,caption=caption),inp);tick=time.perf_counter();self.process.stdin.write(json.dumps(dict(input=str(inp),output=str(out)))+'\n');self.process.stdin.flush()
        if not select.select([self.process.stdout],[],[],240)[0]:raise TimeoutError('ReferDINO timed out; own worker log '+str(self.log.name))
        ack=json.loads(self.process.stdout.readline());assert ack.get('ok'),ack
        z=torch.load(out,map_location='cpu',weights_only=False);assert z['RGB_sha256']==context['RGB_sha256'];torch.save(z,path);receipt.write_text(json.dumps(dict(context=context,sha256=sha(path)),ensure_ascii=False));return z,dict(cache_hit=False,seconds=time.perf_counter()-tick,cache_key=key)
    def close(self):
        if self.process is not None and self.process.poll()is None:
            try:self.process.stdin.write('{"stop":true}\n');self.process.stdin.flush();self.process.wait(timeout=30)
            except (BrokenPipeError,subprocess.TimeoutExpired):self.process.terminate();self.process.wait(timeout=15)
        if self.log:self.log.close()
        self.temp.cleanup()
