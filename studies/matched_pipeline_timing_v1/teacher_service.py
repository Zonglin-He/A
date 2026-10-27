"""Isolated original ReferDINO numeric path over RAM socket, no experiment files."""
from common import *
import sys,gc,random,inspect,textwrap,traceback
from multiprocessing.connection import Connection
def run(fd):
 import torch,numpy as np
 from stvg.runtime import setup
 from stvg.teacher import SingleRefer
 from stvg.teacher_worker import deny_detector
 setup();sys.addaudithook(deny_detector);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 # Only diagnostic mask-summary reductions/assertion are removed. Inference,
 # query selection, native boxes/masks and transfers remain byte-for-byte code.
 code=textwrap.dedent(inspect.getsource(SingleRefer.__call__.__wrapped__))
 old=";mstats=dict(shape=list(m.shape),finite=bool(torch.isfinite(m).all()),foreground_fraction_at_native_grid=float((m>0).float().mean()))"
 assert code.count(old)==1;code=code.replace(old,'')
 old="assert torch.isfinite(b).all()and(b[:,2:]>0).all();";assert code.count(old)==1;code=code.replace(old,'').replace('mask_summary=mstats','mask_summary=None')
 # Decorator is retained and resolved without invoking any model at import.
 scope=dict(vars(sys.modules[SingleRefer.__module__]));exec(compile(code,'<diagnostic-free-native-ReferDINO>','exec'),scope);call=scope['__call__']
 conn=Connection(fd);torch.manual_seed(42);np.random.seed(42);t=time.perf_counter();teacher=SingleRefer();teacher.model.cpu();gc.collect();torch.cuda.empty_cache();load=time.perf_counter()-t
 rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate());conn.send(dict(ready=True,worker=identity(),load_seconds=load,load_audit=teacher.load_audit))
 while True:
  request=conn.recv()
  if request.get('stop'):break
  try:
   teacher.model.cuda();torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3]);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   z=call(teacher,request['frames'],request['caption']);torch.cuda.synchronize();z.update(peak_GPU_bytes=torch.cuda.max_memory_allocated(),peak_reserved_GPU_bytes=torch.cuda.max_memory_reserved(),model_load_seconds=0.,query_forwards=1,cache_hit=False)
   teacher.model.cpu();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();conn.send(dict(ok=True,payload=z))
  except BaseException:
   conn.send(dict(ok=False,error=traceback.format_exc()));raise
 conn.close()
if __name__=='__main__':run(int(sys.argv[1]))
