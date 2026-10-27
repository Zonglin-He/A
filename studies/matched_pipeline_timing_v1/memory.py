"""NVML process-only sampling; no model allocations or disk IO in sampler."""
import ctypes,threading,time
class PI(ctypes.Structure):
 _fields_=[('pid',ctypes.c_uint),('used',ctypes.c_ulonglong),('gi',ctypes.c_uint),('ci',ctypes.c_uint)]
class Monitor:
 def __init__(self,pids):
  self.lib=ctypes.CDLL('libnvidia-ml.so.1');assert self.lib.nvmlInit_v2()==0
  self.h=ctypes.c_void_p();assert self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.h))==0
  self.pids=set(pids);self.samples=[];self.stop_event=threading.Event();self.error=None
 def sample(self):
  b=(PI*128)();n=ctypes.c_uint(128);code=self.lib.nvmlDeviceGetComputeRunningProcesses_v2(self.h,ctypes.byref(n),b)
  if code:raise RuntimeError(f'NVML process query code{code}')
  own={b[i].pid:int(b[i].used)for i in range(n.value)if b[i].pid in self.pids}
  assert all(v<2**60 for v in own.values())
  self.samples.append((time.perf_counter(),sum(own.values()),own))
 def loop(self):
  try:
   while not self.stop_event.wait(.01):self.sample()
  except BaseException as e:self.error=repr(e)
 def start(self):
  self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def stop(self):
  self.stop_event.set();self.thread.join();self.sample();assert self.error is None,self.error
  return dict(sampled_process_peak_bytes=max(x[1]for x in self.samples),samples=len(self.samples),target_PIDs=sorted(self.pids),sampling_interval_seconds=.01,scope='sum only active pipeline PIDs, includes CUDA contexts; unrelated processes excluded; sampled peak can miss sub10ms transients')
