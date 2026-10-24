"""Live current-MAP method: original native Source/8expert, no MBR readout."""
from common import *
import inspect,textwrap,time
import numpy as np,torch
from map_core import map_interval
from stvg.spatial import evidence_dp
from stvg.representation import extract
from stvg.prototype import one_step

def predictor_class():
 import stvg.predictor as old
 code=textwrap.dedent(inspect.getsource(old.BoxFixedPredictor.capture));before="native_interval=final_mbr(U,q)['interval']";assert code.count(before)==1
 scope={**vars(old),'map_interval':map_interval};exec(compile(code.replace(before,'native_interval=map_interval(U,q)'),'<native_capture_current_q0_MAP>','exec'),scope)
 return type('CurrentMAPLivePredictor',(old.BoxFixedPredictor,),{'capture':scope['capture']})
@torch.no_grad()
def finish(cache):
 t=time.perf_counter();tick=t;q0=cache['posterior']['Native'];ids=cache['frame_ids'];aux=evidence_dp(cache['geometry'].for_interval([ids[0],ids[-1]+1]))['final'].numpy();auxs=time.perf_counter()-tick;tick=time.perf_counter();f,_=extract(cache,aux);pool=time.perf_counter()-tick;tick=time.perf_counter();q,_=one_step(f,q0);proto=time.perf_counter()-tick;tick=time.perf_counter();iv=map_interval(ids,q);maps=time.perf_counter()-tick;tick=time.perf_counter();s=evidence_dp(cache['geometry'].for_interval(iv));dps=time.perf_counter()-tick
 return dict(boxes=s['final'].numpy(),interval=iv,positions8=cache['planned_positions'],frame_ids=ids,posterior=q,content_features=f,cost=dict(auxiliary_DP_seconds=auxs,region_pooling_seconds=pool,prototype_seconds=proto,MAP_readout_seconds=maps,final_DP_seconds=dps,finish_wall_seconds=time.perf_counter()-t),GT_read=False)
