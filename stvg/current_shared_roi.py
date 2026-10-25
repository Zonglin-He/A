"""Selected no-shrinkage MAP method; immutable legacy interfaces remain separate."""
import json,hashlib,inspect,textwrap,time,argparse
from pathlib import Path
import numpy as np,torch
from .prototype import one_step
from .spatial import evidence_dp
from .representation import extract
ROOT=Path(__file__).resolve().parents[1]
def profile():
 p=json.loads((ROOT/'profiles/shared_corrected_roi_v1.json').read_text());assert p['kappa']==0 and p['beta']==1 and p['rounds']==1 and p['first_auxiliary_DP'] is False and p['final_interval_DP'] is True
 assert hashlib.sha256((ROOT/p['base_config']).read_bytes()).hexdigest()==p['base_config_sha256'];return p
def map_interval(ids,q):
 q=np.asarray(q,dtype=np.float64);assert q.shape==(len(ids),len(ids)) and np.isfinite(q).all() and (q>=0).all() and abs(q.sum()-1)<1e-10
 a,b=np.unravel_index(int(q.argmax()),q.shape);assert a<=b and q[a,b]>0;return [int(ids[a]),int(ids[b])+1]
def predictor_class():
 from . import predictor as old
 profile();code=textwrap.dedent(inspect.getsource(old.BoxFixedPredictor.capture));before="native_interval=final_mbr(U,q)['interval']";assert code.count(before)==1
 scope={**vars(old),'map_interval':map_interval};exec(compile(code.replace(before,'native_interval=map_interval(U,q)'),'<selected_shared_corrected_ROI_MAP_capture>','exec'),scope)
 return type('SharedCorrectedROIMapLivePredictor',(old.BoxFixedPredictor,),{'capture':scope['capture']})
@torch.no_grad()
def finish(cache):
 profile();t=time.perf_counter();tick=t;ids=cache['frame_ids'];q0=cache['posterior']['Native'];auxs=0.;f,_=extract(cache,np.asarray(cache['E']));pool=time.perf_counter()-tick;tick=time.perf_counter();q,_=one_step(f,q0,kappa=0,beta=1);proto=time.perf_counter()-tick;tick=time.perf_counter();iv=map_interval(ids,q);maps=time.perf_counter()-tick;tick=time.perf_counter();out=evidence_dp(cache['geometry'].for_interval(iv));dps=time.perf_counter()-tick
 return dict(boxes=out['final'].numpy(),interval=iv,positions8=cache['planned_positions'],frame_ids=ids,posterior=q,content_features=f,cost=dict(auxiliary_DP_seconds=auxs,region_pooling_seconds=pool,prototype_seconds=proto,MAP_readout_seconds=maps,final_DP_seconds=dps,finish_wall_seconds=time.perf_counter()-t),GT_read=False)
def main():
 a=argparse.ArgumentParser(description='Current selected kappa0 jointMAP + exactDP');a.add_argument('command',choices=['predict']);a.add_argument('--request',required=True);a.add_argument('--output',required=True);z=a.parse_args()
 from .runtime import setup,lease
 from .decode import decode
 from .inputs import validate_input,validate_metadata
 request=json.loads(Path(z.request).read_text());assert not(set(request)-{'dataset','key','metadata'});validate_metadata(request['metadata']);setup();lock=lease()
 try:
  frames,ids=decode(request['metadata'],request['dataset']);validate_input(frames,ids,request['metadata']);model=predictor_class()(request['dataset']);cap=model.capture(frames,ids,request['metadata']);out=finish(cap);p=Path(z.output);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.savez_compressed(p,FullMAP=out['boxes'],FullMAP_interval=out['interval'],frame_ids=ids,positions8=out['positions8']);p.with_suffix('.json').write_text(json.dumps(dict(profile=profile(),prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),GT_read=False,capture_cost=cap['cost'],finish_cost=out['cost']),indent=2))
 finally:lock.close()
if __name__=='__main__':main()
