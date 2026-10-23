"""Actual native Source/grid no-op plus coupled SimplePath/all-video feedback bridge."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from full import RUN,PUBLIC,NATIVEPRIOR,read,sha,atomic,identity,check
from stvg.runtime import setup,lease,load_source
from stvg.decode import decode
from stvg.query import QuerySubjectParser
from stvg.native import frozen_native_forward
from stvg.prior import native_prior,from_capture
from stvg.predictor import Predictor
from stvg.representation import observe_native_grid,extract
from stvg.visual_cache import VisualCache
from stvg.visual_prefix import native_appearance_grid
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.prototype import one_step
from stvg.temporal import final_mbr
from stvg.paper_panel import derive


def main():
 plan=check(True);setup();torch.set_num_threads(1);lock=lease();started=time.perf_counter()
 items=read(RUN/'promotion_inputs.json');parser=QuerySubjectParser(ROOT/'checkpoints/stanza');vc=VisualCache(4*2**30)
 spec=importlib.util.spec_from_file_location('frozen_prior_spatial',NATIVEPRIOR/'stvg/spatial.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 checks=0;changed_aux=0;changed_pool=0;cases=[]
 atomic(RUN/'bridge_worker.json',identity())
 try:
  for ds in ['hc2','vidstg']:
   model=load_source(read(ROOT/'configs/default.json')['datasets'][ds])
   for item in [x for x in items if x['dataset']==ds][:2]:
    assert sha(item['path'])==item['sha256'] and sha(item['feature_path'])==item['feature_sha256']
    cap=torch.load(item['path'],map_location='cpu',weights_only=False);tr=torch.load(item['feature_path'],map_location='cpu',weights_only=False)
    expected=tr['features']['Native2D'] if isinstance(tr['features'],dict)else tr['features']
    old=legacy.SpatialGeometry.__new__(legacy.SpatialGeometry);old.__dict__.update(cap['geometry'])
    g=SpatialGeometry.from_cached_geometry(cap['geometry']);ids=cap['frame_ids']
    old_aux=legacy.evidence_dp(old.for_interval([ids[0],ids[-1]+1]))['final'].numpy()
    aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy()
    changed_aux+=int(not np.array_equal(aux,old_aux))
    frames,physical=decode(item['row']['metadata'],ds);assert ids==physical
    row=dict(input=item['row']['metadata'],frame_ids=ids,parses={'subject':parser(item['row']['metadata']['caption'])['subject']})
    first=None
    for repeat in range(2):
     torch.manual_seed(20260920);np.random.seed(20260920);model.cuda()
     with torch.no_grad(),observe_native_grid(model)as grid,vc.context(model,frames,ids):batch,recs,base=frozen_native_forward(model,frames,row)
     assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])
     logs=[v.detach().cpu().clone()for v in base.zero['logits']];q0=native_prior(recs,logs,ids);assert np.array_equal(q0,from_capture(cap))
     oldfeatures,_=extract(dict(frame_ids=ids,native_local_grid=grid),old_aux);assert np.array_equal(oldfeatures,expected)
     features,_=extract(dict(frame_ids=ids,native_local_grid=grid),aux)
     manualq,_=one_step(features,q0);manualiv=final_mbr(ids,manualq)['interval'];manualboxes=evidence_dp(g.for_interval(manualiv))['final'].numpy()
     out=Predictor(ds).finish(dict(Source=cap['Source'],E=cap['E'],geometry=g,frame_ids=ids,native_local_grid=grid,native_logits=logs,records=recs,posterior={'Native':q0},intervals={'Native':final_mbr(ids,q0)['interval']},planned_positions=cap['positions8'],cost={},initialization={}))
     assert np.array_equal(out['content_features'],features) and np.array_equal(out['posterior'],manualq) and np.array_equal(out['boxes'],manualboxes) and out['interval']==manualiv
     if first is None:first=(features.copy(),manualq.copy(),manualboxes.copy());changed_pool+=int(not np.array_equal(features,oldfeatures))
     else:
      for a,b in zip(first,(features,manualq,manualboxes)):assert np.array_equal(a,b)
     checks+=1;del batch,base
    # Actual lightweight appearance prefix must reproduce EVERY raw grid cell.
    with vc.context(model,frames,ids):prefix=native_appearance_grid(model,frames,ids,item['row']['metadata'])
    assert all(torch.equal(a,c)and torch.equal(b,d)for(a,b),(c,d)in zip(prefix,grid))
    prefixfeatures,_=extract(dict(frame_ids=ids,native_local_grid=prefix),aux);assert np.array_equal(prefixfeatures,first[0])
    cases.append(dict(dataset=ds,Source_exact=True,native_prior_exact=True,old_pool_exact=True,new_pool_recomputed=True,prefix_raw_grids_exact=True,cold_cache_final_exact=True))
    print(ds,'coupled cold/cachehit Source/nativeprior/newaux/newpool/NoSplit/final exact; prefix raw cells exact',flush=True)
   del model;torch.cuda.empty_cache()
  assert checks==8 and changed_aux>0 and changed_pool>0
  out=dict(status='passed',queries=4,checks=checks,actual_full_Source_calls=8,appearance_prefix_calls=4,new_aux_repool_verified=True,changed_aux_queries=changed_aux,changed_pooled_feature_queries=changed_pool,cases=cases,new_expert=0,new_labels=0,seconds=time.perf_counter()-started,plan_sha256=sha(RUN/'plan.json'),code_sha256=sha(Path(__file__)),visual=vc.stats())
  atomic(RUN/'bridge.json',out);atomic(PUBLIC/'BRIDGE.json',out)
 finally:lock.close()
if __name__=='__main__':main()
