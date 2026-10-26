"""Real retained-grid gamma sensitivity compatibility preflight; no labels/model."""
from review_common import *
import numpy as np,torch
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.representation import extract
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval
from stvg.predictor import output_writeback
from stvg.initialization import initialize

def derive(c,raw,ds,gamma):
 from studies.map_paper_completion_v1.spatial_core import raw_evidence
 cfg=read(R/'configs/default.json')['datasets'][ds];native=np.asarray(c['geometry']['native_boxes']);u,_=initialize(c['Source'],c['positions8'],native,cfg['E_calibration']);E=output_writeback(c['Source'],u,gamma)
 ev=raw_evidence(c,raw);g=SpatialGeometry(ev,E,native);f,_=extract(raw,E);q,_=one_step(f,c['native_prior'],kappa=0,beta=1);iv=map_interval(c['frame_ids'],q);b=evidence_dp(g.for_interval(iv))['final'].numpy()
 return dict(E=E,features=f,posterior=q,interval=np.asarray(iv),boxes=b)

def main():
 gt_guard();torch.set_num_threads(1);p=read(D/'gamma_preflight_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for path,h in p[field].items():assert sha(R/path)==h,path
 reuse=read(selected.PREV/'raw_reuse_index.json');proof=[]
 for ds in COUNTS:
  rows={r['id']:r for r in records(ds)}
  for key,e in list(reuse[ds].items())[:2]:
   c,meta=selected.suite().load_parent(ds,rows[key]);assert sha(e['path'])==e['sha256'];raw=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True);raw['frame_ids']=c['frame_ids']
   arrays={}
   for gamma in [0,.25,.5,.75,1]:
    a=derive(c,raw,ds,gamma);b=derive(c,raw,ds,gamma);assert all(np.array_equal(a[k],b[k])for k in a);arrays[str(gamma)]=dict(repeat_exact=True,actual_E_ROI_repool=True,posterior_and_final_DP_recomputed=True)
    if gamma==read(R/'configs/default.json')['datasets'][ds]['initialization_gamma']:
     assert np.array_equal(a['E'],c['E']);_,ce=paths(ds,key)
     with np.load(selected.D/ce['path'])as z:assert np.array_equal(a['boxes'],z['FullMAP'])and np.array_equal(a['interval'],z['FullMAP_interval'])
   proof.append(dict(dataset=ds,retained_raw_sha256=e['sha256'],parent_capture_sha256=meta['capture_sha256'],five_gamma_values=arrays,current_selected_output_bitwise=True))
 rec=dict(status='passed',cases=proof,GT_read=False,new_Source=0,new_expert=0,new_model=0,new_optimizer=0,new_GT_scoring=0,scope='Four actual retained grids, 5 gamma each twice; no full-domain sensitivity result, no new GPU bridge')
 atomic(D/'gamma_preflight_receipt.json',rec);atomic(P/'gamma_preflight_receipt.json',rec);print(json.dumps({'status':'passed','cases':4,'gamma_points':5,'current_selected_bitwise':True}))
if __name__=='__main__':main()
