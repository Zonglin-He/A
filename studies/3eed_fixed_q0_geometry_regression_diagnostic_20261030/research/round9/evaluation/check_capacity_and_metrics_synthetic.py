"""Pure synthetic CPU fixtures; does not read real data/annotation/model weights."""
import json,hashlib
from pathlib import Path
import numpy as np
from geometry_metrics import *
rng=np.random.default_rng(901)
# Actual final linear-row optimization; distinct norms prevent max-norm falsefit.
hc=rng.normal(size=288);hs=rng.normal(size=288)*3
Wc=rng.normal(size=(3,288));Ws=rng.normal(size=(3,288));bc=rng.normal(size=3);bs=rng.normal(size=3)
base=rng.normal(size=3);source=np.r_[base+Wc@hc+bc,Ws@hs+bs];target=source+np.array([2.,-1.,.3,-2.,1.,.5]);errors={}
for mode,mask in [('C',np.r_[np.ones(3),np.zeros(3)]),('S',np.r_[np.zeros(3),np.ones(3)]),('CS',np.ones(6))]:
 wc,ws,cb,sb=Wc.copy(),Ws.copy(),bc.copy(),bs.copy();e=(source-target)*mask
 # One SGD on .5sum(e^2); lr per independent head1/(||h||²+1).
 wc-=np.outer(e[:3],hc)/(hc@hc+1);cb-=e[:3]/(hc@hc+1)
 ws-=np.outer(e[3:],hs)/(hs@hs+1);sb-=e[3:]/(hs@hs+1)
 out=np.r_[base+wc@hc+cb,ws@hs+sb];direct=np.where(mask,target,source)
 err=float(abs(out-direct).max());assert err<1e-11
 assert np.array_equal(wc,Wc) if mode=='S' else True
 assert np.array_equal(ws,Ws) if mode=='C' else True
 errors[mode]=err
# Same max(norm) LR is NOT exact for both independent heads.
e=source-target;lr=1/max(hc@hc+1,hs@hs+1);wrong=source-lr*e*np.r_[np.full(3,hc@hc+1),np.full(3,hs@hs+1)];assert abs(wrong-target).max()>.1
# Geometry duplicates from expressions+others are merged before competitor max.
t=np.array([0.,0.,0.,2.,2.,2.,0.]);other=t.copy();other[0]=5.;inv=np.array([t,t,other,other]);groups,tg=dedup_inventory(inv,0);assert len(groups)==2
c=spatial_confirmation(t[:6],t,inv,0);assert c['annotated_spatially_confirmed'] and c['geometric_duplicate_records']==2
assert not spatial_confirmation(t[:6],t,np.array([t]),0)['annotated_spatially_confirmed']
assert not spatial_confirmation(other[:6],t,inv,0)['annotated_spatially_confirmed']
# Strict Acc threshold and yaw0 target-dimension replacement not maximalIoU.
boxes=np.array([[0,0,0,1,1,2],[0,0,0,1,2,2]],float);vals,_=candidate_ious(boxes,t);assert vals.tolist()==[.25,.5] and not bool(vals[0]>.25) and not bool(vals[1]>.5)
rotgt=np.array([0.,0.,0.,4.,1.,1.,np.pi/2]);direct=rotgt[:6].copy();alternative=direct.copy();alternative[3:5]=[1.,4.];v,_=candidate_ious(np.array([direct,alternative]),rotgt);assert v[1]>v[0]
# All GT-bearing file access requires main gate; importing metrics has no IO.
out={'state':'synthetic_CPU_fixtures_pass','synthetic_only':True,'linear_L2_exactfit_max_error_by_C_S_CS':errors,'wrong_common_maxnorm_not_exact':True,'dedup_repeated_expression_boxes':True,'empty_competitors_not_identity_confirmed':True,'wrong_target_rejected':True,'strict25_50_thresholds':True,'GT_dimensions_at_yaw0_not_maximal_IoU_demonstrated':True,'no_model_GPU_realdata_GT_file_access':True}
p=Path(__file__).with_name('CPU_SYNTHETIC_FIXTURE.json');assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
