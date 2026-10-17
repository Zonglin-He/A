"""Independent CPU normal-equation replay of every fixed Tube proposal; no GT."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_followups64_v1 import BASE,read,sha,put

def main():
 import numpy as np,torch
 from scipy.special import expit
 from scripts import run_stvg_adjacent_motion_student_v1 as h
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 rr={r['probe_alias']:r for r in read(BASE/'ROSTER_LOCK.json')['records']};rows=[]
 for r in read(BASE/'INPUT_LOCK.json')['records']:
  case,_=load_case(rr[r['case_id']]);assert sha(ROOT/r['proposal_path'])==r['proposal_sha256'];p=torch.load(ROOT/r['proposal_path'],map_location='cpu',weights_only=False);src=case['zero']['boxes'].cpu().double().numpy();assert np.array_equal(src,p['source_cxcywh']);a=h._anchors(r['reference_record'],r['native_frame_ids'],torch);pos=[z['position'] for z in a];ref=np.array([z['box'].tolist() for z in a],dtype=np.float64).reshape(-1,4);n=len(src);dt=np.diff(np.array(r['native_frame_ids'],dtype=np.float64)/case['input']['fps']);errors=[]
  if p['completion']['status']=='completed':
   for drop in [None]+([p['leave_one_reference']['heldout_reference_index']] if p['leave_one_reference'] else []):
    use=[j for j in range(len(pos)) if j!=drop];pp=[pos[j] for j in use];e=ref[use];k=len(pp);x=np.log(src)-np.log1p(-src);y=np.log(e)-np.log1p(-e)-x[pp];S=np.eye(n)[pp];D=np.zeros((n-1,n))
    for j in range(n-1):D[j,j]=-1/np.sqrt(dt[j]);D[j,j+1]=1/np.sqrt(dt[j])
    H=np.eye(n)/n+S.T@S/k+D.T@D/(n-1);rhs=S.T@y/k;z=np.linalg.solve(H,rhs);pred=expit(x+z)
    if drop is None:actual=p['proposal_cxcywh'];assert np.allclose(z,p['residual_logit'],atol=1e-10,rtol=1e-10)
    else:actual=np.array(p['leave_one_reference']['heldout_prediction_cxcywh']);pred=pred[pos[drop]]
    error=float(np.max(np.abs(pred-actual)));assert error<1e-10;errors.append(error)
  else:assert np.array_equal(src,p['proposal_cxcywh'])
  rows.append(dict(case_id=r['case_id'],dataset=r['dataset'],completion_status=p['completion']['status'],reason=p['completion'].get('reason'),independent_maximum_absolute_error=max(errors) if errors else 0.,leave_one_replayed=p['leave_one_reference'] is not None,GT_read=False))
 put(BASE/'tube/ROOT_COMPLETION_MATH_READBACK.json',dict(status='pass',count=64,independent_dense_matrix_normal_equations=True,actual_physical_seconds=True,proposal_and_leave_one_replayed=True,records=rows,GT_read=False));print('TUBE64_INDEPENDENT_NORMAL_EQUATIONS_PASS')
if __name__=='__main__':main()
