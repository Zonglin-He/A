"""Independent source-target replay and oriented-IoU audit of direct teachers."""
from pathlib import Path
import sys,json,numpy as np,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round106';B=R/'crossplatform/round106';O=B/'direct_teacher_supplement_PRIVATE';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou
def main():
 matrix=rd(O/'MATRIX_SEALED.json');tasks=rd(H/'DIRECT_TEACHER_ROWS_PRIVATE.json');lookup={(x['source'],x['target'],x['id_PRIVATE']):x for x in tasks};n=0;error=0.
 for name,h in matrix['directions'].items():
  src,_,tgt=name.split('_');assert sha(O/name/'SEALED.json')==h;seal=rd(O/name/'SEALED.json')
  for f,hh in seal['files'].items():assert sha(O/name/f)==hh
  rows=rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');golds=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json')
  for r in rows:
   z=np.load(O/name/(r['id']+'.npz'));old=np.load(B/name/'samples_PRIVATE'/(r['id']+'.npz'));p=dict(zip(old['arm_names'],old['scores']));rec=rd(B/name/'records_PRIVATE'/(r['id']+'.json'));fit=rec['fits'].get('real');q=np.array(fit['target']) if fit else p['prepared_frozen'];on=rec['families'].get('real',{}).get('updated_output_enabled',False);wanted=[q,q if on else p['real_frozen_fusedABC'],(q+p['real_frozen_C'])/2 if on else p['real_frozen_ensemble']];assert np.array_equal(z['scores'],np.stack(wanted));assert np.array_equal(z['source_boxes'],old['source_boxes']);gt=np.array(golds[r['anno_index']]['source_GT7_float32']);c,s=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
   for arm,pp in zip(z['arm_names'],z['scores']):
    b=z['source_boxes'][pp.argmax()].astype(float);b[3:]=np.maximum(b[3:],1e-6);y=oriented_iou(b[:3],b[3:],np.eye(3),gt[:3],gt[3:6],rot);e=abs(y-lookup[(src,tgt,r['id'])]['arms'][str(arm)]);error=max(error,e);assert e<1e-6;n+=1
 assert n==2304;p=H/'DIRECT_TEACHER_AUDIT.json';p.write_text(json.dumps({'state':'PASS_EXACT_ALREADY_SEALED_TARGETS_COVERAGE_FALLBACKS_AND_INDEPENDENT_ORIENTED_IOU','n':768,'readouts':n,'max_IoU_error':error,'new_model_calls':0,'FINAL_access':False},indent=2)+'\n');print('TEACHER106_AUDIT_PASS',n,error)
if __name__=='__main__':main()
