"""First-two sealed rows only: diagnose numeric layout without GT or rerun."""
from pathlib import Path
import json,hashlib
import numpy as np
from audit_predictions50 import conditional
from audit_prepared50 import oldw,oldread
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def compare(x,y):
 x,y=np.asarray(x),np.asarray(y);return {'shape_equal':x.shape==y.shape,'dtype_x':str(x.dtype),'dtype_y':str(y.dtype),'bitexact':x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes(),'different_values':int((x!=y).sum()),'max_abs_difference':float(abs(x.astype(float)-y.astype(float)).max()) if x.size else 0.}
def layout(x):return {'shape':list(x.shape),'strides':list(x.strides),'C_contiguous':bool(x.flags.c_contiguous),'F_contiguous':bool(x.flags.f_contiguous)}
def main():
 pred=ROOT/'crossplatform/round50/predicted336_attempt001';prep=ROOT/'crossplatform/round50/prepared336_attempt001';auth=read(ROOT/'research/round50/experiment/ROOT_INDEPENDENT_PREDICTION_AUDIT_AUTHORIZATION.json');assert sha(pred/'SEALED.json')==auth['prediction_seal_sha256'];assert read(HERE/'prediction_audit_attempt001/FAILED.json')['error']=='seven_scores'
 out=HERE/'numeric_diagnostic_first2_attempt001';assert not out.exists();out.mkdir();rows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json')[:2];diagnostics=[]
 for i,row in enumerate(rows):
  p=prep/'arrays_PRIVATE'/(row['id']+'.npz');z=load(pred/'samples_PRIVATE'/(row['id']+'.npz'));a=load(p);s=a['source_scores'];e=z['image_e'];pr=a['raw_mask_PR'];c=a['mask_counts'];n=a['allvisible'];pool=a['old_pool'];wide=a['wide_pool'];old=[]
  for pp in (pool,wide):
   w,_,_=oldw(pp,a['legacy_support'],a['legacy_coverage'],a['legacy_projection_iou'],a['source_boxes']);old.append(oldread(s,w,pp,e,pr)['scores'])
  rowdiag={'anonymous_row':i+1,'source':compare(s.astype(float),z['scores'][0]),'oldG':compare(old[0],z['scores'][1]),'wideoldW':compare(old[1],z['scores'][2]),'new_arms':[]}
  for j,(k,tag,pp) in enumerate([(3,'old',pool),(3,'wide',wide),(10,'old',pool),(10,'wide',wide)]):
   aa=a[f'{tag}_k{k}_conditional_association'];rho=a[f'{tag}_k{k}_rho'];selected=c[:,pp].astype(np.float64);den=selected.sum(1);recreated=np.zeros_like(selected);np.divide(selected,den[:,None],out=recreated,where=den[:,None]>0)
   with np.load(p,allow_pickle=False) as source:original_layout=source[f'{tag}_k{k}_conditional_association']
   states={name:conditional(s,arr,rho,pp,e,pr) for name,arr in [('loaded_C_copy',aa),('recreated_original_allocation',recreated),('explicit_C',np.array(aa,order='C',copy=True)),('explicit_F',np.array(aa,order='F',copy=True))]}
   arm={'k':k,'entry':tag,'conditional_stored_vs_recreated':compare(aa,recreated),'stored_original_npy_layout':layout(original_layout),'auditor_loaded_layout':layout(aa),'recreated_layout':layout(recreated),'variants':{}}
   for name,st in states.items():
    arm['variants'][name]={'scores':compare(st['scores'],z['scores'][j+3]),'query_same':int(st['scores'].argmax())==int(z['queries'][j+3]),'components':{key:compare(st[key],z[key][j+3]) for key in ('vision','position','relation')},'conditional_clipped':{key:compare(st[key],z[key][j]) for key in ('conditional_vision_clipped','conditional_position_clipped','conditional_relation_clipped')}}
   rowdiag['new_arms'].append(arm)
  diagnostics.append(rowdiag)
 result={'state':'NO_GT_FIRST2_NUMERIC_LAYOUT_DIAGNOSTIC_COMPLETE_NOT_AUDIT_RERUN','rows':diagnostics,'prediction_immutable_sha256':sha(pred/'SEALED.json'),'program_sha256':sha(__file__),'no_model_GT_or_method_change':True}
 (out/'DIAGNOSTIC.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'state':result['state'],'variants_per_arm':[{'row':r['anonymous_row'],'arms':[{'k':a['k'],'entry':a['entry'],'scores':{name:v['scores'] for name,v in a['variants'].items()}} for a in r['new_arms']]} for r in diagnostics]}))
if __name__=='__main__':main()
