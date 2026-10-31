"""No-GT actual sealed-output adapter check; never opens annotation/score index."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_transfer as s

def main(a):
 s.bound(a.predictions/'SEALED.json',a.seal_sha256);s.bound(a.pairs,a.pairs_sha256)
 seal=s.read(a.predictions/'SEALED.json');pairs=s.read(a.pairs)
 if seal['state']!='sealed_R14_Quad76_Aonly_transfer7arms_before_B_GT' or len(pairs)!=76 or seal['n_pairs']!=76:raise ValueError('Formal76 scope')
 for name,h in seal['files'].items():s.bound(s.member(a.predictions,name),h)
 checks=0;nonzero=0;maxerr={'half':0.,'full':0.};gate=0
 for pair in pairs:
  with np.load(a.predictions/(pair['pair_id']+'.npz'),allow_pickle=False) as z:
   d={k:z[k].copy() for k in s.NPZ_KEYS}
  assert tuple(d['arm_names'].tolist())==s.ARMS;checks+=1
  raw=d['arm_boxes_raw6'];assert raw.shape==(7,6) and raw.dtype==np.float32 and np.isfinite(raw).all();checks+=1
  s.validate_official_clamp(raw,d['arm_boxes_eval6']);checks+=1
  q=int(d['q_source']);assert np.array_equal(d['arm_query_index'],np.full(7,q)) and q==int(d['source_text_scores256'].argmax());checks+=1
  with np.load(pair['B_cache_path'],allow_pickle=False) as z:
   assert np.array_equal(d['source_boxes_raw6'],z['boxes_raw6'][0]) and np.array_equal(d['source_text_scores256'],z['text_scores'][0]) and q==int(z['query_index']);checks+=1
  with np.load(pair['A_cache_path'],allow_pickle=False) as z:
   aq=int(z['query_index']);assert aq==int(d['A_q_source']) and np.array_equal(d['A_source6'],z['boxes_raw6'][0,aq]);checks+=1
  with np.load(pair['A_geometry_path'],allow_pickle=False) as z:
   names=z['arm_names'].tolist();assert np.array_equal(d['A_target6'],z['arm_boxes_raw6'][names.index('B_inequality_full')]);checks+=1
  if bool(d['A_nonzero_residual']):
   nonzero+=1;delta=d['A_target6']-d['A_source6']
   assert np.array_equal(raw[1],raw[0]+delta);checks+=1
   assert np.array_equal(raw[2],raw[0]+np.tile(raw[0,3:]/d['A_source6'][3:],2)*delta);checks+=1
  else:assert np.array_equal(raw,np.tile(raw[0],(7,1)));checks+=1
  gate+=bool(d['A_gate'])
  for label,p,k in [('half',3,4),('full',5,6)]:
   err=float(np.max(np.abs(raw[p].astype(float)-raw[k].astype(float))));maxerr[label]=max(maxerr[label],err)
   assert err<=2e-4;checks+=1
 if nonzero!=49 or nonzero!=seal['nonzero_A_n']:raise ValueError('A signal count')
 result={'state':'passed_actual_formal_Quad76_score_input_bridge_NO_GT_NO_IoU',
  'n':76,'checks':checks,'A_gate_n':int(gate),'A_nonzero_signal_n':nonzero,
  'parameter_kernel_selected_max_abs_error_m':maxerr,
  'prediction_seal_sha256':s.sha(a.predictions/'SEALED.json'),'pairs_sha256':s.sha(a.pairs),
  'program_sha256':s.sha(Path(__file__)),'scorer_sha256':s.sha(Path(s.__file__)),
  'actual_GT_index_annotation_taskIoU_P_image_feature_model_or_GPU_accessed':False,
  'scope':'No-GT scalar/selected predictions and source raw-box/cache bridges only. Does not authorize any target score.'}
 if a.output.exists():raise FileExistsError(a.output)
 s.dump(a.output,result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('predictions','pairs','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--seal-sha256',required=True);p.add_argument('--pairs-sha256',required=True);main(p.parse_args())
