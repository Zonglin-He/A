"""Independent scalar sums, discrete-support checks and saved OPD path diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_stvg_dino_error_attribution32_v1 import BASE,JOINT,OLD,read,put,load,sha
def run():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_decota_paper_main_v1 import read_row
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 from vg_tta.tastvg_oracle_event5_v1 import box_iou,DenseTube
 result=read(BASE/'NUMERIC_RESULTS.json');records=read(JOINT/'PLAN_LOCK.json')['target'];checks=0;details=[]
 targets={ds:truths(ds,{'parents':[r['query_ordinal']for r in records if r['dataset']==ds]})for ds in ['hc2','vidstg']}
 for record,saved in zip(records,result['rows']):
  alias=record['alias'];ds=record['dataset'];ordinal=record['query_ordinal'];row=read_row(ds,ordinal)
  z=load(JOINT/'target'/(alias+'.pt'));p=load(OLD/'predictions'/(alias+'.pt'));gt=targets[ds][0][ordinal];span=targets[ds][1][ordinal]
  ids=row['frame_ids'];positions={a['position']for a in z['expert']['anchors']['single4']};observed={ids[i]for i in positions}
  geometries={}
  for a in z['expert']['anchors']['single4']:
   fid=ids[a['position']];geometries[fid]=dict(in_event=span[0]<=fid<span[1],GT_available=fid in gt,
    unavailable_reason=None if fid in gt else('outside_GT_event'if not span[0]<=fid<span[1]else'no_box_annotation_at_exact_frame'))
  for method,before,after,interval in [('Direct',z['boxes'][0],z['boxes'][1],z['native_interval']),('OPD_current',p['fit']['before'],p['fit']['final'],p['arms']['Full_P7A']['interval'])]:
   # Independently interpolate one frame at a time instead of DenseTube vectorization.
   bbs=[xyxy(v.numpy(),row['input']['width'],row['input']['height'])for v in [before,after]]
   if ds=='hc2':bbs=[np.maximum(b,0)for b in bbs]
   sums={k:[0.,0.,0]for k in ['observed','unobserved_native','benchmark_interpolated']}
   lo,hi=interval;g,h=span;den=max(max(hi,h)-min(lo,g),1)
   for f,truth in gt.items():
    vv=[]
    for b in bbs:
     pred=[float(np.interp(f,ids,b[:,c]))for c in range(4)]
     vv.append(float(box_iou(pred,truth))if ids[0]<=f<=ids[-1]else 0.)
    delta=vv[1]-vv[0];key='observed'if f in observed else('unobserved_native'if f in ids else'benchmark_interpolated')
    sums[key][0]+=100*delta/len(gt)
    if max(lo,g)<=f<min(hi,h):sums[key][1]+=100*delta/den
    sums[key][2]+=1
   for k,(s,v,n)in sums.items():
    ref=saved['methods'][method]['components'][k]
    assert abs(s-ref['dense_s_contribution_pp'])<1e-8 and abs(v-ref['v_contribution_pp'])<1e-8 and n==ref['GT_frames'];checks+=3
  anchors=[]
  for a in z['expert']['anchors']['single4']:
   pos=a['position'];fid=ids[pos]
   if fid not in gt:continue
   quality=float(box_iou(xyxy(np.array(a['box']),row['input']['width'],row['input']['height']),gt[fid]))
   tb=box_iou(xyxy(z['boxes'][0][pos].numpy(),row['input']['width'],row['input']['height']),gt[fid])
   anchors.append(dict(slot=z['expert']['positions4'].index(pos),reference_GT_IoU=quality,reference_better_than_Source=quality>float(tb)+1e-12,
    OPD_before=float(box_iou(xyxy(p['fit']['before'][pos].numpy(),row['input']['width'],row['input']['height']),gt[fid])),
    OPD_after=float(box_iou(xyxy(p['fit']['final'][pos].numpy(),row['input']['width'],row['input']['height']),gt[fid]))))
  paths=[]
  for k in range(len(p['fit']['path'])-1):
   bb=[p['fit']['path'][i]['boxes'].numpy()for i in [k,k+1]]
   tubes=[DenseTube(b,row,gt,span,clip=ds=='hc2')for b in bb];diff=tubes[1].iou-tubes[0].iou;fids=tubes[0].fids
   good_ids={ids[a['position']]for a in z['expert']['anchors']['single4']if ids[a['position']]in gt and box_iou(xyxy(np.array(a['box']),row['input']['width'],row['input']['height']),gt[ids[a['position']]])>=.5}
   obs=np.isin(fids,list(good_ids));unobs=np.isin(fids,ids)&~np.isin(fids,list(observed))
   if obs.any()and unobs.any()and diff[obs].mean()>1e-12 and diff[unobs].mean()<-1e-12:
    paths.append(dict(step=k+1,good_reference_observed_mean_delta_pp=float(100*diff[obs].mean()),unobserved_native_mean_delta_pp=float(100*diff[unobs].mean()),good_reference_anchor_count=int(obs.sum()),unobserved_native_GT_count=int(unobs.sum())))
  details.append(dict(source_alias=alias,dataset=ds,admitted_support=[dict(slot=z['expert']['positions4'].index(a['position']),**geometries[ids[a['position']]])for a in z['expert']['anchors']['single4']],
   anchors=anchors,OPD_rounds_good_reference_observed_up_unobserved_native_down=paths,
   note='IoU>=.5 is only a disclosed geometric diagnostic threshold, not an identity or deployment gate'))
 put(BASE/'ROOT_NUMERIC_AUDIT.json',dict(status='pass',count=32,independent_scalar_checks=checks,scalar_results_sha256=sha(BASE/'NUMERIC_RESULTS.json'),CPU_only=True,details=details))
 print(json.dumps(dict(status='pass',checks=checks,path_witnesses={r['source_alias']:len(r['OPD_rounds_good_reference_observed_up_unobserved_native_down'])for r in details if r['OPD_rounds_good_reference_observed_up_unobserved_native_down']})))
if __name__=='__main__':run()
