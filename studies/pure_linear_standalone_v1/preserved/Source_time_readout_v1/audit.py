from common import *
from score import measure
import argparse

def main(ds):
 validate();sys.addaudithook(guard);torch.set_num_threads(1);start=time.perf_counter();dest=D/ds;seal=read(dest/'sealed.json');sc=read(dest/'score_receipt.json');assert sc['metrics_sha256']==sha(dest/'metrics.json')and sc['seal_sha256']==sha(dest/'sealed.json');truth=labels(ds);m={r['id']:r for r in read(dest/'metrics.json')['private_rows']};assert set(m)==set(seal['files'])=={r['id']for r in rows(ds)};missing=0;affected=0;maxcoord=0.;checks=0
 for i,row in enumerate(rows(ds),1):
  resources(start);again,proof=inputs(ds,row,reference=True);e=seal['files'][row['id']];assert proof==e['proof']and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];missing+=proof['old_uncovered_video_frames'];affected+=int(proof['old_uncovered_video_frames']>0);g=truth[row['id']];gt={int(k):v for k,v in g['boxes'].items()};ivs=evaluation_intervals(ds,row['id'])
  with np.load(D/e['path'])as z:
   assert set(z.files)==set(again)
   for k,v in again.items():
    err=float(np.max(abs(v-z[k])));maxcoord=max(maxcoord,err);assert err<2e-10,(row['id'],k,err);checks+=1
   ix=np.searchsorted(z['frame_ids'],z['anchor_times']);assert np.array_equal(z['boxes_xyxy'][ix],z['anchor_xyxy']);a,b=bounds(ds,row['metadata']);assert z['frame_ids'][0]==a and z['frame_ids'][-1]==b-1 and len(z['frame_ids'])==b-a
   # Independent normalized-coordinate IoU against dense direct saved boxes.
   ids=z['frame_ids'];xy=z['boxes_xyxy'];W,H=row['metadata']['width'],row['metadata']['height'];scale=np.array([W,H,W,H]);f=np.array(sorted(gt));valid=(f>=ids[0])&(f<=ids[-1]);pred=np.zeros((len(f),4));pred[valid]=xy[f[valid]-ids[0]]/scale
   if ds=='hc2':pred=np.maximum(0,pred)
   target=np.array([gt[int(k)]for k in f])/scale;inter=np.maximum(0,np.minimum(pred[:,2:],target[:,2:])-np.maximum(pred[:,:2],target[:,:2])).prod(1);union=np.maximum(0,pred[:,2:]-pred[:,:2]).prod(1)+np.maximum(0,target[:,2:]-target[:,:2]).prod(1)-inter;iou=np.divide(inter,union,out=np.zeros(len(f)),where=union>0);iou[~valid]=0
   for arm,iv in ivs.items():
    aa,bb=map(int,iv);c,d=map(int,g['interval']);use=(f>=max(aa,c))&(f<min(bb,d));v=float(iou[use].sum()/max(max(bb,d)-min(aa,c),1));assert abs(v-m[row['id']]['values'][arm]['v'])<2e-10;checks+=1
   assert m[row['id']]['parent']==str(g['parent'])
  if i%100==0:atomic(dest/'audit_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 rep=read(P/ds/'report.json');mc=0
 for arm in ARMS:
  for f,v in rep['means'][arm].items():
   vv=[r['values'][arm][f]for r in m.values()if r['values'][arm][f]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12;mc+=1
 rec=dict(status='verified_full_direct_raw8_linear_trajectory_cold_inputs_SHA_parent_anchors_complete_tails_normalized_denseIoU_means',queries=len(m),parents=len({r['parent']for r in m.values()}),checks=checks,means_checks=mc,max_independent_corner_roundoff_pixels=maxcoord,old_source_grid_uncovered_video_frames=missing,queries_affected_by_old_missing_coverage=affected,full_video_segment_frames_complete=True,all8rawexpertanchors_exact=True,Source_E_features_mask_posterior_DP_in_interpolation=False,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),score_sha256=sha(dest/'score_receipt.json'),report_sha256=sha(P/ds/'report.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(dest/'audit_receipt.json',rec);atomic(P/ds/'audit_receipt.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);main(p.parse_args().dataset)
