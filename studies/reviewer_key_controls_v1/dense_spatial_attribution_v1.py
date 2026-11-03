"""Exact offline decomposition of the already scored matched-time tube gain.

This supplements the native-frame distance table with the official dense metric
support. All contributions retain ALL queries, including zero-support queries.
"""
from common_v1 import *
import numpy as np,math
NAMES=['observed','unobserved_0_05','unobserved_05_1','unobserved_1_2','unobserved_2_4','unobserved_4_inf']
BASE=R/'outputs/review_strengthening_v1';DEST=D/'dense_spatial_attribution';PUB=P/'dense_spatial_attribution'
def prepare():
 assert not(DEST/'plan_v1.json').exists();files=[]
 for ds in COUNTS:
  files += [BASE/'matched_cpu'/ds/f for f in ['sealed.json','reporting_receipt.json','metrics.json']]
  j=read(selected.RUN/'plan.json')['jobs'][ds];files += [selected.PROVIDER/j['labels'],selected.PROVIDER/j['manifest']]
 p=dict(schema='exact_dense_matched_time_spatial_v_attribution_v1',scope='Only original complete audited same-interval Ours and ExpertInterpolation paths, no new prediction/model/acquisition. Offline GT only. Decompose the exact already-published Table10 per-query vIoU difference, not infer contribution from native-frame means.',groups=NAMES,rule='Ground-truth-annotated dense frame IDs in predicted AND GT half-open interval, official prediction pixel interpolation/no extrapolation. Distances from actual8 physical observed IDs/fps. Observed frames separate; unobserved bins[0,.5),[.5,1),[1,2),[2,4),[4,inf).',contribution='100*sum(frameIoU_Ours-frameIoU_Interpolation on group)/official temporal enclosing-span denominator per query. Sum6groups equals old audited delta-v pointwise. All queries remain denominator, absence of group support contributes0.',statistics='query-equal and parent10000 bootstrap seed20261022 pointwise95%, counts of supported queries/frames also report; not independent-frame tests, not unique causal attribution or new method.',code_sha256={str(Path(__file__).relative_to(R)):sha(Path(__file__))},input_sha256={str(f):sha(f)for f in files});p['fingerprint']=digest(p);atomic(DEST/'plan_v1.json',p);atomic(PUB/'protocol_v1.json',{k:v for k,v in p.items()if k!='input_sha256'});print(p['fingerprint'])
def check():
 p=read(DEST/'plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for k in ['code_sha256','input_sha256']:
  for f,h in p[k].items():assert sha(R/f)==h,f
 return p
def frame_scores(boxes,ids,m,truth,span,manual=False):
 if not manual:
  from stvg.metrics import DenseTube
  z=DenseTube(boxes,dict(frame_ids=ids,input=m),truth,span,clip=m['_dataset']=='hc2');return z.fids,z.iou
 from stvg.metrics_base import xyxy
 b=xyxy(boxes,m['width'],m['height']);b=np.maximum(b,0)if m['_dataset']=='hc2'else b;fids=np.asarray(sorted(truth));coords=np.asarray([np.interp(fids,ids,b[:,k])for k in range(4)]).T;gt=np.asarray([truth[int(t)]for t in fids]);width=np.maximum(0,np.minimum(coords[:,2],gt[:,2])-np.maximum(coords[:,0],gt[:,0]));height=np.maximum(0,np.minimum(coords[:,3],gt[:,3])-np.maximum(coords[:,1],gt[:,1]));inter=width*height;ap=np.maximum(0,coords[:,2]-coords[:,0])*np.maximum(0,coords[:,3]-coords[:,1]);ag=np.maximum(0,gt[:,2]-gt[:,0])*np.maximum(0,gt[:,3]-gt[:,1]);union=ap+ag-inter;iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);iou[(fids<ids[0])|(fids>ids[-1])]=0;return fids,iou
def one(ds,row,g,entry,old,manual=False):
 assert sha(BASE/entry['path'])==entry['sha256']and sha(BASE/entry['receipt_path'])==entry['receipt_sha256'];truth={int(k):v for k,v in g['boxes'].items()};m={**row['metadata'],'_dataset':ds}
 with np.load(BASE/entry['path'])as z:
  ids=z['frame_ids'];pos=z['positions8'];iv=z['Ours_interval'];assert np.array_equal(iv,z['ExpertInterpolation_interval']);obs=ids[pos];fid,a=frame_scores(z['Ours'],ids,m,truth,g['interval'],manual);fid2,b=frame_scores(z['ExpertInterpolation'],ids,m,truth,g['interval'],manual);assert np.array_equal(fid,fid2)
  event=(fid>=max(iv[0],g['interval'][0]))&(fid<min(iv[1],g['interval'][1]));den=max(max(iv[1],g['interval'][1])-min(iv[0],g['interval'][0]),1);observed=np.isin(fid,obs);distance=np.min(np.abs(fid[:,None]-obs[None,:]),axis=1)/m['fps'];bi=np.searchsorted([0,.5,1,2,4,float('inf')],distance,side='right')-1;contribution=[];counts=[];sums=[]
  for j in range(6):
   mask=event&(observed if j==0 else(~observed)&(bi==j-1));dv=float(math.fsum((a[mask]-b[mask]).tolist())*100/den);contribution.append(dv);counts.append(int(mask.sum()));sums.append([float(a[mask].sum()),float(b[mask].sum())])
 expected=100*(old['values']['Ours']['v']-old['values']['ExpertInterpolation']['v']);assert abs(math.fsum(contribution)-expected)<1e-12
 return dict(id=row['id'],parent=old['parent'],contributions=contribution,annotated_frames=counts,IoU_sums=sums,old_delta_v_pp=expected,temporal_denominator=float(den))
def derive(ds):
 check();start=time.perf_counter();seal=read(BASE/'matched_cpu'/ds/'sealed.json');aud=read(BASE/'matched_cpu'/ds/'reporting_receipt.json');assert aud['all_arrays_cold_rederived_exact'];truth=selected.labels(ds);old={r['id']:r for r in read(BASE/'matched_cpu'/ds/'metrics.json')['private_rows']};out=[]
 for i,row in enumerate(selected.records(ds),1):
  resources(start);out.append(one(ds,row,truth[row['id']],seal['files'][row['id']],old[row['id']]))
  if i%500==0:atomic(DEST/ds/'progress.json',dict(stage='derive',completed=i,total=COUNTS[ds],worker=identity()))
 atomic(DEST/ds/'private_rows.json',out);atomic(DEST/ds/'derived_receipt.json',dict(status='derived_exact_sum_to_old_v_delta',queries=len(out),rows_sha256=sha(DEST/ds/'private_rows.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_predictions=0,new_model=0))
def audit(ds):
 check();start=time.perf_counter();out=read(DEST/ds/'private_rows.json');truth=selected.labels(ds);seal=read(BASE/'matched_cpu'/ds/'sealed.json');old={r['id']:r for r in read(BASE/'matched_cpu'/ds/'metrics.json')['private_rows']}
 for row,saved in zip(selected.records(ds),out):
  got=one(ds,row,truth[row['id']],seal['files'][row['id']],old[row['id']],manual=True);assert got['annotated_frames']==saved['annotated_frames']and got['parent']==saved['parent']and got['id']==saved['id'];assert np.allclose(got['contributions'],saved['contributions'],rtol=0,atol=1e-12)and np.allclose(got['IoU_sums'],saved['IoU_sums'],rtol=0,atol=1e-10)
 from statistics_v1 import stats,dump_table
 report={};all_signed=[]
 for j,name in enumerate(NAMES):
  rr=[dict(parent=r['parent'],contribution_pp=r['contributions'][j])for r in out];v=stats(rr,['contribution_pp']);v.update(supported_queries=sum(r['annotated_frames'][j]>0 for r in out),annotated_frames=sum(r['annotated_frames'][j]for r in out));report[name]=v
 oldmean=float(np.mean([r['old_delta_v_pp']for r in out]));assert abs(sum(v['statistics']['contribution_pp']['mean']for v in report.values())-oldmean)<1e-12
 for i,r in enumerate(out,1):all_signed.append(dict(query=f'Q{i:05d}',**{name+'_v_contribution_pp':r['contributions'][j]for j,name in enumerate(NAMES)},total_delta_v_pp=r['old_delta_v_pp']))
 atomic(PUB/ds/'report.json',dict(queries=len(out),parents=len({r['parent']for r in out}),groups=report,exact_old_mean_delta_v_pp=oldmean,all_query_six_bin_sums_match_original_v_delta=True));dump_table(PUB/ds/'all_query_contributions.tsv',all_signed);text=f'# Exact matched-time spatial vIoU attribution: {ds}\n\n{len(out)} original complete audited query paths. Independent audit recomputed pixel interpolation/IoU with a separate coordinate formula. Each six-bin sum reproduces original scored Ours-minus-interpolation vIoU within1e-12pp. No new predictions/model/Source/expert. Offline annotations only.\n\n|Distance/support|Queries with support|Annotated dense frames inside prediction and GT event|Contribution to whole mean vIoU pp [95% CI]|\n|---|---:|---:|---|\n'
 for name,v in report.items():
  z=v['statistics']['contribution_pp'];text+=f"|{name}|{v['supported_queries']}|{v['annotated_frames']}|{z['mean']:+.6f} [{z['CI95'][0]:+.6f}, {z['CI95'][1]:+.6f}]|\n"
 text+=f'\nSum = {oldmean:+.9f}pp, the original full matched-time spatial comparison. Each bin contribution retains ALL queries and zero-support queries, unlike the conditional per-bin IoU table. Confidence intervals are parent10000pointwise intervals; do not sum CI endpoints. Dense GT annotations are real; only predicted boxes are interpolated as required by the official metric. Observed means exactly one of the8physical expert IDs, not a radius. This is an arithmetic decomposition of existing performance, not a causal intervention or evidence of action understanding. Full target data exposed, no model/default changes.\n';(PUB/ds/'ANALYSIS.md').write_text(text);receipt=dict(status='independent_dense_formula_and_exact_old_v_sum_verified_root_exported',queries=len(out),rows_sha256=sha(DEST/ds/'private_rows.json'),report_sha256=sha(PUB/ds/'report.json'),analysis_sha256=sha(PUB/ds/'ANALYSIS.md'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_predictions=0,new_Source=0,new_expert=0,new_model=0);atomic(PUB/ds/'audit_root_receipt.json',receipt);print(ds,'exact dense attribution passed',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','derive','audit']);ap.add_argument('--dataset',choices=COUNTS);x=ap.parse_args();globals()[x.stage](*([]if x.stage=='prepare'else[x.dataset]))
