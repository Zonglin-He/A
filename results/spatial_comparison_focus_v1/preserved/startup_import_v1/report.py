"""Independent scalar/input audit and root reuse-only accuracy-cost export."""
from run import *
def audit():
 check();settings();old.parent.infer_guard();seal=read(D/'sealed.json');assert seal['measurement_sha256']==sha(D/'measurements.json')and seal['calls']==2304
 m=read(D/'measurements.json');items=m['items'];assert len(items)==2304 and len(m['warmup'])==6;keys={(i['dataset'],i['query'],i['repeat'],i['arm'])for i in items};expected={(ds,q,r,a)for ds in DATASETS for q in range(1,129)for r in range(1,4)for a in ARMS};assert keys==expected and len(keys)==len(items)
 for ds in DATASETS:
  for qi,row in enumerate(rows(ds),1):
   v,refs,b=load(ds,row);ih=input_digest(v);qq=[i for i in items if i['dataset']==ds and i['query']==qi]
   for a in ARMS:
    aa=[i for i in qq if i['arm']==a];assert len(aa)==3 and{j['order_position']for j in aa}=={1,2,3}
    for x in aa:assert np.isfinite(x['elapsed_seconds'])and x['elapsed_ns']>0 and x['elapsed_seconds']==x['elapsed_ns']/1e9 and x['all_output_arrays_exact']and x['input_digest']==ih and x['output_digest']==x['reference_array_digest']==array_digest(refs[a])and x['reference_prediction_sha256']==b['reference_sha256']
 ack=dict(status='independent_all2304_tuples_reference_input_SHA_output_hashes_orders_clocks_verified',measurement_sha256=sha(D/'measurements.json'),seal_sha256=sha(D/'sealed.json'),worker=identity(),no_retiming=True,new_Source=0,new_expert=0,new_GPU=0,new_GT=0);atomic(D/'audit_receipt.json',ack);atomic(P/'audit_receipt.json',ack);print('all2304 spatial clocks/input/output references independently verified',flush=True)
def stats_query(q,seed=20261103):
 n=len(q);assert n==128;w=np.random.default_rng(seed).multinomial(n,[1/n]*n,size=10000);x=np.array([[v[a]for a in ARMS]for v in q],float);boot=(w@x)/n;rep={}
 for j,a in enumerate(ARMS):rep[a]=dict(queries=n,repeats_per_query=3,mean_seconds=float(x[:,j].mean()),CI95_mean_seconds=np.quantile(boot[:,j],[.025,.975]).tolist(),p50_query_mean_seconds=float(np.quantile(x[:,j],.5)),p90_query_mean_seconds=float(np.quantile(x[:,j],.9)))
 paired={}
 for a,b in [('Current','Matching'),('Current','PureExpert'),('Matching','PureExpert')]:
  ia,ib=ARMS.index(a),ARMS.index(b);paired[a+'-'+b]=dict(mean_seconds=float((x[:,ia]-x[:,ib]).mean()),CI95_seconds=np.quantile(boot[:,ia]-boot[:,ib],[.025,.975]).tolist(),ratio_of_means=float(x[:,ia].mean()/x[:,ib].mean()),CI95_ratio_of_means=np.quantile(boot[:,ia]/boot[:,ib],[.025,.975]).tolist())
 return dict(methods=rep,paired=paired)
def export():
 check();assert read(D/'audit_receipt.json')['measurement_sha256']==sha(D/'measurements.json');m=read(D/'measurements.json');allsummary={};public=[];accuracy={};names={'PureExpert':'PureMatched','Matching':'Matching','Current':'Current'}
 for ds in DATASETS:
  q=[]
  for i in range(1,129):
   r=dict(query=i)
   for a in ARMS:r[a]=float(np.mean([j['elapsed_seconds']for j in m['items']if j['dataset']==ds and j['query']==i and j['arm']==a]))
   q.append(r);public.append(dict(dataset=ds,**r))
  allsummary[ds]=stats_query(q);ack=read(PREV/ds/'reporting_receipt.json');folder=old.P/'matching_pilot'/ds;rep=read(folder/'report.json');assert sha(folder/'report.json')==ack['report_sha256'];accuracy[ds]=dict(means={a:rep['means'][b]for a,b in names.items()},paired=rep['paired'],reused_report_sha256=ack['report_sha256'],reused_audit_sha256=sha(PREV/ds/'reporting_receipt.json'))
 atomic(P/'timing_summary.json',dict(summary=allsummary,hardware=m['hardware'],measurement_sha256=sha(D/'measurements.json'),aggregation='per-query average3repeats, then128queryequal/128pairedparent10000bootstrap; pointwise',scope='CPU input-ready spatial functions only, no Source/expert/inference/decoding/IO/time-update cost. Current includes its real E initialization/candidate geometry/finalDP; no observed-anchor method change.',warmup=m['warmup']));atomic(P/'reused_accuracy.json',accuracy)
 with(P/'all_measured_calls.tsv').open('w')as f:
  kk=['dataset','query','repeat','arm','order_position','elapsed_ns','elapsed_seconds','all_output_arrays_exact','output_digest','reference_array_digest','input_digest','reference_prediction_sha256'];w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(kk);w.writerows([[r[k]for k in kk]for r in m['items']])
 with(P/'per_query_three_repeat_means.tsv').open('w')as f:
  kk=['dataset','query',*ARMS];w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(kk);w.writerows([[r[k]for k in kk]for r in public])
 text='''# Fixed128 three-strategy spatial accuracy and actual CPU processing cost

Only this spatial comparison is active under the latest user scope. All earlier completed experiments are reused, not rerun; fulltarget acquisition, end-to-end timing, density and other optional experiments are paused. Latest user explicitly cancelled manuscript work. Same original128query/128distinctparents per direction, same8expert boxes/masks/physicalpositions, same original Source feature layer/resolution/preprocessing and same Current event interval. Source-only Matching parameters remain temperature.05/topK10/context20. No new accuracy predictions/GT scoring/model/Source/expert/GPU/optimizer or Current method change. Real algorithm replay below is required timing, not an accuracy rerun.

The official-kernel/source-domain/positive-control/cold-array checks and both target pilots are already complete. This is a Source-feature VideoWalk module adapter, not the original learned encoder or a strong-baseline claim: it is weaker than pure interpolation on source and target. Current retains its previously audited E/expert soft-candidate behavior at observed frames; Pure and Matching exactly retain raw expert anchors. The comparison controls inputs/time/features; it must not be described as fixing Current anchor outputs or validating an intended hard-anchor implementation. No clamp was silently added.

Timing starts with CPU Source features/boxes, native-grid expert masks/boxes/positions and the common event interval ready, and ends with an owned CPU trajectory. Pure measures interpolation and endpoint holding. Matching measures its actual normalization, two-direction propagation and native-resolution mask-to-box readout. Current measures actual shared E initialization, SourceEvidence construction, complete candidate/score/link rebuilding and final interval DP; no cached E/geometry/path is substituted. Source/ReferDINO/decoding/feature extraction/temporal posterior, experimental reference checks/SHA/saving/resource guards are outside this spatial window. Original intrinsic implementation checks remain part of each function. These clocks are NOT end-to-end deployment latency or component-clock subtraction.

Three actual repeats per query; deterministic forward/reverse cyclic arm order gives each strategy each position once. Torch/BLAS/OpenCV one CPU thread, cores14-15/nice19/idleIO, input/output arrays bitwise equal to old reference on every call, six warmups excluded. Query-internal means precede queryequal128parent10kbootstrap; p90 is the distribution of128query means. Shared-host conditions and low-priority scheduling limit extrapolation to throughput/other hardware. Source/expert inherited cost is nonzero.

|Dataset|Strategy|vIoU %|R.3 %|R.5 %|Unobserved IoU %|Spatial mean ms|95% CI mean ms|p90 query mean ms|
|---|---|---:|---:|---:|---:|---:|---|---:|
'''
 for ds in DATASETS:
  for a in ARMS:
   ac=accuracy[ds]['means'][a];t=allsummary[ds]['methods'][a];text+=f"|{ds}|{a}|{100*ac['v']:.6f}|{100*ac['vIoU_at_03']:.6f}|{100*ac['vIoU_at_05']:.6f}|{100*ac['unobserved_actual_sIoU']:.6f}|{1000*t['mean_seconds']:.6f}|[{1000*t['CI95_mean_seconds'][0]:.6f},{1000*t['CI95_mean_seconds'][1]:.6f}]|{1000*t['p90_query_mean_seconds']:.6f}|\n"
 text+='\n|Dataset|Cost comparison|Mean difference ms [95% CI]|Ratio of means [95% CI]|\n|---|---|---|---|\n'
 for ds in DATASETS:
  for name,t in allsummary[ds]['paired'].items():text+=f"|{ds}|{name}|{1000*t['mean_seconds']:+.6f} [{1000*t['CI95_seconds'][0]:+.6f},{1000*t['CI95_seconds'][1]:+.6f}]|{t['ratio_of_means']:.6f} [{t['CI95_ratio_of_means'][0]:.6f},{t['CI95_ratio_of_means'][1]:.6f}]|\n"
 text+='\nAll per-call costs and three-repeat per-query averages are public scalar TSVs. Existing accuracy confidence intervals/positive-negative/severe/rescues/shortlong/distance reports are reused and linked in the original matching pilot, with no new official scoring. No significance-selected sample/parameter changes, equivalence/unseen-test or universal propagation superiority claim. No manuscript edits, extra strategy, end-to-end benchmark, density or fulltarget Source collection was performed.\n'
 (P/'ANALYSIS.md').write_text(text);atomic(D/'root_export_receipt.json',dict(status='root_reused_accuracy_and_actual_three_arm_spatial_timing_exported',analysis_sha256=sha(P/'ANALYSIS.md'),summary_sha256=sha(P/'timing_summary.json'),audit_sha256=sha(D/'audit_receipt.json'),worker=identity(),code_sha256=sha(Path(__file__)),new_accuracy_scores=0,new_Source=0,new_expert=0,new_GPU=0,Current_changed=False,paper_changed=False));print(json.dumps(allsummary),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','export']);globals()[ap.parse_args().stage]()
