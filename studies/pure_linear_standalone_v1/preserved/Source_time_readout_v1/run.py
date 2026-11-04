from common import *
import argparse,ast,csv

def prepare():
 assert not(D/'plan.json').exists();torch.set_num_threads(1);cases=[]
 # No access to actual Source/Full intervals in prediction or this preflight.
 for ds in COUNTS:
  for row in rows(ds)[:2]:
   a,p=inputs(ds,row);b,q=inputs(ds,row,reference=True);assert p==q and np.max(abs(a['boxes_xyxy']-b['boxes_xyxy']))<2e-10
   ix=np.searchsorted(a['frame_ids'],a['anchor_times']);assert np.array_equal(a['boxes_xyxy'][ix],a['anchor_xyxy']);assert np.array_equal(a['boxes_xyxy'][a['frame_ids']<a['anchor_times'][0]],np.repeat(a['anchor_xyxy'][:1],sum(a['frame_ids']<a['anchor_times'][0]),axis=0));assert np.array_equal(a['boxes_xyxy'][a['frame_ids']>a['anchor_times'][-1]],np.repeat(a['anchor_xyxy'][-1:],sum(a['frame_ids']>a['anchor_times'][-1]),axis=0));cases.append(dict(dataset=ds,queries=1,old_uncovered_video_frames=p['old_uncovered_video_frames'],all8exact=True,both_tails_complete=True,independent_bracket_interpolation_exact_within_roundoff=True))
 tree=ast.parse((S/'pure.py').read_text());assert {n.name for n in tree.body if isinstance(n,ast.FunctionDef)}=={'interpolate'};assert all(not isinstance(n,ast.ImportFrom)for n in tree.body)
 atomic(D/'preflight.json',dict(cases=cases,new_model=0,new_Source=0,new_expert=0,new_GT=0))
 files=[CHECK/'teacher_inventory.json',RUN/'plan.json']
 for ds in COUNTS:
  j=fixed(RUN/'plan.json')['jobs'][ds];files += [PROVIDER/j['manifest'],PROVIDER/j['labels'],RUN/ds/'sealed.json',CHECK/ds/'sealed.json']
 p=dict(schema='standalone_raw8_linear_expert_interpolation_v1',authorization='User reports baseline mixed proposed-method ideas; isolate pure raw8corner interpolation and fix missing endpoint video coverage. No current method changes.',counts=COUNTS,one_spatial_algorithm='Only np.interp eight raw expert pixelxyxy boxes over physical anchor times, full video input segment every integer frame, nearest endpoint tails, exact raw anchors. No logsize, E, Source boxes, appearance, mask, prior, posterior, geometry, candidates, transport orDP.',prediction_intervals_read=False,time_protocol='Intervals supplied ONLY after whole-domain prediction seal by scoring. Independent baseline: unchanged original native Source interval. Separate fixed-time spatial diagnostic: unchanged Full interval; NEVER call this independent expert temporal prediction. Same generated expert-only trajectory in both evaluations.',corrected_boundary='Old PureLinearFixed produced only Source native sampling IDs; official linear_interp does not extrapolate outside them. New pure estimator covers full metadata input segment, independently of Source sampling grid. HC [0,frame_count); Vid official [start_frame,end_frame). Negative-coordinate HC clipping remains original benchmark evaluator policy and is applied after full-frame interpolation.',same_old8expert_observations=True,new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_optimizer=0,source_or_method_or_paper_modified=False,whole_domain_seal_separate_officialCPUscore_separate_coldarray_inputSHA_means_audit=True,stats='all queries retained, null support notzero, queryequal10kpairedparentpointwiseCI/exposedcohorts/no tuning; allsigned severe/rescue/shortlong',code_sha256={str(f.relative_to(R)):sha(f)for f in list(S.glob('*.py'))+[R/'stvg/metrics.py',R/'stvg/metrics_base.py',R/'stvg/metrics_hc.py']},input_sha256={str(f):sha(f)for f in files});p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'});atomic(P/'boundary_preflight.json',dict(cases=cases));print(p['fingerprint'],flush=True)

def predict(ds):
 p=validate();os.environ['PREDICTION_ONLY']='1';sys.addaudithook(guard);torch.set_num_threads(1);start=time.perf_counter();files={};dest=D/ds;assert not(dest/'sealed.json').exists()
 for i,row in enumerate(rows(ds),1):
  resources(start);a,proof=inputs(ds,row);path=dest/'predictions'/(row['id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
  with path.open('wb')as f:np.savez_compressed(f,**a)
  with np.load(path)as z:assert all(np.array_equal(v,z[k])for k,v in a.items())
  rec=dict(path=str(path.relative_to(D)),sha256=sha(path),proof=proof,worker=identity(),GT_read=False);rp=dest/'receipts'/(row['id']+'.json');atomic(rp,rec);files[row['id']]={**rec,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}
  if i%100==0:atomic(dest/'progress.json',dict(completed=i,total=COUNTS[ds],worker=identity(),GT_read=False))
 atomic(dest/'sealed.json',dict(files=files,queries=len(files),GT_read=False,no_temporal_interval_in_prediction=True,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,fingerprint=p['fingerprint']));print(ds,'sealed',len(files),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','predict']);a.add_argument('--dataset',choices=COUNTS);x=a.parse_args();globals()[x.stage](*([]if x.stage=='prepare'else[x.dataset]))
