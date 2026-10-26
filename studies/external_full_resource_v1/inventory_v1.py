"""Metadata/storage feasibility only; no labels, model, RGB decoding or release."""
from pathlib import Path
import hashlib,json,math,os,time

R=Path(__file__).resolve().parents[2]
D=R/'outputs/external_full_resource_v1'
P=R/'results/external_full_resource_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def run():
 start=time.perf_counter();plan=read(D/'plan_v1.json')
 for p,h in plan['input_sha256'].items():assert sha(R/p)==h,p
 assert sha(Path(__file__))==plan['code_sha256']
 source=read(R/'outputs/simplepath_full_v2/plan.json');provider=R.parent/'stvg-opd'
 forbidden={(provider/j['labels']).resolve()for j in source['jobs'].values()}
 import sys
 def guard(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path)):
   assert Path(os.fsdecode(args[0])).resolve()not in forbidden,'No GT in resource inventory'
 sys.addaudithook(guard)
 raw=read(R/'outputs/map_paper_completion_v1/raw_reuse_index.json');small=read(R/'outputs/paper_review_suite_v1/selection.json');report={};private={}
 for ds,expected in [('hc2',1902),('vidstg',10303)]:
  j=source['jobs'][ds];mp=provider/j['manifest'];assert sha(mp)==j['manifest_sha256'];rows=[json.loads(s)for s in mp.read_text().splitlines()];assert len(rows)==expected and len({r['id']for r in rows})==expected
  pilot={r['id']for r in small[ds]['confirm']};assert len(pilot)==128 and pilot<={r['id']for r in rows}
  inventory=[];sizes=[];nframes=[];aspect=[];rgbbytes=[];packed=[]
  for row in rows:
   m=row['metadata'];ids=m['frame_ids'];assert ids==sorted(set(ids))and len(ids)>=8
   h,w=m['height'],m['width'];n=len(ids);nframes.append(n);aspect.append(max(h,w)/min(h,w));rgbbytes.append(n*h*w*3);packed.append(n*math.ceil(h*w/8))
   inventory.append(dict(id=row['id'],native_frames=n,height=h,width=w,in_pilot=row['id']in pilot,packed_mask_bytes=n*math.ceil(h*w/8)))
  verified=[]
  for key,e in raw[ds].items():
   p=Path(e['path']);assert p.exists()and sha(p)==e['sha256'];sizes.append(p.stat().st_size);verified.append(dict(id=key,path=str(p),sha256=e['sha256'],bytes=p.stat().st_size))
  assert len(verified)==192 and pilot<={x['id']for x in verified}
  sampled=[r for r in inventory if r['in_pilot']]
  report[ds]=dict(full_queries=expected,completed_ReferFormer_small_queries=128,Source_Edge_small_queries=128,small_Source_Edge_completion_not_claimed=True,native_frames_total=sum(nframes),native_frames_max=max(nframes),native_frames_pilot_total=sum(x['native_frames']for x in sampled),metadata_aspect_ratio_max=max(aspect),largest_decoded_native_RGB_bytes=max(rgbbytes),whole_full_packed_binary_masks_exact_payload_bytes=sum(packed),whole_full_float32_masks_if_all_retained_bytes=4*sum(x['native_frames']*x['height']*x['width']for x in inventory),float_masks_policy='Keep full float outputs for first-two root bridge only; checkpoint every actual float before validation, then only own acknowledged committed float scratch can be disposed. Retain packed masks and SHA/disposal receipts.',retained_raw_candidate_files_SHA_verified=len(verified),retained_raw_candidate_bytes=sum(sizes),raw_candidate_tensor_keys_and_new_expert_repool_eligibility_not_verified_here=True,full_queries_outside_retained192=expected-len(verified),full_queries_outside_completed_ReferFormer128=expected-128,full_queries_outside_Source_Edge128=expected-128,Source_only_Edge_new_Source_required_for_tracking=0,ReferFormer_missing_grid_acquisition_upper_bound=expected-len(verified),ReferFormer_new_expert_upper_bound_remaining=expected-128,model_runtime_unchanged=True)
  private[ds]=dict(manifest_path=str(mp),manifest_sha256=j['manifest_sha256'],all_queries=inventory,verified_retained_raw_candidates=verified)
 put(D/'private_inventory_v1.json',private)
 result=dict(status='full_manifest_metadata_and384_existing_raw_bytes_inventory_only_no_execution_release',domains=report,code_sha256=sha(Path(__file__)),private_inventory_sha256=sha(D/'private_inventory_v1.json'),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_forward=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0,full_execution_release=False,full_execution_plan_frozen=False,paused_CG_gamma_corruption_untouched=True,small_accuracy_not_used_for_resource_selection=True,notes=['Metadata packed-mask bytes are an exact payload count, not total disk usage; PT/ZIP receipts/compacts/features/raw expert outputs add overhead.','192 retained candidate SHA proofs do not establish arbitrary new tensor pooling eligibility; full worker must verify tensor schema, native/old-pool/current-control math and input binding before reuse.','No full latency extrapolation from conditional cached query sums; genuine uncached timing must be measured independently.','Remaining full execution requires both Source-prompt small audits/root publication and complete model/source/checkpoint/first-paid-bridge/resource/cost plan.'])
 put(D/'receipt_v1.json',result);put(P/'metadata_resource_inventory_v1.json',result)
if __name__=='__main__':run()
