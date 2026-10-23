"""Final CG acquisition accounting from sealed receipts and private cache hashes."""
import hashlib,json
from pathlib import Path
R=Path.cwd();D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1/cgstvg_v9'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 receipts=[];proof={};stages={}
 for stage,n in [('screen',128),('confirm',256)]:
  path=D/('cgstvg_v9_'+stage+'_sealed.json');seal=read(path);score=read(D/('cgstvg_v9_'+stage+'_score_receipt.json'))
  assert score['status']=='scored'and score['seal_sha256']==sha(path)and seal['counts']['queries']==n
  rows=[e for files in seal['files'].values()for e in files.values()];assert len(rows)==n
  for e in rows:assert sha(D/e['receipt_path'])==e['receipt_sha256']
  receipts+=rows;stages[stage]=dict(queries=n,source_forwards_in_bound_receipts=sum(e['source_forwards']for e in rows),recorded_new_expert=sum(e['expert_new_forwards']for e in rows),recorded_teacher_cache_hits=sum(e['expert_cache_hit']for e in rows),Source_seconds=sum(e['source_seconds']for e in rows),pipeline_seconds=sum(e['pipeline_seconds']for e in rows),expert_load_seconds_recorded=sum(e['expert_model_load_seconds']for e in rows))
  proof[stage]=dict(seal_sha256=sha(path),score_sha256=sha(D/('cgstvg_v9_'+stage+'_score_receipt.json')))
 teacher_counts={};keys=set();cache_proof={}
 for path in sorted((D/'cgstvg_teacher_cache').glob('*.json')):
  meta=read(path);assert meta['key']==path.stem and meta['key']not in keys;keys.add(meta['key'])
  assert meta['context']['fingerprint']==read(D/'cgstvg_v9_plan.json')['teacher_runtime_fingerprint']
  assert sha(path.with_suffix('.pt'))==meta['sha256'];ds=meta['context']['dataset'];teacher_counts[ds]=teacher_counts.get(ds,0)+1
  cache_proof[path.name]=dict(metadata_sha256=sha(path),payload_sha256=meta['sha256'])
 assert teacher_counts=={'hc2':192,'vidstg':192}and len(keys)==384
 total_source=sum(x['source_forwards_in_bound_receipts']for x in stages.values());recorded_expert=sum(x['recorded_new_expert']for x in stages.values());assert total_source==395 and recorded_expert==381
 out=dict(status='complete_CG_native_acquisition_accounting',stages=stages,unique_same_ReferDINO_joint8_contexts=384,unique_contexts_by_domain=teacher_counts,new_expert_acquisition_recorded_in_final_receipts=381,prior_paid_exact_cache_acquisitions_restored=3,expert_acquisitions_total_once=384,Source_forward_accounting=dict(ordinary_queries=384,native_noop_extra_references=4,additional_stream_parity_replays=7,bound_receipt_completed_total=395,completed_Source_before_v7_readout_failure=1,earlier_duplicate_v4_completed_references=4,total_completed_including_engineering=400,partial_failed_native_OOM_attempts=4,pre_Source_decode_boundary_failure_calls=0),Source_model_loading_seconds=None,Source_model_loading_note='Not separately timed in original frozen Source loader; unavailable, not zero. Model load/init and video decode are outside reported per-query inference scope.',expert_model_loading_note='Recorded separately above; costs are original acquisitions, no cache-as-deployment-speed claim.',cost_scope='Original expert per-context GPU/pre/post cost remains in each query receipt/report even when cache-hit. Nested expert/Source/CPU/pipeline times not summed; partial-failure full duration unknown. Synthetic primitive memory parity measurements and recovery overhead are engineering evidence, not a full deployment benchmark.',proof=proof,private_cache_manifest_sha256=hashlib.sha256(json.dumps(cache_proof,sort_keys=True).encode()).hexdigest(),all_cached_payload_SHA_verified=True,no_GT_or_model_calls=True)
 (P/'COST_ACCOUNTING.json').write_text(json.dumps(out,indent=2)+'\n');(D/'cgstvg_v9_cache_cost_verification_private.json').write_text(json.dumps(cache_proof,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
