"""Read-only saved-receipt/trace summary, with no model, DP, or GT calls."""
from pathlib import Path
import hashlib,json,os,time
from collections import Counter
import torch
ROOT=Path(os.environ.get('STVG_PROJECT_ROOT',Path.cwd())).resolve();TR=ROOT/'outputs/tube_temporal_hc64';OUT=ROOT/'results/tube_temporal_hc64/structure.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def run():
 torch.set_num_threads(1);begin=time.perf_counter();lock=read(TR/'lock.json');seal=read(TR/'sealed.json')
 assert lock['GT_read']is False and seal['GT_read']is False and seal['queries']==64
 assert seal['fingerprint']==lock['fingerprint']=='99b5f8f873b2accb32c0fa46a9e91a9f4e1190d77265023b33c0e4fb25315c9c'
 assert seal['lock_sha256']==sha(TR/'lock.json')and set(seal['files'])==set(lock['ids'])and len(lock['ids'])==64
 for p,d in lock['sources'].items():assert sha(ROOT/p)==d and sha(TR/'source_snapshot'/p)==d
 formal=sha(ROOT/'outputs/formal/state.json');default=sha(ROOT/'configs/default.json')
 assert formal==lock['formal_state_sha256']and default==lock['registered_config_sha256']
 counts=Counter();native_counts=Counter();extra=Counter();layers=Counter();mass_errors=[];text_errors=[];padding_errors=[];counts_by_T=Counter();r_values=[];padding_per_query=[];private=[]
 for item in lock['inputs']:
  key=item['id'];entry=seal['files'][key];rp=TR/entry['receipt_path'];r=read(rp)
  assert sha(rp)==entry['receipt_sha256']and r['GT_read']is False and r['fingerprint']==lock['fingerprint']
  assert r['prediction_path']==entry['path']and r['prediction_sha256']==entry['sha256']
  for stem in ['prediction','trace','capture']:
   path=(TR/r[stem+'_path']).resolve();assert path.is_relative_to(TR.resolve())and sha(path)==r[stem+'_sha256']
  assert all(r[k]for k in ['Source_exact','E_exact','positions_exact','A_reference_exact'])
  assert r['cache_sha256']==item['cache_sha256']and r['mask_sha256']==item['mask_sha256']and r['reference_sha256']==item['reference_sha256']
  assert r['Source448_calls']==r['expert_query_forwards']==r['actual_updates']==0
  assert r['extra_decoder_calls']=={'B_Native224':0,'C_TubeReplay':2,'D_Permuted':2}
  cc=r['capture_cost'];assert cc['Source224_offsets']==2 and cc['native_time_decoder_calls']==cc['native_position_decoder_calls']==8
  assert cc['Source448_calls']==cc['ReferDINO_query_forwards']==0 and cc['ReferDINO_cache_hit']is True and cc['expert_instances']==1
  native_counts.update({k:cc[k]for k in ['Source224_offsets','native_time_decoder_calls','native_position_decoder_calls','Source448_calls','ReferDINO_query_forwards']})
  native_counts['expert_cached_queries']+=int(cc['ReferDINO_cache_hit'])
  t=torch.load(TR/r['trace_path'],map_location='cpu',weights_only=False)
  assert set(t)=={'B_Native224','C_TubeReplay','D_Permuted'}
  B,C,D=t['B_Native224'],t['C_TubeReplay'],t['D_Permuted'];assert B['auxiliary']is None and B['temporal_audit']is None
  T=len(C['auxiliary']['final']);counts_by_T[T]+=1
  for k in ['final','actions','reliability','frame_ids']:assert torch.equal(C['auxiliary'][k],D['auxiliary'][k])
  assert C['auxiliary']['score']==D['auxiliary']['score']
  expected=dict(observed=8,unobserved=T-8,event_edges=T-1,all_nodes=T)
  assert C['auxiliary']['mode']==D['auxiliary']['mode']=='full_timeline'
  assert C['auxiliary']['score_counts']==D['auxiliary']['score_counts']==expected
  counts['queries_same_auxiliary_boxes_actions_score']+=1;counts['auxiliary_nodes']+=T;counts['auxiliary_edges']+=T-1
  query_pad=0
  for a,b in zip(C['temporal_audit']['priors'],D['temporal_audit']['priors']):
   assert a['frame_ids']==b['frame_ids']and a['offset']==b['offset']
   assert torch.equal(a['reliability'],b['reliability'])and torch.equal(a['legal_visual'],b['legal_visual'])
   M,N,legal=a['M'],b['M'],a['legal_visual'];rval=a['reliability']
   assert M.shape==N.shape==legal.shape and rval.shape==(len(M),)
   assert bool(torch.isfinite(M).all())and bool(torch.isfinite(N).all())and bool(torch.isfinite(rval).all())
   assert bool(((M>=0)&(M<=1)).all())and bool(((N>=0)&(N<=1)).all())and bool(((rval>=0)&(rval<=1)).all())
   assert not bool(M[~legal].any())and not bool(N[~legal].any())
   assert a['metadata']['context_scale']==b['metadata']['context_scale']==1.5
   assert a['metadata']['permuted']is False and b['metadata']['permuted']is True
   assert a['metadata']['legal_counts']==b['metadata']['legal_counts']==legal.sum(-1).tolist()
   for ma,mb,l in zip(M,N,legal):
    va,vb=ma[l],mb[l];n=len(va);shift=max(1,n//2)if n else 0
    assert torch.equal(va.sort().values,vb.sort().values)
    assert torch.equal(vb,va.roll(shift))if n else True
    counts['M_histogram_matched_nodes']+=1;counts['M_permutation_changed_nodes']+=int(not torch.equal(va,vb));counts['M_visual_cells']+=len(l)
   query_pad+=int((~legal).sum());counts['legal_visual_cells']+=int(legal.sum());r_values.extend(rval.tolist())
  padding_per_query.append(query_pad);counts['padding_visual_cells']+=query_pad;counts['queries_same_reliability_and_time_support']+=1
  for method,perm in [('B_Native224',False),('C_TubeReplay',False),('D_Permuted',True)]:
   out=t[method];assert out['GT_read']is False and out['cost']==r['costs'][method]and out['cost']['actual_updates']==0
   assert out['spatial']['mode']=='interval'and out['spatial']['costs']['updates']==out['spatial']['costs']['backward']==out['spatial']['costs']['decoder_calls']==0
   extra[method]+=out['cost']['extra_decoder_calls']
   if method=='B_Native224':assert out['cost']['extra_decoder_calls']==0;continue
   audit=out['temporal_audit'];meta=audit['metadata'];cost=audit['cost']
   assert meta['alpha']==.5 and meta['permuted']==perm and meta['GT_read']is False and meta['resolution']==224
   assert cost['offset_time_decoder_calls']==[1,1]and cost['extra_time_decoder_calls']==2
   assert all(cost[k]==0 for k in ['extra_encoder_calls','extra_448_calls','extra_expert_calls','spatial_decoder_calls','optimizer_steps'])
   assert len(audit['attention_audit'])==12
   for z in audit['attention_audit']:
    assert z['text_max_error']==z['padding_max_error']==0 and z['visual_mass_max_error']<2e-6
    mass_errors.append(z['visual_mass_max_error']);text_errors.append(z['text_max_error']);padding_errors.append(z['padding_max_error']);layers[(method,z['heads'],z['queries'])]+=1
  counts['queries']+=1;private.append(dict(id=key,receipt_sha256=sha(rp),trace_sha256=r['trace_sha256'],capture_sha256=r['capture_sha256'],prediction_sha256=r['prediction_sha256']))
  del t
 assert counts['queries']==64
 for p,d in lock['sources'].items():assert sha(ROOT/p)==d
 assert sha(ROOT/'outputs/formal/state.json')==formal and sha(ROOT/'configs/default.json')==default
 rv=torch.tensor(r_values,dtype=torch.float64)
 public=dict(schema='tube_temporal_saved_structure_audit_v1',status='all64_saved_receipts_and_traces_passed',queries=64,GT_read=False,
  fingerprint=lock['fingerprint'],lock_sha256=sha(TR/'lock.json'),seal_sha256=sha(TR/'sealed.json'),
  scope='Read-only hashes and saved tensors; no new model/encoder/decoder/expert/RGB/DP path/candidate/GT scoring',
  identity=dict(source_snapshot_current_hashes_checked=len(lock['sources']),Source_E_positions_A_reference_exact_queries=64,formal_paused_state_hash_unchanged=True,registered_config_hash_unchanged=True),
  auxiliary=dict(same_C_D_boxes_actions_score_queries=64,same_C_D_reliability_physical_support_queries=64,full_timeline_queries=64,physical_node_count_distribution={str(k):v for k,v in sorted(counts_by_T.items())},observed_nodes=64*8,unobserved_nodes=counts['auxiliary_nodes']-64*8,total_nodes=counts['auxiliary_nodes'],all_adjacent_edges=counts['auxiliary_edges'],Strong_interval_used=False,
   reliability=dict(count=len(rv),mean=float(rv.mean()),min=float(rv.min()),max=float(rv.max()),zero_nodes=int((rv==0).sum()),positive_nodes=int((rv>0).sum()),not_event_probability=True),
   evidence='Saved full_timeline mode and explicit all-node/edge denominators, plus prior source review; this audit does not recompute geometry or a new DP path'),
  permutation=dict(legal_M_histogram_bit_exact_nodes=counts['M_histogram_matched_nodes'],M_changed_nodes=counts['M_permutation_changed_nodes'],same_reliability_and_time=True,context_scale=1.5,fixed_legal_cell_half_cycle=True,legal_visual_cells=counts['legal_visual_cells'],padding_visual_cells=counts['padding_visual_cells'],queries_with_nonempty_padding=sum(x>0 for x in padding_per_query),queries_without_padding=sum(x==0 for x in padding_per_query),padding_M_zero=True),
  attention=dict(actual_layer_records=len(mass_errors),records_per_C_query=12,records_per_D_query=12,heads=sorted({x[1]for x in layers}),visual_mass_max_abs_error=max(mass_errors),visual_mass_tolerance=2e-6,text_max_abs_error=max(text_errors),padding_max_abs_error=max(padding_errors),all_records_passed=True,
   nonempty_padding_exercised=any(x>0 for x in padding_per_query),padding_limitation=None if any(x>0 for x in padding_per_query)else'All64 actual visual grids have no padding cells; zero-padding error does not exercise nonempty padding in this dataset'),
  calls=dict(shared_native_totals=dict(native_counts),extra_time_decoder_totals=dict(extra),extra_per_offset_per_C_D_query=[1,1],new_Source448_forwards=0,new_expert_forwards=0,optimizer_updates=0,decoder_backwards=0,A_reference='historical sealed BoxFixed+Strong reused; no new A model calls',cost_boundary='Shared native224 includes actual original-route time/position decoder calls; extra replay counts are reported separately; historical expert/Strong computation is not reclassified as current work'),
  interpretation='Engineering identity/budget/matching audit only; no task-accuracy or reward-quality conclusion')
 save(OUT,public)
 private_receipt=dict(status=public['status'],public_path=str(OUT.relative_to(ROOT)),public_sha256=sha(OUT),audit_script_sha256=sha(__file__),elapsed_seconds=time.perf_counter()-begin,GT_read=False,new_forwards=0,new_DP_paths=0,new_candidates=0,source_fingerprint=lock['fingerprint'],files=private)
 save(TR/'monitor'/'all64_structure.json',private_receipt)
 print(json.dumps({'status':public['status'],'output':str(OUT.relative_to(ROOT)),'sha256':sha(OUT),'attention':public['attention'],'permutation':public['permutation'],'calls':public['calls']},indent=2))
if __name__=='__main__':run()
