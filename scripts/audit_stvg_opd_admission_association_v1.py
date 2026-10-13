"""Independent fixed support/source/noise/objective/Adam and postseal statistics reader."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,sys,time
from scripts.stvg_opd_admission_association_common_v1 import *

def full_readback(phase):
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();verify()
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old()
 from scripts.run_stvg_opd_admission_association_v1 import clean_fit
 from scripts.review_stvg_opd_admission_association_independent_v1 import selector,feedback
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare
 from vg_tta.stvg_opd_admission_association_v1 import objective_audit,math_audit,noise_audit,select,pools_from_expert
 from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
 from scripts.run_decota_paper_main_v1 import read_row,unpack_expert
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 qual=phase=='qualification';oracle=phase=='oracle';bar=read(BASE/('QUALIFICATION_BARRIER.json' if qual else 'ORACLE_PREDICTION_BARRIER.json' if oracle else 'GLOBAL_PREDICTION_BARRIER.json'))
 expected=60 if qual else 64 if oracle else 384;assert bar['logical_complete_fits']==expected and bar['GT_read']==oracle
 for n,h in bar['files'].items():assert sha(ROOT/n)==h
 counts=collections.Counter();comp=dict(tensors=0,tensor_coordinates=0,scalar_values=0);seen=set();d=read(BASE/'DESIGN_LOCK.json')
 for r in d['records']:
  if qual and r['index'] not in d['qualification_indices']:continue
  pair=prior.old.pair_for(r);initial=prior.old.states(pair)['Inherited_order1'];inp=normalized(load(ROOT/r['originals'][0]['input_path']));ex=unpack_expert(inp['expert']);raw=read_row(r['dataset'],r['query_ordinal'])
  actual=select(pools_from_expert(ex,inp['frame_ids']),raw['key']);assert prior.old.digest(actual)==prior.old.digest(read(ROOT/r['selection_path']));independent=selector(actual,raw['key']);counts['independent_enumerated_paths']+=independent['checks']['enumerated_paths']
  assert r['visited']==inp['expert']['actual_observation_positions'];hpos=[q['position'] for q in actual['arms']['H_Top1']['rows']]
  for repeat in ([0,1] if qual else [0]):
   for arm in (['Oracle'] if oracle else ARMS):
    z=resolve(r['index'],arm,phase,repeat);f=z['fit'];fixed=z['fixed_evidence'];p=ROOT/z['canonical_payload_path'];h=z['canonical_payload_sha256']
    assert h==bar['files'][str(p.relative_to(ROOT))] and z['runtime_sha256']==sha(BASE/'RUNTIME.json')
    assert z['logical_GT_assisted']==oracle and f['GT_used']==fixed['GT_assisted'] and (oracle or not f['GT_used'])
    if not oracle:assert evidence_digest(z['selection'])==evidence_digest(actual['arms'][arm])
    assert prior.old.digest(initial)==prior.old.digest(f['initial']) and not torch.count_nonzero(f['initial'][prior.old.Q])
    assert z['source_model_sha256']==r['source_model_sha256']==inp['source_model_state_sha256'] and z['pixel_sha256']==r['pixel_sha256']==inp['pixel_sha256']
    assert z['frame_ids']==raw['frame_ids']==inp['frame_ids'] and z['Native_interval']==inp['interval'] and z['intervals']==r['intervals'] and z['evidence_sha256']==prior.old.digest(inp['expert'])
    assert f['config']==r['config'] and z['Adam_reset'] and z['no_cross_query_state_writeback']
    if str(p) not in seen:
     seen.add(str(p));fb=feedback(dict(f,fixed_evidence=fixed),ex);counts['independent_feedback_rounds']+=fb['rounds'];counts['uniform_observation_rounds']+=fb['uniform_observation_rounds'];assert objective_audit(f,ex,fixed)==z['audit']['objective'];m=math_audit(f);assert m==z['audit']['Adam'];counts['Adam_scalar_checks']+=m['scalar_checks']
     n=noise_audit(f,raw['key'],hpos,z['frame_ids']);assert n==z['audit']['noise'];counts['sample_coordinates']+=n['sample_coordinates']
     counts['physical_payloads_read']+=1
    if arm=='H_Top1':equal(clean_fit(f),bare(load(ROOT/r['A_path'])['fit']),comp);counts['exact_original_H_fits']+=1
    if repeat or (phase=='formal' and r['index'] in d['qualification_indices']):
     q=resolve(r['index'],arm,'qualification',0);equal(clean_fit(f),clean_fit(q['fit']),comp);equal(f['geometry'],q['fit']['geometry'],comp);counts['whole_repeat_or_qualified_formal']+=1
    counts['logical_fits']+=1;counts['logical_rounds']+=len(f['rounds']);counts['logical_empty_fits']+=f['empty'];counts['logical_backwards']+=f['gradient_calls'];assert sha(p)==h
  status(BASE/'ROOT_STATUS.json',dict(status='actual_'+phase+'_source_math_readback',done=counts['logical_fits'],total=expected,CPU_only=True,GT_read=oracle,time=time.time()))
 assert counts['logical_fits']==expected
 return dict(counts),comp

def root(qualification=False):
 if qualification:
  c,comp=full_readback('qualification');write(BASE/'ROOT_QUALIFICATION.json',dict(status='pass',scope='60 logical complete fits, actual canonical payloads, original H whole fits and repeat, native head VJP/objective/Adam/source/noise; technical implementation only',counts=c,full_comparisons=comp,CPU_only=True,GT_read=False,time=time.time()));return
 c,comp=full_readback('formal');o,oc=full_readback('oracle')
 from scripts.score_stvg_opd_admission_association_v1 import assemble,rows_read,summary
 from scripts.audit_stvg_opd_LN_state_intervention_v1 import near,independent_estimate
 rr,ff,rd,ee,pr=assemble(True)
 scalars=near(rr,rows_read(OUT/'ROWS.jsonl.gz'))+near(ff,rows_read(OUT/'FRAME_ROWS.jsonl.gz'))+near(rd,rows_read(OUT/'ROUND_ROWS.jsonl.gz'))+near(ee,rows_read(OUT/'EVIDENCE_ROWS.jsonl.gz'))
 scalars+=near(summary(rr,independent_estimate),read(OUT/'SUMMARY.json'))
 write(BASE/'ROOT_READBACK.json',dict(status='pass',scope='all label-free six-arm fits and separately tagged GT-assisted original-pool Oracle; source/state/pools/fixed teachers/noise/action/likelihood/Adam/dense/frame/parent bootstrap; no independent full decoder Jacobian',counts=c,Oracle_counts=o,full_comparisons=comp,Oracle_comparisons=oc,independent_scalar_checks=scalars,GT_provenance=pr,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json',dict(status='pass',parents=64,logical_fits=448,frame_rows=len(ff),round_rows=len(rd),evidence_rows=len(ee),scalar_checks=scalars,positive_negative_neutral_unknown_complete=True,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'MECHANISM_DECISION.json',dict(status='bounded_admission_association_task_evaluated',scope='fixed historically exposed panel and current candidate pool/geometry/OPD only; Oracle GT-assisted diagnostic; no identity evidence, universal route rejection or automatic promotion',production_promoted=False,main_protocol_complete=False,automatic_further_experiments=False,monitor_recreated=False,time=time.time()))

def public(folder):
 from scripts.score_stvg_opd_admission_association_v1 import rows_read,summary
 from scripts.audit_stvg_opd_LN_state_intervention_v1 import near,independent_estimate
 rr=rows_read(folder/'ROWS.jsonl.gz');assert len(rr)==64;n=near(summary(rr,independent_estimate),read(folder/'SUMMARY.json'))
 return dict(status='pass',scope='all64 anonymized parent outcome rows and independent paired parent bootstrap; all six arms and separately tagged Oracle signs/unknowns',scalar_checks=n,parent_rows=64,CPU_only=True,GT_read=False)
if __name__=='__main__':root(len(sys.argv)>1 and sys.argv[1]=='qualification')
