"""Complete independent source/state/objective/Adam/selection/noise/frame/dense/bootstrap readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,sys,time
from scripts.stvg_opd_real_forward_task_common_v1 import *

def full_readback(qualification=False):
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();verify()
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old()
 from scripts.run_stvg_opd_real_forward_task_v1 import clean_fit,science_geometry
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare
 from vg_tta.stvg_opd_real_forward_task_v1 import objective_audit,math_audit
 from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
 from scripts.run_decota_paper_main_v1 import read_row,unpack_expert
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 phase='qualification' if qualification else 'formal';bar=read(BASE/('QUALIFICATION_BARRIER.json' if qualification else 'GLOBAL_PREDICTION_BARRIER.json'))
 assert bar['actual_complete_GPU_fits']==(32 if qualification else 256) and not bar['GT_read'];counts=collections.Counter();comp=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
 rr=read(BASE/'DESIGN_LOCK.json')['records'];qi=read(BASE/'DESIGN_LOCK.json')['qualification_indices']
 for r in rr:
  if qualification and r['index'] not in qi:continue
  pair=prior.old.pair_for(r);initial=prior.old.states(pair)['Inherited_order1'];inp=normalized(load(ROOT/r['originals'][0]['input_path']));ex=unpack_expert(inp['expert']);raw=read_row(r['dataset'],r['query_ordinal']);traces=[]
  for repeat in ([0,1] if qualification else [0]):
   for arm in ARMS:
    p=BASE/phase/f"{r['index']:05}"/(arm+f'_repeat{repeat}.pt');h=sha(p);assert h==bar['files'][str(p.relative_to(ROOT))];z=load(p);f=z['fit']
    assert z['runtime_sha256']==sha(BASE/'RUNTIME.json') and not z['GT_read'] and not f['GT_used']
    assert prior.old.digest(initial)==prior.old.digest(f['initial']) and not torch.count_nonzero(f['initial'][prior.old.Q])
    assert z['source_model_sha256']==r['source_model_sha256']==inp['source_model_state_sha256'] and z['pixel_sha256']==r['pixel_sha256']==inp['pixel_sha256']
    assert z['frame_ids']==raw['frame_ids']==inp['frame_ids'] and z['Native_interval']==inp['interval'] and z['evidence_sha256']==prior.old.digest(inp['expert'])
    assert f['config']==r['config'] and f['policy_arm']==arm and z['Adam_reset'] and z['no_cross_query_state_writeback']
    assert objective_audit(f,ex)==z['audit']['objective'];m=math_audit(f);assert m==z['audit']['Adam'];counts['Adam_scalar_checks']+=m['Adam']['scalar_checks'];counts['policy_scalar_checks']+=m['policy_scalar_checks']
    n,c=prior.old.noise(f,raw['key']);assert n==z['audit']['noise'];traces.append(n);counts['sample_coordinates']+=c
    if arm=='A':equal(clean_fit(f),bare(load(ROOT/r['A_path'])['fit']),comp);counts['exact_original_A_fits']+=1
    if repeat or (not qualification and r['index'] in qi):
     q=load(BASE/('qualification' if not qualification else phase)/f"{r['index']:05}"/(arm+'_repeat0.pt'));equal(clean_fit(f),clean_fit(q['fit']),comp);equal(science_geometry(f['geometry']),science_geometry(q['fit']['geometry']),comp);counts['whole_repeat_or_qualified_formal']+=1
    counts['fits']+=1;counts['rounds']+=len(f['rounds']);counts['empty_fits']+=f['empty'];counts['backwards']+=f['gradient_calls'];assert sha(p)==h
   assert all(tr==traces[0] for tr in traces)
  status(BASE/'ROOT_STATUS.json',dict(status='complete_actual_qualification_readback' if qualification else 'complete_actual_source_math_policy_readback',done=counts['fits'],total=32 if qualification else 256,CPU_only=True,GT_read=False,time=time.time()))
 assert counts['fits']==(32 if qualification else 256)
 return dict(counts),comp

def root(qualification=False):
 c,comp=full_readback(qualification)
 if qualification:
  write(BASE/'ROOT_QUALIFICATION.json',dict(status='pass',scope='32 complete fits/eight captures/original A reproduction/whole repeats/actual native head VJP/independent objective action Adam policy/noise/source; not task benefit or old coverage gate',counts=c,full_comparisons=comp,CPU_only=True,GT_read=False,time=time.time()));return
 from scripts.score_stvg_opd_real_forward_task_v1 import assemble,rows_read,summary
 from scripts.audit_stvg_opd_LN_state_intervention_v1 import near,independent_estimate
 rr,ff,rd,pr=assemble(True);scalars=near(rr,rows_read(OUT/'ROWS.jsonl.gz'))+near(ff,rows_read(OUT/'FRAME_ROWS.jsonl.gz'))+near(rd,rows_read(OUT/'ROUND_ROWS.jsonl.gz'))+near(summary(rr,independent_estimate),read(OUT/'SUMMARY.json'))
 write(BASE/'ROOT_READBACK.json',dict(status='pass',scope='all256 complete live trajectories, original64 A, source/state/input/noise/native likelihood/action/Adam/proposal selection/application/complete dense and real GT frames/parent bootstrap; not an independent complete decoder Jacobian',counts=c,full_comparisons=comp,independent_scalar_checks=scalars,GT_provenance=pr,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json',dict(status='pass',parents=64,fits=256,frame_rows=len(ff),round_rows=len(rd),scalar_checks=scalars,positive_negative_neutral_unknown_complete=True,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'MECHANISM_DECISION.json',dict(status='bounded_complete_task_evaluated_pending_independent_replication',scope='current fixed historical recipe only; all effect/control/protection/correction uncertainties reported; no universal route rejection or automatic promotion',production_promoted=False,main_protocol_complete=False,automatic_further_experiments=False,monitor_recreated=False,time=time.time()))

def public(folder):
 from scripts.score_stvg_opd_real_forward_task_v1 import rows_read,summary
 from scripts.audit_stvg_opd_LN_state_intervention_v1 import near,independent_estimate
 rr=rows_read(folder/'ROWS.jsonl.gz');s=read(folder/'SUMMARY.json');n=near(summary(rr,independent_estimate),s);ff=rows_read(folder/'FRAME_ROWS.jsonl.gz');rd=rows_read(folder/'ROUND_ROWS.jsonl.gz')
 assert len(rr)==64 and len(rd)==6400 and all(a in ARMS for a in [r['arm'] for r in ff+rd])
 for r in ff:assert abs(r['after']-r['before']-r['delta'])<1e-12;n+=1
 for r in rr:
  for a in ARMS:
   subset=[x for x in ff if x['parent_id']==r['parent_id'] and x['arm']==a]
   for name,pred in [('observed',lambda x:x['visited']),('own_U',lambda x:x['unvisited']),('correct_U',lambda x:x['unvisited'] and x['initially_correct']),('wrong_U',lambda x:x['unvisited'] and not x['initially_correct'])]:
    part=[x for x in subset if pred(x)];assert len(part)==r[a+'_'+name+'_frames']
    for f in ['before','after','delta']:
     val=sum(x[f] for x in part)/len(part) if part else None;n+=near(val,r[a+'_'+name+'_'+f])
 return dict(status='pass',scope='all64 anonymous parents/256 complete fits/6400 rounds/all actual frame scalar arithmetic and independent parent bootstrap, all signs and unknowns',scalar_checks=n,parent_rows=64,frame_rows=len(ff),round_rows=len(rd),CPU_only=True,GT_read=False)
if __name__=='__main__':root(len(sys.argv)>1 and sys.argv[1]=='qualification')
