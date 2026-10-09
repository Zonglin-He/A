"""Bounded no-GT root readback of actual accepted original P4 first fits."""
import collections
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,read,write,sha
activate()
from scripts.decota_matrix_common_v1 import load
from scripts.stvg_opd_paper_later_common_v1 import stage_definition,committed
from scripts.run_stvg_opd_p4_joint_precision001 import verify,dispatch_saved,scientific,REC
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal


def run():
    import torch
    from scripts.run_decota_paper_main_v1 import unpack_expert
    torch.set_num_threads(2);verify()
    root_runtime=read(REC/'ROOT_RESUME_RUNTIME.json')
    for f,h in root_runtime['pins'].items():assert sha(ROOT/f)==h,f
    assert read(BASE/'P4_QUALIFICATION.json')['status']=='pass'
    name='P4_hc2';stage=stage_definition(name);condition='clean';order='order1'
    arms=stage['arms']
    files=[]
    for arm in arms:
        for at in range(8):
            p=BASE/'stages'/name/condition/order/arm/f'{at:05}.pt'
            if not p.exists() or not p.with_suffix('.json').exists():break
            files.append((arm,at,p,sha(p),sha(p.with_suffix('.json'))))
    for arm in arms:assert sum(a==arm for a,_,_,_,_ in files)>=2,'First two accepted fits not ready'
    counts=collections.Counter();records=[];inputs={};first=[]
    for arm in arms:
        previous=None;previous_hash=None
        qualified=BASE/'component_qualification'/name/condition/order/arm/'00000.pt'
        source_initial=load(qualified)['fit']['initial']
        for a,at,p,h,rh in files:
            if a!=arm:continue
            rc=read(p.with_suffix('.json'));z=load(p);fit=z['fit'];cfg=stage['variant_configs'][arm]
            assert rc['sha256']==h and rc['bytes']==p.stat().st_size and not rc['GT_read']
            assert z['stage']==name and z['condition']==condition and z['order']==order and z['arm']==arm
            assert z['arrival']==at and z['query_ordinal']==stage['orders'][order][at]
            assert z['dataset']==stage['dataset'] and z['source']==stage['source'] and z['GT_read'] is False
            assert z['config']==fit['config']==cfg and fit['selected_step']==cfg['steps']
            assert fit['active_parameters']=={'query_only':256,'LN_only':1536}.get(arm,1792)
            assert z['previous_payload_sha256']==previous_hash
            assert z['query_reset'] and z['Adam_reset'] and z['Native_WHEN_fixed']
            assert sum(v.numel() for v in fit['initial'].values())==1792
            assert torch.count_nonzero(fit['initial']['spatial.query_residual'])==0
            expected_initial=source_initial if previous is None else previous
            assert all(torch.equal(v,torch.zeros_like(v) if n=='spatial.query_residual' else expected_initial[n]) for n,v in fit['initial'].items())
            written=committed(fit['initial'],fit['state'],cfg['writeback'])
            assert all(torch.equal(v,z['committed'][n]) for n,v in written.items())
            if arm in ['query_only','joint_alpha0']:
                assert all(torch.equal(v,source_initial[n]) for n,v in z['committed'].items() if n!='spatial.query_residual')
            ip=BASE/z['input']['path'];ih=sha(ip);ir=read(ip.with_suffix('.json'));inp=load(ip)
            assert ih==z['input']['sha256']==ir['sha256'] and not ir['GT_read']
            assert inp['GT_read'] is False and z['interval']==inp['interval']
            assert dispatch_saved(fit,unpack_expert(inp['expert']),z)==z['math_audit']
            inputs[str(ip.relative_to(BASE))]=dict(sha256=ih,bytes=ip.stat().st_size)
            if at<2:
                qp=qualified.with_name(p.name);q=load(qp);cr=REC/'first_formal'/name/condition/order/arm/(p.stem+'.json')
                check=read(cr);comparison=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                assert q['config']==cfg and q['input']['sha256']==ih and check['status']=='pass'
                assert check['qualified_fit_sha256']==sha(qp) and check['GT_read'] is False
                equal(scientific(fit),scientific(q['fit']),comparison);assert comparison==check['complete_fit_bitwise']
                first.append(dict(path=str(cr.relative_to(BASE)),sha256=sha(cr),arm=arm,arrival=at,query_ordinal=z['query_ordinal']))
                counts['actual_qualified_formal_pairs_bitwise']+=1
            previous=z['committed'];previous_hash=h
            counts['accepted_formal_predictions']+=1;counts['complete_fit_math_exact']+=1
            counts['actual_rounds']+=len(fit['rounds']);counts['state_coordinates']+=1792
            records.append(dict(stage=name,condition=condition,order=order,arm=arm,arrival=at,query_ordinal=z['query_ordinal'],
                path=str(p.relative_to(BASE)),sha256=h,receipt_sha256=rh,bytes=p.stat().st_size,input_sha256=ih,
                active_parameters=fit['active_parameters'],complete_math_and_state_chain_pass=True))
    for _,_,p,h,rh in files:assert sha(p)==h and sha(p.with_suffix('.json'))==rh
    for p,rc in inputs.items():assert sha(BASE/p)==rc['sha256']
    receipt=dict(status='pass',scope='bounded no-GT actual accepted P4 first formal prefix math/scoped state/reset/writeback/Native and opaque readback',
        counts=dict(counts),records=records,inputs=inputs,first_formal_receipts=first,
        prediction_bytes=sum(r['bytes'] for r in records),input_bytes=sum(v['bytes'] for v in inputs.values()),
        runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),qualification_sha256=sha(BASE/'P4_QUALIFICATION.json'),
        original_science_and_prediction_bytes_unchanged=True,CPU_only=True,new_model_calls=0,GT_read=False,
        P4_global_seal=False,P4_phase_complete=False,paper_suite_complete=False,time=time.time())
    write(REC/'ROOT_RESUME_READBACK.json',receipt)
    print({k:receipt[k] for k in ['status','scope','counts','prediction_bytes','input_bytes','GT_read']})


if __name__=='__main__':run()
