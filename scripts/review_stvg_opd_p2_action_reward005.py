"""Actual CPU root original-prefix, saved revision, state and resumption audit."""
import copy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p2_action_reward005 import REC,verify,dispatch_saved,component_audit,prior
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha
from scripts.decota_matrix_common_v1 import load
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal


def prefix(paths):
    import torch
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.stvg_opd_paper_later_common_v1 import committed
    chains={};count=size=coordinates=rounds=0;revisions={};records=[]
    for p in sorted(paths):
        rc=read(p.with_suffix('.json'));h=sha(p)
        assert h==rc['sha256'] and p.stat().st_size==rc['bytes'] and rc['GT_read'] is False
        z=load(p);f=z['fit'];ip=BASE/z['input']['path'];assert sha(ip)==z['input']['sha256']
        inp=load(ip);assert inp['GT_read'] is False and z['GT_read'] is False
        equal(z['interval'],inp['interval'],dict(tensors=0,tensor_coordinates=0,scalar_values=0))
        assert z['Native_WHEN_fixed'] and z['query_reset'] and z['Adam_reset']
        checked=dispatch_saved(f,unpack_expert(inp['expert']),z);assert checked==z['math_audit']
        key=(z['stage'],z['condition'],z['order'],z['arm']);previous=chains.get(key)
        assert z['previous_payload_sha256']==(None if previous is None else previous[0])
        if previous is None:
            source=load(BASE/'component_qualification'/z['stage']/z['condition']/'order1'/z['arm']/'00000.pt')['fit']['initial']
            assert all(torch.equal(v,source[n]) for n,v in f['initial'].items())
        else:
            assert all(torch.equal(v,torch.zeros_like(v) if n=='spatial.query_residual' else previous[1][n])
                       for n,v in f['initial'].items())
        assert torch.count_nonzero(f['initial']['spatial.query_residual'])==0
        expected=committed(f['initial'],f['state'],z['config']['writeback'])
        assert all(torch.equal(expected[n],z['committed'][n]) for n in expected)
        assert f['active_parameters']==1792 and f['selected_step']==z['config']['steps']
        assert z['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json')
        assert z['component_runtime_sha256']==sha(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json')
        rev=str(checked.get('revision'));revisions[rev]=revisions.get(rev,0)+1
        chains[key]=(h,z['committed']);count+=1;size+=rc['bytes'];rounds+=len(f['rounds'])
        coordinates+=sum(v.numel() for v in f['state'].values())
        records.append(dict(path=str(p.relative_to(BASE)),sha256=h,bytes=rc['bytes'],receipt_sha256=sha(p.with_suffix('.json'))))
    return dict(accepted_snapshot_predictions=count,accepted_snapshot_bytes=size,rounds_read_back=rounds,
        state_coordinates=coordinates,stream_chains=len(chains),saved_audit_revision_counts=revisions,records=records,
        original_source_resets_verified=True,complete_1792_inheritance_and_writeback=True,
        query_and_Adam_resets_verified=True,Native_WHEN_verified=True,GT_read=False)


def old_integrity():
    capture=read(REC/'CAPTURE_RECEIPT.json');records=capture['records']
    for r in records:
        p=BASE/r['path'];assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes']
        assert sha(p.with_suffix('.json'))==r['receipt_sha256']
        assert sha(REC/'opaque_predictions'/r['path'])==r['sha256']
    for f,rc in capture['opaque_input_snapshot'].items():
        p=BASE/f;assert sha(p)==rc['sha256'] and p.stat().st_size==rc['bytes']
        assert sha(p.with_suffix('.json'))==rc['receipt_sha256']
    return dict(original_prediction_count=len(records),original_prediction_bytes=sum(r['bytes'] for r in records),
        original_input_count=len(capture['opaque_input_snapshot']),original_prediction_and_receipt_bytes_unchanged=True)


def dispatch_contracts():
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from vg_tta.decota_spatial_opd_action_readback_precision005 import FIELD
    controls=read(REC/'qualification/ordinary_controls/GPU_QUALIFICATION.json')['records'];count=0
    for r in controls:
        z=load(BASE/r['path']);inp=load(BASE/z['input']['path']);expert=unpack_expert(inp['expert'])
        oldpath=BASE/'component_qualification'/r['stage']/r['condition']/'order1'/r['arm']/f"{r['arrival']:05}.pt"
        old=load(oldpath);assert sha(oldpath)==r['original_qualification_sha256']
        equal(z['old_control_fit'],old['fit'],dict(tensors=0,tensor_coordinates=0,scalar_values=0))
        assert dispatch_saved(old['fit'],expert,old)==old['math_audit']
        assert dispatch_saved(z['fit'],expert,z)==z['math_audit'];count+=2
    for r in read(prior.REC/'qualification/ordinary_controls/GPU_QUALIFICATION.json')['records']:
        z=load(BASE/r['path']);inp=load(BASE/z['input']['path'])
        assert dispatch_saved(z['fit'],unpack_expert(inp['expert']),z)==z['math_audit'];count+=1
    old4=load(prior.REC/'qualification/failed_query/QUALIFIED_COMPLETE_FIT.pt')
    inp=load(BASE/old4['input']['path']);assert dispatch_saved(old4['fit'],unpack_expert(inp['expert']),old4)==old4['math_audit'];count+=1
    b=read(BASE/'stages/P2_hc2_cross_clean/PREDICTION_BARRIER.json');seen=set()
    for _,r in b['logical_records'].items():
        if r.get('origin_stage','').startswith('P0_'):
            z=load(BASE/r['path']);rev=z['math_audit'].get('revision')
            if rev in seen:continue
            inp=load(BASE/z['input']['path']);expert=unpack_expert(inp['expert'])
            assert dispatch_saved(z['fit'],expert,z,r)==z['math_audit'];seen.add(rev);count+=1
    z=load(REC/'qualification/failed_query/QUALIFIED_COMPLETE_FIT.pt');inp=load(BASE/z['input']['path']);expert=unpack_expert(inp['expert'])
    assert dispatch_saved(z['fit'],expert,z)==z['math_audit'];count+=1
    wrong=copy.deepcopy(z);wrong['math_audit']['revision']='unregistered_P2_actual_action_revision'
    try:dispatch_saved(wrong['fit'],expert,wrong)
    except AssertionError:pass
    else:raise AssertionError('Unknown revision was not rejected')
    wrong=copy.deepcopy(z);wrong['actual_action_runtime_sha256']='0'*64
    try:dispatch_saved(wrong['fit'],expert,wrong)
    except AssertionError:pass
    else:raise AssertionError('Wrong actual-action runtime was not rejected')
    wrong=copy.deepcopy(z['fit']);wrong[FIELD]['trace'][0]['actions'][0,0,0]+=.01
    try:component_audit(wrong,expert)
    except AssertionError:pass
    else:raise AssertionError('Changed original GPU action was not rejected')
    checked=component_audit(z['fit'],expert);wrong=copy.deepcopy(z['math_audit']);wrong['original_sigmoid_CPU32_reward_failures']=-1
    assert checked!=wrong
    return dict(saved_receipt_dispatch_contracts=count,unknown_revision_rejections=1,wrong_runtime_rejections=1,
        changed_GPU_action_rejections=1,rewritten_math_dictionary_rejections=1,
        complete_ordinary_old_control_fit_readbacks=12,old_inline004_complete_fit_readbacks=13,
        old_P0_audit_revisions=list(seen),GT_read=False)


def run(job):
    verify()
    import torch
    torch.set_num_threads(2)
    import vg_tta.stvg_opd_paper_component_audit_v1 as target
    prior.previous.ORIGINAL_COMPONENT_AUDIT=target.audit
    assert read(REC/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
    integrity=old_integrity()
    if job=='before_resume':
        contracts=dispatch_contracts()
        write(REC/'ROOT_RECEIPT_DISPATCH_CONTRACTS.json',dict(status='pass',scope='actual old/current/inline004 qualification and P0 alias exact receipt dispatch with fail-closed rejection controls',
            **contracts,runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
        result=prefix([BASE/r['path'] for r in read(REC/'CAPTURE_RECEIPT.json')['records']])
        assert result['accepted_snapshot_predictions']==467 and result['accepted_snapshot_bytes']==535577607
        write(REC/'ROOT_ORIGINAL_PREFIX_READBACK.json',dict(status='pass',scope='all original accepted P2 prefix mathematics / source and full1792 state chain / opaque bytes; no GT efficacy scoring',
            **result,**integrity,CPU_only=True,new_model_calls=0,time=time.time()))
        print('P2_ACTION005_ROOT_ORIGINAL467_AND_EXACT_RECEIPT_DISPATCH_PASS',flush=True);return
    assert job=='resume'
    first=BASE/'stages/P2_hc2_same_5percent/frame_drop_5/order1/on_policy/00052.pt'
    assert first.exists() and first.with_suffix('.json').exists()
    from vg_tta.decota_spatial_opd_action_readback_precision005 import comparable
    qualified=load(REC/'qualification/failed_query/QUALIFIED_COMPLETE_FIT.pt');formal=load(first)
    counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
    equal(comparable(qualified['fit'],True),comparable(formal['fit'],True),counts)
    assert read(REC/'FIRST_FORMAL_FIT_BITWISE.json')['status']=='pass'
    paths=[p for p in (BASE/'stages').glob('P2_*/**/*.pt') if p.with_suffix('.json').exists()]
    result=prefix(paths);assert result['accepted_snapshot_predictions']>=468
    assert old_integrity()==integrity
    write(REC/'ROOT_RESUME_READBACK.json',dict(status='pass',scope='actual immutable-prefix and first formally resumed fit exact qualification / full1792 chain / receipt-aware mathematics / reset / Native WHEN',
        **result,**integrity,first_missing_formal_fit_and_checks_bitwise=counts,
        first_formal_fit_sha256=sha(first),qualified_fit_sha256=sha(REC/'qualification/failed_query/QUALIFIED_COMPLETE_FIT.pt'),
        P2_global_seal=(BASE/'P2_PREDICTION_BARRIER.json').exists(),paper_suite_complete=False,
        CPU_only=True,new_model_calls=0,time=time.time()))
    print('P2_ACTION005_ROOT_RESUMED_REAL_PREFIX_PASS',result['accepted_snapshot_predictions'],flush=True)


if __name__=='__main__':run(sys.argv[1])
