"""One unchanged original005 guard replay; save the actual incomplete witness."""
import sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.run_stvg_opd_p2_action_reward005 as original
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha
from scripts.decota_matrix_common_v1 import save
REC=BASE/'recovery/P2_matched_gradient_precision_006'


def run():
    original.verify();runtime=read(REC/'REPRODUCTION_RUNTIME.json');cap=read(REC/'CAPTURE_RECEIPT.json')
    assert sha(Path(__file__))==runtime['script_sha256'] and sha(REC/'CAPTURE_RECEIPT.json')==runtime['capture_sha256']
    assert not (REC/'REPRODUCTION_RECEIPT.json').exists()
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import Installation as Inline
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    import scripts.run_stvg_opd_paper_components_v2 as runner
    old_guard=Inline.guard;old_save=runner.save;witness=None;meta_live=None
    def guard(inst,mu,rr,gmu,analytic,error,cfg,k):
        nonlocal witness,meta_live
        try:return old_guard(inst,mu,rr,gmu,analytic,error,cfg,k)
        except AssertionError as exc:
            tb=exc.__traceback__
            while tb.tb_next:tb=tb.tb_next
            assert tb.tb_frame.f_code.co_filename.endswith('decota_spatial_opd_inline_gradient_precision004.py') and tb.tb_lineno==42
            f=sys._getframe(1);core=meta=chart=actions=None
            while f is not None:
                if f.f_code.co_filename.endswith('decota_spatial_opd_tunable_v1.py') and f.f_code.co_name=='fit':core=f.f_locals
                if f.f_code.co_filename.endswith('run_stvg_opd_paper_components_v2.py'):meta=f.f_locals
                if f.f_code.co_filename.endswith('run_stvg_opd_p2_engineering002.py'):chart=f.f_locals.get('installation')
                if f.f_code.co_filename.endswith('run_stvg_opd_p2_action_reward005.py'):actions=f.f_locals.get('actions',actions)
                f=f.f_back
            assert core is not None and meta is not None and chart is not None and actions is not None
            assert meta['at']==cap['arrival'] and meta['q']==cap['query_ordinal'] and meta['arm']==cap['arm']
            assert meta['condition']==cap['condition'] and meta['inputrc']['new_DINO_calls']==0
            assert core['k']==k and core['arm']=='frozen_rollout' and actions.contexts and inst.contexts and chart.contexts
            cpu=tb.tb_frame.f_locals
            witness=dict(status='actual_original_matched_gradient_guard_reproduced',stage=cap['stage'],condition=meta['condition'],
                order=meta['order'],arrival=meta['at'],query_ordinal=meta['q'],arm=meta['arm'],failed_round=k,config=cfg,
                mean=detached(mu,'cpu'),rollout=detached(rr,'cpu'),autograd_mean_gradient=detached(gmu,'cpu'),
                analytic_mean_gradient=detached(analytic,'cpu'),original_inline_error=error,
                original_CPU32_gradient=cpu['cpu32'],original_matched_CPU32_GPU32_error=float(cpu['same'].max()),
                positions=core['positions'],evidence=detached(core['evidence'],'cpu'),initial=detached(core['origin'],'cpu'),
                incomplete_path=detached(core['path'],'cpu'),completed_rounds=detached(core['audits'],'cpu'),
                optimizer=detached(core['opt'].state_dict(),'cpu'),chart_context=detached(chart.contexts[-1],'cpu'),
                actual_action_context=detached(actions.contexts[-1],'cpu'),inline_context=detached(inst.contexts[-1],'cpu'),
                input=meta['inputrc'],previous_payload_sha256=meta['prevhash'][meta['arm']],GT_read=False,
                original_dead_failed_fit_serialized=False,formal_predictions_accepted=0)
            save(REC/'REPRODUCED_INCOMPLETE_FIT_WITNESS.pt',witness);meta_live=meta
            (REC/'reproduced_traceback.txt').write_text(''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)))
            raise
    def reject(path,z):
        if str(path).startswith(str(BASE/'stages')) and isinstance(z,dict) and 'fit' in z:
            raise AssertionError('One diagnostic replay must accept zero new formal predictions')
        return old_save(path,z)
    Inline.guard=guard;runner.save=reject
    try:
        try:original.run('GPU',cap['stage'])
        except AssertionError:assert witness is not None
        else:raise AssertionError('Original matched precision failure did not reproduce')
    finally:Inline.guard=old_guard;runner.save=old_save
    assert state_hash(meta_live['model'].state_dict())==meta_live['modelhash']
    assert state_hash(meta_live['expert'].model.state_dict())==meta_live['experthash']
    write(REC/'REPRODUCTION_RECEIPT.json',dict(status='pass_original_failure_reproduced',scope='one real original005 failing fit with complete-prefix inheritance and actual incomplete matched-gradient witness; no new formal predictions',
        stage=cap['stage'],condition=cap['condition'],order=cap['order'],arrival=cap['arrival'],query_ordinal=cap['query_ordinal'],
        witness_sha256=sha(REC/'REPRODUCED_INCOMPLETE_FIT_WITNESS.pt'),witness_bytes=(REC/'REPRODUCED_INCOMPLETE_FIT_WITNESS.pt').stat().st_size,
        previous_payload_sha256=witness['previous_payload_sha256'],input_sha256=witness['input']['sha256'],
        source_and_expert_unchanged=True,formal_predictions_accepted=0,new_DINO_calls=0,actual_GPU_fits=1,
        original_dead_failed_fit_serialized=False,no_dead_memory_bitwise_claim=True,GT_read=False,
        runtime_sha256=sha(REC/'REPRODUCTION_RUNTIME.json'),time=time.time()))
    print('P2_MATCHED006_ORIGINAL_FAILURE_REPRODUCED_ZERO_FORMAL',flush=True)


if __name__=='__main__':run()
