"""Once-only original failing fit replay from the complete saved formal prefix."""
import sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p2_input_bridge003 import verify,run as original_run
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha
from scripts.decota_matrix_common_v1 import save
REC=BASE/'recovery/P2_inline_gradient_precision_004'


def run():
    verify()
    import torch
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from vg_tta.decota_spatial_opd_chart_revision003 import Installation
    import scripts.run_stvg_opd_paper_components_v2 as runner
    runtime=read(REC/'REPRODUCTION_RUNTIME_revision001.json')
    assert sha(Path(__file__))==runtime['script_sha256']
    assert sha(REC/'CAPTURE_RECEIPT.json')==runtime['capture_sha256']
    assert not (REC/'REPRODUCTION_RECEIPT.json').exists()
    old_init=Installation.__init__;old_save=runner.save;witness=None

    def capture(exc,context):
        nonlocal witness
        tb=exc.__traceback__
        while tb.tb_next:tb=tb.tb_next
        assert tb.tb_frame.f_code.co_filename.endswith('decota_spatial_opd_tunable_v1.py')
        assert tb.tb_lineno==100 and isinstance(exc,AssertionError)
        loc=tb.tb_frame.f_locals
        f=tb.tb_frame.f_back
        while f is not None and not f.f_code.co_filename.endswith('run_stvg_opd_paper_components_v2.py'):
            f=f.f_back
        assert f is not None;meta=f.f_locals
        assert meta['q']==3051 and meta['at']==7 and meta['arm']=='frozen_rollout'
        assert loc['arm']=='frozen_rollout' and meta['condition']=='frame_drop_5'
        assert state_hash(meta['model'].state_dict())==meta['modelhash']
        assert state_hash(meta['expert'].model.state_dict())==meta['experthash']
        witness=dict(status='actual_original_guard_reproduced',stage='P2_hc2_same_5percent',
            condition=meta['condition'],order=meta['order'],arrival=meta['at'],query_ordinal=meta['q'],
            arm=loc['arm'],failed_round=loc['k'],config=loc['cfg'],
            mean=detached(loc['mu'],'cpu'),rollout=detached(loc['rr'],'cpu'),
            autograd_mean_gradient=detached(loc['gmu'],'cpu'),analytic_mean_gradient=detached(loc['analytic'],'cpu'),
            original_inline_error=loc['error'],positions=loc['positions'],evidence=detached(loc['evidence'],'cpu'),
            initial=detached(loc['origin'],'cpu'),incomplete_path=detached(loc['path'],'cpu'),
            completed_rounds=detached(loc['audits'],'cpu'),optimizer=detached(loc['opt'].state_dict(),'cpu'),
            chart_context=detached(context,'cpu'),input=meta['inputrc'],
            previous_payload_sha256=meta['prevhash'][meta['arm']],GT_read=False,
            original_dead_failed_fit_serialized=False,formal_predictions_accepted=0)
        save(REC/'REPRODUCED_INCOMPLETE_FIT_WITNESS.pt',witness)
        (REC/'reproduced_traceback.txt').write_text(''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)))

    def init(self,*args,**kwargs):
        kwargs['failure_callback']=capture
        return old_init(self,*args,**kwargs)

    def reject_new_formal(path,z):
        if str(path).startswith(str(BASE/'stages')) and isinstance(z,dict) and 'fit' in z:
            raise RuntimeError('Original diagnostic replay must accept zero new formal predictions')
        return old_save(path,z)

    Installation.__init__=init;runner.save=reject_new_formal
    try:
        try:original_run('GPU','P2_hc2_same_5percent')
        except AssertionError:
            assert witness is not None
        else:raise AssertionError('Expected actual original in-loop guard did not reproduce')
    finally:Installation.__init__=old_init;runner.save=old_save
    write(REC/'REPRODUCTION_RECEIPT.json',dict(status='pass_original_failure_reproduced',
        scope='one real original failing frozen-rollout fit with saved complete-prefix state reconstruction; no new formal acceptance',
        stage=witness['stage'],condition=witness['condition'],order=witness['order'],
        arrival=witness['arrival'],query_ordinal=witness['query_ordinal'],
        witness_sha256=sha(REC/'REPRODUCED_INCOMPLETE_FIT_WITNESS.pt'),
        witness_bytes=(REC/'REPRODUCED_INCOMPLETE_FIT_WITNESS.pt').stat().st_size,
        previous_payload_sha256=witness['previous_payload_sha256'],input_sha256=witness['input']['sha256'],
        source_and_expert_unchanged=True,formal_predictions_accepted=0,new_DINO_calls=0,
        original_dead_failed_fit_serialized=False,no_dead_memory_bitwise_claim=True,
        GT_read=False,runtime_sha256=sha(REC/'REPRODUCTION_RUNTIME_revision001.json'),time=time.time()))
    print('ORIGINAL_INLINE_GRADIENT_FAILURE_REPRODUCED_ZERO_FORMAL_ACCEPTED',flush=True)


if __name__=='__main__':run()
