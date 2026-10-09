"""One actual full original-reward-failure replay with same-call action logging."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.run_stvg_opd_p2_inline_gradient004 as prior
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha
from scripts.decota_matrix_common_v1 import load,save
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
REC=BASE/'recovery/P2_actual_action_reward_precision_005'


def run():
    prior.verify()
    runtime=read(REC/'REPRODUCTION_RUNTIME.json');assert sha(Path(__file__))==runtime['script_sha256']
    assert sha(REC/'CAPTURE_RECEIPT.json')==runtime['capture_sha256']
    from vg_tta.decota_spatial_opd_action_readback_precision005 import Installation,FIELD
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import comparable
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    import scripts.run_stvg_opd_paper_components_v2 as runner
    original=load(BASE/read(REC/'CAPTURE_RECEIPT.json')['original_failed_fit_path'])
    installation=Installation();old_audit=prior.component_audit;oldsave=runner.save;witness=None
    def audited(fit,expert):
        nonlocal witness
        frame=sys._getframe(1)
        while frame is not None and not frame.f_code.co_filename.endswith('run_stvg_opd_paper_components_v2.py'):
            frame=frame.f_back
        assert frame is not None;loc=frame.f_locals
        assert loc['q']==2676 and loc['at']==52 and loc['arm']=='on_policy'
        counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
        equal(comparable({k:v for k,v in fit.items() if k!=FIELD},True),comparable(original['fit'],True),counts)
        excounts=dict(tensors=0,tensor_coordinates=0,scalar_values=0);equal(detached(expert,'cpu'),original['expert'],excounts)
        assert loc['inputrc']['new_DINO_calls']==0
        assert state_hash(loc['model'].state_dict())==loc['modelhash']
        assert state_hash(loc['expert'].model.state_dict())==loc['experthash']
        try:old_audit(fit,expert)
        except AssertionError as exc:
            tb=exc.__traceback__
            while tb.tb_next:tb=tb.tb_next
            assert tb.tb_frame.f_code.co_filename.endswith('decota_spatial_opd_reward_audit_revision005.py') and tb.tb_lineno==66
        else:raise AssertionError('Original CPU32 sigmoid/reward failure did not repeat')
        witness=dict(fit=fit,expert=detached(expert,'cpu'),stage='P2_hc2_same_5percent',condition=loc['condition'],
            order=loc['order'],arrival=52,query_ordinal=2676,arm='on_policy',config=loc['cfg'],
            input=loc['inputrc'],previous_payload_sha256=loc['prevhash'][loc['arm']],
            original_serialized_full_fit_bitwise=counts,expert_bitwise=excounts,GT_read=False)
        save(REC/'REPRODUCED_COMPLETE_ACTION_READBACK_FIT.pt',witness)
        raise OriginalReproductionComplete()
    class OriginalReproductionComplete(Exception):pass
    def reject(path,z):
        if str(path).startswith(str(BASE/'stages')) and isinstance(z,dict) and 'fit' in z:
            raise AssertionError('Diagnostic replay must accept zero formal predictions')
        return oldsave(path,z)
    prior.component_audit=audited;runner.save=reject
    try:
        try:prior.run('GPU','P2_hc2_same_5percent')
        except OriginalReproductionComplete:assert witness is not None
        else:raise AssertionError('Original diagnostic failed to stop')
    finally:prior.component_audit=old_audit;runner.save=oldsave;installation.uninstall()
    write(REC/'REPRODUCTION_RECEIPT.json',dict(status='pass_original_complete_failure_reproduced',
        scope='one complete real40-round fit with original CPU32 assertion retained and same original GPU action values recorded',
        stage=witness['stage'],condition=witness['condition'],order=witness['order'],arrival=52,query_ordinal=2676,
        original_serialized_complete_fit_bitwise=witness['original_serialized_full_fit_bitwise'],
        expert_bitwise=witness['expert_bitwise'],original_complete_failed_fit_serialized_before_exit=True,
        action_readback_fit_sha256=sha(REC/'REPRODUCED_COMPLETE_ACTION_READBACK_FIT.pt'),
        action_readback_fit_bytes=(REC/'REPRODUCED_COMPLETE_ACTION_READBACK_FIT.pt').stat().st_size,
        input_sha256=witness['input']['sha256'],previous_payload_sha256=witness['previous_payload_sha256'],
        source_and_expert_unchanged=True,formal_predictions_accepted=0,new_DINO_calls=0,GT_read=False,
        actual_GPU_fits=1,runtime_sha256=sha(REC/'REPRODUCTION_RUNTIME.json'),time=time.time()))
    print('P2_ACTUAL_ACTION005_ORIGINAL_COMPLETE_FAILURE_REPLAY_BITWISE_ZERO_FORMAL',flush=True)


if __name__=='__main__':run()
