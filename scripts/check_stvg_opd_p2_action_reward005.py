"""CPU contracts for bound original-action readback; no model, video or GT."""
import copy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p2_inline_gradient004 import verify
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha
from scripts.decota_matrix_common_v1 import load
REC=BASE/'recovery/P2_actual_action_reward_precision_005'


def run():
    verify()
    import numpy as np,torch
    from vg_tta.decota_spatial_opd_action_readback_precision005 import check,geometry32,FIELD,audit
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    torch.set_num_threads(2);rng=np.random.default_rng(5005);valid=[];cases=[]
    for n in [1,2,3,4,16]:
        for width in [.005,.3]:
            for overlap in [True,False]:
                e=np.tile(np.array([.45,.55,width,width],np.float32),(n,1))
                mean=np.log(e/(1-e)).astype(np.float32)
                if not overlap:mean[:,:2]+=np.float32(2.)
                x=(mean[:,None,:]+np.float32(.025)*rng.normal(0,1,(n,32,4))).astype(np.float32)
                actions=torch.sigmoid(torch.tensor(x)).numpy();reward=geometry32(actions,e)
                args=[x,e,reward,actions];check(*args);cases.append(args)
                valid.append(dict(positions=n,tiny_geometry=bool(width<.01),overlap=overlap,pass_=True))
    base=cases[0];bad=[]
    def reject(name,mutate):
        args=copy.deepcopy(base);mutate(args)
        try:check(*args)
        except (AssertionError,ValueError,IndexError):bad.append(dict(name=name,rejected=True))
        else:raise AssertionError('Bad contract was accepted '+name)
    reject('wrong_reward',lambda a:a[2].__setitem__((0,0),a[2][0,0]+np.float32(.05)))
    reject('wrong_action',lambda a:a[3].__setitem__((0,0,0),a[3][0,0,0]+np.float32(.01)))
    reject('wrong_action_with_recomputed_reward',lambda a:(a[3].__setitem__((0,0,0),a[3][0,0,0]+np.float32(.01)),a.__setitem__(2,geometry32(a[3],a[1]))))
    reject('wrong_sample_binding',lambda a:a[0].__setitem__((0,0,0),a[0][0,0,0]+np.float32(1.)))
    reject('wrong_evidence',lambda a:a[1].__setitem__((0,0),np.float32(.1)))
    reject('negative_evidence_width',lambda a:a[1].__setitem__((0,2),np.float32(-.1)))
    reject('zero_evidence_height',lambda a:a[1].__setitem__((0,3),np.float32(0.)))
    reject('missing_action',lambda a:a.__setitem__(3,a[3][:,:31,:]))
    reject('missing_sample',lambda a:a.__setitem__(0,a[0][:,:31,:]))
    reject('wrong_reward_shape',lambda a:a.__setitem__(2,a[2][:,:31]))
    reject('wrong_evidence_shape',lambda a:a.__setitem__(1,a[1][:,:3]))
    reject('nonfinite_sample',lambda a:a[0].__setitem__((0,0,0),np.float32(np.inf)))
    reject('nonfinite_action',lambda a:a[3].__setitem__((0,0,0),np.float32(np.nan)))
    reject('nonfinite_reward',lambda a:a[2].__setitem__((0,0),np.float32(np.inf)))
    reject('nonfloat32_action',lambda a:a.__setitem__(3,a[3].astype(float)+1e-12))
    reject('nonfloat32_sample',lambda a:a.__setitem__(0,a[0].astype(float)+1e-12))
    reject('out_of_range_action',lambda a:a[3].__setitem__((0,0,0),np.float32(1.1)))
    actual=load(REC/'REPRODUCED_COMPLETE_ACTION_READBACK_FIT.pt');f=actual['fit']
    from scripts.run_stvg_opd_p2_inline_gradient004 import previous
    import vg_tta.stvg_opd_paper_component_audit_v1 as raw
    previous.ORIGINAL_COMPONENT_AUDIT=raw.audit
    checked=audit(f,actual['expert'],raw.audit)
    assert checked['original_sigmoid_CPU32_reward_failures']>0
    extra=[]
    for name,mutate in [
        ('wrong_trace_samples',lambda z:z[FIELD]['trace'][0]['samples'].__setitem__((0,0,0),z[FIELD]['trace'][0]['samples'][0,0,0]+.1)),
        ('false_same_call_flag',lambda z:z[FIELD]['trace'][0].__setitem__('same_original_GPU_sigmoid_call',False)),
        ('missing_trace_call',lambda z:z[FIELD]['trace'].pop()),
        ('unknown_action_revision',lambda z:z[FIELD].__setitem__('revision','unknown'))]:
        z=copy.deepcopy(f);mutate(z)
        try:audit(z,actual['expert'],raw.audit)
        except AssertionError:extra.append(dict(name=name,rejected=True))
        else:raise AssertionError('Invalid stored action trace was accepted')
    assert len(valid)==20 and len(bad)==17 and len(extra)==4
    write(REC/'CPU_CONTRACTS.json',dict(status='pass',scope='independent actual-action geometry/readout valid and wrong/malformed rejection contracts plus actual saved complete failure',valid_count=20,rejected_count=21,valid_contracts=valid,rejection_contracts=bad+extra,actual_complete_fit_math=checked,actual_action_fit_sha256=sha(REC/'REPRODUCED_COMPLETE_ACTION_READBACK_FIT.pt'),original_failed_guard_preserved=True,new_model_calls=0,new_DINO_calls=0,GT_read=False,time=time.time()))
    print('P2_ACTION005_CPU_PASS_20_VALID_21_REJECTIONS_ACTUAL_COMPLETE_FAILURE_RETAINED',flush=True)


if __name__=='__main__':run()
