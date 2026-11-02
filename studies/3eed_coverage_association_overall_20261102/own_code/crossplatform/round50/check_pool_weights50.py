"""Pure synthetic invariants; no real cache/media/labels/model access."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
from pool_weights50 import visible_mass_weights,conditional_operator,wide_entry,taxonomy_flags
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/round50/synthetic_attempt001'

def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    results=[];max_constant=0.
    boxes=np.tile(np.array([0,0,0,2,2,2],np.float32),(256,1))
    c=np.zeros((256,3),np.int64);c[0]=[4,0,0];c[1]=[0,1,0];c[3]=[1,2,4]
    n=c.sum(1);n[1]=101;n[3]=20
    u=n-c.sum(1);amb=np.zeros(256,np.int64);pool=np.array([0,1,2],np.int64)
    source=np.zeros(256,np.float32);source[0]=1
    for k in (3,10):
        mass=visible_mass_weights(pool,c,n,u,amb,boxes,k)
        assert np.allclose(mass['conditional_association'][mass['rho']>0].sum(1),1,rtol=0,atol=1e-12)
        assert np.allclose(mass['rho'],c.sum(1)/(n+k),rtol=0,atol=1e-15)
        assert np.allclose(mass['unknown_mass'],mass['unknown_prior_mass']+mass['out_of_pool_foreground_mass']+mass['unassigned_mass']+mass['ambiguous_mass'],rtol=0,atol=1e-12)
        for unmapped_source in (False,True):
            s=source.copy()
            if unmapped_source:s[2]=2
            state=conditional_operator(s,mass['conditional_association'],mass['rho'],pool,np.full(3,13.125),np.tile([.35,-.77],(3,1)))
            for name in ('vision','position','relation'):
                max_constant=max(max_constant,float(abs(state[name]).max()))
                assert abs(state[name]).max()<=1e-12
                assert (state[name][mass['rho']==0]==0).all()
            assert state['query']==int(s.argmax())
        results.append({'check':'constant_e_and_PR_no_instance_delta_mapped_or_unmapped_q0','k':k,'PASS':True})
        varied=conditional_operator(source,mass['conditional_association'],mass['rho'],pool,np.array([-100.,100.,200.]),np.array([[-10,-10],[10,10],[20,-20.]]))
        for name in ('vision','position','relation'):
            assert (abs(varied[name])<=2*mass['rho']+1e-15).all()
            assert (varied[name][mass['rho']==0]==0).all()
        results.append({'check':'weak_rho_contracts_each_clipped_increment_and_zero_rho_exact_zero','k':k,'PASS':True})
    # No selected foreground: the original median fallback must not create a
    # nonzero increment in any row, even with large nonconstant finite evidence.
    empty=visible_mass_weights(pool,np.zeros_like(c),np.zeros_like(n),np.zeros_like(n),np.zeros_like(n),boxes,3)
    zero=conditional_operator(source,empty['conditional_association'],empty['rho'],pool,np.array([200.,-100.,4.]),np.array([[1.,2.],[3.,4.],[5.,6.]]))
    for key in ('vision','position','relation'):
        assert zero[key].tobytes()==np.zeros(256).tobytes()
    results.append({'check':'all_zero_support_is_bitexact_source_increment_zero','PASS':True})
    # Missing/ambiguous/out-of-pool support remains unknown, not reassigned.
    part=visible_mass_weights(np.array([0],np.int64),c,n,u,amb,boxes,3)
    assert part['rho'][3]==1/23 and part['out_of_pool_foreground_mass'][3]==6/23
    assert part['unassigned_mass'][3]==13/23
    results.append({'check':'unknown_quality_not_foreground_renormalized','PASS':True})
    entry=wide_entry(np.array([2],np.int64),c,np.ones(3,bool))
    assert entry['pool'].tolist()==[0,1,2]
    flags=taxonomy_flags('person',np.array([1,2,3],np.int64))
    assert not flags['historical_taxonomy_compatible'].any() and flags['corrected_taxonomy_compatible'].tolist()==[True,False,False]
    results.append({'check':'wide_entrance_and_taxonomy_flags_are_not_a_class_gate','PASS':True})
    # Read code only; do not execute the rejected naive newW readout as an arm.
    previous=ROOT/'research/round50/preexecution_revision001/pool_weights50.py'
    current=Path(__file__).with_name('pool_weights50.py')
    def node(path):return next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='legacy_operator')
    assert ast.dump(node(previous),include_attributes=False)==ast.dump(node(current),include_attributes=False)
    results.append({'check':'legacy_operator_AST_unchanged_from_preexecution_revision001','PASS':True})
    result={'state':'PASS_SYNTHETIC_CONDITIONAL_ASSOCIATION_POSTCLIP_RELIABILITY','checks':results,
        'maximum_constant_e_or_PR_increment':max_constant,'real_data_model_or_label_access':False,
        'old_naive_formula_not_run_as_an_experiment':True,
        'module_sha256':hashlib.sha256(current.read_bytes()).hexdigest()}
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
