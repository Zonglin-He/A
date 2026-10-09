"""Bounded CPU scope/installation contracts and real saved joint-fit arithmetic."""
import copy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,write,sha
activate()
from scripts.decota_matrix_common_v1 import load
from vg_tta.stvg_opd_p3_scope_precision001 import Installation,audit,FIELD,ACTION,REVISION,CHART,ACTION_REVISION
from vg_tta.stvg_opd_paper_component_audit_v1 import audit as original
from vg_tta.decota_fixed_full_audit_v1 import vector
REC=BASE/'P3_scope_engineering_revision001'


def run():
    torch.set_num_threads(2)
    import vg_tta.decota_spatial_opd_tunable_v1 as core
    import vg_tta.stvg_opd_paper_ablations_v1 as component
    old=(core.fit,component.component_fit,core.TrickReplay.values,core.mean_coordinates,component.mean_coordinates)
    x=Installation();new=(core.fit,component.component_fit,core.TrickReplay.values,core.mean_coordinates,component.mean_coordinates)
    with x.original():assert old==(core.fit,component.component_fit,core.TrickReplay.values,core.mean_coordinates,component.mean_coordinates)
    assert new==(core.fit,component.component_fit,core.TrickReplay.values,core.mean_coordinates,component.mean_coordinates)
    x.uninstall();assert old==(core.fit,component.component_fit,core.TrickReplay.values,core.mean_coordinates,component.mean_coordinates)
    valid=0;rejected=[]
    for scope in ['joint','query_only','LN_only']:
        for steps in [1,2,10,40]:
            state={'spatial.query_residual':torch.zeros(256),**{f'LN.{j}':torch.ones(256)*(j%2) for j in range(6)}}
            names=[n for n in state if scope=='joint' or (scope=='query_only')==(n=='spatial.query_residual')]
            boxes=torch.tensor([[.4,.5,.2,.3]])
            fit=dict(arm='on_policy',initial=copy.deepcopy(state),state=copy.deepcopy(state),before=boxes.clone(),final=boxes.clone(),
                selected_step=steps,path=[dict(step=k,state=copy.deepcopy(state),boxes=boxes.clone()) for k in range(steps+1)],
                rounds=[dict(round=k,updated=False,no_information=True) for k in range(steps)],gradient_calls=0,
                active_parameters=len(vector(state,names)),scope=scope,direct_objective=False,optimizer_parameter_names=names,
                positions=[],empty=True,GT_used=False,config=dict(sigma=.025,tau=.05,samples=32,steps=steps,lr=.01,writeback=1/16),
                numerical_chart=dict(revision=CHART,interior_formula='original_torch_logit',endpoint_formula='same_call_finite_native_logit',trace=[]))
            fit[FIELD]=dict(revision=REVISION,rounds=[],CPU_audit_seconds=0.);fit[ACTION]=dict(revision=ACTION_REVISION,trace=[],CPU_action_readback_seconds=0.)
            assert audit(fit,{'observations':{}},original)['status']=='pass';valid+=1
            if steps!=2:continue
            changes={'wrong active size':lambda f:f.update(active_parameters=42),'wrong scope names':lambda f:f.update(optimizer_parameter_names=[]),
                'changed state':lambda f:f['state']['LN.0'].add_(1),'changed final boxes':lambda f:f['final'].add_(.1),
                'false selected round':lambda f:f.update(selected_step=1),'rewritten initial':lambda f:f['initial']['LN.0'].add_(1),
                'unknown gradient revision':lambda f:f[FIELD].update(revision='unknown'),'unknown action revision':lambda f:f[ACTION].update(revision='unknown'),
                'zero sigma':lambda f:f['config'].update(sigma=0),'zero tau':lambda f:f['config'].update(tau=0),
                'fake observation':lambda f:f.update(positions=[0]),'false no-op update':lambda f:f['rounds'][0].update(updated=True)}
            for label,change in changes.items():
                bad=copy.deepcopy(fit);change(bad)
                try:audit(bad,{'observations':{}},original)
                except (AssertionError,KeyError,ValueError):rejected.append(scope+':'+label)
                else:raise AssertionError('Malformed scope accepted '+scope+':'+label)
    from scripts.run_decota_paper_main_v1 import unpack_expert
    actual=[];reject_actual=[]
    for ds in ['hc2','vidstg']:
        for at in [126,127]:
            p=BASE/'stages'/('P2_'+ds+'_same_5percent')/'exposure_5/order1/on_policy'/f'{at:05}.pt'
            z=load(p);fit=copy.deepcopy(z['fit']);fit[FIELD]['revision']=REVISION
            inp=BASE/z['input']['path'];assert sha(inp)==z['input']['sha256'];data=load(inp);assert not data['GT_read'];ex=unpack_expert(data['expert'])
            assert audit(fit,ex,original)['status']=='pass'
            actual.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),rounds=fit['selected_step'],active_parameters=1792,
                interpretation='CPU scoped joint audit of immutable real P2 fit; no new GPU P3 qualification',GT_read=False))
            if fit['positions']:
                bad=copy.deepcopy(fit);bad[ACTION]['trace'][0]['actions'].add_(.01)
                try:audit(bad,ex,original)
                except AssertionError:reject_actual.append(str(p.relative_to(ROOT)))
                else:raise AssertionError('Changed saved action accepted')
    receipt=dict(status='pass',scope='CPU-only scope no-op/install and actual historical joint-fit arithmetic; not P3 GPU qualification',
        valid_scope_contracts=valid,malformed_rejections=len(rejected),rejected=rejected,actual_saved_joint_fits=len(actual),
        actual_saved_fit_records=actual,changed_saved_action_rejections=len(reject_actual),source_AST_assertion_only_checked=True,
        original_hook_restore_checked=True,new_model_calls=0,GT_read=False,actual_GPU_fits=0,time=time.time())
    write(REC/'CPU_CONTRACTS_revision002.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['rejected','actual_saved_fit_records']}))


if __name__=='__main__':run()
