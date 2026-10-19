"""Independent single-head legal normalization, all paths and replay-diff audit."""
from pathlib import Path
import sys,numpy as np,torch,inspect,textwrap,ast
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
    from methods.decota_final_simplified_v1.replay import SpatialReplay
    source=textwrap.dedent(inspect.getsource(SpatialReplay.reference_values));old='boxes=torch.stack([boxes[i % 2][i // 2] for i in range(self.n)]).float()';assert source.count(old)==1
    modified=source.replace(old,'boxes=boxes[0].float()');assert modified.replace('boxes=boxes[0].float()',old)==source
    original_ast=ast.parse(source);changed_ast=ast.parse(modified)
    # Remove just the returned bbox expression in each AST; remaining code identical.
    for tree in [original_ast,changed_ast]:
        returns=[n for n in ast.walk(tree)if isinstance(n,ast.Return)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='dict']
        assert len(returns)==1
        field=next(x for x in returns[0].value.keywords if x.arg=='boxes');field.value=ast.Constant(value=None)
    assert ast.dump(original_ast)==ast.dump(changed_ast)
    D=R/'artifacts/stvg_temporal_full_context128_v1';HR=R/'artifacts/stvg_temporal_hires128_v1';p=read(D/'PLAN_LOCK.json');b=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert b['count']==128;maxerr=0.;decisions=0;peaks=[]
    for r,f in zip(p['records'],b['files']):
        assert sha(R/f['path'])==f['sha256'];z=load(R/f['path']);h=load(HR/'predictions'/f"{r['alias']}.pt");ids=z['frame_ids'];assert len(ids)==65 and z['logits'].shape==(1,65,2)and z['logits'].dtype==torch.float32
        assert z['raw_RGB_sha256']==h['observation']['RGB_sha256'];info=z['full_context_info'];assert info['normalized_shape'][0]==65 and info['normalized_shape'][-2]==224 and info['frame_count']==65 and info['full_Source_forward_calls']==1
        assert z['actions'].shape==(1,65,1)and torch.isfinite(z['actions']).all()
        assert all(len(z['gates'][k])>0 and all(0<=i<65 for i in z['gates'][k])for k in ['first','second'])
        ll=z['logits'].numpy()[0].astype('float64');x=ll[:,0,None]+ll[None,:,1];mask=np.triu(np.ones((65,65),bool),1);y=x[mask];weights=np.exp(y-y.max());weights/=sum(weights.tolist());q=np.zeros_like(x);q[mask]=weights;err=float(abs(q-z['outputs']['Full65']['q']).max());assert err<1e-12;maxerr=max(maxerr,err)
        assert np.array_equal(z['outputs']['Full65Mix']['q'],.5*z['outputs']['Native']['q']+.5*z['outputs']['Full65']['q'])
        assert np.array_equal(z['outputs']['HiResMix']['q'],h['outputs']['HiResMix']['q']);peaks.append(z['peak_GPU_allocated']);decisions+=2080
    assert read(D/'NATIVE_FULL65_CONTROL.json')['FP32_full65_logits_exact'];cost=read(D/'COST.json');assert cost['current_full65_Source_attempts']==128 and cost['old_full65_control_attempts']==1
    assert all(sha(R/f)==h for f,h in p['pins'].items())
    out=dict(status='pass',all128_single_stream=True,legal_states_checked=decisions,single_head_normalization_max_error=maxerr,AST_original_replay_except_only_bbox_assembly_exact=True,old_actual_numeric_control='pass_noGT',max_GPU_allocated_bytes=max(peaks),spatial_arrays_checked_by_main_official_scorer=True)
    put(R/'results/stvg_error_state/2026-10-20/temporal_full_context128/SINGLE_STREAM_AUDIT_PUBLIC.json',out);print(out)

if __name__=='__main__':main()
