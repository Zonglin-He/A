"""Independent CPU signal loss/gradient geometry/GT projection readback. No model or inference."""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a05_signal import D,A0,A3,GAP,SIGNALS,BRANCHES,load
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,total_prior

def main(mode):
    import numpy as np,torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);start=time.monotonic();cfg=read(D/'CONFIG.json');rows=read(D/'INPUTS.json');ids=[0] if mode=='first' else list(range(16))
    check_pins(read(D/'LOCK.json')['pins']);check_pins(read(D/'REFERENCE_SEAL.json')['files'])
    qc=load(D/'QC.pt').double().numpy();report={s:[] for s in SIGNALS};max_loss=chain=rawerr=projectionerr=0.;backwards=0
    if mode=='all':
        assert read(D/'GPU_COMPLETE.json')['seal_sha']==sha(D/'SIGNALS_SEAL.json');check_pins({str(D/k):v for k,v in read(D/'SIGNALS_SEAL.json')['files'].items()})
    norm=lambda a:float(np.sqrt(np.sum(a*a)))
    unit=lambda a:a/norm(a) if norm(a)>0 else np.zeros_like(a)
    def logs(x):
        m=x.max(-1,keepdims=True);return x-m-np.log(np.exp(x-m).sum(-1,keepdims=True))
    with threadpool_limits(limits=4,user_api='blas'):
        for i in ids:
            row=rows[i];ep=D/'episodes'/f'{i:02}';done=read(ep/'COMPLETE.json');check_pins({str(ep/k):v for k,v in done['files'].items()});backwards+=done['backwards']
            ref=load(D/'references'/f'{i:02}.pt');raw=load(ep/'RAW_SIGNALS.pt');meta=read(ep/'SIGNALS.json');inp=read(ep/'INPUT.json')
            gap=GAP/'episodes'/f"{row['gap_index']:04}";oldraw=load(gap/'RAW_DIRECTIONS.pt');trace=load(gap/'BASE_TRACE.pt')
            assert sha(gap/'RAW_DIRECTIONS.pt')==ref['original_GT_gradient_sha'] and sha(gap/'BASE_TRACE.pt')==inp['native_trace_sha']
            assert inp['stock_sha']==ref['stock_sha'] and inp['stock_norm']==ref['stock_norm']
            assert inp['physical_replay_exact'] and not inp['source_GT_loaded_in_worker'] and done['frozen_scope'] and done['optimizer_steps']==0
            a3=A3/'native/episodes'/f"{row['dev64_index']:04}"/'Shared-R16_FACTORS.npy';assert sha(a3)==ref['oracle_factors_sha']
            oracle=unit(np.load(a3).astype(np.float64));assert np.max(abs(oracle-ref['oracle_direction'].numpy()))<1e-12
            gt={};available={}
            for branch,_,_ in BRANCHES:
                # Independent Torch FP64 matmul checks original NumPy preparation.
                projected=(oldraw['gradients'][branch].double()@torch.from_numpy(qc)).numpy()
                saved=ref['GT_gradients_C'][branch].numpy();err=norm(projected-saved)/max(norm(saved),1e-300)
                projectionerr=max(projectionerr,err);assert err<1e-10
                gt[branch]=projected;available[branch]='missing' not in oldraw['objectives'][branch]
                assert available[branch]==ref['GT_available'][branch]
            for signal in SIGNALS:
                gs={};losses={};missing={}
                for branch,j,kind in BRANCHES:
                    g=raw[signal]['gradients'][branch].numpy().astype(np.float64);assert np.isfinite(g).all() and list(g.shape)==[*ref['stock_shape'][:-1],16]
                    gs[branch]=g;v=meta[signal][branch];has=j<len(trace['branches']) and kind in trace['branches'][j]['logits'];missing[branch]=not has
                    if not has:assert 'missing' in v and norm(g)==0;continue
                    expected=trace['branches'][j]['logits'][kind];teacher=expected.numpy().astype(np.float64)
                    student=load(ep/(signal+'_'+branch+'_LOGITS.pt')).numpy().astype(np.float64) if signal=='U-Consistency' else teacher
                    assert student.shape==teacher.shape and student.shape[-1]==(ref['stock_shape'][1] if branch=='event' else 152775)
                    if signal=='U-Entropy':assert v['observed_native_exact']
                    lp=logs(student)
                    if signal=='U-Consistency':lt=logs(teacher);loss=float((np.exp(lt)*(lt-lp)).sum(-1).mean())
                    else:loss=float(-(np.exp(lp)*lp).sum(-1).mean())
                    max_loss=max(max_loss,abs(loss-v['loss']));assert abs(loss-v['loss'])<cfg['loss_absolute_tolerance']
                    rawerr=max(rawerr,abs(norm(g)-v['gradient_norm']));losses[branch]=loss
                    if i==0:
                        gf=load(ep/(signal+'_'+branch+'_GF.pt')).numpy().astype(np.float64);recon=gf@qc
                        err=norm(recon-g)/max(norm(g),1e-300);chain=max(chain,err);assert err<cfg['chain_relative_tolerance']
                direction=-unit(unit(gs['event'])+unit(gs['spatial']));saved=raw[signal]['direction'].numpy();rawerr=max(rawerr,float(np.max(abs(direction-saved))));assert np.max(abs(direction-saved))<1e-10
                defined=norm(direction)>0 and norm(oracle)>0;cos=float(np.sum(direction*oracle)) if defined else None
                dots={b:float(-np.sum(gt[b]*direction)) if available[b] else None for b in gt}
                report[signal].append(dict(index=i,cosine=cos,gate_cosine=cos if defined else 0.,direction_defined=defined,
                  local_descent=dots,GT_available=available,native_missing=missing,gradient_norms={b:norm(g) for b,g in gs.items()},losses=losses,
                  branch_gradient_cosine=float(np.sum(unit(gs['event'])*unit(gs['spatial']))) if all(norm(g)>0 for g in gs.values()) else None))
                print('AUDITED',i,signal,'cos',cos,'descent',dots,flush=True)
    result=dict(status='passed',mode=mode,queries=len(ids),actual_backwards=backwards,max_scalar_loss_error=max_loss,max_first_chain_relative_error=chain,
      max_raw_norm_direction_error=rawerr,max_reference_projection_relative_error=projectionerr,CPU_seconds=time.monotonic()-start,results=report)
    assert rawerr<1e-9
    if mode=='all':
        aggregates={}
        for signal,rr in report.items():
            vals=[x['gate_cosine'] for x in rr];available={b:sum(x['GT_available'][b] for x in rr) for b,_,_ in BRANCHES}
            positive={b:sum(x['local_descent'][b] is not None and x['local_descent'][b]>0 for x in rr) for b in available}
            passed=np.median(vals)>=.10 and all(available[b]>0 and positive[b]/available[b]>=.65 for b in available)
            aggregates[signal]=dict(cosine_mean=float(np.mean(vals)),cosine_median=float(np.median(vals)),cosine_min=float(min(vals)),cosine_max=float(max(vals)),
              defined=sum(x['direction_defined'] for x in rr),count_positive=sum(v>0 for v in vals),positive_GT_descent=positive,available_GT=available,
              positive_GT_descent_fraction={b:positive[b]/available[b] if available[b] else None for b in available},passed=bool(passed))
        result['aggregates']=aggregates;result['passing_signals']=[s for s,v in aggregates.items() if v['passed']]
        result['decision']='conditional_Dev16_one_step_native' if result['passing_signals'] else 'stop_internal_objective_tuning_next_expert_pseudo_gradient_qualification'
    write(D/('ROOT_FIRST_READBACK.json' if mode=='first' else 'ROOT_SIGNAL_READBACK.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['first','all']);a=p.parse_args()
    try:main(a.mode)
    except BaseException as e:
        dest=D/('CPU_'+a.mode+'_FAILURE.json')
        if not dest.exists():write(dest,dict(error=repr(e),traceback=traceback.format_exc()))
        raise
