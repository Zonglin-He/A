"""Independent NumPy full-field and native support audit. No model or CUDA.

First mode audits the fixed first completed episode, never chooses a sample.
All mode requires the full seal, and is mandatory before scoring.
"""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_oracle_mixer_gap import D,E,OLD,SEEDS,ARMS,load
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,tensor_sha

def audit(mode):
    import numpy as np,torch
    torch.set_num_threads(4);start=time.monotonic()
    dest=D/('ROOT_FIRST_RAW_READBACK.json' if mode=='first' else 'ROOT_ALL_RAW_READBACK.json')
    assert not dest.exists();check_pins(read(D/'LOCK.json')['pins'])
    if mode=='all':
        assert read(D/'COMPLETE.json')['seal_sha']==sha(D/'PREDICTIONS_SEAL.json')
        seal=read(D/'PREDICTIONS_SEAL.json')['files'];check_pins({str(D/p):h for p,h in seal.items()})
        assert set(seal)=={str(p.relative_to(D)) for p in (D/'episodes').rglob('*') if p.is_file()}
    basis=load(D/'BASIS.pt').numpy().astype(np.float64);rows=read(D/'INPUTS.json');cfg=read(D/'CONFIG.json')
    errors=[];field_errors=[];ce_errors=[];summaries=[]
    def close(x,y):
        err=abs(float(x)-float(y));errors.append(err);assert err<=1e-9+1e-8*abs(float(y)),(x,y,err)
    def dot(x,y):
        return sum(float(np.sum(x[s:s+128].astype(np.float64)*y[s:s+128].astype(np.float64))) for s in range(0,len(x),128))
    def norm(x):return dot(x,x)**.5
    def cos(x,y):
        n=norm(x)*norm(y);return dot(x,y)/n if n else None
    for i in range(1 if mode=='first' else len(rows)):
        ep=D/'episodes'/f'{i:04}';done=read(ep/'COMPLETE.json')
        check_pins({str(ep/p):h for p,h in done['files'].items()})
        assert done['index']==i and done['optimizer_steps']==0 and done['predictions']==4
        inp=read(ep/'INPUT.json');assert inp['old_input_sha']==sha(E/'episodes'/f'{i:04}'/'INPUT.json')
        for arm in ARMS[:-1]:assert sha(ep/(arm+'.pt'))==sha(E/'episodes'/f'{i:04}'/(arm+'.pt'))
        assert all(read(ep/'BASELINE_CHECK.json')['checks'].values())
        raw=load(ep/'RAW_DIRECTIONS.pt');trace=load(ep/'BASE_TRACE.pt');geo=read(ep/'GEOMETRY.json')
        assert raw['geometry']==geo and raw['stock_sha']==inp['support']['visual_grid']
        gs={k:v.numpy().reshape(-1,2560) for k,v in raw['gradients'].items()}
        ds={k:v.numpy().reshape(-1,2560) for k,v in raw['deltas'].items()}
        assert all(list(v.shape)==raw['stock_shape'] and v.dtype==torch.float32 and torch.isfinite(v).all() for part in ('gradients','deltas') for v in raw[part].values())
        for b,g in gs.items():close(norm(g),geo['gradient_norms'][b])
        c=cos(gs['event'],gs['spatial'])
        if c is None:assert geo['gradient_cosine'] is None
        else:close(c,geo['gradient_cosine'])
        fn=geo['stock_norm'];assert fn>0 and geo['radius']==cfg['radius']
        for arm,delta in ds.items():
            close(norm(delta),geo['delta_norms'][arm]);close(norm(delta)/(fn*cfg['radius']),geo['norm_over_cap'][arm])
            c=cos(delta,ds['oracle'])
            if c is None:assert geo['cosine_to_oracle'][arm] is None
            else:close(c,geo['cosine_to_oracle'][arm])
            for b,g in gs.items():close(-dot(g,delta),geo['descent_dot'][b][arm])
        # FP32 unit/add, FP64 projection, FP32 cast and scaling match the locked convention.
        balanced=np.zeros_like(gs['event'])
        for g in gs.values():
            n=norm(g)
            if n:balanced+=g*np.float32(1/n)
        projected=np.empty_like(balanced)
        for s in range(0,len(balanced),128):projected[s:s+128]=-(balanced[s:s+128].astype(np.float64)@basis@basis.T).astype(np.float32)
        n=norm(projected);expected=projected*np.float32(cfg['radius']*fn/n) if n else projected
        rel=norm(expected-ds['oracle'])/max(norm(expected),1e-30);field_errors.append(rel);assert rel<=2e-6,rel
        for arm in SEEDS:
            coef=raw['coefficients'][arm].numpy().reshape(-1,256);meta=geo['mixer'][arm]
            assert np.isfinite(coef).all() and np.max(np.abs(coef))<=1
            close(norm(coef)/(coef.size**.5),meta['coefficient_rms'])
            close(np.mean(np.abs(coef)>.99),meta['fraction_abs_above_099']);close(np.mean(np.abs(coef)>.999),meta['fraction_abs_above_0999'])
            close(np.mean(1-coef.astype(np.float64)**2),meta['mean_tanh_derivative'])
            err2=0.
            for s in range(0,len(coef),128):
                reconstructed=(coef[s:s+128]*np.float32(meta['scale'])).astype(np.float64)@basis.T
                diff=reconstructed-ds[arm][s:s+128];err2+=float(np.sum(diff*diff))
            rel=err2**.5/max(norm(ds[arm]),1e-30);field_errors.append(rel);assert rel<=2e-6
            assert abs(meta['coefficient_rms']-geo['norm_over_cap'][arm])<2e-6
        backwards=0
        for b,j,kind in [('event',0,'time'),('spatial',1,'coordinate')]:
            obj=raw['objectives'][b]
            if 'missing' in obj:
                assert not np.any(gs[b]);continue
            backwards+=1;fwd=read(ep/(b+'_FORWARD.json'));logits=trace['branches'][j]['logits'][kind]
            assert fwd['exact'] and fwd['native_sha']==fwd['replay_sha']==tensor_sha(logits)
            valid=obj['valid'].numpy();target=obj['targets'].numpy()[valid];x=logits.float().numpy()[valid].astype(np.float64)
            assert x.shape[-1]==obj['classes']==fwd['classes'] and len(target)==obj['actions']==fwd['actions']
            assert obj['classes']==(len(rows[i]['input']['frame_ids']) if b=='event' else 152775)
            mx=x.max(-1);ce=float(np.mean(mx+np.log(np.exp(x-mx[:,None]).sum(-1))-x[np.arange(len(x)),target]))
            err=abs(ce-obj['CE']);ce_errors.append(err);assert err<5e-6,(b,ce,obj['CE'])
        assert backwards==done['backwards']
        oracle=load(ep/'oracle.pt');assert oracle['GT_read'] and oracle['source_GT_oracle'] and not oracle['target_read']
        assert oracle['support']==inp['support'] and oracle['preprocess']==inp['preprocess']
        assert oracle['injection']['common_F_sha']==raw['stock_sha'] and oracle['injection']['same_field_both_passes']
        close(oracle['injection']['relative_norm'],geo['delta_norms']['oracle']/fn)
        summaries.append(dict(index=i,backwards=backwards,missing={b:o['missing'] for b,o in raw['objectives'].items() if 'missing' in o},geometry=geo))
        print('RAW_AUDITED',i+1,flush=True)
    write(dest,dict(status='passed',episodes=len(summaries),backwards=sum(x['backwards'] for x in summaries),
        norm_dot_max_abs=max(errors,default=0),field_relative_L2_max=max(field_errors,default=0),CE_FP64_max_abs=max(ce_errors,default=0),
        cases=summaries,lock_sha=sha(D/'LOCK.json'),auditor_sha=sha(Path(__file__)),CPU_seconds=time.monotonic()-start,
        limits='Stock-F norm comes from actual worker (full stock field not duplicated); full gradients/deltas/coefficients independently reduced. No outcome selection or GPU.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['first','all']);a=p.parse_args()
    try:audit(a.mode)
    except BaseException as e:
        f=D/('RAW_AUDIT_'+a.mode+'_FAILURE.json')
        if not f.exists():write(f,dict(error=repr(e),traceback=traceback.format_exc()))
        raise
