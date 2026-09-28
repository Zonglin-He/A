"""Conditional Dev64 native stage; cannot run unless audited direction gate passes."""
import argparse,gc,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a0_fast_screen import D,OUT,PROTOCOL,load,setup,example,ADAPTER_SHA,labels_for
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,local_dependencies,allocation,tensor_sha

def prepare():
    assert read(D/'ROOT_DIRECTION_READBACK.json')['decision']=='native_dev64'
    dest=D/'native';assert not dest.exists();cfg=read(D/'CONFIG.json')
    paths=local_dependencies([Path(__file__),ROOT/'scripts/score_desta3d_v3_a0_native.py',PROTOCOL,D/'ROOT_DIRECTION_READBACK.json',D/'FINAL.pt',D/'FIT_SEAL.json',D/'CACHE_SEAL.json',D/'DEV64.json'])
    write(dest/'LOCK.json',dict(pins={str(p):sha(p) for p in paths}));write(dest/'REGISTRATION.json',dict(time=time.time(),source_GT_privilege=True,queries=64,new_predictions=64,reused_B1=64,optimizer_steps=0,fresh_read=False))

def run(name):
    import torch
    from vg_tta.desta3d_v3_gap_candidates import StateAwareDirectionMixer
    from vg_tta.desta3d_v3_a0_screen import coefficients,fixed_field
    from vg_tta.desta3d_v3_decomposition import shared_fields,norm
    from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
    from vg_tta.desta3d_v2_prediction_contract import validate_prediction
    from scripts.desta3d_v2_source_fit import prediction_record
    from scripts.desta3d_v2_reference_audit_cached_v3 import details
    from scripts.desta3d_v2_p0 import adapter_sha256
    dest=D/'native';check_pins(read(dest/'LOCK.json')['pins']);check_pins(read(D/'LOCK.json')['pins'])
    assert read(D/'ROOT_DIRECTION_READBACK.json')['decision']=='native_dev64'
    d=OUT/name;assert not d.exists();cfg={**read(D/'CONFIG.json'),'phase_seconds':900}
    write(d/'CONFIG.json',cfg);write(d/'LOCK.json',dict(pins={**read(D/'LOCK.json')['pins'],**read(dest/'LOCK.json')['pins']}))
    with allocation(d) as (cfg,guard):
        pr,model,adapter,unused=setup(cfg['seed']);unused.eval().requires_grad_(False)
        mixer=StateAwareDirectionMixer(load(D/'BASIS.pt'),radius=cfg['radius']).cuda().eval().requires_grad_(False);mixer.load_state_dict(load(D/'FINAL.pt')['mixer'])
        rows=read(D/'DEV64.json');labels=labels_for(rows,'dev')
        for i,row in enumerate(rows):
            guard();ep=dest/'episodes'/f'{i:04}';assert not ep.exists();ep.mkdir(parents=True)
            old=D/'cache/dev'/f'{i:04}';data=load(old/'CACHE.pt');inp=read(old/'INPUT.json');label=labels[row['key']]
            prompt,pre,fields,args=example(pr,model,adapter,row,label);stock=fields['visual_grid'].detach();support={k:tensor_sha(v) for k,v in fields.items() if isinstance(v,torch.Tensor)}
            assert pre==inp['preprocess'] and support==inp['support'] and abs(norm(stock)-data['stock_norm'])<1e-8
            for name,x in zip(['z','qT','qS','evidence8'],args[:-1]):assert torch.equal(x.cpu(),data[name]),name
            gpu={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in data.items()}
            with torch.no_grad():
                coeff=coefficients(mixer,gpu);assert torch.equal(coeff.cpu(),load(D/'terminal_coefficients/dev'/f'{i:04}.pt'))
                delta=fixed_field(mixer,coeff,data['stock_norm']);corrected=stock+delta
                with shared_fields(adapter,{'event':corrected,'spatial':corrected},['event','spatial'],allow_prefix=True) as calls:
                    result=decode_shared_reference_two_pass(model,pr,prompt,adapter,fields)
            pred=prediction_record(result,row,pre,ADAPTER_SHA);pred.update(arm='A0',readout=details(result),support=support,source_GT_privilege=True,GT_read=True,target_read=False,decoder_GT_prefix=False,optimizer_steps=0,
                injection=dict(calls=calls,same_field_both_passes=True,common_F_sha=tensor_sha(stock),corrected_F_sha=tensor_sha(corrected),relative_norm=norm(delta)/norm(stock)))
            assert abs(pred['injection']['relative_norm']-cfg['radius'])<2e-6;validate_prediction(pred,len(row['input']['frame_ids']))
            torch.save(pred,ep/'A0.pt');os.link(old/'B1.pt',ep/'B1.pt');write(ep/'INPUT.json',inp)
            assert adapter_sha256(adapter)==ADAPTER_SHA and all(p.grad is None and not p.requires_grad for mod in (model,adapter,mixer,unused) for p in mod.parameters())
            write(ep/'COMPLETE.json',dict(files={p.name:sha(p) for p in ep.iterdir() if p.is_file()},index=i,physical_replay_exact=True,coefficient_replay_exact=True))
            print('A0_NATIVE',i+1,64,flush=True)
            del data,inp,prompt,pre,fields,args,stock,gpu,coeff,delta,corrected,result,pred;gc.collect();torch.cuda.empty_cache()
        write(dest/'PREDICTIONS_SEAL.json',dict(files={str(p.relative_to(dest)):sha(p) for p in (dest/'episodes').rglob('*') if p.is_file()},predictions=128,queries=64))
        write(dest/'COMPLETE.json',dict(predictions=128,queries=64,seal_sha=sha(dest/'PREDICTIONS_SEAL.json')))
        write(d/'COMPLETE.json',dict(predictions=128,new_native=64,optimizer_steps=0,peak_GPU_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--name');a=p.parse_args();prepare() if a.action=='prepare' else run(a.name)
