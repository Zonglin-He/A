"""Independent NumPy A0.4 terminal coefficient audit and strict Adam/sample restore."""
import argparse,sys,time,random,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a04_screen import A0,A3,D,load
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,total_prior

def main(steps):
    import numpy as np,torch
    from threadpoolctl import threadpool_limits
    from vg_tta.desta3d_v3_a04_factorized import FactorizedDirectionMixer,route
    from vg_tta.optimizer_checkpoint import validate_serialized_optimizer,restore_optimizer
    start=time.monotonic();torch.set_num_threads(4);dest=D/f'S{steps}';assert not (dest/'ROOT_READBACK.json').exists()
    cfg=read(D/'CONFIG.json');check_pins(read(dest/'LOCK.json')['pins']);check_pins(read(D/'CACHE_REUSE.json')['files']);check_pins(read(D/'TARGETS_SEAL.json')['files'])
    assert read(dest/'COMPLETE.json')['seal_sha']==sha(dest/'SEAL.json');check_pins({str(dest/k):v for k,v in read(dest/'SEAL.json')['files'].items()})
    assert read(dest/'RECEIPT.json')['status']=='completed'
    b=np.load(D/'B16.npy');assert np.array_equal(b,np.load(A3/'TRAIN_BASIS.npz')['basis'][:,:16]);assert np.max(abs(b.T@b-np.eye(16)))<1e-12
    ck=load(dest/'FINAL.pt');history=read(dest/'HISTORY.json');report=read(dest/'REPORT.json');assert len(history)==steps==ck['steps']
    validate_serialized_optimizer(ck['optimizer']);torch.manual_seed(cfg['seed']);m=FactorizedDirectionMixer(load(A0/'BASIS.pt'),torch.from_numpy(b))
    assert sum(p.numel() for p in m.parameters())==76688
    assert all(torch.equal(v,ck['initial'][k]) for k,v in m.state_dict().items())
    old=load(A0/'FINAL.pt')['initial'];assert all(torch.equal(v,old[k]) for k,v in ck['initial'].items() if k.startswith(('input.','local.','mix.')))
    m.load_state_dict(ck['mixer']);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=0);restore_optimizer(opt,ck['optimizer'])
    assert len(opt.state)==8 and all(int(v['step'])==steps for v in opt.state.values())
    assert all(isinstance(k,int) for k in ck['optimizer']['state'])
    assert all(p in opt.state for p in m.parameters())
    assert all(all(torch.isfinite(v).all() for v in s.values() if torch.is_tensor(v)) for s in opt.state.values())
    for name in ('union','channel_basis','basis'):assert torch.equal(ck['mixer'][name],ck['initial'][name])
    assert sum(not torch.equal(ck['mixer'][k],v) for k,v in ck['initial'].items())==8
    rng=random.Random(cfg['seed']);order=[]
    for step,row in enumerate(history,1):
        ids=[]
        for _ in range(4):
            if not order:order=list(range(128));rng.shuffle(order)
            ids.append(order.pop())
        assert row['step']==step and row['indices']==ids and set(row['counters'])=={step}
    assert rng.getstate()==ck['sample_rng'] and order==ck['sample_order']
    if steps==2000:
        assert read(D/'S200/ROOT_READBACK.json')['decision']=='continue_same_trajectory_to_2000'
        assert history[:200]==read(D/'S200/HISTORY.json')
        r=read(dest/'RESUME_READBACK.json');assert r['source_sha']==sha(D/'S200/FINAL.pt') and r['actual_start_counter']==200
        assert all(torch.equal(v,load(D/'S200/FINAL.pt')['initial'][k]) for k,v in ck['initial'].items())
    maxerr=0.;projection_error=0.;identity_error=0.;results={};cos=lambda x,y:float(np.sum(x*y)/np.sqrt(np.sum(x*x))/np.sqrt(np.sum(y*y)))
    with threadpool_limits(limits=4,user_api='blas'):
        for split,n in [('train',128),('dev',64)]:
            rows=[]
            for i in range(n):
                a=load(A0/'cache'/split/f'{i:04}'/'CACHE.pt')['oracle_coeff256'].numpy().astype(np.float64)
                p=load(dest/'terminal_coefficients'/split/f'{i:04}.pt').numpy().astype(np.float64);target=a@b;rounded=np.load(D/'targets16'/split/f'{i:04}.npy').astype(np.float64)
                assert p.shape==target.shape and np.isfinite(p).all() and np.linalg.norm(p)>0 and np.linalg.norm(target)>0
                projection_error=max(projection_error,float(np.linalg.norm(target-rounded)/np.linalg.norm(target)))
                c=cos(p,target);f=cos(p@b.T,a);training=cos(p,rounded);energy=float(np.sum(target*target)/np.sum(a*a))
                identity_error=max(identity_error,abs(f-c*np.sqrt(energy)))
                saved=report['results'][split]['rows'][i]
                for v,key in [(c,'cosine'),(f,'full_oracle_cosine'),(training,'training_cosine'),(energy,'projection_energy')]:maxerr=max(maxerr,abs(v-saved[key]))
                rows.append(dict(index=i,cosine=c,full_oracle_cosine=f,projection_energy=energy,training_cosine=training))
            vals=np.array([r['cosine'] for r in rows]);full=np.array([r['full_oracle_cosine'] for r in rows])
            rs=dict(mean=float(vals.mean()),median=float(np.median(vals)),count_gt_point1=int((vals>.1).sum()),count_gt_point3=int((vals>.3).sum()),defined=n,
              terminal_mean_loss=float(1-vals.mean()),rounded_training_target_mean_loss=float(1-np.mean([r['training_cosine'] for r in rows])),full_oracle_mean=float(full.mean()),full_oracle_median=float(np.median(full)))
            for k,v in rs.items():maxerr=max(maxerr,abs(v-report['results'][split][k]))
            results[split]=rs
    assert maxerr<1e-10 and projection_error<2e-6 and identity_error<1e-10
    grad=[h['gradient_norm'] for h in history];gn=dict(min=min(grad),max=max(grad),last=grad[-1],clipped=sum(v>1 for v in grad));assert gn==report['gradient_norm']
    decision=route(results['train']['median'],results['dev']['median'],steps)
    audit=dict(status='passed',steps=steps,decision=decision,coefficient_fields=192,max_numpy_scalar_error=maxerr,
      max_target_projection_rounding_relative_L2=projection_error,max_cosine_factorization_identity_error=identity_error,
      frozen_basis_and_hidden_initialization_exact=True,changed_parameter_tensors=8,integer_Adam_keys=True,live_Parameter_binding=True,
      actual_counters=steps,sample_order_and_RNG_exact=True,continuation_from_step200=(steps==2000),CPU_seconds=time.monotonic()-start)
    write(dest/'ROOT_READBACK.json',audit)
    summary=dict(status='completed_independently_audited',steps=steps,results=results,decision=decision,trainable_parameters=76688,
      gradient_norm=gn,terminal_training_batch_loss=history[-1]['loss'],new_steps=report['new_steps'],PTD_loaded=False,native_predictions=0,fresh_read=False,
      cumulative_GPU_seconds=total_prior(),cap=None,scope='source GT cached direction screen; repeatedly exposed Dev64; not target TTA or learned native utility')
    write(dest/'PUBLIC_REPORT.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('steps',type=int,choices=[200,2000]);main(p.parse_args().steps)
