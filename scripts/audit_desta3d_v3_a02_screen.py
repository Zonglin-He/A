"""CPU-only independent saved coefficient/Adam/readout audit; no model forward."""
import sys,time,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a02_screen import A0,D,ARMS,load
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,total_prior

def main():
    import numpy as np,torch
    from vg_tta.desta3d_v3_a02_global_mean import GlobalMeanDirectionMixer,route
    from vg_tta.optimizer_checkpoint import validate_serialized_optimizer,restore_optimizer
    torch.set_num_threads(4);start=time.monotonic();cfg=read(D/'CONFIG.json');check_pins(read(D/'LOCK.json')['pins'])
    check_pins(read(D/'CACHE_REUSE.json')['files'])
    allresults={};maxerr=0;parameter_counts={};gradient={};losses={};audit={}
    for arm,(width,steps) in ARMS.items():
        dest=D/arm;assert read(dest/'COMPLETE.json')['seal_sha']==sha(dest/'SEAL.json')
        check_pins({str(dest/k):v for k,v in read(dest/'SEAL.json')['files'].items()})
        assert read(dest/'RECEIPT.json')['status']=='completed'
        report=read(dest/'REPORT.json');checkpoint=load(dest/'FINAL.pt');history=read(dest/'HISTORY.json')
        assert len(history)==steps==checkpoint['steps']
        assert torch.equal(checkpoint['mixer']['basis'],load(A0/'BASIS.pt'))
        validate_serialized_optimizer(checkpoint['optimizer'])
        m=GlobalMeanDirectionMixer(load(A0/'BASIS.pt'),hidden_dim=width);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=0)
        assert sum(p.numel() for p in m.parameters())==107648
        restore_optimizer(opt,checkpoint['optimizer']);assert len(opt.state)==8 and all(int(v['step'])==steps for v in opt.state.values())
        assert all(isinstance(k,int) for k in checkpoint['optimizer']['state'])
        assert all(all(torch.isfinite(v).all() for v in s.values() if torch.is_tensor(v)) for s in opt.state.values())
        rng=random.Random(cfg['seed']);order=[];maxhistory=0
        for step,row in enumerate(history,1):
            indices=[]
            for _ in range(4):
                if not order:order=list(range(128));rng.shuffle(order)
                indices.append(order.pop())
            assert row['step']==step and row['indices']==indices and set(row['counters'])=={step}
        changed=sum(not torch.equal(checkpoint['mixer'][k],v) for k,v in checkpoint['initial'].items() if k!='basis');assert changed==8
        if width==128:
            assert all(torch.equal(v,load(A0/'FINAL.pt')['initial'][k]) for k,v in checkpoint['initial'].items())
        if arm=='GMean-S2000':
            assert read(dest/'GMEAN_STEP200_IDENTITY.json')['exact']
            at200=load(dest/'STEP200_IDENTITY.pt');old=load(D/'GMean-S200/FINAL.pt')
            assert all(torch.equal(v,old['mixer'][k]) for k,v in at200['mixer'].items())
            for k,s in at200['optimizer']['state'].items():
                assert all(torch.equal(v,old['optimizer']['state'][k][n]) if torch.is_tensor(v) else v==old['optimizer']['state'][k][n] for n,v in s.items())
        results={}
        for split,n in [('train',128),('dev',64)]:
            values=[]
            for i in range(n):
                p=load(dest/'terminal_coefficients'/split/f'{i:04}.pt').numpy().astype(np.float64).ravel()
                t=load(A0/'cache'/split/f'{i:04}'/'CACHE.pt')['oracle_coeff256'].numpy().astype(np.float64).ravel()
                assert np.isfinite(p).all() and np.isfinite(t).all()
                den=np.sqrt(np.sum(p*p))*np.sqrt(np.sum(t*t));v=float(np.sum(p*t)/den) if den>0 else None
                saved=report['results'][split]['rows'][i]['cosine'];assert (v is None)==(saved is None)
                if v is not None:maxerr=max(maxerr,abs(v-saved));values.append(v)
            results[split]=dict(mean=float(np.mean(values)),median=float(np.median(values)),defined=len(values),count_gt_point1=sum(v>.1 for v in values),count_gt_point3=sum(v>.3 for v in values),terminal_mean_loss=float(1-np.mean(values)))
            for k in ('count_gt_point1','count_gt_point3'):assert results[split][k]==report['results'][split][k]
            for k in ('mean','median','terminal_mean_loss'):maxerr=max(maxerr,abs(results[split][k]-report['results'][split][k]))
        allresults[arm]=results;parameter_counts[arm]=sum(p.numel() for p in m.parameters())
        gradient[arm]=dict(min=min(x['gradient_norm'] for x in history),max=max(x['gradient_norm'] for x in history),last=history[-1]['gradient_norm'],clipped=sum(x['gradient_norm']>1 for x in history))
        assert gradient[arm]==report['gradient_norm'];losses[arm]=history[-1]['loss']
        audit[arm]=dict(steps=steps,occurrences=steps*4,changed_tensors=changed,counters_live_bound=True,integer_keys=True,basis_exact=True)
    assert maxerr<1e-10
    decision=route(allresults['GMean-S200'],allresults['GMean-S2000'])
    seconds={arm:dict(worker=read(OUT/('a02_'+arm)/'RECEIPT.json')['seconds'],wrapper=read(OUT/('a02_'+arm+'_wrapper')/'RECEIPT.json')['seconds']) for arm in ARMS}
    summary=dict(status='completed_independently_audited',results=allresults,parameter_counts=parameter_counts,gradient_norm=gradient,
      terminal_training_batch_loss=losses,decision=decision,native_run=False,fresh_read=False,
      actual_steps=2200,arm_gpu_seconds=seconds,cumulative_GPU_seconds=total_prior(),cap=None)
    baselines={}
    for name,path in [('Local-S200',A0/'DIRECTION_REPORT.json'),('Local-S2000',OUT/'a01_fitability_v1/W128-S2000/REPORT.json')]:
        owner=path.parent;sealname='FIT_SEAL.json' if name=='Local-S200' else 'SEAL.json'
        assert sha(path)==read(owner/sealname)['files'][path.name]
        rs=read(path)['results'];baselines[name]={}
        for split in ['train','dev']:
            vals=[r['cosine'] for r in rs[split]['rows'] if r['cosine'] is not None]
            baselines[name][split]=dict(mean=float(np.mean(vals)),median=float(np.median(vals)),count_gt_point1=sum(v>.1 for v in vals),count_gt_point3=sum(v>.3 for v in vals),defined=len(vals))
    summary['matched_baselines']=baselines
    write(D/'ROOT_READBACK.json',dict(status='passed',max_cosine_summary_error=maxerr,arms=audit,CPU_seconds=time.monotonic()-start))
    write(D/'PUBLIC_REPORT.json',summary);write(D/'COMPLETE.json',dict(status='completed',decision=decision,public_report_sha=sha(D/'PUBLIC_REPORT.json'),readback_sha=sha(D/'ROOT_READBACK.json')))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    from vg_tta.desta3d_v3_oracle_io import OUT
    main()
