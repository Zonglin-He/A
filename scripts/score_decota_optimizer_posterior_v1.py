"""Independent NumPy objectives/optimizer/probability audit after global seal."""
import sys,time,collections,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_optimizer_posterior_common_v1 import *
from scripts.decota_public_result_io_v1 import read
from scripts.score_audit_tastvg_decota_c1_same_domain_v1 import loss_numpy,flat,check_state
from vg_tta.decota_optimizer_posterior_r1_v1 import ARMS
import numpy as np,torch
from scipy.special import logsumexp

def energy(boxes,ex,support):
    terms=[]
    for (_,pos),ob in sorted(ex['observations'].items()):
        q=ob['probe']
        if support in ['admit','top1'] and not q['accepted']:continue
        b=np.asarray(q['boxes'],float).reshape(-1,4);s=np.asarray(q['target_scores'],float);valid=(b[:,2:]>0).all(1);b,s=b[valid],s[valid]
        if not len(s):continue
        if support=='top1':k=s.argmax();b,s=b[k:k+1],s[k:k+1]
        logw=s-logsumexp(s);v=np.asarray(boxes[pos],float)
        p=np.r_[v[:2]-v[2:]/2,v[:2]+v[2:]/2];e=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]
        it=np.maximum(np.minimum(p[2:],e[:,2:])-np.maximum(p[:2],e[:,:2]),0).prod(1)
        un=np.maximum(np.maximum(p[2:]-p[:2],0).prod()+np.maximum(e[:,2:]-e[:,:2],0).prod(1)-it,1e-7)
        terms.append(-logsumexp(logw+it/un))
    return float(np.mean(terms)) if terms else 0.

def stats(rows,fields):
    group=collections.defaultdict(list)
    for r in rows:group[r['source_id']].append([r[k] for k in fields])
    if not group:return dict(cells=0,sources=0,metrics={})
    ids=sorted(group);x=np.array([np.mean(group[i],0) for i in ids]);rng=np.random.default_rng(20261004)
    boot=np.concatenate([x[rng.integers(len(x),size=(100,len(x)))].mean(1) for _ in range(100)])
    ci=np.quantile(boot,[.025,.975],axis=0)
    return dict(cells=len(rows),sources=len(ids),metrics={k:dict(mean=float(x[:,j].mean()),ci95=ci[:,j].tolist(),
        source_values={str(i):float(v) for i,v in zip(ids,x[:,j])}) for j,k in enumerate(fields)},
        bootstrap_draws=10000,seed=20261004)

def audit_sgd_rounding(old,new,gradient,lr):
    old=np.asarray(old,dtype=float);new=np.asarray(new,dtype=float);g=np.asarray(gradient,dtype=float)
    product=-float(np.float32(lr))*g;exact=old+product
    bound=np.abs(np.spacing(product.astype(np.float32))).astype(float)+np.abs(np.spacing(exact.astype(np.float32))).astype(float)+1e-30
    assert np.all(np.abs(new-exact)<=bound),'float32 addition exceeds per-element ULP bound'

SF=['v','t','s','vs_frozen_v','vs_before_v','vs_adam_v','gross_gain','gross_loss']
TF=['v','t','s','vs_native_v','vs_native_t','vs_native_fp64_v','vs_hard_v','recall','precision','span_fraction']

def summary(rows,fields,arms,temporal=False):
    result={}
    for ds in DATASETS:
        result[ds]={}
        for split in ['search','confirm']:
            result[ds][split]={}
            for arm in arms:
                rr=[r for r in rows if r['dataset']==ds and r['split']==split and r['arm']==arm]
                sub={'corruption':[r for r in rr if r['condition']!='clean'],'clean':[r for r in rr if r['condition']=='clean']}
                for c in sorted({r['condition'] for r in rr}):sub['condition:'+c]=[r for r in rr if r['condition']==c]
                for o in ['order1','order2']:sub[o]=[r for r in rr if r['order']==o and r['condition']!='clean']
                if temporal:
                    for cat in ['short','medium','long']:sub[cat]=[r for r in rr if r['duration_group']==cat and r['condition']!='clean']
                    for flag in [True,False]:sub['native_high' if flag else 'native_low']=[r for r in rr if r['native_high']==flag and r['condition']!='clean']
                sums={k:stats(v,fields) for k,v in sub.items()}
                for k,v in sub.items():
                    delta='vs_native_v' if temporal else 'vs_before_v'
                    sums[k]['tails']=dict(harm_gt5pp=sum(r[delta]<-.05 for r in v),harm_gt20pp=sum(r[delta]<-.2 for r in v),
                        gain_gt5pp=sum(r[delta]>.05 for r in v),correctness={str(th):dict(
                            destroyed=sum(r['v']-r[delta]>=th and r['v']<th for r in v),
                            recovered=sum(r['v']-r[delta]<th and r['v']>=th for r in v)) for th in [.3,.5]})
                result[ds][split][arm]=sums
    return result

def run():
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    from scripts import tastvg_decota_c1_common_v1 as c1
    from methods.decota_final_simplified_v1.tensors import state_hash
    cpu=read(BASE/'CPU_RUNTIME_LOCK.json');pins=dict(cpu['pins'])
    for f in sorted((BASE/'cpu_revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    for f,h in pins.items():assert sha(ROOT/f)==h,f
    tick=time.time();torch.set_num_threads(4);b=verify_seal()
    exposure=BASE/'GT_EXPOSURE.json'
    if exposure.exists():assert read(exposure)['barrier_sha256']==sha(BASE/'GLOBAL_PREDICTION_BARRIER.json') and read(exposure)['global_seal_precedes_GT']
    else:write(exposure,dict(time=tick,barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),global_seal_precedes_GT=b['time']<tick,GT_online=False))
    sr=[];tr=[];diagnostics=[];checks=0;tempaudit=[]
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json');labels={s:read(prior.POOL/ds/f'GT_LABELS_{s}.json') for s in p['splits']}
        for split,sp in p['splits'].items():
            for cond in p['conditions']:
                for order,seq in sp['orders'].items():
                    for at,parent in enumerate(seq):
                        row=p['rows'][parent];truth={int(k):v for k,v in labels[split][str(parent)]['truth'].items()};span=labels[split][str(parent)]['span']
                        sf=BASE/ds/'spatial'/split/cond/order/f'{at:05}.pt';x=checked(sf);ref=prior.checked(ROOT/x['prestate_path']);ex=c1.checked(ROOT/x['evidence_path'])['expert']
                        assert sha(ROOT/x['prestate_path'])==x['prestate_sha256'] and sha(ROOT/x['evidence_path'])==x['evidence_sha256']
                        checks+=check_state(x['initial'],ref['initial']);assert torch.equal(x['before'],ref['before']) and not x['GT_read']
                        iv=x['native']['physical_interval'];frozen=DenseTube(x['native']['boxes'].numpy(),row,truth,span,clip=ds=='hc2').score(iv)
                        bef=DenseTube(x['before'].numpy(),row,truth,span,clip=ds=='hc2').score(iv)
                        adam=DenseTube(x['fits']['all_adam']['final'].numpy(),row,truth,span,clip=ds=='hc2').score(iv)
                        for arm,z in x['fits'].items():
                            obj,mode=arm.split('_',1);assert not z['GT_used'] and z['parameter_count']==1792
                            checks+=check_state(z['initial'],x['initial']);names=list(z['initial']);selected=min(range(len(z['path'])),key=lambda j:z['path'][j]['loss'])
                            if mode.endswith('authority'):
                                authorities=[]
                                for ob in ex['observations'].values():
                                    q=ob['probe'];bb=np.asarray(q['boxes'],float).reshape(-1,4);sc=np.asarray(q['target_scores'],float)[(bb[:,2:]>0).all(1)]
                                    if len(sc):
                                        logw=sc-logsumexp(sc);authorities.append(1. if len(sc)==1 else 1+float((np.exp(logw)*logw).sum())/np.log(len(sc)))
                                auth=float(np.mean(authorities)) if authorities else 0.
                                assert abs(auth-z['authority'])<5e-7;checks+=1
                            assert selected==z['selected_step'];proposal=z['path'][selected]['state'];checks+=check_state(z['proposal_state'],proposal)
                            if mode=='post_authority':
                                used={n:x['initial'][n]+z['authority']*(proposal[n]-x['initial'][n]) for n in x['initial']};checks+=check_state(used,z['state'])
                            else:checks+=check_state(proposal,z['state']);assert torch.equal(z['final'],z['path'][selected]['boxes'])
                            m=np.zeros(1792);v=np.zeros(1792);pathv=[];gn=[];pn=[];bn=[];un=[];arith=0
                            for j,h in enumerate(z['path']):
                                actual=loss_numpy(h['boxes'].numpy(),ex['anchors']['single4']) if obj=='direct' else energy(h['boxes'].numpy(),ex,obj)
                                assert abs(actual-h['loss'])<(2e-5 if obj=='direct' else 2e-6),(arm,actual,h['loss']);checks+=1
                                pathv.append(DenseTube(h['boxes'].numpy(),row,truth,span,clip=ds=='hc2').score(iv)['v'])
                                bn.append(float((h['boxes']-x['before']).abs().mean()));pn.append(float((flat(h['state'],names)-flat(x['initial'],names)).norm()))
                                if 'update' not in h:continue
                                u=h['update'];g=u['gradient'].numpy().astype(float);assert np.isfinite(g).all()
                                if mode=='sgd':expected=-z['lr']*g
                                else:
                                    m=.9*m+.1*g;v=.999*v+.001*g*g
                                    expected=-z['lr']*(m/(1-.9**(j+1)))/(np.sqrt(v/(1-.999**(j+1)))+1e-8)
                                err=float(np.max(np.abs(expected-u['raw'].numpy())));arith=max(arith,err)
                                if mode=='sgd':
                                    # Audit the actual float32 parameter addition, including cancellation.
                                    oldvec=flat(h['state'],names).numpy().astype(float);newvec=flat(z['path'][j+1]['state'],names).numpy().astype(float)
                                    audit_sgd_rounding(oldvec,newvec,g,z['lr'])
                                else:assert err<3e-6,(arm,err)
                                assert torch.equal(flat(z['path'][j+1]['state'],names)-flat(h['state'],names),u['raw']);checks+=1792*3
                                gn.append(float(np.linalg.norm(g)));un.append(float(u['raw'].norm()))
                            fast=DenseTube(z['final'].numpy(),row,truth,span,clip=ds=='hc2').score(iv);off=official(z['final'].numpy(),row,truth,span,iv,ds)
                            assert max(abs(fast[k]-off[k]) for k in fast)<2e-12;assert off['t']==frozen['t'];checks+=4
                            rec=dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,arm=arm,**off,
                                vs_frozen_v=off['v']-frozen['v'],vs_before_v=off['v']-bef['v'],vs_adam_v=off['v']-adam['v'],
                                gross_gain=max(off['v']-bef['v'],0),gross_loss=max(bef['v']-off['v'],0),
                                prestate_sha256=state_hash(x['initial']),payload_sha256=sha(sf))
                            obsids={row['frame_ids'][pos] for _,pos in ex['observations']}
                            dt0=DenseTube(x['before'].numpy(),row,truth,span,clip=ds=='hc2');dt1=DenseTube(z['final'].numpy(),row,truth,span,clip=ds=='hc2')
                            omask=np.isin(dt0.fids,list(obsids));observed_delta=float((dt1.iou[omask]-dt0.iou[omask]).mean()) if omask.any() else None
                            unobserved_delta=float((dt1.iou[~omask]-dt0.iou[~omask]).mean()) if (~omask).any() else None
                            used_loss=loss_numpy(z['final'].numpy(),ex['anchors']['single4']) if obj=='direct' else energy(z['final'].numpy(),ex,obj)
                            sr.append(rec);diagnostics.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,arm=arm,
                                own_objective_losses=[h['loss'] for h in z['path']],posthoc_GT_vIoU=pathv,selected_step=selected,GT_selected_step=False,
                                gradient_norms=gn,step_parameter_displacements=un,parameter_displacements=pn,functional_displacements=bn,
                                used_functional_displacement=float((z['final']-x['before']).abs().mean()),authority=z['authority'],lr=z['lr'],
                                used_loss=used_loss,max_optimizer_arithmetic_error=arith,
                                proxy_improved_GT_harmed=off['v']<bef['v'] and used_loss<z['path'][0]['loss'],
                                proposal_proxy_improved_GT_harmed=pathv[selected]<pathv[0] and z['path'][selected]['loss']<z['path'][0]['loss'],
                                used_functional_exceeds_full_proposal=float((z['final']-x['before']).abs().mean())>bn[selected]+1e-12,
                                reused=arm=='all_adam',gradient_calls=z['gradient_calls'],support_frames=len(z.get('frame_metadata',[])),
                                observed_GT_frames=int(omask.sum()),unobserved_GT_frames=int((~omask).sum()),
                                observed_GT_IoU_delta=observed_delta,unobserved_GT_IoU_delta=unobserved_delta))
                        tf=BASE/ds/'temporal'/cond/f'{parent:05}.pt';t=checked(tf);old=c1.checked(c1.BASE/ds/'temporal'/cond/f'{parent:05}.pt')
                        # Independently reconstruct span prior, action cost and centre-preserving posterior.
                        if order=='order1':
                            data,_=c1.cache(ds,row['pool_parent'],cond)
                            for o,ev,z,record in zip(t['offsets'],old['evidence']['offsets'],data['prediction']['logits'],data['records']):
                                zz=z.reshape(-1,2).numpy().astype(float);ii,jj=np.triu_indices(len(zz),1);lp=zz[ii,0]+zz[jj,1];lp-=logsumexp(lp)
                                u=ev['standardized_logits'].numpy();w=ev['omega'].numpy();not_a=-np.logaddexp(0,u);cp=np.r_[0,np.cumsum(w*u)]
                                cost=-(w*not_a).sum()-(cp[jj+1]-cp[ii]);assert np.allclose(cost,ev['cost'].numpy(),atol=2e-12,rtol=0)
                                assert np.allclose(lp,o['logp0'],atol=2e-12,rtol=0)
                                lf=lp-cost;lf-=logsumexp(lf);assert np.allclose(lf,o['logfull'],atol=2e-12,rtol=0)
                                ids=np.asarray(record['frame_ids']);centres=ids[ii]+ids[jj]+1;le=np.empty_like(lp)
                                for c in np.unique(centres):
                                    a=centres==c;le[a]=logsumexp(lp[a])+lp[a]-cost[a]-logsumexp(lp[a]-cost[a])
                                assert np.allclose(le,o['logextent'],atol=2e-12,rtol=0);checks+=len(lp)*4
                            tempaudit.append(dict(dataset=ds,condition=cond,source_id=parent,offsets=[{k:v for k,v in o.items() if k in ['centre_marginal_max_error','centre_groups','singleton_groups','native_KL_full','native_KL_extent','beta']} for o in t['offsets']]))
                        native_before=DenseTube(x['before'].numpy(),row,truth,span,clip=ds=='hc2').score(t['intervals']['native'])
                        hard_before=DenseTube(x['before'].numpy(),row,truth,span,clip=ds=='hc2').score(t['intervals']['hard'])
                        # Matched numerical readout control: beta0 in the same FP64 policy.
                        raw0=[]
                        for oo,ev in zip(t['offsets'],old['evidence']['offsets']):
                            k=int(np.argmax(oo['logp0']));ii,jj=oo['ij'][:,k]
                            # Use exact native physical frame IDs, not midpoint cell edges.
                            ri=row['frame_ids'][len(raw0)::2];raw0.append([ri[ii],ri[jj]+1])
                        iv64=[min(a[0] for a in raw0),max(a[1] for a in raw0)]
                        native64=DenseTube(x['before'].numpy(),row,truth,span,clip=ds=='hc2').score(iv64)
                        cliplen=row['frame_ids'][-1]+1-row['frame_ids'][0];frac=(span[1]-span[0])/cliplen
                        for arm,interval in t['intervals'].items():
                            metrics=official(x['before'].numpy(),row,truth,span,interval,ds);dense=DenseTube(x['before'].numpy(),row,truth,span,clip=ds=='hc2').score(interval)
                            assert max(abs(metrics[k]-dense[k]) for k in dense)<2e-12
                            overlap=max(0,min(interval[1],span[1])-max(interval[0],span[0]));checks+=3
                            tr.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,arm=arm,**metrics,
                                vs_native_v=metrics['v']-native_before['v'],vs_native_t=metrics['t']-native_before['t'],vs_native_fp64_v=metrics['v']-native64['v'],
                                native_fp64_changes_original_readout=iv64!=t['intervals']['native'],vs_hard_v=metrics['v']-hard_before['v'],
                                recall=overlap/(span[1]-span[0]),precision=overlap/(interval[1]-interval[0]),span_fraction=(interval[1]-interval[0])/cliplen,
                                duration_group='short' if frac<1/3 else 'medium' if frac<2/3 else 'long',native_high=native_before['t']>=.5))
                    print('R1_CPU',ds,split,cond,order,len(sr),checks,flush=True)
    assert len(sr)==11520 and len(tr)==5760 and len(tempaudit)==576
    ss=summary(sr,SF,ARMS);ts=summary(tr,TF,['native','hard','full','extent','pm'],True)
    write(PUB/'SPATIAL_ROWS.json',sr);write(PUB/'TEMPORAL_ROWS.json',tr);write(PUB/'SPATIAL_DIAGNOSTICS.json',diagnostics)
    write(PUB/'SPATIAL_SUMMARY.json',ss);write(PUB/'TEMPORAL_SUMMARY.json',ts);write(PUB/'TEMPORAL_PROBABILITY_AUDIT.json',tempaudit)
    write(PUB/'CALIBRATION.json',{ds:read(BASE/ds/'CALIBRATION.json') for ds in DATASETS})
    write(PUB/'ROOT_AUDIT.json',dict(status='pass',checks=checks,spatial_arrivals=1152,spatial_arm_cells=11520,temporal_arm_cells=5760,
        GT_after_global_seal=True,all_same_P1_prestates=True,independent_loss_optimizer_and_probability=True,official_dense_agreement=True,time=time.time(),seconds=time.time()-tick))
    temporal_pass={a:all(ts[ds][sp][a]['corruption']['metrics'][delta]['ci95'][0]>0 for delta in ['vs_native_v','vs_native_fp64_v'] for ds in DATASETS for sp in ['search','confirm']) for a in ['full','extent']}
    candidates=['all_sgd','all_lr_authority','all_post_authority'];dominant=[]
    for a in candidates:
        good=all(ss[ds]['search'][a]['corruption']['metrics']['vs_before_v']['mean']>=ss[ds]['search']['all_adam']['corruption']['metrics']['vs_before_v']['mean'] and
            ss[ds]['search'][a]['corruption']['tails']['harm_gt20pp']<=ss[ds]['search']['all_adam']['corruption']['tails']['harm_gt20pp'] for ds in DATASETS)
        if good:dominant.append(a)
    winner=max(dominant,key=lambda a:sum(ss[d]['search'][a]['corruption']['metrics']['vs_before_v']['mean'] for d in DATASETS)) if dominant else 'all_adam'
    write(PUB/'DECISION.json',dict(temporal_posterior_qualification=temporal_pass,T1='eligible' if any(temporal_pass.values()) else 'skipped_by_prelocked_posterior_condition',
        spatial_optimizer_search_winner=winner,selection_uses_confirmation=False,method_promoted=False,R1_complete=True,full_route_complete=False,
        next='root: quantify mechanism contrasts; proceed track evidence with one selected optimizer family',universal_temporal_impossibility_claim=False))
    public_check(PUB)

def public_check(folder):
    p=Path(folder);sr=read(p/'SPATIAL_ROWS.json');tr=read(p/'TEMPORAL_ROWS.json');ss=read(p/'SPATIAL_SUMMARY.json');ts=read(p/'TEMPORAL_SUMMARY.json')
    assert len(sr)==11520 and len(tr)==5760
    assert summary(sr,SF,ARMS)==ss and summary(tr,TF,['native','hard','full','extent','pm'],True)==ts
    for r in sr:
        assert 0<=r['v']<=1 and r['gross_gain']==max(r['vs_before_v'],0) and r['gross_loss']==max(-r['vs_before_v'],0)
    for d in read(p/'SPATIAL_DIAGNOSTICS.json'):
        assert not d['GT_selected_step'] and d['selected_step']==min(range(len(d['own_objective_losses'])),key=lambda k:d['own_objective_losses'][k])
    result=dict(status='pass',spatial_rows=len(sr),temporal_rows=len(tr),all_public_aggregates_recomputed=True,time=time.time())
    if not (p/'PUBLIC_AUDIT.json').exists():write(p/'PUBLIC_AUDIT.json',result)
    return result

if __name__=='__main__':run()
