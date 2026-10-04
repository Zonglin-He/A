"""Independent exact path, optimizer and dense-scoring audits, post stage seal."""
import sys,time,math,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_actuation_scope_common_v1 import *
from scripts.score_decota_optimizer_posterior_v1 import stats,energy,audit_sgd_rounding
from scripts.score_audit_tastvg_decota_c1_same_domain_v1 import flat,check_state
import numpy as np,torch
from scipy.special import logsumexp

FIELDS=['v','t','s','vs_frozen_v','vs_before_v','vs_anchor_v','gross_gain','gross_loss']

def iou(a,b,giou=False):
    a=np.asarray(a,float);b=np.asarray(b,float)
    aa=np.r_[a[:2]-a[2:]/2,a[:2]+a[2:]/2];bb=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]
    inter=np.maximum(np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2]),0).prod()
    union=max(a[2:].prod()+b[2:].prod()-inter,1e-7);v=inter/union
    if giou:
        area=max(np.maximum(np.maximum(aa[2:],bb[2:])-np.minimum(aa[:2],bb[:2]),0).prod(),1e-7)
        v-=(area-union)/area
    return float(v)

def frame_data(ex,ids):
    ff=[]
    for (_,pos),ob in ex['observations'].items():
        q=ob['probe'];bb=np.asarray(q['boxes'],float).reshape(-1,4);ss=np.asarray(q['target_scores'],float);valid=(bb[:,2:]>0).all(1)
        if valid.any():ff.append((pos,bb[valid],ss[valid]))
    return sorted(ff,key=lambda f:(ids[f[0]],f[0]))

def path_numpy(ex,before,ids):
    import itertools
    ff=frame_data(ex,ids);paths=list(itertools.product(*[range(len(f[1])) for f in ff])) if ff else []
    values=np.zeros(len(paths),dtype=float);ix=np.asarray(paths,dtype=int)
    for t,(pos,boxes,scores) in enumerate(ff):
        native=np.array([iou(before[pos],b) for b in boxes]);values+=scores[ix[:,t]]+native[ix[:,t]]
        if t:
            table=np.array([[iou(a,b) for b in boxes] for a in ff[t-1][1]])
            values+=table[ix[:,t-1],ix[:,t]]
    logp=values-logsumexp(values) if len(values) else np.array([])
    auth=1. if len(paths)==1 else max(0.,min(1.,1+float((np.exp(logp)*logp).sum())/math.log(len(paths)))) if paths else 0.
    return ff,paths,logp,auth

def path_energy(boxes,ff,paths,logp,kind):
    if not ff:return 0.
    ix=np.asarray(paths,dtype=int);rewards=np.zeros(len(paths),dtype=float)
    for t,(pos,candidates,_) in enumerate(ff):
        similarities=np.array([iou(boxes[pos],b,kind=='track_giou') for b in candidates])
        rewards+=similarities[ix[:,t]]
    return float(-logsumexp(logp+rewards))

def auth_frame(ff):
    vals=[]
    for _,bb,s in ff:
        lp=s-logsumexp(s);vals.append(1. if len(s)==1 else 1+float((np.exp(lp)*lp).sum())/math.log(len(s)))
    return float(np.mean(vals)) if vals else 0.

def aggregate(rows):
    out={}
    for ds in DATASETS:
        out[ds]={}
        for split in ['search','confirm']:
            out[ds][split]={}
            for arm in sorted({r['arm'] for r in rows}):
                rr=[r for r in rows if r['dataset']==ds and r['split']==split and r['arm']==arm]
                groups={'corruption':[r for r in rr if r['condition']!='clean'],'clean':[r for r in rr if r['condition']=='clean']}
                for o in ['order1','order2']:groups[o]=[r for r in rr if r['order']==o and r['condition']!='clean']
                for c in sorted({r['condition'] for r in rr}):groups['condition:'+c]=[r for r in rr if r['condition']==c]
                out[ds][split][arm]={}
                for name,gg in groups.items():
                    z=stats(gg,FIELDS);z['tails']={f'harm_gt{th}pp':sum(r['vs_before_v']<-th/100 for r in gg) for th in [5,20]}
                    z['proxy_improved_GT_harmed']=sum(r['proxy_improved_GT_harmed'] for r in gg)
                    out[ds][split][arm][name]=z
    return out

def run(stage):
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    from scripts import tastvg_decota_c1_common_v1 as c1
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.c1_enabling_tricks_v1 import QUERY
    verify();cpu=read(BASE/'CPU_LOCK.json')
    expected_score=cpu['score_sha256']
    for rev in sorted((BASE/'revisions').glob('*.json')):
        expected_score=read(rev)['pin_overrides'].get('scripts/score_decota_actuation_scope_v1.py',expected_score)
    assert sha(ROOT/'scripts/score_decota_actuation_scope_v1.py')==expected_score
    barrier=read(BASE/stage/'GLOBAL_PREDICTION_BARRIER.json');assert barrier['arrivals']==1152 and not barrier['GT_read']
    for f,h in barrier['files'].items():assert sha(BASE/f)==h
    exposure=BASE/stage/'GT_EXPOSURE.json'
    if exposure.exists():assert read(exposure)['barrier_sha256']==sha(BASE/stage/'GLOBAL_PREDICTION_BARRIER.json')
    else:write(exposure,dict(time=time.time(),barrier_sha256=sha(BASE/stage/'GLOBAL_PREDICTION_BARRIER.json'),GT_online=False))
    torch.set_num_threads(4);rows=[];diag=[];checks=0;tick=time.time()
    anchor={ (r['dataset'],r['split'],r['condition'],r['order'],r['arrival']):r['v'] for r in read(r1.PUB/'SPATIAL_ROWS.json') if r['arm']==read(r1.PUB/'DECISION.json')['spatial_optimizer_search_winner'] }
    for ds in DATASETS:
        p=read(r1.BASE/ds/'PLAN.json');labels={s:read(c1.POOL/ds/f'GT_LABELS_{s}.json') for s in p['splits']}
        for split,sp in p['splits'].items():
            for cond in p['conditions']:
                for order,seq in sp['orders'].items():
                    for at,parent in enumerate(seq):
                        row=p['rows'][parent];truth={int(k):v for k,v in labels[split][str(parent)]['truth'].items()};span=labels[split][str(parent)]['span']
                        f=BASE/stage/ds/split/cond/order/f'{at:05}.pt';x=checked(f);ex=c1.checked(ROOT/x['evidence_path'])['expert']
                        assert sha(ROOT/x['evidence_path'])==x['evidence_sha256'] and sha(ROOT/x['prestate_path'])==x['prestate_sha256']
                        old=r1.checked(ROOT/x['prestate_path']) if 'decota_optimizer_posterior' in x['prestate_path'] else r1.prior.checked(ROOT/x['prestate_path'])
                        checks+=check_state(old['initial'],x['initial']);assert torch.equal(x['before'],old['before'])
                        iv=x['native']['physical_interval'];dt=lambda b:DenseTube(b.numpy(),row,truth,span,clip=ds=='hc2')
                        frozen=dt(x['native']['boxes']).score(iv);before=dt(x['before']).score(iv)
                        ff,paths,lp,pa=path_numpy(ex,x['before'].numpy(),row['frame_ids']);fa=auth_frame(ff)
                        for name,z in x['fits'].items():
                            checks+=check_state(z['initial'],x['initial']);cfg=z['config'];kind=cfg['evidence'];mode=cfg['optimizer'];scope=z['scope'];history=z['path']
                            selected=min(range(len(history)),key=lambda j:history[j]['loss']);assert selected==z['selected_step']
                            checks+=check_state(z['proposal_state'],history[selected]['state'])
                            if kind.startswith('track'):
                                assert len(paths)==z['metadata']['path_count'] and np.allclose(np.exp(lp),z['metadata']['probabilities'],atol=2e-6,rtol=0)
                            auth=fa if cfg.get('authority_source')=='frame' else pa if kind.startswith('track') else fa
                            assert abs(auth-z['authority'])<5e-7
                            auth=z['authority'] # Actual FP32 coefficient, after independent FP64 validation.
                            def objective(boxes):
                                if kind.startswith('track'):return path_energy(boxes,ff,paths,lp,kind)
                                ee=energy(boxes,ex,'all');return ee*len(ff) if kind=='frame_sum' else ee
                            used={n:v.clone() for n,v in z['proposal_state'].items()}
                            if mode=='post_authority' or cfg.get('authority',False) and mode!='sgd':used={n:x['initial'][n]+auth*(v-x['initial'][n]) for n,v in used.items()}
                            if scope=='small_LN':used={n:v if n==QUERY else x['initial'][n]+auth*(v-x['initial'][n]) for n,v in used.items()}
                            checks+=check_state(used,z['state']);names=list(x['initial']) if scope!='u_only' else [QUERY];size=sum(x['initial'][n].numel() for n in names)
                            assert size==z['active_parameters'] and size in [256,1792]
                            m=np.zeros(size);v=np.zeros(size);pv=[];gn=[];un=[];fd=[];errmax=0.
                            for j,h in enumerate(history):
                                loss=objective(h['boxes'].numpy());assert abs(loss-h['loss'])<8e-6,(stage,name,loss,h['loss']);checks+=1
                                pv.append(dt(h['boxes']).score(iv)['v']);fd.append(float((h['boxes']-x['before']).abs().mean()))
                                if 'update' not in h:continue
                                u=h['update'];assert u['names']==names;g=u['gradient'].numpy().astype(float)
                                if u['optimizer']=='sgd':expect=-u['lr']*g
                                else:
                                    m=.9*m+.1*g;v=.999*v+.001*g*g;expect=-u['lr']*(m/(1-.9**(j+1)))/(np.sqrt(v/(1-.999**(j+1)))+1e-8)
                                error=float(np.max(np.abs(expect-u['raw'].numpy())));errmax=max(errmax,error)
                                if u['optimizer']=='sgd':audit_sgd_rounding(flat(h['state'],names).numpy(),flat(history[j+1]['state'],names).numpy(),g,u['lr'])
                                else:assert error<3e-6
                                assert torch.equal(flat(history[j+1]['state'],names)-flat(h['state'],names),u['raw']);checks+=size*3
                                if scope=='u_only':
                                    for n in x['initial']:
                                        if n!=QUERY:assert torch.equal(h['state'][n],x['initial'][n])
                                gn.append(float(np.linalg.norm(g)));un.append(float(u['raw'].norm()))
                            if scope=='u_only' and not z['empty']:
                                ss=z['slow_LN'];names=ss['names'];g=ss['gradient'].numpy().astype(float);expect=-ss['lr']*g if ss['optimizer']=='sgd' else -ss['lr']*g/(np.abs(g)+1e-8)
                                assert np.max(np.abs(expect-ss['raw'].numpy()))<3e-6;writevec=flat(z['write_proposal'],names)-flat(z['state'],names)
                                a=auth if mode=='post_authority' or cfg.get('authority',False) and mode!='sgd' else 1.
                                assert torch.max((writevec-ss['raw']*a).abs())<3e-6;assert torch.equal(z['write_proposal'][QUERY],z['state'][QUERY]);checks+=1536*3
                            else:checks+=check_state(z['write_proposal'],z['state'])
                            off=official(z['final'].numpy(),row,truth,span,iv,ds);dense=dt(z['final']).score(iv);assert max(abs(off[k]-dense[k]) for k in dense)<2e-12;checks+=3
                            usedloss=objective(z['final'].numpy());key=(ds,split,cond,order,at)
                            rec=dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,arm=name,**off,
                                vs_before_v=off['v']-before['v'],vs_frozen_v=off['v']-frozen['v'],vs_anchor_v=off['v']-anchor[key],
                                gross_gain=max(off['v']-before['v'],0),gross_loss=max(before['v']-off['v'],0),proxy_improved_GT_harmed=usedloss<history[0]['loss'] and off['v']<before['v'],
                                own_online_trajectory=False,payload_sha256=sha(f),prestate_sha256=state_hash(x['initial']))
                            rows.append(rec);diag.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,arm=name,
                                selected_step=selected,own_losses=[h['loss'] for h in history],used_loss=usedloss,posthoc_GT_vIoU=pv,GT_selected_step=False,
                                gradient_norms=gn,step_parameter_displacements=un,functional_displacements=fd,used_functional_displacement=float((z['final']-x['before']).abs().mean()),
                                write_LN_displacement=float(flat(z['write_proposal'],[n for n in x['initial'] if n!=QUERY]).sub(flat(x['initial'],[n for n in x['initial'] if n!=QUERY])).norm()),
                                authority=auth,frame_authority=fa,path_authority=pa,path_count=len(paths),gradient_calls=z['gradient_calls'],optimizer_max_error=errmax))
                    print('SCOPE_AUDIT',stage,ds,split,cond,order,len(rows),checks,flush=True)
    out=PUB/stage;write(out/'ROWS.json',rows);write(out/'DIAGNOSTICS.json',diag);sums=aggregate(rows);write(out/'SUMMARY.json',sums)
    write(out/'ROOT_AUDIT.json',dict(status='pass',checks=checks,arrivals=1152,arm_cells=len(rows),independent_path_energy_optimizer_scope=True,
        GT_after_global_seal=True,official_dense_agreement=True,seconds=time.time()-tick,time=time.time()))
    if stage=='R3':
        eligible=[]
        for a in ['track','track_authority']:
            good=all(sums[d]['search'][a]['corruption']['metrics']['vs_before_v']['mean']>=sums[d]['search']['frame_sum']['corruption']['metrics']['vs_before_v']['mean'] and
                sums[d]['search'][a]['corruption']['metrics']['vs_anchor_v']['mean']>=0 and
                sums[d]['search'][a]['corruption']['tails']['harm_gt20pp']<=read(r1.PUB/'SPATIAL_SUMMARY.json')[d]['search'][read(r1.PUB/'DECISION.json')['spatial_optimizer_search_winner']]['corruption']['tails']['harm_gt20pp'] for d in DATASETS)
            if good:eligible.append(a)
        winner=max(eligible,key=lambda a:sum(sums[d]['search'][a]['corruption']['metrics']['vs_before_v']['mean'] for d in DATASETS)) if eligible else 'frame'
        config={d:dict(read(BASE/'R2_SELECTION.json')['config'][d],evidence=winner,authority=winner=='track_authority',authority_source='path' if winner=='track_authority' else 'frame') for d in DATASETS}
        write(BASE/'R3_SELECTION.json',dict(config=config,winner=winner,selection_uses_confirmation=False,GIoU='eligible' if eligible else 'skipped_unqualified_track',time=time.time()))
        if not eligible:write(BASE/'R3_FINAL_SELECTION.json',read(BASE/'R3_SELECTION.json'))
    elif stage=='R3G':
        previous=read(PUB/'R3'/'SUMMARY.json');orig=read(BASE/'R3_SELECTION.json');arm=orig['winner']
        good=all(sums[d]['search']['track_giou']['corruption']['metrics']['vs_before_v']['mean']>=previous[d]['search'][arm]['corruption']['metrics']['vs_before_v']['mean'] and
            sums[d]['search']['track_giou']['corruption']['tails']['harm_gt20pp']<=previous[d]['search'][arm]['corruption']['tails']['harm_gt20pp'] for d in DATASETS)
        final=dict(orig,config={d:dict(c,evidence='track_giou') for d,c in orig['config'].items()} if good else orig['config'],winner='track_giou' if good else arm,GIoU_selected=good)
        write(BASE/'R3G_SELECTION.json',final);write(BASE/'R3_FINAL_SELECTION.json',final)
    elif stage=='R4':
        eligible=[]
        for a in ['u_only','small_LN']:
            if all(sums[d]['search'][a]['corruption']['metrics']['vs_before_v']['mean']>=sums[d]['search']['joint']['corruption']['metrics']['vs_before_v']['mean'] and
                sums[d]['search'][a]['corruption']['tails']['harm_gt20pp']<=sums[d]['search']['joint']['corruption']['tails']['harm_gt20pp'] for d in DATASETS):eligible.append(a)
        winner=max(eligible,key=lambda a:sum(sums[d]['search'][a]['corruption']['metrics']['vs_before_v']['mean'] for d in DATASETS)) if eligible else 'joint'
        write(BASE/'R4_SELECTION.json',dict(config=read(BASE/'R3_FINAL_SELECTION.json')['config'],scope=winner,selection_uses_confirmation=False,time=time.time()))
    write(out/'DECISION.json',read(BASE/f'{stage}_SELECTION.json'));public_check(out)

def public_check(folder):
    from scripts.decota_public_result_io_v1 import read as public_read
    folder=Path(folder);rows=public_read(folder/'ROWS.json');diag=public_read(folder/'DIAGNOSTICS.json');assert aggregate(rows)==public_read(folder/'SUMMARY.json')
    assert len(rows)==len(diag) and len({(r['dataset'],r['split'],r['condition'],r['order'],r['arrival']) for r in rows})==1152
    for d in diag:assert not d['GT_selected_step'] and d['selected_step']==min(range(len(d['own_losses'])),key=lambda k:d['own_losses'][k])
    for r in rows:assert 0<=r['v']<=1 and r['gross_gain']==max(r['vs_before_v'],0)
    result=dict(status='pass',rows=len(rows),all_aggregates_recomputed=True,time=time.time())
    if not (folder/'PUBLIC_AUDIT.json').exists():write(folder/'PUBLIC_AUDIT.json',result)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['R3','R3G','R4']);a=p.parse_args();run(a.stage)
