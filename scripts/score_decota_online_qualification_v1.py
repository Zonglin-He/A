"""Audit genuinely evolving LN chains and nested schedules after global seal."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_actuation_scope_common_v1 import *
from scripts.score_decota_actuation_scope_v1 import path_numpy,path_energy,energy,auth_frame,stats
from scripts.score_decota_optimizer_posterior_v1 import audit_sgd_rounding
from scripts.score_audit_tastvg_decota_c1_same_domain_v1 import flat,check_state
import numpy as np,torch

FIELDS=['v','t','s','vs_frozen_v','before_vs_frozen_v','vs_before_v','vs_episodic_v','gross_gain','gross_loss']

def aggregate(rows):
    out={}
    for ds in DATASETS:
        out[ds]={}
        for split in ['search','confirm']:
            out[ds][split]={}
            for stream in sorted({r['stream'] for r in rows}):
                rr=[r for r in rows if r['dataset']==ds and r['split']==split and r['stream']==stream]
                groups={'corruption':[r for r in rr if r['condition']!='clean'],'clean':[r for r in rr if r['condition']=='clean']}
                for o in ['order1','order2']:groups[o]=[r for r in rr if r['order']==o and r['condition']!='clean']
                for e in [True,False]:groups['expert' if e else 'nonexpert']=[r for r in rr if r['expert']==e and r['condition']!='clean']
                for c in sorted({r['condition'] for r in rr}):groups['condition:'+c]=[r for r in rr if r['condition']==c]
                out[ds][split][stream]={}
                for name,gg in groups.items():
                    z=stats(gg,FIELDS);z['tails']={f'harm_gt{th}pp':sum(r['vs_frozen_v']<-th/100 for r in gg) for th in [5,20]}
                    z['correctness']={str(th):dict(destroyed=sum(r['frozen_v']>=th and r['v']<th for r in gg),recovered=sum(r['frozen_v']<th and r['v']>=th for r in gg)) for th in [.3,.5]}
                    out[ds][split][stream][name]=z
    return out

def audit_fit(z,x,ex,row):
    from vg_tta.c1_enabling_tricks_v1 import QUERY
    checks=check_state(z['initial'],x['initial']);cfg=z['config'];kind=cfg['evidence'];mode=cfg['optimizer'];scope=z['scope'];hist=z['path']
    ff,paths,lp,pa=path_numpy(ex,x['before'].numpy(),row['frame_ids']);fa=auth_frame(ff)
    auth=fa if cfg.get('authority_source')=='frame' else pa if kind.startswith('track') else fa;assert abs(auth-z['authority'])<5e-7
    auth=z['authority'] # Interpolation uses the audited actually executed FP32 scalar.
    selected=min(range(len(hist)),key=lambda j:hist[j]['loss']);assert selected==z['selected_step'];checks+=check_state(z['proposal_state'],hist[selected]['state'])
    used={n:v.clone() for n,v in z['proposal_state'].items()}
    if mode=='post_authority' or cfg.get('authority',False) and mode!='sgd':used={n:x['initial'][n]+auth*(v-x['initial'][n]) for n,v in used.items()}
    if scope=='small_LN':used={n:v if n==QUERY else x['initial'][n]+auth*(v-x['initial'][n]) for n,v in used.items()}
    checks+=check_state(used,z['state']);names=list(x['initial']) if scope!='u_only' else [QUERY];sz=sum(x['initial'][n].numel() for n in names);m=np.zeros(sz);v=np.zeros(sz)
    for j,h in enumerate(hist):
        ee=path_energy(h['boxes'].numpy(),ff,paths,lp,kind) if kind.startswith('track') else energy(h['boxes'].numpy(),ex,'all')*(len(ff) if kind=='frame_sum' else 1)
        assert abs(ee-h['loss'])<8e-6;checks+=1
        if 'update' not in h:continue
        u=h['update'];g=u['gradient'].numpy().astype(float)
        if u['optimizer']=='sgd':expect=-u['lr']*g
        else:
            m=.9*m+.1*g;v=.999*v+.001*g*g;expect=-u['lr']*(m/(1-.9**(j+1)))/(np.sqrt(v/(1-.999**(j+1)))+1e-8)
        if u['optimizer']=='sgd':audit_sgd_rounding(flat(h['state'],names).numpy(),flat(hist[j+1]['state'],names).numpy(),g,u['lr'])
        else:assert np.max(np.abs(expect-u['raw'].numpy()))<3e-6
        assert torch.equal(flat(hist[j+1]['state'],names)-flat(h['state'],names),u['raw']);checks+=sz*3
        if scope=='u_only':
            for n in x['initial']:
                if n!=QUERY:assert torch.equal(x['initial'][n],h['state'][n])
    if scope=='u_only' and not z['empty']:
        u=z['slow_LN'];g=u['gradient'].numpy().astype(float);expect=-u['lr']*g if u['optimizer']=='sgd' else -u['lr']*g/(np.abs(g)+1e-8)
        assert np.max(np.abs(expect-u['raw'].numpy()))<3e-6
        a=auth if mode=='post_authority' or cfg.get('authority',False) and mode!='sgd' else 1.
        assert torch.max((flat(z['write_proposal'],u['names'])-flat(z['state'],u['names'])-u['raw']*a).abs())<3e-6;checks+=1536*3
    else:checks+=check_state(z['write_proposal'],z['state'])
    return checks

def run():
    from scripts import tastvg_decota_c1_common_v1 as c1
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    from vg_tta.decota_actuation_scope_v1 import commit_state
    from vg_tta.c1_enabling_tricks_v1 import QUERY
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();b=read(BASE/'online'/'GLOBAL_PREDICTION_BARRIER.json');assert b['arrivals']==4608
    for f,h in b['files'].items():assert sha(BASE/f)==h
    write(BASE/'online'/'GT_EXPOSURE.json',dict(time=time.time(),barrier_sha256=sha(BASE/'online'/'GLOBAL_PREDICTION_BARRIER.json'),GT_online=False))
    torch.set_num_threads(4);lock=read(BASE/'ONLINE_LOCK.json');rows=[];diag=[];checks=0;tick=time.time();episodes={}
    for ds in DATASETS:
        p=read(r1.BASE/ds/'PLAN.json');labels={s:read(c1.POOL/ds/f'GT_LABELS_{s}.json') for s in p['splits']};plan=lock['plans'][ds]
        for stream in plan['streams']:
            for split,sp in p['splits'].items():
                rate=1. if stream=='episodic' else float(stream.split('_')[1]);chosen=plan['schedules'][split][str(rate)]
                for cond in p['conditions']:
                    for order,seq in sp['orders'].items():
                        prev=None;prevsha=None
                        for at,parent in enumerate(seq):
                            row=p['rows'][parent];f=BASE/'online'/ds/stream/split/cond/order/f'{at:05}.pt';x=checked(f);expert=parent in chosen
                            assert x['expert']==expert and not x['GT_read'] and x['query_reset'] and x['optimizer_reset'] and torch.count_nonzero(x['initial'][QUERY])==0
                            assert x['previous_payload_sha256']==prevsha
                            initial={n:torch.zeros_like(v) if n==QUERY else v for n,v in (prev if prev is not None else x['source_state']).items()};checks+=check_state(initial,x['initial'])
                            if expert:
                                ep=ROOT/x['evidence_path'];assert sha(ep)==x['evidence_sha256'];ex=c1.checked(ep)['expert'];checks+=audit_fit(x['fit'],x,ex,row)
                                commit=commit_state(x['initial'],x['fit']['write_proposal']);checks+=check_state(commit,x['committed']);assert torch.equal(x['after'],x['fit']['final'])
                            else:
                                checks+=check_state(x['initial'],x['committed']);assert x['fit'] is None and torch.equal(x['before'],x['after'])
                            if stream!='episodic':prev=x['committed'];prevsha=sha(f)
                            truth={int(k):v for k,v in labels[split][str(parent)]['truth'].items()};span=labels[split][str(parent)]['span'];iv=x['interval'];nativeiv=x['native']['physical_interval']
                            dt=lambda boxes:DenseTube(boxes.numpy(),row,truth,span,clip=ds=='hc2')
                            before=dt(x['before']).score(nativeiv);frozen=dt(x['native']['boxes']).score(nativeiv);after=dt(x['after']).score(iv);off=official(x['after'].numpy(),row,truth,span,iv,ds)
                            assert max(abs(after[k]-off[k]) for k in off)<2e-12;checks+=3
                            key=(ds,split,cond,order,at)
                            if stream=='episodic':episodes[key]=after['v']
                            spatial=dt(x['after']).score(nativeiv)
                            rows.append(dict(dataset=ds,stream=stream,split=split,condition=cond,order=order,arrival=at,source_id=parent,expert=expert,
                                **off,frozen_v=frozen['v'],before_v=before['v'],vs_frozen_v=off['v']-frozen['v'],before_vs_frozen_v=before['v']-frozen['v'],vs_before_v=off['v']-before['v'],
                                vs_episodic_v=off['v']-episodes[key],gross_gain=max(off['v']-frozen['v'],0),gross_loss=max(frozen['v']-off['v'],0),
                                current_spatial_gain=spatial['v']-before['v'],current_temporal_gain=off['v']-spatial['v'],payload_sha256=sha(f),prestate_sha256=state_hash(x['initial']),committed_state_sha256=state_hash(x['committed'])))
                            diag.append(dict(dataset=ds,stream=stream,split=split,condition=cond,order=order,arrival=at,source_id=parent,expert=expert,
                                fit_seconds=x['fit_seconds'],cached_backward_calls=0 if x['fit'] is None else x['fit']['gradient_calls'],logical_observation_requests=4 if expert else 0,
                                actual_new_DINO=0,new_backbone=0,spatial_current_gain=spatial['v']-before['v'],temporal_current_gain=off['v']-spatial['v']))
                            diag[-1].update(actual_cached_backward_calls=x.get('actual_cached_backwards',diag[-1]['cached_backward_calls']),reused_saved_stream=x.get('reused_saved_stream',False))
                            if x.get('reused_saved_stream'):
                                assert sha(ROOT/x['reuse_path'])==x['reuse_sha256'];checks+=1
                        print('ONLINE_AUDIT',ds,stream,split,cond,order,len(rows),checks,flush=True)
    assert len(rows)==4608;out=PUB/'online';write(out/'ROWS.json',rows);write(out/'DIAGNOSTICS.json',diag);write(out/'SUMMARY.json',aggregate(rows))
    write(out/'CONFIGURATION.json',lock);write(out/'ROOT_AUDIT.json',dict(status='pass',checks=checks,independent_state_chain_optimizer_objective=True,official_dense_agreement=True,arrivals=4608,seconds=time.time()-tick,time=time.time()))
    sums=read(out/'SUMMARY.json');cross_pass=all(sums[d]['confirm']['budget_1.0']['corruption']['metrics']['vs_frozen_v']['ci95'][0]>0 for d in DATASETS)
    write(out/'DECISION.json',dict(cross_domain='eligible' if cross_pass else 'skipped_unqualified_R5_confirmation',prelocked_lower95_positive_both=True,
        selection_uses_confirmation=False,method_promoted=False,time=time.time()))
    public_check(out)

def public_check(folder):
    folder=Path(folder);rr=read(folder/'ROWS.json');assert len(rr)==4608 and aggregate(rr)==read(folder/'SUMMARY.json')
    for r in rr:
        assert 0<=r['v']<=1 and abs(r['vs_frozen_v']-(r['v']-r['frozen_v']))<1e-12
        assert abs(r['vs_before_v']-(r['current_spatial_gain']+r['current_temporal_gain']))<1e-12
    result=dict(status='pass',rows=4608,all_aggregates_recomputed=True,time=time.time())
    if not (folder/'PUBLIC_AUDIT.json').exists():write(folder/'PUBLIC_AUDIT.json',result)
    return result

if __name__=='__main__':run()
