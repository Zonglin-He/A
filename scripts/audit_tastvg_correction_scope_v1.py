"""Independent dense formulas, SGD/block/reset contracts and scalar readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_correction_scope_common_v1 import *
from scripts.tastvg_cpu_handoff_v1 import verify_cpu
from scripts.audit_tastvg_negative_evidence_v1 import Checker,shstate,metric,geometry,ranks,summary as independent_summary,independent_tails

def public(directory=PUB):
    tick=time.monotonic();directory=Path(directory);ck=Checker()
    cfg=read(directory/'CONFIG.json');bar=read(directory/'PREDICTION_BARRIER.json')
    assert cfg['full_reset'] and cfg['parameters']==1792 and cfg['bundles']==BUNDLES
    assert not cfg['GT_used_for_updates'] and not cfg['GT_used_for_selection'] and not cfg['production_promoted']
    assert bar['GT_read'] is False and bar['time']<bar['score_start_time']
    life=read(directory/'LIFECYCLE_ROWS.json');matrix=read(directory/'SCOPE_ROWS.json')
    grouped=read(directory/'DONOR_SCOPE_ROWS.json');saved=read(directory/'SUMMARY.json')
    assert len(life)==1152 and sum(r['expert_scheduled'] for r in life)==288
    assert len(matrix)==cfg['scope_target_cells'] and len(grouped)==cfg['donor_balanced_scope_rows']
    contrasts=cfg['life_contrasts']
    for r in life:
        for a,b in contrasts:
            for m in ('v','t','s'):ck.close(r[a+'_minus_'+b+'_'+m],r[a+'_'+m]-r[b+'_'+m],'life_differences')
        for m in ('v','t','s'):ck.close(r['E_before_'+m],r['Frozen_'+m],'episodic_source_readout')
        if not r['expert_scheduled']:
            for setting in ('','_free','_fast'):
                for m in ('v','t','s'):ck.close(r['E_after'+setting+'_'+m],r['Frozen'+setting+'_'+m],'nonexpert_reset')
    for r in matrix:
        for b in BLOCKS:
            for m in ('v','t','s'):ck.close(r[b+'_minus_pre_'+m],r[b+'_'+m]-r['pre_'+m],'scope_differences')
            d=r[b+'_minus_pre_v'];ck.close([r[b+'_gross_gain_pre'],r[b+'_gross_loss_pre']],[max(d,0),max(-d,0)],'scope_gross')
            ck.close(r[b+'_t'],r['pre_t'],'fixed_temporal_scope')
    pools=collections.defaultdict(list)
    for r in matrix:
        for s in r['roles']:
            if s in SCOPES:pools[(r['cell_key'],s)].append(r)
    assert len(pools)==len(grouped)
    for r in grouped:
        rr=pools[(r['cell_key'],r['scope'])];assert r['target_count']==len(rr)
        for k,v in r.items():
            if k.startswith(tuple(b+'_' for b in BLOCKS)) or k.startswith('pre_'):
                ck.close(v,np.mean([a[k] for a in rr]),'donor_balance')
    bykey=collections.defaultdict(list)
    for r in matrix:bykey[r['cell_key']].append(r)
    for rr in bykey.values():
        a=next(r for r in rr if 'matched_cross_baseline' in r['roles'])
        b=next(r for r in rr if 'different_corruption' in r['roles'])
        assert a['target_source_id']==b['target_source_id'] and a['target_condition']!=b['target_condition']
        for block in BLOCKS:ck.close(b[block+'_cross_minus_same_v'],b[block+'_minus_pre_v']-a[block+'_minus_pre_v'],'paired_cross')
    for ds in DATASETS:
        for sp in ('search','confirm'):
            for group,z in saved['lifecycle'][ds][sp].items():
                rr=[r for r in life if r['dataset']==ds and r['split']==sp and
                    (group=='all' or (r['condition']=='clean')==(group=='clean')) and
                    (group not in ('expert_corrupt','nonexpert_corrupt') or r['expert_scheduled']==(group=='expert_corrupt'))]
                fields=list(z['metrics']);calc=independent_summary(rr,fields)
                for k,v in calc.items():ck.tree(v,z[k],'life_bootstrap')
                ck.tree({a+'_minus_'+b:independent_tails(rr,a,b) for a,b in contrasts},z['negative_tails'],'life_tails')
                for o in ('order1','order2'):ck.tree(independent_summary([r for r in rr if r['order']==o],fields),z['orders'][o],'life_orders')
            for group,scopes in saved['scope'][ds][sp].items():
                for scope,z in scopes.items():
                    rr=[r for r in grouped if r['dataset']==ds and r['split']==sp and r['scope']==scope and
                        (group=='all' or (r['condition']=='clean')==(group=='clean'))]
                    fields=list(z['metrics']);calc=independent_summary(rr,fields)
                    for k,v in calc.items():ck.tree(v,z[k],'scope_bootstrap')
                    raw=[r for r in matrix if r['dataset']==ds and r['split']==sp and scope in r['roles'] and
                         (group=='all' or (r['condition']=='clean')==(group=='clean'))]
                    assert z['target_cells']==len(raw)
                    ck.tree({b+'_minus_pre':independent_tails(raw,b,'pre') for b in BLOCKS},z['negative_tails'],'scope_tails')
                    for o in ('order1','order2'):ck.tree(independent_summary([r for r in rr if r['order']==o],fields),z['orders'][o],'scope_orders')
                    tz=saved['target_cluster_sensitivity'][ds][sp][group][scope]
                    ck.tree(independent_summary([dict(r,source_id=r['target_source_id']) for r in raw],fields),tz,'target_bootstrap')
                cross=[r for r in matrix if r['dataset']==ds and r['split']==sp and 'different_corruption' in r['roles'] and
                       (group=='all' or (r['condition']=='clean')==(group=='clean'))]
                z=saved['cross_corruption'][ds][sp][group]
                ck.tree(independent_summary(cross,list(z['metrics'])),z,'cross_bootstrap')
    out=dict(status='pass',checks=dict(ck.count),scalar_checks=sum(ck.count.values()),max_errors=dict(ck.errors),
        CPU_wall_seconds=time.monotonic()-tick,time=time.time(),raw_GT_required=False)
    return out

def root():
    import torch
    verify_cpu(BASE)
    torch.set_num_threads(2);tick=time.monotonic();ck=Checker();bar=seal();cohort=read(BASE/'COHORT.json')
    exposure=read(BASE/'GT_EXPOSURE.json');assert bar['time']<exposure['time']
    labels={ds:{sp:read(POOL/ds/f'GT_LABELS_{sp}.json') for sp in ('search','confirm')} for ds in DATASETS}
    ar=cohort['alternatives'];parents=sorted(map(int,ar));ap=dict(rows=[ar[str(k)]['row'] for k in parents],orders={'all':list(range(len(parents)))})
    from scripts.diagnose_tastvg_pipeline_cpu_v1 import truth
    ad,asp,provenance=truth('P1',ap);assert provenance==exposure['alternative_GT_inputs']
    ag={p:dict(truth=ad[j],span=asp[j]) for j,p in enumerate(parents)}
    life={r['cell_key']:r for r in read(PUB/'LIFECYCLE_ROWS.json')}
    matrix={(r['cell_key'],r['target_index']):r for r in read(PUB/'SCOPE_ROWS.json')}
    sources={ds:checked(BASE/ds/'SOURCE_STATE.pt')['state'] for ds in DATASETS}
    def equal(a,b,kind):
        assert set(a)==set(b)
        for n,v in a.items():ck.close(v,b[n],kind,0)
    def trace(z,ds):
        state=z['pre_state'];assert shstate(state)==z['pre_state_sha256']
        for step in z['update_steps']:
            equal(step['pre_state'],state,'episode_state_chain');u=step['update']
            assert shstate(step['pre_state'])==step['pre_state_sha256'] and shstate(step['post_state'])==step['post_state_sha256']
            if u is not None:
                assert u['lr']==BUNDLES[ds]['lr'] and u['teacher_temperature']==BUNDLES[ds]['teacher_temperature'] and u['student_temperature']==1.
                for n,v in step['pre_state'].items():
                    expected=v.clone().add_(u['gradients'][n],alpha=-u['lr'])
                    ck.close(expected,step['post_state'][n],'episode_SGD',2e-7,3e-6)
                cs=torch.stack([q['prediction']['boxes'] for q in step['candidates']])
                ck.close(cs[0],step['prediction']['boxes'],'refreshed_candidate_zero',0)
                d=geometry(step['prediction']['boxes'],cs,u['coefficients'])
                lp=(-d).log_softmax(0);lq=(-torch.as_tensor(ranks(np.array(step['rewards'])),dtype=d.dtype)/u['teacher_temperature']).log_softmax(0)
                ck.close(d,u['distances'],'episode_geometry',3e-6,2e-6)
                ck.close(lp.exp(),u['p'],'episode_student',2e-6)
                ck.close(lq.exp(),u['q'],'episode_rank_target',2e-6)
                ck.close(float((lp.exp()*(lp-lq)).sum()),u['loss_before'],'episode_loss',5e-6,4e-6)
            else:equal(step['pre_state'],step['post_state'],'episode_noop')
            state=step['post_state']
        equal(state,z['post_state'],'episode_final_chain')
    episode_seen=set()
    for ds in DATASETS:
        context=checked(BASE/ds/'TEXT_CONTEXT.pt');sim=context['cosine'];ck.close(context['vectors'].numpy()@context['vectors'].numpy().T,sim,'context_cosine',1e-14)
        targetlock=read(BASE/ds/'TARGET_LOCK.json');assert targetlock['time']<bar['time']
        assert sha(BASE/ds/'TEXT_CONTEXT.pt')==targetlock['context_sha256']
        tl={key(i['cell']):i for i in targetlock['rows']};p=plan(ds)
        for c in [v for v in cohort['cells'] if v['dataset']==ds]:
            old=oldcell(c);data,_=capture(ds,c['parent'],c['condition']);r=life[key(c)]
            row=p['rows'][c['parent']];g=labels[ds][c['split']][str(c['parent'])];tt={int(k):v for k,v in g['truth'].items()}
            native=data['prediction'];fixed=native['physical_interval'];ab=old['slow']
            if c['scheduled']:
                matched=oldchecked(local_payload_path(c));ma=matched['arms']['rank_native']
                equal(matched['pre_state'],old['pre_state'],'post_output_matched_prestate')
                equal(ma['post_state'],old['post_state'],'post_output_matched_poststate')
                mp=ma['steps'][0]['pre_prediction'];mq=ma['steps'][-1]['post_prediction']
                for field in ('boxes','indices'):
                    ck.close(mp[field],ab[field],'post_output_matched_pre_prediction',0)
                aa=old.get('post_prediction',mq)
                for field in ('boxes','indices'):
                    ck.close(aa[field],mq[field],'post_output_matched_prediction',0)
            else:
                equal(old['pre_state'],old['post_state'],'nonexpert_post_noop')
                assert not old['updated'] and old['pre_sha']==old['post_sha']
                aa=old.get('post_prediction',ab)
                for field in ('boxes','indices'):ck.close(aa[field],ab[field],'nonexpert_post_prediction',0)
            oi=old['final_indices'];ai=[row['frame_ids'][oi[0]],row['frame_ids'][oi[1]]+1]
            if c['scheduled']:
                e=checked(episode_path(ds,c['parent'],c['condition']));z=e['result'];sig=(ds,c['parent'],c['condition'])
                equal(z['pre_state'],sources[ds],'full_source_reset');assert shstate(sources[ds])==e['source_state_sha256']
                assert z['compute']['spatial_provider_calls']==1 and z['compute']['temporal_provider_calls']==1
                if sig not in episode_seen:trace(z,ds);episode_seen.add(sig)
                for m in ('boxes','indices'):ck.close(z['prediction'][m],native[m],'source_suffix_parity',0)
                ep,ea=z['prediction'],z['post_prediction'];ef,efafter=z['output_prediction'],z['after_fast_prediction']
            else:ep=ea=ef=efafter=native
            arms=dict(Frozen=native,A_before=ab,A_after=aa,E_before=ep,E_after=ea)
            fast=dict(Frozen=native,A_before=dict(ab,physical_interval=ai),A_after=dict(aa,physical_interval=ai),E_before=ef,E_after=efafter)
            for a,pred in arms.items():
                for setting,pr,ii in [('',pred,fixed),('_free',pred,None),('_fast',fast[a],None)]:
                    m=metric(pr,row,tt,g['span'],ds,ii)
                    for k,v in m.items():ck.close(v,r[a+setting+'_'+k],'independent_lifecycle_dense',1e-9)
            if not c['scheduled']:continue
            x=checked(matrix_path(c));donor=oldchecked(local_payload_path(c));post=donor['arms']['rank_native']['post_state'];pre=donor['pre_state']
            equal(pre,old['pre_state'],'original_donor_pre');equal(post,old['post_state'],'original_donor_post')
            assert x['donor_receipt_sha256']==sha(Path(str(local_payload_path(c))+'.pt'))
            seq=plan(ds)['splits'][c['split']]['orders'][c['order']]
            future=[j for j in range(c['arrival']+1,len(seq)) if j%4!=0 and p['rows'][seq[j]]['input']['video_sha256']!=row['input']['video_sha256']]
            nt=min(future,key=lambda j:(-sim[c['parent'],seq[j]],j));ft=min(future,key=lambda j:(sim[c['parent'],seq[j]],j))
            targets=tl[key(c)]['targets'];assert len(targets)==len(x['targets'])
            for j,q in enumerate(x['targets']):
                t=q['target'];assert t==targets[j];r=matrix[(key(c),j)]
                if 'semantic_near' in t['roles']:assert t['arrival']==nt
                if 'semantic_far' in t['roles']:assert t['arrival']==ft
                if 'different_video_same_corruption' in t['roles']:assert t['arrival'] in future and t['parent']==seq[t['arrival']]
                tr=ar[str(c['parent'])]['row'] if t['alternative'] else p['rows'][t['parent']]
                tg=ag[c['parent']] if t['alternative'] else labels[ds][c['split']][str(t['parent'])]
                tt={int(k):v for k,v in tg['truth'].items()}
                if t['alternative']:
                    assert tr['input']['video_sha256']==row['input']['video_sha256'] and tr['input']['caption']!=row['input']['caption']
                ii=q['before']['physical_interval'];baseline=metric(q['before'],tr,tt,tg['span'],ds,ii)
                for k,v in baseline.items():ck.close(v,r['pre_'+k],'independent_scope_baseline',1e-9)
                for b in BLOCKS:
                    partial=post if b=='full' else {n:post[n] if (n=='spatial.query_residual' if b=='query' else f'.{b}.' in n) else v for n,v in pre.items()}
                    assert shstate(partial)==q['state_hashes'][b];ck.count['exact_block_masks']+=1
                    m=metric(q['after'][b],tr,tt,tg['span'],ds,ii)
                    for k,v in m.items():ck.close(v,r[b+'_'+k],'independent_scope_dense',1e-9)
                    for setting,interval in [('free',None),('source_fixed',q['source_interval'])]:
                        a=metric(q['after'][b],tr,tt,tg['span'],ds,interval);v=metric(q['before'],tr,tt,tg['span'],ds,interval)
                        for k in a:ck.close(a[k]-v[k],r[b+'_'+setting+'_minus_pre_'+k],'independent_scope_controls',1e-9)
                if 'self' in t['roles']:
                    ck.close(q['before']['boxes'],donor['pre_prediction']['boxes'],'donor_self_pre',0)
                    ck.close(q['after']['full']['boxes'],donor['arms']['rank_native']['post_prediction']['boxes'],'donor_self_full',0)
    out=dict(status='pass',checks=dict(ck.count),scalar_checks=sum(ck.count.values()),max_errors=dict(ck.errors),
        episode_unique=len(episode_seen),global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        independent_dense=True,full_reset=True,blocks_nonadditive=True,CPU_wall_seconds=time.monotonic()-tick,time=time.time())
    write(BASE/'ROOT_AUDIT.json',out);write(PUB/'ROOT_AUDIT.json',out)
    pa=public();write(BASE/'PUBLIC_EXPORT_AUDIT.json',pa);write(PUB/'PUBLIC_AUDIT.json',pa)
    print('ROOT_AUDIT_PASS',out['scalar_checks'],'PUBLIC',pa['scalar_checks'],flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:print(__import__('json').dumps(public(Path(sys.argv[1]))))
    else:root()
