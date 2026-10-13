"""Independent CPU box/selection/state/Adam and anonymous public arithmetic."""
import collections,json,os,time
import numpy as np
from scripts.stvg_opd_forward_candidate_common_v1 import *

def iou(p,e):
    p,e=np.asarray(p,np.float64),np.asarray(e,np.float64)
    a=np.concatenate((p[:,:2]-p[:,2:]/2,p[:,:2]+p[:,2:]/2),1)
    b=np.concatenate((e[:,:2]-e[:,2:]/2,e[:,:2]+e[:,2:]/2),1)
    inter=np.maximum(0,np.minimum(a[:,2:],b[:,2:])-np.maximum(a[:,:2],b[:,:2])).prod(1)
    area=np.maximum(0,a[:,2:]-a[:,:2]).prod(1);other=np.maximum(0,b[:,2:]-b[:,:2]).prod(1)
    return inter/np.maximum(1e-7,area+other-inter)

def stats(rows):
    result={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds and r['repeat']==0]
        positive=[r for r in rr if r['reason']=='positive_original_gain']
        better=[r for r in positive if r['useful_alternatives']]
        reductions=[(r['candidates']['A']['D']-r['candidates'][r['selected']]['D'])/r['candidates']['A']['D'] for r in positive if r['candidates']['A']['D']>0]
        costs=[r for r in rows if r['dataset']==ds and r['repeat']==1 and r['original_informative']]
        prior_costs=[r['prior_F_local_seconds'] for r in costs]
        screen=[r['screen_seconds'] for r in costs]
        result[ds]=dict(boundaries=len(rr),positive_original_gain=len(positive),nonpositive_original_gain=sum(r['reason']=='nonpositive_original_gain' for r in rr),
          empty_observed=sum(r['reason']=='empty_observed' for r in rr),empty_unobserved=sum(r['reason']=='empty_unobserved' for r in rr),
          useful_alternative_boundaries=len(better),useful_fraction=len(better)/len(positive) if positive else None,
          positive_queries=len({r['parent_id'] for r in positive}),useful_queries=len({r['parent_id'] for r in better}),
          selected_mix=dict(collections.Counter(r['selected'] for r in rr)),
          median_selected_drift_reduction_fraction=float(np.median(reductions)) if reductions else None,
          measured_repeat1_screen_seconds_median=float(np.median(screen)) if screen else None,
          matched_prior_F_seconds_median=float(np.median(prior_costs)) if prior_costs else None,
          descriptive_screen_to_prior_F_ratio=float(np.median(screen)/np.median(prior_costs)) if screen else None,
          candidate_cost_seconds_median={n:float(np.median([r['candidates'][n]['seconds'] for r in costs])) for n in ['A','LN','Query','Half']} if costs else None,
          repeat_exact=sum(r.get('repeat_exact') is True for r in rows if r['dataset']==ds and r['repeat']==1),
          useful_space_condition=bool(positive and len(better)/len(positive)>=.5))
    return result

def root():
    import torch
    torch.set_num_threads(2);assert not torch.cuda.is_initialized();verify()
    from scripts.audit_stvg_opd_observation_response_projection_qualification_final_v1 import adam
    b=read(BASE/'GLOBAL_READOUT_BARRIER.json');assert b['status']=='sealed' and b['count']==48
    design={r['index']:r for r in read(BASE/'DESIGN_LOCK.json')['records']};rows=[];scalar=statecoords=adamcoords=boxcoords=0
    for rel,h in b['files'].items():
        assert sha(ROOT/rel)==h;z=load(ROOT/rel);r=design[z['index']];A=load(ROOT/r['A_path'])['fit'];k=z['boundary']
        assert z['runtime_sha256']==sha(BASE/'RUNTIME.json') and z['A_complete_fit_sha256']==r['A_sha256'] and not z['GT_read']
        assert z['pixel_sha256']==r['pixel_sha256'] and z['source_model_sha256']==r['source_model_sha256']
        inp=prior.old.normalized(load(ROOT/r['originals'][0]['input_path'])) if hasattr(prior.old,'normalized') else None
        if inp is None:
            from scripts.stvg_opd_legacy_input_bridge003 import normalized
            inp=normalized(load(ROOT/r['originals'][0]['input_path']))
        assert z['frame_ids']==inp['frame_ids'] and z['Native_interval']==inp['interval']
        assert z['positions']==A['positions'] and z['visited']==r['visited']
        assert z['unvisited']==[i for i in range(len(z['frame_ids'])) if i not in set(r['visited'])]
        from scripts.run_decota_paper_main_v1 import unpack_expert
        from vg_tta.decota_optimizer_posterior_r1_v1 import Energy
        ex=unpack_expert(inp['expert']);en=Energy(ex,A['before'],'top1')
        assert [int(x[0]) for x in en.frames]==z['positions']
        expected_targets=torch.cat([x[1] for x in en.frames],0) if en.frames else torch.empty((0,4))
        assert torch.equal(expected_targets,z['targets'])
        for name,orig in [('entering',A['before']),('current',A['path'][k]['boxes'])]:assert torch.equal(z[name],orig);boxcoords+=orig.numel()
        assert torch.equal(z['outputs']['A'],A['path'][k+1]['boxes']);boxcoords+=z['outputs']['A'].numel()
        for name,orig in [('before_state',A['path'][k]['state']),('original_after_state',A['path'][k+1]['state'])]:
            assert prior.old.digest(z[name])==prior.old.digest(orig);statecoords+=1792
        witness_path=prior.BASE/'qualification'/f"{z['index']:05}"/f'P_k{k}_repeat{z["repeat"]}.pt'
        assert sha(witness_path)==z['proposal_witness_sha256'];w=load(witness_path)['witness']
        assert bool(w['updated'])==z['original_informative']
        if w['updated']:
            adamcoords+=adam(w,r['config'])
            assert torch.equal(w['gradient'],A['path'][k]['update']['gradient'])
            assert prior.old.digest(w['rollout'])==prior.old.digest(A['rounds'][k]['rollout'])==z['original_noise_sha256']
        else:assert prior.old.digest(z['before_state'])==prior.old.digest(z['original_after_state'])
        for arm in ['A','LN','Query','Half']:
            for n,v in z['candidate_states'][arm].items():
                before=z['before_state'][n].numpy();after=z['original_after_state'][n].numpy()
                if arm=='A':expected=after
                elif arm=='LN':expected=before if n=='spatial.query_residual' else after
                elif arm=='Query':expected=after if n=='spatial.query_residual' else before
                else:expected=before+np.float32(.5)*(after-before)
                assert np.array_equal(v.numpy(),expected);statecoords+=v.numel()
        scores={};O=z['positions'];U=z['unvisited'];ref=z['entering'].numpy().astype(float);cur=z['current'].numpy().astype(float)
        for arm in ['A','LN','Query','Half']:
            boxes=z['outputs'][arm].numpy().astype(float);assert np.isfinite(boxes).all();boxcoords+=boxes.size
            G=float((iou(boxes[O],z['targets'].numpy())-iou(cur[O],z['targets'].numpy())).mean()) if O else None
            D=float(np.square(boxes[U]-ref[U]).sum()/len(U)) if U else None
            for n,v in [('G',G),('D',D)]:
                assert (v is None and z['scores'][arm][n] is None) or (v is not None and abs(v-z['scores'][arm][n])<1e-12);scalar+=1
            scores[arm]={'G':G,'D':D,'seconds':z['candidate_seconds'][arm]}
        if scores['A']['G'] is None:selected='A';feasible=[];reason='empty_observed'
        elif scores['A']['G']<=0:selected='A';feasible=[];reason='nonpositive_original_gain'
        elif scores['A']['D'] is None:selected='A';feasible=[];reason='empty_unobserved'
        else:
            reason='positive_original_gain';feasible=[n for n in ['A','LN','Query','Half'] if scores[n]['G']>=.9*scores['A']['G']]
            selected=min(feasible,key=lambda n:(scores[n]['D'],['A','LN','Query','Half'].index(n)))
        useful=[n for n in feasible if n!='A' and scores['A']['D']-scores[n]['D']>1e-10+64*np.finfo(np.float32).eps*scores['A']['D']]
        assert [selected,feasible,reason,useful]==[z['selected'],z['feasible'],z['reason'],z['useful_alternatives']];scalar+=4
        if not z['original_informative']:assert all(torch.equal(z['outputs'][n],z['current']) for n in ['A','LN','Query','Half'])
        if z['repeat']:
            other=load(BASE/'comparisons'/f"{z['index']:05}"/f'boundary{k}_repeat0.pt')
            exact=all(prior.old.digest(other[n])==prior.old.digest(z[n]) for n in ['entering','current','before_state','original_after_state','candidate_states','outputs','scores','selected','feasible','reason','useful_alternatives'])
            assert exact==z['repeat_exact'];scalar+=1
        priorF=load(prior.BASE/'qualification'/f"{z['index']:05}"/f'F_k{k}_repeat{z["repeat"]}.pt')['witness']['seconds']
        rows.append(dict(parent_id=z['index'],dataset=z['dataset'],boundary=k,repeat=z['repeat'],observed_count=len(O),unobserved_count=len(U),
          original_informative=z['original_informative'],candidates=scores,selected=selected,feasible=feasible,reason=reason,useful_alternatives=useful,
          repeat_exact=z.get('repeat_exact'),screen_seconds=z['screen_seconds'],prior_F_local_seconds=priorF,GT_read=False))
        assert sha(ROOT/rel)==h
    assert len(rows)==48;ss=stats(rows);exact=all(x['repeat_exact'] for x in rows if x['repeat']==1)
    eligible=exact and all(s['useful_space_condition'] for s in ss.values())
    write(OUT/'BOUNDARY_ROWS.json',rows);write(OUT/'STATISTICS.json',dict(status='complete',datasets=ss,real_forward_repeat_exact=exact,useful_space_criterion_met=eligible,
      GT_read=False,formal_fits_executed=0,new_training_fits=0,new_DINO_calls=0))
    write(BASE/'ROOT_READBACK.json',dict(status='pass',scope='complete real candidate outputs/source-state-pixel-frame-native/Adam and noise custody/block-mask/half-state/independent IoU-drift-choice/full anonymous readback; not GT efficacy',
      count=48,total=48,candidate_readouts=192,scalar_checks=scalar,parameter_coordinates=statecoords,box_coordinates=boxcoords,
      independent_Adam_coordinates=adamcoords,CPU_only=True,GT_read=False,time=time.time()))
    write(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json',dict(status='pass',scope='all48 boundary rows/all192 candidate metrics/costs/choices/unknown and no-op labels',count=48,total=48,scalar_checks=scalar,CPU_only=True,GT_read=False,time=time.time()))
    write(BASE/'TECHNICAL_DECISION.json',dict(status='useful_forward_space_task_validation_protocol_pending' if eligible else 'stop_fixed_forward_candidate_recipe_before_GT',
      scope='current eight historical queries/local saved A boundaries only; independent full task protocol and GT efficacy unexecuted',datasets=ss,
      useful_space_criterion_met=eligible,real_forward_repeat_exact=exact,formal_fits_executed=0,GT_read=False,CPU_only=True,time=time.time()))
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_view_public_archive_final',scope='all sealed real-forward comparison/source/math/choice rows complete',count=48,total=48,CPU_only=True,GT_read=False,time=time.time()))
    print(json.dumps(dict(status='root_pass',datasets=ss,useful_space_criterion_met=eligible)))

def public(folder):
    folder=Path(folder);rows=read(folder/'BOUNDARY_ROWS.json');saved=read(folder/'STATISTICS.json');checks=0
    for r in rows:
        assert not r['GT_read'];c=r['candidates'];a=c['A'];order=['A','LN','Query','Half']
        if a['G'] is None:f=[];selected='A';reason='empty_observed'
        elif a['G']<=0:f=[];selected='A';reason='nonpositive_original_gain'
        elif a['D'] is None:f=[];selected='A';reason='empty_unobserved'
        else:
            f=[n for n in order if c[n]['G']>=.9*a['G']];selected=min(f,key=lambda n:(c[n]['D'],order.index(n)));reason='positive_original_gain'
        u=[n for n in f if n!='A' and a['D']-c[n]['D']>1e-10+64*np.finfo(np.float32).eps*a['D']]
        assert [f,selected,reason,u]==[r['feasible'],r['selected'],r['reason'],r['useful_alternatives']];checks+=4
    assert stats(rows)==saved['datasets'];checks+=2
    exact=all(r['repeat_exact'] for r in rows if r['repeat']==1)
    assert exact==saved['real_forward_repeat_exact'] and (exact and all(x['useful_space_condition'] for x in saved['datasets'].values()))==saved['useful_space_criterion_met']
    return dict(status='pass',scope='all anonymous48 rows/192 candidate comparisons/unknown paths/choices/fractions/cost statistics independent arithmetic',count=48,total=48,scalar_checks=checks,CPU_only=True,GT_read=False)

if __name__=='__main__':root()
