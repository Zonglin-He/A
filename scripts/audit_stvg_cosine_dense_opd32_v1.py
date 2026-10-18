"""Independent actual KL steps, cosine queries, unchanged supports and metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,argparse,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_cosine_dense_opd32_v1 import D,P,PRE,OLDP,CODE,verify
import torch,numpy as np

def audit():
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
    from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
    p=verify();torch.set_num_threads(4);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['all32_before_NEW_GT'];res=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in res['rows']};checks=collections.Counter();maxR=0.;center_changes={ds:0 for ds in ['hc2','vidstg']}
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in center_changes};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
    for rec in p['records']:
        a=rec['alias'];ds=rec['dataset'];q=rec['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));old=load(PRE/'feedback'/(a+'.pt'));x=load(PRE/'features'/(a+'.pt'));E=x['E'];pos=o['planned_positions'];n=len(E);allowed=torch.ones(n,dtype=torch.bool);allowed[pos]=False;eps=old['feedback']['noise'];soft=old['feedback']['cosine_all_soft_maps'];perm=o['permutation'];cfg=p['configuration']
        assert np.array_equal(o['outputs']['E_MBR'],E.numpy())and o['interval']==x['interval'];assert torch.equal(o['direct']['candidates'],old['feedback']['candidates'])and (o['direct']['feedback']['prob']-old['feedback']['cosine']).abs().max()<1e-12;checks['cached_initial_candidate_probability_match']+=1
        fixed,on=o['fits']['Fixed_dense'],o['fits']['On_policy_dense'];assert torch.equal(fixed['path'][1]['boxes'],on['path'][1]['boxes'])and torch.equal(fixed['rounds'][0]['gradient'],on['rounds'][0]['gradient'])and torch.equal(fixed['rounds'][0]['feedback']['prob'],on['rounds'][0]['feedback']['prob']);checks['first_step_matched']+=1
        center_changes[ds]+=bool(on['path'][1]['eta'].abs().max()>0)
        for arm,fit in o['fits'].items():
            assert torch.count_nonzero(fit['path'][0]['eta'])==0 and torch.equal(fit['path'][0]['boxes'],E);checks['zero_state_exact_E']+=1
            for j,t in enumerate(fit['rounds']):
                sampler=torch.zeros_like(t['eta_before'])if arm=='Fixed_dense'else fit['path'][j]['eta'];assert torch.equal(sampler,t['sampling_center']);checks['actual_sampling_center']+=1
                sigma=E[:,2:]*cfg['center_scale'];centers=(E[:,:2]+sigma*sampler)[:,None]+sigma[:,None]*eps[j];assert torch.equal(centers,t['candidates'][...,:2]);checks['same_noise_current_candidate_centers']+=1
                b=t['candidates'];valid=x['valid'][perm]if arm=='Shuffled_OPD'else x['valid'];ss=soft[:,perm]if arm=='Shuffled_OPD'else soft;H,W=valid.shape[-2:];area=cell_overlap(b,H,W);ok=(area*valid[:,None]).sum((-1,-2))>1e-12;assert torch.equal(ok,t['feedback']['valid'])
                if len(ss):
                    inter=(area[None]*ss[:,:,None].double()).sum((-1,-2));den=b[None,...,2]*b[None,...,3]*H*W+ss.double().sum((-1,-2))[:,:,None]-inter;rr=torch.where(den>0,inter/den,torch.zeros_like(inter));error=float((rr-t['feedback']['per_anchor_rewards']).abs().max());maxR=max(error,maxR);assert error<1e-12;rewards=rr.mean(0);checks['actual_cosine_reward_scalars']+=rr.numel()
                else:rewards=torch.zeros_like(t['feedback']['rewards'])
                lp=-.5*eps[j].square().sum(-1);safe=ok.clone();safe[~safe.any(1)]=True;lp[~safe]=-torch.inf;prior=lp.softmax(-1);pt=(lp+rewards/cfg['temperature']).softmax(-1)
                informative=ok.any(1)&((rewards.masked_fill(~ok,-torch.inf).amax(-1)-rewards.masked_fill(~ok,torch.inf).amin(-1))>1e-12)if len(ss)else torch.zeros(n,dtype=torch.bool);pt[~informative]=prior[~informative];assert (pt-t['feedback']['prob']).abs().max()<1e-12;checks['actual_teacher_distribution_scalars']+=pt.numel()
                d=sampler[:,None]+eps[j];student=-.5*(d-t['eta_before'][:,None]).square().sum(-1);student[~safe]=-torch.inf;sp=student.softmax(-1);assert (sp-t['student_prob']).abs().max()<1e-12
                gradient=((sp-pt)[...,None]*d).sum(1);gradient[~(allowed&informative)]=0;assert (gradient-t['gradient']).abs().max()<1e-12;independent_new=t['eta_before']-cfg['step_size']*gradient;assert (independent_new-t['eta_after']).abs().max()<1e-12;new=t['eta_before']-cfg['step_size']*t['gradient'];assert torch.equal(new,t['eta_after'])and torch.equal(new,fit['path'][j+1]['eta']);checks['analytic_KL_actual_step']+=1
                box=E.clone();box[:,:2]=E[:,:2]+E[:,2:]*cfg['center_scale']*new;box[pos]=E[pos];assert torch.equal(box,fit['path'][j+1]['boxes']);checks['actual_external_output_readout']+=1
                if not len(ss):assert torch.equal(box,E);checks['no_reference_strict_noop']+=1
            for entry in fit['path']:assert torch.equal(entry['boxes'][:,2:],E[:,2:])and torch.equal(entry['boxes'][pos],E[pos]);checks['all_step_sizes_observed_fixed']+=1
        for arm,b in o['outputs'].items():assert np.array_equal(b[:,2:],E[:,2:].numpy())and np.array_equal(b[pos],E[pos].numpy());checks['all_five_outputs_size_anchors_fixed']+=1
        independent_metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q],rows[a],checks)
    for ds,g in res['groups'].items():
        rr=[r for r in res['rows']if r['dataset']==ds]
        for key,v in g['paired'].items():
            a,b=key.split('_minus_')
            for ep in ['v','unobserved_actual_s']:
                values=np.array([100*(r['values'][a][ep]-r['values'][b][ep])for r in rr if r['values'][a][ep]is not None and r['values'][b][ep]is not None]);ix=np.random.default_rng(20261018).integers(len(values),size=(10000,len(values)));ci=np.quantile(values[ix].mean(1),[.025,.975]);assert abs(float(values.mean())-v[ep]['mean'])<1e-12 and max(abs(ci-np.array(v[ep]['CI95'])))<1e-12;checks['independent_primary_paired_summaries']+=1
    assert not read(OLDP/'GATE.json')['passed'];verify(full=True);assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),max_independent_reward_error=maxR,actual_second_round_nonzero_center_queries=center_changes,
        all32_before_NEW_GT=True,historical_dev_exposure=True,all_three_update_arms_actually_run=True,ridge_gate_unchanged_failed=True,new_GPU_model_DINO_calls=0));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),flush=True)

def report():
    from vg_tta.stvg_cosine_dense_opd32_v1 import ARMS,FIT_ARMS
    z=read(P/'RESULTS_PUBLIC.json');p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');put(P/'CONFIGURATION.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']})
    put(P/'MODEL_PROVENANCE.json',dict(**read(OLDP/'MODEL_PROVENANCE.json'),new_model_calls_this_run=0,new_DINO_calls_this_run=0,features_and_soft_maps_cached=True))
    cost=dict(new_GPU_seconds=0,total_Source_captures_completed=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,new_DINO_calls=0,
        CPU_construction_and_three_fits_seconds=bar['CPU_construction_and_three_fits_seconds'],three_fit_seconds=sum(r['compute']['three_fit_seconds']for r in z['rows']),fit_episodes=96,scheduled_rounds=192,old_Source_E_DINO_and_map_cost_excluded=True);put(P/'COST_BREAKDOWN.json',cost)
    def ci(t):return 'unknown'if t['mean']is None else f"{t['mean']:+.6f} [{t['CI95'][0]:+.6f},{t['CI95'][1]:+.6f}]"
    lines=['# Cached cosine-feedback output-distribution OPD32','',
        'Actual five-output experiment on exposed NH16/NV16; one query/parent. Independent of the failed ridge teacher; its NO-GO remains unchanged. All3 external-center update branches truly executed, no ranking pre-gate used. No new pretrained model/backbone/DINO call, no historical Before, no QR/LN update.','',
        'Fixed E/MBR, only independent per-frame center-state means change outside four planned DINO nodes. E widths/heights remain exact. Same original two17-candidate antithetic noise batches, sigma0.25*E size, feedback tau0.1, per-frame KL step0.5,2 rounds/readstep2. Teacher = Gaussian sampling-prior tilt by the equally averaged all-E-anchor cosine soft-IoU rewards. No ridge probability mixture. Soft evidence is frozen and non-calibrated.','',
        'Fixed samples both batches about E. On-policy has EXACT first step with Fixed; second batch follows its updated state. Shuffled_OPD uses the same policy/update but cyclically deranges only unobserved target-frame evidence. Raw-feature-before-matching shuffle equivalence checked on a fixed example/domain, no model rerun. Direct readout picks an actual candidate under the original34-state distribution. No averaging endpoints or post-fit E.','',
        '| Output | HC complete vIoU % | Vid complete vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |','|---|---:|---:|---:|---:|']
    for a in ARMS:lines.append('| '+a+' | '+' | '.join(f"{100*z['groups'][ds]['metrics'][a][ep]:.6f}"for ds,ep in [('hc2','v'),('vidstg','v'),('hc2','unobserved_actual_s'),('vidstg','unobserved_actual_s')])+' |')
    for ds,g in z['groups'].items():
        lines+=['',f'## {ds}: paired query differences, pp','',
            '| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | v positive/negative/unchanged | v losses below -5pp |','|---|---:|---:|---:|---:|']
        for key,t in g['paired'].items():
            v=t['v'];lines.append(f"| {key} | {ci(v)} | {ci(t['unobserved_actual_s'])} | {v['positive']}/{v['negative']}/{v['neutral']} | {v['severe_loss_below_minus5pp']} |")
        lines+=['',f"Active all-E queries {g['active']}/16; all16 remain. tIoU is EXACT fixed MBR for all outputs. Unknown actual-node support stays null.",'',f"Complete v contributions by actual observed / actual unobserved / official-interpolated frame support: {g['interpolation_v_contributions_pp']}"]
    lines+=['','## All signed case changes versus E, pp','','| Alias | Domain | Cosine readout | Fixed | On-policy | Shuffled |','|---|---|---:|---:|---:|---:|']
    for r in z['rows']:lines.append('| '+r['alias']+' | '+r['dataset']+' | '+' | '.join(f"{100*(r['values'][a]['v']-r['values']['E_MBR']['v']):+.6f}"for a in ARMS[1:])+' |')
    lines+=['','## Offline common-bin strata','']
    for ds,sub in z['strata'].items():
        for key,values in sub.items():
            for name,g in values.items():
                t=g['paired']['On_policy_dense_minus_E_MBR'];lines.append(f"{ds} {key}={name}: n={g['count']}; on-policy−E v {ci(t['v'])}; unobserved actual s {ci(t['unobserved_actual_s'])}.")
    lines+=['','All32 five-output predictions/paths sealed before NEW GT evaluation. Cohort and teacher choice have previous target-GT development exposure; not independent confirmation. No GT in candidate construction, feedback, updates, fit-step selection or readout. All32 no-ref/no-update/failures retained. Paired query=parent bootstrap10000 seed20261018, not candidate-level independence.','',
        'Per-frame categorical Gaussian KL gradients are independently verified against actual state/output changes. This is output-space on-policy distillation, not pretrained-weight adaptation or an autoregressive GKD replication. Dense visual evidence covers existing actual decoder nodes; interpolated benchmark frames are not additional expert observations.','',
        'Cosine gave positive weighted ranking on the previous perturbed pool; that was NOT an E gain. This run measures the actual two-step outputs, not the mean candidate IoU. Sampling-prior and feedback scales are fixed, no sweep/safety gate/third round/second backbone is added. Future expansion is not automatic.','',
        'Actual added work: '+str(cost), '','Existing source/expert/correspondence generation is excluded; CPU cached runtime is not first-use deployment latency.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,(ds,g)in zip(axes,z['groups'].items()):
        ts=[g['paired'][a+'_minus_E_MBR']['v']for a in ARMS[1:]];ax.bar(range(4),[t['mean']for t in ts],yerr=[[t['mean']-t['CI95'][0]for t in ts],[t['CI95'][1]-t['mean']for t in ts]],capsize=3);ax.set_xticks(range(4),['Readout','Fixed','On-policy','Shuffled']);ax.axhline(0,color='k',lw=.6);ax.set(title=ds,ylabel='Complete vIoU difference vs E (pp)')
    fig.tight_layout();fig.savefig(P/'COMPARISON.svg');fig.savefig(D/'COMPARISON.png',dpi=150);plt.close(fig)
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=CODE+[str(f.relative_to(ROOT))for f in P.iterdir()if f.is_file()]));put(D/'REPORT_COMPLETION.json',dict(status='complete',count=32))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','report']);a=ap.parse_args();globals()[a.stage]()
