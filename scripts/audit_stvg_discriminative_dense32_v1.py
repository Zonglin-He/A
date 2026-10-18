"""Independent ridge equations/ROI exclusion/weighted quality and metric audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_discriminative_dense32_v1 import D,P,PRE,CODE,verify
import numpy as np
import torch

def independent_metrics(o,row,gt,span,rows,checks):
    ds=o['dataset'];ids=np.asarray(o['frame_ids']);fid=np.array(sorted(gt));truth=np.array([gt[int(i)]for i in fid]);iv=o['interval'];inside=(fid>=max(iv[0],span[0]))&(fid<min(iv[1],span[1]));den=max(max(iv[1],span[1])-min(iv[0],span[0]),1)
    for arm,b in o['outputs'].items():
        xy=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
        if ds=='hc2':xy=np.maximum(xy,0)
        pred=np.column_stack([np.interp(fid,ids,xy[:,j])for j in range(4)]);inter=np.maximum(0,np.minimum(pred[:,2:],truth[:,2:])-np.maximum(pred[:,:2],truth[:,:2])).prod(1)
        union=np.maximum(0,pred[:,2:]-pred[:,:2]).prod(1)+np.maximum(0,truth[:,2:]-truth[:,:2]).prod(1)-inter;iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);iou[(fid<ids[0])|(fid>ids[-1])]=0
        tests=dict(v=float(iou[inside].sum()/den),s=float(iou.mean()),t=max(0,min(iv[1],span[1])-max(iv[0],span[0]))/den)
        for key,mask in [('observed_actual_s',np.isin(fid,ids[o['planned_positions']])),('unobserved_actual_s',np.isin(fid,ids)&~np.isin(fid,ids[o['planned_positions']]))]:tests[key]=float(iou[mask].mean())if mask.any()else None
        for k,v in tests.items():
            got=rows['values'][arm][k];assert got is None if v is None else abs(v-got)<2e-10;checks['independent_metric_scalars']+=1

def audit():
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
    p=verify();torch.set_num_threads(4);checks=collections.Counter();z=read(P/'PHASE1_RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']}
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/shared_text_query32/RESULTS_PUBLIC.json')['rows']}
    bar=read(D/'FEEDBACK_BARRIER.json');assert bar['all32_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    for rec in p['records']:
        a=rec['alias'];ds=rec['dataset'];q=rec['query_ordinal'];o=load(D/'feedback'/(a+'.pt'));f=o['feedback'];feat=load(D/'features'/(a+'.pt'));E=feat['E'];pos=o['planned_positions'];b=f['candidates'];row=plans[ds][q]
        assert torch.equal(f['teacher_readout'][:,2:],E[:,2:])and torch.equal(f['teacher_readout'][pos],E[pos]);checks['readout_sizes_and_observed_exact_E']+=1
        for t in o['teachers']:
            phi=t['phi'];aa=t['weights'];yy=t['labels'];A=phi.T@(aa[:,None]*phi)+p['configuration']['ridge']*torch.eye(phi.shape[1],dtype=torch.float64);rhs=phi.T@(aa*yy);assert (A@t['w']-rhs).abs().max()<1e-10
            assert abs(float(aa[yy>0].sum())-.5)<1e-12 and abs(float(aa[yy<0].sum())-.5)<1e-12;checks['ridge_equations_and_class_balance']+=1
            i=t['position'];pool=next(v for v in feat['frames']if v['position']==i);cc=torch.tensor(pool['boxes']);bg=t['background_boxes'];lo=bg[:,:2]-bg[:,2:]/2;hi=bg[:,:2]+bg[:,2:]/2;cl=cc[:,:2]-cc[:,2:]/2;ch=cc[:,:2]+cc[:,2:]/2
            assert ((torch.minimum(hi[:,None],ch)-torch.maximum(lo[:,None],cl)).clamp_min(0).prod(-1)==0).all();checks['background_excludes_all_candidates']+=1
            # Re-pool actual frozen grid, independently of the teacher pool helper.
            boxes=torch.cat((t['positive'][None],bg));area=cell_overlap(boxes,*feat['maps'].shape[-2:])*feat['valid'][i];raw=(area[:,None]*feat['maps'][i].double()[None]).sum((-1,-2))/area.sum((-1,-2)).clamp_min(1e-12)[:,None];norm=raw/raw.norm(dim=-1,keepdim=True).clamp_min(1e-12);pp=torch.cat((norm,torch.ones(len(norm),1)),1)[t['kept']];assert abs(pp-phi).max()<1e-12;checks['actual_feature_ROI_pool']+=1
        # Candidate-level IoU and scalar quality, without ranking scorer helpers.
        ids=np.array(o['frame_ids']);known=np.array([int(i)in gt[ds][0][q]for i in ids]);mask=known&~np.isin(np.arange(len(ids)),pos);truth_quality=[]
        for i in np.flatnonzero(mask):
            bb=b[i].numpy();xy=np.c_[bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            target=np.array(gt[ds][0][q][int(ids[i])]);inter=np.maximum(0,np.minimum(xy[:,2:],target[2:])-np.maximum(xy[:,:2],target[:2])).prod(-1);union=np.maximum(0,xy[:,2:]-xy[:,:2]).prod(-1)+np.maximum(0,target[2:]-target[:2]).prod()-inter;ii=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
            ps=[f['real']['prob'][i].numpy(),f['real']['prior'][i].numpy(),f['cosine'][i].numpy(),f['shuffled']['prob'][i].numpy()];truth_quality.append([sum(float(x)*float(y)for x,y in zip(pp,ii))for pp in ps]);checks['independent_frame_weighted_quality']+=len(ps)
        if truth_quality:
            vv=np.mean(truth_quality,axis=0);assert abs(vv[0]-vv[1]-rows[a]['supports']['unobserved']['rank_vs_prior']['ridge'])<1e-12;checks['independent_primary_query_rank']+=1
        for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
            v=rows[a]['values']['E_MBR'][k];old=prior[a]['values']['E_MBR'][k];assert v is None and old is None or v is not None and old is not None and abs(v-old)<1e-12;checks['exact_stored_E_metric']+=1
        independent_metrics(o,row,gt[ds][0][q],gt[ds][1][q],rows[a],checks)
    if read(P/'GATE.json')['passed']:
        zz=read(P/'PHASE2_RESULTS_PUBLIC.json');rr={r['alias']:r for r in zz['rows']}
        for r in p['records']:
            a=r['alias'];o=load(D/'predictions'/(a+'.pt'));feat=load(D/'features'/(a+'.pt'));E=feat['E'];pos=feat['planned_positions'];f,g=o['fits']['Fixed_dense'],o['fits']['On_policy_dense'];assert torch.equal(f['path'][1]['boxes'],g['path'][1]['boxes']);checks['matched_first_update']+=1
            for fit in [f,g]:
                for j,t in enumerate(fit['rounds']):
                    expected=((t['student_prob']-t['feedback']['prob'])[...,None]*t['standardized_candidates']).sum(1);allowed=torch.ones(len(E),dtype=torch.bool);allowed[pos]=False;expected[~(allowed&t['feedback']['informative'])]=0;assert abs(expected-t['gradient']).max()<1e-12;assert torch.equal(t['eta_before']-p['configuration']['step_size']*expected,t['eta_after']);checks['analytic_KL_actual_update']+=1
                assert torch.equal(fit['final'][:,2:],E[:,2:])and torch.equal(fit['final'][pos],E[pos]);checks['fixed_widths_and_observed']+=1
            q=r['query_ordinal'];ds=r['dataset'];independent_metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q],rr[a],checks)
    else:assert not (D/'ADAPTATION_BARRIER.json').exists()and not(D/'predictions').exists();checks['failed_gate_no_adaptation']=1
    verify(full=True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),phase1_all32_before_GT=True,phase2_conditional=read(P/'GATE.json')['passed'],pretrained_weights_frozen=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),flush=True)

def report():
    p=verify();z=read(P/'PHASE1_RESULTS_PUBLIC.json');gate=read(P/'GATE.json');bar=read(D/'FEATURE_BARRIER.json');fb=read(D/'FEEDBACK_BARRIER.json')
    put(P/'CONFIGURATION.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']});put(P/'MODEL_PROVENANCE.json',read(ROOT/'results/stvg_error_state/2026-10-18/shared_text_query32/MODEL_PROVENANCE.json'))
    cost=dict(GPU_allocation_seconds=bar['GPU_allocation_seconds'],peak_allocated_bytes=bar['peak_allocated_bytes'],total_Source_captures_completed=32,new_DINO_calls=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,CPU_feedback_seconds=fb['CPU_feedback_seconds'],CPU_adaptation_seconds=read(D/'ADAPTATION_BARRIER.json')['CPU_adaptation_seconds']if gate['passed']else 0.,phase2_ran=gate['passed'],old_Source_E_DINO_cost_excluded=True);put(P/'COST_BREAKDOWN.json',cost)
    lines=['# Target/background ridge teacher and conditional dense distillation32','',
        'Same exposed NH16/NV16, one query/parent, opposite-source TA-STVG. No new DINO or pretrained network update. E/final-MBR fixed. Ridge is per-query/per-anchor disposable target/background fitting, not an offline-trained DiMP or a new video model.','',
        'Appearance maps are frozen input_proj outputs BEFORE text fusion. All32 new Source captures match saved boxes/logits exactly. Background boxes avoid every retained DINO candidate; nonselected candidates remain unknown. Ridge0.01, balanced class weights0.5+0.5, bias regularized. Per-anchor teacher probabilities retained and mixed. Candidates: two initial batches of17 center/antithetic offsets, fixed E size, center std0.25*size. Teacher probabilities explicitly tilt a Gaussian proximity prior; ranking relative that prior separates visual signal from the prior.','',
        '| Domain | Ridge − prior Δrank pp [95% CI] | Ridge − old cosine pp [95% CI] | Ridge − shuffled pp [95% CI] | E vIoU % | Teacher readout vIoU % |','|---|---:|---:|---:|---:|---:|']
    def ci(x):return 'unknown'if x['mean']is None else f"{x['mean']:+.6f} [{x['CI95'][0]:+.6f},{x['CI95'][1]:+.6f}]"
    for ds,g in z['groups'].items():
        u=g['supports']['unobserved'];lines.append(f"| {ds} | {ci(u['rank_vs_prior']['ridge'])} | {ci(u['ridge_minus_control']['cosine'])} | {ci(u['ridge_minus_control']['shuffled'])} | {100*g['metrics']['E_MBR']['v']:.6f} | {100*g['metrics']['Teacher_readout']['v']:.6f} |")
    lines+=['','Gate was fixed BEFORE GT: both domains require all three parent-mean differences above zero. Confidence intervals describe uncertainty and are not silently substituted for this point gate.','',str(gate),'',
        'Phase2 status: '+('ACTUALLY RUN, two external-center KL steps per branch. Same first step/noise/query budget; round2 only sampling center follows current state in on-policy arm.'if gate['passed']else'NOT RUN because the frozen teacher failed the prespecified gate. No fixed dense fit, dynamic OPD, optimizer or temperature search was executed. No OPD effectiveness result can be inferred.'),'',
        '| Domain | Selected E anchors | Usable classifiers | Active queries | No usable teacher | Unknown actual GT frames |','|---|---:|---:|---:|---:|---:|']
    for ds,g in z['groups'].items():lines.append(f"| {ds} | {g['E_anchor_selected']} | {g['usable_teachers']} | {g['active']}/16 | {16-g['active']} | {sum(r['unknown_GT_actual_frames']for r in z['rows']if r['dataset']==ds)} |")
    for ds,g in z['groups'].items():
        lines+=['',f'## {ds} signed results and strata','',f"Teacher readout minus E: {g['readout_minus_E']}",'',f"All supports (rank vs uniform AND prior; far actual frames included): {g['supports']}"]
        for key,vals in z['strata'][ds].items():
            for name,gg in vals.items():lines.append(f"{key}={name}, n={gg['count']}: ridge-prior {ci(gg['supports']['unobserved']['rank_vs_prior']['ridge'])}; readout v {ci(gg['readout_minus_E']['v'])}; unobserved actual s {ci(gg['readout_minus_E']['unobserved_actual_s'])}.")
    if gate['passed']:lines+=['','## Actual phase2 outputs',str(read(P/'PHASE2_RESULTS_PUBLIC.json')['groups'])]
    lines+=['','All32 phase1 features, teachers, distributions and E/readout predictions were sealed before phase1 GT. Missing GT is unknown, not a background label or score of zero. All queries/no-ref cases retained. Old common motion/size bins used offline only. No old model-assisted identity labels are promoted to new human truth.','',
        'If phase2 ran, its global gate depended on phase1 offline target GT. This is disclosed development exposure. Online construction/updates read no GT; phase2 predictions sealed before phase2 scoring. Neither phase is independent confirmation.','',
        'Direct readout is a mechanism control, not the proposed OPD method. Actual decoder nodes provide feedback; official interpolation is not a new visual observation. Widths, four planned anchor positions and time remain E/exact MBR. Wrong E identities can still train wrong appearance models. Coarse pooled features and score calibration remain limitations; one failed ridge configuration does not disprove discriminative tracking.','',
        'Measured costs: '+str(cost),'',
        'References are mechanism context only: [DiMP official project](https://visionml.github.io/dimp/), [ATOM official project](https://visionml.github.io/atom/), [official PyTracking](https://github.com/visionml/pytracking). Their trained task-specific components are not used here. No new download or pretrained tracker is introduced.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(ds,g)in zip(axes,z['groups'].items()):
        u=g['supports']['unobserved'];vals=[u['rank_vs_prior']['ridge'],u['ridge_minus_control']['cosine'],u['ridge_minus_control']['shuffled']];ax.bar(range(3),[x['mean']for x in vals],yerr=[[x['mean']-x['CI95'][0]for x in vals],[x['CI95'][1]-x['mean']for x in vals]],capsize=3);ax.set_xticks(range(3),['vs prior','vs cosine','vs shuffled']);ax.axhline(0,color='k',lw=.6);ax.set(title=ds,ylabel='Unobserved actual-frame Δrank (pp)')
    fig.tight_layout();fig.savefig(P/'COMPARISON.svg');fig.savefig(D/'COMPARISON.png',dpi=150);plt.close(fig)
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=CODE+[str(f.relative_to(ROOT))for f in P.iterdir()if f.is_file()]));put(D/'REPORT_COMPLETION.json',dict(status='complete',count=32,phase2_ran=gate['passed']))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','report']);a=ap.parse_args();globals()[a.stage]()
