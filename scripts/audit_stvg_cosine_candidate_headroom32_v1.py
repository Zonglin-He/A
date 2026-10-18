"""Independent raw-cache quality, finite-product oracle, aggregate audit and report."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,argparse,itertools,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.run_stvg_cosine_candidate_headroom32_v1 import D,P,OLD,OLDP,CODE,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def properties():
    from vg_tta.stvg_cosine_candidate_headroom32_v1 import oracle,iou,xy
    row=dict(input=dict(width=100,height=100));E=np.array([[.4,.5,.2,.3]]*3);b=np.repeat(E[:,None],3,axis=1);b[:,1,0]+=.12;b[:,2,0]-=.12;ids=np.array([0,2,4]);gt={f:[40,35,60,65]for f in range(5)};v=np.ones((3,3),bool);o=oracle(b,v,E,ids,[0],row,'vidstg',gt,[0,5],[0,5]);allvalues=[]
    for choice in itertools.product(range(3),repeat=2):
        chosen=np.array([0,*choice]);x=xy(b[np.arange(3),chosen],row,'vidstg');total=0
        for f in range(5):total+=float(iou(np.array([np.interp(f,ids,x[:,j])for j in range(4)]),np.array(gt[f])))
        allvalues.append(total/5)
    assert abs(max(allvalues)-o['optimum_v'])<1e-12
    return dict(status='pass',DP_matches_exhaustive_synthetic_product=True,E_in_pool=True,scope='offline finite support oracle only')

def audit():
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
    from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
    p=verify();torch.set_num_threads(4);props=properties();z=read(P/'RESULTS_PUBLIC.json');rr={r['alias']:r for r in z['rows']};checks=collections.Counter();gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
    for f in read(D/'DIAGNOSTIC_COMPLETION.json')['files']:assert sha(ROOT/f['path'])==f['sha256']
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];old=load(OLD/'predictions'/(a+'.pt'));d=load(D/'diagnostics'/(a+'.pt'));row=plans[ds][q];truth,span=gt[ds][0][q],gt[ds][1][q];b=old['direct']['candidates'].numpy();E=old['outputs']['E_MBR'];pos=old['planned_positions'];exact=d['exact_oracle'];N,K=b.shape[:2]
        for t in np.flatnonzero(d['mask']):
            vals=np.array([iou_gt(bb,truth[int(old['frame_ids'][t])],row,ds)for bb in b[t]]);assert np.max(abs(vals-d['quality'][t]))<1e-12;checks['independent_candidate_IoU_scalars']+=len(vals)
        assert np.array_equal(exact['boxes'],b[np.arange(N),exact['choice']])and exact['allowed'][np.arange(N),exact['choice']].all();assert np.array_equal(exact['boxes'][pos],E[pos])and np.array_equal(exact['boxes'][:,2:],E[:,2:]);checks['oracle_actual_candidate_size_anchor_legality']+=1
        outputs=dict(E_MBR=E,Actual_node_oracle=d['node_oracle'],Finite_pool_full_v_oracle=exact['boxes'],On_policy_dense=old['outputs']['On_policy_dense'],Cosine_readout=old['outputs']['Cosine_readout'])
        independent_metrics(dict(dataset=ds,frame_ids=old['frame_ids'],interval=old['interval'],planned_positions=pos,outputs=outputs),row,truth,span,rr[a],checks)
        assert abs(rr[a]['values']['Finite_pool_full_v_oracle']['v']-exact['optimum_v'])<2e-10;assert rr[a]['values']['Actual_node_oracle']['unobserved_actual_s']>=rr[a]['values']['E_MBR']['unobserved_actual_s']-1e-12;checks['oracle_objective_and_actual_node_ceiling']+=1
        # Reverse recursion independently consumes saved edge objectives.
        v=exact['last_cost'].copy();v[~exact['allowed'][-1]]=-np.inf
        for t in range(N-2,-1,-1):v=(exact['costs'][t]+v[None]).max(1);v[~exact['allowed'][t]]=-np.inf
        assert abs(v.max()/exact['denominator']-exact['optimum_v'])<1e-12;checks['reverse_DP_objective']+=1
        for f in d['frame_rows']:
            t=f['position'];delta=d['quality'][t]-d['base_quality'][t];valid=old['direct']['feedback']['valid'][t].numpy().copy();valid[0]=True
            assert abs(f['best_gain_pp']-100*max(float(delta[valid].max()),0))<1e-12;checks['per_frame_headroom_summary']+=1
    for ds,g in z['groups'].items():
        rows=[r for r in z['rows']if r['dataset']==ds]
        for arm,v in g['gain_vs_E'].items():
            for ep,t in v.items():
                arr=np.array([100*(r['values'][arm][ep]-r['values']['E_MBR'][ep])for r in rows]);ix=np.random.default_rng(20261018).integers(len(arr),size=(10000,len(arr)));assert abs(arr.mean()-t['mean'])<1e-12 and max(abs(np.quantile(arr[ix].mean(1),[.025,.975])-np.array(t['CI95'])))<1e-12;checks['independent_paired_oracle_and_actual_summaries']+=1
    verify();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),properties=props,GT_only_offline_diagnostic=True,no_new_models_updates_candidates=True));print(dict(checks),flush=True)

def report():
    z=read(P/'RESULTS_PUBLIC.json');p=verify();put(P/'CONFIGURATION.json',{k:v for k,v in p.items()if k not in ['records','created_epoch']});put(P/'MODEL_PROVENANCE.json',read(OLDP/'MODEL_PROVENANCE.json'))
    put(P/'COST_BREAKDOWN.json',dict(new_GPU_seconds=0,total_Source_captures_completed=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,new_DINO_calls=0,CPU_diagnostic_seconds=z['CPU_seconds'],new_updates=0,new_candidates=0,old_cost_excluded=True))
    def ci(v):return 'unknown'if v['mean']is None else f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}], n={v['parents']}"
    lines=['# E-relative candidate headroom and cosine ranking diagnosis32','',
        'Only read existing commit30746bcc initial34-state candidate grids, rewards/posteriors and actual update paths. Same exposed NH16/NV16 one-query/parent. No new network/DINO call, candidate generation, student update, sweep, dataset expansion, or change to original decisions. Ground truth is explicitly used ONLY for this offline diagnosis.','',
        'E included as a valid baseline even if its own ROI has no grid support; alternatives require original saved visibility. Planned four nodes fixed E; widths/height/MBR unchanged. Missing actual-node GT stays unknown. Two center duplicates remain in34 states; do not treat candidates as independent data.','',
        'Two oracle concepts are different: actual-node per-frame GT best maximizes known-node sIoU; exact finite-product DP maximizes complete official vIoU including interpolation. The former is NOT an upper bound on interpolated complete v. The latter is an upper bound only within this original finite product, may switch identity/produce implausible trajectories, and uses GT at benchmark frames. Neither is deployable or a measured TTA gain.','',
        '| Diagnostic / existing output | HC complete vIoU % | Vid complete vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |','|---|---:|---:|---:|---:|']
    for arm in z['groups']['hc2']['metrics']:lines.append('| '+arm+' | '+' | '.join(f"{100*z['groups'][ds]['metrics'][arm][ep]:.6f}"for ds,ep in [('hc2','v'),('vidstg','v'),('hc2','unobserved_actual_s'),('vidstg','unobserved_actual_s')])+' |')
    for ds,g in z['groups'].items():
        lines+=['',f'## {ds} finite-pool ability','',f"Queries={g['count']}; teacher={g['active_teacher_queries']}; known unobserved actual nodes={g['known_unobserved_nodes']}; unknown={g['unknown_unobserved_nodes']}; counts={g['frame_counts']}."]
        for arm,v in g['gain_vs_E'].items():lines.append(f"{arm}−E: v {ci(v['v'])}; unobserved actual s {ci(v['unobserved_actual_s'])}.")
        lines+=['',f'## {ds} E-relative ranking','',
            'Each statistic averages eligible frames within each query, then averages queries; below n is query/parent support, never frame-pair or candidate count. Good/harm strict difference1e-10; meaningful groups +/-5IoU pp. Raw reward and posterior evaluated separately, score ties1e-12. Posterior includes geometric proximity prior.','',
            '| Quantity | All known unobserved actual nodes, query-balanced | Only within fixed MBR |','|---|---:|---:|']
        for key in ['best_gain_pp','has_gain_1pp','has_gain_5pp','has_gain_10pp','raw_auc','posterior_auc','meaningful_raw_auc','meaningful_posterior_auc','strong_pool_raw_top_good','strong_pool_posterior_top_good','strong_pool_raw_top_harmful','strong_pool_posterior_top_harmful','strong_pool_no_good_outranks_E_raw','strong_pool_no_good_outranks_E_posterior','posterior_good_but_On_nonpositive']:
            lines.append('| '+key+' | '+ci(g['query_balanced']['unobserved_known'][key])+' | '+ci(g['query_balanced']['unobserved_known_inside_MBR'][key])+' |')
        lines+=['','Fractions/AUC are in [0,1], not pp; best_gain_pp is IoU percentage points. Supports vary when no good/bad pair, no teacher or no >=5pp-headroom frame exists. For teacher-only headroom, see RESULTS_PUBLIC.teacher_available_groups; no-teacher queries remain in all-query capacity/full-metric denominator.','',f"Zero-full-v cases: {g['zero_v_cases']}"]
    lines+=['','## All query diagnostic and actual changes versus E, pp','','| Alias | Domain | Exact full-v oracle Δv | Actual-node oracle Δactual-s | On Δv | On Δactual-s |','|---|---|---:|---:|---:|---:|']
    for r in z['rows']:
        e=r['values']['E_MBR'];lines.append('| '+r['alias']+' | '+r['dataset']+' | '+' | '.join(f"{100*(r['values'][arm][ep]-e[ep]):+.6f}"for arm,ep in [('Finite_pool_full_v_oracle','v'),('Actual_node_oracle','unobserved_actual_s'),('On_policy_dense','v'),('On_policy_dense','unobserved_actual_s')])+' |')
    lines+=['','Not a claim that a GT oracle provides usable visual evidence. Raw reward AUC near chance suggests feedback limits in this pool; posterior can additionally favor E because of its sampling prior. Teacher top being useful and actual KL output not useful is a descriptive conversion discrepancy, not proof that the optimizer is broken or that argmax would be a good deployment rule.','',
        'Previous ridge/actual OPD failed/inconclusive decisions preserved. No automated next experiment is authorized by this report. CPU cached diagnosis excludes historical model/expert/feature cost.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(ds,g)in zip(axes,z['groups'].items()):
        arms=['Finite_pool_full_v_oracle','On_policy_dense','Cosine_readout'];v=[g['gain_vs_E'][a]['v']['mean']for a in arms];ax.bar(range(3),v);ax.set_xticks(range(3),['GT finite-pool oracle','Actual OPD','Cosine readout'],rotation=10);ax.axhline(0,color='k',lw=.5);ax.set(title=ds,ylabel='Complete vIoU difference vs E (pp)')
    fig.tight_layout();fig.savefig(P/'HEADROOM.svg');fig.savefig(D/'HEADROOM.png',dpi=150);plt.close(fig)
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=CODE+[str(f.relative_to(ROOT))for f in P.iterdir()if f.is_file()]));put(D/'REPORT_COMPLETION.json',dict(status='complete',count=32))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['properties','audit','report']);x=a.parse_args();v=globals()[x.stage]();print(v if v else 'complete',flush=True)
