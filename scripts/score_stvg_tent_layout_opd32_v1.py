"""Scorer-only recovery: compare saved torch boxes with explicit tensor E.

All128 fits and the global32 prediction barrier remain byte-identical. Original
runner and failure preserved; no model/inference/update rerun or metric change.
"""
from scripts.run_stvg_tent_layout_opd32_v1 import *
def score():
    import torch,numpy as np
    torch.set_num_threads(4)
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    from vg_tta.stvg_tent_layout_opd32_v1 import ARMS,FIT_ARMS
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_NEW_GT']and bar['all_four_update_arms_actually_run']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
    strata=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json')['rows'];strata={ds:{r['query_ordinal']:r for r in rr}for ds,rr in strata.items()};prior={r['alias']:r for r in read(OLDP/'RESULTS_PUBLIC.json')['rows']};rows=[]
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));m=metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q]);s=strata[ds][q]
        for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
            x,y=m['values']['E_MBR'][k],prior[a]['values']['E_MBR'][k];assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12
        fits={}
        for arm,f in o['fits'].items():
            residual=(f['final'][:,:2]-o['outputs']['E_MBR'][:,:2])/(.25*o['outputs']['E_MBR'][:,2:])
            fits[arm]=dict(rounds=len(f['rounds']),state_norm=float(f['path'][-1]['state'].norm()),output_L2=float((f['final']-o['outputs']['E_MBR']).norm()),changed_actual_frames=int((f['final'][:,:2]!=torch.as_tensor(o['outputs']['E_MBR'])[:,:2]).any(1).sum()),residual_first_difference_L2=float(residual.diff(dim=0).norm()),residual_second_difference_L2=float(residual.diff(n=2,dim=0).norm()),teacher_ESS=[v.get('teacher_ESS',float((1/v['feedback']['prob'].square().sum(-1)).mean()))for v in f['rounds']],prior_ESS=[v.get('prior_ESS',float((1/v['feedback']['prior'].square().sum(-1)).mean()))for v in f['rounds']])
        rows.append(dict(alias=a,dataset=ds,active=o['active'],target_area_bin=s['bins']['target_area'],motion_bin=s['bins']['speed_sizes_sec'],query_form=s['query_form'],fits=fits,compute=o['compute'],basis_rank=o['basis_rank'],eligible_nodes=int((o['B'].sum(-1)>0).sum()),**m))
    eps=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'];contrasts=[(a,'E_MBR')for a in ARMS[1:]]+[('Joint_live',a)for a in ARMS[1:]if a!='Joint_live']
    def aggregate(rr):
        return dict(count=len(rr),parents=len(rr),active=sum(r['active']for r in rr),metrics={a:{k:mean([r['values'][a][k]for r in rr])for k in eps}for a in ARMS},
            paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in eps if k!='t'}for a,b in contrasts},
            interpolation_v_contributions_pp={a:{support:mean([r['interpolation_effects'][a][support]['v_contribution_pp']for r in rr])for support in ['observed_actual','unobserved_actual','benchmark_interpolated']}for a in ARMS[1:]})
    groups={ds:aggregate([r for r in rows if r['dataset']==ds])for ds in ['hc2','vidstg']};subgroups={ds:{key:{str(v):aggregate([r for r in rows if r['dataset']==ds and r[key]==v])for v in sorted({r[key]for r in rows if r['dataset']==ds},key=str)}for key in ['target_area_bin','motion_bin','active']}for ds in groups}
    put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,strata=subgroups,all32_outputs_before_NEW_GT=True,cohort_historically_development_exposed=True,independent_confirmation=False,all_four_update_arms_actually_run=True,config=p['configuration'],MBR_fixed=True,no_parameter_search=True))
    put(D/'CPU_SCORE_COMPLETION.json',dict(status='complete',count=32));print({ds:{a:g['metrics'][a]['v']for a in ARMS}for ds,g in groups.items()},flush=True)

if __name__=='__main__':score()
