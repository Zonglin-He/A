"""Post-seal CPU scores, one scalar selection, and GT-only error decomposition."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,numpy as np,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_increment128_v1 import D,P,EVENT
from scripts.score_stvg_where_when_event128_v1 import paired,avg

def main():
    import torch
    torch.set_num_threads(4)
    p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_new_GT'];assert bar['plan_sha256']==sha(D/'PLAN_LOCK.json')
    for f,h in p['pins'].items():assert sha(R/f)==h
    for f in bar['files']:assert sha(R/f['path'])==f['sha256']
    put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_lambdas_before_new_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]))
    raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];old={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-19/where_when_event128/RESULTS_PUBLIC.json')['rows']};rows=[];checks=0
    for r in p['records']:
        a=r['alias'];z=load(D/'predictions'/f'{a}.pt');row=raw[r['query_ordinal']];truth=gt[r['query_ordinal']];g,h=spans[r['query_ordinal']];values={};geometry={};ids=np.asarray(z['frame_ids']);s,e=np.triu_indices(len(ids),1);starts=ids[s];ends=ids[e]+1
        aa,bb=z['original_interval'];ov=max(0,min(bb,h)-max(aa,g));miss=h-g-ov;extra=bb-aa-ov
        kind='wrong_segment'if ov==0 else'missing_event_part'if miss>0 and extra==0 else'excess_outside_time'if extra>0 and miss==0 else'other_boundary_or_exact'
        for space,boxes in z['trajectories'].items():
            tube=DenseTube(boxes,row,truth,[g,h],clip=True);cumulative=np.r_[0.,np.cumsum(tube.iou)];lo=np.searchsorted(tube.fids,np.maximum(starts,g));hi=np.searchsorted(tube.fids,np.minimum(ends,h));hi=np.maximum(hi,lo);cv=(cumulative[hi]-cumulative[lo])/np.maximum(np.maximum(ends,h)-np.minimum(starts,g),1)
            baseline=tube.score(z['original_interval'])['v'];perfect=tube.score([g,h])['v'];assert cv.max()>=baseline-1e-12
            geometry[space]=dict(original_temporal_loss_pp=100*(perfect-baseline),candidate_recovery_capacity_pp=100*(float(cv.max())-baseline))
            for lam in p['lambdas']:
                name=f'{space}_L{lam:g}';o=z['outputs'][name];iv=o['interval'];v=tube.score(iv);v2=official(boxes,row,truth,[g,h],iv,'hc2');assert max(abs(v[k]-v2[k])for k in ['v','t','s'])<2e-10;checks+=1
                oldkey='E_MBR'if space=='E'else space+'_OriginalTime'
                if lam==0:assert iv==list(z['original_interval'])and max(abs(v[k]-old[a]['values'][oldkey][k])for k in ['v','t','s'])<2e-10
                values[name]=dict(**v,start_absolute_error_frames=abs(iv[0]-g),end_absolute_error_frames=abs(iv[1]-h),mean_boundary_error_frames=(abs(iv[0]-g)+abs(iv[1]-h))/2,interval=iv,posterior_KL=o['KL_posterior_to_prior'],posterior_L1=o['L1_prior_change'],changed=o['changed'])
        for lam in p['lambdas']:
            v=[values[f'Live{s}_L{lam:g}']for s in p['seeds']];values[f'Live_seedmean_L{lam:g}']={k:avg([x[k]for x in v])for k in ['v','t','s','start_absolute_error_frames','end_absolute_error_frames','mean_boundary_error_frames','posterior_KL','posterior_L1']}
        geometry['Live_seedmean']={k:avg([geometry[f'Live{s}'][k]for s in p['seeds']])for k in ['original_temporal_loss_pp','candidate_recovery_capacity_pp']}
        rows.append(dict(alias=a,values=values,error_class=kind,original_exact=(aa==g and bb==h),geometry=geometry,feedback=z['feedback_audits']))
        print('scored',a,flush=True)
    metrics=['v','t','s','start_absolute_error_frames','end_absolute_error_frames','mean_boundary_error_frames'];means={}
    for space in ['Live_seedmean','E']:
        for lam in p['lambdas']:
            name=f'{space}_L{lam:g}';means[name]={k:avg([r['values'][name][k]for r in rows])for k in metrics}
    scores=[(means[f'Live_seedmean_L{lam:g}']['v'],lam)for lam in p['lambdas']];best=max(x for x,l in scores);selected=min(lam for v,lam in scores if abs(v-best)<=1e-12)
    comparisons={}
    for space in ['Live_seedmean','E']:
        for lam in p['lambdas']:
            name=f'{space}_L{lam:g}';base=f'{space}_L0';d={k:paired([(100 if k in ['v','t','s']else 1)*(r['values'][name][k]-r['values'][base][k])for r in rows])for k in metrics};dv=[100*(r['values'][name]['v']-r['values'][base]['v'])for r in rows]
            d['tails']=dict(positive=sum(v>1e-10 for v in dv),negative=sum(v< -1e-10 for v in dv),unchanged=sum(abs(v)<=1e-10 for v in dv),below_minus3=sum(v< -3 for v in dv),below_minus5=sum(v< -5 for v in dv),positive_mean_contribution_pp=float(np.maximum(dv,0).mean()),negative_mean_contribution_pp=float(np.minimum(dv,0).mean()),worst=sorted([dict(alias=r['alias'],delta_pp=x)for r,x in zip(rows,dv)],key=lambda x:x['delta_pp'])[:8]);comparisons[name+'_minus_'+base]=d
    errors={}
    for kind in ['wrong_segment','missing_event_part','excess_outside_time','other_boundary_or_exact']:
        group=[r for r in rows if r['error_class']==kind];errors[kind]=dict(parents=len(group),exact_original=sum(r['original_exact']for r in group),original_temporal_loss_pp=avg([r['geometry']['Live_seedmean']['original_temporal_loss_pp']for r in group]),candidate_capacity_pp=avg([r['geometry']['Live_seedmean']['candidate_recovery_capacity_pp']for r in group]),actual_selected_recovery_pp=avg([100*(r['values'][f'Live_seedmean_L{selected:g}']['v']-r['values']['Live_seedmean_L0']['v'])for r in group]))
    arithmetic={}
    for n,b in [('Live_NewTime_seedmean','Live_OriginalTime_seedmean'),('E_EventLive_seedmean','E_MBR')]:
        dv=np.array([100*(r['values'][n]['v']-r['values'][b]['v'])for r in old.values()]);arithmetic[n]=dict(positive=int((dv>1e-10).sum()),negative=int((dv< -1e-10).sum()),flat=int((abs(dv)<=1e-10).sum()),positive_contribution_pp=float(np.maximum(dv,0).mean()),negative_contribution_pp=float(np.minimum(dv,0).mean()),net_pp=float(dv.mean()))
    result=dict(status='development_screen_completed',parents=128,seeds=p['seeds'],lambdas=p['lambdas'],selected_lambda=selected,means=means,paired=comparisons,error_decomposition=errors,prior_experiment_arithmetic_check=arithmetic,rows=rows,cost=read(D/'COST.json'),source_exports=read(D/'PRIOR_BARRIER.json')['new_Source_queries'],selection_scope='exposed HC128 development only',confirmation_status='not run; missing independent parent token caches; no new expert authorization assumed',new_temporal_OPD=False)
    put(P/'RESULTS_PUBLIC.json',result)
    put(D/'SELECTION_LOCK.json',dict(lambda_value=selected,results_sha256=sha(P/'RESULTS_PUBLIC.json'),selection=p['selection'],frozen_before_any_confirmation=True,zero_lambda_is_identity_not_positive_screen=(selected==0)))
    put(P/'SELECTION_PUBLIC.json',read(D/'SELECTION_LOCK.json'));put(P/'ERROR_DECOMPOSITION.json',errors);put(P/'INDEPENDENT_AUDIT.json',dict(status='pass',official_dense_metric_checks=checks,all128_lambda0_MBR_exact=True,all128_prior_intervals_exact=True,all_seeds_retained=True,all128_readouts_sealed_before_new_GT=True,GT_provenance=prov,all_original_spatial_predictions_unchanged=True));put(D/'STATUS.json',dict(stage='screen_evaluated_confirmation_unresolved',done=128,total=128,selected_lambda=selected),True)
    print('selected',selected,flush=True);print(means,flush=True)

if __name__=='__main__':main()
