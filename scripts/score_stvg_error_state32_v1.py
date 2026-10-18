"""Dense postseal evaluation; no output correction is fitted in this scorer."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_error_state32_v1 import D,P,BASES,REFS,ARMS,verify
from vg_tta.stvg_error_state_correction_v1 import encode


def mean(values):
    v=[float(x)for x in values if x is not None]
    return float(np.mean(v))if v else None


def pair(rows,fn):
    v=[fn(r)for r in rows];v=np.array([x for x in v if x is not None],float)
    if not len(v):return dict(mean=None,CI95=None,parents=0,positive=0,negative=0,neutral=0,severe_harm_below_minus5pp=0)
    idx=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)))
    return dict(mean=float(v.mean()),CI95=np.quantile(v[idx].mean(1),[.025,.975]).tolist(),parents=len(v),
        positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),neutral=int((abs(v)<=1e-10).sum()),
        severe_harm_below_minus5pp=int((v< -5).sum()),per_parent_pp=v.tolist())


def difference(r,base,ref,control,endpoint,clock='final_MBR'):
    g=r['values'][base][ref]['GlobalError'];b=r['values'][base][ref][control]
    a=g['clocks'][clock][endpoint]if endpoint in ['v','t','s']else g['supports'][endpoint]['s']
    c=b['clocks'][clock][endpoint]if endpoint in ['v','t','s']else b['supports'][endpoint]['s']
    return None if a is None or c is None else 100*(a-c)


def summarize(rows):
    result={}
    for base in BASES:
        result[base]={}
        for ref in REFS:
            metrics={}
            for arm in ARMS:
                vv=[r['values'][base][ref][arm]for r in rows]
                supports=list(vv[0]['supports'])if vv else []
                metrics[arm]=dict(clocks={c:{k:mean([v['clocks'][c][k]for v in vv])for k in ['v','t','s','R03','R05']}for c in ['final_MBR','Native','projection','marginal_MAP']},
                    supports={s:dict(s=mean([v['supports'][s]['s']for v in vv]),evaluable_parents=sum(v['supports'][s]['s']is not None for v in vv),known_frames=sum(v['supports'][s]['frames']for v in vv))for s in supports},
                    anchor_z_RMSE=mean([v['anchor_z_RMSE']for v in vv]),anchor_IoU=mean([v['anchor_IoU']for v in vv]),
                    unobserved_center_residual_RMSE=mean([v['unobserved_center_residual_RMSE']for v in vv]),
                    unobserved_log_size_residual_RMSE=mean([v['unobserved_log_size_residual_RMSE']for v in vv]))
            pairs={control:{ep:pair(rows,lambda r,c=control,e=ep:difference(r,base,ref,c,e))for ep in ['v','s','observed_decoder','unobserved_decoder','unobserved_decoder_in_MBR','benchmark_interpolated','far_unobserved_decoder']}for control in ['Base','LocalAnchor']}
            result[base][ref]=dict(queries=len(rows),parents=len(rows),no_reference_queries=sum(r['reference_counts'][ref]==0 for r in rows),
                total_references=sum(r['reference_counts'][ref]for r in rows),metrics=metrics,global_minus=pairs)
    return result


def evaluate():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
    torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_predictions_sealed_before_dense_GT_evaluation']
    dbar=read(D/'DINO_FULL_BARRIER.json');assert sha(D/'DINO_FULL_BARRIER.json')==bar['DINO_full_barrier_sha256']
    for f in bar['files']+dbar['files']:assert sha(ROOT/f['path'])==f['sha256']
    started=time.perf_counter();targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets}
    sf=ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json';assert sha(sf)==read(D/'RUNTIME_LOCK.json')['strata_sha256']
    features={ds:{v['query_ordinal']:v for v in vv}for ds,vv in read(sf)['rows'].items()};rows=[];checks=0;time_checks=0;local_checks=0;global_checks=0
    for r in p['records']:
        z=load(D/'dino_full'/(r['alias']+'.pt'));oracle=load(D/'oracle'/(r['alias']+'.pt'));ds=r['dataset'];q=r['query_ordinal'];row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q];ids=np.array(z['frame_ids']);planned=np.array(z['planned_positions']);feat=features[ds][q]
        known=np.array([int(fid)in gt for fid in ids]);scale=np.array([row['input']['width'],row['input']['height']]*2);xy=np.zeros((len(ids),4))
        for pos in np.flatnonzero(known):xy[pos]=np.asarray(gt[int(ids[pos])])/scale
        gt_b=np.column_stack(((xy[:,:2]+xy[:,2:])/2,xy[:,2:]-xy[:,:2]));unobs=known&~np.isin(np.arange(len(ids)),planned)
        far=np.min(abs(ids[:,None]-ids[planned][None,:]),axis=1)>=.25*(ids[-1]+1-ids[0]);inside=(ids>=z['intervals']['final_MBR'][0])&(ids<z['intervals']['final_MBR'][1])
        masks=dict(observed_decoder=known&~unobs,unobserved_decoder=unobs,unobserved_decoder_in_MBR=unobs&inside,far_unobserved_decoder=unobs&far,all_known_decoder=known)
        references=dict(DINO_full=dict(positions=z['reference_positions'],targets=z['reference_targets']),**oracle['references']);values={};diagnostic={};contributions={}
        for base in BASES:
            values[base]={};diagnostic[base]={};contributions[base]={};baseline=z['bases'][base];bz=encode(baseline)
            for ref in REFS:
                o=z['corrections'][base]if ref=='DINO_full'else oracle['corrections'][ref][base]
                pos=references[ref]['positions'];target=references[ref]['targets'];bb=dict(Base=baseline,LocalAnchor=o['local'],GlobalError=o['global_error']);values[base][ref]={};dense={}
                assert len(pos)==o['reference_count'];assert np.array_equal(o['local'][np.setdiff1d(np.arange(len(ids)),pos)],baseline[np.setdiff1d(np.arange(len(ids)),pos)]);local_checks+=1
                assert not pos or np.array_equal(o['local'][pos],target);local_checks+=1
                if not pos:assert all(np.array_equal(b,baseline)for b in bb.values())
                else:
                    err=encode(target)-bz[pos];assert np.max(abs(err.mean(0)-o['delta']))<1e-12;assert o['after_z_SSE']<=o['before_z_SSE']+1e-12;global_checks+=2
                diagnostic[base][ref]={k:float(o[k])for k in ['before_z_SSE','after_z_SSE','mean_residual_error','center_displacement_max_error','log_size_change_max_error']}
                for arm,b in bb.items():
                    tube=DenseTube(b,row,gt,span,clip=ds=='hc2');dense[arm]=tube;v={}
                    for clock,iv in z['intervals'].items():
                        met=official(b,row,gt,span,iv,ds);ind=tube.score(iv);assert all(abs(met[k]-ind[k])<2e-10 for k in ['v','t','s']);checks+=3
                        met.update(R03=float(met['v']>.3),R05=float(met['v']>.5));v[clock]=met
                        if arm!='Base':assert met['t']==values[base][ref]['Base']['clocks'][clock]['t'];time_checks+=1
                    predxy=np.column_stack((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2))*scale
                    if ds=='hc2':predxy=np.maximum(predxy,0)
                    iou=np.zeros(len(ids));iou[known]=box_iou(predxy[known],xy[known]*scale)
                    support={name:dict(frames=int(mask.sum()),s=mean(iou[mask]))for name,mask in masks.items()}
                    benchmark=~np.isin(tube.fids,ids);support['benchmark_interpolated']=dict(frames=int(benchmark.sum()),s=mean(tube.iou[benchmark]))
                    counts=len(pos);evalpos=[i for i in pos if known[i]]
                    anchor_iou=mean(iou[evalpos]);anchor_rmse=float(np.sqrt(np.square(encode(target)-encode(b)[pos]).mean()))if counts else None
                    ez=encode(gt_b[unobs])-encode(b[unobs])if unobs.any()else np.empty((0,4))
                    values[base][ref][arm]=dict(clocks=v,supports=support,anchor_z_RMSE=anchor_rmse,anchor_IoU=anchor_iou,
                        unobserved_center_residual_RMSE=float(np.sqrt(np.square(ez[:,:2]).mean()))if len(ez)else None,
                        unobserved_log_size_residual_RMSE=float(np.sqrt(np.square(ez[:,2:]).mean()))if len(ez)else None,
                        out_of_image_center_frames=int(((b[:,:2]<0)|(b[:,:2]>1)).any(1).sum()))
                # Attribution to observed nodes, other decoder frames and benchmark
                # interpolated frames sums exactly to fixed-time full delta-v.
                anchor=np.isin(dense['Base'].fids,ids[planned]);native=np.isin(dense['Base'].fids,ids);iv=z['intervals']['final_MBR'];inter=(dense['Base'].fids>=max(iv[0],span[0]))&(dense['Base'].fids<min(iv[1],span[1]));den=max(max(iv[1],span[1])-min(iv[0],span[0]),1)
                comps={}
                for arm in ['LocalAnchor','GlobalError']:
                    delta=dense[arm].iou-dense['Base'].iou;parts={name:float(delta[mask&inter].sum()/den*100)for name,mask in dict(observed=anchor,unobserved_decoder=~anchor&native,benchmark_interpolated=~native).items()}
                    exact=100*(values[base][ref][arm]['clocks']['final_MBR']['v']-values[base][ref]['Base']['clocks']['final_MBR']['v']);assert abs(sum(parts.values())-exact)<1e-9;checks+=1;comps[arm]=parts
                contributions[base][ref]=comps
            # Descriptive variation of true correction error, not an online rule.
            kp=np.flatnonzero(known);speed={}
            if len(kp)>1:
                seconds=np.diff(ids[kp])/row['input']['fps'];tz=encode(gt_b[kp]);error=tz-bz[kp]
                gs=np.linalg.norm(np.diff(tz[:,:2],axis=0),axis=1)/seconds;es=np.linalg.norm(np.diff(error[:,:2],axis=0),axis=1)/seconds
                speed=dict(GT_center_speed=mean(gs),correction_error_center_speed=mean(es),error_to_GT_speed_ratio=float(es.mean()/gs.mean())if gs.mean()>0 else None)
            diagnostic[base]['variation']=speed
        rows.append(dict(alias=r['alias'],dataset=ds,parent_alias=r['alias'],reference_counts={k:len(v['positions'])for k,v in references.items()},
            planned_slots=len(planned),known_decoder_frames=int(known.sum()),unknown_decoder_frames=int((~known).sum()),
            event_length_bin=feat['bins']['event_fraction'],motion_bin=feat['bins']['speed_sizes_sec'],target_area_bin=feat['bins']['target_area'],query_form=feat['query_form'],
            values=values,diagnostics=diagnostic,contributions=contributions))
        print('EVALUATED',r['alias'],flush=True)
    groups={ds:summarize([r for r in rows if r['dataset']==ds])for ds in targets};strata={ds:{key:{v:summarize([r for r in rows if r['dataset']==ds and r[key]==v])for v in sorted({r[key]for r in rows if r['dataset']==ds})}for key in ['motion_bin','target_area_bin','event_length_bin','query_form']}for ds in targets}
    flags={}
    for ref in REFS:
        flags[ref]={ds:all(groups[ds]['Before'][ref]['global_minus']['Base'][ep]['mean']is not None and groups[ds]['Before'][ref]['global_minus']['Base'][ep]['mean']>0 for ep in ['v','unobserved_decoder'])for ds in groups}
    gtgood=all(flags['GT_common'].values());matchedgood=all(flags['DINO_common'].values());deploygood=all(flags['DINO_full'].values())
    action='stop_this_constant_four_parameter_version'if not gtgood else'reference_reliability_limits_matched_correction'if not matchedgood else'deployment_full_reference_support_does_not_pass'if not deploygood else'positive_development_diagnostic_only_no_automatic_OPD'
    decision=dict(status='complete',primary='Before/final_MBR',positive_unobserved_and_v_means=flags,
        GT_common_positive_both=gtgood,DINO_common_positive_both=matchedgood,DINO_full_positive_both=deploygood,
        next_action=action,new_models_or_OPD_authorized=False,no_auto_confirmation=True,
        mean_screen_not_significance_proof=True,prior_MBRT0_failure_preserved=True)
    P.mkdir(parents=True,exist_ok=True);put(P/'RESULTS.json',dict(status='complete',count=32,groups=groups,strata=strata,rows=rows,
        no_reference_in_denominator=True,unknown_not_zero_filled=True,all_DINO_full_sealed_before_GT=True,
        oracle_and_common_sets_not_deployable=True,output_intervention_not_parameter_TTA=True,fixed_original_inherited_state_not_stream_replay=True))
    put(P/'DECISION.json',decision);put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_read=True,
        result_sha256=sha(P/'RESULTS.json'),metric_checks=checks,time_invariant_checks=time_checks,
        local_node_checks=local_checks,global_LS_checks=global_checks,CPU_score_seconds=time.perf_counter()-started))
    put(D/'STATUS.json',dict(status='complete',stage='measured_pending_independent_audit_publication',count=32),True)
    print('DECISION',action,flush=True)


if __name__=='__main__':evaluate()
