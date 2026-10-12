"""Actual postseal root readback, independent paired statistics and figures."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,sha,load
BASE=ROOT/'artifacts/stvg_opd_supplement_v1'
PUB=ROOT/'results/stvg_opd_supplement/2026-10-11'
FIG=ROOT/'output/pdf/stvg_opd_supplement'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
OLDPUB=ROOT/'results/stvg_opd_paper_hc2_revision/2026-10-08'

def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]

def independent_stats(part,fields,balanced=False):
    # First average orders within each official query, then queries within parent.
    queries=collections.defaultdict(list);seen=set()
    for r in part:
        key=(r['query_ordinal'],r['order'],r.get('condition','clean'))
        assert key not in seen;seen.add(key);queries[r['query_ordinal']].append(r)
    parents=collections.defaultdict(list);qvals=[]
    for q,group in sorted(queries.items()):
        assert len({r['source_id'] for r in group})==1
        v=[math.fsum(r[k] for r in group)/len(group) for k in fields]
        parents[group[0]['source_id']].append(v);qvals.append(v)
    assert parents
    ids=sorted(parents);mat=np.array([[math.fsum(v[j] for v in parents[i])/len(parents[i]) for j in range(len(fields))] for i in ids])
    if balanced:assert len({len(queries[q]) for q in queries})==1
    rng=np.random.default_rng(20261011)
    samples=np.concatenate([mat[rng.integers(0,len(ids),(100,len(ids)))].mean(1) for _ in range(100)])
    ci=np.quantile(samples,[.025,.975],axis=0)
    return dict(sources=len(ids),queries=len(queries),cells=len(part),metrics={k:dict(mean=float(mat[:,j].mean()),query_macro=float(np.mean([v[j] for v in qvals])),ci95=ci[:,j].tolist(),parent_values=[dict(source_id=i,value=float(mat[t,j])) for t,i in enumerate(ids)],negative_parent_count=int((mat[:,j]<0).sum()),positive_parent_count=int((mat[:,j]>0).sum()),harm_gt5pp_sources=int((mat[:,j]<-.05).sum()),harm_gt20pp_sources=int((mat[:,j]<-.2).sum()),worst_parent=float(mat[:,j].min()),best_parent=float(mat[:,j].max())) for j,k in enumerate(fields)})

def compare_stats(a,b):
    assert a['sources']==b['sources'] and a['cells']==b['cells'];checks=2
    for k,m in a['metrics'].items():
        n=b['metrics'][k]
        for field in ['mean','query_macro','worst_parent','best_parent']:
            assert abs(m[field]-n[field])<4e-12;checks+=1
        assert np.max(np.abs(np.asarray(m['ci95'])-n['ci95']))<4e-12;checks+=2
        for field in ['negative_parent_count','positive_parent_count','harm_gt5pp_sources','harm_gt20pp_sources']:
            assert m[field]==n[field];checks+=1
        assert len(m['parent_values'])==len(n['parent_values'])
        for x,y in zip(m['parent_values'],n['parent_values']):assert x['source_id']==y['source_id'] and abs(x['value']-y['value'])<4e-12;checks+=1
    return checks

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelcolor':'#26343b','text.color':'#26343b','axes.titleweight':'normal'})
    FIG.mkdir(parents=True,exist_ok=True)
    return plt

def save_figure(fig,name,folder):
    folder.mkdir(parents=True,exist_ok=True);fig.savefig(folder/(name+'.png'),dpi=240,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    import shutil
    shutil.copyfile(FIG/(name+'.pdf'),folder/(name+'.pdf'))

def orders():
    from scripts.stvg_opd_order_supplement_common_v1 import verify,stages,audit,scientific,ARMS,QUERY
    from scripts.run_stvg_opd_order_supplement_v1 import validate_lifecycle
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    import torch
    torch.set_num_threads(2);verify();b=BASE/'orders';p=PUB/'orders';done=read(b/'CPU_COMPLETION.json');assert done['new_formal_fits']==2304
    saved=read(p/'ORDER_STATISTICS.json')['statistics'];progress=read(p/'STREAM_PROGRESS.json')['curves'];count=coords=fitrounds=first=checks=0;allrows={};ind={};bound={}
    for name,s in stages().items():
        path=p/name/'ROWS.jsonl.gz';assert sha(path)==saved[name]['rows_sha256'];rr=rows(path);allrows[name]=rr
        cells={(r['arm'],r['order'],r['query_ordinal']):r for r in rr};assert len(cells)==1152
        original=read(OLD/'stages'/s['matched_full_stage']/'PREDICTION_BARRIER.json');sourcebinding=original['logical_records']['clean/order1/on_policy/00000'];sourcepath=OLD/sourcebinding['path'];assert sha(sourcepath)==sourcebinding['sha256'];source=load(sourcepath)['fit']['initial']
        barrier=read(b/'stages'/name/'PREDICTION_BARRIER.json');sourcechecks=0
        for order,seq in s['orders'].items():
            for arm in ARMS:
                previous=prevhash=None
                for at,q in enumerate(seq):
                    binding=barrier['logical_records'][f'clean/{order}/{arm}/{at:05}'];f=ROOT/binding['path'];z=load(f);assert sha(f)==binding['sha256'];fit=z['fit'];validate_lifecycle(z,previous,prevhash)
                    assert all(n in source and fit['initial'][n].shape==source[n].shape for n in fit['initial'])
                    if at==0:
                        assert all(torch.equal(fit['initial'][n],v) for n,v in source.items());sourcechecks+=1
                    assert torch.count_nonzero(fit['initial'][QUERY])==0
                    ip=ROOT/z['input']['path'];assert sha(ip)==z['input']['sha256'];inp=normalized(load(ip));ex=unpack_expert(inp['expert']);assert audit(fit,ex)==z['math_audit']
                    assert fit['config']==s['variant_configs'][arm] and fit['selected_step']==s['variant_configs'][arm]['steps'];previous=z['committed'];prevhash=sha(f)
                    if at<2:
                        qf=b/'qualification'/name/'clean'/order/arm/f.name;qualified=load(qf);c=dict(tensors=0,tensor_coordinates=0,scalar_values=0);equal(scientific(fit),scientific(qualified['fit']),c);first+=1;coords+=c['tensor_coordinates']
                    fitrounds+=len(fit['rounds']);count+=1
                print('ORDER_ROOT_FULL_STREAM',name,order,arm,flush=True)
        bound[name]=dict(first_source_states_actual_original_fit_sha256=sha(sourcepath),source_reset_checks=sourcechecks,original_source_model_sha256=barrier['source_model_state_sha256'])
        paired=[]
        for r in rr:
            if r['arm']!='full':continue
            z=dict(r)
            for a in ['alpha0','alpha1']:
                ctrl=cells[a,r['order'],r['query_ordinal']];assert ctrl['arrival']==r['arrival'] and ctrl['Frozen_v']==r['Frozen_v']
                z['Full_minus_'+a+'_v']=r['After_v']-ctrl['After_v']
            paired.append(z)
        fields=['After_v','Before_v','Frozen_v','delta_current_v','delta_inherited_v','Full_minus_alpha0_v','Full_minus_alpha1_v']
        ind[name]=dict(pooled=independent_stats(paired,fields,True),per_order={o:independent_stats([r for r in paired if r['order']==o],fields,True) for o in s['orders']},arm_absolute={a:independent_stats([r for r in rr if r['arm']==a],['After_v','Before_v','Frozen_v','After_s','After_t'],True) for a in ARMS})
        checks+=compare_stats(ind[name]['pooled'],saved[name]['pooled'])
        for o in s['orders']:checks+=compare_stats(ind[name]['per_order'][o],saved[name]['per_order'][o])
        for a in ARMS:checks+=compare_stats(ind[name]['arm_absolute'][a],saved[name]['arm_absolute'][a])
        for o in s['orders']:
            for a in ARMS:
                for cell in progress[name][o][a]:
                    stat=independent_stats([r for r in rr if r['order']==o and r['arm']==a and cell['start']<=r['arrival']<cell['end']],['Before_v','After_v','Frozen_v','delta_inherited_v','delta_current_v'],True)
                    checks+=compare_stats(stat,cell['statistics'])
    assert count==2304 and first==36
    write(b/'ROOT_ALL_FITS_AND_STATISTICS.json',dict(status='pass',complete_fit_math=count,complete_fit_rounds=fitrounds,full_state_coordinates=count*1792,actual_first_formal_pairs=first,bitwise_tensor_coordinates=coords,independent_statistics_comparisons=checks,source_reset_against_actual_original_fit=bound,all_negative_results_retained=True,independent_full_decoder_Jacobian=False,time=time.time()))
    write(p/'ROOT_ORDER_STATISTICS.json',dict(status='pass',statistics=ind,bootstrap=10000,seed=20261011,intervals_unadjusted=True,orders_not_independent_populations=True))
    plt=plotting();fig,axs=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
    labels={'O_hc2_cross_clean':'VidSTG → HC-STVG2','O_vidstg_cross_clean':'HC-STVG2 → VidSTG'}
    for ax,(name,s) in zip(axs,stages().items()):
        for i,o in enumerate(s['orders']):
            for a,dy,color in [('alpha0',-.12,'#217ca3'),('alpha1',.12,'#dc7555')]:
                m=ind[name]['per_order'][o]['metrics']['Full_minus_'+a+'_v'];mean=m['mean']*100;lo,hi=np.asarray(m['ci95'])*100
                ax.errorbar(mean,i+dy,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=3,label='Full − '+a.replace('alpha','α=') if i==0 else None)
        ax.axvline(0,color='#aeb6b9',lw=1);ax.set_title(labels[name]);ax.set_yticks(range(3),['Order 3','Order 4','Order 5']);ax.invert_yaxis();ax.set_xlabel('Paired ΔvIoU (pp), 95% CI');ax.grid(axis='x',alpha=.15)
    axs[0].legend(frameon=False,loc='best',fontsize=10);fig.tight_layout();save_figure(fig,'new_order_state_contrasts',p);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(11,6),sharex=True);colors={'full':'#217ca3','alpha0':'#8b9094','alpha1':'#dc7555'}
    for row,(name,s) in enumerate(stages().items()):
        for col,o in enumerate(s['orders']):
            ax=axs[row,col]
            for a in ARMS:
                cells0=progress[name][o][a];x=[(r['start']+r['end'])/2 for r in cells0]
                for field,style in [('Before_v','--'),('After_v','-')]:
                    y=[r['statistics']['metrics'][field]['mean']*100 for r in cells0];ax.plot(x,y,color=colors[a],ls=style,lw=1.5,label=('Full' if a=='full' else a.replace('alpha','α='))+' '+field.split('_')[0])
            native=[r['statistics']['metrics']['Frozen_v']['mean']*100 for r in progress[name][o]['full']];ax.plot(x,native,':',color='#26343b',lw=1,label='Native source');ax.set_title(labels[name]+' / '+o.replace('order','Order '),fontsize=10);ax.grid(alpha=.13)
            if col==0:ax.set_ylabel('vIoU (%)')
            if row==1:ax.set_xlabel('Arrival (fixed groups of 16)')
    handles,legends=axs[0,0].get_legend_handles_labels();fig.legend(handles,legends,loc='lower center',ncol=4,frameon=False,fontsize=10);fig.tight_layout(rect=[0,.12,1,1]);save_figure(fig,'new_order_before_after_streams',p);plt.close(fig)
    write(b/'ROOT_FIGURE_COMPLETION.json',dict(status='pending_actual_PDF_pixel_view',png_pdf_pairs=2,new_model_calls=0,root_counts=read(b/'ROOT_ALL_FITS_AND_STATISTICS.json'),time=time.time()))
    print('ORDER_ROOT_FIGURES_COMPLETE_PENDING_VIEW',flush=True)

def baselines():
    from scripts.stvg_opd_baseline_supplement_common_v1 import verify,design,audit_baseline,runtime_sha,load
    from scripts.run_stvg_opd_baseline_supplement_v1 import equal_state
    from scripts.decota_paper_common_v1 import digest
    import torch
    torch.set_num_threads(2);verify();b=BASE/'baselines';p=PUB/'baselines';d=design();assert read(b/'FORMAL_CPU_COMPLETION.json')['counts']['arrivals']==82710
    selected=read(b/'SELECTED_CONFIGS.json');devstats=read(p/'DEVELOPMENT_STATISTICS.json')['statistics'];formstats=read(p/'FORMAL_STATISTICS.json')['statistics'];out={};newfit=entropy=statechecks=qualpairs=0;devfits=devchain=devqual=0;reusefits=reusechain=reusesnapshots=0;inputchecks=parentchecks=lr0checks=0
    for ds,s in d['datasets'].items():
        plan=read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')
        ids={name:i for i,name in enumerate(sorted({r['source'] for r in plan['rows']}))}
        excluded={ids[name] for name in s['excluded_primary_parent_sources']};assert len(excluded)==32
        def check_input(z,c):
            q=z['query_ordinal'];row=plan['rows'][q]
            assert z['dataset']==ds and z['source']==s['source'] and z['candidate']==c and z['config']==d['candidates'][c]['config']
            expected_input_sha=hashlib.sha256(json.dumps(row['input'],sort_keys=True).encode()).hexdigest()
            assert z['frame_ids']==row['frame_ids'] and z['input_metadata_sha256']==expected_input_sha and z['runtime_sha256']==runtime_sha()
        # Independently reread every prelocked development trajectory, including
        # source resets and qualification/formal pairing, after the dev seal.
        for c in d['candidates']:
            dest=b/'development'/ds/c;barrier=read(dest/'PREDICTION_BARRIER.json')
            source=load(b/'qualification'/ds/c/'order1/00000.pt')['state_before']
            assert barrier['actual_source_model_hash']==read(b/'qualification'/ds/c/'PREDICTION_BARRIER.json')['actual_source_model_hash']
            for order,seq in barrier['orders'].items():
                previous=None
                for at,q in enumerate(seq):
                    f=dest/order/f'{at:05}.pt';z=load(f)
                    assert sha(f)==barrier['files'][str(f.relative_to(ROOT))] and z['query_ordinal']==q
                    check_input(z,c);inputchecks+=1
                    assert audit_baseline(z)==z['math_audit'];devfits+=1
                    equal_state(source if previous is None else previous,z['state_before'])
                    if previous is not None:devchain+=1
                    previous=z['state_after']
                    if order=='order1' and at<2:
                        qz=load(b/'qualification'/ds/c/order/f.name)
                        keys=['Before','After','audit','state_before','state_after','closure_evidence']
                        equal_state({k:z[k] for k in keys},{k:qz[k] for k in keys});devqual+=1
                        lr0=load(b/'qualification'/ds/c/'lr0_controls'/order/f.name);audit_baseline(lr0)
                        assert all(torch.equal(x,y) for x,y in zip(lr0['state_before']['parameters'],lr0['state_after']['parameters']));lr0checks+=1
            print('BASELINE_ROOT_DEVELOPMENT',ds,c,flush=True)
        # Selection is independently reconstructed from the complete prelocked grid.
        for method in ['TENT','SAR']:
            candidates=[]
            for c,cfg in d['candidates'].items():
                if cfg['method']!=method:continue
                rr=rows(p/'development'/ds/c/'ROWS.jsonl.gz');stat=independent_stats(rr,['After_v','delta_v'],True)
                old=devstats[ds][c]['all_population'];assert abs(stat['metrics']['After_v']['mean']-old['values']['After_v']['parent_macro'])<4e-12
                candidates.append((c,stat['metrics']['After_v']['mean'],stat['metrics']['delta_v']['harm_gt20pp_sources'],cfg['config']['lr'],cfg['config'].get('margin_fraction',0)))
            best=max(x[1] for x in candidates);tied=[x for x in candidates if best-x[1]<=1e-12];winner=min(tied,key=lambda x:(x[2],x[3],x[4]));assert winner[0]==selected['selected'][ds][method]['candidate']
        fullpath=OLDPUB/f'P1_{ds}/ROWS.jsonl.gz';full={(r['query_ordinal'],r['order']):r for r in rows(fullpath)};out[ds]={}
        for method,choice in selected['selected'][ds].items():
            c=choice['candidate'];dest=b/'formal'/ds/c;barrier=read(dest/'PREDICTION_BARRIER.json');rr=rows(p/'formal'/ds/c/'ROWS.jsonl.gz');assert len(rr)==len(full)==s['formal_arrivals'];joined=[]
            for r in rr:
                old=full[r['query_ordinal'],r['order']];assert old['arrival']==r['arrival'] and old['source_id']==r['source_id']
                assert r['source_id']==ids[plan['rows'][r['query_ordinal']]['source']] and r['is_development_parent']==(r['source_id'] in excluded);parentchecks+=1
                z=dict(r,OPD_v=old['After_v'],Source_v=old['Frozen_v'])
                z['selected_minus_Source_v']=r['After_v']-old['Frozen_v'];z['OPD_minus_selected_v']=old['After_v']-r['After_v'];joined.append(z)
            if not barrier.get('reuse_identical_default'):
                for order,seq in barrier['orders'].items():
                    previous=None
                    for at,q in enumerate(seq):
                        f=dest/order/f'{at:05}.pt';z=load(f);assert sha(f)==barrier['files'][str(f.relative_to(ROOT))];assert audit_baseline(z)==z['math_audit'];newfit+=1;entropy+=z['math_audit']['native_entropy_closures']
                        check_input(z,c);inputchecks+=1
                        if previous is not None:equal_state(previous,z['state_before']);statechecks+=1
                        previous=z['state_after']
                        if at<2:
                            qf=b/'formal_qualification'/ds/c/order/f.name;qz=load(qf);keys=['Before','After','audit','state_before','state_after','closure_evidence'];equal_state({k:z[k] for k in keys},{k:qz[k] for k in keys});qualpairs+=1
                    print('BASELINE_ROOT_COMPLETE_STREAM',ds,method,order,flush=True)
            else:
                assert barrier['new_inference_arrivals']==0 and barrier['original_barrier_sha256']==sha(ROOT/'artifacts/decota_paper_experiments_v1/baselines'/f'{method}_{ds}'/'PREDICTION_BARRIER.json')
                # Source/parameter/optimizer saved-state scope is in exact reuse receipt;
                # original per-query GPU gradient tensors were not saved.
                from scripts.decota_paper_common_v1 import digest
                oldbase=ROOT/'artifacts/decota_paper_experiments_v1/baselines'/f'{method}_{ds}'
                oldbarrier=read(oldbase/'PREDICTION_BARRIER.json');plan=read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')
                source=load(b/'qualification'/ds/c/'order1/00000.pt')['state_before']
                sourceflat=np.concatenate([v.numpy().reshape(-1) for v in source['parameters']]).astype(np.float32)
                assert read(oldbase/'PARAMETER_SCOPE.json')['names']==source['names']
                for order,seq in plan['orders'].items():
                    previous=None;previous_sha=None
                    for at,q in enumerate(seq):
                        f=oldbase/order/f'{at:05}.npz';rc=read(f.with_suffix('.json'))
                        assert sha(f)==rc['sha256']==oldbarrier['files'][str(f.relative_to(ROOT/'artifacts/decota_paper_experiments_v1'))]
                        with np.load(f,allow_pickle=False) as a:
                            md=json.loads(a['metadata'].tobytes());sb=a['state_before'];sa=a['state_after']
                            assert md['parent']==q and md['order']==order and md['arrival']==at and md['previous_payload_sha256']==previous_sha
                            assert digest(sb.tolist())==md['parameter_before_sha256'] and digest(sa.tolist())==md['parameter_after_sha256']
                            assert np.array_equal(sourceflat if previous is None else previous,sb)
                            if previous is not None:reusechain+=1
                            if md.get('optimizer_sample'):
                                binding=md['optimizer_sample'];sf=ROOT/'artifacts/decota_paper_experiments_v1'/binding['path'];sample=load(sf)
                                assert sha(sf)==binding['sha256'] and sample['config']==choice['config'] and sample['arrivals']==at+1 and sample['names']==source['names']
                                assert np.array_equal(np.concatenate([v.numpy().reshape(-1) for v in sample['parameters']]).astype(np.float32),sa)
                                assert sample['optimizer']['param_groups'][0]['lr']==choice['config']['lr'];reusesnapshots+=1
                            previous=sa.copy();previous_sha=rc['sha256'];reusefits+=1
            for scope,part in [('primary_excluding_32_dev_parents',[r for r in joined if not r['is_development_parent']]),('all_population',joined)]:
                stat=independent_stats(part,['After_v','Before_v','After_s','After_t','After_R30','After_R50','delta_v','OPD_v','Source_v','selected_minus_Source_v','OPD_minus_selected_v'])
                ref=formstats[ds][c]['primary' if scope.startswith('primary') else 'all_population'];assert stat['sources']==ref['parents'] and stat['cells']==ref['cells']
                for field in ['After_v','Before_v','After_s','After_t','After_R30','After_R50','delta_v']:
                    assert abs(stat['metrics'][field]['mean']-ref['values'][field]['parent_macro'])<4e-12 and np.max(np.abs(np.asarray(stat['metrics'][field]['ci95'])-ref['values'][field]['ci95']))<4e-12
                stat.update(actual_update_arrivals=sum(r['updates']>0 for r in part),optimizer_steps=sum(r['updates'] for r in part),skip_steps=sum(r['skips'] for r in part),recoveries=sum(r['recoveries'] for r in part),update_rate=sum(r['updates']>0 for r in part)/len(part))
                out[ds].setdefault(method,dict(candidate=c,config=choice['config'],exact_default_reuse=bool(barrier.get('reuse_identical_default'))))[scope]=stat
    write(p/'ROOT_MATCHED_MAIN_COMPARISON.json',dict(status='pass',datasets=out,selection_only_same_historical_dev=True,main_method_reselected=False,formal_dev_excluded=True,bootstrap=10000,seed=20261011,unadjusted_CI=True))
    assert devfits==1152 and devqual==lr0checks==36 and parentchecks==82710 and newfit+reusefits==82710
    write(b/'ROOT_ALL_SELECTED_FITS_READBACK.json',dict(status='pass',selected_logical_arrivals=82710,new_complete_fits_rechecked=newfit,independent_entropy_closures=entropy,full_optimizer_state_chain_pairs=statechecks,qualified_formal_pairs=qualpairs,complete_development_fits_rechecked=devfits,development_state_chain_pairs=devchain,development_qualified_pairs=devqual,independent_LR0_math_and_parameter_checks=lr0checks,explicit_input_runtime_binding_checks=inputchecks,independent_formal_parent_exclusion_checks=parentchecks,reused_prediction_parameter_records_rechecked=reusefits,reused_parameter_chain_pairs=reusechain,reused_sparse_optimizer_snapshots_rechecked=reusesnapshots,reused_default_gradient_claim=False,old_default_full_saved_state_scope_only=True,old_optimizer_state_available_every_query=False,GT_after_selected_formal_global=True,time=time.time()))
    plt=plotting();fig,axs=plt.subplots(1,2,figsize=(9,3.7),sharey=False)
    for ax,(ds,z) in zip(axs,out.items()):
        for i,method in enumerate(['TENT','SAR']):
            for field,dy,color,label in [('selected_minus_Source_v',-.12,'#217ca3','Calibrated − Source'),('OPD_minus_selected_v',.12,'#dc7555','OPD − Calibrated')]:
                m=z[method]['primary_excluding_32_dev_parents']['metrics'][field];mean=m['mean']*100;lo,hi=np.asarray(m['ci95'])*100;ax.errorbar(mean,i+dy,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=3,label=label if i==0 else None)
        ax.axvline(0,color='#aeb6b9',lw=1);ax.set_yticks([0,1],['TENT','SAR']);ax.invert_yaxis();ax.set_title('HC-STVG2' if ds=='hc2' else 'VidSTG');ax.set_xlabel('Primary paired ΔvIoU (pp), 95% CI');ax.grid(axis='x',alpha=.15)
    axs[0].legend(frameon=False,fontsize=9);fig.tight_layout();save_figure(fig,'calibrated_baseline_main_comparison',p);plt.close(fig)
    report=['# Development-calibrated native baselines','','Same historical 32 development parents per target; prelocked grid; complete dev seal before dev GT. Formal primary excludes these parents.','','| Target | Method | Chosen LR | SAR margin | Primary vIoU | Query-macro vIoU | Updated arrivals | OPD minus baseline (pp) | 95% CI |','|---|---|---:|---:|---:|---:|---|---:|---|']
    for ds,z in out.items():
        for method,m in z.items():
            stat=m['primary_excluding_32_dev_parents'];effect=stat['metrics']['OPD_minus_selected_v'];v=stat['metrics']['After_v'];report.append(f"| {ds} | {method} | {m['config']['lr']} | {m['config'].get('margin_fraction','—')} | {v['mean']*100:.3f} | {v['query_macro']*100:.3f} | {stat['actual_update_arrivals']}/{stat['cells']} | {effect['mean']*100:+.3f} | [{effect['ci95'][0]*100:+.3f}, {effect['ci95'][1]*100:+.3f}] |")
    report+=['','These are native STVG adaptations of classification TENT/SAR: decoder LayerNorm affine scope, eval dropout, endpoint plus actionness entropy, and native post-update readout. SAR retains both entropy reliability gates and recovery. Calibration changes only the predeclared LR/margin grid; no forced updates, formal selection or Ours retuning. Exact original default reuse preserves original predictions and saved state; original per-query gradient tensors are unavailable for such reused defaults.']
    (p/'REPORT.md').write_text('\n'.join(report)+'\n');write(b/'ROOT_FIGURE_COMPLETION.json',dict(status='pending_actual_PDF_pixel_view',png_pdf_pairs=1,time=time.time()))

def cost():
    b=BASE/'cost';p=PUB/'cost';datasets={};allcount=0
    from scripts.stvg_opd_design_ablation_common_v1 import audit,scientific
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    import torch
    torch.set_num_threads(2);matched=mathrounds=bitwise_coordinates=0
    keys=['decode_preprocess_seconds','native_student_prefix_seconds','fresh_expert_seconds','adaptation_seconds','final_readout_seconds','total_seconds']
    for ds in ['hc2','vidstg']:
        rc=read(b/ds/'COMPLETION.json');path=p/ds/'QUERY_TIMINGS.json';assert sha(path)==rc['rows_sha256'];z=read(path);rr=z['rows'];assert len(rr)==16
        for r in rr:assert abs(math.fsum(r[k] for k in keys[:-1])+r['dispatcher_seconds']-r['total_seconds'])<1e-7
        datasets[ds]=dict(queries=16,resident_model_and_expert_startup_seconds=z['resident_model_and_expert_startup_seconds'],timings={k:dict(median=float(np.median([r[k] for r in rr])),p95=float(np.quantile([r[k] for r in rr],.95))) for k in keys},CUDA_peak_allocated_max=max(r['CUDA_peak_allocated'] for r in rr),CUDA_peak_reserved_max=max(r['CUDA_peak_reserved'] for r in rr),actual_new_DINO_calls=sum(r['actual_new_DINO_calls'] for r in rr),all_slow_and_no_admitted_queries_retained=True);allcount+=len(rr)
        for f,h in rc['files'].items():
            path=ROOT/f;assert sha(path)==h;z=load(path);accept=read(path.with_suffix('.json'));assert accept['status']=='pass' and accept['sha256']==h
            binding=next(v for v in read(b/'DESIGN_LOCK.json')['stages'][ds]['bindings'] if v['query_ordinal']==z['query_ordinal'])
            original=ROOT/binding['path'];assert sha(original)==z['original_fit_sha256']==binding['sha256'];old=load(original)
            ip=OLD/old['input']['path'];assert sha(ip)==z['original_input_sha256']==old['input']['sha256'];inp=normalized(load(ip))
            assert z['pixel_sha256']==inp['pixel_sha256'] and z['frame_ids']==inp['frame_ids']
            assert z['native']['indices']==inp['indices'] and z['native']['physical_interval']==inp['interval'] and torch.equal(z['native']['boxes'].cpu(),inp['native_boxes'])
            audit(z['fit'],unpack_expert(z['expert']));count=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
            equal(scientific(z['fit']),scientific(old['fit']),count);assert count==accept['original_full_fit_bitwise']
            matched+=1;mathrounds+=len(z['fit']['rounds']);bitwise_coordinates+=count['tensor_coordinates']
        print('COST_ROOT_COMPLETE_MATCHED_FITS',ds,flush=True)
    assert allcount==32
    assert matched==32
    write(b/'ROOT_COMPLETE_TIMED_FIT_READBACK.json',dict(status='pass',complete_matched_fits=matched,complete_math_rounds=mathrounds,bitwise_tensor_coordinates=bitwise_coordinates,native_pixel_frame_interval_box_bindings=True,independent_full_decoder_Jacobian=False,GT_read=False,time=time.time()))
    write(p/'END_TO_END_SUMMARY.json',dict(status='actual_complete_query_timing',datasets=datasets,query_count=32,weights_resident=True,cold_application_decode_cache=True,OS_page_cache_cold=False,saved_query_parse=True,engineering_recorder_in_timed_implementation=True,final_readout_counted_once=True,GT_read=False))
    plt=plotting();fig,axs=plt.subplots(1,2,figsize=(9,3.7));labels=['Decode','Student','Expert','Adapt','Readout']
    for ax,(ds,z) in zip(axs,datasets.items()):
        med=[z['timings'][k]['median'] for k in keys[:-1]];p95=[z['timings'][k]['p95'] for k in keys[:-1]];x=np.arange(5);ax.bar(x-.18,med,.36,color='#217ca3',label='Median');ax.bar(x+.18,p95,.36,color='#dc7555',label='p95');ax.set_xticks(x,labels,rotation=20);ax.set_title('HC-STVG2' if ds=='hc2' else 'VidSTG');ax.set_ylabel('Wall time (s)');ax.grid(axis='y',alpha=.15)
    axs[0].legend(frameon=False);fig.tight_layout();save_figure(fig,'cold_decode_end_to_end_cost',p);plt.close(fig)
    write(b/'ROOT_FIGURE_COMPLETION.json',dict(status='pending_actual_PDF_pixel_view',matched_complete_fits=32,png_pdf_pairs=1,GT_read=False,time=time.time()))

if __name__=='__main__':
    {'orders':orders,'baselines':baselines,'cost':cost}[sys.argv[1]]()
