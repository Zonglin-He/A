"""Finite CPU readback of already globally sealed P2 evidence. No model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections, gzip, json, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,status,sha,load
BASE=ROOT/'artifacts/stvg_opd_mechanism_review_v1'
PUB=ROOT/'results/stvg_opd_mechanism_review/2026-10-11'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
PAPER=ROOT/'artifacts/decota_paper_experiments_v1'
GROUPS=['observed','unobserved_decoder','interpolated','outside_native']

def partition(fids,decoder,observed,interval):
    fids=np.asarray(fids);inside=(fids>=interval[0])&(fids<interval[1]);obs=np.isin(fids,observed);dec=np.isin(fids,decoder)
    masks=[inside&obs,inside&~obs&dec,inside&~obs&~dec,~inside]
    assert np.all(np.sum(masks,axis=0)==1)
    return dict(zip(GROUPS,masks))

def independent_score(ious,fids,span,interval):
    a,z=interval;g,h=span;assert z>a and h>g
    overlap=max(0,min(z,h)-max(a,g));mask=(fids>=max(a,g))&(fids<min(z,h))
    return dict(v=float(np.sum(ious[mask])/(max(z,h)-min(a,g))),t=float(overlap/(z-a+h-g-overlap)),s=float(np.mean(ious)))

def contracts():
    m=partition([0,1,2,3,4],[0,2,4],[2],[1,4])
    assert [np.flatnonzero(m[k]).tolist() for k in GROUPS]==[[2],[],[1,3],[0,4]]
    # Annotation endpoint, disjoint/partial intervals and zero-extrapolation support.
    f=np.arange(5);iou=np.array([0.,.5,1.,.5,0.])
    assert independent_score(iou,f,[1,4],[0,5])==dict(v=.4,t=.6,s=.4)
    assert independent_score(iou,f,[1,4],[5,7])['v']==0
    assert not (.5<.5) and not (-.05<-.05)
    return dict(status='pass',valid_contracts=6,scope='disjoint partitions, half-open intervals, exact threshold boundaries; not GPU qualification',GT_read=False)

def prepare():
    if (BASE/'CPU_RUNTIME.json').exists():return verify()
    assert read(OLD/'P2_PREDICTION_BARRIER.json')['status']=='sealed'
    assert read(OLD/'P2_ROOT_CLOSING_RECEIPT.json')['status']=='complete'
    write(BASE/'CPU_CONTRACTS.json',contracts())
    files=['protocols/stvg_opd_mechanism_review_v1.md','scripts/analyze_stvg_opd_mechanism_review_v1.py',
      'scripts/review_stvg_opd_mechanism_review_v1.py','scripts/render_stvg_opd_mechanism_review_v1.py',
      'scripts/score_stvg_opd_p1_v1.py','vg_tta/tastvg_oracle_event5_v1.py','vg_tta/decota_paper_controls_v1.py',
      'scripts/review_stvg_opd_revised_p1_v2.py','scripts/stvg_opd_paper_later_common_v1.py',
      'scripts/stvg_opd_legacy_input_bridge003.py','scripts/run_decota_paper_main_v1.py',
      'methods/CURRENT_METHOD.json','methods/decota_spatial_opd_v1/configs.json',
      str((OLD/'P2_ROOT_CLOSING_RECEIPT.json').relative_to(ROOT)),str((OLD/'P2_PREDICTION_BARRIER.json').relative_to(ROOT)),
      str((OLD/'LATER_DESIGN_LOCK.json').relative_to(ROOT)),
      'results/stvg_opd_paper_hc2_revision/2026-10-08/P2/ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json']
    design=read(OLD/'LATER_DESIGN_LOCK.json')
    files += [str((OLD/'stages'/s/'PREDICTION_BARRIER.json').relative_to(ROOT)) for s in design['stages'] if s.startswith('P2_')]
    write(BASE/'CPU_RUNTIME.json',dict(status='locked_before_new_diagnostic_GT',pins={f:sha(ROOT/f) for f in files},GT_read=False,time=time.time()))
    return verify()

def verify():
    r=read(BASE/'CPU_RUNTIME.json')
    for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
    return r

def stats(rows,fields):
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['dataset'],r['source_id'])].append(r)
    result={}
    for field in fields:
        vals=np.array([np.mean([r[field] for r in rr if r.get(field) is not None]) for rr in groups.values() if any(r.get(field) is not None for r in rr)])
        if len(vals)==0:result[field]=dict(parents=0,mean=None,ci95=None);continue
        rng=np.random.default_rng(20261011);boot=[]
        for _ in range(100):boot.append(vals[rng.integers(0,len(vals),(100,len(vals)))].mean(1))
        result[field]=dict(parents=len(vals),mean=float(vals.mean()),ci95=np.quantile(np.concatenate(boot),[.025,.975]).tolist(),
          parent_negative_tail_5pp=int((vals<-.05).sum()),parent_p05=float(np.quantile(vals,.05)),cells=sum(r.get(field) is not None for r in rows))
    return dict(parent_sources=len(groups),cells=len(rows),metrics=result)

def aggregate(frames,keys,fields):
    by=collections.defaultdict(list)
    for r in frames:by[tuple(r[k] for k in keys)].append(r)
    return [{**dict(zip(keys,k)),**{f:float(np.mean([r[f] for r in rr])) for f in fields},'frames':len(rr)} for k,rr in by.items()]

QUALITY_FIELDS=['expert_GT_IoU','student_expert_IoU','max_first_reward','first_zero_fraction','all_zero_fraction',
 'Full_current','Direct_current','Full_minus_Direct_current','Full_minus_Direct_final','Before_difference',
 'Full_harm','Direct_harm','Direct_minus_Full_current']
PROP_FIELDS=['After_minus_Before','After_minus_Native','After_minus_DINO','Direct_minus_Before','DINO_minus_Native',
 'Native','Before','After','DINO','Direct','After_task_gated','Native_task_gated']

def summarize(quality,frames,cells):
    quality_stats={};prop_stats={};oracle_stats={};triggers=[]
    settings=sorted({r['setting'] for r in cells})
    for setting in settings:
        for pop in ['all_historical','excluding_development']:
            eligible=lambda r:r['setting']==setting and (pop=='all_historical' or not r['development'])
            qq=[r for r in quality if eligible(r)];ff=[r for r in frames if eligible(r)];cc=[r for r in cells if eligible(r)]
            item={}
            for q in ['correct','incorrect','all']:
                for a in ['zero','positive','all']:
                    rr=[r for r in qq if (q=='all' or r['quality']==q) and (a=='all' or r['reachability']==a)]
                    group=aggregate(rr,['dataset','source_id','condition','order'],QUALITY_FIELDS)
                    item[q+'/'+a]=dict(statistics=stats(group,QUALITY_FIELDS),expert_frame_cells=len(rr),round_frame_cells=sum(r['rounds'] for r in rr),
                       all_first_rewards_zero=sum(r['reachability']=='zero' for r in rr),constant_nonzero_first_rewards=sum(r['first_constant_nonzero'] for r in rr))
            quality_stats[setting+'/'+pop]=item
            pp={}
            pergroup={g:aggregate([r for r in ff if r['group']==g],['dataset','source_id','condition','order'],PROP_FIELDS) for g in GROUPS}
            for g,rows in pergroup.items():pp[g]=dict(statistics=stats(rows,PROP_FIELDS),frame_cells=sum(r['frames'] for r in rows))
            observed_parents={(r['dataset'],r['source_id']) for r in pergroup['observed']}
            common={}
            for g in GROUPS[1:]:
                intersection=observed_parents&{(r['dataset'],r['source_id']) for r in pergroup[g]}
                common[g]=dict(common_parents=len(intersection),observed=stats([r for r in pergroup['observed'] if (r['dataset'],r['source_id']) in intersection],PROP_FIELDS),
                    comparison=stats([r for r in pergroup[g] if (r['dataset'],r['source_id']) in intersection],PROP_FIELDS))
            prop_stats[setting+'/'+pop]=dict(all_eligible=pp,common_parent_intersections=common)
            fields=['Native_v','SpatialOracle_v','TemporalOracle_v','JointOracle_v','SupportJointOracle_v',
              'Spatial_headroom','Temporal_headroom','Native_event_coverage','unsupported_event_fraction','no_admission','no_update','missing_expert_GT']
            oracle_stats[setting+'/'+pop]=dict(pooled=stats(cc,fields),per_order={o:stats([r for r in cc if r['order']==o],fields) for o in sorted({r['order'] for r in cc})})
            if pop=='all_historical':
                correct=[r for r in qq if r['quality']=='correct'];zero=[r for r in correct if r['reachability']=='zero']
                parent_count=len({(r['dataset'],r['source_id']) for r in zero});fraction=len(zero)/len(correct) if correct else 0
                benefit=item['correct/all']['statistics']['metrics']['Direct_minus_Full_current']
                fire=parent_count>=10 and fraction>=.25 and benefit['ci95'] is not None and benefit['ci95'][0]>0
                triggers.append(dict(setting=setting,correct_expert_observations=len(correct),zero_correct_observations=len(zero),
                    zero_correct_parents=parent_count,zero_correct_fraction=fraction,Direct_minus_Full_current=benefit,triggered=fire))
    return dict(status='complete_postseal_CPU_diagnosis',quality=quality_stats,propagation=prop_stats,oracles=oracle_stats,
        pilot_resource_trigger=dict(status='triggered' if any(r['triggered'] for r in triggers) else 'not_triggered',settings=triggers),
        bootstrap=10000,seed=20261011,unadjusted_descriptive_CI=True,historically_exposed=True,
        Q_threshold=.5,harm_delta_threshold=-.05,actual_GPU_reward_used=True,GT_only_offline_after_old_global_seal=True,
        identity_switching_not_inferred=True,new_model_calls=0)

def run():
    verify();import torch;torch.set_num_threads(2)
    from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate
    activate()
    from scripts.stvg_opd_paper_later_common_v1 import phases,stage_definition,record_path,input_equivalent
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    from vg_tta.decota_paper_controls_v1 import dino_refine
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    PUB.mkdir(parents=True,exist_ok=True)
    write(BASE/'GT_EXPOSURE.json',dict(status='offline_only_after_complete_P2_and_actual_phase_root',GT_read=True,
      global_barrier_sha256=sha(OLD/'P2_PREDICTION_BARRIER.json'),P2_root_closing_sha256=sha(OLD/'P2_ROOT_CLOSING_RECEIPT.json'),
      runtime_sha256=sha(BASE/'CPU_RUNTIME.json'),new_model_calls=0,time=time.time()))
    quality=[];frames=[];cells=[];bindings=[];checks=0;start=time.time()
    design=read(OLD/'DESIGN_LOCK.json')
    for name in phases()['P2']:
        stage=stage_definition(name);ds=stage['dataset'];gt,spans,provenance=truths(ds,stage);plan=read(PAPER/ds/'PLAN.json')
        srcs={s:i for i,s in enumerate(sorted({r['source'] for r in plan['rows']}))}
        dev={plan['rows'][q]['source'] for q in design['datasets'][ds]['development_query_ordinals']}
        barrier=read(OLD/'stages'/name/'PREDICTION_BARRIER.json');assert barrier['status']=='sealed' and not barrier['GT_read']
        for condition in stage['conditions']:
            for order,seq in stage['orders'].items():
                for at,q in enumerate(seq):
                    loaded={}
                    for arm in ['on_policy','direct_L1_GIoU']:
                        path,b=record_path(barrier,condition,order,arm,at);rc=read(path.with_suffix('.json'))
                        assert sha(path)==rc['sha256']==b['sha256'] and path.stat().st_size==rc['bytes'] and not rc['GT_read']
                        z=load(path);ip=OLD/z['input']['path'];assert sha(ip)==z['input']['sha256']
                        inp=normalized(load(ip));loaded[arm]=(z,inp);bindings.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),input_path=str(ip.relative_to(ROOT)),input_sha256=sha(ip)))
                    z,inp=loaded['on_policy'];direct,dinp=loaded['direct_L1_GIoU'];input_equivalent(inp,dinp)
                    row=plan['rows'][q];assert inp['frame_ids']==row['frame_ids'] and z['interval']==direct['interval']==inp['interval']
                    fit=z['fit'];dfit=direct['fit'];ex=unpack_expert(inp['expert']);support=top1_support(ex);assert [p for p,e in support]==fit['positions']==dfit['positions']
                    refine,refmeta=dino_refine(inp['native_boxes'].numpy(),row['frame_ids'],ex)
                    tubes={};values={}
                    for label,boxes in [('Native',inp['native_boxes'].numpy()),('Before',fit['before'].numpy()),('After',fit['final'].numpy()),
                          ('DirectBefore',dfit['before'].numpy()),('Direct',dfit['final'].numpy()),('DINO',refine)]:
                        tube=DenseTube(boxes,row,gt[q],spans[q],clip=ds=='hc2');tubes[label]=tube
                        v=official(boxes,row,gt[q],spans[q],z['interval'],ds);ind,fids,ious=dense_metric(boxes,row,gt[q],spans[q],z['interval'],ds)
                        assert all(abs(v[k]-ind[k])<2e-10 for k in v) and np.allclose(tube.iou,ious,rtol=0,atol=2e-10)
                        values[label]=v;checks+=3
                    native=tubes['Native'];fids=native.fids;span=spans[q];interval=z['interval'];event=(fids>=span[0])&(fids<span[1])
                    saved=(fids>=row['frame_ids'][0])&(fids<=row['frame_ids'][-1]);inside=(fids>=interval[0])&(fids<interval[1])
                    spatial=official(inp['native_boxes'].numpy(),row,gt[q],span,interval,ds,gt_space=True)
                    temporal=official(inp['native_boxes'].numpy(),row,gt[q],span,span,ds)
                    joint=official(inp['native_boxes'].numpy(),row,gt[q],span,span,ds,gt_space=True)
                    for val,ious,when in [(spatial,np.ones(len(fids)),interval),(temporal,native.iou,span),(joint,np.ones(len(fids)),span)]:
                        iv=independent_score(ious,fids,span,when);assert all(abs(val[k]-iv[k])<2e-10 for k in val);checks+=3
                    sj=independent_score(saved.astype(float),fids,span,span)
                    common=dict(dataset=ds,source_id=srcs[row['source']],condition=condition,order=order,
                      setting=ds+('/cross_clean' if condition=='clean' else '/same_corruption'),development=row['source'] in dev,arrival=at)
                    cells.append(dict(common,Native_v=values['Native']['v'],SpatialOracle_v=spatial['v'],TemporalOracle_v=temporal['v'],JointOracle_v=joint['v'],SupportJointOracle_v=sj['v'],
                      Spatial_headroom=spatial['v']-values['Native']['v'],Temporal_headroom=temporal['v']-values['Native']['v'],
                      Native_event_coverage=float(np.mean(inside[event])),unsupported_event_fraction=float(np.mean(~saved[event])),
                      no_admission=float(not support),no_update=float(fit['gradient_calls']==0),missing_expert_GT=sum(row['frame_ids'][p] not in gt[q] for p,e in support),
                      values=values,oracle_values=dict(Spatial=spatial,Temporal=temporal,Joint=joint,SupportJoint=sj)))
                    masks=partition(fids,row['frame_ids'],[row['frame_ids'][p] for p,e in support],interval)
                    for g,mask in masks.items():
                        for i in np.flatnonzero(mask&event):
                            v={label:float(tubes[label].iou[i]) for label in tubes}
                            frames.append(dict(common,group=g,anonymous_GT_event_index=int(i),saved_decoder_support=bool(saved[i]),
                              Native=v['Native'],Before=v['Before'],After=v['After'],DINO=v['DINO'],Direct=v['Direct'],
                              After_minus_Before=v['After']-v['Before'],After_minus_Native=v['After']-v['Native'],After_minus_DINO=v['After']-v['DINO'],
                              Direct_minus_Before=v['Direct']-v['DirectBefore'],DINO_minus_Native=v['DINO']-v['Native'],
                              After_task_gated=v['After']*float(inside[i]),Native_task_gated=v['Native']*float(inside[i])))
                    for j,(p,e) in enumerate(support):
                        fid=row['frame_ids'][p]
                        if fid not in gt[q]:continue
                        i=int(np.flatnonzero(fids==fid)[0]);rewards=[r['rollout']['rewards'].numpy()[j] for r in fit['rounds']]
                        assert rewards and all(len(r)==32 for r in rewards)
                        first=rewards[0];qual=float(box_iou(xyxy(np.asarray(e),row['input']['width'],row['input']['height']),gt[q][fid]))
                        sr=float(box_iou(xyxy(fit['before'][p].numpy(),row['input']['width'],row['input']['height']),xyxy(np.asarray(e),row['input']['width'],row['input']['height'])))
                        fc=float(tubes['After'].iou[i]-tubes['Before'].iou[i]);dc=float(tubes['Direct'].iou[i]-tubes['DirectBefore'].iou[i])
                        quality.append(dict(common,observation_index=j,expert_GT_IoU=qual,student_expert_IoU=sr,max_first_reward=float(first.max()),
                          first_zero_fraction=float(np.mean(first==0)),all_zero_fraction=float(np.mean(np.stack(rewards)==0)),
                          quality='correct' if qual>=.5 else 'incorrect',reachability='zero' if first.max()==0 else 'positive',
                          first_constant_nonzero=bool(first.min()==first.max() and first.max()>0),rounds=len(rewards),
                          Full_current=fc,Direct_current=dc,Full_minus_Direct_current=fc-dc,Direct_minus_Full_current=dc-fc,
                          Full_minus_Direct_final=float(tubes['After'].iou[i]-tubes['Direct'].iou[i]),Before_difference=float(tubes['Before'].iou[i]-tubes['DirectBefore'].iou[i]),
                          Full_harm=float(fc<-.05),Direct_harm=float(dc<-.05)))
                    if at%16==0:
                        state=dict(status='CPU_reading_only_complete_sealed_P2',stage=name,condition=condition,order=order,arrival=at,
                          cells=len(cells),CPU_only=True,new_model_calls=0,GT_read='offline_after_global_seal',pid=os.getpid(),time=time.time())
                        status(BASE/'CPU_STATUS.json',state);print('MECHANISM_CPU_PROGRESS',name,condition,order,at,len(cells),flush=True)
    assert len(cells)==1792 and len(bindings)==3584
    for name,rows in [('QUALITY_ROWS',quality),('FRAME_ROWS',frames),('CELL_ROWS',cells)]:
        with gzip.open(PUB/(name+'.jsonl.gz'),'wt',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    write(BASE/'PRIVATE_INPUT_BINDINGS.json',dict(status='all_sealed_fit_input_bytes_checked',records=bindings,GT_read=True))
    summary=summarize(quality,frames,cells);write(PUB/'SUMMARY.json',summary)
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_independent_root_statistics_and_PDF_view_publication',scope='sealed_P2_diagnostic_only',
      matched_cells=len(cells),complete_sealed_fit_reads=len(bindings),quality_frame_cells=len(quality),propagation_frame_cells=len(frames),independent_dense_oracle_scalar_checks=checks,
      new_model_calls=0,CPU_only=True,GT_read='offline_after_original_global_seal',
      files={str(p.relative_to(ROOT)):sha(p) for p in PUB.iterdir() if p.is_file()},elapsed_seconds=time.time()-start,time=time.time()))
    status(BASE/'CPU_STATUS.json',dict(status='pending_independent_root_statistics_and_PDF_view_publication',matched_cells=len(cells),CPU_only=True,new_model_calls=0,time=time.time()))

if __name__=='__main__':
    try:prepare() if len(sys.argv)>1 and sys.argv[1]=='prepare' else run()
    except BaseException:
        p=BASE/'CPU_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc());raise
