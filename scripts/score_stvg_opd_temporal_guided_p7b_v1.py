"""Complete sealed P7-B source/math/state/dense and coupling diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_temporal_guided_p7_common_v1 import *
from scripts.score_stvg_opd_temporal_capture_p7a_v1 import array,near,rows_read,rows_write
PUB=OUT/'P7_B';ARMS=['A_original','B_posthoc','C_temporal_guided_OPD','D_temporal_guided_Direct']

def seal():
    from scripts.run_stvg_opd_temporal_guided_p7b_v1 import verify_b
    verify_b();b=read(B/'GLOBAL_PREDICTION_BARRIER.json');assert b['status']=='sealed' and b['actual_complete_GPU_fits']==1024 and not b['GT_read']
    assert b['expert_barrier_sha256']==sha(B/'EXPERT_EVIDENCE_BARRIER.json');return b

def aggregate(rows):
    result={}
    for ds in ['hc2','vidstg']:
        result[ds]={}
        for subset in ['all128','previous_P6_32','added96']:
            result[ds][subset]={}
            for order in ['both','order1','order2']:
                rr=[r for r in rows if r['dataset']==ds and (subset=='all128' or r['previous_P6']==(subset=='previous_P6_32')) and (order=='both' or r['order']==order)]
                by=collections.defaultdict(list)
                for r in rr:by[r['parent_id']].append(r)
                fields=[k for k in rr[0] if k not in ['public_cell','parent_id','previous_P6','dataset','order']]
                block={}
                for f in fields:
                    vals=[math.fsum(r[f] for r in rs if r[f] is not None)/sum(r[f] is not None for r in rs) for rs in by.values() if any(r[f] is not None for r in rs)]
                    x=np.asarray(vals,float)
                    if not len(x):block[f]=dict(parents=0,mean=None,ci95=None,undefined_parents=len(by));continue
                    gen=np.random.default_rng(20261012);bootstrap=[]
                    for _ in range(100):bootstrap.append(x[gen.integers(len(x),size=(100,len(x)))].mean(1))
                    block[f]=dict(parents=len(x),mean=float(x.mean()),ci95=np.quantile(np.concatenate(bootstrap),[.025,.975]).tolist(),
                        undefined_parents=len(by)-len(x),parent_values=x.tolist(),harm_5pp=int(sum(x<-.05)),harm_20pp=int(sum(x<-.2)),
                        positive_parents=int(sum(x>1e-12)),negative_parents=int(sum(x< -1e-12)))
                result[ds][subset][order]=dict(cells=len(rr),parents=len(by),statistics=block)
    return result

def actual_rows(independent=False):
    import torch
    torch.set_num_threads(2);barrier=seal()
    from scripts.score_stvg_opd_temporal_capture_p7a_v1 import context_truth
    gt,spans=context_truth()
    from scripts.run_decota_paper_main_v1 import read_row,unpack_expert
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from vg_tta.tastvg_oracle_event5_v1 import official,box_iou
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import mathematical_readback
    from scripts.stvg_opd_design_ablation_common_v1 import audit as OPD_audit
    from vg_tta.stvg_opd_paper_component_audit_v1 import audit as Direct_audit
    rows=[];counts=collections.Counter();eb=read(B/'EXPERT_EVIDENCE_BARRIER.json')
    for r in read(BASE/'DESIGN_LOCK.json')['records']:
        ds=r['dataset'];q=r['query_ordinal'];raw=read_row(ds,q);truth=gt[ds][q];span=spans[ds][q]
        a=load(A/'readouts'/f"{r['public_cell']:05}.pt");old=load(ROOT/r['original_Full']['path']);inp=normalized(load(ROOT/r['original_Full']['input_path']))
        assert sha(ROOT/r['original_Full']['path'])==r['original_Full']['sha256'] and sha(ROOT/r['original_Full']['input_path'])==r['original_Full']['input_sha256']
        ep=B/'expert_I1'/f"{r['capture_index']:05}.pt";assert sha(ep)==eb['files'][str(ep.relative_to(ROOT))];ex=load(ep);expert=unpack_expert(ex['expert'])
        boxes=dict(A_original=a['boxes']['OPD'],B_posthoc=a['boxes']['OPD']);intervals=dict(A_original=a['intervals']['T0'],B_posthoc=a['intervals']['T1'])
        for arm in ARMS[2:]:
            p=B/'formal'/f"{r['public_cell']:05}"/(arm+'_repeat0.pt');h=sha(p);assert h==barrier['files'][str(p.relative_to(ROOT))];z=load(p)
            assert z['interval']==a['intervals']['T1']==ex['interval'] and z['Native_interval']==inp['interval']
            assert z['pixel_sha256']==inp['pixel_sha256']==ex['pixel_sha256'] and z['frame_ids']==inp['frame_ids']
            assert z['source_model_sha256']==inp['source_model_state_sha256']
            fit=z['fit'];assert all(torch.equal(v,old['fit']['initial'][k]) for k,v in fit['initial'].items());assert torch.equal(fit['before'],old['fit']['before'])
            assert fit['positions']==[int(v['position']) for v in ex['expert']['anchors']['single4']] and fit['config']==r['config'] and not fit['GT_used']
            if independent:
                checked=mathematical_readback(fit);assert checked==z['math_audit']['state_Adam_audit'];counts['math_scalars']+=checked['scalar_checks']
                assert (OPD_audit if arm==ARMS[2] else Direct_audit)(fit,expert)==z['math_audit']['objective_audit'];counts['whole_math_fits']+=1
            boxes[arm]=fit['final'];intervals[arm]=z['interval'];counts['full_state_coordinates']+=1792*(len(fit['path'])+2);counts['complete_fits']+=1
            assert sha(p)==h;counts['double_hashes']+=2
        row={k:r[k] for k in ['public_cell','dataset','parent_id','order','previous_P6']};curves={}
        for arm in ARMS:
            dense,fids,iou=dense_metric(array(boxes[arm]),raw,truth,span,intervals[arm],ds);curves[arm]=iou
            if not independent:counts['official_dense_values']+=near(dense,official(array(boxes[arm]),raw,truth,span,intervals[arm],ds))
            for metric,value in dense.items():row[f'{arm}_{metric}']=value
            for t in [.3,.5]:row[f'{arm}_R{int(t*100)}']=float(dense['v']>t)
        for pair,left,right in [('coupling_C_minus_B',ARMS[2],ARMS[1]),('D_minus_B',ARMS[3],ARMS[1]),('C_minus_D',ARMS[2],ARMS[3]),('B_minus_A',ARMS[1],ARMS[0])]:
            for metric in ['v','t','s','R30','R50']:row[f'{pair}_{metric}']=row[f'{left}_{metric}']-row[f'{right}_{metric}']
        assert row['A_original_s']==row['B_posthoc_s'] and row['B_posthoc_t']==row['C_temporal_guided_OPD_t']==row['D_temporal_guided_Direct_t']
        ids=np.asarray(sorted(truth));event=(ids>=span[0])&(ids<span[1]);real=np.isin(ids,raw['frame_ids']);supported=(ids>=raw['frame_ids'][0])&(ids<=raw['frame_ids'][-1])
        original_visited=inp['expert']['actual_observation_positions'];new_visited=ex['expert']['actual_observation_positions']
        union_frames=[raw['frame_ids'][p] for p in sorted(set(original_visited)|set(new_visited))]
        obs=np.isin(ids,union_frames);a0,b0=intervals['A_original'];a1,b1=intervals['B_posthoc'];I0=(ids>=a0)&(ids<b0);I1=(ids>=a1)&(ids<b1)
        masks=dict(common_unobserved_real=event&real&~obs,visited_union_real=event&real&obs,interpolation_only=event&supported&~real,
            gained_event_outside_Native=event&I1&~I0,gained_event_unobserved_real=event&I1&~I0&real&~obs,
            lost_event_support=event&I0&~I1,event_outside_Native=event&~I0,unsupported_event=event&~supported)
        for name,mask in masks.items():
            row[f'{name}_frames']=int(mask.sum())
            for pair,left,right in [('C_minus_B',ARMS[2],ARMS[1]),('D_minus_B',ARMS[3],ARMS[1]),('C_minus_D',ARMS[2],ARMS[3])]:
                row[f'{pair}_{name}_IoU']=float((curves[left]-curves[right])[mask].mean()) if mask.any() else None
        # Offline spatial diagnostic only; these GT values never choose I1 or
        # observations, and all DINO/fit outputs are globally sealed first.
        newids=[raw['frame_ids'][p] for p in new_visited];oldids=[raw['frame_ids'][p] for p in original_visited]
        row['new_visited_frames']=len(newids);row['old_visited_frames']=len(oldids);row['new_positions_not_visited_before']=len(set(newids)-set(oldids))
        row['I1_visited_inside_event']=sum(span[0]<=f<span[1] for f in newids);row['I1_visited_outside_event']=sum(not span[0]<=f<span[1] for f in newids)
        row['I1_planned_positions']=ex['planned_observations'];row['I1_actual_DINO_calls_unique_pack']=ex['actual_new_DINO_calls'];row['I1_admitted_positions']=len(ex['expert']['anchors']['single4'])
        row['DINO_pack_actual_seconds']=ex['actual_seconds'];row['DINO_pack_forward_seconds']=ex['actual_forward_seconds']
        # Allocation quality is IoU, not an inferred identity-error label.
        quality=[]
        for v in ex['expert']['anchors']['single4']:
            frame=raw['frame_ids'][v['position']]
            if frame in truth and span[0]<=frame<span[1]:
                pred=xyxy(np.array([v['box']]),raw['input']['width'],raw['input']['height'])[0]
                if ds=='hc2':pred=np.maximum(pred,0)
                quality.append(float(box_iou(pred,np.asarray(truth[frame]))))
        row['I1_offline_admitted_expert_quality']=float(np.mean(quality)) if quality else None
        rows.append(row);status(B/('ROOT_STATUS.json' if independent else 'CPU_STATUS.json'),dict(status='actual_full_math_dense_coupling_root' if independent else 'actual_postseal_all512_coupling_scores',done=len(rows),total=512,GT_read=True,CPU_only=True,time=time.time()))
    assert len(rows)==512;return rows,dict(counts)

def score():
    barrier=seal();write(B/'GT_EXPOSURE.json',dict(status='after_all1024_fit_and_all_A_B_bindings_sealed',barrier_sha256=sha(B/'GLOBAL_PREDICTION_BARRIER.json'),GT_for_choice=False,time=time.time()))
    rows,c=actual_rows();rows_write(PUB/'ROWS.jsonl.gz',rows);write(PUB/'SUMMARY.json',aggregate(rows))
    write(B/'CPU_COMPLETION.json',dict(status='pending_actual_independent_root_PDF_view_publication_archive',cells=512,arms=4,logical_outputs=2048,
        new_formal_complete_GPU_fits=1024,counts=c,files={str(f.relative_to(ROOT)):sha(f) for f in [PUB/'ROWS.jsonl.gz',PUB/'SUMMARY.json']},CPU_only=True,GT_read=True,time=time.time()))

def root():
    assert read(B/'CPU_COMPLETION.json')['GT_read'];fresh,counts=actual_rows(True);old=rows_read(PUB/'ROWS.jsonl.gz');checks=near(fresh,old)
    # Independent parent bootstrap builds the complete resampling array and
    # recomputes each saved statistic from all anonymous rows.
    summary=read(PUB/'SUMMARY.json')
    for ds,data in summary.items():
        for subset,orders in data.items():
            for order,block in orders.items():
                rr=[r for r in fresh if r['dataset']==ds and (subset=='all128' or r['previous_P6']==(subset=='previous_P6_32')) and (order=='both' or r['order']==order)]
                by=collections.defaultdict(list)
                for r in rr:by[r['parent_id']].append(r)
                assert len(by)==block['parents']
                for field,v in block['statistics'].items():
                    values=[math.fsum(r[field] for r in group if r[field] is not None)/sum(r[field] is not None for r in group) for group in by.values() if any(r[field] is not None for r in group)]
                    if not values:checks+=near(dict(parents=0,mean=None,ci95=None,undefined_parents=len(by)),v);continue
                    x=np.array(values);gen=np.random.default_rng(20261012);indices=gen.integers(len(x),size=(10000,len(x)));bootstrap=x[indices].sum(1)/len(x)
                    expected=dict(parents=len(x),mean=math.fsum(x)/len(x),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),undefined_parents=len(by)-len(x),parent_values=x.tolist(),
                        harm_5pp=int(sum(x<-.05)),harm_20pp=int(sum(x<-.2)),positive_parents=int(sum(x>1e-12)),negative_parents=int(sum(x< -1e-12)))
                    checks+=near(expected,v)
    write(B/'ROOT_READBACK.json',dict(status='pass',scope='all1024 actual complete fits/grad-Adam-state-objective/full source-input-expert binding/dense2048readouts/observed-unobserved/gained-lost/allsubset bootstrap',
        scalar_checks=checks,counts=counts,cells=512,parents=256,CPU_only=True,GT_read=True,time=time.time()))
    conclusions={}
    for ds in ['hc2','vidstg']:
        stats=summary[ds]['added96']['both']['statistics'];c=stats['coupling_C_minus_B_v'];u=stats['C_minus_B_common_unobserved_real_IoU'];d=stats['C_minus_D_v']
        conclusions[ds]=dict(coupling=c,unobserved_real=u,OPD_minus_Direct=d,
            supports_positive_coupling=bool(c['ci95'][0]>0),supports_unobserved_real_improvement=bool(u['ci95'] is not None and u['ci95'][0]>0),
            does_not_establish_OPD_necessity=not bool(d['ci95'][0]>0))
    write(B/'CANDIDATE_MECHANISM_DECISION.json',dict(status='measured_after_actual_complete_root',settings=conclusions,
        misallocation_hypothesis_supported_in_both_added96=all(v['supports_positive_coupling'] and v['supports_unobserved_real_improvement'] for v in conclusions.values()),
        bidirectional_joint_optimization=False,temporal_parameter_TTA=False,production_promoted=False,GT_online=False,
        full_official_query_expansion_authorized=False,time=time.time()))

if __name__=='__main__':{'score':score,'root':root}[sys.argv[1]]()
