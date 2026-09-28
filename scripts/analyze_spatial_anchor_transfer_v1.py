"""Offline GT response matrices; no result-based deployment selection."""
import argparse,collections,itertools,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import spearmanr
from scripts.decota_matrix_common_v1 import read,write,load,save,sha
from scripts.run_spatial_anchor_transfer_v1 import OUT,verify
from scripts.analyze_spatial10_components_v1 import checked_score,aggregate,summary
from vg_tta.box_stability_diagnostics_v1 import overlap
METRICS=['sIoU','vIoU_corrected','tIoU']


def mean(a,mask):return float(np.asarray(a)[mask].mean()) if mask.any() else None
def rho(a,b):return float(spearmanr(a,b).statistic) if len(a)>2 and np.ptp(a)>0 and np.ptp(b)>0 else None


def cached():
    p=verify();assert sha(p['labels'])==p['labels_sha256'];labels=read(p['labels']);rows=[];anchors=[];pairs=[];matrices={}
    for r in p['rows']:
        assert sha(r['single'])==r['single_sha256'];z=load(r['single']);x=load(r['path']);gt=labels[r['key']]
        ids=z['frame_ids'];valid=np.array(gt['valid'],bool);truth=np.array(gt['boxes']);native=z['baseline']['boxes'].numpy()
        nq=overlap(native,truth);observed=np.zeros(len(ids),bool);observed[x['actual_observation_positions']]=True;unobserved=valid&~observed
        base,_=checked_score(native,gt,ids,z['baseline']['indices']);current,cq=checked_score(z['current']['boxes'].numpy(),gt,ids,z['current']['indices'])
        aa=[];matrix=[];displacements=[];metrics=[]
        for i,fit in enumerate(z['fits']):
            b=fit['prediction']['boxes'].numpy();q=overlap(b,truth);d=q-nq;pos=fit['anchor']['position']
            expert=np.array([fit['anchor']['box']]);quality=float(overlap(expert,truth[[pos]])[0]) if valid[pos] else None
            before=float(overlap(native[[pos]],expert)[0]);after=float(overlap(b[[pos]],expert)[0]);local=float(d[pos]) if valid[pos] else None
            u=mean(d,unobserved);adv=quality-float(nq[pos]) if quality is not None else None
            if quality is None:category='unknown_reference'
            elif adv<=-.05 and after-before>.001 and u is not None and u<-.001:category='worse_reference_propagation'
            elif local>.001 and u is not None and u<-.001:category='transfer_failure'
            elif adv>=.05 and local<=.001:category='absorption_failure'
            elif local>.001 and u is not None and u>.001:category='absorption_and_transfer'
            else:category='mixed_or_neutral'
            disp=np.linalg.norm(b-native,axis=1);dist=np.abs(np.array(ids)-ids[pos])/x['input']['fps']
            geom=1-overlap(np.repeat(truth[[pos]],len(ids),axis=0),truth)
            metric,_=checked_score(b,gt,ids,z['baseline']['indices']);metrics.append(metric)
            row=dict(key=r['key'],source=r['source'],cohort=r['cohort'],anchor_index=i,position=pos,frame_id=ids[pos],
                category=category,teacher_iou=quality,teacher_advantage=adv,native_local=float(nq[pos]) if valid[pos] else None,
                local_gain=local,teacher_agreement_before=before,teacher_agreement_after=after,unobserved_gain=u,
                unobserved_count=int(unobserved.sum()),all_gain=metric['sIoU']-base['sIoU'],v_gain=metric['vIoU_corrected']-base['vIoU_corrected'],
                selected_step=fit['selected_step'],loss_initial=fit['losses'][0],loss_selected=fit['losses'][fit['selected_step']],
                local_displacement=float(disp[pos]),unobserved_displacement=mean(disp,unobserved),
                rho_distance=rho(dist[unobserved],d[unobserved]),rho_GT_geometry=rho(geom[unobserved],d[unobserved]) if valid[pos] else None)
            aa.append(row);anchors.append(row);matrix.append(np.where(valid,d,np.nan));displacements.append(disp)
        for a,b in itertools.combinations(aa,2):
            if a['teacher_iou'] is None or b['teacher_iou'] is None:continue
            if abs(a['teacher_iou']-b['teacher_iou'])<=.05:
                ma,mb=matrix[a['anchor_index']],matrix[b['anchor_index']]
                opposite=((ma>.001)&(mb<-.001))|((ma<-.001)&(mb>.001))
                pairs.append(dict(key=r['key'],source=r['source'],cohort=r['cohort'],a=a['anchor_index'],b=b['anchor_index'],
                    quality_difference=abs(a['teacher_iou']-b['teacher_iou']),utility_difference=abs(a['all_gain']-b['all_gain']),
                    both_quality_ge_half=a['teacher_iou']>=.5 and b['teacher_iou']>=.5,
                    response_rms=float(np.sqrt(np.mean((ma[valid]-mb[valid])**2))),
                    opposite_sign_fraction=float(opposite[valid].mean())))
        # Tensor files retain missing GT as NaN; never replace with background labels.
        dest=OUT/'matrices'/f'{r["key"].replace(":","_")}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,M=np.array(matrix).reshape(-1,len(ids)),displacement=np.array(displacements).reshape(-1,len(ids)),
            native_quality=np.where(valid,nq,np.nan),current_quality=np.where(valid,cq,np.nan),valid=valid,observed=observed,
            frame_ids=np.array(ids),time_seconds=(np.array(ids)-ids[0])/x['input']['fps'],anchor_positions=[a['position'] for a in aa])
        matrices[str(dest)]=sha(dest)
        rows.append(dict(key=r['key'],group=r['source'],cohort=r['cohort'],caption=x['input']['caption'],anchors=len(aa),
            arms={'Native':base,'Current_C':current,'Best_single_oracle':max(metrics,key=lambda m:m['sIoU']) if metrics else base},
            unobserved_current_gain=mean(np.array(cq)-nq,unobserved)))
    stats={}
    for c in p['configs']:
        ar=[a for a in anchors if a['cohort']==c];rr=[r for r in rows if r['cohort']==c];pr=[a for a in pairs if a['cohort']==c]
        stats[c]=dict(summary=summary(rr,['Native','Current_C','Best_single_oracle'],METRICS,[('Current_C','Native')]),
            anchor_attempts=len(ar),counts=dict(collections.Counter(a['category'] for a in ar)),
            category_sources={cat:len({a['source'] for a in ar if a['category']==cat}) for cat in set(a['category'] for a in ar)},
            unobserved_current_gain=aggregate([r['unobserved_current_gain'] for r in rr],[r['group'] for r in rr]),
            same_quality_pairs=len(pr),same_quality_pair_sources=len({a['source'] for a in pr}),
            same_quality_utility_gap=aggregate([a['utility_difference'] for a in pr],[a['source'] for a in pr]),
            descriptive={k:aggregate([a[k] for a in ar],[a['source'] for a in ar]) for k in ['rho_distance','rho_GT_geometry']})
    write(OUT/'ANCHORS.json',anchors);write(OUT/'CACHED_SOURCES.json',rows);write(OUT/'MATCHED_PAIRS.json',pairs)
    write(OUT/'CACHED_SUMMARY.json',stats);write(OUT/'MATRIX_MANIFEST.json',matrices)
    chosen=[]
    categories=['absorption_and_transfer','absorption_failure','transfer_failure','worse_reference_propagation']
    for c in p['configs']:
        ar=sorted([a for a in anchors if a['cohort']==c and a['teacher_iou'] is not None],key=lambda a:(a['key'],a['anchor_index']))
        used=set()
        for category in categories:
            match=next((a for a in ar if a['category']==category and a['key'] not in used),None)
            if match is not None:chosen.append(dict(key=match['key'],cohort=c,anchor_index=match['anchor_index'],category=category));used.add(match['key'])
        for a in ar:
            if len(used)>=4:break
            if a['key'] not in used:chosen.append(dict(key=a['key'],cohort=c,anchor_index=a['anchor_index'],category='coverage_fill'));used.add(a['key'])
    write(OUT/'INTERVENTION_LOCK.json',dict(rows=chosen,label_informed_case_selection=True,deployment_GT=False,
        selection_rule='first lexicographic distinct source per category then fill, pre-perturbation',anchor_results_sha256=sha(OUT/'ANCHORS.json')))
    print('CACHED',len(rows),'anchors',len(anchors),'selected',chosen)
    for c,s in stats.items():print(c,s['counts'])


def evaluate():
    p=verify();barrier=read(OUT/'PREDICTION_BARRIER.json');assert len(barrier['files'])==64
    for f,h in barrier['files'].items():assert sha(f)==h
    labels=read(p['labels']);rows=[];changes=[]
    for r in p['rows']:
        x=load(OUT/'runs'/f'{r["key"].replace(":","_")}.pt');gt=labels[r['key']];old=load(r['path'])
        valid=np.array(gt['valid'],bool);observed=np.zeros(len(valid),bool);observed[old['actual_observation_positions']]=True;u=valid&~observed
        arms={}
        for name,z in x['outputs'].items():arms[name]=checked_score(z['boxes'].numpy(),gt,x['frame_ids'],x['indices'])[0]
        rows.append(dict(key=r['key'],group=r['source'],cohort=r['cohort'],arms=arms))
        if x['interventions']:
            base=x['interventions']['repeat'];b0=base['boxes'].numpy();q0=overlap(b0,np.array(gt['boxes']));pos=base['anchor']['position']
            for name,z in x['interventions'].items():
                if name=='repeat':continue
                b=z['boxes'].numpy();q=overlap(b,np.array(gt['boxes']));fixed=z['path'][base['selected_step']]['boxes'].numpy()
                fq=overlap(fixed,np.array(gt['boxes']));disp=np.linalg.norm(b-b0,axis=1)
                inputnorm=float(np.linalg.norm(z['info']['actual_delta']));response=mean(disp,u)
                record=dict(key=r['key'],source=r['source'],cohort=r['cohort'],condition=name,
                    selected_step=z['selected_step'],original_step=base['selected_step'],step_changed=z['selected_step']!=base['selected_step'],
                    local_delta=float(q[pos]-q0[pos]) if valid[pos] else None,unobserved_delta=mean(q-q0,u),
                    all_delta=mean(q-q0,valid),fixed_step_unobserved_delta=mean(fq-q0,u),
                    local_box_response=float(disp[pos]),unobserved_box_response=response,input_box_change=inputnorm,
                    response_input_ratio=response/inputnorm if response is not None and inputnorm>0 else None)
                changes.append(record)
    names=['Native','Own','Donor','Donor_normmatched','Mean_LOSO_normmatched'];comparisons=[('Own',n) for n in names if n!='Own']
    stats={c:summary([r for r in rows if r['cohort']==c],names,METRICS,comparisons) for c in p['configs']}
    write(OUT/'SWAP_SOURCES.json',rows);write(OUT/'SWAP_SUMMARY.json',stats);write(OUT/'PERTURBATION_RESULTS.json',changes)
    sensitivity={c:{'n':len([r for r in changes if r['cohort']==c]),
        'sources':len({r['source'] for r in changes if r['cohort']==c}),
        'best_step_changed':sum(r['step_changed'] for r in changes if r['cohort']==c),
        'statistics':{k:aggregate([r[k] for r in changes if r['cohort']==c],[r['source'] for r in changes if r['cohort']==c]) for k in
         ['unobserved_delta','local_delta','fixed_step_unobserved_delta','response_input_ratio']}} for c in p['configs']}
    write(OUT/'PERTURBATION_SUMMARY.json',sensitivity)
    for c,s in stats.items():print(c,{n:{k:round(s['arms'][n][k]['mean']*100,3) for k in METRICS} for n in names})
    print('sensitivity',sensitivity)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['cached','evaluate']);x=a.parse_args();cached() if x.command=='cached' else evaluate()
