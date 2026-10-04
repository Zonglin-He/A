"""Post-seal mechanism readback. No online rule or selection is modified."""
import sys,time,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_actuation_scope_common_v1 import *
from scripts.score_decota_actuation_scope_v1 import stats,path_numpy
import numpy as np

def run():
    from scripts import tastvg_decota_c1_common_v1 as c1
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,box_iou
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    import torch
    verify();torch.set_num_threads(4)
    contrasts={};cases={};details=[];resources={};label_provenance={}
    for ds in DATASETS:
        label_provenance[ds]={}
        for split in ['search','confirm']:
            exposure=c1.POOL/ds/f'GT_EXPOSURE_{split}.json';labels=c1.POOL/ds/f'GT_LABELS_{split}.json'
            old=read(exposure);assert sha(labels)==old['labels_sha256']
            label_provenance[ds][split]=dict(labels_sha256=sha(labels),upstream_exposure_sha256=sha(exposure),
                upstream_annotation_hashes=old['hashes'],sources=old['sources'],historical_exposure=True,
                labels_unchanged_from_predecessor=True)
    for stage in ['R3','R3G','R4']:
        folder=PUB/stage
        if not folder.exists():continue
        assert read(folder/'ROOT_AUDIT.json')['status']=='pass'
        barrier=read(BASE/stage/'GLOBAL_PREDICTION_BARRIER.json');assert barrier['arrivals']==1152
        rr=read(folder/'ROWS.json');lookup={(r['dataset'],r['split'],r['condition'],r['order'],r['arrival'],r['arm']):r for r in rr}
        pairs={'track_minus_frame_sum':('track','frame_sum'),'track_authority_minus_track':('track_authority','track')} if stage=='R3' else {'u_only_minus_joint':('u_only','joint'),'small_LN_minus_joint':('small_LN','joint')} if stage=='R4' else {}
        contrasts[stage]={};cases[stage]=[r for r in rr if r['dataset']=='vidstg' and r['source_id']==35 and r['condition']=='exposure_5' and r['order']=='order1']
        resources[stage]={}
        for ds in DATASETS:
            contrasts[stage][ds]={};p=read(r1.BASE/ds/'PLAN.json')
            resources[stage][ds]={k:v for k,v in read(BASE/stage/ds/'PREDICTION_BARRIER.json').items() if k!='files'}
            for split in ['search','confirm']:
                contrasts[stage][ds][split]={}
                for name,(a,b) in pairs.items():
                    z=[]
                    for r in rr:
                        if r['dataset']!=ds or r['split']!=split or r['condition']=='clean' or r['arm']!=a:continue
                        other=lookup[(ds,split,r['condition'],r['order'],r['arrival'],b)]
                        z.append(dict(source_id=r['source_id'],delta_v=r['v']-other['v']))
                    contrasts[stage][ds][split][name]=stats(z,['delta_v'])
            labels=read(c1.POOL/ds/'GT_LABELS_confirm.json')
            for r in rr:
                if r['dataset']!=ds or r['split']!='confirm':continue
                row=p['rows'][r['source_id']];truth={int(k):v for k,v in labels[str(r['source_id'])]['truth'].items()};span=labels[str(r['source_id'])]['span']
                f=BASE/stage/ds/'confirm'/r['condition']/r['order']/f"{r['arrival']:05}.pt"
                assert sha(f)==r['payload_sha256'] and barrier['files'][str(f.relative_to(BASE))]==sha(f)
                x=checked(f);fit=x['fits'][r['arm']];ex=c1.checked(ROOT/x['evidence_path'])['expert']
                d0=DenseTube(x['before'].numpy(),row,truth,span,clip=ds=='hc2');d1=DenseTube(fit['final'].numpy(),row,truth,span,clip=ds=='hc2')
                positions={key[1] for key,ob in ex['observations'].items() if len(ob['probe']['boxes'])}
                observed=np.isin(d0.fids,[row['frame_ids'][i] for i in positions]);delta=d1.iou-d0.iou
                rec={k:r[k] for k in ['dataset','split','condition','order','arrival','source_id','arm']}
                rec.update(stage=stage,observed_GT_delta=float(delta[observed].mean()) if observed.any() else None,
                    unobserved_GT_delta=float(delta[~observed].mean()) if (~observed).any() else None,
                    observed_GT_frames=int(observed.sum()),unobserved_GT_frames=int((~observed).sum()),
                    used_proxy_improved_GT_harmed=r['proxy_improved_GT_harmed'],current_vIoU_delta=r['vs_before_v'])
                # A diagnostic of the fixed posterior, never a GT-selected inference path.
                ff,paths,lp,pa=path_numpy(ex,x['before'].numpy(),row['frame_ids']);known=[];per=[]
                for pos,bb,ss in ff:
                    fid=int(row['frame_ids'][pos]);valid=fid in truth and span[0]<=fid<span[1];known.append(valid)
                    boxes=xyxy(bb,row['input']['width'],row['input']['height'])
                    if ds=='hc2':boxes=np.maximum(boxes,0)
                    per.append(box_iou(boxes,np.array(truth[fid])) if valid else np.zeros(len(bb)))
                if any(known):
                    qualities=np.array([np.mean([per[j][path[j]] for j in range(len(ff)) if known[j]]) for path in paths])
                    rec.update(path_GT_observed_support_frames=sum(known),path_best_observed_GT_IoU=float(qualities.max()),
                        path_posterior_mean_observed_GT_IoU=float(np.exp(lp)@qualities),
                        path_MAP_observed_GT_IoU=float(qualities[np.argmax(lp)]),
                        path_mass_observed_GT_IoU_ge_half=float(np.exp(lp)[qualities>=.5].sum()))
                else:rec.update(path_GT_observed_support_frames=0)
                details.append(rec)
    write(PUB/'MECHANISM_CONTRASTS.json',contrasts);write(PUB/'SOURCE35_MATCHED_CASES.json',cases)
    write(PUB/'LABEL_PROVENANCE_AUDIT.json',dict(status='pass',datasets=label_provenance,raw_annotations_exported=False))
    write(PUB/'OBSERVATION_AND_PATH_DIAGNOSTICS.json',details)
    summary={}
    for stage in contrasts:
        summary[stage]={}
        for ds in DATASETS:
            summary[stage][ds]={}
            for arm in sorted({d['arm'] for d in details if d['stage']==stage}):
                rr=[d for d in details if d['stage']==stage and d['dataset']==ds and d['condition']!='clean' and d['arm']==arm]
                z={}
                for fld in ['observed_GT_delta','unobserved_GT_delta','path_best_observed_GT_IoU','path_posterior_mean_observed_GT_IoU','path_MAP_observed_GT_IoU','path_mass_observed_GT_IoU_ge_half']:
                    z[fld]=stats([dict(source_id=d['source_id'],value=d[fld]) for d in rr if d.get(fld) is not None],['value'])
                z['used_proxy_improved_GT_harmed']=sum(d['used_proxy_improved_GT_harmed'] for d in rr)
                z['no_observed_GT_support']=sum(d['path_GT_observed_support_frames']==0 for d in rr)
                summary[stage][ds][arm]=z
    write(PUB/'MECHANISM_DIAGNOSTIC_SUMMARY.json',summary)
    resources['online']={ds:{k:v for k,v in read(BASE/'online'/ds/'PREDICTION_BARRIER.json').items() if k!='files'} for ds in DATASETS}
    write(PUB/'RESOURCE_RECEIPT.json',dict(stages=resources,new_DINO=0,new_backbone=0,wall_is_not_GPU_kernel=True))
    write(PUB/'ENGINEERING_REVISION_RECEIPT.json',dict(revisions=[read(f) for f in sorted((BASE/'revisions').glob('*.json'))],time=time.time(),predictions_changed_after_GT=False))

if __name__=='__main__':run()
