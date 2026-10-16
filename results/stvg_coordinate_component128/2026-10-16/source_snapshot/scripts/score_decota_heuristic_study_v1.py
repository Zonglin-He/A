"""Outer dev-only selection, sealed evaluation predictions, then GT diagnostics.

No evaluation labels are opened by tune() or predict(). GT-anchor results are
created only inside score(), explicitly labelled oracle, on common GT support.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_heuristic_study_v1 import *
from methods.decota_refine_v1.api import refine
from vg_tta.native_coverage_calibration_v1 import native_at_length
from vg_tta.posterior_mass_coverage_v1 import select as pm_select
from vg_tta.metrics import interval_from_logits
from vg_tta.st_component_diagnostics_v1 import frame_iou,physical_score


def gt_payload(p,r,g,split,bridges=None):
    if split=='development':
        assert sha(p['dev_label_manifest'])==p['dev_label_sha256']
        rr=next(x for x in read(p['dev_label_manifest'])['groups'][g] if x['index']==r['input']['index'])
        assert sha(rr['labels_path'])==rr['label_sha256'];x=load(rr['labels_path'])
        assert x['video_target']['frames_id']==r['input']['frame_ids']
        return x['targets'],tuple(x['annotation'][k] for k in ['tube_start_frame','tube_end_frame'])
    assert bridges is not None
    targets,_,gt=bridges[r['parent']].labels(r['input']);return targets,gt


def gt_array(targets,ids,gt):
    valid=(np.asarray(ids)>=gt[0])&(np.asarray(ids)<gt[1]);boxes=np.zeros((len(ids),4),np.float32)
    for i,t in enumerate(targets):
        assert (len(t['boxes'])>0)==bool(valid[i])
        if valid[i]:boxes[i]=t['boxes'][0].float().numpy()
    assert valid.any();return boxes,valid


def numpy_iou(boxes,truth,valid):
    b=np.asarray(boxes,dtype=np.float64);g=np.asarray(truth,dtype=np.float64)
    lo=np.maximum(b[:,:2]-b[:,2:]/2,g[:,:2]-g[:,2:]/2)
    hi=np.minimum(b[:,:2]+b[:,2:]/2,g[:,:2]+g[:,2:]/2)
    inter=np.maximum(hi-lo,0).prod(1);union=b[:,2:].prod(1)+g[:,2:].prod(1)-inter
    out=inter/np.maximum(union,1e-12);out[~valid]=np.nan;return out


def metric(boxes,truth,valid,ids,gt,ij):
    qual=numpy_iou(boxes,truth,valid);a,c=ids[ij[0]],ids[ij[1]]+1
    result=physical_score(qual,ids,gt,(a,c));inter=max(0,min(c,gt[1])-max(a,gt[0]))
    horizon=ids[-1]+1-ids[0]
    result.update(recall=inter/(gt[1]-gt[0]),precision=inter/(c-a),span=(c-a)/horizon,
        duration_MAE=abs((c-a)-(gt[1]-gt[0]))/horizon,
        center_MAE=abs((c+a)-(gt[1]+gt[0]))/(2*horizon))
    return result,qual


def pack(p,r,b,g,split):
    n=native_row(r,b);t=load(tpath(b,g,split,r['ordinal']));assert not t['GT_used']
    return dict(row=r,native=n,temporal=t,ids=n['frame_ids'],base=n['predictions']['frozen']['boxes'].float().cpu(),
        ni=t['intervals']['frozen'],ij=t['intervals']['decota'],evidence=evidence(r,b,n),
        probes={m:expert_probes(g,split,r['ordinal'],m) for m in ['parsed','noun','full']},
        nms={m:{} for m in ['parsed','noun','full']})


def spatial(x,family,cfg,*,extent=None):
    ij=x['ij'] if extent is None else extent;ni=x['ni'];cfg=dict(cfg)
    mode=family if family in ['uniform','no_new','no_evidence','full_uniform'] or family.startswith('random') else 'current'
    text='noun' if family=='text_noun' else 'full' if family=='text_full' else 'parsed'
    if family=='gate_none':cfg.update(phrase_threshold=0.,distinct_margin=0.)
    if family=='gate_score':cfg['distinct_margin']=0.
    positions=select_positions(x['ids'],ij,ni,x['evidence'],cfg['keyframes'],mode)
    probes=x['probes'][text];bypos={p['position']:p for p in probes}
    phrase,entity=text_spec(x['row'],text)
    if phrase and entity in phrase:
        assert set(positions)<=set(bypos),(family,set(positions)-set(bypos))
    pp=anchors(probes,positions,cfg,x['nms'][text])
    boxes,audit=refine(x['base'],pp,x['ids'],family if family in ['absolute','direct'] else 'residual')
    return dict(boxes=boxes,indices=list(ij),positions=positions,
        called=[i for i in positions if i in bypos],pseudo=pp,text=text,
        expert_seconds=sum(bypos[i]['seconds'] for i in positions if i in bypos),audit=audit)


def prior(z,ids,kind,value):
    if kind=='fraction':return native_at_length(z,ids,fraction=value)['indices']
    if kind in ['tau','tau_native']:
        ij=pm_select(z,ids,tau=value)['indices']
        return ij if kind=='tau' else native_at_length(z,ids,reference=ij)['indices']
    if kind=='ramp':
        z=z.detach().cpu().double().reshape(len(ids),2).clone();u=torch.tensor((np.asarray(ids)-ids[0])/max(ids[-1]-ids[0],1))
        z[:,0]-=value*u;z[:,1]+=value*u
        extent=interval_from_logits(z[None])
        return extent
    raise ValueError(kind)


def tune(b,g):
    p=plan();torch.set_num_threads(2);dest=OUT/'selection'/b/g
    if (dest/'selected.json').exists():return
    xs=[pack(p,r,b,g,'development') for r in p['rows']['development'][g]]
    labels=[]
    for x in xs:
        targets,gt=gt_payload(p,x['row'],g,'development');truth,valid=gt_array(targets,x['ids'],gt)
        labels.append((truth,valid,gt))
    trials=[];chosen={};cache={}
    def evaluate(family,cfg):
        key=(family,json.dumps(cfg,sort_keys=True))
        if key in cache:return cache[key]
        vals=[]
        for x,(truth,valid,gt) in zip(xs,labels):
            methods=[f'random{s}' for s in SEEDS] if family=='random' else [family]
            vv=[]
            for m in methods:
                pred=spatial(x,m,cfg);vv.append(metric(pred['boxes'],truth,valid,x['ids'],gt,pred['indices'])[0]['vIoU_corrected'])
            vals.append(float(np.mean(vv)))
        val=float(np.mean(vals));cache[key]=val
        trials.append(dict(family=family,config=dict(cfg),utility=val,source_utilities=vals))
        return val
    for family in FAMILIES:
        cfg=dict(SPATIAL);initial=evaluate(family,cfg)
        for sweep in range(2):
            for k,grid in p['grids'].items():
                if family=='gate_none' and k!='nms_iou':continue
                if family=='gate_score' and k=='distinct_margin':continue
                options=[({**cfg,k:v},evaluate(family,{**cfg,k:v})) for v in sorted(set([cfg[k]]+grid))]
                best=max(v for _,v in options)
                cfg=min((c for c,v in options if v>=best-1e-10),key=lambda c:(abs(c[k]-SPATIAL[k]),c[k]))
        chosen[family]=dict(config=cfg,dev_fixed=initial,dev_tuned=evaluate(family,cfg))
        print('GATE_SELECTED',b,g,family,chosen[family],flush=True)
    priors={}
    for kind,grid in [('fraction',p['prior_grids']['fraction']),('tau',p['prior_grids']['tau']),('tau_native',p['prior_grids']['tau']),('ramp',p['prior_grids']['ramp'])]:
        values=[]
        for value in grid:
            vv=[]
            for x,(truth,valid,gt) in zip(xs,labels):
                ij=prior(x['native']['native_logits'],x['ids'],kind,value)
                vv.append(metric(x['base'],truth,valid,x['ids'],gt,ij)[0]['vIoU_corrected'])
            values.append(dict(value=value,utility=float(np.mean(vv))))
        priors[kind]=dict(selected=max(values,key=lambda x:x['utility']),trials=values)
    # Match donor's mean predicted extent to own on dev without any GT.
    scales={};own=np.mean([fraction(x['ij'],x['ids']) for x in xs])
    for seed in SEEDS:
        fs=np.array([fraction(x['temporal']['intervals'][f'donor{seed}'],x['ids']) for x in xs]);lo,hi=0.,100.
        for _ in range(60):
            mid=(lo+hi)/2
            if np.clip(fs*mid,0,1).mean()<own:lo=mid
            else:hi=mid
        scales[str(seed)]=dict(scale=(lo+hi)/2,own_mean=own,donor_mean=float(fs.mean()),GT_used=False)
    write(dest/'selected.json',dict(families=chosen,priors=priors,donor_scales=scales,trials=trials,
        development_sources=[r['input']['source'] for r in p['rows']['development'][g]],evaluation_labels_used=False,
        algorithm_sha256=sha(__file__),lock_sha256=sha(OUT/'lock.json')))


def barrier():
    p=plan();f=OUT/'selection_barrier.json'
    if f.exists():return
    files=[]
    for b in BS:
        for g in GS:
            q=OUT/'selection'/b/g/'selected.json';s=read(q);assert not s['evaluation_labels_used']
            files.append(dict(backbone=b,group=g,path=str(q),sha256=sha(q)))
    write(f,dict(receipts=files,created=time.time(),algorithm_sha256=sha(__file__),GT_used_for_test_adaptation=False))


def predict(b,g):
    p=plan();selbar=read(OUT/'selection_barrier.json')
    receipt=next(r for r in selbar['receipts'] if r['backbone']==b and r['group']==g)
    assert sha(receipt['path'])==receipt['sha256'];s=read(receipt['path']);assert sha(__file__)==selbar['algorithm_sha256']
    dest=OUT/'predictions'/b/g;receipts=[];torch.set_num_threads(2)
    for r in p['rows']['evaluation'][g]:
        j=r['ordinal'];f=dest/f'{j:03d}.pt'
        if not f.exists():
            x=pack(p,r,b,g,'evaluation');pred={}
            for k in ['frozen','decota','coupled_coverage','center']+[f'{prefix}{seed}' for prefix in ['place_random','donor','shuffle'] for seed in SEEDS]:
                pred['time_'+k]=dict(boxes=x['base'],indices=x['temporal']['intervals'][k],called=[],pseudo=[])
            for kind,z in s['priors'].items():
                pred['time_prior_'+kind]=dict(boxes=x['base'],indices=list(prior(x['native']['native_logits'],x['ids'],kind,z['selected']['value'])),called=[],pseudo=[])
            for seed in SEEDS:
                fr=fraction(x['temporal']['intervals'][f'donor{seed}'],x['ids'])*s['donor_scales'][str(seed)]['scale']
                ij=native_at_length(x['native']['native_logits'],x['ids'],fraction=min(fr,1.))['indices']
                pred[f'time_donor_scaled{seed}']=dict(boxes=x['base'],indices=list(ij),called=[],pseudo=[])
            for family in FAMILIES:
                names=[f'random{seed}' for seed in SEEDS] if family=='random' else [family]
                for name in names:
                    for variant,cfg in [('fixed',SPATIAL),('retuned',s['families'][family]['config'])]:pred[f'{name}_{variant}']=spatial(x,name,cfg)
            pred['space_native']=spatial(x,'main',SPATIAL,extent=x['ni'])
            # Matched 2x2: identical full-video uniform anchors and boxes,
            # differing ONLY in native vs adapted final temporal interval.
            shared=spatial(x,'full_uniform',SPATIAL)
            pred['matched_space_native']={**shared,'indices':x['ni']}
            pred['matched_space_decota']={**shared,'indices':x['ij']}
            for k in p['budget_grid']:pred[f'budget{k}']=spatial(x,'main',{**SPATIAL,'keyframes':k})
            main=pred['main_fixed'];ex=main['boxes'].clone();pp=main['pseudo']
            if pp:
                ps=sorted(pp,key=lambda q:q['position']);base=x['base'];a,z=ps[0]['position'],ps[-1]['position']
                for pos in range(x['ij'][0],x['ij'][1]+1):
                    if a<=pos<=z:continue
                    anchor=ps[0] if pos<a else ps[-1]
                    v=base[pos]+torch.tensor(anchor['box'])-base[anchor['position']]
                    from torchvision.ops import box_convert
                    v=box_convert(box_convert(v[None],'cxcywh','xyxy').clamp(0,1),'xyxy','cxcywh')[0]
                    if bool((v[2:]>0).all()):ex[pos]=v
            pred['nearest_extrapolation']={**main,'boxes':ex}
            old=load(PREV/'evaluation'/b/g/f'{j:03d}.pt')['predictions']['old_method']
            assert list(old['indices'])==list(main['indices'])
            assert torch.equal(old['boxes'].float(),main['boxes']),('stable output mismatch',b,g,j)
            save(f,dict(input=r['input'],parent=r['parent'],frame_ids=x['ids'],predictions=pred,
                candidate_count=x['temporal']['candidate_count'],donors=x['temporal']['donors'],
                GT_used=False,stable_main_exact=True,selection_sha256=sha(OUT/'selection_barrier.json')))
        receipts.append(dict(path=str(f),sha256=sha(f),ordinal=j,source=r['input']['source']))
        print('PREDICTIONS',b,g,j+1,len(p['rows']['evaluation'][g]),flush=True)
    f=dest/'barrier.json'
    if not f.exists():write(f,dict(receipts=receipts,GT_used=False))


def safe_mean(a):
    vals=np.asarray(a,dtype=float);return float(np.nanmean(vals)) if np.isfinite(vals).any() else None


def summarize(values,sources,seed=20260910):
    from collections import defaultdict
    d=defaultdict(list)
    for v,s in zip(values,sources):
        if v is not None and np.isfinite(v):d[s].append(v)
    if not d:return dict(mean=None,ci95=None,sources=0)
    a=np.array([np.mean(v) for k,v in sorted(d.items())]);n=len(a)
    bs=a[np.random.default_rng(seed).integers(n,size=(10000,n))].mean(1)
    k=int(.1*n);z=np.sort(a)
    return dict(mean=float(a.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),sources=n,median=float(np.median(a)),
        trimmed_mean=float(z[k:n-k].mean()),win_neutral_loss=[int((a>.001).sum()),int((abs(a)<=.001).sum()),int((a<-.001).sum())])


def score():
    from vg_tta.dense_expansion_data_v1 import DenseBridge
    from vg_tta.tg_spatial_tta_v1 import choose_candidate
    p=plan();torch.set_num_threads(2);receipts=[]
    for b in BS:
        for g in GS:
            bar=read(OUT/'predictions'/b/g/'barrier.json');assert not bar['GT_used']
            assert len(bar['receipts'])==len(p['rows']['evaluation'][g])
            for rr in bar['receipts']:
                assert sha(rr['path'])==rr['sha256'];receipts.append(dict(backbone=b,group=g,**rr))
    f=OUT/'prediction_barrier.json'
    if not f.exists():write(f,dict(receipts=receipts,GT_used=False,created=time.time()))
    bridges={k:DenseBridge(v) for k,v in p['evaluation_label_specs'].items()};rows=[]
    for rr in receipts:
        b,g,j=rr['backbone'],rr['group'],rr['ordinal'];r=p['rows']['evaluation'][g][j]
        result=load(rr['path']);x=pack(p,r,b,g,'evaluation');ids=x['ids']
        targets,gt=gt_payload(p,r,g,'evaluation',bridges);truth,valid=gt_array(targets,ids,gt)
        pred=result['predictions'];main=pred['main_fixed'];positions=main['called']
        # Identical anchor-support masks for observed-expert and GT oracle.
        common=[q for q in main['pseudo'] if valid[q['position']]]
        gtanchors=[{**q,'box':truth[q['position']].tolist()} for q in common]
        oracles={}
        for tag,aa in [('expert_common',common),('GT_anchor_oracle',gtanchors)]:
            for mode in ['direct','absolute','residual']:
                boxes,_=refine(x['base'],aa,ids,mode)
                name=tag+'_'+mode;pred[name]=dict(boxes=boxes,indices=x['ij'],called=positions,pseudo=aa)
                oracles[name]=dict(boxes=boxes,indices=x['ij'],GT_used=True,diagnostic_only=True)
        of=OUT/'oracles'/b/g/f'{j:03d}.pt'
        if not of.exists():save(of,oracles)
        common_uncalled=valid.copy()
        for name,v in pred.items():
            if not name.startswith('time_'):
                common_uncalled[v['called']]=False
        main_uncalled=valid.copy();main_uncalled[positions]=False
        qs={};ms={}
        for name,z in pred.items():
            m,q=metric(z['boxes'],truth,valid,ids,gt,z['indices']);qs[name]=q
            m.update(main_uncalled_GT=safe_mean(q[main_uncalled]),common_uncalled_GT=safe_mean(q[common_uncalled]),
                called=len(z['called']),accepted=len(z['pseudo']),expert_seconds=z.get('expert_seconds',0.))
            ms[name]=m
        # Treat random repeats as repeated conditions, average within source.
        for pre,suf,out in [('random','_fixed','random_fixed_mean'),('random','_retuned','random_retuned_mean'),
            ('time_place_random','','time_random_mean'),('time_donor','','time_donor_mean'),
            ('time_donor_scaled','','time_donor_scaled_mean'),('time_shuffle','','time_shuffle_mean')]:
            ms[out]={k:safe_mean([ms[f'{pre}{seed}{suf}'][k] for seed in SEEDS]) for k in ms[f'{pre}{SEEDS[0]}{suf}'] if isinstance(ms[f'{pre}{SEEDS[0]}{suf}'][k],(int,float)) or ms[f'{pre}{SEEDS[0]}{suf}'][k] is None}
        reliability=[];bypos={q['position']:q for q in x['probes']['parsed']}
        for pos in positions:
            if pos not in bypos:continue
            z=bypos[pos];c=choose_candidate(z['all_boxes'],z['all_phrase_scores'],dict(nms_iou=.5,phrase_threshold=0.,distinct_margin=0.))
            if not c['accepted']:continue
            labeled=bool(valid[pos]);delta=None
            if labeled:
                bb=x['base'].clone();bb[pos]=torch.tensor(c['box']);quality=numpy_iou(bb,truth,valid)
                delta=float(quality[pos]-qs['time_frozen'][pos])
            reliability.append(dict(position=pos,score=c['score'],margin=c['margin'],labeled=labeled,delta_iou=delta))
        # Independent reference evaluator spot-check all primary coordinates.
        maxerr=0.
        for name in ['time_frozen','main_fixed','absolute_fixed','GT_anchor_oracle_residual']:
            q,_=frame_iou(pred[name]['boxes'],targets,ids,gt)
            maxerr=max(maxerr,float(np.nanmax(abs(q-qs[name]))));assert maxerr<3e-6
        duration=(gt[1]-gt[0])/(ids[-1]+1-ids[0]);native_recall=ms['time_frozen']['recall']
        rows.append(dict(backbone=b,group=g,source=r['input']['source'],ordinal=j,metrics=ms,
            event_group='short' if duration<.33 else 'medium' if duration<.67 else 'long',
            native_coverage_group='low' if native_recall<.33 else 'medium' if native_recall<.67 else 'high',
            gt_fraction=duration,candidate_count=result['candidate_count'],expert_reliability=reliability,
            parser_rule=r['visual_query']['rule'],distinctive=r['visual_query']['distinctive'],
            parser_empty=not bool(r['visual_query']['phrase']),common_uncalled_count=int(common_uncalled.sum()),
            main_uncalled_count=int(main_uncalled.sum()),oracle_common_anchors=len(common),numpy_iou_maxerr=maxerr))
    summary={}
    pairs=[('main_fixed','space_native'),('main_fixed','time_decota'),('time_decota','time_frozen'),
        ('matched_space_decota','matched_space_native'),('time_decota','time_coupled_coverage'),
        ('time_decota','time_center'),('time_decota','time_random_mean'),('time_decota','time_donor_mean'),
        ('time_decota','time_donor_scaled_mean'),('time_decota','time_shuffle_mean')]
    pairs += [('time_decota','time_prior_'+k) for k in ['fraction','tau','tau_native','ramp']]
    pairs += [('main_fixed',name+'_fixed') for name in FAMILIES if name not in ['main','random']]
    pairs += [('main_fixed','random_fixed_mean'),('main_fixed','nearest_extrapolation')]
    pairs += [('main_retuned',name+'_retuned') for name in FAMILIES if name not in ['main','random']]
    pairs += [('main_retuned','random_retuned_mean'),('main_retuned','main_fixed')]
    pairs += [(tag+'_residual',tag+'_'+mode) for tag in ['expert_common','GT_anchor_oracle'] for mode in ['direct','absolute']]
    fields=['vIoU_corrected','sIoU','tIoU','recall','precision','span','duration_MAE','main_uncalled_GT','common_uncalled_GT']
    for b in BS:
        for g in GS:
            rr=[r for r in rows if r['backbone']==b and r['group']==g];sources=[r['source'] for r in rr]
            absolute={m:{f:summarize([r['metrics'][m].get(f) for r in rr],sources) for f in fields} for m in rr[0]['metrics']}
            contrasts={}
            for a,z in pairs:
                contrasts[a+'__minus__'+z]={f:summarize([None if r['metrics'][a].get(f) is None or r['metrics'][z].get(f) is None else r['metrics'][a][f]-r['metrics'][z][f] for r in rr],sources) for f in fields}
            contrasts['deployment_interaction']={'vIoU_corrected':summarize([r['metrics']['main_fixed']['vIoU_corrected']-r['metrics']['space_native']['vIoU_corrected']-r['metrics']['time_decota']['vIoU_corrected']+r['metrics']['time_frozen']['vIoU_corrected'] for r in rr],sources)}
            strata={}
            for grouping,levels in [('event_group',['short','medium','long']),('native_coverage_group',['low','medium','high'])]:
                strata[grouping]={}
                for level in levels:
                    subset=[r for r in rr if r[grouping]==level]
                    strata[grouping][level]={m:summarize([r['metrics'][m]['vIoU_corrected'] for r in subset],[r['source'] for r in subset]) for m in ['time_frozen','time_decota','time_prior_fraction','time_prior_tau','time_donor_mean','time_shuffle_mean','main_fixed','space_native']}
            movable=[r for r in rr if r['candidate_count']>1]
            placement={z:summarize([r['metrics']['time_decota']['vIoU_corrected']-r['metrics'][z]['vIoU_corrected'] for r in movable],[r['source'] for r in movable]) for z in ['time_center','time_random_mean','time_coupled_coverage']}
            reliability={}
            for threshold in [.0,.15,.25,.35,.5]:
                for margin in [0.,.025,.05,.1]:
                    vv=[];allcalled=accepted=labelcount=0
                    for r in rr:
                        aa=r['expert_reliability'];bb=[x for x in aa if x['score']>=threshold and x['margin']>=margin]
                        allcalled+=len(aa);accepted+=len(bb);labelcount+=sum(x['labeled'] for x in bb)
                        vv.append(safe_mean([q['delta_iou'] for q in bb if q['labeled']]))
                    reliability[f'{threshold}_{margin}']=dict(accepted=accepted,total_candidates=allcalled,labeled_accepted=labelcount,
                        acceptance_rate=accepted/max(allcalled,1),source_macro_delta_iou=summarize(vv,sources))
            counts={m:dict(mean_called=float(np.mean([r['metrics'][m]['called'] for r in rr])),
                mean_accepted=float(np.mean([r['metrics'][m]['accepted'] for r in rr])),
                empty=sum(r['metrics'][m]['accepted']==0 for r in rr),
                expert_seconds=float(np.mean([r['metrics'][m]['expert_seconds'] for r in rr]))) for m in rr[0]['metrics'] if not m.startswith('time_')}
            summary[b+'_'+g]=dict(sources=len(rr),absolute=absolute,contrasts=contrasts,strata=strata,
                movable_sources=len(movable),movable_placement_contrasts=placement,reliability=reliability,counts=counts,
                parser_empty=sum(r['parser_empty'] for r in rr),retrospective=True)
            print('SCORED',b,g,{m:round(absolute[m]['vIoU_corrected']['mean']*100,4) for m in ['time_frozen','time_decota','space_native','main_fixed','main_retuned','absolute_fixed']},flush=True)
    write(OUT/'scored_rows.json',rows);write(OUT/'summary.json',summary)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['tune','barrier','predict','score']);ap.add_argument('--backbone',choices=BS);ap.add_argument('--group',choices=GS)
    a=ap.parse_args()
    if a.stage=='tune':tune(a.backbone,a.group)
    elif a.stage=='barrier':barrier()
    elif a.stage=='predict':predict(a.backbone,a.group)
    else:score()
