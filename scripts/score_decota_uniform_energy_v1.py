"""Development-only gate selection, then sealed evaluation and energy QA."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_uniform_energy_v1 import *
from scripts.score_decota_heuristic_study_v1 import gt_payload,gt_array,metric,summarize,safe_mean
from scripts.audit_decota_heuristic_study_v1 import independent_quality
from methods.decota_refine_uniform_v1.api import solve_anchor_energy


def tune(b,g):
    torch.set_num_threads(2);p,prior=plan();dest=OUT/'selection'/b/g/'selected.json'
    if dest.exists():return
    xs=[pack(p,prior,r,b,g,'development') for r in prior['rows']['development'][g]]
    labels=[]
    for x in xs:
        targets,gt=gt_payload(prior,x['row'],g,'development');truth,valid=gt_array(targets,x['ids'],gt)
        labels.append((truth,valid,gt))
    trials=[];cache={};chosen={}
    def evaluate(family,cfg):
        key=(family,json.dumps(cfg,sort_keys=True))
        if key not in cache:
            selector,mode=family.split('_');vv=[]
            for x,(truth,valid,gt) in zip(xs,labels):
                pred=spatial(x,selector,mode,cfg)
                vv.append(metric(pred['boxes'],truth,valid,x['ids'],gt,pred['indices'])[0]['vIoU_corrected'])
            cache[key]=float(np.mean(vv));trials.append(dict(family=family,config=dict(cfg),utility=cache[key],source_utilities=vv))
        return cache[key]
    for family in FAMILIES:
        cfg=dict(SPATIAL);initial=evaluate(family,cfg)
        for sweep in range(p['gate_sweeps']):
            for k,grid in p['gate_grid'].items():
                options=[({**cfg,k:v},evaluate(family,{**cfg,k:v})) for v in sorted(set([cfg[k]]+grid))]
                best=max(v for _,v in options)
                cfg=min((c for c,v in options if v>=best-1e-10),key=lambda c:(abs(c[k]-SPATIAL[k]),c[k]))
        chosen[family]=dict(config=cfg,dev_fixed=initial,dev_tuned=evaluate(family,cfg))
        print('SELECT',b,g,family,chosen[family],flush=True)
    write(dest,dict(families=chosen,trials=trials,evaluation_labels_used=False,
        development_sources=[r['input']['source'] for r in prior['rows']['development'][g]],
        lock_sha256=sha(OUT/'lock.json')))


def barrier():
    p,prior=plan();receipts=[]
    for b in BS:
        for g in GS:
            f=OUT/'selection'/b/g/'selected.json';s=read(f);assert not s['evaluation_labels_used']
            receipts.append(dict(backbone=b,group=g,path=str(f),sha256=sha(f)))
    write(OUT/'selection_barrier.json',dict(receipts=receipts,created=time.time(),evaluation_labels_used=False))


def energy_check(base,pseudo,ids,absolute,energy):
    error=float(torch.max(abs(absolute-energy)))
    assert error<=2e-7
    out=dict(output_max_error=error,bit_equal=torch.equal(absolute,energy))
    if len(pseudo)>=2:
        pp=sorted(pseudo,key=lambda q:q['position']);pos=[q['position'] for q in pp]
        targets=np.asarray([q['box'] for q in pp],dtype=np.float32)
        hull,values,audit=solve_anchor_energy(ids,pos,targets)
        interp=np.column_stack([np.interp(np.asarray(ids)[hull],np.asarray(ids)[pos],targets[:,k]) for k in range(4)])
        out.update(audit,FP64_interp_max_error=float(np.max(abs(interp-values))))
        assert out['FP64_interp_max_error']<1e-9 and audit['stationarity_max']<1e-10
        assert torch.equal(absolute[:pos[0]],base[:pos[0]]) and torch.equal(absolute[pos[-1]+1:],base[pos[-1]+1:])
    return out


def score():
    from vg_tta.dense_expansion_data_v1 import DenseBridge
    torch.set_num_threads(2);p,prior=plan();bar=read(OUT/'prediction_barrier.json')
    assert not bar['GT_used'] and bar['selection_sha256']==sha(OUT/'selection_barrier.json')
    # Evaluation labels are first opened here, only after all legal predictions.
    bridges={k:DenseBridge(v) for k,v in prior['evaluation_label_specs'].items()}
    rows=[];qa=[];independent_errors=[]
    for rr in bar['receipts']:
        assert sha(rr['path'])==rr['sha256'];z=load(rr['path']);assert not z['GT_used']
        b,g,j=rr['backbone'],rr['group'],rr['ordinal'];r=prior['rows']['evaluation'][g][j]
        x=pack(p,prior,r,b,g,'evaluation');ids=x['ids'];targets,gt=gt_payload(prior,r,g,'evaluation',bridges)
        truth,valid=gt_array(targets,ids,gt);pred=z['predictions'];checks={};oracle={}
        for selector in ['current','uniform','minimax']:
            for variant in ['fixed','retuned']:
                a=pred[selector+'_absolute_'+variant];e=pred[selector+'_energy_'+variant]
                assert a['pseudo']==e['pseudo'] and a['indices']==e['indices']
                checks[selector+'_'+variant]=energy_check(x['base'],a['pseudo'],ids,a['boxes'],e['boxes'])
            a=pred[selector+'_absolute_fixed'];common=[q for q in a['pseudo'] if valid[q['position']]]
            gtanchors=[{**q,'box':truth[q['position']].tolist()} for q in common]
            for tag,aa in [('expert_common',common),('GT_anchor_oracle',gtanchors)]:
                for mode in ['direct','residual','absolute','minimum_energy']:
                    boxes,audit=reconstruct(x['base'],aa,ids,mode)
                    oracle[selector+'_'+tag+'_'+mode]=dict(boxes=boxes,indices=x['ij'],pseudo=aa,
                        called=a['called'],positions=a['positions'],audit=audit,GT_used=True,diagnostic_only=True)
                aa_abs=oracle[selector+'_'+tag+'_absolute'];aa_en=oracle[selector+'_'+tag+'_minimum_energy']
                checks[selector+'_'+tag]=energy_check(x['base'],aa,ids,aa_abs['boxes'],aa_en['boxes'])
        save(OUT/'oracles'/b/g/f'{j:03d}.pt',oracle)
        qualities={};metrics={}
        for name,v in {**pred,**oracle}.items():
            m,q=metric(v['boxes'],truth,valid,ids,gt,v['indices']);qualities[name]=q
            independent=independent_quality(v['boxes'],truth,valid)
            a,c=ids[v['indices'][0]],ids[v['indices'][1]]+1;tt=np.asarray(ids)
            intersection=valid&(tt>=a)&(tt<c);union=(tt>=min(a,gt[0]))&(tt<max(c,gt[1]))
            independent_v=float(independent[intersection].sum()/max(1,int(union.sum())))
            err=abs(independent_v-m['vIoU_corrected']);independent_errors.append(err);assert err<1e-10
            m.update(called=len(v['called']),accepted=len(v['pseudo']),
                radius=v.get('radius'),radius_fraction=v.get('radius_fraction'),accepted_radius=v.get('accepted_radius'),
                expert_seconds=v.get('expert_seconds'),reconstruction_seconds=v.get('reconstruction_seconds'))
            metrics[name]=m
        pairs=[]
        for variant in ['fixed','retuned']:
            pairs += [('uniform_absolute_'+variant,'current_residual_'+variant),
                      ('uniform_absolute_'+variant,'uniform_residual_'+variant),
                      ('uniform_absolute_'+variant,'current_absolute_'+variant),
                      ('minimax_absolute_'+variant,'uniform_absolute_'+variant)]
            pairs += [(s+'_energy_'+variant,s+'_absolute_'+variant) for s in ['current','uniform','minimax']]
        contrasts={}
        for a,c in pairs:
            mask=valid.copy();mask[list(set(pred[a]['called']+pred[c]['called']))]=False
            contrasts[a+'__minus__'+c]=dict(
                vIoU_corrected=metrics[a]['vIoU_corrected']-metrics[c]['vIoU_corrected'],
                sIoU=metrics[a]['sIoU']-metrics[c]['sIoU'],
                common_uncalled_sIoU=safe_mean((qualities[a]-qualities[c])[mask]),
                common_uncalled_frames=int(mask.sum()))
        for variant in ['fixed','retuned']:
            def vv(sel,mode):return metrics[sel+'_'+mode+'_'+variant]['vIoU_corrected']
            contrasts['selection_propagation_interaction_'+variant]=dict(vIoU_corrected=
                vv('uniform','absolute')-vv('uniform','residual')-vv('current','absolute')+vv('current','residual'))
        rows.append(dict(backbone=b,group=g,ordinal=j,source=r['input']['source'],metrics=metrics,contrasts=contrasts,
            temporal_equal=all(v['indices']==x['ij'] for name,v in pred.items() if name!='frozen'),
            minimax_certificate=z['minimax_certificate']))
        qa.append(dict(backbone=b,group=g,ordinal=j,checks=checks))
    summary={}
    for b in BS:
        for g in GS:
            rr=[r for r in rows if r['backbone']==b and r['group']==g];sources=[r['source'] for r in rr]
            assert len(rr)==len(set(sources))==len(prior['rows']['evaluation'][g])
            absolute={name:{m:summarize([r['metrics'][name][m] for r in rr],sources)
                for m in ['vIoU_corrected','sIoU','tIoU','called','accepted','radius','radius_fraction','accepted_radius','expert_seconds','reconstruction_seconds']}
                for name in rr[0]['metrics']}
            contrast={name:{m:summarize([r['contrasts'][name][m] for r in rr],sources)
                for m in rr[0]['contrasts'][name] if m!='common_uncalled_frames'} for name in rr[0]['contrasts']}
            # Separate index path recomputes primary means and paired bootstrap.
            sorted_rows=sorted(rr,key=lambda z:z['source'])
            ix=np.random.default_rng(20260910).integers(len(rr),size=(10000,len(rr)))
            for name,z in contrast.items():
                a=np.array([r['contrasts'][name]['vIoU_corrected'] for r in sorted_rows])
                ci=np.percentile(np.take(a,ix).mean(1),[2.5,97.5])
                assert np.max(abs(ci-z['vIoU_corrected']['ci95']))<1e-12
            summary[b+'_'+g]=dict(absolute=absolute,contrasts=contrast)
    write(OUT/'scored_rows.json',rows);write(OUT/'summary.json',summary);write(OUT/'energy_checks.json',qa)
    checks=[c for r in qa for c in r['checks'].values()]
    write(OUT/'QA_RESULTS.json',dict(legal_prediction_sources=len(rows),unique_evaluation_sources=len({r['source'] for r in rows}),
        independent_metrics=len(independent_errors),max_metric_error=max(independent_errors),
        energy_pairs=len(checks),energy_bit_equal=sum(c['bit_equal'] for c in checks),
        max_energy_output_error=max(c['output_max_error'] for c in checks),
        max_energy_FP64_error=max(c.get('FP64_interp_max_error',0) for c in checks),
        max_stationarity=max(c.get('stationarity_max',0) for c in checks),
        source_disjoint=True,all_temporal_outputs_unchanged=all(r['temporal_equal'] for r in rows),
        all_sources_retained=True,all_old_control_predictions_exact=True,spatial_backward_calls=0,
        all_minimax_certificates_valid=True,paired_bootstrap_independently_checked=True))
    print('QA',read(OUT/'QA_RESULTS.json'),flush=True)
    for cell,v in summary.items():
        print('RESULT',cell,{m:round(v['absolute'][m]['vIoU_corrected']['mean']*100,4) for m in ['frozen','time_only','current_residual_fixed','uniform_absolute_fixed','uniform_energy_fixed','minimax_absolute_fixed','uniform_absolute_retuned','minimax_absolute_retuned']},flush=True)


def live(b,g):
    """New deployment route vs sealed predictions; native/TTA and expert run live."""
    from scripts.run_decota_refine_v1 import student
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert,choose_candidate
    from vg_tta.dense_expansion_data_v1 import decode_raw
    from methods.decota_refine_uniform_v1.predictor import UniformDeCoTARefinePredictor
    p,prior=plan();configure();rows=prior['rows']['evaluation'][g]
    allx=[load(OUT/'predictions'/b/g/f'{j:03d}.pt') for j in range(len(rows))]
    enough=next((j for j,x in enumerate(allx) if len(x['predictions']['uniform_absolute_fixed']['pseudo'])>=2),0)
    empty=next((j for j,x in enumerate(allx) if not x['predictions']['uniform_absolute_fixed']['pseudo']),0)
    model=student(read(ROOT/'artifacts/decota_refine_v1/lock.json'),b,g);expert=SpatialExpert(prior['expert_snapshot'])
    cfg={**SPATIAL,**{k:p['temporal_configs'][b][g][k] for k in ['lr','steps']}};results=[]
    for j in sorted(set([enough,empty])):
        row=rows[j];raw,ids=decode_raw(row['input'])
        # The first two cases use the real frozen expert, not a proposal replay.
        predictor=UniformDeCoTARefinePredictor(model,expert,None,backbone=b,config=cfg)
        out=predictor.predict(raw,ids,row['input'],subject=row['student_subject']['subject'],parsed_visual_query=row['visual_query'])
        expected=allx[j]['predictions']['uniform_absolute_fixed']
        assert out['keyframes']==expected['positions'] and out['indices']==expected['indices']
        assert torch.equal(out['boxes'],expected['boxes'])
        assert out['pseudo']==expected['pseudo'] and out['spatial_backward_calls']==0 and not out['GT_used']
        results.append(dict(ordinal=j,source=row['input']['source'],selector='uniform',reconstruction='absolute',
            native_temporal_live=True,expert_live=True,all_outputs_exact=True,accepted=len(out['pseudo']),state_reset=out['state_reset_checked']))
        print('LIVE',b,g,results[-1],flush=True)
    # Also exercise the optional minimax+energy entry point on a nonempty case.
    j=enough;row=rows[j];raw,ids=decode_raw(row['input'])
    predictor=UniformDeCoTARefinePredictor(model,expert,None,backbone=b,config=cfg,selector='minimax',reconstruction='minimum_energy')
    out=predictor.predict(raw,ids,row['input'],subject=row['student_subject']['subject'],parsed_visual_query=row['visual_query'])
    expected=allx[j]['predictions']['minimax_energy_fixed']
    assert out['keyframes']==expected['positions'] and out['indices']==expected['indices'] and torch.equal(out['boxes'],expected['boxes'])
    results.append(dict(ordinal=j,source=row['input']['source'],selector='minimax',reconstruction='minimum_energy',
        native_temporal_live=True,expert_live=True,all_outputs_exact=True,state_reset=out['state_reset_checked']))
    write(OUT/'live'/f'{b}_{g}.json',results)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['tune','barrier','score','live']);ap.add_argument('--backbone',choices=BS);ap.add_argument('--group',choices=GS)
    a=ap.parse_args()
    if a.stage=='tune':tune(a.backbone,a.group)
    elif a.stage=='barrier':barrier()
    elif a.stage=='score':score()
    else:live(a.backbone,a.group)
