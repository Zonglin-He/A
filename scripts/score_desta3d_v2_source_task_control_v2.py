"""Sealed source-train positive control: scalar geometry and raw CPU audits."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.score_desta3d_v2_reference_audit import (score_tube_independently,summarize_parents,
    summarize_arm,paired_parent_bootstrap,binary_auc)
from vg_tta.desta3d_v2_prediction_contract import validate_prediction
RUN=ROOT/'artifacts/desta3d_v2/tta_v2/source_task_control_v2'
OUT=RUN/'independent_readback_v1'
ARMS=['no_update','unlabeled','supervised'];METRICS=['vIoU','sIoU','tIoU']
LABEL=RUN/'SOURCE_RECORDS.json'

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def arr(t):return t.numpy().astype(np.float64)
def statehash(state):
    class State:
        def state_dict(self):return state
    return adapter_sha256(State())

def controls():
    pred={'frame_ids':[0,1,2],'positions':[0,1,2],
        'boxes_cxcywh':torch.tensor([[.5,.5,.5,.5]]*3),'geometry_valid':torch.tensor([True]*3),
        'interval':[0,2],'format_ok':True}
    lab={'frame_ids':[0,1,2],'boxes_xyxy':[[.25,.25,.75,.75]]*3,
        'box_valid':[True]*3,'event_active':[True]*3,'event_interval':{'begin_fid':0,'end_fid':3}}
    a=score_tube_independently(pred,lab);assert all(a[m]==1. for m in METRICS)
    pred['geometry_valid'][1]=False;pred['boxes_cxcywh'][1]=0
    b=score_tube_independently(pred,lab);assert b['tIoU']==1. and b['vIoU']==b['sIoU']==2/3
    pred['format_ok']=False;assert all(score_tube_independently(pred,lab)[m]==0. for m in METRICS)
    assert binary_auc([0,1,0,1],[0,1,0,1])==1. and binary_auc([1,1],[0,1]) is None
    parents={'a':{m:.2 for m in METRICS},'b':{m:.8 for m in METRICS}}
    assert paired_parent_bootstrap(parents,parents,'vIoU')['bootstrap_ci95_pp']==[0.,0.]
    return {'status':'passed','controls':5,'GPU':False,'GT_read':False}

def preflight():
    paths=[Path(__file__),ROOT/'scripts/score_desta3d_v2_reference_audit.py',
        ROOT/'scripts/desta3d_source_fit_v1.py',ROOT/'vg_tta/metrics.py',
        ROOT/'vg_tta/desta3d_v2_prediction_contract.py',RUN/'CONFIG.json',RUN/'INPUTS.json']
    save_once(RUN/'SCORER_PREFLIGHT.json',{**controls(),'time':time.time(),
        'pins':{str(p):sha(p) for p in paths},'source_label_sha':sha(LABEL),
        'source_labels_already_used_in_worker':True,'target_data_read':False})

def score():
    torch.set_num_threads(2)
    assert not OUT.exists()
    for p,h in read(RUN/'SCORER_PREFLIGHT.json')['pins'].items():assert sha(Path(p))==h
    done=read(RUN/'COMPLETE.json');seal=read(RUN/'ALL_PREDICTIONS_SEAL.json')
    assert done['episodes']==16 and done['optimizer_steps']==96 and done['predictions']==48
    assert done['seal_sha']==sha(RUN/'ALL_PREDICTIONS_SEAL.json')
    for p,h in seal['pins'].items():assert sha(Path(p))==h,p
    for p,h in read(RUN/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    cfg=read(RUN/'CONFIG.json');rows=read(RUN/'INPUTS.json');assert len(rows)==len({r['source'] for r in rows})==16
    initial=load(Path(cfg['checkpoint']['checkpoint']))['adapter'];digest=statehash(initial)
    assert digest==cfg['checkpoint']['adapter_sha256']
    payloads={};updates={a:[] for a in ARMS[1:]};rawstats=[];identities=[];maxsum=0.;maxnorm=0.;maxdot=0.;deltaerr=0.
    for i,row in enumerate(rows):
        ep=RUN/'episodes'/f'{i:02}';ident=read(ep/'INPUT_IDENTITY.json')
        assert ident['key']==row['key'] and ident['initial_adapter_sha']==digest
        assert all(read(ep/'INITIAL_REPEAT_CHECK.json').values())
        ec=read(ep/'EPISODE_COMPLETE.json');assert ec['reset_exact'] and ec['optimizer_steps']==6
        for p,h in ec['pins'].items():assert sha(Path(p))==h
        payloads[row['key']]={}
        for arm in ARMS:
            p=load(ep/(arm+'.pt'));validate_prediction(p,len(row['input']['frame_ids']))
            assert p['key']==row['key'] and p['source']==row['source']
            assert p['frame_ids']==row['input']['frame_ids'] and p['video_sha256']==row['input']['video_sha256']
            assert p['preprocess']==ident['preprocess'] and p['sourcefit_adapter_sha']==digest
            assert not p['decoder_uses_GT'] and not p['target_GT_read'] and not p['unlabeled_update_uses_GT']
            assert p['source_labels_used_for_update']==(arm=='supervised')
            payloads[row['key']][arm]=p
            if arm=='no_update':assert p['adapter_sha']==digest;continue
            summ=read(ep/arm/'SUMMARY.json');final=load(ep/arm/'FINAL_CALIBRATION.pt')
            assert len(final)==10 and sum(v.numel() for v in final.values())==66816
            assert statehash({**initial,**final})==p['adapter_sha']==summ['after_adapter_sha']
            assert all(not v or n in final for n,v in summ['changed_tensors'].items())
            deltas=[];steps=[]
            for j in range(1,4):
                r=load(ep/arm/f'step{j}.pt');g=r['raw'];u=r['update'];names=r['ordered_names']
                assert names==list(final) and r['parameter_count']==66816 and r['backbone_frozen']
                assert all(v.shape==(66816,) and torch.isfinite(v).all() for v in g.values())
                assert r['source_labels_used_for_update']==(arm=='supervised') and not r['target_inputs_read']
                assert u['live_parameter_binding'] and set(u['actual_Adam_counters'])==set(names)
                assert set(u['actual_Adam_counters'].values())=={j}
                task=(g['task_event']+g['task_spatial']).numpy().astype(np.float64)
                ul=arr(g['unlabeled_total']);delta=arr(g['actual_delta']);deltas.append(delta)
                component=sum(arr(g[k]) for k in ['latent','referent','event','parameter_anchor','alignment'])
                err=float(np.max(np.abs(component-ul)));maxsum=max(maxsum,err);assert np.allclose(component,ul,atol=1e-7,rtol=2e-4)
                applied=ul if arm=='unlabeled' else task
                norm=float(np.linalg.norm(applied));maxnorm=max(maxnorm,abs(norm-u['pre_clip_norm']))
                assert abs(norm-u['pre_clip_norm'])<max(1e-7,norm*2e-6)
                dot=float(np.dot(task,delta));maxdot=max(maxdot,abs(dot-u['task_dot_delta']['dot']));assert abs(dot-u['task_dot_delta']['dot'])<1e-12
                denom=np.linalg.norm(task)*np.linalg.norm(ul)
                cos=float(np.dot(task,ul)/denom) if denom else None
                assert cos is None or abs(cos-u['task_vs_unlabeled']['cosine'])<1e-12
                sd={'key':row['key'],'source':row['source'],'arm':arm,'step':j,
                    'task_CE_before':r['task']['total'],'unlabeled_loss_before':r['unlabeled']['total'],
                    'task_dot_actual_Adam_delta':dot,'task_vs_unlabeled_cosine':cos,
                    'task_norm':float(np.linalg.norm(task)),'unlabeled_norm':float(np.linalg.norm(ul)),
                    'clip_triggered':u['clip_triggered'],'actual_delta_norm':float(np.linalg.norm(delta)),
                    'task_token_counts':{b:r['task'][b]['tokens'] for b in ['event','spatial']},
                    'branch_injection':{b:{k:r['task'][b][k] for k in ['relative_injection_norm','cast_changed_elements']} for b in ['event','spatial']}}
                steps.append(sd);rawstats.append(sd)
            end=np.concatenate([arr(final[n]).reshape(-1) for n in names])-np.concatenate([arr(initial[n]).reshape(-1) for n in names])
            de=float(np.max(np.abs(sum(deltas)-end)));deltaerr=max(deltaerr,de);assert de<1e-10
            updates[arm].append({'key':row['key'],'source':row['source'],'task_CE_before':steps[0]['task_CE_before'],
                'task_CE_after':summ['task_after']['total'],'task_CE_delta':summ['task_after']['total']-steps[0]['task_CE_before'],
                'unlabeled_before':steps[0]['unlabeled_loss_before'],'unlabeled_after':summ['unlabeled_after'],'steps':steps})
        identities.append({'key':row['key'],'source':row['source'],'frame_count':len(row['input']['frame_ids']),
            'formats':{a:p['format_ok'] for a,p in payloads[row['key']].items()},'initial_task_repeat_exact':True})
    save_once(OUT/'PRE_SCORE_AUDIT.json',{'time':time.time(),'status':'passed','predictions':48,'optimizer_steps':96,
        'queries':16,'parents':16,'seal_sha':sha(RUN/'ALL_PREDICTIONS_SEAL.json'),
        'source_labels_already_used_for_supervised_updates_and_side_diagnostics':True,'target_data_read':False,
        'raw_checks':{'component_sum_max_error':maxsum,'preclip_norm_max_error':maxnorm,'task_dot_delta_max_error':maxdot,'three_delta_telescope_error':deltaerr},
        'identities':identities,'scorer_sha':sha(Path(__file__))})
    assert sha(LABEL)==read(RUN/'SCORER_PREFLIGHT.json')['source_label_sha']
    labels={r['key']:r for r in read(LABEL)};results={a:[] for a in ARMS};case=[]
    for row in rows:
        k=row['key'];label=labels[k]
        for arm,p in payloads[k].items():
            metrics=score_tube_independently(p,label)
            results[arm].append({'key':k,'source':row['source'],'metrics':metrics,
                'event_AUROC':binary_auc(label['event_active'],p['event_logits'].reshape(-1).tolist()),
                'interval':p['interval'],'invalid_geometry_boxes':int((~p['geometry_valid']).sum())})
        base=payloads[k]['no_update']
        case.append({'key':k,'source':row['source'],'intervals':{a:p['interval'] for a,p in payloads[k].items()},
            'event_references':{a:p['readout']['spatial_reference_token_ids'] for a,p in payloads[k].items()}})
    parents={a:summarize_parents(x) for a,x in results.items()}
    pairs=[('unlabeled','no_update'),('supervised','no_update'),('supervised','unlabeled')]
    comparisons={a+'_minus_'+b:{m:paired_parent_bootstrap(parents[a],parents[b],m) for m in METRICS} for a,b in pairs}
    retention={}
    for m in ['vIoU','tIoU']:
        good={r['key'] for r in results['no_update'] if r['metrics'][m]>.5}
        retention[m]={'definition':'B1 no-update > 0.5; retained if candidate > 0.5','eligible':len(good),
            'arms':{a:{'retained':sum(r['key'] in good and r['metrics'][m]>.5 for r in x),
                       'lost':[r['key'] for r in x if r['key'] in good and r['metrics'][m]<=.5]} for a,x in results.items()}}
    report={'status':'completed_source_train_diagnostic','source_label_sha':sha(LABEL),
        'audit_sha':sha(OUT/'PRE_SCORE_AUDIT.json'),'arms':{a:{'query_rows':x,'summary':summarize_arm(x,parents[a])} for a,x in results.items()},
        'comparisons':comparisons,'retention':retention,'updates':updates,'cases':case,'raw_step_diagnostics':rawstats,
        'scope':'16 preselected source training parents, one query each; supervised diagnostic uses labels; no target input, selection or generalization claim',
        'CE_not_native_cached_likelihood':True,'seed':20260927,'CI':'descriptive paired-parent bootstrap, no multiplicity correction'}
    save_once(OUT/'REPORT.json',report)
    lines=['# 源任务监督正控：固定16父源、同66816参数、3步','',report['scope'],'',
        '所有48原生预测、96步原始向量先封存并核验；源标签已用于监督臂及旁路任务诊断，不宣称worker未读GT。','',
        '|状态|父源宏 vIoU %|sIoU %|tIoU %|','|---|---:|---:|---:|']
    for a in ARMS:lines.append('|'+a+'|'+'|'.join(f"{report['arms'][a]['summary']['parent_macro'][m]*100:.6f}" for m in METRICS)+'|')
    lines+=['','|比较|Δv pp|95% CI|正/负/零父源|>5pp损害|','|---|---:|---|---|---:|']
    for name,c in comparisons.items():
        v=c['vIoU'];lines.append(f"|{name}|{v['mean_delta_pp']:+.6f}|{v['bootstrap_ci95_pp']}|{v['positive_parents']}/{v['negative_parents']}/{v['zero_parents']}|{v['severe_loss_below_minus5pp']}|")
    for arm,x in updates.items():
        ds=[r['task_CE_delta'] for r in x];rs=[r for r in rawstats if r['arm']==arm]
        lines+=['',f"{arm}: task CE下降 {sum(d<0 for d in ds)}/16，均值变化 {np.mean(ds):+.8f}；task梯度·实际Adam位移<0 {sum(r['task_dot_actual_Adam_delta']<0 for r in rs)}/48。CE/一阶下降不等于native tube改善。"]
    lines+=['','原生GT teacher forcing与cached free decode不同；3步正控不能排除训练充分性。保留全部病例、格式失败、连续损害与阈值保持。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save_once(OUT/'COMPLETE.json',{'status':'independently_scored','report_sha':sha(OUT/'REPORT.json'),'source_only':True})
    print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','score']);a=p.parse_args().action
    try:{'preflight':preflight,'score':score}[a]()
    except BaseException:
        if a=='score' and not (OUT/'FAILURE.json').exists():save_once(OUT/'FAILURE.json',{'time':time.time(),'failure':traceback.format_exc()})
        raise
