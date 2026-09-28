"""Source-only sealed FP32 task-head control, independent scalar geometry."""
import argparse,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.score_desta3d_v2_reference_audit import score_tube_independently,summarize_parents,summarize_arm,paired_parent_bootstrap,binary_auc
from scripts.score_desta3d_v2_source_task_control_v2 import controls,statehash
from vg_tta.desta3d_v2_prediction_contract import validate_prediction
RUN=ROOT/'artifacts/desta3d_v2/tta_v2/source_fp32_head_control_v1';OUT=RUN/'independent_readback_v1'
ARMS=['no_update','unlabeled','supervised','supervised_fp32'];NEW=ARMS[-1];MS=['vIoU','sIoU','tIoU']

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def a(x):return x.numpy().astype(np.float64)

def preflight():
    save_once(RUN/'SCORER_PREFLIGHT.json',{**controls(),'time':time.time(),'source_GT_already_supervised':True,
        'source_label_sha':sha(RUN/'SOURCE_RECORDS.json'),'pins':{str(p):sha(p) for p in [Path(__file__),
        ROOT/'scripts/score_desta3d_v2_reference_audit.py',ROOT/'scripts/crosscheck_desta3d_v2_source_fp32_control.py',RUN/'CONFIG.json',RUN/'INPUTS.json']}})

def score():
    torch.set_num_threads(2);assert not OUT.exists()
    for p,h in read(RUN/'SCORER_PREFLIGHT.json')['pins'].items():assert sha(Path(p))==h
    done=read(RUN/'COMPLETE.json');assert done['episodes']==16 and done['new_steps']==48 and done['predictions']==64
    assert done['seal_sha']==sha(RUN/'ALL_PREDICTIONS_SEAL.json')
    for p,h in read(RUN/'ALL_PREDICTIONS_SEAL.json')['pins'].items():assert sha(Path(p))==h,p
    for p,h in read(RUN/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    cfg=read(RUN/'CONFIG.json');rows=read(RUN/'INPUTS.json');initial=load(Path(cfg['checkpoint']['checkpoint']))['adapter'];digest=statehash(initial)
    payloads={};updates=[];rawstats=[];sumerr=normerr=doterr=deltaerr=0.
    from scripts.desta3d_v2_source_fp32_head_control import same
    for i,row in enumerate(rows):
        ep=RUN/'episodes'/f'{i:02}';ec=read(ep/'EPISODE_COMPLETE.json');ident=read(ep/'INPUT_IDENTITY.json')
        assert ec['reset_exact'] and ec['original_B1_native_exact'] and ec['new_steps']==3
        assert same(load(ep/'REPLAY_NO_UPDATE.pt'),load(ep/'no_update.pt'))
        payloads[row['key']]={}
        for arm in ARMS:
            p=load(ep/(arm+'.pt'));validate_prediction(p,len(row['input']['frame_ids']))
            assert p['key']==row['key'] and p['source']==row['source'] and p['frame_ids']==row['input']['frame_ids']
            assert p['preprocess']==ident['preprocess'] and p['sourcefit_adapter_sha']==digest
            assert not p['decoder_uses_GT'] and not p['target_GT_read'] and not p['unlabeled_update_uses_GT']
            if arm=='no_update':assert p['adapter_sha']==digest
            payloads[row['key']][arm]=p
        final=load(ep/NEW/'FINAL_CALIBRATION.pt');summary=read(ep/NEW/'SUMMARY.json');opt=load(ep/NEW/'FINAL_OPTIMIZER.pt')
        assert len(final)==10 and sum(x.numel() for x in final.values())==66816
        assert statehash({**initial,**final})==payloads[row['key']][NEW]['adapter_sha']==summary['after_adapter_sha']
        assert all(not v or k in final for k,v in summary['changed_tensors'].items())
        assert all(isinstance(k,int) for k in opt['state']) and len(opt['state'])==10
        assert {int(v['step']) for v in opt['state'].values()}=={3}
        deltas=[];steps=[]
        for j in range(1,4):
            s=load(ep/NEW/f'step{j}.pt');g=s['raw'];u=s['update'];names=s['ordered_names']
            assert names==list(final) and s['parameter_count']==66816 and s['backbone_frozen'] and s['only_calibration_trainable']
            assert s['source_supervised_update'] and not s['target_data']
            assert all(v.shape==(66816,) and torch.isfinite(v).all() for v in g.values())
            assert u['live_parameter_binding'] and set(u['actual_Adam_counters'].values())=={j}
            summed=a(g['task_event'])+a(g['task_spatial']);task=a(g['task_total']);delta=a(g['actual_delta']);deltas.append(delta)
            se=float(np.max(np.abs(summed-task)));sumerr=max(sumerr,se);assert se<1e-7
            ne=abs(float(np.linalg.norm(task))-u['pre_clip_norm']);normerr=max(normerr,ne);assert ne<max(1e-7,np.linalg.norm(task)*2e-6)
            dot=float(task@delta);de=abs(dot-u['task_dot_actual_delta']);doterr=max(doterr,de);assert de<1e-12
            x={'key':row['key'],'source':row['source'],'step':j,'BF16_CE_before':s['BF16']['total'],'FP32_CE_before':s['FP32']['total'],
                'task_dot_actual_delta':dot,'norm':float(np.linalg.norm(task)),'clip':u['clip_triggered'],'delta_norm':float(np.linalg.norm(delta))}
            steps.append(x);rawstats.append(x)
        end=np.concatenate([a(final[n]).ravel()-a(initial[n]).ravel() for n in names]);e=float(np.max(np.abs(sum(deltas)-end)))
        deltaerr=max(deltaerr,e);assert e<1e-10
        old=read(ep/'OLD_SUPERVISED_DUAL_CE.json');assert old['adapter_sha']==payloads[row['key']]['supervised']['adapter_sha']
        updates.append({'key':row['key'],'source':row['source'],'BF16_before':steps[0]['BF16_CE_before'],'FP32_before':steps[0]['FP32_CE_before'],
            'new_BF16_after':summary['BF16_after']['total'],'new_FP32_after':summary['FP32_after']['total'],
            'old_BF16_after':old['BF16']['total'],'old_FP32_after':old['FP32']['total'],'steps':steps})
    save_once(OUT/'PRE_SCORE_AUDIT.json',{'status':'passed','time':time.time(),'predictions':64,'new_steps':48,'source_queries':16,'parents':16,
        'source_GT_already_used_in_supervised_worker':True,'target_data':False,'exact_B1_native_replays':16,'raw_sum_error':sumerr,
        'norm_error':normerr,'dot_error':doterr,'delta_telescope_error':deltaerr,'seal_sha':sha(RUN/'ALL_PREDICTIONS_SEAL.json')})
    lp=RUN/'SOURCE_RECORDS.json';assert sha(lp)==read(RUN/'SCORER_PREFLIGHT.json')['source_label_sha']
    labels={r['key']:r for r in read(lp)};results={arm:[] for arm in ARMS};cases=[]
    for row in rows:
        key=row['key'];label=labels[key]
        for arm,p in payloads[key].items():
            m=score_tube_independently(p,label)
            results[arm].append({'key':key,'source':row['source'],'metrics':m,'event_AUROC':binary_auc(label['event_active'],p['event_logits'].reshape(-1).tolist()),
                'interval':p['interval'],'invalid_geometry_boxes':int((~p['geometry_valid']).sum())})
        cases.append({'key':key,'source':row['source'],'intervals':{a:p['interval'] for a,p in payloads[key].items()}})
    parents={a:summarize_parents(x) for a,x in results.items()}
    pairs=[(NEW,'no_update'),(NEW,'supervised'),(NEW,'unlabeled'),('supervised','no_update')]
    comparisons={a+'_minus_'+b:{m:paired_parent_bootstrap(parents[a],parents[b],m) for m in MS} for a,b in pairs}
    retention={}
    for m in ['vIoU','tIoU']:
        good={r['key'] for r in results['no_update'] if r['metrics'][m]>.5}
        retention[m]={'eligible':len(good),'definition':'B1>0.5 retained if candidate>0.5','arms':{a:{'retained':sum(r['key'] in good and r['metrics'][m]>.5 for r in x),
            'lost':[r['key'] for r in x if r['key'] in good and r['metrics'][m]<=.5]} for a,x in results.items()}}
    ces={}
    for which in ['old','new']:
        ces[which]={}
        for dtype in ['BF16','FP32']:
            d=np.array([u[f'{which}_{dtype}_after']-u[f'{dtype}_before'] for u in updates])
            ces[which][dtype]={'mean_delta':float(d.mean()),'decreased_queries':int((d<0).sum()),'increased_queries':int((d>0).sum()),'deltas':d.tolist()}
    report={'status':'completed_source_training_diagnostic','source_label_sha':sha(lp),'audit_sha':sha(OUT/'PRE_SCORE_AUDIT.json'),
        'arms':{a:{'query_rows':x,'summary':summarize_arm(x,parents[a])} for a,x in results.items()},'comparisons':comparisons,'retention':retention,
        'updates':updates,'raw_step_diagnostics':rawstats,'cases':cases,'CE_same_definition_changes':ces,
        'scope':'16 fixed source training parents, supervised diagnostic only, no target, one seed, uncorrected descriptive parent CI',
        'all_loss_and_native_negative_cases_retained':True}
    save_once(OUT/'REPORT.json',report)
    lines=['# 源16：仅最终task-head FP32，固定3步监督正控','','64原生预测全seal、原48复用逐query实际B1身份/输出核验，48新步骤及完整raw/最终Adam审计后评分。源GT用于明示监督/诊断，非未曝光评估。','',
        '|状态|vIoU %|sIoU %|tIoU %|','|---|---:|---:|---:|']
    for a in ARMS:lines.append('|'+a+'|'+'|'.join(f"{report['arms'][a]['summary']['parent_macro'][m]*100:.6f}" for m in MS)+'|')
    lines+=['','|比较|Δv pp|95% CI|正/负/零父源|>5pp损害|','|---|---:|---|---|---:|']
    for k,c in comparisons.items():
        v=c['vIoU'];lines.append(f"|{k}|{v['mean_delta_pp']:+.6f}|{v['bootstrap_ci95_pp']}|{v['positive_parents']}/{v['negative_parents']}/{v['zero_parents']}|{v['severe_loss_below_minus5pp']}|")
    lines+=['','|训练臂|CE读出定义|最终−初始均值|下降query|','|---|---|---:|---:|']
    for x,v in ces.items():
        for dtype,d in v.items():lines.append(f"|{x}|{dtype}|{d['mean_delta']:+.8f}|{d['decreased_queries']}/16|")
    lines+=['',f"新FP32梯度·实际Adamdelta负向 {sum(x['task_dot_actual_delta']<0 for x in rawstats)}/48；clip {sum(x['clip'] for x in rawstats)}/48。",'','原GT teacher-forcing非cached-native；CE下降不等于定位收益。全部病例/连续损害/阈值保持均在REPORT.json。不得跨两种不同CE定义比较刻度。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');save_once(OUT/'COMPLETE.json',{'status':'independently_scored','report_sha':sha(OUT/'REPORT.json')})
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','score']);a=p.parse_args().action
    try:{'preflight':preflight,'score':score}[a]()
    except BaseException:
        if a=='score' and not (OUT/'FAILURE.json').exists():save_once(OUT/'FAILURE.json',{'error':traceback.format_exc(),'time':time.time()})
        raise
