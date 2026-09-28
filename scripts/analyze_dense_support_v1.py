"""F39 offline metrics, development-only common selection, all-source retention."""
import argparse
import math
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.prepare_dense_support_v1 import OUT,plan
from scripts.analyze_spatial10_components_v1 import labels_for,checked_score,summary,aggregate
from scripts.run_dense_support_v1 import dest,existing
METRICS=['vIoU_corrected','sIoU','tIoU','temporal_recall','temporal_precision','span','start_error_seconds','end_error_seconds']

def config_name(lr,b):return f'T1_lr{lr:g}_b{b:g}'
def behavior(old,new):
    a,b=old;c,d=new
    if (a,b)==(c,d):return 'no_op'
    if c<=a and d>=b:return 'expansion'
    if c>=a and d<=b:return 'shrink'
    return 'shift_or_mixed'

def score(stage):
    p=plan();role='development' if stage!='eval' else 'evaluation'
    wanted=[r for rows in p['rows'].values() for r in rows if r['f37_role']==role]
    gt=labels_for(wanted);rows=[]
    names=['T0']+([config_name(lr,b) for lr in p['temporal']['lr'] for b in p['temporal']['beta']] if stage=='dev' else ['T1','T2','T3','T4'])
    for r in wanted:
        key=r['key'];path=dest(stage,key);assert existing(path),path
        x=load(path);fits=x['fits'];ids=x['frame_ids'];arms={};details={}
        if stage=='dev_controls':
            dpath=dest('dev',key);assert existing(dpath)
            chosen=read(OUT/'TEMPORAL_SELECTION.json')['chosen']
            fits={**fits,'T1':load(dpath)['fits'][chosen['name']]}
        a,b=gt[key]['interval'];native=x['T0']['indices'];old=[ids[native[0]],ids[native[1]]+1]
        for n in names:
            fit=fits.get(n);y=fit['final'] if fit else x['T0']
            metric,_=checked_score(y['boxes'],gt[key],ids,y['indices'])
            s,e=y['indices'];new=[ids[s],ids[e]+1];fps=r['input']['fps']
            metric.update(start_error_seconds=abs(new[0]-a)/fps,end_error_seconds=abs(new[1]-b)/fps)
            arms[n]=metric
            details[n]=dict(parameter_TTA=bool(fit and fit['parameter_TTA']),updated=bool(fit and fit['state_delta']>0),
                parameter_change_count=sum(int(bool((v!=fit['path'][0]['state'][k]).any())) for k,v in fit['state'].items()) if fit else 0,
                scalar_change_count=sum(int((v!=fit['path'][0]['state'][k]).sum()) for k,v in fit['state'].items()) if fit else 0,
                state_delta=fit['state_delta'] if fit else 0,best_step=fit['best_step'] if fit else 0,
                physical_interval=new,behavior=behavior(old,new),interval_changed=old!=new,
                duration_change_frames=(new[1]-new[0])-(old[1]-old[0]),
                center_change_frames=(sum(new)-sum(old))/2,
                initial_loss=fit['path'][0]['loss'] if fit else None,final_loss=fit['final']['loss'] if fit else None,
                initial_event=fit['path'][0]['parts']['event'] if fit else None,final_event=fit['final']['parts']['event'] if fit else None,
                final_keep=fit['final']['parts']['keep'] if fit else None,
                gradient_first=fit['path'][0].get('gradient_norm',0) if fit else 0,
                accepted_steps=sum(t.get('accepted',False) for t in fit['path']) if fit else 0,
                backward=fit['backwards'] if fit else 0,seconds=fit['seconds'] if fit else 0,
                failure=fit.get('failure') if fit else None)
            assert metric['sIoU']==arms['T0']['sIoU']
        event_frac=(b-a)/(ids[-1]+1-ids[0]);event_group='short_0_20' if event_frac<=.2 else 'medium_20_50' if event_frac<=.5 else 'long_50_100plus'
        teacher=[]
        from sklearn.metrics import roc_auc_score
        for t in x['teacher']:
            f=np.array(t['frame_ids']);y=((f>=a)&(f<b)).astype(int);prob=t['a'].numpy()
            teacher.append(dict(positions=len(f),prob_mean=float(prob.mean()),prob_quantiles=np.quantile(prob,[.1,.5,.9]).tolist(),
                event_positions=int(y.sum()),auroc=float(roc_auc_score(y,prob)) if 0<y.sum()<len(y) else None,
                GT_fg_prob=float(prob[y==1].mean()) if y.any() else None,
                GT_bg_prob=float(prob[y==0].mean()) if (y==0).any() else None))
        rows.append(dict(key=key,cohort=r['key'].split(':')[0],group=r['group'],source=r['source'],role=role,
            query=r['input']['caption'],frame_ids=ids,GT_interval=[a,b],native_physical_interval=old,
            GT_event_fraction=event_frac,event_length_group=event_group,teacher_offline_diagnostic=teacher,
            arms=arms,details=details,path=str(path),sha256=sha(path),wrong_query=x['wrong_query'],
            full_model_reinsertion=x['full_reinsertion'],teacher_full_model_exact=x['teacher_full_model_exact'],
            independent_source=True,history='Previously exposed F36/F37/F38 pool; not untouched test'))
    pairs=[(n,'T0') for n in names[1:]]+([('T1',n) for n in ['T2','T3','T4']] if stage!='dev' else [])
    summaries={}
    for c in p['rows']:
        rr=[r for r in rows if r['cohort']==c]
        s=summary(rr,names,METRICS,pairs)
        s['behavior']={n:{k:sum(r['details'][n]['behavior']==k for r in rr) for k in ['expansion','shrink','shift_or_mixed','no_op']} for n in names}
        s['updates']={n:dict(updated=sum(r['details'][n]['updated'] for r in rr),
            interval_changed=sum(r['details'][n]['interval_changed'] for r in rr),
            nonzero_gradient=sum(r['details'][n]['gradient_first']>0 for r in rr),
            numerical_failures=sum(r['details'][n]['failure'] is not None for r in rr),
            seconds_mean=float(np.mean([r['details'][n]['seconds'] for r in rr]))) for n in names}
        s['event_length_diagnostic']={g:summary([r for r in rr if r['event_length_group']==g],names,METRICS,pairs)
            for g in ['short_0_20','medium_20_50','long_50_100plus']}
        s['duplicate_source_queries']=len(rr)-len({r['group'] for r in rr})
        summaries[c]=s
    result=dict(rows=rows,summary=summaries,names=names,stage=stage,metrics=METRICS,
        units='raw fractions; report means percent and differences pp; timing seconds',
        GT_use='offline scoring and dev configuration selection only; length groups never deployment rules',
        historical_exposure=True,untouched=False,lock_sha256=sha(OUT/'LOCK.json'))
    write(OUT/(stage.upper()+'_RESULTS.json'),result)
    if stage=='dev':
        table=[]
        for lr in p['temporal']['lr']:
            for b in p['temporal']['beta']:
                n=config_name(lr,b)
                ds={c:summaries[c]['contrasts'][n+' - T0']['vIoU_corrected']['mean'] for c in summaries}
                table.append(dict(name=n,lr=lr,beta=b,delta_by_direction=ds,common_delta=float(np.mean(list(ds.values())))))
        best=max(x['common_delta'] for x in table)
        tied=[x for x in table if best-x['common_delta']<=1e-12]
        chosen=min(tied,key=lambda x:(not(x['lr']==.001 and x['beta']==.1),abs(math.log10(x['lr']/.001)),x['beta'],x['lr']))
        write(OUT/'TEMPORAL_SELECTION.json',dict(chosen=chosen,all_configs=table,steps=5,common_config=True,
            locked_before_eval=True,eval_results_read=False,created_unix=__import__('time').time(),
            dev_results_sha256=sha(OUT/'DEV_RESULTS.json'),criterion=p['temporal']['selection']))
        print('COMMON SELECTED',chosen,flush=True)
    for c,s in summaries.items():
        print(stage,c,{n:{k:round(m[k]['mean']*100,4) for k in METRICS[:3]} for n,m in s['arms'].items()},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['dev','dev_controls','eval']);args=ap.parse_args();score(args.stage)
