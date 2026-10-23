"""IndependentCPU labels ONLY after all outputs seal; risk-first selection then ONCE confirmation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,json,math
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(Path(__file__).parent))
from run import D,P,verify,CONFIGS
from full import PROVIDER,sha,read,atomic,identity
from full_score import metric,cluster
import numpy as np,torch

def evaluate(stage):
 p=verify();seal=read(D/stage/'sealed.json');n=192 if stage=='dev'else 64
 assert seal['count']==n and seal['GT_read']is False
 if stage=='confirm':assert seal['frozen_config_sha256']==sha(D/'frozen_config.json')
 for e in seal['files'].values():assert sha(D/e['path'])==e['sha256']
 arms=[*CONFIGS,'Source','CurrentRegion','NativeG6']if stage=='dev'else['Selected','K16','Source','CurrentRegion','NativeG6'];base='K16_B100'if stage=='dev'else'K16';out={};checks=0;errmax=0
 for ds in ['hc2','vidstg']:
  label=PROVIDER/p['labels'][ds]['path'];assert sha(label)==p['labels'][ds]['sha256'];truth={x['id']:x for x in map(json.loads,label.read_text().splitlines())if x['dataset']==ds};rows=[]
  for j,i in enumerate(x for x in p[stage+'_items']if x['dataset']==ds):
   r=i['row'];g=truth[r['id']];span=g['interval'];boxes={int(k):v for k,v in g['boxes'].items()};values={};e=seal['files'][i['alias']]
   with np.load(D/e['path'],allow_pickle=False)as z:
    for a in arms:values[a],err=metric(z[a],r,boxes,span,z[a+'_interval'].tolist(),z['positions8']);checks+=1;errmax=max(errmax,err)
   rows.append(dict(query=i['alias'],parent=f'P{j+1:03d}',duration_group='short_le5s'if(span[1]-span[0])/r['metadata']['fps']<=5 else'long_gt5s',values=values))
  assert len(rows)==n//2;means={a:{f:float(np.mean([x['values'][a][f]for x in rows if x['values'][a][f]is not None]))if any(x['values'][a][f]is not None for x in rows)else None for f in rows[0]['values'][a]}for a in arms};risks={}
  for a in arms:
   risks[a]={}
   for b in ['NativeG6',base,'Source','CurrentRegion']:
    loss=np.array([max(0.,x['values'][b]['t']-x['values'][a]['t'],x['values'][b]['v']-x['values'][a]['v'])for x in rows]);severe=loss>.05;risks[a][b]=dict(union_count=int(severe.sum()),union_fraction=float(severe.mean()),time_count=sum(x['values'][a]['t']-x['values'][b]['t']<-.05 for x in rows),v_count=sum(x['values'][a]['v']-x['values'][b]['v']<-.05 for x in rows),worst10_mean_loss_pp=float(100*np.sort(loss)[-math.ceil(len(rows)*.1):].mean()))
  pairs={a+'-'+b:cluster(rows,a,b)for a in arms if a not in ['Source','CurrentRegion','NativeG6']for b in [base,'NativeG6','Source','CurrentRegion']if a!=b};groups={g:dict(queries=sum(x['duration_group']==g for x in rows),deltas_Source={a:{f:float(np.mean([100*(x['values'][a][f]-x['values']['Source'][f])for x in rows if x['duration_group']==g]))if any(x['duration_group']==g for x in rows)else None for f in ['t','v']}for a in arms})for g in ['short_le5s','long_gt5s']};out[ds]=dict(means=means,risks=risks,paired=pairs,groups=groups,rows=rows)
 return dict(stage=stage,domains=out,checks=checks,max_metric_error=errmax,seal_sha256=sha(D/stage/'sealed.json'),plan_sha256=sha(D/'plan.json'))

def write(report,stage):
 atomic(P/(stage+'_report.json'),report);head='dataset\tquery\tmethod\tbaseline\tdelta_t_pp\tdelta_v_pp';lines=[head];bad=[head]
 base='K16_B100'if stage=='dev'else'K16'
 for ds,d in report['domains'].items():
  for row in d['rows']:
   for a,v in row['values'].items():
    for b in [base,'Source','NativeG6','CurrentRegion']:
     dt=100*(v['t']-row['values'][b]['t']);dv=100*(v['v']-row['values'][b]['v']);line=f"{ds}\t{row['query']}\t{a}\t{b}\t{dt:.12g}\t{dv:.12g}";lines.append(line)
     if min(dt,dv)<-5:bad.append(line)
 (P/(stage+'_signed.tsv')).write_text('\n'.join(lines)+'\n');(P/(stage+'_severe.tsv')).write_text('\n'.join(bad)+'\n')

def dev_score():
 assert not(D/'frozen_config.json').exists();torch.set_num_threads(1);started=time.perf_counter();r=evaluate('dev');candidates={}
 for a,(k,beta)in CONFIGS.items():
  guard={ds:all(100*(d['means'][a][f]-d['means']['NativeG6'][f])>=-.1 for f in ['t','v'])and 100*(d['means'][a]['v']-d['means']['K16_B100']['v'])>=-.5 and d['means'][a]['v']>d['means']['Source']['v']for ds,d in r['domains'].items()};risk=[d['risks'][a]['NativeG6']for d in r['domains'].values()];obj=[max(x['union_fraction']for x in risk),float(np.mean([x['union_fraction']for x in risk])),float(np.mean([x['worst10_mean_loss_pp']for x in risk])),-float(np.mean([d['means'][a]['v']for d in r['domains'].values()]))];candidates[a]=dict(eligible=all(guard.values()),guards=guard,objective=obj,kappa=k,beta=beta)
 eligible=[a for a,v in candidates.items()if v['eligible']];winner=min(eligible,key=lambda a:(candidates[a]['objective'],abs(CONFIGS[a][0]-16)+abs(CONFIGS[a][1]-1)))if eligible else'K16_B100';k,beta=CONFIGS[winner];r.update(candidates=candidates,winner=winner,scoring_seconds=time.perf_counter()-started);write(r,'dev');frozen=dict(winner=winner,kappa=k,beta=beta,no_more_tuning=True,one_shared_pair=True,dev_report_sha256=sha(P/'dev_report.json'),plan_sha256=sha(D/'plan.json'),confirmation_results_read=False);atomic(D/'frozen_config.json',frozen);atomic(P/'frozen_config.json',frozen);atomic(D/'dev/score_receipt.json',dict(report_sha256=sha(P/'dev_report.json'),seal_sha256=r['seal_sha256'],**identity()));atomic(D/'state.json',dict(status='frozen_awaiting_confirmation',**frozen));print(json.dumps(dict(winner=winner,candidates=candidates),indent=2));assert not torch.cuda.is_initialized()

def confirm_score():
 assert not(D/'confirm/score_receipt.json').exists();torch.set_num_threads(1);started=time.perf_counter();r=evaluate('confirm');f=read(D/'frozen_config.json');guards={ds:d['risks']['Selected']['NativeG6']['union_count']<=d['risks']['K16']['NativeG6']['union_count']and all(100*(d['means']['Selected'][x]-d['means']['K16'][x])>=-.5 for x in ['t','v'])for ds,d in r['domains'].items()};reduction=sum(d['risks']['K16']['NativeG6']['union_count']-d['risks']['Selected']['NativeG6']['union_count']for d in r['domains'].values());passed=all(guards.values());deployed=dict(kappa=f['kappa'],beta=f['beta'])if passed else dict(kappa=16,beta=1.);r.update(guards=guards,severe_queries_reduced_total=reduction,risk_reduction_supported=passed and reduction>0,deployed=deployed,selected=f,scoring_seconds=time.perf_counter()-started,search_stopped=True);write(r,'confirm');decision=dict(status='completed_scored',deployed=deployed,confirmation_passed=passed,risk_reduction_supported=r['risk_reduction_supported'],all_failures_preserved=True,Native2D_user_selected_as_main=True,not_untouched_test=True,no_further_selection=True,confirm_report_sha256=sha(P/'confirm_report.json'));atomic(P/'DECISION.json',decision);atomic(D/'state.json',decision);atomic(D/'confirm/score_receipt.json',dict(report_sha256=sha(P/'confirm_report.json'),seal_sha256=r['seal_sha256'],**identity()));print(json.dumps(decision,indent=2));assert not torch.cuda.is_initialized()
if __name__=='__main__':globals()[sys.argv[1]]()
