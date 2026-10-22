"""Score development only, freeze once, then confirmation only; never retune."""
import os,sys,json,time,re
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.target_semantic128_shrink_v1.run import *
import numpy as np,torch
from research.native_event_target_semantic_v1.score import evaluate

def eligible(domain,a):
 rows=domain['rows'];mean=domain['means'];d=lambda base,m:100*(mean[a][m]-mean[base][m]);wins=sum(x['values'][a]['t']>x['values']['NativeG6']['t']+1e-12 for x in rows);harms=sum(x['values'][a]['t']<x['values']['NativeG6']['t']-1e-12 for x in rows);sev=lambda arm,m:sum(x['values'][arm][m]<x['values']['NativeG6'][m]-.05 for x in rows)
 group={g:float(np.mean([100*(x['values'][a]['t']-x['values']['Source']['t'])for x in rows if x['duration_group']==g]))for g in ['short_le5s','long_gt5s']};ok=all(d(b,m)>0 for b in ['NativeG6','Source']for m in ['t','v'])and all(v>0 for v in group.values())and wins>harms and all(sev(a,m)<=sev('Loop2',m)for m in ['t','v'])
 return dict(passed=bool(ok),group_t_vs_Source_pp=group,wins=wins,harms=harms,severe={m:sev(a,m)for m in ['t','v']},Loop2severe={m:sev('Loop2',m)for m in ['t','v']})

def truths(path,keys):
 # Bytes of the official file are hashed/read, but only requested split lines are
 # JSON-parsed/evaluated. Confirmation coordinates are never interpreted in dev.
 out={}
 for raw in path.read_bytes().splitlines():
  m=re.search(rb'"id"\s*:\s*"([^"\\]+)"',raw)
  if m and m.group(1).decode()in keys:
   row=json.loads(raw);assert row['id']in keys;out[row['id']]=row
 assert set(out)==keys;return out

def score(split,selected):
 p=verify();s=read(D/'sealed.json');assert s['count']==256 and not s['GT_read']and s['plan_sha256']==sha(D/'plan.json');assert not(D/(split+'_receipt.json')).exists()
 for e in s['files'].values():assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256']
 reports={};checks=0;error=0;start=time.perf_counter()
 for ds in ['hc2','vidstg']:
  items=[x for x in p['items']if x['dataset']==ds and x['split']==split];lf=p['labels'][ds];assert sha(PROVIDER/lf['path'])==lf['sha256'];gt=truths(PROVIDER/lf['path'],{x['row']['id']for x in items});rows=[];assert len(items)==64 and len({x['parent']for x in items})==64
  for item in items:
   key=item['alias'];e=s['files'][key];truth=gt[item['row']['id']];values={};fixed={}
   with np.load(D/e['path'],allow_pickle=False)as z:
    for a in selected:
     values[a],err=evaluate(z[a],item['row'],truth,z[a+'_interval'].tolist(),z['positions8'],ds);error=max(error,err);checks+=1;fixed[a],err=evaluate(z['NativeG6'],item['row'],truth,z[a+'_interval'].tolist(),z['positions8'],ds);error=max(error,err);checks+=1
   rows.append(dict(query=key,parent=f'p{len(rows)+1:03d}',values=values,fixed_space=fixed,duration_group='short_le5s'if(truth['interval'][1]-truth['interval'][0])/item['row']['metadata']['fps']<=5 else'long_gt5s'));put(D/'score_progress.json',dict(split=split,dataset=ds,completed=len(rows),total=64))
  ix=np.random.default_rng(p['bootstrap_seed']).integers(0,64,(10000,64));means={a:{m:float(np.mean([r['values'][a][m]for r in rows if r['values'][a][m]is not None]))for m in rows[0]['values'][a]}for a in selected};paired={}
  compare=[(a,b)for a in selected if a not in ['NativeG6','Source']for b in ['NativeG6','Source','Loop2','Loop1']if a!=b]
  if split=='confirm':
   f=read(D/'frozen_config.json');compare +=[(f['shrink'],b)for b in [f['weak']]+[f['shrink']+suffix for suffix in ['Random','WrongTime','Reflected']]]
  for a,b in compare:
   paired[a+'-'+b]={}
   for m in ['t','v','unobserved_s','boundary_MAE_seconds']:
    valid=[r for r in rows if r['values'][a][m]is not None and r['values'][b][m]is not None];delta=np.array([r['values'][a][m]-r['values'][b][m]for r in valid])*(1 if m.endswith('seconds')else 100);benefit=-delta if m.endswith('seconds')else delta;mix=ix if len(valid)==64 else np.random.default_rng(p['bootstrap_seed']).integers(0,len(valid),(10000,len(valid)));paired[a+'-'+b][m]=dict(mean=float(delta.mean()),CI95=np.quantile(delta[mix].mean(1),[.025,.975]).tolist(),valid_queries=len(valid),excluded_empty_queries=64-len(valid),improved=int((benefit>1e-10).sum()),harmed=int((benefit< -1e-10).sum()),severe_below_minus5pp=int((delta< -5).sum())if not m.endswith('seconds')else None)
  reports[ds]=dict(queries=64,parents=64,means=means,paired=paired,rows=rows)
 out=dict(split=split,domains=reports,plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),checks=checks,max_metric_error=error,scored_arms=selected,other_split_not_JSON_parsed=True,new_model=0,new_predictions=0,seconds=time.perf_counter()-start,exposure=p['split']);put(P/(split+'_report.json'),out);header='dataset\tquery\tmethod\tvIoU\ttIoU\tdelta_v_G6_pp\tdelta_t_G6_pp\tdelta_v_Source_pp\tdelta_t_Source_pp\tboundary_MAE_seconds';lines=[header];severe=[header]
 for ds,rr in reports.items():
  for row in rr['rows']:
   for a,v in row['values'].items():
    b=row['values']['NativeG6'];src=row['values']['Source'];line=f"{ds}\t{row['query']}\t{a}\t{v['v']:.12g}\t{v['t']:.12g}\t{100*(v['v']-b['v']):.12g}\t{100*(v['t']-b['t']):.12g}\t{100*(v['v']-src['v']):.12g}\t{100*(v['t']-src['t']):.12g}\t{v['boundary_MAE_seconds']:.12g}";lines.append(line)
    if min(v['v']-b['v'],v['t']-b['t'])<-.05:severe.append(line)
 (P/(split+'_signed.tsv')).write_text('\n'.join(lines)+'\n');(P/(split+'_severe.tsv')).write_text('\n'.join(severe)+'\n');put(D/(split+'_receipt.json'),dict(report_sha256=sha(P/(split+'_report.json')),seal_sha256=sha(D/'sealed.json'),scored_arms=selected,checks=checks,CUDA_initialized=torch.cuda.is_initialized()));return out

def development():
 out=score('develop',ARMS+['Source']);selection={}
 for family,names in [('shrink',SHRINK),('weak',WEAK)]:
  candidates=[]
  for i,a in enumerate(names):
   stats={ds:eligible(v,a)for ds,v in out['domains'].items()};utility=float(np.mean([100*(v['means'][a]['v']-v['means']['NativeG6']['v'])for v in out['domains'].values()]));candidates.append(dict(arm=a,eligible=all(x['passed']for x in stats.values()),utility_complete_v_pp=utility,domain_checks=stats,order=i))
  pool=[x for x in candidates if x['eligible']]or candidates;best=max(pool,key=lambda x:(x['utility_complete_v_pp'],-x['order']));selection[family]=dict(selected=best['arm'],development_eligible=best['eligible'],candidates=candidates)
 assert not(D/'frozen_config.json').exists();f=dict(shrink=selection['shrink']['selected'],weak=selection['weak']['selected'],selection=selection,development_report_sha256=sha(P/'develop_report.json'),development_receipt_sha256=sha(D/'develop_receipt.json'),plan_sha256=sha(D/'plan.json'),frozen_before_confirmation_GT=True,frozen_epoch=time.time(),no_reselection=True);put(D/'frozen_config.json',f);put(P/'FROZEN_CONFIG.json',f);put(D/'state.json',dict(status='config_frozen_before_confirmation',fingerprint=read(D/'plan.json')['fingerprint']));print(json.dumps(f,indent=2))

def confirmation():
 f=read(D/'frozen_config.json');assert f['development_report_sha256']==sha(P/'develop_report.json')and f['development_receipt_sha256']==sha(D/'develop_receipt.json');selected=BASE+[f['shrink'],f['weak']]+[f['shrink']+suffix for suffix in ['Random','WrongTime','Reflected']]+['Source'];assert len(set(selected))==len(selected);out=score('confirm',selected);a=f['shrink'];checks={ds:eligible(v,a)for ds,v in out['domains'].items()};matched={ds:{s:100*(v['means'][a]['t']-v['means'][a+s]['t'])for s in ['Random','WrongTime','Reflected']}for ds,v in out['domains'].items()};oldsevere=sum(sum(x['Loop2severe'].values())for x in checks.values());newsevere=sum(sum(x['severe'].values())for x in checks.values());qualified=f['selection']['shrink']['development_eligible']and all(x['passed']for x in checks.values())and all(x>0 for v in matched.values()for x in v.values())and(newsevere<oldsevere if oldsevere else newsevere==0)
 weak={ds:v['paired'][a+'-'+f['weak']]for ds,v in out['domains'].items()};result=dict(qualified_for_new_frozen_full=bool(qualified),chosen_shrink=a,chosen_weak_control=f['weak'],development_eligible=f['selection']['shrink']['development_eligible'],confirmation_checks=checks,matched_time_deltas_pp=matched,serious_decline_total=dict(shrink=newsevere,original_loop2=oldsevere),shrink_vs_weak=weak,prototype_estimation_specific_advantage=all(v['t']['CI95'][0]>0 and v['v']['mean']>=0 for v in weak.values()),confirmation_report_sha256=sha(P/'confirm_report.json'),frozen_config_sha256=sha(D/'frozen_config.json'),Source_failures_not_reclassified=True,no_new_full_started=True);put(P/'QUALIFICATION.json',result);put(D/'state.json',dict(status='completed_scored_qualified'if qualified else'completed_scored_not_qualified',fingerprint=read(D/'plan.json')['fingerprint']));print(json.dumps(result,indent=2))
if __name__=='__main__':
 torch.set_num_threads(1);globals()[sys.argv[1]]()
