"""CPU-only dev selection and ONCE frozen confirmation; never reselect onconfirm."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(Path(__file__).parent))
from run import D,P,DEV,verify,VALUES
from full import PROVIDER,sha,read,atomic,identity
from full_score import metric,cluster
import numpy as np,torch

def evaluate(stage):
 p=verify();seal=read(D/stage/'sealed.json');assert seal['count']==64 and seal['GT_read']is False
 if stage=='confirm':assert seal['frozen_config_sha256']==sha(D/'frozen_config.json')
 for e in seal['files'].values():assert sha(D/e['path'])==e['sha256']
 atomic(D/stage/'pre_GT_check.json',dict(all64_predictions_sealed_and_hashed=True,seal_sha256=sha(D/stage/'sealed.json'),GPU_initialized=False))
 arms=['K4','K16','K64','Source','CurrentRegion','NativeG6']if stage=='dev'else['Selected','K16','Source','CurrentRegion','NativeG6'];domains={};checks=0;errmax=0
 for ds in ['hc2','vidstg']:
  label=PROVIDER/p['labels'][ds]['path'];assert sha(label)==p['labels'][ds]['sha256'];truth={x['id']:x for x in map(json.loads,label.read_text().splitlines())if x['dataset']==ds};rows=[]
  for j,item in enumerate(x for x in p[stage+'_items']if x['dataset']==ds):
   key=item['alias'];r=item['row'];gt=truth[r['id']];span=gt['interval'];boxes={int(k):v for k,v in gt['boxes'].items()};values={}
   with np.load(D/seal['files'][key]['path'],allow_pickle=False)as z:
    for a in arms:
     values[a],err=metric(z[a],r,boxes,span,z[a+'_interval'].tolist(),z['positions8']);checks+=1;errmax=max(errmax,err)
   rows.append(dict(query=key,parent=f'p{j+1:03d}',duration_group='short_le5s'if(span[1]-span[0])/r['metadata']['fps']<=5 else'long_gt5s',values=values))
  assert len(rows)==32;means={a:{f:float(np.mean([x['values'][a][f]for x in rows if x['values'][a][f]is not None]))if any(x['values'][a][f]is not None for x in rows)else None for f in rows[0]['values'][a]}for a in arms};severe={a:{f:sum(x['values'][a][f]-x['values']['NativeG6'][f]<-.05 for x in rows)for f in ['v','t']}for a in arms};paired={a+'-'+b:cluster(rows,a,b)for a in arms if a not in ['Source','CurrentRegion','NativeG6']for b in ['K16','CurrentRegion','Source','NativeG6']if a!=b};groups={g:dict(queries=sum(x['duration_group']==g for x in rows),delta_Source={a:{f:float(np.mean([100*(x['values'][a][f]-x['values']['Source'][f])for x in rows if x['duration_group']==g]))if any(x['duration_group']==g for x in rows)else None for f in ['t','v']}for a in arms})for g in ['short_le5s','long_gt5s']}
  domains[ds]=dict(queries=32,means=means,severe_vsG6=severe,paired=paired,groups=groups,rows=rows)
 return dict(stage=stage,domains=domains,checks=checks,max_metric_error=errmax,seal_sha256=sha(D/stage/'sealed.json'))

def write_signed(report,stage):
 header='dataset\tquery\tmethod\tvIoU\ttIoU\tdelta_v_K16_pp\tdelta_t_K16_pp\tdelta_v_G6_pp\tdelta_t_G6_pp';lines=[header];bad=[header]
 for ds,d in report['domains'].items():
  for row in d['rows']:
   for a,v in row['values'].items():
    k=row['values']['K16'];g=row['values']['NativeG6'];line=f"{ds}\t{row['query']}\t{a}\t{v['v']:.12g}\t{v['t']:.12g}\t{100*(v['v']-k['v']):.12g}\t{100*(v['t']-k['t']):.12g}\t{100*(v['v']-g['v']):.12g}\t{100*(v['t']-g['t']):.12g}";lines.append(line)
    if min(v['v']-g['v'],v['t']-g['t'],v['v']-k['v'],v['t']-k['t'])<-.05:bad.append(line)
 (P/f'{stage}_signed.tsv').write_text('\n'.join(lines)+'\n');(P/f'{stage}_severe.tsv').write_text('\n'.join(bad)+'\n')

def dev_score():
 assert not(D/'frozen_config.json').exists();torch.set_num_threads(1);start=time.perf_counter();report=evaluate('dev');candidates={}
 for k in VALUES:
  a='K'+str(k);gates={ds:all(100*(d['means'][a][f]-d['means']['K16'][f])>=-.25-1e-10 and d['severe_vsG6'][a][f]<=d['severe_vsG6']['K16'][f]for f in ['t','v'])for ds,d in report['domains'].items()};gain=float(np.mean([100*(d['means'][a]['v']-d['means']['K16']['v'])for d in report['domains'].values()]));candidates[k]=dict(gates=gates,eligible=all(gates.values()),macro_v_gain_pp=gain)
 eligible=[k for k,c in candidates.items()if c['eligible']];assert 16 in eligible;preference={16:0,64:1,4:2};selected=sorted(eligible,key=lambda k:(-candidates[k]['macro_v_gain_pp'],preference[k]))[0]
 report.update(selection_candidates=candidates,selected_kappa=selected,scoring_seconds=time.perf_counter()-start,plan_sha256=sha(D/'plan.json'));atomic(P/'dev_report.json',report);write_signed(report,'dev');frozen=dict(kappa=selected,one_shared_value_for_both_directions=True,no_more_tuning=True,selection_data='EXPOSED devHC32/Vid32 only',plan_sha256=sha(D/'plan.json'),dev_report_sha256=sha(P/'dev_report.json'),dev_seal_sha256=sha(D/'dev/sealed.json'),confirm_parents_fixed_before_devscore=True,confirmation_results_read=False);atomic(D/'frozen_config.json',frozen);atomic(P/'frozen_config.json',frozen);atomic(D/'dev/score_receipt.json',dict(report_sha256=sha(P/'dev_report.json'),seal_sha256=sha(D/'dev/sealed.json'),**identity()));atomic(D/'state.json',dict(status='frozen_awaiting_confirmation',kappa=selected));assert not torch.cuda.is_initialized();print(json.dumps(dict(candidates=candidates,selected=selected),indent=2))

def confirm_score():
 assert not(D/'confirm/score_receipt.json').exists();torch.set_num_threads(1);start=time.perf_counter();report=evaluate('confirm');frozen=read(D/'frozen_config.json');gates={ds:all(100*(d['means']['Selected'][f]-d['means']['K16'][f])>=-.25-1e-10 and d['severe_vsG6']['Selected'][f]<=d['severe_vsG6']['K16'][f]for f in ['t','v'])for ds,d in report['domains'].items()};gain=float(np.mean([100*(d['means']['Selected']['v']-d['means']['K16']['v'])for d in report['domains'].values()]));supported=all(gates.values())and gain>1e-10 and frozen['kappa']!=16
 report.update(selected_kappa=frozen['kappa'],frozen_config_sha256=sha(D/'frozen_config.json'),transfer_screen=gates,macro_v_gain_pp=gain,parameter_improvement_supported=supported,stop_search=True,no_other_candidate_or_strengths_tested_after_confirmation=True,scoring_seconds=time.perf_counter()-start,retained_candidate_kappa=frozen['kappa']if supported else16);atomic(P/'confirm_report.json',report);write_signed(report,'confirm');atomic(P/'DECISION.json',dict(selected_on_dev=frozen['kappa'],confirmation_supports_parameter_improvement=supported,retained_kappa=report['retained_candidate_kappa'],search_stopped=True,main_full_not_automatically_replaced=True,not_statistical_equivalence=True,historical_exposure_disclosed=True,confirm_report_sha256=sha(P/'confirm_report.json')));atomic(D/'confirm/score_receipt.json',dict(report_sha256=sha(P/'confirm_report.json'),seal_sha256=sha(D/'confirm/sealed.json'),**identity()));atomic(D/'state.json',dict(status='completed_scored',parameter_improvement_supported=supported,retained_kappa=report['retained_candidate_kappa']));assert not torch.cuda.is_initialized();print(json.dumps(dict(kappa=frozen['kappa'],gates=gates,gain=gain,supported=supported,retained=report['retained_candidate_kappa']),indent=2))
if __name__=='__main__':globals()[sys.argv[1]]()
