"""Fixed additional-parent recheck; revised tradeoff locked before these labels."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.general_region_recheck_v2.run import *
from research.general_region_v1.score import evaluate,truths,candidate_ranks,summary
import numpy as np,torch

def main():
    torch.set_num_threads(1);p=verify();s=read(D/'sealed.json');assert s['count']==64 and not s['GT_read']and s['plan_sha256']==sha(D/'plan.json')and not(D/'score_receipt.json').exists()
    for e in s['files'].values():
        for f,h in [('path','sha256'),('trace_path','trace_sha256'),('capture_path','capture_sha256')]:assert sha(D/e[f])==e[h]
    put(D/'pre_GT_receipt.json',dict(all64_pred_trace_capture_hash_verified=True,seal_sha256=sha(D/'sealed.json'),GT_read=False));put(D/'scorer.json',who());reports={};checks=0;maxerr=0;start=time.perf_counter()
    for ds in ['hc2','vidstg']:
        items=[x for x in p['items']if x['dataset']==ds];assert len(items)==len({x['parent']for x in items})==32;lf=p['labels'][ds];assert sha(PROVIDER/lf['path'])==lf['sha256'];gt=truths(PROVIDER/lf['path'],{x['row']['id']for x in items});rows=[]
        for item in items:
            key=item['alias'];e=s['files'][key];truth=gt[item['row']['id']];values={};fixed={};tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu')
            with np.load(D/e['path'],allow_pickle=False)as z:
                for a in ARMS:
                    values[a],err=evaluate(z[a],item['row'],truth,z[a+'_interval'].tolist(),z['positions8'],ds);checks+=1;maxerr=max(maxerr,err)
                    fixed[a],err=evaluate(z['NativeG6'],item['row'],truth,z[a+'_interval'].tolist(),z['positions8'],ds);checks+=1;maxerr=max(maxerr,err)
                ranks,pairs=candidate_ranks(tr,z['frame_ids'],truth['interval']);moves={a:(z[a+'_interval']-z['NativeG6_interval']).tolist()for a in ARMS}
            rows.append(dict(query=key,parent=f'p{len(rows)+1:03d}',former_split='additional_parent_fixed_recheck',duration_group='short_le5s'if(truth['interval'][1]-truth['interval'][0])/item['row']['metadata']['fps']<=5 else'long_gt5s',values=values,fixed_space=fixed,ranks=ranks,matched_pairs=pairs,movement=moves,feedback_std=float(tr['feedback']['L'][PRIMARY].std()),k16_strength_factor=tr['feedback']['k16_strength_factor'],target_reflection_overlap=float(np.mean([x['target_reflected_soft_overlap']for x in tr['pooling']])),fallback_frames=sum(x['fallback']for x in tr['pooling'])))
            put(D/'score_progress.json',dict(dataset=ds,completed=len(rows),total=32))
        result=summary(rows,p['bootstrap_seed']);means=result['means'];paired=result['paired'];severe=result['severe'];groups=result['groups'];result['original_strict_development_checks']=result['checks']
        result['checks']=dict(Source_G6_actual_t_v_positive=all(means[PRIMARY][m]>means[b][m]for b in ['Source','NativeG6']for m in ['t','v']),
            K16_t_v_within_minus025pp=all(paired[PRIMARY+'-K16'][m]['mean']>=-.25 for m in ['t','v']),Source_short_long_within_minus025pp=all(x['count']>0 and x['t_Source_pp']>=-.25 for x in groups.values()),
            time_wins_at_least_harms=paired[PRIMARY+'-NativeG6']['t']['improved']>=paired[PRIMARY+'-NativeG6']['t']['harmed'],severe_t_v_no_more_than_K16_plus_one=all(severe[PRIMARY][m]<=severe['K16'][m]+1 for m in ['t','v']))
        result['passed']=all(result['checks'].values());result['target_location_specific_advantage_established']=all(paired[PRIMARY+'-'+b]['t']['CI95'][0]>0 for b in ['FullGrid','Reflected']);reports[ds]=result
    passed=all(x['passed']for x in reports.values());report=dict(status='completed_fixed_additional_parent_recheck',primary=PRIMARY,domains=reports,checks=checks,max_metric_error=maxerr,passed_both_domains=passed,original_development_gate_still_failed=True,user_selected_after_development=True,new_full_started=False,plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),prediction_seconds=s['seconds'],counts=s['counts'],scoring_seconds=time.perf_counter()-start)
    put(P/'report.json',report);put(P/'QUALIFICATION.json',dict(qualified_for_new_frozen_full=passed,original_strict_development_passed=False,latest_tradeoff_disclosed=True,domain_checks={k:v['checks']for k,v in reports.items()},new_full_started=False,old_K16_paused=True,report_sha256=sha(P/'report.json')))
    header='dataset\tquery\tmethod\tvIoU\ttIoU\tdelta_v_G6_pp\tdelta_t_G6_pp\tdelta_v_Source_pp\tdelta_t_Source_pp\tboundary_MAE_seconds';lines=[header];bad=[header]
    for ds,rr in reports.items():
        for row in rr['rows']:
            b=row['values']['NativeG6'];src=row['values']['Source']
            for a,v in row['values'].items():
                line=f"{ds}\t{row['query']}\t{a}\t{v['v']:.12g}\t{v['t']:.12g}\t{100*(v['v']-b['v']):.12g}\t{100*(v['t']-b['t']):.12g}\t{100*(v['v']-src['v']):.12g}\t{100*(v['t']-src['t']):.12g}\t{v['boundary_MAE_seconds']:.12g}";lines.append(line)
                if min(v['v']-b['v'],v['t']-b['t'])<-.05:bad.append(line)
    (P/'signed.tsv').write_text('\n'.join(lines)+'\n');(P/'severe.tsv').write_text('\n'.join(bad)+'\n');assert not torch.cuda.is_initialized();verify()
    put(D/'score_receipt.json',dict(report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),checks=checks,max_metric_error=maxerr,CUDA_initialized=False,**who()));put(D/'state.json',dict(status='completed_scored_qualified'if passed else'completed_scored_not_qualified',fingerprint=p['fingerprint']))
    print(json.dumps({k:dict(checks=v['checks'],vsG6=v['paired'][PRIMARY+'-NativeG6'],vsK16=v['paired'][PRIMARY+'-K16'],groups=v['groups'],severe={a:v['severe'][a]for a in [PRIMARY,'K16']})for k,v in reports.items()},indent=2),flush=True)
if __name__=='__main__':main()
