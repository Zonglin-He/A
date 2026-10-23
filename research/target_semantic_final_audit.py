"""Postseal final-round verification/case summary. No new inference or GT read."""
import sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from research.target_semantic_final_v1.run import D,P,A,B,PRIMARY,verify,read,put,sha

def main():
    torch.set_num_threads(1);start=time.perf_counter();p=verify();s=read(D/'sealed.json');report=read(P/'report.json')
    assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json')
    assert report['development_passed'] is False
    rows=[];normerr=0;zero_controls=0
    for item in p['items']:
        e=s['files'][item['alias']];assert sha(D/e['trace_path'])==e['trace_sha256'] and sha(D/e['path'])==e['sha256']
        tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu')['feedback'];L=tr['L'];norm=np.linalg.norm(L[PRIMARY])
        for a in ['UniformMatched','PermutedMatched']:
            if np.linalg.norm(L[a])==0 and norm>0:zero_controls+=1
            normerr=max(normerr,float(abs(np.linalg.norm(L[a])-norm)))
        for v in tr['prototypes']:
            assert np.array_equal(np.sort(v['reliability']),np.sort(v['reliability'][tr['permutation']]))
        old=next(x for x in report['domains'][item['dataset']]['rows'] if x['query']==item['alias'])
        rows.append(dict(query=item['alias'],dataset=item['dataset'],r_mean=old['reliability_mean'],r_std=old['reliability_std'],cosine=old['contrast_cosine'],uniform_factor=old['uniform_factor']))
    processes=[]
    for n in ['launch.json','score_launch.json']:
        x=read(D/n);f=Path('/proc')/str(x['pid'])/'stat';actual=None
        if f.exists():actual=int(f.read_text().rsplit(')',1)[1].split()[19])
        processes.append(dict(pid=x['pid'],expected_startticks=x['startticks'],actual_startticks=actual,exited=actual!=x['startticks']))
    assert all(x['exited'] for x in processes)
    cases={}
    for ds,v in report['domains'].items():
        ordered=sorted(v['rows'],key=lambda x:x['values'][PRIMARY]['t']-x['values']['NativeG6']['t'])
        cases[ds]=[]
        for row in ordered[:3]+ordered[-3:]:
            b=row['values']['NativeG6'];a=row['values'][PRIMARY];matched=row['values']['PermutedMatched'];uni=row['values']['UniformMatched']
            cases[ds].append(dict(query=row['query'],group=row['duration_group'],native_t=100*b['t'],delta_t_G6_pp=100*(a['t']-b['t']),delta_v_G6_pp=100*(a['v']-b['v']),
                permuted_delta_t_pp=100*(matched['t']-b['t']),uniform_delta_t_pp=100*(uni['t']-b['t']),r_mean=row['reliability_mean'],r_std=row['reliability_std'],contrast_cosine=row['contrast_cosine']))
    out=dict(kind='Postseal numerical/provenance/process verification_NOTnewcandidate',code_sha256=sha(Path(__file__)),plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),
        report_sha256=sha(P/'report.json'),exact_old_baselines_queries=s['original_base_outputs_bitwise'],official_metric_checks=report['checks'],max_metric_error=report['max_metric_error'],
        matched_L2_max_absolute_error=normerr,degenerate_strength_controls=zero_controls,
        sensitivity_summary={ds:{key:float(np.median([x[key] for x in rows if x['dataset']==ds and x[key] is not None])) for key in ['r_mean','r_std','cosine','uniform_factor']} for ds in ['hc2','vidstg']},
        cases_rule='Top3 harms and top3 rescues by primary t delta vsG6 in each direction, afterseal; no GT-selected inference/gating or selective omission.',cases=cases,
        processes=processes,paper48_provider434_unchanged=True,all256_prediction_and_trace_bytes_verified=True,
        last_round_stop=True,new_confirmation_started=False,new_full_started=False,new_model=0,new_GT_read=0,new_predictions=0,seconds=time.perf_counter()-start)
    put(P/'VERIFICATION.json',out);put(D/'process_exit.json',dict(processes=processes,all_owned_exited=True))
    print(out['sensitivity_summary']);print('max_strength_error',normerr)

if __name__=='__main__':main()
