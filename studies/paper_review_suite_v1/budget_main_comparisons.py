"""All fixed confirmation arms versus the frozen Hybrid8 main, after scoring.

No models, new predictions or new GT scoring. These are descriptive paired
comparisons, not a parameter-selection rule.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
from full import RUN,read,sha,atomic
from full_score import cluster
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1/budget_v2'


def main():
    score=read(D/'budget_v2_confirm_score_receipt.json');seal=D/'budget_v2_confirm_sealed.json'
    assert score['checks']==2560 and score['seal_sha256']==sha(seal)and read(seal)['GT_read']is False
    selection=read(D/'selection.json');eligible=read(D/'budget_v2_qualification.json')['eligible']
    for ds,groups in selection.items():
        old={x['id']:x for x in read(RUN/ds/'metrics.json')['private_rows']}
        dest=P/ds/'confirm';signed=list(csv.DictReader((dest/'signed.tsv').open(),delimiter='\t'));report=read(dest/'report.json')
        lookup={(x['query'],x['configuration']):x for x in signed};allrows=[];result={}
        for name in eligible:
            conf=report['configs'][name];assert conf['queries']==128
            rows=[]
            for i,item in enumerate(groups['confirm'],1):
                value=lookup[(f'Q{i:03d}',name)];base=old[item['id']]
                vals={'MainHybrid8':dict(base['values']['Full']),name:dict(t=float(value['t']),v=float(value['v']))}
                rows.append(dict(parent=base['parent'],values=vals,duration_group=base['duration_group']))
                dt=100*(vals[name]['t']-vals['MainHybrid8']['t']);dv=100*(vals[name]['v']-vals['MainHybrid8']['v'])
                if name=='Hybrid8':assert dt==0 and dv==0
                allrows.append([f'Q{i:03d}',name,base['duration_group'],dt,dv])
            result[name]=dict(paired_MainHybrid8=cluster(rows,name,'MainHybrid8',fields=[('t',100),('v',100)]),cost_ratio_to_MainHybrid8=conf['mean_actual_GPU_forward_seconds']/report['configs']['Hybrid8']['mean_actual_GPU_forward_seconds'])
        atomic(dest/'paired_main.json',dict(comparisons=result,scope='Post-seal descriptive paired comparisons of all10 fixed candidates against original main Hybrid8; no parameter selection and no new scoring.'))
        with(dest/'paired_main_signed.tsv').open('w')as out:
            writer=csv.writer(out,delimiter='\t',lineterminator='\n');writer.writerow(['query','configuration','duration_group','delta_t_vs_main_pp','delta_v_vs_main_pp']);writer.writerows(allrows)
    atomic(D/'budget_v2_confirm_main_comparisons_receipt.json',dict(status='complete',code_sha256=sha(Path(__file__)),score_receipt_sha256=sha(D/'budget_v2_confirm_score_receipt.json'),new_predictions=0,new_GT_scoring=0,main_Hybrid8_scores_exact=True))


if __name__=='__main__':main()
