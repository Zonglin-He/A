"""Postseal224 GT evaluation plus saved32/A/B matched scalar reuse."""
from pathlib import Path
import copy,gc,json,math,sys,time,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_coverage224_v1 import BASE,EXP,CORE,verify,read,sha,put,old
from scripts.stvg_overnight_score_v1 import aggregate,seal
def main():
    verify()
    from scripts import stvg_overnight_score_v1 as scorer
    # Original postseal scoring path, variable cohort224; same metrics/support.
    scorer.c=types.SimpleNamespace(BASE=BASE,verify=verify,score=old.score);scorer.score('expansion224')
    first=read(old.BASE/'ablation_noCoverage/POSTSEAL_RESULTS.json');new=read(EXP/'POSTSEAL_RESULTS.json');A=old.score('A');B=old.score('B');ai={(r['case_id'],r['arm']):r for r in A['rows']};bi={(r['case_id'],r['arm']):r for r in B['rows']};allr=read(CORE/'ROSTER_LOCK.json')['records'];parts=[]
    for name,source,n in [('initial32',first,32),('additional224',new,224),('complete256',dict(rows=first['rows']+new['rows']),256)]:
        ids={r['case_id'] for r in source['rows']};assert len(ids)==n;idx={(r['case_id'],r['arm']):r for r in source['rows']};rows=[]
        for cid in [r['probe_alias'] for r in allr if r['probe_alias'] in ids]:
            for control in ('Frozen','Temporal'):assert idx[cid,control]['metrics']==ai[cid,control]['metrics']==bi[cid,control]['metrics']
            assert idx[cid,'LocalB']['metrics']==bi[cid,'B']['metrics']
            for arm,row in [('Frozen',idx[cid,'Frozen']),('Temporal',idx[cid,'Temporal']),('A',ai[cid,'A']),('B',bi[cid,'B']),('B_noCoverage',idx[cid,'Candidate'])]:rows.append(dict(row,arm=arm))
        s,p=aggregate(rows,n);parts.append(dict(partition=name,count=n,per_domain=n//2,rows=rows,summary=s,paired=p))
    put(BASE/'MATCHED_RESULTS256.json',dict(status='complete',count=256,primary='B_noCoverage_minus_A',secondary='B_noCoverage_minus_B',records=parts,all_A_B_Frozen_Temporal_saved_scores_reused=True,additional_GT_sources_only=224,initial32_preserved=True,development_only=True,semantic_status='separate original fixed512-window review, pending actual labels',no_automatic_confirmation_or_method_selection=True,GT_read=True))
    put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=256,GT_read=True,additional_new_predictions=224,original_reused_predictions=32,result_sha256=sha(BASE/'MATCHED_RESULTS256.json')));print('ALL256_THREE_PARTITIONS_POSTSEAL_COMPLETE')
if __name__=='__main__':main()
