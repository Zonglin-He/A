"""Offline P0 metrics after all four arms and independent readbacks seal.

No best-arm choice is fed back to inference or fitting. Ranking/coverage are
postseal diagnostics, separately labelled from final Student localization.
"""
from pathlib import Path
import sys,math,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_evidence_budget_p0_003 as study
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=study.BASE

def main():
    import torch,numpy as np
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    study.verify();pred=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert pred['status']=='complete'and pred['count']==16 and not pred['GT_read']
    for rel,h in pred['bindings'].items():assert sha(ROOT/rel)==h
    rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:study.load_case(r)[0]for r in rr};payloads={};bindings={}
    for arm in study.ARMS:
        b=BASE/arm;bar=read(b/'STUDENT_GLOBAL_BARRIER.json');root=read(b/'ROOT_READBACK.json')
        assert bar['status']=='complete'and root['status']=='pass'and bar['count']==root['count']==16
        assert root['student_barrier_sha256']==sha(b/'STUDENT_GLOBAL_BARRIER.json')
        payloads[arm]={}
        for f in bar['files']:
            assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
            assert not d['GT_read'] and d['source_case_sha256']==cases[d['case_id']]['source_case_sha256'];payloads[arm][d['case_id']]=d
        bindings[arm]=sha(b/'STUDENT_GLOBAL_BARRIER.json')
    write(BASE/'POSTSEAL_GT_EXPOSURE.json',dict(status='after_all_four_arms_and_independent_root',count=16,GT_read=True,prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json')))
    targets,gtbinding=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[];ranking=[]
    aliases={}
    for ds,prefix in [('hc2','PHC'),('vidstg','PVID')]:
        rs=[r for r in rr if r['dataset']==ds]
        aliases.update({r['probe_alias']:prefix+f'{i:03}'for i,r in enumerate(rs,1)})
    for rec in rr:
        cid=rec['probe_alias'];case=cases[cid];target=targets[cid];qtype='question'if('question'in rec['query_type']or'interrog'in rec['query_type'])else'statement'
        candidate_t=[]
        for cand in case['temporal_candidates']:
            scores,_=metric._score_student_arm(dict(boxes=case['native_boxes_xyxy_normalized'],interval=cand['interval_frames']),case,target,set(),official,independent)
            candidate_t.append(scores['m_tIoU'])
        for arm in ['Frozen','Projection',*study.ARMS]:
            interval=case['Native_interval']if arm=='Frozen'else case['Projection_interval']if arm=='Projection'else payloads[arm][cid]['interval']
            scores,_=metric._score_student_arm(dict(boxes=case['native_boxes_xyxy_normalized'],interval=interval),case,target,set(),official,independent)
            rows.append(dict(case_id=aliases[cid],dataset=rec['dataset'],query_type=qtype,arm=arm,metrics=scores,source_geometry_unchanged=True))
            if arm not in study.ARMS:continue
            rewards=[];coverage=[];costs=[]
            for cand in case['temporal_candidates']:
                p=BASE/'teacher'/arm/(cid+'_'+cand['canonical_id']+'.json');r=read(p);rewards.append(r['reward'])
                m=r['metadata'];fids=m.get('sampled_physical_frame_ids',[]);a,z=target['span']
                coverage.append(dict(unique_frames=len(set(fids)),requested_frames=m['requested_frames'],event_hit_frames=sum(a<=f<=z for f in fids),
                    candidate_id=cand['canonical_id'],unknown_input=r['status']!='actual_frozen_teacher'))
                if r['status']=='actual_frozen_teacher':costs.append(dict(caption_seconds=r['caption_seconds'],alignment_seconds=r['alignment_seconds'],GPU_peak_bytes=r['GPU_peak_bytes'],CPU_process_peak_bytes=r['CPU_process_peak_bytes'],
                    visual_tokens=r['qualification']['actual_processed_visual_tokens'],video_grid_thw=r['qualification']['video_grid_thw'],caption_input_tokens=r['caption_input_tokens'],caption_output_tokens=r['caption_output_tokens'],alignment_input_tokens=r['alignment_input_tokens'],truncated=r['output_truncated_at_cap']))
            comparable=[]
            for i in range(len(rewards)):
                for j in range(i):
                    if rewards[i]is None or rewards[j]is None or abs(candidate_t[i]-candidate_t[j])<1e-9:continue
                    d=rewards[i]-rewards[j];t=candidate_t[i]-candidate_t[j];comparable.append(1. if d*t>0 else .5 if d==0 else 0.)
            valid=[i for i,r in enumerate(rewards)if r is not None]
            selected=max(valid,key=lambda i:(rewards[i],-i))if valid else None
            ranking.append(dict(case_id=aliases[cid],dataset=rec['dataset'],query_type=qtype,arm=arm,rewards=rewards,candidate_tIoU=candidate_t,
                candidate_pool_best_tIoU=max(candidate_t)if candidate_t else None,teacher_selected_candidate_tIoU=candidate_t[selected]if selected is not None else None,
                ranking_pair_count=len(comparable),ranking_pair_accuracy=math.fsum(comparable)/len(comparable)if comparable else None,coverage=coverage,costs=costs))
    summary=[];paired=[]
    for ds in ('hc2','vidstg'):
        for group in ('all','question','statement'):
            sub=[r for r in rows if r['dataset']==ds and(group=='all'or r['query_type']==group)]
            if not sub:continue
            for arm in ['Frozen','Projection',*study.ARMS]:
                sr=[r for r in sub if r['arm']==arm]
                summary.append(dict(dataset=ds,query_group=group,arm=arm,count=len(sr),metrics={m:math.fsum(r['metrics'][m]for r in sr)/len(sr)for m in ('m_vIoU','m_tIoU','m_sIoU_dense_GT')}))
            ix={(r['case_id'],r['arm']):r for r in sub}
            for arm in ('B','C','D'):
                ids=[r['case_id']for r in sub if r['arm']==arm]
                for m in ('m_vIoU','m_tIoU','m_sIoU_dense_GT'):
                    d={cid:100*(ix[cid,arm]['metrics'][m]-ix[cid,'A']['metrics'][m])for cid in ids};v=list(d.values())
                    paired.append(dict(dataset=ds,query_group=group,contrast=arm+'_minus_A',metric=m,count=len(v),per_source_delta_pp=d,mean_delta_pp=math.fsum(v)/len(v),positive=sum(x>1e-6 for x in v),negative=sum(x< -1e-6 for x in v),neutral=sum(abs(x)<=1e-6 for x in v)))
    out=BASE/'complete';out.mkdir(exist_ok=False)
    write(out/'POSTSEAL_ROWS.json',dict(status='complete',count=16,row_count=len(rows),rows=rows,GT_read=True))
    write(out/'POSTSEAL_SUMMARY.json',dict(status='complete',summary=summary,paired=paired,GT_read=True))
    write(out/'POSTSEAL_RANKING.json',dict(status='complete',records=ranking,GT_read=True,GT_ranking_diagnostic_only=True))
    write(out/'CPU_COMPLETION.json',dict(status='complete',count=16,row_count=len(rows),GT_read=True,CPU_only=True,bootstrap_executed=False,CURRENT_promoted=False,
        bindings=bindings,prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),scorer_sha256=sha(Path(__file__)),
        output_sha256={n:sha(out/n)for n in ('POSTSEAL_ROWS.json','POSTSEAL_SUMMARY.json','POSTSEAL_RANKING.json')}))
    print(json.dumps({'status':'complete','count':16,'summary':[s for s in summary if s['query_group']=='all']},ensure_ascii=False))

if __name__=='__main__':main()
