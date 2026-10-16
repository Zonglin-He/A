"""Distinguish common observation clocks from actual gradient-anchor support."""
from pathlib import Path
import gc,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE as DEV,read,put,sha
def main():
    import torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    b=DEV if sys.argv[1]=='development'else DEV/'confirmation';n=32 if b==DEV else 64
    d=read(b/'EXPERIMENT_POSTSEAL_RESULTS.json');bar=read(b/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete'and bar['count']==n and d['prediction_barrier_sha256']==sha(b/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json')
    rr=read(b/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in rr};targets,gt=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();idx={(r['case_id'],r['arm']):r['metrics']for r in d['rows']};rows=[]
    files={t:{f['case_id']:f for f in read(b/'trials'/t/'STUDENT_GLOBAL_BARRIER.json')['files']}for t in ('E0','E1')}
    for rec in rr:
        cid=rec['probe_alias'];docs={}
        for t in ('E0','E1'):
            f=files[t][cid];assert sha(ROOT/f['path'])==f['sha256'];docs[t]=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
        for tid,mode,arm in [('E0','M2','E0'),('E1','M2','E1'),('E1','M0','E2')]:
            doc=docs[tid];fit=doc['fits'][mode];pos=set(fit['observation_positions'])if fit is not None else set();out={};audit={}
            for name,pred in [('Student',doc['arms']['TS_'+mode]),('Temporal',doc['arms']['T'])]:
                metrics,a=metric._score_student_arm(pred,cases[cid],targets[cid],pos,official,independent)
                for k in ('m_vIoU','m_tIoU','m_sIoU_dense_GT'):assert metrics[k]==idx[cid,arm if name=='Student'else'Temporal'][k]
                out[name]={k:metrics[k]for k in ('m_sIoU_supervised_GT','m_sIoU_unsupervised_GT')};audit[name]={k:a['support'][k]for k in ('supervised_anchor_count','supervised_gt_known_frame_count','unsupervised_gt_known_frame_count','off_grid_gt_known_frame_count')}
            delta={k:100*(out['Student'][k]-out['Temporal'][k])if out['Student'][k]is not None and out['Temporal'][k]is not None else None for k in out['Student']}
            rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=arm,actual_gradient_anchor_count=len(pos),metrics=out,delta_pp=delta,support_counts=audit,GT_read=True))
        del docs;gc.collect()
    summary=[]
    for ds in ('hc2','vidstg'):
        for arm in ('E0','E1','E2'):
            for k in ('m_sIoU_supervised_GT','m_sIoU_unsupervised_GT'):
                allr=[r for r in rows if r['dataset']==ds and r['arm']==arm];v=[r['delta_pp'][k]for r in allr if r['delta_pp'][k]is not None]
                summary.append(dict(dataset=ds,arm=arm,metric=k,mean_student_minus_Temporal_pp=math.fsum(v)/len(v)if v else None,valid=len(v),unknown=len(allr)-len(v),positive=sum(x>1e-6 for x in v),negative=sum(x< -1e-6 for x in v),neutral=sum(abs(x)<=1e-6 for x in v)))
    put(b/'ACTUAL_GRADIENT_SUPPORT_DIAGNOSIS.json',dict(status='complete',count=n,rows=rows,summary=summary,GT_read=True,scope='additional postseal diagnostic only; does not choose recipe or change predictions',common_metric_note='main five-arm metric names supervised/unsupervised partition the four pre-fixed observation positions, including unused/rejected ones; this supplement instead uses actual positive-weight gradient anchors per updater',complement='all other known GT frames, including off-Source-grid frames, as in the original official/independent scorer',source_result_sha256=sha(b/'EXPERIMENT_POSTSEAL_RESULTS.json'),GT_bindings=gt));print(json.dumps(dict(status='complete',count=n,summary=summary)))
if __name__=='__main__':main()
