"""CPU-only sealed evaluation, with signed benefit/harm counts as primary outputs."""
import os,sys,time,json
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.joint_event_v2.run import R,D,P,ARMS,sha,read,put,verify,records
from research.boundary_feedback_v1.score import evaluate
import numpy as np,torch

def main():
    torch.set_num_threads(1);plan=verify();seal=read(D/'sealed.json');assert seal['count']==64 and not seal['GT_read']and not seal['CUDA_initialized']
    assert seal['plan_sha256']==sha(D/'plan.json')and not(D/'score_receipt.json').exists()
    for e in seal['files'].values():assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256']
    put(D/'pre_GT_validation.json',dict(count=64,seal_sha256=sha(D/'sealed.json'),all_bound_files_verified=True,CUDA_initialized=torch.cuda.is_initialized()))
    labels=R/'data/evaluation/development_labels.jsonl';assert sha(labels)==plan['labels_sha256']
    truth={g['id']:g for g in map(json.loads,labels.read_text().splitlines())if g['dataset']=='hc2'}
    prior={x['query']:x for x in read(R/'results/boundary_feedback_hc64_v1/report.json')['rows']}
    rows=[];checks=0;error=0.;arms=ARMS+['Source'];begin=time.perf_counter()
    for row in records():
        k=row['id'];e=seal['files'][k];gt=truth[k];vals={};fixed={}
        with np.load(D/e['path'],allow_pickle=False)as z:
            for arm in arms:
                vals[arm],v=evaluate(z[arm],row,gt,z[arm+'_interval'].tolist(),z['positions8']);checks+=1;error=max(error,v)
                fixed[arm],v=evaluate(z['NativeG6'],row,gt,z[arm+'_interval'].tolist(),z['positions8']);checks+=1;error=max(error,v)
            assert vals['NativeG6']==prior[k]['values']['NativeG6']and vals['Source']==prior[k]['values']['Source']
            # Signed final-interval movements only; no private coordinates exported.
            movements={a:(z[a+'_interval']-z['NativeG6_interval']).tolist()for a in ARMS}
        tr=torch.load(D/e['trace_path'],map_location='cpu',weights_only=False)
        dt=(gt['interval'][1]-gt['interval'][0])/row['metadata']['fps']
        parent=str(gt.get('parent',row['metadata']['source']))
        rows.append(dict(query=k,parent=parent,duration_group='short_le5s'if dt<=5 else 'long_gt5s',values=vals,fixed_space=fixed,
                         decision=tr['details'],text_metadata=tr['text_metadata'],movement=movements,
                         projection={name:entry['skills']for name,entry in tr['semantic'].items()}))
        put(D/'score_progress.json',dict(completed=len(rows),total=64))
    parent=sorted({x['parent']for x in rows});assert len(parent)==64
    mapping={x:f'p{i+1:03d}'for i,x in enumerate(parent)}
    put(D/'parent_mapping.json',mapping)
    for x in rows:x['parent']=mapping[x['parent']]
    ix=np.random.default_rng(20261022).integers(0,64,(10000,64))
    def means(field):return {a:{m:float(np.mean([r[field][a][m]for r in rows if r[field][a][m]is not None]))for m in rows[0][field][a]}for a in arms}
    comparisons=[(a,'NativeG6')for a in ARMS if a!='NativeG6']+[('ExcessGuard','Excess'),('ExcessGuard','ShiftExcessGuard'),('ExcessGuard','RawExcessGuard'),('ExcessGuard','Source')]
    paired={}
    for a,b in comparisons:
        result={}
        for metric in ['v','t','unobserved_s','boundary_MAE_seconds']:
            scale=1 if metric.endswith('seconds')else 100;delta=np.array([r['values'][a][metric]-r['values'][b][metric]for r in rows])*scale
            improve=int((delta>1e-10).sum());harm=int((delta< -1e-10).sum());changed=improve+harm
            result[metric]=dict(mean=float(delta.mean()),CI95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist(),improved=improve,harmed=harm,unchanged=64-changed,
                                harmed_fraction_all=harm/64,harmed_fraction_changed=harm/changed if changed else None,severe_below_minus5pp=int((delta< -5).sum())if scale==100 else None)
        paired[a+'-'+b]=result
    groups={g:{a:{m:float(np.mean([100*(r['values'][a][m]-r['values']['NativeG6'][m])for r in rows if r['duration_group']==g]))for m in ['v','t']}for a in ARMS if a!='NativeG6'}for g in ['short_le5s','long_gt5s']}
    mechanism={}
    for a in ['ExcessGuard','ShiftExcessGuard','RawExcessGuard']:
        mechanism[a]=dict(accepted=sum(bool(r['decision'][a]['accept'])for r in rows),predictive=sum(bool(r['decision'][a].get('predictive',False))for r in rows),agreement=sum(bool(r['decision'][a].get('agreement',False))for r in rows),unsupported=sum(r['decision'][a].get('reason')=='no_valid_semantic_contrast'for r in rows))
    result=dict(status='completed_scored_development',queries=64,parents=64,means=means('values'),fixed_space_means=means('fixed_space'),paired=paired,groups_vs_G6=groups,mechanism=mechanism,rows=rows,
        checks=checks,max_metric_error=error,all64_seal_before_GT=True,plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),
        mean_cached_feedback_seconds=float(np.mean([e['feedback_seconds']for e in seal['files'].values()])),
        CPU_text_and_projection_cost_included=True,end_to_end_cost_measured=False,full_status='paused_by_user',score_seconds=time.perf_counter()-begin)
    put(P/'report.json',result)
    header='query\tmethod\tvIoU\ttIoU\tunobserved_sIoU\tdelta_v_G6_pp\tdelta_t_G6_pp\tboundary_MAE_seconds';lines=[header];severe=[header]
    for r in rows:
        for a in arms:
            v=r['values'][a];b=r['values']['NativeG6'];s=f"{r['query']}\t{a}\t{v['v']:.12g}\t{v['t']:.12g}\t{v['unobserved_s']:.12g}\t{100*(v['v']-b['v']):.12g}\t{100*(v['t']-b['t']):.12g}\t{v['boundary_MAE_seconds']:.12g}"
            lines.append(s)
            if v['v']-b['v']<-.05 or v['t']-b['t']<-.05:severe.append(s)
    (P/'signed.tsv').write_text('\n'.join(lines)+'\n');(P/'severe_declines.tsv').write_text('\n'.join(severe)+'\n')
    put(D/'score_receipt.json',dict(report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),checks=checks,max_error=error,CUDA_initialized=torch.cuda.is_initialized()))
    put(D/'state.json',dict(status='completed_scored',fingerprint=plan['fingerprint']))
    print(json.dumps(dict(means=result['means'],paired=paired,groups=groups,mechanism=mechanism)))
if __name__=='__main__':main()
