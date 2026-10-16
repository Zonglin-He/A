"""Independent final arithmetic, anonymous-export and publication receipt audit.

Saved data only; no GPU, model loading, labels or new task scoring.
"""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,read,put,sha
def check_scores(b,folder,prefix,n):
    d=read(b/('POSTSEAL_RESULTS.json'if folder=='temporal001'else'EXPERIMENT_POSTSEAL_RESULTS.json'));public=read(ROOT/'results/stvg_instance_evidence32/2026-10-16'/folder/'RESULTS.json');rows=d['rows'];aliases={c:f'{prefix}{i:03}'for i,c in enumerate(sorted({r['case_id']for r in rows}),1)}
    public_idx={(r['case_id'],r['arm']):r['metrics']for r in public['rows']};assert len(public_idx)==len(rows)
    cells=nulls=0
    for r in rows:
        assert public_idx[aliases[r['case_id']],r['arm']]==r['metrics'];cells+=len(r['metrics']);nulls+=sum(v is None for v in r['metrics'].values())
    idx={(r['case_id'],r['arm']):r for r in rows}
    for s in d['summary']:
        vv=[r['metrics'][s['metric']]for r in rows if r['dataset']==s['dataset']and r['arm']==s['arm']];vals=[v for v in vv if v is not None];mean=math.fsum(vals)/len(vals)if vals else None
        assert len(vv)==n//2 and len(vals)==s['valid']and len(vv)-len(vals)==s['unknown'];assert mean==s['mean']
    for p in d['paired']:
        a,c=p['comparison'].split('_minus_');ids=[r['case_id']for r in rows if r['dataset']==p['dataset']and r['arm']==a];vv={i:100*(idx[i,a]['metrics'][p['metric']]-idx[i,c]['metrics'][p['metric']])for i in ids if idx[i,a]['metrics'][p['metric']]is not None and idx[i,c]['metrics'][p['metric']]is not None}
        assert vv==p['per_source_delta_pp'];vals=list(vv.values());assert (math.fsum(vals)/len(vals)if vals else None)==p['mean_delta_pp'];assert sum(v>1e-6 for v in vals)==p['positive']and sum(v< -1e-6 for v in vals)==p['negative']and sum(abs(v)<=1e-6 for v in vals)==p['neutral']
    return dict(group=folder,count=n,row_count=len(rows),metric_cells=cells,null_cells=nulls,all_source_metrics_and_summaries_pairs_exact=True,source_sha256=sha(b/('POSTSEAL_RESULTS.json'if folder=='temporal001'else'EXPERIMENT_POSTSEAL_RESULTS.json')))
def main():
    rows=[check_scores(BASE/'temporal','temporal001','T',32),check_scores(BASE,'experiment001','E',32),check_scores(BASE/'confirmation','confirmation001','C',64)]
    groups=('pool001','temporal001','experiment001','support001','interpretation001','confirmation001');pub=[]
    for g in groups:
        b=BASE/'publication'/g;m=read(b/'EXPORT_MANIFEST.json');r=read(b/'FINAL_GITHUB_RECEIPT.json');http=read(b/'REMOTE_BYTES_READBACK.json');assert r['status']==http['status']=='pass'and http['actual_remote_HTTP_bytes_SHA256_Gitblob_checked']
        assert r['commit_sha']==http['commit_sha']and r['file_count']==len(m['files'])==http['file_count']and r['bytes']==sum(f['bytes']for f in m['files'])==http['bytes']
        for f in m['files']:assert sha(ROOT/f['path'])==f['sha256']and(ROOT/f['path']).stat().st_size==f['bytes']
        pub.append(dict(group=g,commit_sha=r['commit_sha'],tree_sha=r['tree_sha'],files=r['file_count'],bytes=r['bytes'],remote_receipt_sha256=sha(b/'REMOTE_BYTES_READBACK.json')))
    rosters=[read(BASE/'ROSTER_LOCK.json')['records'],read(BASE/'confirmation/ROSTER_LOCK.json')['records']];parents=[r['parent_sha256']for rr in rosters for r in rr];assert len(parents)==len(set(parents))==96
    put(BASE/'ROOT_STUDY_SAVED_RESULTS_READBACK.json',dict(status='pass',count=96,CPU_only=True,GT_read=False,saved_postseal_metric_values_used=True,groups=rows,publications=pub,primary_logical_rows=sum(r['row_count']for r in rows),primary_metric_cells=sum(r['metric_cells']for r in rows),primary_null_cells=sum(r['null_cells']for r in rows),no_new_inference_or_GT_scoring=True,all_unique_parent_counts_exact=True,scope='saved arithmetic/anonymous exports/actual existing HTTP receipts, not a new full backward or repeated remote network fetch'))
    print(json.dumps(dict(status='pass',groups=rows,publications=pub)))
if __name__=='__main__':main()
