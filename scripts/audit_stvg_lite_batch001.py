"""Independently compare every public scalar and paired case to saved results."""
import ast
import hashlib
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_lite_event_support_v1'
OUT=ROOT/'results/stvg_lite_event_support/2026-10-16/batch001'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    receipt=read(OUT/'BATCH_RECEIPT.json');assert receipt['whole_study_FINAL_complete'] is False
    tids=receipt['spatial_blocks']+receipt['temporal_blocks']
    raw=read(BASE/'trials/L000/POSTSEAL_ROWS.json')['rows'];aliases={}
    for ds,prefix in [('hc2','HC'),('vidstg','VID')]:
        ids=sorted({r['case_id'] for r in raw if r['dataset']==ds});assert len(ids)==64
        aliases.update({cid:f'{prefix}{i:03}' for i,cid in enumerate(ids,1)})
    public={};total=nulls=cells=paircells=0
    for tid in tids:
        p=BASE/'temporal/revision003' if tid=='TEMPORAL' else BASE/'trials'/tid
        if tid=='TEMPORAL':
            pre=read(p/'CPU_OUTPUT_BARRIER.json')
            assert pre['status']=='complete' and pre['count']==128 and pre['GT_read'] is False
        rr=read(p/'POSTSEAL_ROWS.json')['rows'];pp=read(OUT/'trials'/tid/'ROWS.json')['rows'];public[tid]=pp
        assert len(rr)==len(pp)==(768 if tid=='TEMPORAL' else 512)
        for r,v in zip(rr,pp):
            assert set(v)=={'case_id','dataset','arm','status','metrics','support_counts'}
            assert (v['case_id'],v['dataset'],v['arm'],v['status'])==(aliases[r['case_id']],r['dataset'],r['arm'],r['status'])
            assert v['metrics']==r['metrics']
            for value in v['metrics'].values():
                assert value is None or math.isfinite(value);nulls+=value is None
            total+=1
        originals={(s['dataset'],s['arm'],s['metric']):s for s in read(p/'POSTSEAL_SUMMARY.json')['summary']}
        for s in read(OUT/'trials'/tid/'SUMMARY.json')['summary']:
            old=originals[s['dataset'],s['arm'],s['metric']]
            assert s['n_valid']==old['n_valid'] and s['n_unknown']==old['n_unknown']
            assert s['mean'] is old['mean'] is None or abs(s['mean']-old['mean'])<1e-12
            if 'unknown_case_ids' in s:
                assert sorted(s['unknown_case_ids'])==sorted(aliases[c] for c in old['unknown_case_ids'])
            cells+=1
    def check_pair(pair,a,aa,b,bb):
        ds=pair['dataset'];m=pair['metric']
        x={r['case_id']:r['metrics'][m] for r in a if r['dataset']==ds and r['arm']==aa}
        y={r['case_id']:r['metrics'][m] for r in b if r['dataset']==ds and r['arm']==bb}
        delta={c:(x[c]-y[c])*100 for c in sorted(x) if x[c] is not None and y[c] is not None}
        assert delta==pair['per_source_delta_pp']
        assert len(delta)==pair['n_pairs']
        assert pair['positive_case_ids']==[c for c,v in delta.items() if v>1e-10]
        assert pair['negative_case_ids']==[c for c,v in delta.items() if v<-1e-10]
        assert pair['neutral_case_ids']==[c for c,v in delta.items() if abs(v)<=1e-10]
        assert pair['unknown_case_ids']==sorted(set(x)-set(delta))
        assert len(delta)+len(pair['unknown_case_ids'])==64
        if delta:assert abs(sum(delta.values())/len(delta)-pair['mean_delta_pp'])<1e-12
    for tid in tids:
        for pair in read(OUT/'trials'/tid/'PAIRED.json')['paired']:
            aa,bb=pair['comparison'].split(' minus ');check_pair(pair,public[tid],aa,public[tid],bb);paircells+=1
    for pair in read(OUT/'CROSS_TRIAL_PAIRED.json')['paired']:
        a,b=pair['comparison'].split(' minus ');check_pair(pair,public[a],'TS_M2',public[b],'TS_M2');paircells+=1
    sb=read(OUT/'SOURCE_BINDINGS.json')
    for rel,h in sb['measurement_sources'].items(): assert sha(ROOT/rel)==h
    for r in sb['source_snapshot']:
        assert sha(ROOT/r['path'])==r['original_sha256']
        p=OUT/'source_snapshot'/r['path'];assert sha(p)==r['public_sha256'];ast.parse(p.read_text())
    forbidden={'query_text','caption','raw_output','video_path','physical_frame_ids','supervised_positions',
               'box_xyxy','box_cxcywh','optimizer_state','GT_coordinates','weights','gradient'}
    def scan(x):
        if isinstance(x,dict):
            for k,v in x.items():assert k not in forbidden,k;scan(v)
        elif isinstance(x,list):
            for v in x:scan(v)
        elif isinstance(x,str):assert str(Path.home())+'/' not in x and '.codex/attachments/' not in x
    for p in OUT.rglob('*'):
        if p.is_file():
            assert p.suffix in {'.json','.py','.md','.png'}
            if p.suffix=='.json':scan(read(p))
            if p.suffix in {'.py','.md'}:
                text=p.read_text();assert '/home/' not in text and '.codex/attachments' not in text
                assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{30,}',text),p
    assert total==receipt['logical_score_rows']==4864
    proof=dict(status='pass',rows_checked=total,metric_values_checked=total*5,null_values_preserved=nulls,
               summary_cells_checked=cells,paired_cells_checked=paircells,source_snapshot_count=len(sb['source_snapshot']),
               all_signs_and_unknown_preserved=True,new_GT_read=False,new_inference=False,
               source_bindings_sha256=sha(OUT/'SOURCE_BINDINGS.json'),audit_source_sha256=sha(Path(__file__)))
    (OUT/'ROOT_PUBLIC_SCALAR_AUDIT.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))

if __name__=='__main__': main()
