"""Independent numeric/anonymous-field check of the explicit measurement export."""
import ast
import hashlib
import json
import math
from pathlib import Path

R=Path(__file__).resolve().parents[1]
O=R/'results/stvg_coordinate_component128/2026-10-16'
S=R/'artifacts/stvg_soft_ep_coordinate128_v1'
C=R/'artifacts/stvg_component_ablation128_v1'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report=read(O/'REPORT_RECEIPT.json'); assert report['logical_score_rows']==52864
    raw=read(S/'trials/D000/POSTSEAL_SCORE_ROWS.json')['rows']
    aliases={}
    for ds,prefix in [('hc2','HC'),('vidstg','VID')]:
        ids=sorted({r['case_id'] for r in raw if r['dataset']==ds});assert len(ids)==64
        aliases.update({c:f'{prefix}{i:03}' for i,c in enumerate(ids,1)})
    def rename(x):
        if isinstance(x,dict):return {k:rename(v) for k,v in x.items()}
        if isinstance(x,list):return [rename(v) for v in x]
        return aliases.get(x,x) if isinstance(x,str) else x
    rows_checked=0;scalars=0;nulls=0;summary_checked=0;paired_checked=0
    for trial in report['trials']:
        tid=trial['trial_id'];public=O/'trials'/tid
        if tid.startswith(('D','E','F')):
            base=S/'trials'/tid;rowfile='POSTSEAL_SCORE_ROWS.json';summaryfile='POSTSEAL_SCORE_SUMMARY.json';pairedfile='POSTSEAL_PAIRED_CONTRASTS.json'
        elif tid=='TEMPORAL':
            base=C/'temporal_component/revision005';rowfile='TEMPORAL_COMPONENT_SCORE_ROWS.json';summaryfile='TEMPORAL_COMPONENT_SCORE_SUMMARY.json';pairedfile='TEMPORAL_COMPONENT_SCORE_PAIRED.json'
        else:
            base=C/('native_clock/revision002/trials/N000' if tid=='N000' else 'trials/'+tid);rowfile='POSTSEAL_ROWS.json';summaryfile='POSTSEAL_SUMMARY.json';pairedfile='POSTSEAL_PAIRED.json'
        rr=read(base/rowfile)['rows'];pp=read(public/'ROWS.json')['rows'];assert len(rr)==len(pp)==trial['row_count']
        for raw,pub in zip(rr,pp):
            assert pub['case_id']==aliases[raw['case_id']] and pub['dataset']==raw['dataset'] and pub['arm']==raw['arm']
            assert pub['metrics']==raw['metrics']
            assert set(pub)=={'case_id','dataset','arm','status','metrics','support_counts'}
            for value in raw['metrics'].values():
                scalars+=1;nulls+=value is None
                assert value is None or math.isfinite(value)
            rows_checked+=1
        summary=read(public/'SUMMARY.json')['summary'];assert summary==rename(read(base/summaryfile)['summary'])
        for cell in summary:
            metric=cell['metric'];group=[r for r in pp if r['dataset']==cell['dataset'] and r['arm']==cell['arm']]
            if metric in group[0]['metrics']:
                values=[r['metrics'][metric] for r in group if r['metrics'][metric] is not None]
                assert len(values)==cell['n_valid'] and len(group)-len(values)==cell['n_unknown']
                if values: assert abs(sum(values)/len(values)-cell['mean'])<1e-12
            summary_checked+=1
        pairs=read(public/'PAIRED.json')['paired_contrasts'];assert pairs==rename(read(base/pairedfile)['paired_contrasts']);paired_checked+=len(pairs)
    source=read(O/'SOURCE_BINDINGS.json')
    for rel,expected in source['measurement_sources'].items():assert sha(R/rel)==expected
    for record in source['source_snapshot']:
        assert sha(R/record['path'])==record['original_sha256']
        p=O/'source_snapshot'/record['path'];assert sha(p)==record['public_snapshot_sha256'];ast.parse(p.read_text())
    forbidden=['query_text','caption','raw_output','video_path','physical_frame_ids','supervised_positions','supervised_physical_frame_ids','box_xyxy','box_cxcywh','optimizer_state','GT_coordinates','weights']
    def scan(d):
        if isinstance(d,dict):
            for k,v in d.items():assert k not in forbidden,k;scan(v)
        elif isinstance(d,list):
            for v in d:scan(v)
        elif isinstance(d,str):assert '/home/wwww/' not in d and '.codex/attachments/' not in d
    for p in O.rglob('*'):
        if p.is_file():
            assert p.suffix in {'.json','.py','.png'},p
            if p.suffix=='.json':scan(read(p))
            elif p.suffix=='.py':assert (str(Path.home()/'.codex/attachments')+'/').encode() not in p.read_bytes()
    contrasts=read(O/'CROSS_TRIAL_PAIRED.json')['paired']
    for cell in contrasts:
        values=[r['delta_pp'] for r in cell['rows']]
        assert len(values)==cell['n_pairs']
        if values:assert abs(sum(values)/len(values)-cell['mean_delta_pp'])<1e-12
        assert sum(x>1e-10 for x in values)==cell['positive']
        assert sum(x<-1e-10 for x in values)==cell['negative']
        assert sum(abs(x)<=1e-10 for x in values)==cell['neutral']
    proof=dict(status='pass',scope='all exported measurement rows/metrics/nulls/paired sets and explicit source snapshots; no raw GT or model access',rows_checked=rows_checked,metric_scalars_checked=scalars,null_metric_values_preserved=nulls,summary_cells_checked=summary_checked,original_paired_cells_checked=paired_checked,cross_trial_paired_cells_checked=len(contrasts),source_snapshot_count=len(source['source_snapshot']),source_bindings_sha256=sha(O/'SOURCE_BINDINGS.json'),reader_sha256=sha(Path(__file__)),new_GT_read=False,new_model_inference=False,bootstrap_executed=False)
    assert rows_checked==52864
    (O/'ROOT_PUBLIC_SCALAR_AUDIT.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))


if __name__=='__main__':main()
