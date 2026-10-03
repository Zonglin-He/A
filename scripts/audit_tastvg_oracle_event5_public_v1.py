"""Independent anonymous result audit: no GT, raw tensors, models or GPU access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,collections
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_text())
def run(directory):
    p=Path(directory);checks=collections.Counter();dslist=['vidstg','hc2'];splits=['search','confirm'];total=donors=0
    def close(a,b):
        assert np.isfinite(a).all() and np.isfinite(b).all();np.testing.assert_allclose(a,b,atol=1e-10,rtol=0);checks['scalars']+=np.asarray(a).size
    def summary(rows,z):
        keys=list(z['metrics']);ss=sorted({r['source_id'] for r in rows});order=sorted({r['order'] for r in rows})
        assert z['cells']==len(rows) and z['sources']==len(ss)
        if not rows:return
        matrix=[]
        for s in ss:
            byorder=[]
            for o in order:
                rr=[r for r in rows if r['source_id']==s and r['order']==o]
                if rr:
                    conds=sorted({r['condition'] for r in rr});byorder.append(np.mean([np.mean([[r[k] for k in keys] for r in rr if r['condition']==c],axis=0) for c in conds],axis=0))
            matrix.append(np.mean(byorder,axis=0))
        matrix=np.array(matrix);rng=np.random.default_rng(20261003)
        bootstrap=np.concatenate([matrix[rng.integers(0,len(ss),(100,len(ss)))].mean(1) for _ in range(100)])
        ci=np.percentile(bootstrap,[2.5,97.5],axis=0)
        for j,k in enumerate(keys):
            m=z['metrics'][k];a=matrix[:,j];loo=(a.sum()-a)/(len(a)-1) if len(a)>1 else a
            close(a.mean(),m['mean']);close(ci[:,j],m['ci95']);close([loo.min(),loo.max()],m['leave_one_out_range'])
            close([a[i] for i in range(len(a))],[m['source_values'][str(s)] for s in ss]);close(np.mean([r[k] for r in rows]),m['cell_macro'])
    for ds in dslist:
        for split in splits:
            rows=read(p/'experiment1'/split/ds/'ORACLE_ROWS.json');z=read(p/'experiment1'/split/ds/'SUMMARY.json')
            assert len(rows)==(384 if split=='search' else 192)
            assert len({r['source_id'] for r in rows})==(32 if split=='search' else 16)
            for r in rows:
                close(r['Joint_GT_v'],1);close(r['GT_space_v'],r['A_t'])
                close(r['H_temporal'],r['GT_time_v']-r['A_v']);close(r['H_spatial'],r['GT_space_v']-r['A_v'])
                assert r['H_temporal']>=-1e-12 and r['H_spatial']>=-1e-12
                if r['expert_scheduled']:
                    matrix=np.array(r['joint_candidate_v']);sv=np.array(r['spatial_candidate_v']);tv=np.array(r['temporal_candidate_v'])
                    assert matrix.shape==(8,9) and len(sv)==9 and len(tv)==8
                    close(matrix[:,0],tv);close(sv[0],r['A_v']);close(tv.max(),r['temporal_oracle_v']);close(sv.max(),r['spatial_oracle_v']);close(matrix.max(),r['joint_oracle_v'])
                    close(matrix[r['T_index'],0],r['T_v']);close(r['H_temporal'],r['H_selection']+r['H_coverage'])
                    close(r['H_spatial'],r['H_spatial_selection']+r['H_spatial_coverage'])
                    close(r['T_remaining'],r['temporal_oracle_v']-r['T_v']);close(r['T_recovered'],r['T_v']-r['A_v'])
                    close(r['joint_minus_best_single'],matrix.max()-max(tv.max(),sv.max()))
                    assert matrix.max()>=max(tv.max(),sv.max())-1e-12
                    checks['candidate_matrices']+=1
                total+=1
            for group in ['corruption','clean']:
                for sub in ['all','expert','nonexpert']:
                    rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
                    summary(rr,z[group][sub])
            rows=read(p/'experiment2'/split/ds/'INTERVENTION_ROWS.json');z=read(p/'experiment2'/split/ds/'SUMMARY.json')
            assert len(rows)==(96 if split=='search' else 48)
            for r in rows:
                close(r['A_GT'],r['probe_GT_v'][0]);close(r['A_actual'],r['probe_A_v'][0]);close(r['spatial_oracle_GT'],max(r['probe_GT_v']))
                for b in ['U','U2','R','GT_event']:
                    k=r['evidence'].get(b,{}).get('selected',0)
                    for interval,base in [('GT','A_GT'),('A','A_actual')]:
                        close(r[f'{b}_select_{interval}'],r[f'probe_{interval}_v'][k])
                        for kind in ['select','temp']:close(r[f'delta_{b}_{kind}_{interval}'],r[f'{b}_{kind}_{interval}']-r[base])
                    if b in r['evidence']:
                        e=r['evidence'][b];assert e['observed_frames']==5 and e['empty_observed_frames']==5-e['valid_frames']
                        assert 0<=e['event_scorable_valid_frames']<=e['event_valid_frames']<=e['valid_frames']<=5
                        if b=='GT_event':assert e['event_observed_frames']==5
                if not r['eligible']:close(r['GT_event_temp_GT'],r['A_GT']);close(r['GT_event_select_GT'],r['A_GT'])
                for b in ['R','U2']:
                    for kind in ['select','temp']:
                        for interval in ['GT','A']:close(r[f'GT_minus_{b}_{kind}_{interval}'],r[f'GT_event_{kind}_{interval}']-r[f'{b}_{kind}_{interval}'])
                donors+=1
            for group in ['corruption','clean']:
                for sample in ['eligible_matched','all_scheduled_with_noop_unsupported']:
                    rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and (sample!='eligible_matched' or r['eligible'])];m=z[group][sample];summary(rr,m)
                    for b in ['U','U2','R','GT_event']:
                        ee=[r['evidence'][b] for r in rr if b in r['evidence']];q=m['quality'][b]
                        assert q['requests']==len(ee) and q['empty_requests']==sum(e['valid_frames']==0 for e in ee)
                        if q['event_scorable_valid_frames']:
                            close(q['event_box_GT_IoU_valid_weighted'],sum(e['event_iou_sum'] for e in ee)/sum(e['event_scorable_valid_frames'] for e in ee))
                        for interval in ['GT','A']:
                            e=m['execution'][b][interval];sel=f'delta_{b}_select_{interval}';tmp=f'delta_{b}_temp_{interval}'
                            assert e['selected_better_but_update_harm']==sum(r[sel]>1e-12 and r[tmp]<-1e-12 for r in rr)
                            assert e['serious_harm_gt5pp']==sum(r[tmp]<-.05 for r in rr)
    notesfile=p/'PAIRED_DIAGNOSTIC_NOTES.json'
    if notesfile.exists():
        notes=read(notesfile)
        for ds in dslist:
            for split in splits:
                n=notes[f'{split}/{ds}'];m=read(p/'experiment1'/split/ds/'SUMMARY.json')['corruption']['expert']['metrics']
                close(n['temporal_coverage_fraction_of_headroom_ratio_of_macro_means'],m['H_coverage']['mean']/m['H_temporal']['mean'])
                rows=read(p/'experiment2'/split/ds/'INTERVENTION_ROWS.json')
                rr=[dict(r,GT_time_spatial_candidate_headroom=r['spatial_oracle_GT']-r['A_GT']) for r in rows if r['eligible'] and r['condition']!='clean']
                summary(rr,n['spatial_candidate_increment_at_GT_time'])
    assert total==1152 and donors==288 and checks['candidate_matrices']==288
    result=dict(status='pass',arrivals=total,donors=donors,checks=dict(checks),anonymous_only=True,GT_read=False,GPU_initialized=False)
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':run(sys.argv[1])
