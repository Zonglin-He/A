"""Independent CPU algebra readback from saved shared-stem vectors and deltas."""
from pathlib import Path
import hashlib, json, math, statistics
import torch
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/desta3d_v2/aux_backflow_v1'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):
    assert math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-10), (a,b)


def main():
    seal = read(BASE/'panel/COMPLETE.json')
    for p,h in seal['windows_sha'].items(): assert sha(Path(p))==h
    assert sha(BASE/'PANEL_START.pt')==seal['snapshot_sha']
    start=read(BASE/'PANEL_START.json')
    assert sha(ROOT/'artifacts/desta3d_v2/source_fit/LATEST.pt')==start['original_checkpoint_sha']
    rows=[];queries=[];all_groups=[]
    for wi in range(4):
        report=read(BASE/'panel'/f'WINDOW_{wi}.json')
        assert report['restored_exactly']
        rawpath=Path(report['raw_stem_path']);assert sha(rawpath)==report['raw_stem_sha']
        raw=torch.load(rawpath,map_location='cpu',weights_only=False)
        v={k:x.double() for k,x in raw['vectors'].items()}
        assert raw['numel']==19968 and all(x.numel()==19968 and torch.isfinite(x).all() for x in v.values())
        assert torch.allclose(v['task'],v['event']+v['spatial'],rtol=3e-5,atol=1e-10)
        tn,an=float(v['task'].norm()),float(v['aux'].norm())
        ratio=an/tn if tn else None
        cosine=float(v['task'].dot(v['aux']))/(tn*an) if tn*an else None
        close(tn,report['gradients']['shared_stem']['norms']['task'])
        close(an,report['gradients']['shared_stem']['norms']['aux'])
        item={'window':wi,'task_norm':tn,'aux_norm':an,'ratio':ratio,'cosine':cosine,
              'raw_B0_task_factor':1+ratio*cosine if ratio is not None and cosine is not None else None,'modes':{}}
        for mode in ('B0','B1','B2'):
            c=1. if mode=='B0' else 0. if mode=='B1' else (min(1.,.25*tn/(an+1e-12)) if tn else 0.)
            logged=report['updates'][mode]
            close(c,logged['combination']['aux_reader_backflow_coefficient'])
            assert logged['combination']['aux_readout_head_coefficient']==1
            delta=v['actual_delta_'+mode];dn=float(delta.norm());dot=float(v['task'].dot(delta))
            close(dn,logged['actual_AdamW_delta']['shared_stem']['delta_norm'])
            close(dot,logged['actual_AdamW_delta']['shared_stem']['task_dot_delta'])
            item['modes'][mode]={'c':c,'stem_actual_delta_norm':dn,'stem_task_dot_delta':dot,
                  'stem_descent_first_order_proxy':dot<0,
                  'all_params_task_dot_delta_worker_only':logged['actual_AdamW_delta']['all']['task_dot_delta']}
        rows.append(item)
        for query in report['queries']:
            g=query['gradient_groups']['shared_stem']
            queries.append({'key':query['key'],'source':query['source'],
                'ratio':g['weighted_aux_task_ratio'],'cosine':g['cosines']['task__aux'],
                'raw_B0_task_factor':g['raw_SGD_task_factor']})
    assert len(queries)==len({q['source'] for q in queries})==16
    eq=read(BASE/'panel/REAL_B0_EQUIVALENCE.json')
    assert eq['passed'] and all(x['passed'] for x in eq['checks'].values())
    result={'raw_stem_independent_recompute_pass':True,'queries':16,'parents':16,'windows':4,
            'diagnostic_optimizer_steps':12,'persisted_updates':0,'old_checkpoint_unchanged':True,
            'snapshot_epoch':start['epoch'],'snapshot_cursor':start['cursor'],'B_steps_before_panel':start['stage_steps'],
            'query_ratio_min_median_max':[min(q['ratio'] for q in queries),statistics.median(q['ratio'] for q in queries),max(q['ratio'] for q in queries)],
            'query_raw_B0_ascent_count':sum(q['raw_B0_task_factor']<0 for q in queries),
            'actual_stem_descent_windows':{a:sum(r['modes'][a]['stem_descent_first_order_proxy'] for r in rows) for a in ('B0','B1','B2')},
            'B0_first_query_equivalence_max_group_relative_l2':max(c['relative_l2'] for c in eq['checks'].values()),
            'windows':rows,'per_query':queries,'target_GT_read':False,'source_validation_GT_read':False,
            'interpretation':'source-only fixed panel, raw shared-stem vectors independently recalculated; full-group deltas worker summaries only; task dot delta is a first-order proxy, not observed finite-step loss or vIoU',
            'passed_engineering_gate_for_fixed_B_contrast':True}
    # Keep count distinct from the row list for machine readers.
    result['window_count']=4
    path=BASE/'panel/ROOT_READBACK.json'
    assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n')
    text=['# Fixed source auxiliary-gradient / AdamW panel','',
          f"Same original checkpoint E{start['epoch']} cursor {start['cursor']}, {start['stage_steps']} B updates.",
          '16 source training parents, four accumulation windows, 12 discarded optimizer updates. No validation/target GT.',
          f"Per-query weighted aux/task norm min/median/max: {result['query_ratio_min_median_max']}.",
          f"Raw B0 gradient ascent factor: {result['query_raw_B0_ascent_count']}/16 queries.",
          '', '| Window | B0 task dot delta | B1 task dot delta | B2 task dot delta | B2 c |', '|---|---:|---:|---:|---:|']
    for r in rows:
        text.append('| '+str(r['window'])+' | '+' | '.join(f"{r['modes'][a]['stem_task_dot_delta']:.8g}" for a in ('B0','B1','B2'))+f" | {r['modes']['B2']['c']:.8g} |")
    text+=['','Negative dot product denotes a local first-order descent proxy on the shared stem only. Existing Adam moments, clipping and other parameter groups matter. This table does not measure post-step task loss or full-tube utility. All updates were discarded and the original source checkpoint remained byte-identical.','',
           'The fixed B0/B1/B2 training comparison is still required regardless of which sign wins this local panel.']
    (BASE/'panel/REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('windows','per_query')},indent=2))


if __name__=='__main__':main()
