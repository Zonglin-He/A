"""Independent CPU-only scoring of the completely sealed local224 panel."""
import os,sys,csv,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.local224.run import OUT,PUBLIC,NAMES,validate,read,write,sha,inactive
from scripts.interval_pool.score import score_boxes,summarize
from scripts.joint_boundary_readout.score import paired

def evaluate():
    torch.set_num_threads(1);os.nice(15);p=validate();seal=read(OUT/'sealed.json');s=read(OUT/'state.json')
    assert inactive(s['process']) and seal['queries']==64 and seal['GT_read']is False and seal['fingerprint']==p['fingerprint']
    assert not(PUBLIC/'report.json').exists()
    for r in seal['files']:
        assert sha(OUT/r['prediction_path'])==r['prediction_sha256'] and sha(OUT/r['posterior_path'])==r['posterior_sha256']
    assert sha(ROOT/p['labels']['path'])==p['labels']['sha256']
    labels={r['id']:r for r in [json.loads(l)for l in (ROOT/p['labels']['path']).read_text().splitlines()if l.strip()]}
    rows=[];signed=[];error=0.;checks=0
    old=read(ROOT/'results/tube_fusion_hc64/report.json')
    for item,receipt in zip(p['items'],seal['files']):
        assert item['query']==receipt['query'];g=labels[item['id']];record=item['record'];gt={int(k):v for k,v in g['boxes'].items()};z=np.load(OUT/receipt['prediction_path']);values={};fixed={}
        for arm in NAMES:
            interval=z[arm+'_interval'].tolist()
            values[arm],e=score_boxes(z[arm],record,gt,interval,g['interval'],z['positions8'].tolist());error=max(error,e);checks+=1
            fixed[arm],e=score_boxes(z['FixedSpatial'],record,gt,interval,g['interval'],z['positions8'].tolist());error=max(error,e);checks+=1
        prev=next(r for r in old['queries_anonymous']if r['query']==item['query'])
        for arm,ref in [('Source','Source_Native224'),('Global6','Global6'),('Strong','Strong_Reference')]:
            for k in ['v','t','start_error_seconds','end_error_seconds']:assert abs(values[arm][k]-prev['values'][ref][k])<2e-10
        row=dict(query=item['query'],parent=g['parent'],values=values,fixed_spatial=fixed,crop_stats=receipt['crop_stats']);rows.append(row)
        for arm in NAMES:
            for base in ['Global6','GlobalPlusFull224','GlobalPlusShifted224','Strong']:
                if arm==base:continue
                a,b=values[arm],values[base];signed.append(dict(query=item['query'],method=arm,baseline=base,vIoU_delta_pp=100*(a['v']-b['v']),tIoU_delta_pp=100*(a['t']-b['t']),fixed_space_vIoU_delta_pp=100*(fixed[arm]['v']-fixed[base]['v']),start_error_delta_seconds=a['start_error_seconds']-b['start_error_seconds'],end_error_delta_seconds=a['end_error_seconds']-b['end_error_seconds'],rescued_half=b['t']<.5<=a['t'],harmed_half=a['t']<.5<=b['t'],severe_v_loss=a['v']-b['v'] < -.05,severe_t_loss=a['t']-b['t'] < -.05))
    assert len(rows)==len({r['parent']for r in rows})==64 and checks==768
    comparisons={a+'-'+b:paired(rows,a,b)for a,b in [('GlobalPlusLocal224','Global6'),('GlobalPlusLocal224','GlobalPlusFull224'),('GlobalPlusLocal224','GlobalPlusShifted224'),('GlobalPlusLocal224','Strong'),('GlobalPlusLocal224','Source'),('GlobalPlusFull224','Global6'),('GlobalPlusShifted224','Global6')]}
    subsets={}
    for name,rr in [('area_lt_half',[r for r in rows if r['crop_stats']['area_fraction']<.5]),('area_ge_half',[r for r in rows if r['crop_stats']['area_fraction']>=.5]),('near_full_area_ge90',[r for r in rows if r['crop_stats']['area_fraction']>=.9]),('Global6_wrong_half',[r for r in rows if r['values']['Global6']['t']<.5]),('Global6_correct_half',[r for r in rows if r['values']['Global6']['t']>=.5])]:
        subsets[name]=dict(queries=len(rr),comparison=paired(rr,'GlobalPlusLocal224','Global6')if rr else None)
    crop_summary={k:dict(mean=float(np.mean([r['crop_stats'][k]for r in rows])),median=float(np.median([r['crop_stats'][k]for r in rows])))for k in ['area_fraction','vertical_magnification','horizontal_magnification','window_iou','real_predicted_box_mean_iou','shifted_predicted_box_mean_iou']}
    branches={}
    for label in ['Local224','Shifted224']:
        rr=[r['branches'][label]for r in seal['files']]
        branches[label]={k:float(np.mean([r[k]for r in rr]))for k in ['crop_seconds','network_wall_seconds','posterior_seconds']}
        branches[label]['module_seconds']={k:float(np.mean([r['module_seconds'][k]for r in rr]))for k in rr[0]['module_seconds']}
        branches[label]['mean_input_height']=float(np.mean([r['input_shape'][-2]for r in rr]));branches[label]['mean_input_width']=float(np.mean([r['input_shape'][-1]for r in rr]))
        branches[label]['incremental_wall_including_shared_decode_parse']=float(np.mean([r['decode_seconds']+r['parse_seconds']+r['branches'][label]['crop_seconds']+r['branches'][label]['network_wall_seconds']+r['branches'][label]['posterior_seconds']for r in seal['files']]))
    result=dict(status='completed',exposure='HC64 repeatedly exposed development; fixed one configuration; no independent confirmation or production adoption.',queries=64,means=summarize(rows,NAMES,'values'),fixed_spatial_means=summarize(rows,NAMES,'fixed_spatial'),paired=comparisons,subsets=subsets,crop_summary=crop_summary,queries_anonymous=[{k:v for k,v in r.items()if k!='parent'}for r in rows],costs=dict(branches=branches,shared_decode_mean_seconds=float(np.mean([r['decode_seconds']for r in seal['files']])),model_load_seconds=seal['model_load_seconds'],parser_load_seconds=seal['parser_load_seconds'],engineering_whole224_calls=1,all128_local_views_total_seconds=seal['seconds'],scope='Actual128 local views with one prefix/two time passes each. Nested module timings must not be added. Fixed Global6/eight-expert/spatial cached costs excluded; deployment would also pay them. Model load is one-time; stages include host staging. Strong is a historical precision reference; no fresh matched448 timing, therefore no controlled speedup factor.'),proof=dict(all64_sealed_before_labels=True,seal_sha256=sha(OUT/'sealed.json'),plan_fingerprint=p['fingerprint'],official_dense_checks=checks,official_dense_max_error=error,engineering=read(OUT/'engineering.json'),new_local_views=128,new_whole224_engineering_views=1,new_experts=0,new448=0,updates=0,CUDA_initialized=torch.cuda.is_initialized()),decision=dict(production_changed=False,no_parameter_sweep=True,root_review_required=True))
    PUBLIC.mkdir(parents=True,exist_ok=True);write(PUBLIC/'report.json',result)
    with(PUBLIC/'signed_queries.tsv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(signed[0]),delimiter='\t');w.writeheader();w.writerows(signed)
    lines=['# Stable trajectory-crop local224: fixed HC64 trial','',result['exposure'],'','| Method | Full vIoU % | tIoU % | Unobserved sIoU % | Boundary MAE s |','|---|---:|---:|---:|---:|']
    for arm,m in result['means'].items():lines.append(f"| {arm} | {100*m['m_vIoU']:.4f} | {100*m['m_tIoU']:.4f} | {100*m['m_sIoU_unobserved_actual']:.4f} | {m['boundary_MAE_seconds']:.4f} |")
    lines+=['','| Comparison | tIoU pp [95% CI] | Full vIoU pp [95% CI] | Fixed-space vIoU pp | t rescue / harm |','|---|---:|---:|---:|---:|']
    for name,d in comparisons.items():
        def fmt(x):return f"{x['mean']:+.4f} [{x['CI95'][0]:.4f},{x['CI95'][1]:.4f}]"
        lines.append(f"| {name} | {fmt(d['t'])} | {fmt(d['v'])} | {d['fixed_space_v']['mean']:+.4f} | {d['recovered_tIoU_half']} / {d['lost_tIoU_half']} |")
    lines+=['','All signed changes, severe losses, boundary effects, stable-crop overlap and observed/unobserved space are in JSON/TSV. No per-query GT branch selection.','',result['costs']['scope']]
    (PUBLIC/'report.md').write_text('\n'.join(lines)+'\n');write(OUT/'score_receipt.json',dict(status='scored',report_sha256=sha(PUBLIC/'report.json'),signed_sha256=sha(PUBLIC/'signed_queries.tsv')))
    s['status']='scored_awaiting_root_analysis_and_publication';write(OUT/'state.json',s)
    print(json.dumps(dict(means=result['means'],paired=result['paired'],costs=result['costs']),indent=2))
if __name__=='__main__':evaluate()
