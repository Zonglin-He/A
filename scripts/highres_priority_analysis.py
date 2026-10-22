"""Sealed-output diagnosis: resolution/layers/offsets, rescues and latent change.

No new inference or parameter search; all label use follows the complete seal.
"""
import os,sys,json,csv
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.highres_priority.run import SUP,OUT,PUBLIC,validate,read,write,sha,inactive
from scripts.joint_boundary_readout.score import paired
from scripts.interval_pool.score import summarize

def analyze():
    torch.set_num_threads(1);p=validate();seal=read(SUP/'sealed.json');state=read(SUP/'state.json');assert seal['queries']==63 and not seal['GT_read'] and inactive(state['process'])
    report=read(PUBLIC/'report.json');assert sha(PUBLIC/'report.json')==read(SUP/'score_receipt.json')['report_sha256']
    local=read(ROOT/'results/local224_v1/report.json');localrows={r['query']:r for r in local['queries_anonymous']};common=report['queries_anonymous'];rows=[]
    for r in common:
        if r['query']=='H001':continue
        lr=localrows[r['query']]
        for m,old in [('Only224six','Global6'),('Fusion','Strong')]:
            for field in ['v','t','start_error_seconds','end_error_seconds']:assert abs(r['values'][m][field]-lr['values'][old][field])<2e-10
        v=dict(r['values'],Local224=lr['values']['GlobalPlusLocal224'],Full224SingleFusion=lr['values']['GlobalPlusFull224']);fixed=dict(r['fixed_spatial'],Local224=lr['fixed_spatial']['GlobalPlusLocal224'],Full224SingleFusion=lr['fixed_spatial']['GlobalPlusFull224'])
        rows.append(dict(query=r['query'],values=v,fixed_spatial=fixed))
    assert len(rows)==63
    names=list(rows[0]['values']);pairs=[('Only448','Only224six'),('Fusion','Only448'),('High6','Only448'),('High6Fusion','Fusion'),('Only224six','Only224Last'),('High6','Only224six'),('SingleFusion0','Fusion'),('SingleFusion1','Fusion'),('SingleHigh0','Only448'),('SingleHigh1','Only448'),('Local224','Fusion'),('Local224','SingleFusion0')]
    comp={a+'-'+b:paired(rows,a,b)for a,b in pairs}
    # The same two-layer choices at each resolution. Matched63 only.
    layer_panel=summarize(rows,['Only224Last','Only224six','Only448','High6'],'values')
    threshold_patterns={}
    conditions=dict(high_rescues=lambda g,h,f:g<.5<=h,fusion_only_rescues=lambda g,h,f:g<.5 and h<.5<=f,high_harms=lambda g,h,f:h<.5<=g,fusion_harms=lambda g,h,f:f<.5<=g,high_good_lost_in_fusion=lambda g,h,f:f<.5<=h)
    for tag,condition in conditions.items():threshold_patterns[tag]=[r['query']for r in common if condition(*[r['values'][m]['t']for m in ['Only224six','Only448','Fusion']])]
    cases=[]
    for r in common:
        g,h,f=[r['values'][m]for m in ['Only224six','Only448','Fusion']]
        if not((g['t']<.5<=f['t'])or(f['t']<.5<=g['t'])or(g['t']<.5<=h['t'])or(h['t']<.5<=g['t'])):continue
        cases.append(dict(query=r['query'],type='fusion_rescue'if g['t']<.5<=f['t']else'fusion_harm'if f['t']<.5<=g['t']else'high_rescue'if g['t']<.5<=h['t']else'high_harm',t_global=100*g['t'],t_high=100*h['t'],t_fusion=100*f['t'],t_local=100*localrows[r['query']]['values']['GlobalPlusLocal224']['t'],high_start_error_change=h['start_error_seconds']-g['start_error_seconds'],high_end_error_change=h['end_error_seconds']-g['end_error_seconds'],fusion_start_error_change=f['start_error_seconds']-g['start_error_seconds'],fusion_end_error_change=f['end_error_seconds']-g['end_error_seconds'],fusion_full_v_change_pp=100*(f['v']-g['v'])))
    # New feature evidence versus a mere logit temperature/shift: rank changes.
    features=[];high_prefix=[]
    assert sha(ROOT/p['labels']['path'])==p['labels']['sha256']
    labels={r['id']:r for r in [json.loads(l)for l in (ROOT/p['labels']['path']).read_text().splitlines()if l.strip()]}
    rr={r['query']:r for r in rows}
    for item,receipt in zip(p['items'],seal['files']):
        assert sha(SUP/receipt['posterior_path'])==receipt['posterior_sha256'];v=torch.load(SUP/receipt['posterior_path'],map_location='cpu',weights_only=False);cap=torch.load(ROOT/item['inputs']['capture'],map_location='cpu',weights_only=False)
        cosine=[];rank=[];peak=[];ratio=[]
        for offset in [0,1]:
            low=cap['temporal_cache']['offsets'][offset]['hidden'];high=v['high_hidden'][offset];assert low.shape==high.shape
            cosine.append(torch.nn.functional.cosine_similarity(low,high,dim=-1).mean((1,2)).tolist())
            lz=cap['temporal_cache']['offsets'][offset]['native_logits'].numpy()[0];hz=v['logits'][offset].numpy()[0]
            rank.append([float(spearmanr(lz[:,k],hz[:,k]).statistic)for k in [0,1]]);peak.append([bool(np.argmax(lz[:,k])!=np.argmax(hz[:,k]))for k in [0,1]])
            gh,gw=v['high_grid_sizes'][offset];lh,lw=v['low_grid_sizes'][offset];ratio.append((gh*gw)/(lh*lw))
        qg=v['posteriors']['Only224six'];qh=v['posteriors']['Only448'];mix=.5*qg+.5*qh
        def kl(a,b):
            valid=a>0;return float((a[valid]*np.log(a[valid]/b[valid])).sum())
        truth=labels[item['id']];boxes=np.array(list(truth['boxes'].values()),float);H=item['record']['metadata']['height'];pixel_height=float(np.mean(boxes[:,3]-boxes[:,1])*224/H)
        pack=torch.load(ROOT/item['inputs']['pack'],map_location='cpu',weights_only=False);candidates=np.asarray(pack['candidate_intervals']);a,b=truth['interval'];overlap=np.maximum(0,np.minimum(b,candidates[:,1])-np.maximum(a,candidates[:,0]));oracle=float((overlap/((b-a)+(candidates[:,1]-candidates[:,0])-overlap)).max())
        features.append(dict(query=item['anonymous'],cosine_by_layer=np.mean(cosine,axis=0).tolist(),start_logit_rank_spearman=float(np.mean(np.array(rank)[:,0])),end_logit_rank_spearman=float(np.mean(np.array(rank)[:,1])),start_argmax_changes=int(np.array(peak)[:,0].sum()),end_argmax_changes=int(np.array(peak)[:,1].sum()),spatial_tokens_ratio=float(np.mean(ratio)),posterior_JS=.5*(kl(qg,mix)+kl(qh,mix)),mean_GT_target_height_at224_pixels=pixel_height,high_minus_global_t_pp=100*(rr[item['anonymous']]['values']['Only448']['t']-rr[item['anonymous']]['values']['Only224six']['t']),low13_candidate_oracle_t=oracle,high_above_low13_oracle=bool(rr[item['anonymous']]['values']['Only448']['t']>oracle+1e-10),high_half_success_missing_from_low13=bool(rr[item['anonymous']]['values']['Only448']['t']>=.5 and oracle<.5)))
        high_prefix.append(receipt['prefix_seconds'])
    feature_summary=dict(mean_cosine_by_layer=np.mean([r['cosine_by_layer']for r in features],axis=0).tolist(),start_rank_spearman_mean=float(np.mean([r['start_logit_rank_spearman']for r in features])),end_rank_spearman_mean=float(np.mean([r['end_logit_rank_spearman']for r in features])),offset_start_argmax_changes=sum(r['start_argmax_changes']for r in features),offset_end_argmax_changes=sum(r['end_argmax_changes']for r in features),offset_comparisons=126,mean_spatial_tokens_ratio=float(np.mean([r['spatial_tokens_ratio']for r in features])),median_GT_target_height_at224_pixels=float(np.median([r['mean_GT_target_height_at224_pixels']for r in features])),target_pixel_height_vs_high_gain_spearman=float(spearmanr([r['mean_GT_target_height_at224_pixels']for r in features],[r['high_minus_global_t_pp']for r in features]).statistic))
    feature_summary['low13_oracle_mean_t']=float(np.mean([r['low13_candidate_oracle_t']for r in features]));feature_summary['high_above_low13_oracle_queries']=[r['query']for r in features if r['high_above_low13_oracle']];feature_summary['high_half_success_missing_from_low13_queries']=[r['query']for r in features if r['high_half_success_missing_from_low13']]
    gains={}
    for arm in ['SingleFusion0','SingleFusion1','High6Fusion']:
        gains[arm]={}
        for field in ['t','v']:
            g=np.mean([r['values']['Only224six'][field]for r in rows]);f=np.mean([r['values']['Fusion'][field]for r in rows]);a=np.mean([r['values'][arm][field]for r in rows]);gains[arm][field]=dict(gain_over_global_pp=100*(a-g),gap_to_old_fusion_pp=100*(a-f),gain_retained_fraction=(a-g)/(f-g)if abs(f-g)>1e-12 else None)
    prefix=np.array(high_prefix);cost=dict(mean_two448_seconds=float(np.mean([r['Source448_seconds']for r in seal['files']])),mean_decode_seconds=float(np.mean([r['decode_seconds']for r in seal['files']])),mean_prefix0_seconds=float(prefix[:,0].mean()),mean_prefix1_seconds=float(prefix[:,1].mean()),scope='Both native448 offsets measured. Prefix stage timings are not single-view deployment wall; no mean single-view speed claim.')
    smoke=ROOT/'outputs/highres_single_smoke_v1/receipt.json'
    if smoke.exists():cost['single_view_one_case_engineering']=read(smoke)
    write(SUP/'private_feature_diagnostics.json',dict(features=features))
    public_features=[{k:v for k,v in r.items()if k!='mean_GT_target_height_at224_pixels'}for r in features]
    result=dict(status='completed_root_analysis',matched_queries=63,common_queries=64,exposure=report['exposure'],means63=summarize(rows,names,'values'),fixed_space_means63=summarize(rows,names,'fixed_spatial'),layer_2x2=layer_panel,paired63=comp,threshold_patterns64=threshold_patterns,all_crossing_cases64=cases,features=public_features,feature_summary=feature_summary,descriptive_gain_retention=gains,cost=cost,production_changed=False,source_reference='https://arxiv.org/html/2502.11168v1',source_fact='Original paper reports random-resized-crop training with short side420 and ResNet101/VideoSwin/RoBERTa. Nearer training scale is plausible, not causally isolated here.',interpretation_bounds='The resolution change recomputes visual/motion/text fusion and native query flow. Endpoint rank changes refute a pure scalar temperature account, but do not assign causal shares to small details, context or training-scale alignment. Latent cosine is descriptive, not a correctness signal. Local224 did not create comparable pixel scale; no actionness used as new algorithm input.',proof=dict(main63_sealed_before_labels=True,old_Global6_Strong_recovered_exactly_all63=True,shared_local_baselines_same=True,no_new_model_calls_in_analysis=True))
    write(PUBLIC/'analysis.json',result)
    fields=list(cases[0])
    with(PUBLIC/'crossing_cases.tsv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(cases)
    lines=['# Native448 decomposition and why local224 did not replace it','',report['exposure'],'','Common4 arms use64; all following resolution/layer/offset/local comparisons use exactly63. Single offset0 is fixed primary; offset1 sensitivity is reported without posthoc selection. No weight scan or production change.','','| Method, matched63 | tIoU % | Full vIoU % | Boundary MAE s |','|---|---:|---:|---:|']
    for name,m in result['means63'].items():lines.append(f"| {name} | {100*m['m_tIoU']:.4f} | {100*m['m_vIoU']:.4f} | {m['boundary_MAE_seconds']:.4f} |")
    lines+=['','| Comparison, matched63 | tIoU pp [95% CI] | Full vIoU pp [95% CI] | Fixed-space vIoU pp |','|---|---:|---:|---:|']
    for name,x in comp.items():
        fmt=lambda d:f"{d['mean']:+.4f} [{d['CI95'][0]:.4f},{d['CI95'][1]:.4f}]"
        lines.append(f"| {name} | {fmt(x['t'])} | {fmt(x['v'])} | {x['fixed_space_v']['mean']:+.4f} |")
    lines+=['','## All threshold-crossing cases','','| Query | Category | G t% | H t% | Fusion t% | Local t% | H start error delta s | H end error delta s |','|---|---|---:|---:|---:|---:|---:|---:|']
    for x in cases:lines.append(f"| {x['query']} | {x['type']} | {x['t_global']:.3f} | {x['t_high']:.3f} | {x['t_fusion']:.3f} | {x['t_local']:.3f} | {x['high_start_error_change']:+.3f} | {x['high_end_error_change']:+.3f} |")
    lines+=['','## Input and model evidence','',json.dumps(feature_summary,indent=2),'',result['interpretation_bounds'],'',
        'The local224 trial produced55 whole-frame windows and effective scale only1.0014x vertical/1.0090x horizontal on average. It changed single-offset/last-layer weighting while failing to add448-level pixels. Actual448 increases spatial tokens and recomputes the native upstream evidence; late224 replay cannot reconstitute downsampled RGB. This does not prove all224 representations lack event information.',
        '',result['source_fact']+' [Primary paper](https://arxiv.org/html/2502.11168v1).','',
        'Full saved/paused queue remains6088/10303; original deferred plan retained untouched. All signed losses and severe tails remain in report.json/signed_queries.tsv. No GT-based query selection, offset selection or deployment promotion.']
    (PUBLIC/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(means63=result['means63'],paired63=comp,patterns=threshold_patterns,features=feature_summary,gain_retention=gains,cost=cost),indent=2))

if __name__=='__main__':analyze()
