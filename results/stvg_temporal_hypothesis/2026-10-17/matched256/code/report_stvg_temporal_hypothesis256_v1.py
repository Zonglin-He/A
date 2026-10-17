"""Anonymous original256 B/Frozen/T2 result export and independent checks."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_hypothesis256_v1 import BASE,OLD,T2,read,write,sha
OUT=ROOT/'results/stvg_temporal_hypothesis/2026-10-17/matched256'
def independent(rows,summary,paired):
    import numpy as np
    from scipy.stats import t
    idx={(r['case_id'],r['arm']):r['metrics'] for r in rows};checks=0
    for s in summary:
        ids={r['case_id'] for r in rows if r['dataset']==s['dataset']};v=[idx[c,s['arm']][s['metric']] for c in ids if idx[c,s['arm']][s['metric']] is not None];assert len(v)==s['valid'] and s['unknown']==len(ids)-len(v);assert not v or abs(math.fsum(v)/len(v)-s['mean'])<1e-12;checks+=1
    for p in paired:
        a,b=p['comparison'].split('_minus_');ids={r['case_id'] for r in rows if r['dataset']==p['dataset']};v={c:100*(idx[c,a][p['metric']]-idx[c,b][p['metric']]) for c in ids if idx[c,a][p['metric']] is not None and idx[c,b][p['metric']] is not None};assert v==p['per_source_delta_pp'];vv=list(v.values());n=len(v);m=math.fsum(vv)/n if n else None;assert m==p['mean_delta_pp'] and p['valid']==n and p['unknown']==len(ids)-n
        if n>1:
            e=float(t.ppf(.975,n-1)*np.std(vv,ddof=1)/math.sqrt(n));assert np.allclose([m-e,m+e],p['paired_source_t95_interval_pp'],atol=1e-11,rtol=0)
        assert [sum(x>1e-6 for x in vv),sum(x< -1e-6 for x in vv),sum(abs(x)<=1e-6 for x in vv)]==[p[k] for k in ['positive','negative','neutral']];checks+=n
    return checks
def report():
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    from scripts.stvg_temporal_hypothesis_fit32_v1 import CFG
    d=read(BASE/'POSTSEAL_RESULTS.json');assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)};checks=independent(d['rows'],d['summary'],d['paired'])
    for s in d['strata']:
        ids={u['case_id'] for u in d['usage'] if u['reused_prior32']==(s['name']=='prior32')};checks+=independent([r for r in d['rows'] if r['case_id'] in ids],s['summary'],s['paired'])
    def anon(x):
        if isinstance(x,dict):return {aliases.get(k,k):anon(v) for k,v in x.items() if k!='GT_bindings'}
        if isinstance(x,list):return [anon(v) for v in x]
        return aliases.get(x,x) if isinstance(x,str) else x
    OUT.mkdir(parents=True,exist_ok=False);write(OUT/'RESULTS.json',anon(d));method=dict(count=256,per_domain=128,old32_reused=32,added224=True,development_exposed=True,Source='cross-domain TA-STVG',source_checkpoint_sha256_by_target={k:sorted({r['source_checkpoint_sha256'] for r in rr if r['dataset']==k}) for k in ['hc2','vidstg']},Teacher='frozen STVG-R1-7B',revision='c0db00d019bf746f54b67770b82261b2161fed59',frames_per_hypothesis=8,hypotheses=2,Native='actual FP32 two-offset native MAP merge',Alternative='highest Native joint state with different actual uniform8 frame set',probability='conditional on two retained joint offset states, not merged-interval marginal',reward='three-label P(A)-P(B), not calibrated semantic correctness',config=CFG,parameters=512,external_Actionness_Projection=False,internal_frozen_action_gates=True,Spatial='disabled; Source boxes exact',unchanged_B_fitter_sha256=sha(ROOT/'scripts/stvg_temporal_hypothesis_fit32_v1.py'),old32_result_sha256=sha(OLD/'POSTSEAL_RESULTS.json'),same256_T2_result_sha256=sha(T2/'POSTSEAL_RESULTS.json'),CURRENT_changed=False);write(OUT/'ACTUAL_METHOD.json',method)
    teacher=read(BASE/'TEACHER_BARRIER.json');calls=teacher['calls'];write(OUT/'COST.json',dict(new_visual_description_generations=calls['captions'],new_consistency_prefills=calls['prefills'],extra_prefix_qualifications=calls['extra_prefix_qualifications'],Teacher_model_load_seconds=teacher['load_seconds'],old32_complete_outputs_reused=32,old32_historical_calls_not_zero=True,new_actual_temporal_optimized_queries=sum(u['optimizer_steps']>0 for u in d['usage'] if not u['reused_prior32']),all256_optimized_queries=sum(u['optimizer_steps']>0 for u in d['usage']),new_optimizer_steps=sum(u['optimizer_steps'] for u in d['usage'] if not u['reused_prior32']),new_Spatial_fits=0,new_T2_fits=0,Source_captures=0,stage_costs=[dict(stage=p.parent.name,**read(p)) for p in sorted((BASE/'execution').glob('*/WORKER_EXIT.json'))],cache_continuation_not_fresh_end_to_end_speedup=True))
    recovery=BASE/'recovery/missing_prefix002/PREFIX_RECONSTRUCTION_COMPLETION.json'
    if recovery.exists():
        q=read(recovery);assert q['status']=='pass' and q['count']==2;c=read(OUT/'COST.json');c.update(fresh_frozen_Source_prefix_forwards=2,missing_prefix_recovery_preserved_readouts=33,Teacher_and_fit_repeated_for_recovery=0,original_readout_failure_preserved=True);write(OUT/'COST.json',c,mutable=True)
    write(OUT/'SOURCE_BINDINGS.json',dict(count=256,records=[dict(case_id=aliases[r['probe_alias']],dataset=r['dataset'],parent_sha256=r['parent_sha256'],query_key_sha256=r['query_key_sha256'],source_checkpoint_sha256=r['source_checkpoint_sha256']) for r in rr],private_text_and_media_excluded=True))
    lines=['# Original256: Native Temporal B versus matched Frozen and T2','','Same original HC128/Vid128 exposed development parents. The current B implementation and all per-query parameters are unchanged. Old32 complete results are reused after exact source/hypothesis/pixel/Teacher/fit/readout identity; only remaining224 acquire new evidence and fit. This is not an independent confirmation or a single-factor I2/Actionness ablation. No Spatial adaptation/Full run, new Source capture, T2 rerun, Prompt/parameter search or CURRENT promotion.','']
    def table(s,heading):
        out=[heading,'','|Domain|Arm|vIoU %|tIoU %|fixed dense sIoU %|R@.3 %|R@.5 %|','|---|---|---:|---:|---:|---:|---:|']
        for ds in ['hc2','vidstg']:
            for arm in ['Frozen','T2','B']:
                def val(k):return next(x['mean'] for x in s['summary'] if x['dataset']==ds and x['arm']==arm and x['metric']==k)
                out.append('|'+ds+'|'+arm+'|'+'|'.join(f'{100*val(k):.6f}' for k in ['m_vIoU','m_tIoU','m_sIoU_dense_GT','R_vIoU_at_0p3','R_vIoU_at_0p5'])+'|')
        out+=['','|Domain|Contrast|Metric|Delta pp|paired source t95 pp|+/-/neutral/unknown|','|---|---|---|---:|---|---|']
        for p in s['paired']:
            if p['metric'] in ['m_vIoU','m_tIoU']:out.append(f"|{p['dataset']}|{p['comparison']}|{p['metric']}|{p['mean_delta_pp']:.6f}|{p['paired_source_t95_interval_pp']}|{p['positive']}/{p['negative']}/{p['neutral']}/{p['unknown']}|")
        return out+['']
    lines+=table(d,'## Complete256 (primary)')
    for s in d['strata']:lines+=table(s,'## '+s['name']+' / '+str(s['count'])+' sources')
    lines+=['Different sampled inputs do not establish different event semantics. Nonzero P(A)-P(B) differences and conditional targets are uncalibrated numerical evidence, not verified referent/event labels. Captions never see the original Query; consistency evaluation does. Invalid/truncated/equal/missing-hypothesis cases remain exact Native and stay in all denominators. No third hypothesis or retry.','', 'Native candidate prior is tilted with tau1; zero-init512 endpoint adaptation minimizes joint-state candidate KL and sum offset Native-grid trust KL(lambda1). Original LBFGS50/Armijo remains, and reaching50 iterations is not proof of full convergence. Final intervals come from actual complete native-grid decode. Frozen and T2 are independently rescored from sealed predictions and checked exactly against old scalars. All B Source boxes remain exact. All256 prediction/root barriers precede new GT scoring.','', 'These are exposed development sources, including the previously read32; no unseen confirmation/noninferiority claim follows automatically. The Teacher was trained on target-domain datasets and exact checkpoint-parent overlap is unresolved. No identity/role semantic panel is completed by numerical metrics or engineering pixel inspection. Other-backbone plug-and-play is unmeasured; current native two-offset512 interface and internal frozen gates remain.','']
    if recovery.exists():lines+=['Engineering recovery: two original Source prefix cache receipts were absent. The first readout worker stopped after33 new complete readouts, before any new GT scoring. Original failure/log/runtime and those33 outputs are preserved. Two full frozen prefixes were reconstructed from SHA-matched original video pixels and checked exactly against sealed Source logits, hidden states, boxes, records and Native intervals. Remaining191 readouts then continued with unchanged fitted parameters; no Teacher generation or fitting was repeated and the original deadline was retained.','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');write(OUT/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=256,checks=checks,all_signed_unknown_and_strata_reconstructed=True,baseline_scalar_checks=d['baseline_scalar_checks'],GT_read=False))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4));arms=['Frozen','T2','B']
    for ax,ds in zip(axs,['hc2','vidstg']):
        vals=[100*next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']=='m_vIoU') for a in arms];ax.bar(arms,vals);ax.set_title(ds+' / all128');ax.set_ylabel('vIoU (%)')
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom')
        ax.set_ylim(0,max(vals)*1.15)
    fig.tight_layout();fig.savefig(OUT/'effects.png',dpi=170);plt.close(fig)
    for n in ['stvg_temporal_hypothesis256_v1','continue_stvg_temporal_hypothesis256_v1','continue_stvg_temporal_hypothesis256_recovery002','score_stvg_temporal_hypothesis256_v1','report_stvg_temporal_hypothesis256_v1','publish_stvg_temporal_hypothesis256_v1','verify_stvg_temporal_hypothesis256_remote_v1','stvg_temporal_hypothesis_fit32_v1','stvg_temporal_hypothesis32_v1','test_stvg_temporal_hypothesis_fit32_v1','stvg_temporal_semantic_view_fit32_v1']:
        p=OUT/'code'/(n+'.py');p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/(n+'.py')).read_bytes()))
    (OUT/'PROTOCOL.md').write_text((ROOT/'protocols/stvg_temporal_hypothesis256_v1.md').read_text());decision();paths={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()};verify_privacy(paths);write(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=256,checks=checks,GT_read=False));write(BASE/'publication/matched256/EXPORT_MANIFEST.json',dict(status='complete',count=256,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print('HYPOTHESIS256_EXPORT_READY',checks)
def decision():
    d=read(BASE/'POSTSEAL_RESULTS.json');contrasts=[{k:v for k,v in p.items() if k!='per_source_delta_pp'} for p in d['paired'] if p['comparison'].startswith('B_minus_') and p['metric'] in ['m_vIoU','m_tIoU']];write(OUT/'DECISION.json',dict(status='no_independent_Temporal_gain_established',scope='same original exposed development256 only',B_minus_Frozen_mean_negative_both_domains=True,B_minus_Frozen_t95_crosses_zero_both_domains=True,B_minus_T2_t95_below_zero_both_domains_and_metrics=True,contrasts=contrasts,source_subset_alone_explanation_not_sufficient=True,Actionness_alone_causal_attribution_not_established=True,not_single_factor_I2_comparison=True,no_Full256_run=True,no_promotion=True,no_automatic_parameter_search=True,CURRENT_changed=False))
    p=OUT/'RESULTS.md';p.write_text(p.read_text()+'\nMeasured decision: B-minus-Frozen vIoU means are -0.132646pp HC and -0.064423pp Vid; both paired95% intervals cross zero. B-minus-T2 is -4.813401pp HC and -0.507402pp Vid, with both paired intervals entirely below zero. Thus the current Native-only B does not retain old T2 benefit on the original256; the earlier32 subset alone cannot explain the method gap. This is not an isolated Actionness removal or I2-swap experiment, so the gap is not fully attributed to either component. No Full256 effect or universal failure is inferred.\n')
def render():
    import numpy as np
    from PIL import Image,ImageDraw
    assert read(BASE/'INPUT_BARRIER.json')['count']==256;rr=read(BASE/'ROSTER_LOCK.json')['records'];jm={r['case_id']:r for r in read(BASE/'INPUT_BARRIER.json')['jobs']};files=[]
    for ds in ['hc2','vidstg']:
        r=next(r for r in rr if r['dataset']==ds and not jm[r['probe_alias']]['reused_prior32']);cid=r['probe_alias'];j=read(ROOT/jm[cid]['path']);canvas=Image.new('RGB',(1600,800),(20,20,20));dr=ImageDraw.Draw(canvas)
        for row,h in enumerate(j['selection']['hypotheses']):
            if 'video_path' not in h:continue
            v=np.load(ROOT/h['video_path'],allow_pickle=False)
            for col,i in enumerate([0,2,5,7]):
                im=Image.fromarray(v[i].transpose(1,2,0));im.thumbnail((390,360));canvas.paste(im,(col*400,row*400+30));dr.text((col*400+6,row*400+8),('Native' if row==0 else 'Alternative')+' / frame '+str(h['sampled_physical_frame_ids'][i]),fill='white')
        p=BASE/('private_first_new_hypothesis_'+ds+'.png');canvas.save(p);files.append(dict(dataset=ds,path=str(p.relative_to(ROOT)),sha256=sha(p),engineering_pixels_only=True));print(p)
    write(BASE/'ROOT_PIXEL_VIEW_BINDING.json',dict(status='prepared_pending_actual_view',count=2,files=files,GT_read=False))
if __name__=='__main__':globals()[sys.argv[1]]()
