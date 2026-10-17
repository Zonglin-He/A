"""Independent scalar/coverage audit and anonymous report for the one Direct correction."""
from pathlib import Path
import inspect,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_direct_reliable32_v1 import BASE,OLD,read,sha,put
OUT=ROOT/'results/stvg_r1_direct_reliable32/2026-10-17'
def report():
    import numpy as np
    from scipy.stats import t
    from collections import Counter
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json');d=read(BASE/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)};idx={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};checks=0
    for s in d['summary']:
        ids=[r['probe_alias'] for r in rr if r['dataset']==s['dataset']];v=[idx[c,s['arm']][s['metric']] for c in ids if idx[c,s['arm']][s['metric']] is not None]
        assert len(v)==s['valid'] and 16-len(v)==s['unknown'];assert not v or abs(math.fsum(v)/len(v)-s['mean'])<1e-12;checks+=1
    for p in d['paired']:
        a,b=p['comparison'].split('_minus_');v={r['probe_alias']:100*(idx[r['probe_alias'],a][p['metric']]-idx[r['probe_alias'],b][p['metric']]) for r in rr if r['dataset']==p['dataset'] and idx[r['probe_alias'],a][p['metric']] is not None and idx[r['probe_alias'],b][p['metric']] is not None};assert v==p['per_source_delta_pp'];n=len(v);vs=list(v.values());mean=math.fsum(vs)/n if n else None;assert mean==p['mean_delta_pp'] and n==p['valid'] and 16-n==p['unknown'];ci=None
        if n>1:
            e=float(t.ppf(.975,n-1)*np.std(vs,ddof=1)/math.sqrt(n));ci=[mean-e,mean+e]
        assert (ci is None and p['paired_source_t95_interval_pp'] is None) or np.allclose(ci,p['paired_source_t95_interval_pp'],rtol=0,atol=1e-11)
        assert [sum(x>1e-6 for x in vs),sum(x< -1e-6 for x in vs),sum(abs(x)<=1e-6 for x in vs)]==[p[k] for k in ['positive','negative','neutral']];checks+=n
    def anon(v):
        if isinstance(v,dict):return {aliases.get(k,k):anon(x) for k,x in v.items() if k!='GT_bindings'}
        if isinstance(v,list):return [anon(x) for x in v]
        if isinstance(v,str):
            if v in aliases:return aliases[v]
            for cid,a in aliases.items():
                if v.startswith(cid+'_'):return a+v[len(cid):]
        return v
    OUT.mkdir(parents=True,exist_ok=False);quality=[];oldq=read(OLD/'POSTSEAL_RESULTS.json')['reference_quality'];dec=read(BASE/'R1_FIXED_REFERENCE_LOCK.json')['decisions']
    for ds in ['hc2','vidstg']:
        qs=[q for q in d['reference_quality'] if q['dataset']==ds];known=[q for q in qs if q['GT_known']];vals=[q['conditional_IoU'] for q in known if q['conditional_IoU'] is not None];oq=[q for q in oldq if q['dataset']==ds];ok=[q for q in oq if q['GT_known']]
        quality.append(dict(dataset=ds,planned=64,new_reference_available=sum(q['reference_available'] for q in qs),old_reference_available=sum(q['reference']['DIRECT']['available'] for q in oq),
            GT_known=len(known),GT_unknown=64-len(known),new_success_count=sum(bool(q['success_at_0p5']) for q in known),old_success_count=sum(bool(q['reference']['DIRECT']['success_at_0p5']) for q in ok),
            new_conditional_IoU_mean=math.fsum(vals)/len(vals) if vals else None,new_conditional_count=len(vals),
            new_status_counts=dict(Counter(q['status'] for q in dec if q['dataset']==ds)),new_actual_fit_sources=sum(u['actual_fit'] for u in d['usage'] if u['dataset']==ds),
            invalid_abstained_missing_are_failures_on_known=True,geometric_IoU_not_identity_labels=True))
    for name,obj in [('RESULTS.json',d),('REFERENCE_QUALITY.json',quality),('RAW_RESPONSE_AUDIT.json',read(BASE/'RAW_RESPONSE_AUDIT.json')),('TRAINING_EXPOSURE.json',read(BASE/'TRAINING_EXPOSURE.json')),
        ('METHOD_LOCK.json',read(BASE/'METHOD_LOCK.json')),('QUALIFICATION_COMPLETION.json',read(BASE/'QUALIFICATION_COMPLETION.json')),
        ('ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,all_signed_unknown_pairs_checked=True,new_GT_reader=False))]:put(OUT/name,anon(obj))
    put(OUT/'COST.json',dict(stages=[dict(stage=p.parent.name,**read(p)) for p in sorted((BASE/'execution').glob('*/WORKER_EXIT.json'))],new_R1_generations=128,new_Temporal=0,new_DINO=0,
        actual_new_Spatial_fits=sum(u['actual_fit'] for u in d['usage']),both_D2_readouts_no_Student=True,old_prediction_reuse_cost_not_zero=True))
    def value(ds,a,k):return next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']==k)
    lines=['# Reliable R1 Direct feedback: one uniform interface correction','',
        'Same32 exposed development parents (HC16/Vid16),128 fixed T2 anchors, original unmarked context4 and standalone original-aspect image. All128 old raw responses were audited without GT or new model calls; malformed unique numeric literals were never silently admitted. One source-independent synthetic-grammar-tested wire requests an answer-wrapped four-number pixel array or null. It is not claimed calibrated on real Source images. Processor pixels/clock, original checkpoint, greedy256 cap/repetition penalty and original attention backend remain fixed. There is no Prompt search, duplicate-answer recovery, coordinate-unit guessing/clamp or OPD parameter search.', '',
        'R1_FIXED_OPD uses unchanged original1792-parameter episodic Gaussian32/response/fullstep rollback/reject-stop. R1_FIXED_D2 uses exactly the same new references and unchanged physical-time Source-relative center/log-size propagation. R1_ORIGINAL_D2 independently uses the original Direct references, so a DINO/R1-ID propagation result is not substituted. All prediction/root families were sealed before GT; no-reference and extreme sources remain in all32. Original Frozen/T2/WF/Direct/DINO predictions and saved scalars match. This is development evidence, not a new independent confirmation or production promotion.', '',
        'Official model card lists HCSTVG-v1/v2 and VidSTG training domains. The paper reports removing HC validation/test duplicates from training; the local release lacks a complete checkpoint-bound parent manifest, so exact sample overlap is unresolved. No inference-time target GT does not imply a target-domain-unseen teacher. The teacher was trained around visually marked object IDs, not direct coordinate regression. Original sources: https://huggingface.co/XiaowenZhang/stvg-r1-model-7b ; https://arxiv.org/html/2602.11730v1 ; https://github.com/XiaowZhang00/STVG-R1 .', '',
        '| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |','|---|---|---:|---:|---:|---:|---:|']
    arms=['Frozen','T2','Current_WF','R1_DIRECT','D0','R1_ORIGINAL_D2','R1_FIXED_OPD','R1_FIXED_D2']
    for ds in ['hc2','vidstg']:
        for a in arms:lines.append('|'+ds+'|'+a+'|'+'|'.join('unknown' if (v:=value(ds,a,k)) is None else f'{100*v:.6f}' for k in ['m_vIoU','m_sIoU_dense_GT','m_tIoU','R_vIoU_at_0p3','R_vIoU_at_0p5'])+'|')
    lines+=['','| Domain | Contrast | delta vIoU pp | paired t95 pp | + / - / neutral / unknown |','|---|---|---:|---|---|']
    for p in d['paired']:
        if p['metric']=='m_vIoU' and p['comparison'] in ['R1_FIXED_OPD_minus_R1_DIRECT','R1_FIXED_OPD_minus_T2','R1_FIXED_OPD_minus_Current_WF','R1_FIXED_D2_minus_R1_FIXED_OPD','R1_ORIGINAL_D2_minus_R1_DIRECT']:
            lines.append(f"|{p['dataset']}|{p['comparison']}|{p['mean_delta_pp']:.6f}|{p['paired_source_t95_interval_pp']}|{p['positive']}/{p['negative']}/{p['neutral']}/{p['unknown']}|")
    lines+=['','REFERENCE_QUALITY.json reports coverage and successes over every GT-known planned anchor, counting invalid/missing/abstained outputs as failures. Conditional geometry and all nonanchor/source-paired metrics are retained separately. No syntax or IoU proxy is presented as independent identity/role review. No further automatic Prompt or optimizer search; Qwen model continuation remains paused by user.']
    doc=ROOT/'docs/STVG_R1_DIRECT_RELIABLE32_001.md';doc.write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,['hc2','vidstg']):
        aa=['Current_WF','R1_DIRECT','R1_ORIGINAL_D2','R1_FIXED_OPD','R1_FIXED_D2'];ax.bar(aa,[100*(value(ds,a,'m_vIoU')-value(ds,'T2','m_vIoU')) for a in aa]);ax.axhline(0,color='black',lw=.7);ax.set_title(ds+' all16 parents');ax.tick_params(axis='x',rotation=45);ax.set_ylabel('vIoU delta from T2 (pp)')
    fig.tight_layout();fig.savefig(OUT/'effects.png',dpi=160);plt.close(fig)
    for n in ['audit_stvg_r1_direct_interface_v1.py','stvg_r1_direct_reliable32_v1.py','continue_stvg_r1_direct_reliable32_v1.py','report_stvg_r1_direct_reliable32_v1.py','test_stvg_r1_direct_reliable32_v1.py']:
        p=OUT/'code'/n;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/n).read_bytes()))
    paths={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT)),'protocols/stvg_r1_direct_reliable32_v1.md'}
    verify_privacy(paths);put(BASE/'publication/reliable32/EXPORT_MANIFEST.json',dict(status='complete',count=32,paths=sorted(paths),files=[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha(ROOT/p)) for p in sorted(paths)]));put(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,GT_read=False));print('R1_RELIABLE32_EXPORT_READY',checks)
def render():
    from scripts import report_stvg_qwen35_geometry32_v1 as old
    src=inspect.getsource(old.render).replace('QWEN_OPD','R1_FIXED_OPD').replace('QWEN_D2','R1_FIXED_D2').replace('QWEN_REFERENCE_LOCK','R1_FIXED_REFERENCE_LOCK').replace("a.startswith('QWEN')","a.startswith('R1_FIXED')").replace('same Qwen reference','same R1 reference')
    ns=dict(old.__dict__);ns.update(BASE=BASE);exec(compile(src,__file__,'exec'),ns);ns['render']()
if __name__=='__main__':{'report':report,'render':render}[sys.argv[1]]()
