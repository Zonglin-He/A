"""Independent saved-scalar audit, anonymous export and fixed private pixel panel."""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_qwen35_geometry32_v1 import BASE,PREVIOUS,COMPLEMENT,read,sha,write
OUT=ROOT/'results/stvg_qwen35_geometry32/2026-10-17/measured32'
def report():
    import numpy as np
    from scipy.stats import t
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json')
    d=read(BASE/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)}
    idx={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};checks=0
    for s in d['summary']:
        ids=[r['probe_alias'] for r in rr if r['dataset']==s['dataset']];v=[idx[c,s['arm']][s['metric']] for c in ids if idx[c,s['arm']][s['metric']] is not None]
        assert len(v)==s['valid'] and 16-len(v)==s['unknown'];assert not v or abs(math.fsum(v)/len(v)-s['mean'])<1e-12;checks+=1
    for p in d['paired']:
        a,b=p['comparison'].split('_minus_');v={r['probe_alias']:100*(idx[r['probe_alias'],a][p['metric']]-idx[r['probe_alias'],b][p['metric']]) for r in rr if r['dataset']==p['dataset'] and idx[r['probe_alias'],a][p['metric']] is not None and idx[r['probe_alias'],b][p['metric']] is not None}
        assert v==p['per_source_delta_pp'];n=len(v);vs=list(v.values());mean=math.fsum(vs)/n if n else None;assert mean==p['mean_delta_pp'] and n==p['valid'] and 16-n==p['unknown'];ci=None
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
    OUT.mkdir(parents=True,exist_ok=False)
    quality=[]
    for ds in ['hc2','vidstg']:
        qs=[q for q in d['reference_quality'] if q['dataset']==ds];known=[q for q in qs if q['GT_known']];vals=[q['conditional_IoU'] for q in known if q['conditional_IoU'] is not None]
        quality.append(dict(dataset=ds,planned=len(qs),available=sum(q['reference_available'] for q in qs),GT_known=len(known),GT_unknown=len(qs)-len(known),
            success_at_0p5_over_all_GT_known=sum(bool(q['success_at_0p5']) for q in known)/len(known) if known else None,
            conditional_IoU_mean=math.fsum(vals)/len(vals) if vals else None,conditional_count=len(vals),missing_or_abstention_count_as_failure_on_known=True,semantic_identity_not_inferred=True))
    for name,obj in [('RESULTS.json',d),('REFERENCE_QUALITY.json',quality),('QUALIFICATION_COMPLETION.json',read(BASE/'QUALIFICATION_COMPLETION.json')),
        ('METHOD_PLAN.json',read(BASE/'METHOD_PLAN.json')),('ENVIRONMENT_LOCK.json',read(BASE/'ENVIRONMENT_LOCK.json')),
        ('ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,all_signed_unknown_pairs_checked=True,new_GT_reader=False))]:write(OUT/name,anon(obj))
    write(OUT/'COST.json',dict(stages=[dict(stage=p.parent.name,**read(p)) for p in sorted((BASE/'execution002').glob('*/WORKER_EXIT.json'))],
        formal_Qwen_generations=128,parent_qualification_requests_included=4,new_Temporal=0,new_DINO=0,old_baseline_cost_not_zero=True,
        actual_Student_fits=sum(r['actual_fit'] for r in d['usage']),D2_parameter_updates=0))
    def value(ds,a,k):return next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']==k)
    lines=['# Qwen3.5 independent sparse geometry32: original OPD versus same-reference propagation','',
        'Fixed exposed HC16/Vid16 parents, same128 T2 anchors, original unmarked context4 and standalone anchor image. Qwen3.5-9B BF16 uses one fixed non-thinking normalized1000 protocol. Four prelocked parent interface requests count among128 formal requests; no outcome retry, Prompt sweep or model substitution. Original R1 temporal feedback is retained: this is not a single-Teacher system. Native processor resize/clock and actual generation are bound by the qualification receipt. The historical uniform context-clock approximation is retained and disclosed; source timestamps are preserved in private input audits.', '',
        'QWEN_OPD uses unchanged original1792-parameter episodic Gaussian/response/fullstep rollback/reject-stop. QWEN_D2 uses the exact same sparse Qwen references and unchanged Source-relative center/log-size interpolation in physical time inside T2, with no Student updates. Missing/invalid/abstained references remain in the denominator. Both complete prediction families and roots precede GT. Frozen, T2, WF, original and revised R1 Direct+OPD, revised R1 same-reference D2, R1 Heads and D0/D1/D2 are exact matched old predictions; their cost is not zero. All negative/neutral/unknown source deltas and extreme cases are retained. Development32 intervals quantify uncertainty, not a mandatory per-domain exclusion-of-zero selection gate. No domain splicing, new independent confirmation, production promotion or temporal-Teacher migration.', '',
        '| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |','|---|---|---:|---:|---:|---:|---:|']
    arms=['Frozen','T2','Current_WF','R1_DIRECT','R1_FIXED_OPD','R1_FIXED_D2','R1_HEADS','D0','D1','D2','QWEN_OPD','QWEN_D2']
    for ds in ['hc2','vidstg']:
        for a in arms:lines.append('|'+ds+'|'+a+'|'+'|'.join('unknown' if (v:=value(ds,a,k)) is None else f'{100*v:.6f}' for k in ['m_vIoU','m_sIoU_dense_GT','m_tIoU','R_vIoU_at_0p3','R_vIoU_at_0p5'])+'|')
    lines+=['','| Domain | Contrast | delta vIoU pp | paired t95 pp | + / - / neutral / unknown |','|---|---|---:|---|---|']
    for p in d['paired']:
        if p['metric']=='m_vIoU' and p['comparison'] in ['QWEN_OPD_minus_R1_FIXED_OPD','QWEN_OPD_minus_R1_FIXED_D2','QWEN_D2_minus_R1_FIXED_D2','QWEN_OPD_minus_Current_WF','QWEN_OPD_minus_D1','QWEN_OPD_minus_D0','QWEN_D2_minus_QWEN_OPD','QWEN_D2_minus_D2']:
            lines.append(f"|{p['dataset']}|{p['comparison']}|{p['mean_delta_pp']:.6f}|{p['paired_source_t95_interval_pp']}|{p['positive']}/{p['negative']}/{p['neutral']}/{p['unknown']}|")
    lines+=['','RESULTS.json includes all paired sources, nonanchor geometry and usage. REFERENCE_QUALITY.json separates overall known-position success from conditional IoU and unknowns. These geometric proxies do not replace semantic identity/role review; the original semantic512 panel remains44 model-assisted/468 pending.']
    doc=ROOT/'docs/STVG_QWEN35_GEOMETRY32_MEASURED_001.md';doc.write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,['hc2','vidstg']):
        aa=['Current_WF','R1_FIXED_OPD','R1_FIXED_D2','QWEN_OPD','QWEN_D2'];ax.bar(aa,[100*(value(ds,a,'m_vIoU')-value(ds,'T2','m_vIoU')) for a in aa]);ax.axhline(0,color='black',lw=.7);ax.set_title(ds+' all16 parents');ax.tick_params(axis='x',rotation=40);ax.set_ylabel('vIoU delta from T2 (pp)')
    fig.tight_layout();fig.savefig(OUT/'effects.png',dpi=160);plt.close(fig)
    for name in ['stvg_qwen35_geometry_run_v2.py','continue_stvg_qwen35_geometry32_002.py','report_stvg_qwen35_geometry32_v2.py','stvg_qwen35_geometry_adapters_v2.py','test_stvg_qwen35_geometry_resume002.py','stvg_qwen35_geometry_run_v1.py','stvg_qwen35_geometry32_v1.py','stvg_qwen35_geometry_adapters_v1.py','stvg_qwen35_geometry_adapters_v1.py','stvg_qwen35_geometry_contract_v1.py','test_stvg_qwen35_geometry_run_v1.py']:
        p=OUT/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
    paths={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT)),'protocols/stvg_qwen35_geometry32_run002.md'}
    verify_privacy(paths);write(BASE/'publication/measured32/EXPORT_MANIFEST.json',dict(status='complete',count=32,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]))
    write(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,new_GT_reader=False));print('QWEN32_EXPORT_READY',checks)

def render():
    import numpy as np,torch
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    assert read(BASE/'CPU_COMPLETION.json')['status']=='complete';rr=read(BASE/'ROSTER_LOCK.json')['records'];jobs=read(BASE/'INPUT_JOB_LOCK.json')['records']
    refs={r['case_id']:{a['position']:a for a in r['anchors']} for r in read(BASE/'QWEN_REFERENCE_LOCK.json')['records']};bm={r['case_id']:r for r in read(BASE/'BASELINE_BARRIER.json')['files']}
    bars={'QWEN_OPD':BASE/'arms/QWEN_OPD/trials/G32/GLOBAL_PREDICTION_BARRIER.json','QWEN_D2':BASE/'D2_GLOBAL_PREDICTION_BARRIER.json'}
    fig,axes=plt.subplots(2,3,figsize=(14,8));bindings=[]
    for row,ds in enumerate(['hc2','vidstg']):
        rec=next(r for r in rr if r['dataset']==ds);cid=rec['probe_alias'];j=next(j for j in jobs if j['case_id']==cid);pos=j['position'];assert sha(ROOT/j['image']['path'])==j['image']['sha256']
        rgb=np.load(ROOT/j['image']['path'],allow_pickle=False);H,W=rgb.shape[:2];old=torch.load(ROOT/bm[cid]['path'],map_location='cpu',weights_only=False);preds={'Current_WF':old['arms']['TS_M2']}
        for a,p in bars.items():
            f=next(f for f in read(p)['files'] if f['case_id']==cid);assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);preds[a]=d['arm'] if a=='QWEN_D2' else d['arms']['TS_M2']
        for col,(a,p) in enumerate(preds.items()):
            ax=axes[row,col];ax.imshow(rgb);x,y,u,v=map(float,p['boxes'][pos]);ax.add_patch(Rectangle((x*W,y*H),(u-x)*W,(v-y)*H,fill=False,color='cyan',lw=1.5,label='actual output'))
            ref=refs[cid].get(pos)
            if a.startswith('QWEN') and ref:
                x,y,w,h=ref['box_cxcywh'];ax.add_patch(Rectangle(((x-w/2)*W,(y-h/2)*H),w*W,h*H,fill=False,color='yellow',lw=1.4,label='same Qwen reference'))
            ax.set_title(ds+' fixed first source / '+a);ax.axis('off');ax.legend(fontsize=6)
        bindings.append(dict(case_id=cid,job_id=j['job_id'],input_sha256=j['image']['sha256']))
    p=BASE/'private_actual_output_panel.png';fig.tight_layout();fig.savefig(p,dpi=130);plt.close(fig)
    write(BASE/'ROOT_PIXEL_VIEW_BINDING.json',dict(status='prepared_pending_actual_view',count=2,path=str(p.relative_to(ROOT)),sha256=sha(p),records=bindings,GT_read=False));print(p)
if __name__=='__main__':{'report':report,'render':render}[sys.argv[1]]()
