"""Anonymous teacher-family comparison, same B3 evidence and Student."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_qwen_b3_temporal32_v1 import BASE,read,write,sha
OUT=ROOT/'results/stvg_qwen_temporal/2026-10-17/b3_matched32'
def report():
    from scripts.report_stvg_native_head_adapter32_v1 import audit
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    d=read(BASE/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json');checks=audit(d,rr);a=dict(status='pass',count=32,checks=checks,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'));write(BASE/'ROOT_SCALAR_READBACK.json',a);aliases={r['probe_alias']:f'R{i:03d}'for i,r in enumerate(rr,1)}
    def anon(v):
        if isinstance(v,dict):return {aliases.get(k,k):anon(x)for k,x in v.items()if k!='GT_bindings'}
        if isinstance(v,list):return [anon(x)for x in v]
        return aliases.get(v,v)if isinstance(v,str)else v
    OUT.mkdir(parents=True,exist_ok=False);write(OUT/'RESULTS.json',anon(d));write(OUT/'ROOT_SCALAR_READBACK.json',a);write(OUT/'METHOD_LOCK.json',read(BASE/'METHOD_LOCK.json'));write(OUT/'QUALIFICATION_COMPLETION.json',read(BASE/'QUALIFICATION_COMPLETION.json'));write(OUT/'SOURCE_BINDINGS.json',dict(count=32,records=[dict(case_id=aliases[r['probe_alias']],dataset=r['dataset'],parent_sha256=r['parent_sha256'],source_checkpoint_sha256=r['source_checkpoint_sha256'])for r in rr]));write(OUT/'COST.json',dict(Teacher_calls=read(BASE/'TEACHER_BARRIER.json')['calls'],stages=[dict(stage=p.parent.name,**read(p))for p in (BASE/'execution').glob('*/WORKER_EXIT.json')],R1_reused_cost_not_zero=True,new_Spatial_fits=0))
    def val(ds,arm,k):return next(s['mean']for s in d['summary']if s['dataset']==ds and s['arm']==arm and s['metric']==k)
    lines=['# Qwen3.5 versus R1, unchanged B3 temporal adaptation32','','Same exposed HC16/Vid16, exact original B3 two hypothesis intervals and marked8frame NPY observations. Qwen3.5-9B BF16 replaces frozen R1 only. Same caption text task, original A/B/C support/contradiction/uncertain alignment, r=P(A)-P(B),160token cap, candidate prior/target/likelihood merged interval masses, tau1/full-grid trust1/512parameter originalLBFGS50/Armijo, actual FP32 head reinsertion and complete marginal MAP. Native non-thinking Qwen processor/template are architecture-specific, so this compares the whole caption+judge Teacher interface, not only the text judge. No new Actionness, hypotheses, Prompt or parameter search, no Spatial updates or deployment change. All outputs sealed before GT.','','|Domain|Arm|vIoU %|tIoU %|dense sIoU %|','|---|---|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        for arm in ['Frozen','MARGINAL_NO_UPDATE','R1_B3','QWEN_B3','QWEN_TARGET_HARD']:lines.append('|'+ds+'|'+arm+'|'+'|'.join('unknown'if(v:=val(ds,arm,k))is None else f'{100*v:.6f}'for k in ['m_vIoU','m_tIoU','m_sIoU_dense_GT'])+'|')
    lines+=['','All-source paired t95/negative tails/unknowns and original R1/Qwen rewards/posteriors, allowed-label mass, target changes, fit KL and steps are in RESULTS.json. No-information sources retain Source marginal MAP and remain in the denominator. Confidence is not calibrated accuracy; GT interval-order diagnostics are not new semantic labels. Existing semantic panel remains44 model-assisted/468pending. No independent confirmation or cross-backbone evidence. No mixed Full is automatically launched.']
    for ds in ['hc2','vidstg']:
        u=[x for x in d['usage']if x['dataset']==ds];info=[x for x in u if x['Qwen_informative']];unc=[p[2]for x in u for p in x['Qwen_posteriors']];lines.append(f'\n{ds}: informative Qwen {len(info)}/16 versus R1 {sum(x["R1_informative"]for x in u)}/16; uncertain posterior mean {math.fsum(unc)/len(unc)if unc else None}; accepted optimizer steps {sum(x["optimizer_steps"]for x in u)}. These are interface/transfer diagnostics, not semantic correctness.')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');(OUT/'PROTOCOL.md').write_bytes((ROOT/'protocols/stvg_qwen_b3_temporal32_v1.md').read_bytes())
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));aa=['MARGINAL_NO_UPDATE','R1_B3','QWEN_B3']
    for ax,ds in zip(axes,['hc2','vidstg']):ax.bar(['No update','R1','Qwen'],[100*val(ds,x,'m_vIoU')for x in aa]);ax.set_title(ds+' fixed32 cohort / all16');ax.set_ylabel('vIoU (%)')
    fig.tight_layout();fig.savefig(OUT/'effects.png',dpi=160);plt.close(fig)
    for rel in ['scripts/stvg_qwen_b3_temporal32_v1.py','scripts/continue_stvg_qwen_b3_temporal32_v1.py','scripts/report_stvg_qwen_b3_temporal32_v1.py','vg_tta/stvg_native_head_residual32_v1.py']:
        p=OUT/'code'/Path(rel).name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/rel).read_bytes()))
    paths={str(p.relative_to(ROOT))for p in OUT.rglob('*')if p.is_file()};verify_privacy(paths);write(BASE/'publication/temporal32/EXPORT_MANIFEST.json',dict(status='complete',count=32,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]));print('QWEN_TEMPORAL_REPORT_READY',checks,flush=True)
if __name__=='__main__':report()
