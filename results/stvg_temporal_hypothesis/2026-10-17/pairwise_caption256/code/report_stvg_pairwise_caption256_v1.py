"""Independent all-source scalar/order audit and anonymous finite256 export."""
from pathlib import Path
import inspect,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_pairwise_caption256_v1 import BASE,read,write,sha
OUT=ROOT/'results/stvg_temporal_hypothesis/2026-10-17/pairwise_caption256'
def report():
    from scripts import report_stvg_native_head_adapter32_v1 as auditmod
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    import numpy as np
    d=read(BASE/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json')
    source=inspect.getsource(auditmod.audit).replace('16-len(v)','128-len(v)').replace('16-n','128-n');ns=dict(auditmod.__dict__);exec(compile(source,__file__,'exec'),ns);checks=ns['audit'](d,rr);audit=dict(status='pass',count=256,checks=checks,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'));write(BASE/'ROOT_SCALAR_READBACK.json',audit)
    aliases={r['probe_alias']:f'R{i:03d}'for i,r in enumerate(rr,1)}
    def anon(v):
        if isinstance(v,dict):return {aliases.get(k,k):anon(x)for k,x in v.items()if k!='GT_bindings'}
        if isinstance(v,list):return [anon(x)for x in v]
        return aliases.get(v,v)if isinstance(v,str)else v
    OUT.mkdir(parents=True,exist_ok=False);write(OUT/'RESULTS.json',anon(d));write(OUT/'ROOT_SCALAR_READBACK.json',audit);write(OUT/'METHOD_LOCK.json',read(BASE/'METHOD_LOCK.json'));write(OUT/'SOURCE_BINDINGS.json',dict(count=256,records=[dict(case_id=aliases[r['probe_alias']],dataset=r['dataset'],parent_sha256=r['parent_sha256'],source_checkpoint_sha256=r['source_checkpoint_sha256'])for r in rr]));write(OUT/'COST.json',dict(new_caption_generations=0,new_Spatial_fits=0,Teachers={k:{a:b for a,b in read(BASE/k/'TEACHER_BARRIER.json').items()if a in ['text_prefills','input_tokens','load_seconds']}for k in ['R1','QWEN']},stages=[dict(stage=p.parent.name,**read(p))for p in(BASE/'execution').glob('*/WORKER_EXIT.json')],historical_caption_acquisition_cost_not_zero=True))
    signals=[]
    def sign(x):return int(x>0)-int(x<0)
    for ds in ['hc2','vidstg']:
        use=[u for u in d['usage']if u['dataset']==ds]
        for kind in ['R1','QWEN']:
            counts=dict(total=128,eligible=0,informative=0,GT_interval_order_agrees=0,GT_interval_order_disagrees=0,GT_or_reward_tied_or_unknown=0,old_agrees_on_matched_support=0,old_disagrees_on_matched_support=0,matched_informative_support=0,order_argmax_agrees=0);margins=[];tie=[];ins=[];oldall=[0,0,0]
            for u in use:
                v=u['judges'][kind];counts['eligible']+=v['eligible'];counts['informative']+=not v['no_information'];hm=u['hypothesis_metrics'];gt=None if len(hm)!=2 or any(h['tIoU']is None for h in hm)else hm[0]['tIoU']-hm[1]['tIoU'];old=u['old_independent_rewards'];od=None if len(old)!=2 or any(x is None for x in old)else old[0]-old[1]
                if gt is not None and od is not None and gt!=0 and od!=0:oldall[0 if sign(gt)==sign(od)else 1]+=1
                else:oldall[2]+=1
                rw=v['rewards'];diff=None if any(x is None for x in rw)else rw[0]-rw[1]
                if gt is None or diff is None or gt==0 or diff==0:counts['GT_or_reward_tied_or_unknown']+=1
                else:
                    counts['GT_interval_order_agrees'if sign(gt)==sign(diff)else'GT_interval_order_disagrees']+=1
                    if od is not None and od!=0:counts['matched_informative_support']+=1;counts['old_agrees_on_matched_support'if sign(gt)==sign(od)else'old_disagrees_on_matched_support']+=1
                if v['eligible']:counts['order_argmax_agrees']+=v['sampled_order_argmax_agrees'];margins.append(v['order_margin_difference']);tie.append(v['posterior'][2]);ins.append(v['posterior'][3])
            signals.append(dict(dataset=ds,teacher=kind,**counts,old_all_support_agree_disagree_unknown=oldall,mean_order_margin_difference=math.fsum(margins)/len(margins)if margins else None,mean_tie_posterior=math.fsum(tie)/len(tie)if tie else None,mean_both_insufficient_posterior=math.fsum(ins)/len(ins)if ins else None,optimizer_steps=sum(u['judges'][kind]['optimizer_steps']for u in use)))
    write(OUT/'SIGNAL_DIAGNOSTICS.json',dict(status='complete',count=256,records=signals,GT_interval_order_only_a_localization_proxy=True))
    def val(ds,arm,k):return next(s['mean']for s in d['summary']if s['dataset']==ds and s['arm']==arm and s['metric']==k)
    lines=['# Joint pairwise Query-Caption consistency, fixed development256','','Same original exposed HC128/Vid128, exact frozen I1/I2 and512 old R1 captions; no new video/caption generation. Same original512 integrated temporal head, merged prior/target, tau1/full-grid trust1, LBFGS50/Armijo and exact marginal MAP. Only text judgment changes. Both Teachers judge identical captions, both display orders, remapped and averaged; A/B directional, C indistinguishable and D both insufficient nondirectional. No GT selection, no confidence/disagreement filtering; old truncated descriptions retain original no-update admission. No Actionness, new Spatial fit or production promotion.','','|Domain|Temporal arm|vIoU %|tIoU %|dense sIoU %|','|---|---|---:|---:|---:|']
    arms=['Frozen','T2','MARGINAL_NO_UPDATE','R1_INDEPENDENT_B3','R1_PAIRWISE','QWEN_PAIRWISE','R1_TARGET_HARD','QWEN_TARGET_HARD']
    for ds in ['hc2','vidstg']:
        for arm in arms:lines.append('|'+ds+'|'+arm+'|'+'|'.join('unknown'if(v:=val(ds,arm,k))is None else f'{100*v:.6f}'for k in ['m_vIoU','m_tIoU','m_sIoU_dense_GT'])+'|')
    lines+=['','Main contrast is each new fitted arm minus MARGINAL_NO_UPDATE, then versus old R1_INDEPENDENT_B3. Frozen also changes the decoder; its difference is not pure adaptation. Full signed per-source paired t95/tails/unknowns retained, no significance-only decision gate. Target-hard reads use the tilted candidate posterior and are not Student upper bounds. Both Teacher arms were fitted regardless of ranking; no offline GT sorting was used online. The comparison tests text judge and task, not new visual understanding. Model posterior is uncalibrated.','', '## Feedback and actual update diagnostics','']
    for s in signals:lines.append(f"{s['dataset']} / {s['teacher']}: {s['informative']}/128 informative, {s['optimizer_steps']} accepted steps; interval-order proxy {s['GT_interval_order_agrees']} agree / {s['GT_interval_order_disagrees']} disagree / {s['GT_or_reward_tied_or_unknown']} tied or unknown. Both original and matched-support old ordering are in SIGNAL_DIAGNOSTICS.json. Order argmax agreement {s['order_argmax_agrees']}/{s['eligible']}; average mapped margin disagreement {s['mean_order_margin_difference']}; tie/insuff posterior {s['mean_tie_posterior']}/{s['mean_both_insufficient_posterior']}.")
    lines+=['','This single revision does not establish semantic truth, independent confirmation, or cross-backbone transfer. Existing512-window semantic panel remains44 model-assisted/468pending. No further prompt, hypothesis, temperature, optimizer or reward-weight search is triggered. Raw query/caption/Teacher responses, media, GT coordinates and model tensors are excluded from the public export.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');(OUT/'PROTOCOL.md').write_bytes((ROOT/'protocols/stvg_pairwise_caption256_v1.md').read_bytes())
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4));aa=['MARGINAL_NO_UPDATE','R1_INDEPENDENT_B3','R1_PAIRWISE','QWEN_PAIRWISE']
    for a,ds in zip(ax,['hc2','vidstg']):a.bar(['No update','Independent R1','Pairwise R1','Pairwise Qwen'],[100*val(ds,x,'m_vIoU')for x in aa]);a.set_title(ds+' all128 / development');a.set_ylabel('vIoU (%)');a.tick_params(axis='x',rotation=15)
    fig.tight_layout();fig.savefig(OUT/'effects.png',dpi=160);plt.close(fig)
    for n in ['stvg_pairwise_caption256_v1.py','continue_stvg_pairwise_caption256_v1.py','report_stvg_pairwise_caption256_v1.py','publish_stvg_pairwise_caption256_v1.py','stvg_temporal_marginal_train_fit256_v1.py','stvg_temporal_marginal_decode_util256_v1.py']:
        p=OUT/'code'/n;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/n).read_bytes()))
    paths={str(p.relative_to(ROOT))for p in OUT.rglob('*')if p.is_file()};verify_privacy(paths);write(BASE/'publication/pairwise256/EXPORT_MANIFEST.json',dict(status='complete',count=256,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]));print('PAIRWISE_REPORT_READY',checks,flush=True)
if __name__=='__main__':report()
