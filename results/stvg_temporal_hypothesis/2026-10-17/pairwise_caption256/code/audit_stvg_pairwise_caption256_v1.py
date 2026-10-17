"""Postseal independent feedback/actual-FP32 transfer diagnostics; no new fits."""
from pathlib import Path
import gc,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_pairwise_caption256_v1 import BASE,read,write,sha,verify
OUT=ROOT/'results/stvg_temporal_hypothesis/2026-10-17/pairwise_caption256'
def main():
    import numpy as np,torch
    from scipy.special import logsumexp
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_marginal_train_fit256_v1 as fit
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    verify();assert read(BASE/'CPU_COMPLETION.json')['count']==256;assert read(BASE/'ROOT_SCALAR_READBACK.json')['status']=='pass';d=read(BASE/'POSTSEAL_RESULTS.json');uu={u['case_id']:u for u in d['usage']};rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}'for i,r in enumerate(rr,1)};records=[];checks=0
    for r,pf in zip(rr,read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['files']):
        cid=r['probe_alias'];assert cid==pf['case_id']and sha(ROOT/pf['path'])==pf['sha256'];p=torch.load(ROOT/pf['path'],map_location='cpu',weights_only=False);c,_=load_case(r);off=fit.native.offsets(c,_hidden_from_source_case(c));u=uu[cid]
        for kind in ['R1','QWEN']:
            f=torch.load(BASE/kind/'fits'/(cid+'.pt'),map_location='cpu',weights_only=False)['fit'];pre=fit.prepare(off,f['hypotheses'],f['rewards']);q=pre['q'];p0=pre['Native_conditional'];lw=[]
            zs=p['actual_integrated_head_logits'][kind];scores=[]
            for o,z in zip(off,zs):
                a=z.detach().cpu().numpy()[0].astype(np.float64);scores.append(a[o['ij'][0],0]+a[o['ij'][1],1])
            for event in pre['events']:lw.append(fit.event_logweight(off,scores,event,False)[0])
            lw=np.asarray(lw);pa=np.exp(lw-logsumexp(lw));assert np.max(abs(q-f['Teacher_conditional']))<1e-12 and np.max(abs(p0-f['Native_conditional']))<1e-12;L,g,_=fit.loss_gradient(pre,f['theta'].reshape(-1),'marginal_interval',fit.CFG);assert abs(L-f['path'][-1]['loss'])<1e-10
            valid=not f['no_information'];goal=float(np.log(q[0]/q[1])-np.log(p0[0]/p0[1]));move=float(np.log(pa[0]/pa[1])-np.log(p0[0]/p0[1]));rec=dict(case_id=aliases[cid],dataset=r['dataset'],teacher=kind,no_information=f['no_information'],steps=f['optimizer_steps'],Native_conditional=list(p0),Teacher_conditional=list(q),actual_FP32_conditional=list(pa),target_logodds_change=goal,actual_logodds_change=move,moves_in_Teacher_direction=(goal*move>0)if valid else None,target_fraction=move/goal if valid and goal else None,final_gradient_norm=float(np.linalg.norm(g)),initial_candidate_KL=f['path'][0]['terms']['candidate_kl'],final_candidate_KL=f['path'][-1]['terms']['candidate_kl'],MAP_changed=p['arms'][kind+'_PAIRWISE']['interval']!=p['arms']['MARGINAL_NO_UPDATE']['interval'])
            if f['no_information']:assert f['optimizer_steps']==0 and np.count_nonzero(f['theta'])==0
            records.append(rec);checks+=3
        del c,p,off;gc.collect()
    def sign(v):return int(v>0)-int(v<0)
    summary=[]
    for ds in ['hc2','vidstg']:
        for kind in ['R1','QWEN']:
            use=[v for v in records if v['dataset']==ds and v['teacher']==kind];valid=[v for v in use if not v['no_information']];matched=dict(count=0,new_agree=0,new_disagree=0,old_agree=0,old_disagree=0);orderflips=0
            for u in d['usage']:
                if u['dataset']!=ds:continue
                old=u['old_independent_rewards'];v=u['judges'][kind];new=v['rewards'];hs=u['hypothesis_metrics']
                if len(hs)==2 and all(h['tIoU']is not None for h in hs)and len(old)==len(new)==2 and all(x is not None for x in old+new):
                    g=hs[0]['tIoU']-hs[1]['tIoU'];a=old[0]-old[1];b=new[0]-new[1]
                    if g and a and b:matched['count']+=1;matched['new_agree'if sign(g)==sign(b)else'new_disagree']+=1;matched['old_agree'if sign(g)==sign(a)else'old_disagree']+=1
                op=v['order_posteriors']
                if op:orderflips+=sign(op[0][0]-op[0][1])!=sign(op[1][1]-op[1][0])
            summary.append(dict(dataset=ds,teacher=kind,total=128,informative=len(valid),actual_optimized=sum(v['steps']>0 for v in use),accepted_steps=sum(v['steps']for v in use),actual_FP32_moves_in_Teacher_direction=sum(v['moves_in_Teacher_direction']for v in valid),median_target_logodds_fraction=float(np.median([v['target_fraction']for v in valid if v['target_fraction'] is not None]))if any(v['target_fraction'] is not None for v in valid)else None,mean_initial_candidate_KL=math.fsum(v['initial_candidate_KL']for v in use)/128,mean_final_candidate_KL=math.fsum(v['final_candidate_KL']for v in use)/128,MAP_changed=sum(v['MAP_changed']for v in use),display_order_direction_flips=orderflips,matched_GT_interval_order_support=matched))
    result=dict(status='pass',count=256,checks=checks,records=records,summary=summary,new_model_calls=0,new_fits=0,GT_read=True,GT_order_only_proxy=True);write(BASE/'ROOT_TRANSFER_READBACK.json',result);write(OUT/'TRANSFER_DIAGNOSTICS.json',result)
    # Raw reward choice is distinct from the prior-tilted target and Student MAP.
    # Derive only from already postseal-scored hypothesis metrics, no GT routing or new inference.
    idx={(x['case_id'],x['arm']):x['metrics']for x in d['rows']};rawrows=[]
    for u in d['usage']:
        cid=u['case_id'];ds=u['dataset'];base=idx[cid,'MARGINAL_NO_UPDATE']
        rawrows.append(dict(case_id=aliases[cid],dataset=ds,arm='MARGINAL_NO_UPDATE',metrics={k:base[k]for k in ['m_vIoU','m_tIoU']}))
        for kind in ['R1','QWEN']:
            v=u['judges'][kind];hm=u['hypothesis_metrics'];i=int(np.argmax(v['rewards']))if not v['no_information']else None
            m={k:base[k]for k in ['m_vIoU','m_tIoU']}if i is None else dict(m_vIoU=hm[i]['vIoU'],m_tIoU=hm[i]['tIoU'])
            rawrows.append(dict(case_id=aliases[cid],dataset=ds,arm=kind+'_RAW_REWARD_HARD',metrics=m))
    rs,rp=subset_aggregate(rawrows);write(OUT/'RAW_REWARD_READOUT.json',dict(status='complete',count=256,rows=rawrows,summary=rs,paired=rp,derived_from_already_scored_fixed_hypotheses=True,no_information_Source_marginal_noop=True,new_GT_metric_calls=0,new_model_calls=0,new_fits=0,not_Student_upper_bound=True))
    (OUT/'code'/Path(__file__).name).write_text(public_source_text(Path(__file__).read_bytes()))
    recovery=BASE/'recovery/readout002'
    if recovery.exists():
        old=read(recovery/'PARTIAL_REUSE_LOCK.json');fresh=list((recovery/'fresh_prefix').glob('*.json'))
        assert old['count']==49 and len(fresh)==2 and all(read(p)['status']=='pass'for p in fresh)
        write(OUT/'READOUT_RECOVERY.json',dict(status='pass',count=256,partial49_reused_byte_exact=True,remaining207_readouts_only=True,known_retired_Source_prefixes_rebuilt=2,original_pixels_and_full_Source_logits_hidden_boxes_exact=True,Teacher_or_optimizer_rerun=False,deadline_reset=False,original_failure_and_cost_preserved=True))
        for name in ['stvg_pairwise_caption256_recovery002.py','continue_stvg_pairwise_caption256_recovery002.py']:(OUT/'code'/name).write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
    p=OUT/'REPORT.md';lines=['\n## Independent actual-output transfer audit\n']
    for s in summary:lines.append(str(s))
    lines+=['\nRAW_REWARD_READOUT.json separately reports direct raw preference selection versus Source marginal, derived solely from already scored fixed-hypothesis metrics. It is distinct from prior-tilted TARGET_HARD and actual Student adaptation; none is an upper bound. Readout recovery preserved49 predictions and all992 Teacher prefills/all fits; only207 remaining readouts resumed. Two previously known retired Source prefixes were recomputed from exact original pixels and checked against saved records/logits/hidden/boxes; no Teacher or optimizer was retried, original deadline/failure/cost retained.']
    p.write_text(p.read_text()+'\n'.join(lines)+'\n')
    manifest=BASE/'publication/pairwise256/EXPORT_MANIFEST.json';j=read(manifest);paths={str(p.relative_to(ROOT))for p in OUT.rglob('*')if p.is_file()};verify_privacy(paths);j.update(paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]);write(manifest,j,mutable=True);print('PAIRWISE_TRANSFER_AUDIT',checks,flush=True)
if __name__=='__main__':main()
