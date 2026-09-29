"""Independent CPU readback for PartA and post-seal oracle evaluation."""
import sys,math,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.stats import rankdata,spearmanr
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_causal_round2_v1 import BASE,OUT,ARMS,verify
from scripts.score_tastvg_evidence_vulnerability_v1 import self_metrics,arr
from scripts.analyze_spatial10_components_v1 import checked_score

METRICS=['sIoU','tIoU','vIoU_corrected','tube_supported_sIoU']

def stats(values):
    v=np.array([x for x in values if x is not None and np.isfinite(x)],float)
    if not len(v):return dict(n=0,mean=None,median=None,ci95=None,min=None,max=None)
    rng=np.random.default_rng(20260929);b=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),ci95=np.quantile(b,[.025,.975]).tolist(),min=float(v.min()),max=float(v.max()))

def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);keep=np.isfinite(x)&np.isfinite(y);x=x[keep];y=y[keep]
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return dict(n=len(x),rho=None,ci95=None,reason='constant_or_insufficient')
    r=float(spearmanr(x,y).statistic);rng=np.random.default_rng(20260929);ii=rng.integers(len(x),size=(10000,len(x)))
    a=rankdata(x[ii],axis=1);b=rankdata(y[ii],axis=1);a-=a.mean(1)[:,None];b-=b.mean(1)[:,None];den=np.sqrt((a*a).sum(1)*(b*b).sum(1));vv=(a*b).sum(1)[den>0]/den[den>0]
    return dict(n=len(x),rho=r,ci95=np.quantile(vv,[.025,.975]).tolist(),valid_bootstraps=len(vv))

def cos(a,b):
    a=np.concatenate([arr(t).reshape(-1) for t in a]);b=np.concatenate([arr(t).reshape(-1) for t in b]);n=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/n) if n else None

def vectors(data,gradient,branch):
    result=[]
    for v,g in zip(data['views'],gradient):
        n=v['info']['fea_map_size'][0]*v['info']['fea_map_size'][1]
        result.append(g[:n] if branch=='S' else g[-n:] if branch=='T' else torch.cat([g[:n],g[-n:]],0))
    return result

def energy(xs):return sum(float((arr(x)**2).sum()) for x in xs)

def rank_energy(xs):
    gram=np.zeros((256,256),float)
    for x in xs:
        m=arr(x).reshape(-1,256);gram+=m.T@m
    e=np.maximum(np.linalg.eigvalsh(gram),0)[::-1];total=e.sum()
    return {str(r):float(e[:r].sum()/total) if total else None for r in (8,16,32,64)}

def audit_causal(p):
    barrier=read(OUT/'CAUSAL_BARRIER.json');assert len(barrier['files'])==64
    results=[];errors=[];pool_errors=[]
    for row in p['rows']:
        stem=row['key'].replace(':','_');f=OUT/'causal'/f'{stem}.pt';assert sha(f)==barrier['files'][str(f.relative_to(OUT))];z=load(f);base=load(BASE/'capture'/f'{stem}.pt');attack=load(BASE/'attack'/f'{stem}_r0.02.pt')['selected']
        for cell in z['rows']:
            stage=cell['stage'];branch=cell['branch'];offsets=[]
            for j,(v,b,n) in enumerate(zip(base['views'],base['evidence'],attack['evidence'])):
                count=v['info']['fea_map_size'][0]*v['info']['fea_map_size'][1];sel=b[f'selected_stage{stage}'];assert sel==n[f'selected_stage{stage}']
                h=arr(v['H']);hp=arr(v['H']+attack['delta'][j]);h=h[:count] if branch=='app' else h[-count:];hp=hp[:count] if branch=='app' else hp[-count:]
                a=arr(b[f'ASA{stage}_{branch}']);ap=arr(n[f'ASA{stage}_{branch}']);qq={}
                for key,aa,hh in [('base',a,h),('A',ap,h),('H',a,hp),('AH',ap,hp)]:
                    q=(hh.transpose(1,0,2)[sel]*aa[:,:,None]).mean((0,1));saved=arr(cell['vectors'][j][key]);rel=float(np.linalg.norm(q-saved)/max(np.linalg.norm(q),1e-30));assert rel<2e-6,(row['key'],key,rel);pool_errors.append(rel);qq[key]=saved
                diffs={k:qq[k]-qq['base'] for k in ('A','H','AH')};effects={}
                for k,d in diffs.items():
                    q=qq[k];bq=qq['base'];effects[k]=dict(relative_norm=float(np.linalg.norm(d)/max(np.linalg.norm(bq),1e-30)),cosine_drift=float(1-np.dot(q,bq)/max(np.linalg.norm(q)*np.linalg.norm(bq),1e-30)))
                    for name,value in effects[k].items():errors.append(abs(value-cell['decomposition'][j]['effects'][k][name]))
                ratio=float(np.linalg.norm(diffs['AH'])/max(np.linalg.norm(diffs['A'])+np.linalg.norm(diffs['H']),1e-30))
                errors.append(abs(ratio-cell['decomposition'][j]['cancellation_ratio']));offsets.append(dict(effects=effects,cancellation_ratio=ratio))
            conditions={}
            for key,c in cell['conditions'].items():
                pm=self_metrics(cell['baseline_stage_prediction'],c['prediction'])
                for name,value in pm.items():errors.append(abs(float(value)-float(c['preservation'][name])))
                conditions[key]=dict(decoder=pm)
                if stage==1:
                    pm=self_metrics(base['prediction'],c['final_routed_prediction'])
                    for name,value in pm.items():errors.append(abs(float(value)-float(c['final_routed_preservation'][name])))
                    conditions[key]['final_routed']=pm
            results.append(dict(key=row['key'],cohort=row['cohort'],stage=stage,branch=branch,offsets=offsets,conditions=conditions))
    assert max(errors,default=0)<2e-5,max(errors)
    out={}
    for cohort in ('hcstvg1_test','vidstg_test','all'):
        for stage in (1,2):
            for branch in ('app','motion'):
                rr=[r for r in results if r['stage']==stage and r['branch']==branch and (cohort=='all' or r['cohort']==cohort)]
                x=dict(eligible_queries=len(rr),decomposition={},decoder={},cancellation_ratio=stats([np.mean([o['cancellation_ratio'] for o in r['offsets']]) for r in rr]))
                for mode in ('A','H','AH'):
                    x['decomposition'][mode]={m:stats([np.mean([o['effects'][mode][m] for o in r['offsets']]) for r in rr]) for m in ('relative_norm','cosine_drift')}
                    x['decoder'][mode]=dict(self_vIoU=stats([r['conditions'][mode]['decoder']['self_vIoU'] for r in rr]),interval_changed=sum(not r['conditions'][mode]['decoder']['interval_exact'] for r in rr))
                    if stage==1:x['decoder'][mode]['final_routed']=dict(self_vIoU=stats([r['conditions'][mode]['final_routed']['self_vIoU'] for r in rr]),interval_changed=sum(not r['conditions'][mode]['final_routed']['interval_exact'] for r in rr))
                out[f'{cohort}/stage{stage}/{branch}']=x
    write(OUT/'analysis/PARTA_ROWS.json',results);write(OUT/'analysis/PARTA_SUMMARY.json',out);write(OUT/'analysis/PARTA_AUDIT.json',dict(status='pass',queries=64,metric_comparisons=len(errors),max_metric_error=max(errors),pooling_checks=len(pool_errors),max_pooling_relative_error=max(pool_errors),GT_read=False))
    print('PartA independently read back',len(results),'eligible stage-branch cells')

def run():
    torch.set_num_threads(4);p=verify();barrier=read(OUT/'ORACLE_BARRIER.json');assert len(barrier['files'])==448
    for f,h in barrier['files'].items():assert sha(OUT/f)==h
    gt=read(OUT/'GT_SUBSET.json');vv={r['key']:r for r in read(OUT/'VULNERABILITY_LOCK.json')['rows']};rows=[];checks=[];reinsert=0;grad_contracts=[]
    for row in p['rows']:
        key=row['key'];stem=key.replace(':','_');base=load(BASE/'capture'/f'{stem}.pt');z=load(OUT/'oracle'/f'{stem}.pt');truth=gt[key];valid=np.asarray(truth['valid'],bool);ids=base['frame_ids']
        def score(pred):
            m,q=checked_score(pred['boxes'],truth,ids,pred['indices']);selected=np.zeros(len(ids),bool);l,r=pred['indices'];selected[l:r+1]=True
            m['tube_supported_sIoU']=float(q[valid&selected].sum()/valid.sum()) if valid.any() else None
            m['valid_GT_frames_excluded']=int((valid&~selected).sum());return m
        baseline=score(base['prediction']);rr=dict(key=key,cohort=row['cohort'],source=row['source'],VA=vv[key]['VA'],VAmean=vv[key]['VAmean'],baseline=baseline,arms={},swapped={},gradient={},target_exceptions=z['initial']['target_exceptions'])
        if z['native_contract']:grad_contracts.append(z['native_contract'])
        for arm in ARMS:
            a=load(z['arms'][arm]['path']);assert sha(z['arms'][arm]['path'])==z['arms'][arm]['sha256'];assert a['all_parameters_frozen'];assert a['backwards']==10
            n2=0
            for v,d in zip(base['views'],a['delta']):
                count=v['info']['fea_map_size'][0]*v['info']['fea_map_size'][1];assert not torch.count_nonzero(d[count:-count]);n2+=energy([d])
                if arm.startswith('OS') and arm not in ('OST','Oselective'):assert not torch.count_nonzero(d[-count:])
                if arm.startswith('OT'):assert not torch.count_nonzero(d[:count])
            assert math.sqrt(n2)<=a['cap']*(1+1e-6)
            last_loss=z['initial']['losses'];accepted=0;trialcount=0;descent_rejections=0;preservation_rejections=0;loss0=z['initial']['losses'];check_initial=True
            phase_ref=base['prediction'];current_pred=base['prediction']
            for step in a['path']:
                if arm=='Oselective' and step['step']==5:phase_ref=current_pred
                for k in ('S','T','ST'):assert abs(step['current_loss'][k]-last_loss[k])<2e-5
                for trial in step['trials']:
                    trialcount+=1;pm=self_metrics(phase_ref,trial['prediction'])
                    for k,v in pm.items():checks.append(abs(float(v)-float(trial['preservation'][k])))
                    task=step['task'];tol=1e-8*max(1,abs(step['current_loss'][task]));descent=trial['loss'][task]<step['current_loss'][task]-tol
                    if arm=='Oselective':descent=descent and trial['loss']['ST']<step['current_loss']['ST']-1e-8*max(1,abs(step['current_loss']['ST']))
                    preserved=True if step['protect'] is None else pm['interval_exact'] if step['protect']=='T' else pm['mean_native_interval_box_iou']>=.95
                    assert bool(descent)==trial['descent'] and bool(preserved)==trial['preserved'] and bool(descent and preserved)==trial['accepted']
                    descent_rejections+=not descent;preservation_rejections+=not preserved
                    if trial['accepted']:assert trial is step['trials'][-1];last_loss=trial['loss'];current_pred=trial['prediction'];accepted+=1
                assert (step['accepted_alpha']>0)==step['trials'][-1]['accepted']
            for k in ('S','T','ST'):assert abs(last_loss[k]-a['loss'][k])<2e-5
            assert torch.equal(a['prediction']['boxes'],current_pred['boxes'])
            assert a['prediction']['physical_interval']==current_pred['physical_interval']
            if a.get('reinsertion'):assert a['reinsertion']['full_pipeline_exact'];reinsert+=1
            rr['arms'][arm]=dict(metrics=score(a['prediction']),preservation=self_metrics(base['prediction'],a['prediction']),loss=a['loss'],initial_loss=loss0,accepted=accepted,no_op_steps=10-accepted,trial_count=trialcount,loss_rejections=descent_rejections,preservation_rejections=preservation_rejections,relative_norm=math.sqrt(n2)/a['stock_visual_norm'],first_step_metrics=score(a['path'][0]['trials'][-1]['prediction']) if a['path'][0]['accepted_alpha'] else baseline)
        for task,s in z['swapped_probes'].items():rr['swapped'][task]=dict(metrics=score(s['prediction']),loss=s['loss'],accepted=any(x['accepted'] for x in s['trials']),scope=s['scope'],gradient_norm=s['gradient_norm'])
        gg=z['initial']['gradients'];attack=load(BASE/'attack'/f"{stem}_r{vv[key]['rho']:g}.pt")
        for task in ('S','T'):
            e={b:energy(vectors(base,gg[task],b)) for b in ('S','T')};rr['gradient'][task]=dict(energy=e,appearance_fraction=e['S']/sum(e.values()) if sum(e.values()) else None,aligned_fraction=e[task]/sum(e.values()) if sum(e.values()) else None,rank_retention=rank_energy(vectors(base,gg[task],task)))
        raw=cos(vectors(base,attack['selected']['delta'],'S'),vectors(base,gg['S'],'S'));rr['gradient']['attack_cos_raw_S']=raw;rr['gradient']['attack_cos_descent_S']=None if raw is None else -raw
        joint=[a+b for a,b in zip(gg['S'],gg['T'])];rr['gradient']['joint_rank_retention']=rank_energy(vectors(base,joint,'ST'))
        rows.append(rr);print('SCORED',len(rows),64,key,flush=True)
    assert max(checks,default=0)<1e-8 and len(grad_contracts)==2 and reinsert==4
    summaries={};assoc={};protection={}
    for cohort in ('hcstvg1_test','vidstg_test','all'):
        rr=[r for r in rows if cohort=='all' or r['cohort']==cohort];s=dict(n=len(rr),B0={m:stats([r['baseline'][m] for r in rr]) for m in METRICS},arms={},gradient={})
        for arm in ARMS:
            a=dict(absolute={},delta={},tails={},retention={},accepted=stats([r['arms'][arm]['accepted'] for r in rr]),all_noop=sum(r['arms'][arm]['accepted']==0 for r in rr),interval_changed=sum(not r['arms'][arm]['preservation']['interval_exact'] for r in rr),self_box_iou=stats([r['arms'][arm]['preservation']['mean_native_interval_box_iou'] for r in rr]),self_vIoU=stats([r['arms'][arm]['preservation']['self_vIoU'] for r in rr]))
            for m in METRICS:
                vals=[r['arms'][arm]['metrics'][m] for r in rr];ds=[None if r['baseline'][m] is None or r['arms'][arm]['metrics'][m] is None else r['arms'][arm]['metrics'][m]-r['baseline'][m] for r in rr]
                a['absolute'][m]=stats(vals);a['delta'][m]=stats(ds);good=[r for r in rr if r['baseline'][m] is not None and r['baseline'][m]>=.5];a['retention'][m]=dict(denominator=len(good),retained=sum(r['arms'][arm]['metrics'][m] is not None and r['arms'][arm]['metrics'][m]>=.5 for r in good));a['tails'][m]=dict(positive=sum(v is not None and v>1e-12 for v in ds),negative=sum(v is not None and v< -1e-12 for v in ds),harm_gt5pp=sum(v is not None and v<-.05 for v in ds),worst=min([v for v in ds if v is not None],default=None))
            s['arms'][arm]=a
        for task in ('S','T'):s['gradient'][task]=dict(aligned_energy=stats([r['gradient'][task]['aligned_fraction'] for r in rr]),rank={rank:stats([r['gradient'][task]['rank_retention'][rank] for r in rr]) for rank in ('8','16','32','64')})
        s['gradient']['joint_rank']={rank:stats([r['gradient']['joint_rank_retention'][rank] for r in rr]) for rank in ('8','16','32','64')}
        s['gradient']['attack_cos_raw_S']=stats([r['gradient']['attack_cos_raw_S'] for r in rr]);s['gradient']['attack_cos_descent_S']=stats([r['gradient']['attack_cos_descent_S'] for r in rr]);summaries[cohort]=s
        correlations={}
        for feature in ('VA','VAmean'):
            x=[r[feature] for r in rr]
            for name,values in [('baseline_spatial_error',[1-r['baseline']['sIoU'] if r['baseline']['sIoU'] is not None else np.nan for r in rr]),('spatial_oracle_gain',[r['arms']['OS']['metrics']['sIoU']-r['baseline']['sIoU'] if r['baseline']['sIoU'] is not None else np.nan for r in rr]),('joint_oracle_gain',[r['arms']['OST']['metrics']['vIoU_corrected']-r['baseline']['vIoU_corrected'] if r['baseline']['vIoU_corrected'] is not None else np.nan for r in rr])]:correlations[feature+'/'+name]=correlation(x,values)
        assoc[cohort]=correlations;ps={}
        for bare,protected,target in [('OS','OS_PT','sIoU'),('OT','OT_PS','tIoU')]:
            valid=[r for r in rr if r['baseline'][target] is not None];bg=np.array([r['arms'][bare]['metrics'][target]-r['baseline'][target] for r in valid]);pg=np.array([r['arms'][protected]['metrics'][target]-r['baseline'][target] for r in valid]);positive=bg>1e-12
            ps[protected]=dict(bare_positive_queries=int(positive.sum()),positive_opportunity_gain=float(bg[positive].sum()),protected_gain_on_same_queries=float(pg[positive].sum()),useful_gain_retention=float(pg[positive].sum()/bg[positive].sum()) if positive.any() else None,paired_target_gain_delta=stats((pg-bg).tolist()),bare_zero_or_negative_queries=int((~positive).sum()))
        for task,aligned,metric in [('S','OS','sIoU'),('T','OT','tIoU')]:
            ps['swapped_'+task]=dict(aligned_first_step_gain=stats([r['arms'][aligned]['first_step_metrics'][metric]-r['baseline'][metric] for r in rr if r['baseline'][metric] is not None]),swapped_one_step_gain=stats([r['swapped'][task]['metrics'][metric]-r['baseline'][metric] for r in rr if r['baseline'][metric] is not None]))
        protection[cohort]=ps
    write(OUT/'analysis/ORACLE_ROWS.json',rows);write(OUT/'analysis/ORACLE_SUMMARY.json',summaries);write(OUT/'analysis/ASSOCIATIONS.json',assoc);write(OUT/'analysis/PRESERVATION.json',protection)
    write(OUT/'analysis/ORACLE_AUDIT.json',dict(status='pass',queries=64,primary_arms_including_B0=7,adaptation_outputs=384,adaptation_backwards=3840,independent_trial_metric_comparisons=len(checks),max_metric_error=max(checks),native_contracts=grad_contracts,full_final_reinsertions=reinsert,all_text_zero=True,all_radii_valid=True,GT_oracle=True,no_unlabeled_efficacy_claim=True))
    print('Round2 oracle readback complete')

if __name__=='__main__':
    torch.set_num_threads(4)
    if len(sys.argv)>1 and sys.argv[1]=='causal':audit_causal(verify())
    else:run()
