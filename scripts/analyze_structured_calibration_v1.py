"""Independent F44 arithmetic/state/score audit and complete factorial report.

No fitting, no configuration changes, no GT-dependent online choices.
"""
import collections
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.special import expit,log_expit,logsumexp
from sklearn.metrics import roc_auc_score
from scripts.decota_matrix_common_v1 import read,write,status,load,sha
from scripts.structured_temporal_calibration_v1 import OUT,PARENT,plan,cache_for,recorded,config_id,METRICS,choose
from scripts.analyze_structured_temporal_v1 import independent_loss,change
from scripts.analyze_spatial10_components_v1 import labels_for,checked_score,summary,aggregate
from scripts.audit_spatial4_attribute_v1 import aggregate_check
from vg_tta.dense_support_tuning_v1 import state_hash


def independent_evidence(logits,teacher,records,cfg):
    offsets=[]
    for z,t,r in zip(logits,teacher,records):
        x=t['raw_logits'].numpy().reshape(-1).astype(float)
        median=np.median(x);mad=np.median(abs(x-median))
        zz=(x-cfg['center_fraction']*median)/(mad+cfg['epsilon'])
        la,lb=log_expit(zz),log_expit(-zz)
        f=np.asarray(r['frame_ids'],float);edges=np.r_[f[0],(f[:-1]+f[1:])/2,f[-1]+1]
        w=np.diff(edges);w/=w.sum()
        z=z.numpy().reshape(-1,2).astype(float)
        s,e=np.triu_indices(len(z),1);raw=z[s,0]+z[e,1];lp=raw-logsumexp(raw)
        inside=(np.arange(len(f))[None]>=s[:,None])&(np.arange(len(f))[None]<=e[:,None])
        cost=-(w[None]*np.where(inside,la[None],lb[None])).sum(1)
        g=-cost+cfg['prior_weight']*lp;top=int(g.argmax())
        offsets.append(dict(x=x,z=zz,a=expit(zz),w=w,edges=edges,center=median,mad=mad,
            lp=lp,cost=cost,g=g,s=s,e=e,top=top,interval=[int(f[s[top]]),int(f[e[top]])+1]))
    return offsets


def stats(rows,arms,contrasts,counts):
    s=summary(rows,arms,METRICS,contrasts);n,error=aggregate_check(rows,s)
    counts['statistic_cells']+=n
    assert error<1e-12
    return s


def main():
    torch.set_num_threads(2)
    p=plan();manifest=[r for rr in p['rows'].values() for r in rr];bykey={r['key']:r for r in manifest}
    data=read(OUT/'ALL_SOURCE_RESULTS.json');rows=data['rows'];assert len(rows)==113
    assert len({r['key'] for r in rows})==len({r['group'] for r in rows})==113
    gt=labels_for(manifest);caches={r['key']:cache_for(r) for r in manifest}
    for c,rr in p['rows'].items():
        a=[r for r in rr if r['f44_role']=='development'];b=[r for r in rr if r['f44_role']=='second_validation']
        assert len(a)==32 and len(b)==(25 if c=='hcstvg1_test' else 24)
        for field in ['source','group']:assert not {r[field] for r in a}&{r[field] for r in b}
        assert not {r['input']['video_sha256'] for r in a}&{r['input']['video_sha256'] for r in b}
    counts=collections.Counter();maxloss=0.;maxproj=0.;fit_details={};old={r['key']:r for r in read(PARENT/'ALL_SOURCE_RESULTS.json')['rows']}
    allfiles=sorted((OUT/'fits').rglob('*.pt'))+sorted((OUT/'noops').rglob('*.pt'))
    for path in allfiles:
        assert recorded(path);tr=load(path);cache=caches[tr['key']]
        teacher=cache['wrong_teacher'] if tr['kind']=='wrong_query' else cache['rolled_teacher'] if tr['kind']=='time_shift' else cache['teacher']
        ev=independent_evidence(cache['zero']['logits'],teacher,cache['records'],tr['config'])
        assert 'rho' not in tr['evidence'] and 'native_delta' not in tr['config']
        for a,t in zip(ev,tr['evidence']['offsets']):
            err=max(np.max(abs(a['cost']-t['cost'].numpy())),np.max(abs(a['g']-t['score'].numpy())))
            maxproj=max(maxproj,float(err));assert err<1e-8,(path,err)
            assert a['top']==t['target'] and a['interval']==t['interval']
            assert np.max(abs(a['a']-t['a'].numpy()))<1e-12
            assert np.max(abs(a['w']-t['omega'].numpy()))<1e-12
            assert len(t['raw_logits'])==t['valid_positions'] and not t['a'].requires_grad
        for i,st in enumerate(tr['path']):
            lv,nll,hinge,kl=independent_loss(st['logits'],ev,tr['config'],1.,1.)
            err=max(abs(lv-st['loss']),abs(nll-st['parts']['nll']),abs(hinge-st['parts']['hinge']),abs(kl-st['parts']['kl']))
            maxloss=max(maxloss,float(err));assert err<1e-8,(path,i,err)
            assert not any(k in st for k in ['guard','decision_score','native_drops'])
            if i and not tr['path'][i-1]['accepted']:
                assert st['state_sha256']==tr['path'][i-1]['state_sha256']
                assert all(torch.equal(a,b) for a,b in zip(st['logits'],tr['path'][i-1]['logits']))
            if i<tr['steps']:
                assert not st['unused_parameters'] and math.isfinite(st['gradient_norm'])
                assert st['optimizer_restored_on_reject']==(not st['accepted'])
            counts['loss_states']+=1
        chosen=0
        for i,st in enumerate(tr['path'][1:],1):
            if st['loss']<tr['path'][chosen]['loss']-1e-12:chosen=i
        assert chosen==tr['best_step']
        assert state_hash(tr['state'])==tr['path'][chosen]['state_sha256']
        assert tr['final']['physical_interval']==tr['path'][chosen]['physical_interval']
        assert torch.equal(tr['final']['boxes'],cache['T0']['boxes'])
        if tr['kind']=='P2':
            assert not tr['parameter_TTA'] and tr['parameters']==sum(z.numel() for z in cache['zero']['logits'])
            assert all(torch.equal(tr['state']['output.'+str(i)],z) for i,z in enumerate(tr['final']['logits']))
        else:assert tr['parameters']==66306 and tr['parameter_TTA']
        assert tr['source_restored'] and tr['teacher_frozen'] and not tr['GT_online']
        if tr['role']=='noop':
            assert all(x['state_delta']==0 for x in tr['path']);counts['noops']+=1
        else:counts[tr['kind']+'_fits']+=1
        counts['fits']+=1;counts['backwards']+=tr['backwards'];counts['fit_milliseconds']+=round(tr['seconds']*1000)
        fit_details[str(path)]=dict(key=tr['key'],kind=tr['kind'],role=tr['role'],config=tr['config'],
            accepted_steps=sum(x.get('accepted',False) for x in tr['path']),best_step=tr['best_step'],
            state_delta=tr['state_delta'],loss_reduction=tr['path'][0]['loss']-tr['final']['loss'],
            remaining_hinge=tr['final']['parts']['hinge'],final_kl=tr['final']['parts']['kl'],
            both_raw_offsets_match_teacher=all(t['target_indices']==pair for t,pair in zip(tr['evidence']['offsets'],tr['final']['raw_offset_indices'])),
            both_teacher_offsets_native=all(t['target_indices']==t['native_indices'] for t in tr['evidence']['offsets']))
        # Match the entire five-step old B trajectory, not just its metric.
        if tr['role']!='noop' and tr['kind']=='P1' and tr['config']['center_fraction']==tr['config']['prior_weight']==1.:
            ref=old[tr['key']]['fit_references']['B_match'];prior=load(ref['path'])
            for a,b in zip(tr['path'],prior['path'][:6]):
                assert a['state_sha256']==b['state_sha256'] and a['loss']==b['loss']
                assert a['indices']==b['indices']
                # Old dev trajectories continue to30; their step5 records contain
                # a proposal toward step6, which is outside this five-step run.
                if a['step']<tr['steps']:assert a.get('trials')==b.get('trials')
            assert prior['prefixes']['5']['loss']['best_step']==tr['best_step']
            counts['F43_B_match_exact_trajectories']+=1
    assert counts['fits']==1364 and counts['noops']==8 and counts['F43_B_match_exact_trajectories']==113,counts
    assert counts['fits']<=p['caps']['fits'] and counts['backwards']<=p['caps']['backwards']
    grid={}
    for r in rows:
        cache=caches[r['key']];grid[r['key']]=read(r['grid_reference'])['cells']
        assert len(grid[r['key']])==9
        for arm,interval in r['intervals'].items():
            ii=[cache['frame_ids'].index(interval[0]),cache['frame_ids'].index(interval[1]-1)]
            m,_=checked_score(cache['T0']['boxes'],gt[r['key']],cache['frame_ids'],ii)
            for k in METRICS:assert m[k]==r['arms'][arm][k],(r['key'],arm,k)
            counts['rescored_conditions']+=1
        for cell in grid[r['key']].values():
            for name in ['P0','P1']:
                interval=cell[name+'_interval'];ids=cache['frame_ids'];ii=[ids.index(interval[0]),ids.index(interval[1]-1)]
                m,_=checked_score(cache['T0']['boxes'],gt[r['key']],ids,ii)
                for k in METRICS:assert m[k]==cell[name][k]
                counts['rescored_conditions']+=1
        if r['role']=='second_validation':
            full=read(OUT/'full_replay'/(r['key'].replace(':','_')+'.json'))
            assert set(full['audits'])=={'P1','wrong_query','time_shift'}
            for a in full['audits'].values():
                assert a['full_model'] and a['source_restored'] and a['box_max_error']==a['logit_max_error']==0
                counts['full_model_replays']+=1
    assert counts['full_model_replays']==147
    selection=read(OUT/'SELECTION.json')
    for c in p['rows']:
        dev=read(OUT/'development'/f'{c}.json')
        assert choose(dev)['config']==selection['choices'][c]['config']
        for cell in dev.values():
            assert len(cell['rows'])==32
            assert abs(cell['mean_delta_v']-np.mean([x['delta_v'] for x in cell['rows']]))<1e-12
            for x in cell['rows']:assert bykey[x['key']]['f44_role']=='development'
    names=list(rows[0]['arms']);comparisons=[(n,'T0') for n in names if n!='T0']+[
        ('P1','P0'),('P1','P2'),('P1','wrong_query'),('P1','time_shift'),('P1','F43_C'),('P1','F43_B_match')]
    sums={};grids={};factors={};behavior={};cases=[]
    for cohort in p['rows']:
        sums[cohort]={};grids[cohort]={};factors[cohort]={};behavior[cohort]={}
        for role in ['development','second_validation']:
            rr=[r for r in rows if r['cohort']==cohort and r['role']==role]
            sums[cohort][role]=stats(rr,names,comparisons,counts)
            grids[cohort][role]={};factors[cohort][role]={};behavior[cohort][role]={}
            for cfg in p['menu']:
                cid=config_id(cfg)
                sr=[{**r,'arms':{n:grid[r['key']][cid][n] for n in ['T0','P0','P1']}} for r in rr]
                grids[cohort][role][cid]=dict(config=cfg,**stats(sr,['T0','P0','P1'],[('P1','T0'),('P0','T0'),('P1','P0')],counts))
            # Each contrast keeps one row per source, including interaction DD.
            for method in ['P0','P1']:
                effects={}
                def vector(a,l):return np.array([grid[r['key']][config_id(dict(center_fraction=a,prior_weight=l))][method]['vIoU_corrected'] for r in rr])
                def add(label,v):effects[label]=aggregate(v.tolist(),[r['group'] for r in rr])
                for l in [.1,.3,1.]:
                    for a in [0.,.5]:add(f'alpha{a}-alpha1_at_lambda{l}',vector(a,l)-vector(1.,l))
                for a in [0.,.5,1.]:
                    for l in [.1,.3]:add(f'lambda{l}-lambda1_at_alpha{a}',vector(a,l)-vector(a,1.))
                for a in [0.,.5]:
                    for l in [.1,.3]:add(f'interaction_alpha{a}_lambda{l}',(vector(a,l)-vector(1.,l))-(vector(a,1.)-vector(1.,1.)))
                factors[cohort][role][method]=effects
            for name in names:
                behavior[cohort][role][name]=dict(sources=len(rr),
                    interval_changes=dict(collections.Counter(change(r['intervals']['T0'],r['intervals'][name]) for r in rr)),
                    parameters_changed=sum(r['fit_references'].get(name,{}).get('state_delta',0)>0 for r in rr),
                    equal_direct=sum(r['intervals'][name]==r['intervals']['P0'] for r in rr),
                    different_direct_examples=[dict(key=r['key'],delta_v_pp=100*(r['arms'][name]['vIoU_corrected']-r['arms']['P0']['vIoU_corrected']))
                                               for r in rr if r['intervals'][name]!=r['intervals']['P0']])
                if name in ['P1','P2','wrong_query','time_shift']:
                    behavior[cohort][role][name]['both_raw_offsets_match_own_teacher']=sum(
                        fit_details[r['fit_references'][name]['path']]['both_raw_offsets_match_teacher'] for r in rr)
        for r in [r for r in rows if r['cohort']==cohort]:
            cache=caches[r['key']];event=gt[r['key']]['interval'];details={}
            for cid,cell in grid[r['key']].items():
                tr=load(cell['fit']);offsets=[]
                for t in tr['evidence']['offsets']:
                    f=np.array(t['frame_ids']);ij=t['ij'].numpy();distance=abs(f[ij[0]]-event[0])+abs(f[ij[1]]+1-event[1]);near=int(distance.argmin())
                    y=(f>=event[0])&(f<event[1]);auc=float(roc_auc_score(y,t['raw_logits'].numpy())) if len(set(y))==2 else None
                    native=t['native'];top=t['target'];w=t['omega'].numpy()
                    offsets.append(dict(native_interval=t['native_interval'],projected=t['interval'],
                        nearest_GT_grid=[int(f[ij[0,near]]),int(f[ij[1,near]])+1],actionness_auc=auc,
                        mean_raw=float(np.mean(expit(t['raw_logits'].numpy()))),mean_normalized=float(t['a'].mean()),
                        time_weighted_raw=float(w@expit(t['raw_logits'].numpy())),time_weighted_normalized=float(w@t['a'].numpy()),
                        median=t['median'],mad=t['mad'],D_native=float(t['cost'][native]),D_target=float(t['cost'][top]),D_nearGT=float(t['cost'][near]),
                        G_native=float(t['score'][native]),G_target=float(t['score'][top]),G_nearGT=float(t['score'][near]),
                        weighted_prior_penalty_nearGT=float(tr['config']['prior_weight']*(t['logp0'][native]-t['logp0'][near])),
                        support_saturation_fraction=float(((t['a']<1e-6)|(t['a']>1-1e-6)).double().mean())))
                details[cid]=dict(config=cell['config'],P0=cell['P0'],P1=cell['P1'],
                    P0_interval=cell['P0_interval'],P1_interval=cell['P1_interval'],offsets=offsets,
                    parameter_delta=tr['state_delta'],best_step=tr['best_step'])
            tr=load(r['fit_references']['P1']['path']);states=[]
            for st in tr['path']:
                m,_=checked_score(cache['T0']['boxes'],gt[r['key']],cache['frame_ids'],st['indices'])
                states.append(dict(step=st['step'],loss=st['loss'],interval=st['physical_interval'],metrics=m,
                    state_delta=st['state_delta'],selected=st['step']==tr['best_step']))
            cases.append(dict(**r,delta_v=r['arms']['P1']['vIoU_corrected']-r['arms']['T0']['vIoU_corrected'],
                event_fraction=(event[1]-event[0])/(r['frame_ids'][-1]+1-r['frame_ids'][0]),
                selected_path=states,all_nine_costs=details,nearest_GT_only_offline=True))
    lengths={}
    for cohort in p['rows']:
        lengths[cohort]={}
        def group(x):return 'short_0_.2' if x<=.2 else 'middle_.2_.6' if x<=.6 else 'long_.6_1' if x<=1 else 'longer_than_observed_grid'
        for label in ['short_0_.2','middle_.2_.6','long_.6_1','longer_than_observed_grid']:
            keys={r['key'] for r in cases if r['cohort']==cohort and r['role']=='second_validation' and group(r['event_fraction'])==label}
            rr=[r for r in rows if r['key'] in keys]
            if rr:lengths[cohort][label]=stats(rr,['T0','P0','P1','P2'],[('P1','T0'),('P1','P0')],counts)
    write(OUT/'SUMMARY.json',sums);write(OUT/'GRID_RESULTS.json',grids);write(OUT/'FACTOR_EFFECTS.json',factors)
    write(OUT/'BEHAVIOR.json',behavior);write(OUT/'LENGTH_GROUPS.json',lengths)
    write(OUT/'FIT_DIAGNOSTICS.json',fit_details)
    write(OUT/'CASE_ANALYSIS.json',dict(cases=sorted(cases,key=lambda r:r['delta_v']),all_sources_kept=True,GT_online=False))
    for f,h in p['protected_pins'].items():assert sha(ROOT/f)==h,f
    tests=['tests/test_structured_calibration_driver_v1.py','tests/test_structured_temporal_calibration_v1.py','tests/test_structured_temporal_v1.py',
           'tests/test_dense_support_temporal_v1.py','tests/test_dense_support_tuning_v1.py']
    result=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','-m','pytest','-q',*tests],cwd=ROOT,capture_output=True,text=True)
    write(OUT/'TEST_RECEIPT.json',dict(command=result.args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode==0,result.stdout+result.stderr
    audit=dict(status='pass',counts=dict(counts),max_independent_projection_error=maxproj,max_independent_loss_error=maxloss,
        all_source_scoring_checked=True,source_split_checked=True,teacher_frozen=True,GT_online=False,
        protected_files=len(p['protected_pins']),spatial_changed=False,full_model_max_box_error=0,full_model_max_logit_error=0,
        actual_new_DINO=0,actual_spatial_fits=0,registries_changed=False,independent_test=False)
    write(OUT/'AUDIT.json',audit)
    write(OUT/'CONFIG.json',dict(fixed=p['fixed'],numerical=p['numerical'],selection=selection,spatial=p['spatial']))
    report=['# F44：接口校准3×3与参数价值——全部有界实验完成','',
        'TA-STVG，两源checkpoint，精确A4空间状态只读。每方向32开发，HC25/Vid24第二级验证，113历史来源/查询；没有独立test。九条件全部保留，最终配置仅开发选择，第二级验证没有重选。','']
    for cohort in p['rows']:
        report+=['## '+cohort,'',f"开发锁：`{selection['choices'][cohort]}`。",'']
        for role in ['development','second_validation']:
            report+=['### '+role+'：九条件','',
                '| α | λ | P0 vIoU % | P1 vIoU % | P1−T0 pp | 95% CI pp | P1−P0 pp |',
                '|---:|---:|---:|---:|---:|---|---:|']
            for g in grids[cohort][role].values():
                cfg=g['config'];d=g['contrasts']['P1 - T0']['vIoU_corrected'];di=g['contrasts']['P1 - P0']['vIoU_corrected']
                report.append(f"| {cfg['center_fraction']} | {cfg['prior_weight']} | {100*g['arms']['P0']['vIoU_corrected']['mean']:.3f} | {100*g['arms']['P1']['vIoU_corrected']['mean']:.3f} | {100*d['mean']:+.3f} | [{100*d['ci95'][0]:+.3f},{100*d['ci95'][1]:+.3f}] | {100*di['mean']:+.3f} |")
            report+=['','### '+role+'：同选定teacher的完整系统','',
                '| 条件 | vIoU % | 固定sIoU % | tIoU % | Δv pp | 95% CI pp |','|---|---:|---:|---:|---:|---|']
            s=sums[cohort][role]
            for name in names:
                vals=[100*s['arms'][name][k]['mean'] for k in METRICS[:3]]
                d=s['contrasts'][name+' - T0']['vIoU_corrected'] if name!='T0' else dict(mean=0,ci95=[0,0])
                report.append(f"| {name} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {100*d['mean']:+.3f} | [{100*d['ci95'][0]:+.3f},{100*d['ci95'][1]:+.3f}] |")
            report+=['',f"实际P1变化：`{behavior[cohort][role]['P1']}`。",'']
    report+=['## 核验与范围','',f'实际审计：`{audit}`。',
        '', 'P2是自由logit而非网络参数TTA；相同数值LR不意味着同等函数步幅，未额外为P2调参。P0使用P1开发所选teacher，参数使用比较条件于该teacher，不是各算法独立大调参后的最优比较。',
        'CI为10000来源配对bootstrap/20260914，无搜索多重校正。中性.1pp，所有来源/负例/no-op保留。空间s为固定合法GT帧集合，时间改区间不改变该s。',
        '只读父/A4与CURRENT保持；本轮机制和资源裁决见CLAIMS.md，不由均值自动推广到SOTA/全量/所有时间TTA。']
    (OUT/'TEMPORAL_RESULTS.md').write_text('\n'.join(report)+'\n')
    status(OUT/'STATUS.json',dict(stage='audited_pending_interpretation',finished=False))
    print(json.dumps(dict(audit=audit,validation={c:{n:100*x['vIoU_corrected']['mean'] for n,x in roles['second_validation']['arms'].items()} for c,roles in sums.items()}),indent=2),flush=True)


if __name__=='__main__':main()
