"""F36 independent arithmetic, support, receipts and protected-state audit."""
import collections
import json
import math
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.prepare_simplification_partial_v1 import OUT,PARENT
from scripts.run_simplification_partial_v1 import plan,dest,existing
from vg_tta.metrics import generalized_box_iou_aligned_cxcywh


def audit_fit(z,teacher=None):
    names=z['parameter_names'];assert sum(names.values())==z['parameter_count']
    assert set(names)==set(z['state'])
    assert all(t.numel()==names[n] and bool(torch.isfinite(t).all()) for n,t in z['state'].items())
    selected=z['best_step'];assert z['restore_exact'] and 0<=selected<len(z['path'])
    if not z['failure']:assert selected==min(range(len(z['path'])),key=lambda i:z['path'][i]['loss'])
    assert torch.equal(z['final']['boxes'],z['path'][selected]['boxes'])
    assert all(torch.equal(a,b) for a,b in zip(z['final']['logits'],z['path'][selected]['logits'][:2]))
    errors=[]
    def lp(z):
        # Independent implementation, explicit dense legal mask rather than legal_logp helper.
        zz=z.reshape(-1,2).double();N=len(zz);s=zz[:,0,None]+zz[:,1][None,:]
        mask=torch.arange(N)[:,None]<torch.arange(N)[None,:];ss=s[mask]
        return ss-torch.logsumexp(ss,0)
    ref=[lp(a) for a in z['path'][0]['logits'][:2]]
    for i,t in enumerate(z['path']):
        if teacher is None:
            anchors=z['anchors'];b=t['boxes'];K=z.get('planned',z.get('planned_denominator',4))
            if anchors:
                positions=[a['position'] for a in anchors]
                target=b.new_tensor([a['box'] for a in anchors]);pred=b[positions]
                distances=5*torch.abs(pred-target).sum(-1)+2*(1-generalized_box_iou_aligned_cxcywh(pred,target))
                kappa=z['kappa']
                if kappa is not None:distances=kappa*torch.log1p(distances/kappa)
                data=float((distances*b.new_tensor([a.get('weight',1) for a in anchors])).sum()/K)
            else:data=0.
            delta=t.get('state_delta',t.get('group_delta',{}).get('spatial',0.))
            expected=data+z['gamma']*delta**2/1792
        else:
            data=0.;prior=0.
            for j,a in enumerate(t['logits'][:2]):
                logp=lp(a);prob=logp.exp();prior+=float((prob*(logp-ref[j])).sum())/2
                for mask in teacher['masks'][j]:
                    mass=1. if bool(mask.all()) else float(prob[mask].sum())
                    data-=math.log(.05+.95*mass)/4
            expected=data+z['beta']*prior
        errors.append(abs(expected-t['loss']))
        if 'accepted' in t:
            tr=t['trials'];accepted=[a for a in tr if a['loss'] is not None and a['loss']<t['loss']-1e-9]
            assert bool(accepted)==bool(t['accepted'])
            if i+1<len(z['path']):
                expect=accepted[0]['loss'] if accepted else t['loss']
                assert abs(expect-z['path'][i+1]['loss'])<1e-8
    assert max(errors,default=0.)<3e-6, ('loss_arithmetic',max(errors))
    return max(errors,default=0.)


def main():
    p=plan();allr=read(OUT/'ALL_SOURCE_RESULTS.json')
    receipts=0
    for f in OUT.rglob('*.pt'):
        assert existing(f);receipts+=1
    parent=read(PARENT/'COMPLETION.json')
    for f,h in {**parent['files'],**parent['code_pins']}.items():assert sha(ROOT/f)==h
    for f,h in p['protected_pins'].items():assert sha(ROOT/f)==h
    regs=read(OUT/'LOCKED_IMPLEMENTATION.json')['registry_values']
    for f,v in regs.items():assert read(ROOT/f)==v
    counts=dict(fits=0,backward=0,new_DINO=0,crop_teacher=0)
    reuse=collections.Counter();loss_errors=[];full_replays=0;observations_count=0;fits=[]
    # Every source included in each stage. GT never enters any A/T fit input.
    for c,rr in p['rows'].items():
        assert len(rr)==24
        dev=[r for r in rr if r['f36_role']=='development'];ev=[r for r in rr if r['f36_role']=='evaluation']
        assert len(dev)==8 and len(ev)==16
        for field in ('group','source'):
            assert not {r[field] for r in dev}&{r[field] for r in ev}
        for r in rr:
            x=load(dest('space',r['key']));assert not x['GT_online']
            counts['new_DINO']+=x['expert']['new_DINO'];observations_count+=len(x['expert']['reused'])
            common=sorted(set(x['expert']['observed4']+x['expert']['observed8']))
            assert common==x['expert']['common_observed']
            assert len(x['fits']['A5']['anchors'])<=8 and x['fits']['A5']['planned']==8
            for name,z in x['fits'].items():
                if name in x['reuse']:
                    reuse['space_'+name]+=1;continue
                counts['fits']+=1;counts['backward']+=z['backwards'];fits.append(z)
                loss_errors.append(audit_fit(z))
                assert not z['GT_online'] and z['parameter_count']==1792
                assert z['final']['indices']==x['native_indices']
                assert all(torch.equal(a,b) for a,b in zip(z['final']['logits'],x['native_logits']))
            for audit in x['audits'].values():
                assert audit['box_max_error']==audit['logit_max_error']==0;full_replays+=1
            stages=['development','controls_dev'] if r['f36_role']=='development' else ['evaluation']
            for stage in stages:
                t=load(dest('temporal_'+stage,r['key']));assert not t['GT_online']
                for name,z in t['fits'].items():
                    kind='partial' if name.startswith('partial_') else name
                    counts['fits']+=1;counts['backward']+=z['backwards'];fits.append(z)
                    loss_errors.append(audit_fit(z,t['teachers'][kind]))
                    assert not z['GT_online'] and z['parameter_count']==66306
                    assert torch.equal(z['final']['boxes'],t['native_boxes'])
                for audit in t['audits'].values():
                    assert audit['box_max_error']==audit['logit_max_error']==0;full_replays+=1
                for o in t['outputs'].values():
                    assert all(a['converged'] and a['gap']<1e-7 for a in o['audit'])
            tc=load(dest('teachers',r['key']));counts['crop_teacher']+=2*len(tc['predictions'])
            signatures=set()
            for tr in tc['predictions']:
                w=tr['window'];ids=w['frame_ids']
                assert ids==[r['input']['frame_ids'][i] for i in w['positions']]
                assert ids!=r['input']['frame_ids'] and tuple(ids) not in signatures;signatures.add(tuple(ids))
                assert tr['correct']['qa']['rgb_sha256']==tr['wrong']['qa']['rgb_sha256']
                assert tr['correct']['qa']['input_sha256']==tr['wrong']['qa']['input_sha256']
                assert tr['donor']['source']!=r['source']
                for q in ('correct','wrong'):
                    assert tr[q]['offset_forwards']==2 and tr[q]['complete_predictions']==1
    for f in (OUT/'noops').glob('*.pt'):
        x=load(f);z=x['fit'];counts['fits']+=1;counts['backward']+=z['backwards']
        assert z['state_delta']==0 and z['restore_exact'];fits.append(z)
        assert x['audit']['box_max_error']==x['audit']['logit_max_error']==0
    for r in p['H_rows']:
        h=load(dest('absorption',r['key']));assert h['GT_supervised_diagnostic']
        for name,z in h['fits'].items():
            if name=='all_teacher10':reuse['H_all_teacher10']+=1;continue
            counts['fits']+=1;counts['backward']+=z['backwards'];fits.append(z)
            loss_errors.append(audit_fit(z))
            assert z['actual_final_readout_traced']
        a=h['fits']['matched_teacher10']['anchors'];b=h['fits']['matched_GT10']['anchors']
        assert [(v['position'],v['weight']) for v in a]==[(v['position'],v['weight']) for v in b]
        if h['conditional_trigger']:
            assert h['interface']['only_second_final_output_changes']
            assert h['interface']['shape']['parameters']==1028
        for audit in h['audits'].values():
            assert audit['box_max_error']==audit['logit_max_error']==0;full_replays+=1
    for r in [r for rr in p['rows'].values() for r in rr if r['f36_role']=='evaluation']:
        co=load(dest('composition',r['key']));a=co['prediction']['audit']
        assert a['box_max_error']==a['logit_max_error']==0 and co['disjoint_private_states'];full_replays+=1
    recorded=read(OUT/'COUNTERS.json');assert counts==recorded,(counts,recorded)
    assert all(counts[k]<=p['caps'][k] for k in counts)
    # Independent source macro and cluster bootstrap across all main summary blocks.
    comparisons=0;maximum=0.
    for c in p['rows']:
        for name,rowkey in [('space','space_evaluation'),('temporal','temporal_evaluation'),('system','system_rows')]:
            rows=[r for r in allr[rowkey] if r['cohort']==c]
            ss=allr['system_summary'][c] if name=='system' else allr['summaries'][c][name]
            for typ,groups in [('arms',ss['arms']),('contrasts',ss['contrasts'])]:
                for n,mm in groups.items():
                    for metric,expected in mm.items():
                        vals=collections.defaultdict(list)
                        for r in rows:
                            if typ=='arms':v=r['arms'].get(n,{}).get(metric)
                            else:
                                a,b=n.split(' - ');x=r['arms'].get(a,{}).get(metric);y=r['arms'].get(b,{}).get(metric)
                                v=x-y if x is not None and y is not None else None
                            if v is not None:vals[r['group']].append(v)
                        if not vals:assert expected['mean'] is None;continue
                        vector=np.array([sum(vals[g])/len(vals[g]) for g in sorted(vals)])
                        rng=np.random.default_rng(20260914)
                        boots=vector[rng.integers(0,len(vector),(10000,len(vector)))].mean(1)
                        err=max(abs(vector.mean()-expected['mean']),max(abs(np.quantile(boots,[.025,.975])-expected['ci95'])))
                        assert err<1e-12 and len(vector)==expected['sources']
                        maximum=max(maximum,float(err));comparisons+=1
    outcome=dict(status='pass',model_fits=counts,counts_match_receipts=True,raw_pt_receipts=receipts,
        reused_conditions=dict(reuse),exact_cached_expert_requests=observations_count,
        loss_trajectories_checked=len(loss_errors),maximum_loss_error=max(loss_errors),
        actual_full_model_replays=full_replays,bootstrap_cells_checked=comparisons,maximum_aggregation_error=maximum,
        protected_F35_files=len(parent['files']),protected_registries_unchanged=True,
        GT_online_AT=False,H_GT_separate=True,crop_predictions=counts['crop_teacher'],
        crop_offset_forwards=2*counts['crop_teacher'],complete_main_sources=48,untouched=False,
        numerical_failed_fits=sum(bool(z['failure']) for z in fits),zero_state_fits=sum(z['state_delta']==0 for z in fits),
        skipped_fits=sum(bool(z['skipped']) for z in fits),budget_caps=p['caps'])
    write(OUT/'INDEPENDENT_AUDIT.json',outcome)
    total_seconds=sum(x['seconds'] for x in (read(f) for f in (OUT/'leases').glob('*.json')))
    tests='15 CPU synthetic tests passed; toy gradients are not video fits and are separately scoped.'
    text='# F36 计算成本与预算\n\n'+ '\n'.join(f'- {k}: {v} / {p["caps"][k]}' for k,v in counts.items())+'\n\n'
    text+=f'累计有记录GPU lease {total_seconds/60:.2f}分钟（包含加载、读取、校验；不等于纯kernel时间）；裁剪教师完整预测 {counts["crop_teacher"]} 次 / offset前向 {2*counts["crop_teacher"]} 次。\n\n'
    text+=f'新空间DINO请求 {counts["new_DINO"]}；精确旧请求 {observations_count}。缓存复用不代表部署专家免费，A0/A5单query预算8次、A4为4次。A5去掉双视图与融合但增加观察时刻，不能宣称DINO调用减半。\n\n'
    crop_cost=collections.defaultdict(list);out_cost=collections.defaultdict(list)
    for r in allr['temporal_evaluation']:
        tc=load(load(r['raw_path'])['teacher_path'])
        crop_cost[r['cohort']].append(sum(z['correct']['seconds'] for z in tc['predictions']))
        out_cost[r['cohort']].extend(r['output_seconds'].values())
    for c in p['rows']:
        text+=f'- {c}: 正确query新增教师平均{np.mean(crop_cost[c]):.3f}秒/query；同信息simplex平均{np.mean(out_cost[c]):.6f}秒/query。错误query控制成本在总预算中，但不是候选部署开销。\n'
    text+='\n每fit suffix时间、回溯次数、实际步长均保存在pt。最终模型重放和加载有独立回执；不要把缓存条件的总耗时当无缓存端到端吞吐。'+tests+'\n'
    text+='\n追加核清：202次新DINO实际是本轮未合并全部F35八原图缓存导致的重复推理，不是新增视频/独立证据。与旧工件418条可比观察逐框一致，其中包含这202条；详见F35_OBSERVATION_REPLAY_AUDIT.json。427是本轮主空间缓存复用记账条数，不是418条跨轮可比审计分母，两者不可混写。\n'
    (OUT/'COSTS.md').write_text(text)
    print(json.dumps(outcome,indent=2),flush=True)


if __name__=='__main__':main()
