"""Three-arm paired source statistics, current-policy and reward reconstruction."""
import sys,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha,status
from scripts.score_tastvg_spatial_online_opd_s1_v1 import macro
from scripts.audit_tastvg_spatial_s06_s1_public_v1 import online
from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
BASE=ROOT/'artifacts/tastvg_spatial_rank_s11_v1';RAW=ROOT/'artifacts/tastvg_spatial_online_opd_s1_v1';S0=ROOT/'artifacts/tastvg_spatial_expansion_s0_v1';NATIVE=ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1'

def run():
    torch.set_num_threads(4);rr={a:read((RAW if a=='raw' else BASE/a)/'ROWS.json') for a in ['raw','rank','norm']};radius=read(NATIVE/'PARAMETER_SUPPORT.json')['radius'];diags={};resources={}
    for arm,rows in rr.items():
        out=RAW if arm=='raw' else BASE/arm;bar=read(out/'PREDICTION_BARRIER.json')
        for f,h in {**bar['files'],**bar['final_states']}.items():assert sha(out/f)==h
        ups=[r for r in rows if r['updated']];d={}
        for key,get in [('gradient_norm',lambda r:r['update_diagnostics']['gradient_norm']),('step_norm',lambda r:r['step_displacement']),('q_range',lambda r:np.ptp(r['update_diagnostics']['q'])),('p_range',lambda r:np.ptp(r['update_diagnostics']['p'])),('step_to_probe_ratio',lambda r:r['step_displacement']/radius),('KL_before',lambda r:r['update_diagnostics']['loss_before']),('KL_after',lambda r:r['update_diagnostics']['loss_after'])]:
            values=[get(r) for r in ups];d[key]=dict(min=float(min(values)),median=float(np.median(values)),max=float(max(values)))
        d['updates']=len(ups);d['KL_decreased']=sum(r['update_diagnostics']['loss_after']<r['update_diagnostics']['loss_before'] for r in ups);d['query_gradient_nonzero']=sum(r['update_diagnostics']['query_gradient_norm']>0 for r in ups);d['LN_gradient_nonzero']=sum(r['update_diagnostics']['LN_gradient_norm']>0 for r in ups);d['final_state_norms']={r['condition']:r['state_displacement'] for r in rows if r['arrival']==15};diags[arm]=d
        if arm=='raw':continue
        count=changed=empty=reward_checks=nonexpert=0
        for r in rows:
            x=load(out/'online'/r['condition']/f"{r['parent']:03}.pt")
            if not r['expert_scheduled']:
                assert 'candidates' not in x and 'rewards' not in x and 'valid_expert_frames' not in x;nonexpert+=1;continue
            assert torch.equal(x['candidates'][0]['prediction']['boxes'],x['prediction']['boxes']);count+=1
            source=load(S0/'capture'/r['condition']/f"{r['parent']:03}.pt")['prediction'];changed+=int(not torch.equal(x['prediction']['boxes'],source['boxes']))
            expert=load(S0/'expert'/r['condition']/f"{r['parent']:03}.pt");assert expert['pixel_sha256']==x['pixel_sha256']
            val=rewards([c['prediction']['boxes'].numpy() for c in x['candidates']],expert['boxes'],expert['valid'])
            if val is None:assert x['rewards'] is None and not x['updated'];empty+=1
            else:np.testing.assert_allclose(val,x['rewards'],atol=1e-14,rtol=0);reward_checks+=len(val)
        write(out/'CURRENT_POLICY_AUDIT.json',dict(status='pass',current_center_exact=count,learned_center_sets=changed,empty_expert_noops=empty,nonexpert_without_expert=nonexpert,reconstructed_rewards=reward_checks))
        write(out/'PUBLIC_AUDIT.json',online(out))
        allocations=[read(p) for p in sorted((out/'allocations').glob('*.json'))];assert len(allocations)==1 and allocations[0]['status']=='completed';resources[arm]=dict(GPU_process_seconds=sum(v['seconds'] for v in allocations),GPU_failures=0,backward_steps=len(ups),current_policy_candidates=9*count,new_expert_calls=0,new_encoder_captures=0,learned_full_reinsertions=4)
    write(BASE/'DIAGNOSTICS.json',diags);write(BASE/'RESOURCES.json',dict(arms=resources,total_GPU_process_seconds=sum(r['GPU_process_seconds'] for r in resources.values()),raw_reused=True))
    paired=[]
    for i,raw in enumerate(rr['raw']):
        x={a:rr[a][i] for a in rr};assert all((r['parent'],r['condition'],r['arrival'],r['expert_scheduled'])==(raw['parent'],raw['condition'],raw['arrival'],raw['expert_scheduled']) for r in x.values())
        r={k:raw[k] for k in ['parent','condition','arrival','expert_scheduled']}
        for metric in ['s','t','v']:
            for a,b in [('rank','raw'),('norm','raw'),('norm','rank')]:r[f'{a}_minus_{b}_{metric}']=x[a]['online_'+metric]-x[b]['online_'+metric]
            for a in rr:r[f'{a}_minus_frozen_{metric}']=x[a]['online_minus_frozen_'+metric]
        paired.append(r)
    summary={};sources=[]
    for group in ['corruption']+read(BASE/'rank/LOCK.json')['conditions']:
        seq=[r for r in paired if (r['condition']!='clean' if group=='corruption' else r['condition']==group)];summary[group]={}
        for subset in ['all','nonexpert','expert','nonexpert_after_first_write']:
            rs=[r for r in seq if subset=='all' or (subset=='nonexpert' and not r['expert_scheduled']) or (subset=='expert' and r['expert_scheduled']) or (subset=='nonexpert_after_first_write' and not r['expert_scheduled'] and r['arrival']>4)]
            summary[group][subset]=dict(cells=len(rs),sources=len({r['parent'] for r in rs}),metrics={k:macro(rs,k) for k in paired[0] if '_minus_' in k})
            if subset=='nonexpert' and group in ['corruption','clean']:
                for parent in sorted({r['parent'] for r in rs}):sources.append(dict(parent=parent,group=group,**{k:float(np.mean([r[k] for r in rs if r['parent']==parent])) for k in paired[0] if '_minus_' in k}))
    write(BASE/'ROWS.json',paired);write(BASE/'SUMMARY.json',summary);write(BASE/'SOURCE_EFFECTS.json',sources)
    p=read(RAW/'PROVENANCE.json');p.update(experiment='S1.1 Raw / Rank / Rank+Normalized-Step',cells_per_arm=96,total_new_cells=192,raw_reused=True,raw_barrier_sha256=sha(RAW/'PREDICTION_BARRIER.json'),raw_rows_sha256=sha(RAW/'ROWS.json'),new_arms={a:dict(lock_sha256=sha(BASE/a/'LOCK.json'),protocol_and_code_pins=read(BASE/a/'LOCK.json')['pins'],prediction_barrier_sha256=sha(BASE/a/'PREDICTION_BARRIER.json')) for a in ['rank','norm']});p.pop('lock_sha256');p.pop('protocol_and_code_pins');write(BASE/'PROVENANCE.json',p)
    for g in ['corruption','clean']:
        print(g,{k:100*v['mean'] for k,v in summary[g]['nonexpert']['metrics'].items() if k.endswith(('_s','_v'))})
    print(diags)

if __name__=='__main__':run()
