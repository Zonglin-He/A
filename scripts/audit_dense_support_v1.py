"""F39 independent NumPy loss enumeration, state, receipts, aggregation audit."""
import collections
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp,expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.prepare_dense_support_v1 import OUT,plan
from scripts.run_dense_support_v1 import dest,existing
from scripts.audit_spatial4_attribute_v1 import aggregate_check

def independent(logits,refs,teacher,beta):
    ev=[];kl=[];rs=[]
    for z,ref,t in zip(logits,refs,teacher):
        z=z.numpy().reshape(-1,2).astype(float);ref=ref.numpy().reshape(-1,2).astype(float)
        n=len(z);s,e=np.triu_indices(n,1)
        lp=z[s,0]+z[e,1];lp-=logsumexp(lp)
        p0=ref[s,0]+ref[e,1];p0-=logsumexp(p0)
        li=[];lo=[]
        for i in range(n):
            inside=(s<=i)&(i<=e)
            for vals,mask in [(li,inside),(lo,~inside)]:
                val=logsumexp(lp[mask]) if mask.any() else math.log(1e-12)
                vals.append(float(np.clip(val,math.log(1e-12),math.log1p(-1e-12))))
        a=t['a'].numpy().astype(float);w=t['omega'].numpy()
        ev.append(float(-(w*(a*np.array(li)+(1-a)*np.array(lo))).sum()))
        kl.append(float((np.exp(p0)*(p0-lp)).sum()));rs.append(np.exp(li))
    return float(np.mean(ev)+beta*np.mean(kl)),float(np.mean(ev)),float(np.mean(kl)),rs

def main():
    torch.set_num_threads(2);p=plan();rr=[r for v in p['rows'].values() for r in v]
    assert len(rr)==48 and len({r['key'] for r in rr})==48
    assert len({r['group'] for r in rr})==48
    for c,rows in p['rows'].items():
        dev=[r for r in rows if r['f37_role']=='development'];ev=[r for r in rows if r['f37_role']=='evaluation']
        assert len(dev)==8 and len(ev)==16
        for key in ('source','group'):assert not {r[key] for r in dev}&{r[key] for r in ev}
        assert not {r['input']['video_sha256'] for r in dev}&{r['input']['video_sha256'] for r in ev}
    counts=dict(fits=0,backward=0,new_DINO=0);stages=collections.Counter();errors=[];replays=0;teacher_positions=0
    spatial_counts=[];temporal_counts=[];full_field_checks=0;forced=collections.Counter();times=[]
    for stage in ['dev','dev_controls','eval']:
        rows=[r for r in rr if r['f37_role']==('evaluation' if stage=='eval' else 'development')]
        assert len(list((OUT/stage).glob('*.pt')))==len(rows)
        for r in rows:
            path=dest(stage,r['key']);assert existing(path);x=load(path)
            assert not x['GT_online'] and x['new_DINO']==0 and x['teacher_full_model_exact']
            old=load(x['A4_reference']['path']);assert sha(x['A4_reference']['path'])==x['A4_reference']['sha256']
            a4=old['fits']['A4'];assert torch.equal(x['T0']['boxes'],a4['final']['boxes'])
            assert x['T0']['indices']==old['native_indices'] and x['qa']==old['qa']
            assert sum(v.numel() for v in a4['state'].values())==1792
            spatial_counts.append(1792)
            full_field_checks+=2;teacher_positions+=sum(len(t['a']) for t in x['teacher'])
            for j,(t,w,rolled) in enumerate(zip(x['teacher'],x['wrong_teacher'],x['rolled_teacher'])):
                f=np.asarray(t['frame_ids']);edges=np.r_[f[0],(f[:-1]+f[1:])/2,f[-1]+1]
                assert np.array_equal(t['cell_edges'].numpy(),edges)
                assert np.allclose(t['omega'],np.diff(edges)/np.diff(edges).sum(),atol=1e-15,rtol=0)
                assert len(t['a'])==len(f)==len(x['records'][j]['frame_ids']) and bool(t['valid_mask'].all())
                assert t['sigmoid_applications']==1 and t['semantic_threshold'] is None
                assert np.max(np.abs(expit(t['raw_logits'].numpy().astype(float))-t['a'].numpy()))<1e-7
                assert not t['a'].requires_grad
                assert rolled['roll']==len(f)//2
                assert torch.equal(torch.roll(t['a'],len(f)//2),rolled['a'])
                assert w['frame_ids']==t['frame_ids']
            assert x['wrong_query']==r['F39_wrong_query'] and x['wrong_query']['source']!=r['source']
            if stage=='dev_controls':
                prior=load(dest('dev',r['key']))
                for a,b in zip(prior['teacher'],x['teacher']):assert torch.equal(a['raw_logits'],b['raw_logits'])
                for a,b in zip(prior['wrong_teacher'],x['wrong_teacher']):assert torch.equal(a['raw_logits'],b['raw_logits'])
            for name,z in x['fits'].items():
                counts['fits']+=1;stages[stage]+=1;counts['backward']+=z['backwards'];times.append(z['seconds'])
                assert z['steps']==5 and z['backwards']==5 and z['failure'] is None
                assert z['teacher_frozen'] and z['actual_best_state_verified'] and z['spatial_invariant']
                assert not z['GT_online']
                if name=='T2':
                    assert z['output_optimization'] and z['optimizer']=='Adam'
                    assert all(n.startswith('output.') for n in z['state'])
                else:
                    assert z['parameters']==66306 and z['optimizer']=='AdamW'
                    assert all(n.startswith('head.') for n in z['state']);temporal_counts.append(66306)
                t=x['wrong_teacher'] if name=='T3' else x['rolled_teacher'] if name=='T4' else x['teacher']
                for row in z['path']:
                    assert torch.equal(row['boxes'],a4['final']['boxes'])
                    v,ev,kl,rs=independent(row['logits'],x['T0']['logits'],t,z['beta'])
                    err=max(abs(v-row['loss']),abs(ev-row['parts']['event']),abs(kl-row['parts']['keep']))
                    assert err<1e-10,(r['key'],name,row['step'],err);errors.append(err)
                    for j,oa in enumerate(row['parts']['offsets']):
                        assert len(oa['inclusion'])==len(t[j]['a'])
                        assert np.max(np.abs(rs[j]-oa['inclusion']))<1e-10
                        forced['zero']+=len(oa['forced_zero']);forced['one']+=len(oa['forced_one'])
                    if 'accepted' in row:
                        assert row['gradient_norm']>=0 and not row['unused_parameters']
                        if row['accepted']:assert row['trials'][-1]['loss']<row['loss']-1e-9
                        else:assert row['optimizer_restored_on_reject']
                    if row['step']>0:
                        prev=z['path'][row['step']-1]
                        if not prev['accepted']:assert all(torch.equal(v,prev['state'][k]) for k,v in row['state'].items())
                best=z['path'][z['best_step']]
                assert best['loss']==min(q['loss'] for q in z['path'])
                assert all(torch.equal(v,best['state'][k]) for k,v in z['state'].items())
            for a in [x['native_full_model']]+list(x['full_reinsertion'].values()):
                assert a['full_model'] and a['source_restored'] and a['box_max_error']==a['logit_max_error']==0;replays+=1
    for path in (OUT/'noops').glob('*.pt'):
        assert existing(path);x=load(path);z=x['fit'];counts['fits']+=1;stages['noops']+=1;counts['backward']+=z['backwards']
        assert z['state_delta']==0 and z['final']['indices']==z['path'][0]['indices']
        assert all(torch.equal(a,b) for a,b in zip(z['final']['logits'],z['path'][0]['logits']))
        assert x['audit']['box_max_error']==x['audit']['logit_max_error']==0;replays+=1
    assert stages==dict(dev=96,dev_controls=48,eval=128,noops=4),stages
    assert counts==dict(fits=276,backward=1362,new_DINO=0),counts
    assert all(v<=p['caps'][k] for k,v in counts.items())
    cells=0;aggregate_errors=[]
    for stage in ['DEV','DEV_CONTROLS','EVAL']:
        x=read(OUT/(stage+'_RESULTS.json'))
        for c,s in x['summary'].items():
            rows=[r for r in x['rows'] if r['cohort']==c]
            n,e=aggregate_check(rows,s);cells+=n;aggregate_errors.append(e)
            for g,sub in s['event_length_diagnostic'].items():
                n,e=aggregate_check([r for r in rows if r['event_length_group']==g],sub);cells+=n;aggregate_errors.append(e)
    sel=read(OUT/'TEMPORAL_SELECTION.json')
    assert sel['dev_results_sha256']==sha(OUT/'DEV_RESULTS.json') and not sel['eval_results_read']
    assert sel['chosen']['common_delta']==max(r['common_delta'] for r in sel['all_configs'])
    assert all(read(path)['completed']>sel['created_unix'] for path in (OUT/'eval').glob('*.json'))
    config=read(OUT/'CONFIG.json')
    assert config['temporal']['enabled'] and not config['temporal']['automatic_spatial_only_fallback']
    assert config['temporal']['lr']==sel['chosen']['lr'] and config['temporal']['beta']==sel['chosen']['beta']
    prior=read(ROOT/'artifacts/decota_spatial4_weighted_attribute_v1/EVAL_RESULTS.json')
    now=read(OUT/'EVAL_RESULTS.json');baseline_errors=[]
    for c,s in now['summary'].items():
        for m in ('vIoU_corrected','sIoU','tIoU'):
            error=abs(s['arms']['T0'][m]['mean']-prior['summary'][c]['arms']['T0'][m]['mean'])
            baseline_errors.append(error);assert error<1e-12
    command=['bash','scripts/with_local_cuda.sh','.conda/tubedetr/bin/python','-B','-m','pytest','-q',
             'tests/test_dense_support_temporal_v1.py','tests/test_simplification_partial_v1.py']
    tick=time.time();test=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    write(OUT/'TEST_RECEIPT.json',dict(command=command,exit_code=test.returncode,seconds=time.time()-tick,stdout=test.stdout,stderr=test.stderr))
    assert test.returncode==0 and 'passed' in test.stdout
    out=dict(status='pass',counts=counts,fit_stages=dict(stages),unique_queries=48,unique_sources=48,
        development_sources=16,evaluation_sources=32,duplicates=0,historically_exposed=True,untouched=False,
        protected_files=len(p['protected_pins']),both_registries_and_A4_readonly=True,
        independent_loss_states=len(errors),maximum_loss_error=max(errors),
        independent_statistics_cells=cells,maximum_aggregation_error=max(aggregate_errors),
        F38_A4_T0_reproduction_max_error=max(baseline_errors),
        actual_full_model_reinsertions=replays,full_model_box_logit_max_error=0,
        final_actionness_full_field_offset_checks=full_field_checks,teacher_position_checks=teacher_positions,
        structural_forced_occurrences_across_all_trajectories=dict(forced),
        spatial_parameters=1792,temporal_network_parameters=66306,total_network_parameters=68098,
        temporal_network_fits=len(temporal_counts),noops=4,optimizer_and_parameter_rollback_verified=True,
        all_valid_positions_used=True,exactly_one_sigmoid=True,wrong_query_same_geometry=True,
        time_roll_histogram_preserved=True,GT_online=False,tests=test.stdout.strip(),
        optimizer_seconds=dict(sum=sum(times),mean=float(np.mean(times)),median=float(np.median(times))),
        limitations=['Full-model checks apply to initial and selected actual states, not every backtracking proposal.',
          'Timing separates cached temporal optimizer from whole-system inference; not an end-to-end speed benchmark.',
          'Historically exposed small source-isolated pool, not untouched/full-dataset/SOTA evidence.',
          'T2 has a different parametrization and no symmetric hyperparameter selection; not an upper bound.'])
    write(OUT/'AUDIT.json',out);print(out,flush=True)

if __name__=='__main__':main()
