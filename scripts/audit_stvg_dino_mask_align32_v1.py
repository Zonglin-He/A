"""Independent update, mask, layer trace and complete metric verification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_mask_align32_v1 import D,P,PRE,verify
from vg_tta.stvg_error_state_correction_v1 import encode,decode
from scripts.score_stvg_joint_robust_correction32_v1 import paired


def main(dynamic=False):
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube
    import torch
    p=verify();folder=D/'dynamic'if dynamic else D;result=read(P/('DYNAMIC_RESULTS_PUBLIC.json'if dynamic else'RESULTS_PUBLIC.json'))
    assert read(folder/'GLOBAL_PREDICTION_BARRIER.json')['count']==32
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};torch.set_num_threads(4)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};c=collections.Counter();maxupdate=0.
    for rec,rowresult in zip(p['records'],result['rows']):
        z=load(folder/'predictions'/(rec['alias']+'.pt'));e=load(ROOT/rec['prior_prediction']);source=np.asarray(e['outputs']['Source_no_correction'],float);chart=encode(source);scale=z['scale'];mu=z['initial_mean']
        assert not z['GT_read']and z['pretrained_weights_frozen_exact'];ds=rec['dataset'];q=rec['query_ordinal'];row=plans[ds][q]
        paths=[]
        if not dynamic:
            old=load(PRE/'predictions'/(rec['alias']+'.pt'));assert np.array_equal(z['actions'],old['actions']);c['exact_original_action_arrays']+=1
            for m,f in z['matched'].items():paths.append((m,z['actions'],f,z['fits'][m],z['traces']))
        elif z['active']:
            before=mu.copy()
            for step in z['fits']['B_aligned_dynamic']['rounds']:
                assert np.array_equal(step['mean_before'],before);paths.append(('B_aligned',step['actions'],dict(targets=step['targets'],details=step['details']),step,step['traces']));before=step['mean_after']
        for m,actions,f,info,traces in paths:
            expected=[]
            for a,d in zip(actions,f['details']):
                good=[encode(np.array([v['box']]))[0]-chart[pos]for pos,v in d['anchors'].items()if v['score']>=.35]
                y=np.mean(good,0)/scale if good else a
                expected.append(y);assert d['accepted_anchors']==len(good);c['teacher_actions_reconstructed']+=1
                for v in d['anchors'].values():assert v['accepted']==(v['score']>=.35);c['admissions']+=1
            expected=np.asarray(expected);assert np.max(abs(expected-f['targets']))<1e-12
            new=info['mean_before']+(expected-actions).sum(0)/24;maxupdate=max(maxupdate,float(abs(new-info['mean_after']).max()));assert np.array_equal(new,info['mean_after']);c['updates_reconstructed']+=1
            if not (expected-actions).any():assert np.array_equal(info['mean_after'],info['mean_before']);c['exact_identity_updates']+=1
            for pos,fr in traces.items():
                trace=fr[m];k=len(trace['mask_slots']);n=900+k;mask=np.full((1,1,n,n),np.finfo(np.float32).min,np.float32);mask[:,:,:900,:900]=0
                for j,slot in enumerate(trace['mask_slots']):mask[:,:,900+j,:900]=0;mask[:,:,900+j,int(slot)]=np.finfo(np.float32).min;mask[:,:,900+j,900+j]=0
                assert hashlib.sha256(mask.tobytes()).hexdigest()==trace['mask_sha256'];c['actual_mask_hashes']+=1
                assert trace['boxes'].shape==(6,k,4)and trace['scores'].shape==(6,k)and trace['references'].shape==(6,k,4);c['six_layer_traces']+=1
                if m=='B_aligned':assert np.array_equal(trace['content_slots'],trace['mask_slots'])
                else:assert np.array_equal(trace['mask_slots'],trace['nearest_slots'])
                ids=fr['candidate_indices'];inv=fr['inverse']
                for j,ix in enumerate(ids):
                    v=f['details'][ix]['anchors'][pos];assert np.array_equal(v['box'],trace['boxes'][-1,inv[j]])and v['score']==float(trace['scores'][-1,inv[j]]);c['final_trace_to_feedback']+=1
            if not dynamic:
                calculated=decode(chart+info['mean_after']*scale)if not info['all_identity_exact_noop']else z['outputs']['E_MBR']
                assert np.array_equal(calculated,z['outputs'][m+'_once']);c['complete_arrays_reconstructed']+=1
        for pos,native in z['native'].items():
            assert native['initial_references'].shape==(900,4)and native['layer_boxes'].shape==(6,900,4)and native['layer_scores'].shape==(6,900);c['native_initial_and_layers']+=1
        for arm,b in z['outputs'].items():
            independent=DenseTube(b,row,gt[ds][0][q],gt[ds][1][q],clip=ds=='hc2').score(z['interval'])
            for k in ['v','t','s']:assert abs(independent[k]-rowresult['values'][arm][k])<2e-10;c['independent_full_metrics']+=1
    for ds,g in result['groups'].items():
        rr=[x for x in result['rows']if x['dataset']==ds]
        for name,pair in g['paired'].items():
            a,b=name.split('_minus_');expected=paired([100*(x['values'][a]['v']-x['values'][b]['v'])for x in rr]);assert expected==pair['v'];c['paired_query_summaries']+=1
    if not dynamic:
        expected=all(g['feedback']['B_aligned'].get('accepted_refs',0)>g['feedback']['A_unaligned'].get('accepted_refs',0)and g['paired']['B_aligned_once_minus_E_MBR']['v']['mean']>0 for g in result['groups'].values())
        assert expected==result['decision']['pass_for_dynamic'];c['prelocked_continuation']+=1
    put(P/('DYNAMIC_ROOT_AUDIT.json'if dynamic else'ROOT_INDEPENDENT_AUDIT.json'),dict(status='pass',count=32,checks=dict(c),maximum_update_error=maxupdate,pretrained_weights_unchanged=True,GT_only_after_full_barrier=True,no_GT_in_updates=True))
    print(dict(status='pass',checks=dict(c)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--dynamic',action='store_true');main(a.parse_args().dynamic)
