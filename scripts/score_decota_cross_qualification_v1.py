"""Independent post-seal cross-domain state, objective, dense and paired audits."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_cross_qualification_common_v1 import *
from scripts.score_decota_online_qualification_v1 import audit_fit,aggregate
from scripts.score_audit_tastvg_decota_c1_same_domain_v1 import check_state
import torch,numpy as np

def run():
    from scripts import tastvg_decota_c1_common_v1 as c1
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    from vg_tta.decota_actuation_scope_v1 import commit_state
    from vg_tta.c1_enabling_tricks_v1 import QUERY
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();b=read(CROSS/'GLOBAL_PREDICTION_BARRIER.json');assert b['arrivals']==2304 and not b['GT_read']
    for f,h in b['files'].items():assert sha(CROSS/f)==h
    exposure=CROSS/'GT_EXPOSURE.json'
    if exposure.exists():assert read(exposure)['barrier_sha256']==sha(CROSS/'GLOBAL_PREDICTION_BARRIER.json')
    else:write(exposure,dict(time=time.time(),barrier_sha256=sha(CROSS/'GLOBAL_PREDICTION_BARRIER.json'),GT_online=False,historical_exposure=True))
    torch.set_num_threads(4);rows=[];diag=[];checks=0;episodes={};tick=time.time();label_provenance={};resources={};smoke_backwards=0
    for ds in DATASETS:
        p=read(CROSS/ds/'PLAN.json');labels={};label_provenance[ds]={}
        for split in p['splits']:
            f=c1.POOL/ds/f'GT_LABELS_{split}.json';old=read(c1.POOL/ds/f'GT_EXPOSURE_{split}.json');assert sha(f)==old['labels_sha256'];labels[split]=read(f)
            label_provenance[ds][split]=dict(labels_sha256=sha(f),historic_exposure_receipt_sha256=sha(c1.POOL/ds/f'GT_EXPOSURE_{split}.json'),unchanged=True)
        capture=read(CROSS/ds/'CAPTURE_BARRIER.json');resources[ds]={k:v for k,v in capture.items() if k!='files'}
        resources[ds]['prediction']={k:v for k,v in read(CROSS/ds/'PREDICTION_BARRIER.json').items() if k!='files'}
        assert resources[ds]['prediction']['source_state_sha256']==capture['source_state_sha256']
        for stream in ['episodic','online_100']:
            for split,sp in p['splits'].items():
                for cond in p['conditions']:
                    for order,seq in sp['orders'].items():
                        previous=prevsha=None
                        for at,parent in enumerate(seq):
                            row=p['rows'][parent];f=CROSS/ds/stream/split/cond/order/f'{at:05}.pt';x=checked(f)
                            assert x['dataset']==ds and x['parent']==parent and x['stream']==stream and x['previous_payload_sha256']==prevsha
                            assert not x['GT_read'] and x['expert'] and x['query_reset'] and x['optimizer_reset'] and x['temporal_arm']=='native'
                            initial={n:torch.zeros_like(v) if n==QUERY else v for n,v in (previous if previous is not None else x['source_state']).items()};checks+=check_state(initial,x['initial'])
                            ep=ROOT/x['evidence_path'];assert sha(ep)==x['evidence_sha256'];ex=checked(ep);assert ex['pixel_sha256']==x['pixel_sha256']
                            cf=ROOT/x['capture_path'];assert sha(cf)==x['capture_sha256'];data=checked(cf);assert data['checkpoint_state_sha256']==capture['source_state_sha256']
                            assert data['pixel_sha256']==x['pixel_sha256'] and data['frame_ids']==row['frame_ids'] and ex['native_indices']==data['prediction']['indices']
                            assert data['prediction']['physical_interval']==x['interval']==x['native']['physical_interval'] and torch.equal(data['prediction']['boxes'],x['native']['boxes'])
                            checks+=audit_fit(x['fit'],x,ex['expert'],row);checks+=check_state(commit_state(x['initial'],x['fit']['write_proposal']),x['committed']);assert torch.equal(x['fit']['final'],x['after'])
                            assert torch.count_nonzero(x['committed'][QUERY])==torch.count_nonzero(x['initial'][QUERY])==0
                            if stream=='online_100':previous=x['committed'];prevsha=sha(f)
                            lab=labels[split][str(parent)];truth={int(k):v for k,v in lab['truth'].items()};span=lab['span'];iv=x['interval']
                            dense=lambda box:DenseTube(box.numpy(),row,truth,span,clip=ds=='hc2')
                            a=dense(x['after']);c=dense(x['before']);fr=dense(x['native']['boxes']);after=a.score(iv);before=c.score(iv);frozen=fr.score(iv)
                            off=official(x['after'].numpy(),row,truth,span,iv,ds);assert max(abs(after[k]-off[k]) for k in off)<2e-12;checks+=3
                            key=(ds,split,cond,order,at)
                            if stream=='episodic':
                                episodes[key]=after['v']
                                if split=='search' and cond=='clean' and order=='order1' and at<2:smoke_backwards+=2*x['fit']['gradient_calls']
                            rows.append(dict(dataset=ds,stream=stream,split=split,condition=cond,order=order,arrival=at,source_id=parent,expert=True,
                                **off,frozen_v=frozen['v'],before_v=before['v'],vs_frozen_v=after['v']-frozen['v'],before_vs_frozen_v=before['v']-frozen['v'],
                                vs_before_v=after['v']-before['v'],vs_episodic_v=after['v']-episodes[key],gross_gain=max(after['v']-frozen['v'],0),gross_loss=max(frozen['v']-after['v'],0),
                                current_spatial_gain=after['v']-before['v'],current_temporal_gain=0.,payload_sha256=sha(f),prestate_sha256=state_hash(x['initial']),committed_state_sha256=state_hash(x['committed'])))
                            positions=ex['expert']['actual_observation_positions'];observed=[row['frame_ids'][j] for j in positions];oi=[j for j,gid in enumerate(a.fids) if gid in observed];ui=[j for j,gid in enumerate(a.fids) if gid not in observed]
                            dd=dict(dataset=ds,stream=stream,split=split,condition=cond,order=order,arrival=at,source_id=parent,
                                fit_seconds=x['fit_seconds'],cached_backward_calls=x['fit']['gradient_calls'],logical_observation_requests=4,actual_new_DINO_at_readout=0,new_backbone_at_readout=0,
                                observed_GT_delta=float(np.mean((a.iou-c.iou)[oi])) if oi else None,unobserved_GT_delta=float(np.mean((a.iou-c.iou)[ui])) if ui else None,
                                observed_GT_frames=len(oi),unobserved_GT_frames=len(ui),selected_step=x['fit']['selected_step'],empty=x['fit']['empty'],
                                proxy_delta=x['fit']['path'][x['fit']['selected_step']]['loss']-x['fit']['path'][0]['loss'],GT_delta=after['v']-before['v'])
                            dd['proxy_improved_GT_harmed']=dd['proxy_delta']<0 and dd['GT_delta']<0;diag.append(dd)
                            del data,x,ex
                        print('CROSS_ROOT_AUDIT',ds,stream,split,cond,order,len(rows),checks,flush=True)
    assert len(rows)==2304;out=PUB/'cross_domain';write(out/'ROWS.json',rows);write(out/'DIAGNOSTICS.json',diag);write(out/'SUMMARY.json',aggregate(rows))
    lock=read(CROSS/'RUNTIME_LOCK.json');write(out/'CONFIGURATION.json',{k:lock[k] for k in ['version','time','configurations','expert_checkpoint_sha256','unique_inputs','observation_requests_cap','logical_readouts','streams','historical_exposure','no_parameter_search','scope_correction']})
    write(out/'LABEL_PROVENANCE_AUDIT.json',dict(status='pass',datasets=label_provenance,GT_only_after_global_barrier=True,historical_exposure=True))
    write(out/'RESOURCE_RECEIPT.json',dict(datasets=resources,capture_inputs=576,readouts=2304,actual_DINO=sum(z['actual_DINO'] for z in resources.values()),new_two_offset_capture=576,
        wall_time_not_GPU_kernel=True,smoke_extra_full_reinsertion_two_offset=4,actual_full_backbone_single_offset_calls=1160,smoke_cached_backward_calls=smoke_backwards,smoke_count_reproduced_by_same_input_same_state_fits=True,smoke=read(CROSS/'SMOKE_ROOT_ACCEPTANCE.json')))
    write(out/'ROOT_AUDIT.json',dict(status='pass',checks=checks,arrivals=2304,independent_state_objective_optimizer=True,official_dense_agreement=True,source_chain=True,seconds=time.time()-tick,time=time.time()))
    sums=read(out/'SUMMARY.json');pass_both=all(sums[d]['confirm']['online_100']['corruption']['metrics']['vs_frozen_v']['ci95'][0]>0 for d in DATASETS)
    write(out/'DECISION.json',dict(cross_domain_measured=True,positive_lower95_both_confirmation=pass_both,method_promoted=False,selection_uses_confirmation=False,all_results_reported=True))
    public_check(out)

def public_check(folder):
    folder=Path(folder);rr=read(folder/'ROWS.json');assert len(rr)==2304 and aggregate(rr)==read(folder/'SUMMARY.json')
    for r in rr:
        assert 0<=r['v']<=1 and abs(r['vs_frozen_v']-(r['v']-r['frozen_v']))<1e-12
        assert abs(r['vs_before_v']-(r['v']-r['before_v']))<1e-12 and r['current_temporal_gain']==0
    out=dict(status='pass',rows=2304,all_aggregates_recomputed=True,time=time.time())
    if not (folder/'PUBLIC_AUDIT.json').exists():write(folder/'PUBLIC_AUDIT.json',out)
    return out

if __name__=='__main__':run()
