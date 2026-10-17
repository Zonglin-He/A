"""Independent saved mathematics and actual complete-model prediction audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,write,sha,verify,load_case,event_output_positions


def audit_m0(f,ref,key,cfg):
    import numpy as np
    from scripts.audit_stvg_soft_ep_v1 import arr,err,iou,treeeq,rollout_audit,_optimizer
    assert f['config']==cfg and f['seed_key']==key and f['active_parameters']==1792 and not f['GT_read']
    assert f['validation']=='weighted_reference_IoU_non_decrease_1e-7_then_stop'
    anchors=[a for a in ref['anchors']if a['weight']>0]
    pos=[a['position']for a in anchors];ev=np.asarray([a['box_cxcywh']for a in anchors]);weights=np.asarray([a['weight']for a in anchors])
    assert f['observation_positions']==pos
    previous=f['initial_state'];opt=None;accepted=proposed=0;stopped=False
    for step,row in enumerate(f['path']):
        assert row['step']==step and treeeq(row['state'],previous)
        if opt is not None:assert treeeq(row['optimizer_before'],opt)
        if row.get('skipped'):
            assert stopped and row['reason']=='stopped_after_rejected_step'
            continue
        assert not stopped
        rollout_audit(dict(row,boxes_before=row['boxes'],per_anchor_loss=row['anchor_loss']),ev,weights,cfg,pos,key,step)
        delta=0.0
        if row['proposed']:
            proposed+=1
            a=dict(row,state_before=row['state'],gradient=row['update']['gradient'],original_adam_delta=row['update']['raw'])
            assert max(_optimizer(a,list(previous)))<3e-6
            off=0;direction=arr(row['update']['raw'])
            for name,old in previous.items():
                n=arr(old).size;wanted=arr(old)+direction[off:off+n].reshape(arr(old).shape);off+=n
                assert err(wanted,row['proposed_state'][name])<3e-7
            before=iou(arr(row['boxes'])[pos],ev);after=iou(arr(row['proposed_boxes'])[pos],ev)
            assert err(before,row['reference_iou_before'])<3e-6 and err(after,row['reference_iou_proposed'])<3e-6
            # Saved FP32 reduction provides the exact threshold bit; float64
            # independently checks geometry without changing admission.
            saved=float((weights*(arr(row['reference_iou_proposed'])-arr(row['reference_iou_before']))).sum()/weights.sum())
            assert bool(row['accepted'])==(saved>=-1e-7)or abs(saved+1e-7)<2e-7
        else:
            assert not row['rollout']['informative'] and row['update'] is None and not row['accepted']
        if row['accepted']:
            accepted+=1;assert treeeq(row['update_state'],row['proposed_state']) and treeeq(row['optimizer_after'],row['optimizer_proposed'])
        else:
            assert treeeq(row['update_state'],previous) and treeeq(row['optimizer_after'],row['optimizer_before'])
            stopped=bool(row['proposed'])
        previous=row['update_state'];opt=row['optimizer_after']
    assert treeeq(previous,f['final_state']) and accepted==f['optimizer_steps'] and proposed==f['proposed_optimizer_steps']
    return dict(accepted=accepted,proposals=proposed)


def main():
    import torch,numpy as np
    from scripts.audit_stvg_soft_ep_v1 import arr,treeeq
    from scripts.audit_stvg_spatial_lite_fit002 import fit_audit as audit_lite
    from scripts.review_stvg_component_trial_v1 import fit_audit as audit_full
    from scripts.run_stvg_table1_student004 import stable_fit_key
    from scripts.stvg_lite_recovery002_common import verify_recovery
    verify_recovery();verify();cohort,tid=sys.argv[1:3];scope=BASE if cohort=='development'else BASE/'confirmation';count=128 if cohort=='development'else 64
    roster_path=ROOT/'artifacts/stvg_soft_ep_coordinate128_v1/ROSTER_LOCK.json'if cohort=='development'else scope/'ROSTER_LOCK.json'
    roster={r['probe_alias']:r for r in read(roster_path)['records']};assert len(roster)==count
    b=scope/'trials'/tid;rt=read(b/'RUNTIME.json');design=read(b/'DESIGN_LOCK.json')
    recipe=read(BASE/'FINAL_RECIPE_LOCK.json')
    assert design['recipe_sha256']==sha(BASE/'FINAL_RECIPE_LOCK.json') and recipe['status']=='locked'
    assert design['recipe']==recipe['selected_spatial_recipe'] and design['response_scope']==recipe['selected_response_scope']
    for ds in ('hc2','vidstg'):
        assert design['configs'][ds]['M0']==design['configs'][ds]['M2']==recipe['configs'][ds]
    for rel,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/rel)==h,rel
    assert rt['pins']['scripts/review_stvg_lite_complete002.py']==sha(Path(__file__))
    bar=read(b/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete' and bar['count']==count and not bar['GT_read'] and bar['runtime_sha256']==sha(b/'RUNTIME.json')
    refs={r['case_id']:r for r in read(b/'REFERENCE_LOCK.json')['records']};times={r['case_id']:r for r in read(b/'TEMPORAL_LOCK.json')['records']}
    seen=set();counters={mode:dict(accepted=0,proposals=0,no_reference=0)for mode in ('M0','M2')}
    for entry in bar['files']:
        p=ROOT/entry['path'];assert sha(p)==entry['sha256'] and sha(ROOT/entry['receipt_path'])==entry['receipt_sha256']
        receipt=read(ROOT/entry['receipt_path']);assert receipt['sha256']==sha(p) and receipt['status']=='complete' and not receipt['GT_read']
        d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];assert cid in roster and cid not in seen;seen.add(cid)
        rec=roster[cid];case,cp=load_case(rec);ref=refs[cid];ds=rec['dataset'];key=stable_fit_key(ds,rec['query_key_sha256'])
        assert d['source_restored'] and not d['GT_read'] and d['runtime_sha256']==sha(b/'RUNTIME.json') and d['source_file_sha256']==sha(cp) and d['source_case_sha256']==case['source_case_sha256']
        assert d['reference_records']==ref and d['temporal_record']==times[cid]
        assert np.array_equal(d['arms']['F']['boxes'],case['native_boxes_xyxy_normalized']) and np.array_equal(d['arms']['T']['boxes'],d['arms']['F']['boxes'])
        assert d['arms']['F']['interval']==case['Native_interval'] and d['arms']['T']['interval']==times[cid]['selected_interval']
        if design['reference_clock']=='native':assert times[cid]['selected_interval']==case['Native_interval']
        for mode in ('M0','M2'):
            f=d['fits'][mode];out=d['readouts'][mode]
            if not ref['anchors']:
                assert f is out is None
                for prefix in ('S_','TS_'):assert np.array_equal(d['arms'][prefix+mode]['boxes'],d['arms']['F']['boxes'])
                counters[mode]['no_reference']+=1;continue
            assert f['config']==design['configs'][ds][mode] and f['seed_key']==key and treeeq(f['initial_state'],case['source_light_initial_state'])
            if mode=='M0':audit=audit_m0(f,ref,key,design['configs'][ds][mode])
            elif design['recipe']=='full':audit=audit_full(f,ref,{})
            else:
                positions=event_output_positions(case['frame_ids'],times[cid]['selected_interval'])if design['response_scope']=='event_support'else None
                audit=audit_lite(f,ref,design['recipe'],response_positions=positions)
            for k in ('accepted','proposals'):counters[mode][k]+=audit[k]
            assert torch.equal(out['boxes'].cpu(),f['final_cached_decoder_boxes'].cpu())
            bb=arr(out['boxes']).astype(float);xy=np.concatenate([bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2],axis=1)
            for prefix in ('S_','TS_'):assert np.array_equal(xy,d['arms'][prefix+mode]['boxes'])
            assert d['arms']['S_'+mode]['interval']==case['Native_interval'] and d['arms']['TS_'+mode]['interval']==times[cid]['selected_interval']
    assert len(seen)==len(bar['files'])==count
    write(b/'ROOT_READBACK.json',dict(status='pass',count=count,counters=counters,GT_read=False,CPU_only=True,runtime_sha256=sha(b/'RUNTIME.json'),barrier_sha256=sha(b/'STUDENT_GLOBAL_BARRIER.json'),reader_recovery_sha256=sha(BASE/'RECOVERY_RUNTIME002.json'),scope='all source, saved rollout/loss/Adam/solver/rollback and actual complete-model final output; no independent full backward'))
    files=list(bar['files'])+[dict(path=x['receipt_path'],sha256=x['receipt_sha256'])for x in bar['files']]
    for name in ('ROOT_READBACK.json','STUDENT_GLOBAL_BARRIER.json','RUNTIME.json','DESIGN_LOCK.json','REFERENCE_LOCK.json','TEMPORAL_LOCK.json'):
        p=b/name;files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    write(b/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=count,files=files,GT_read=False,root_readback_sha256=sha(b/'ROOT_READBACK.json'),student_barrier_sha256=sha(b/'STUDENT_GLOBAL_BARRIER.json'),scorer_sha256=sha(ROOT/'scripts/score_stvg_lite_complete_v1.py')))
    print('LITE_COMPLETE_ROOT_PASS',cohort,tid,count,flush=True)


if __name__=='__main__':main()
