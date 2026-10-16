"""Independent pre-GT saved-witness and actual full-readout study check."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,COMP,SEARCH,read,write,sha,load_case,roster,verify,event_output_positions


def main():
    import torch
    import numpy as np
    from scripts.audit_stvg_soft_ep_v1 import arr,treeeq
    from scripts.review_stvg_component_trial_v1 import fit_audit as audit_full
    from scripts.audit_stvg_spatial_lite_fit_v1 import fit_audit as audit_lite
    from scripts.run_stvg_table1_student004 import stable_fit_key
    verify();b=BASE/'trials'/sys.argv[1];rt=read(b/'RUNTIME.json');design=read(b/'DESIGN_LOCK.json')
    for rel,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/rel)==h,rel
    bar=read(b/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete' and bar['count']==128 and not bar['GT_read']
    assert bar['runtime_sha256']==sha(b/'RUNTIME.json')
    refs={r['case_id']:r for r in read(b/'REFERENCE_LOCK.json')['records']};times={r['case_id']:r for r in read(b/'TEMPORAL_LOCK.json')['records']};rr={r['probe_alias']:r for r in roster()}
    counts=dict(accepted=0,proposals=0,no_reference=0,full_readouts=0,baseline_exact=0)
    seen=set()
    for x in bar['files']:
        p=ROOT/x['path'];receipt=p.with_suffix('.receipt.json')
        assert sha(p)==x['sha256'] and sha(receipt)==x['receipt_sha256']
        d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];assert cid in rr and cid not in seen;seen.add(cid)
        r=rr[cid];case,sp=load_case(r);ref=refs[cid];ds=r['dataset']
        assert d['source_restored'] and not d['GT_read'] and d['runtime_sha256']==sha(b/'RUNTIME.json')
        assert d['source_file_sha256']==sha(sp) and d['source_case_sha256']==case['source_case_sha256']
        assert d['reference_records']==ref and d['temporal_record']==times[cid]
        assert np.array_equal(d['arms']['F']['boxes'],case['native_boxes_xyxy_normalized']) and np.array_equal(d['arms']['T']['boxes'],d['arms']['F']['boxes'])
        assert d['arms']['F']['interval']==case['Native_interval'] and d['arms']['T']['interval']==times[cid]['selected_interval']
        f=d['fits']['M2'];o=d['readouts']['M2']
        if not ref['anchors']:
            assert f is o is None
            assert np.array_equal(d['arms']['S_M2']['boxes'],d['arms']['F']['boxes']) and np.array_equal(d['arms']['TS_M2']['boxes'],d['arms']['T']['boxes'])
            counts['no_reference']+=1;continue
        assert f['config']==design['configs'][ds]['M2'] and f['settings']==design['response_settings'][ds]['M2']
        assert f['seed_key']==stable_fit_key(ds,r['query_key_sha256']) and treeeq(f['initial_state'],case['source_light_initial_state'])
        assert f['study_recipe']==design['recipe'] and f['study_response_scope']==design['response_scope']
        assert f['predicted_event_interval']==times[cid]['selected_interval']
        if design['recipe']=='full':
            audit=audit_full(f,ref,{})
            if b.name=='L000':
                parent=torch.load(COMP/'trials/A000/student'/(cid+'.pt'),map_location='cpu',weights_only=False)
                assert torch.equal(f['final_cached_decoder_boxes'],parent['fits']['M2']['final_cached_decoder_boxes'])
                assert treeeq(f['final_state'],parent['fits']['M2']['final_state'])
                counts['baseline_exact']+=1
        else:
            support=event_output_positions(case['frame_ids'],times[cid]['selected_interval']) if design['response_scope']=='event_support' else None
            audit=audit_lite(f,ref,design['recipe'],response_positions=support)
        counts['accepted']+=int(audit['accepted']);counts['proposals']+=int(audit['proposals'])
        assert torch.equal(o['boxes'].cpu(),f['final_cached_decoder_boxes'].cpu());counts['full_readouts']+=1
        bb=arr(o['boxes']).astype(float);xy=np.concatenate([bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2],axis=1)
        assert np.array_equal(xy,d['arms']['S_M2']['boxes']) and np.array_equal(xy,d['arms']['TS_M2']['boxes'])
        assert f['matched_efficiency']['fit_wall_cuda_synchronized_seconds']>=0
        assert not f['matched_efficiency']['qualification_enabled']
    assert len(seen)==128
    write(b/'ROOT_READBACK.json',dict(status='pass',count=128,counters=counts,GT_read=False,CPU_only=True,runtime_sha256=sha(b/'RUNTIME.json'),barrier_sha256=sha(b/'STUDENT_GLOBAL_BARRIER.json'),scope='complete saved source/rollout/loss/Adam/solver/state/rollback and actual full-model readout; not independent full backward'))
    files=list(bar['files'])+[{'path':x['receipt_path'],'sha256':x['receipt_sha256']}for x in bar['files']]
    for p in [b/n for n in ['DESIGN_LOCK.json','RUNTIME.json','ROOT_READBACK.json','STUDENT_GLOBAL_BARRIER.json','REFERENCE_LOCK.json','TEMPORAL_LOCK.json']]:files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    write(b/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=128,files=files,GT_read=False,student_barrier_sha256=sha(b/'STUDENT_GLOBAL_BARRIER.json'),root_readback_sha256=sha(b/'ROOT_READBACK.json'),scorer_sha256=sha(ROOT/'scripts/measure_stvg_lite_event_v1.py')))
    print('ROOT_LITE_EVENT_PASS',b.name,128,flush=True)


if __name__=='__main__':main()
