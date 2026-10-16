"""Saved-evidence closure for the bounded phase one, never launches research."""
from pathlib import Path
import collections,json,math,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_referent_evidence_p0_v1 import BASE,CONTROL,PRIOR,read,put,sha,verify,UNIT
def audit():
    import numpy as np,torch
    from scripts.review_stvg_referent_evidence_p0_v1 import summary_rows
    verify();verify('SOURCE')
    for rel,h in read(BASE/'ROOT_REVIEW_RUNTIME.json')['pins'].items():assert sha(ROOT/rel)==h,rel
    original=read(BASE/'PHASE2_DECISION.json');assert original['status']=='not_supported_stop_this_expansion'
    assert read(BASE/'valid_content/SENSITIVITY_COMPLETION.json')['does_not_replace_primary_gate']
    d=read(CONTROL/'POSTSEAL_RESULTS.json');s,p=summary_rows(d['rows']);assert s==d['summary']and p==d['paired'];fits=0
    for item in read(CONTROL/'trials/SOURCE/STUDENT_GLOBAL_BARRIER.json')['files']:
        assert sha(ROOT/item['path'])==item['sha256'];x=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False);assert x['source_restored']and np.array_equal(x['arms']['T']['boxes'],x['arms']['TS_M2']['boxes'])and x['arms']['T']['interval']==x['arms']['TS_M2']['interval']
        f=x['fits']['M2']
        if f:
            fits+=1;assert f['optimizer_steps']==0 and all(torch.equal(v,f['final_state'][k])for k,v in f['initial_state'].items())
    assert fits==23
    for variant in (BASE,BASE/'valid_content'):
        r=read(variant/'RESPONSE_POSTSEAL_DIAGNOSIS.json');assert len(r['frame_rows'])==3072 and len(r['source_rows'])==96 and len(r['pair_rows'])==1152
        for z in r['summary']:
            vals=[q[z['metric']]for q in r['source_rows']if q['dataset']==z['dataset']and q['strategy']==z['strategy']and q[z['metric']]is not None];assert z['valid']==len(vals)and z['mean']==(math.fsum(vals)/len(vals)if vals else None)
        for z in read(variant/'RESPONSE_GLOBAL_BARRIER.json')['records']:assert sha(ROOT/z['path'])==z['sha256']
    for group in ('source_control001','response001'):
        b=BASE/'publication'/group;receipt=read(b/'FINAL_GITHUB_RECEIPT.json');remote=read(b/'REMOTE_BYTES_READBACK.json');assert receipt['status']==remote['status']=='pass'
        for z in read(b/'EXPORT_MANIFEST.json')['files']:assert sha(ROOT/z['path'])==z['sha256']
    exits={}
    for mode in ('control_worker','control_root','responses'):
        x=read(BASE/'execution'/mode/'WORKER_EXIT.json');assert x['exit_code']==0;exits[mode]=x['wall_seconds']
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MainPID,ActiveState,Result'],text=True).splitlines());assert props['MainPID']=='0'and props['Result']=='success'
    put(BASE/'ROOT_STUDY_READBACK.json',dict(status='pass',count=32,response_subset_count=16,Source_control_actual_fits=23,complete_model_readouts=23,Source_control_accepted_steps=0,Source_control_nonzero_final=0,native_vision_passes=64,new_decoder_generation=0,evidence_conditioned_student_updates=0,whole_A_B_C_run=False,primary_and_sensitivity_scalar_reaggregation=True,GT_read=True,worker_seconds=exits,unit_final=props,scope='all saved scalar/selection/math/receipt checks and actual full Student readouts; not a new independent backward or semantic identity GT'))
    print(json.dumps(dict(status='pass',count=32,worker_seconds=exits)))
if __name__=='__main__':audit()
