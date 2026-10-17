"""Fixed I1/I2 and feedback, marginal interval training, real integrated readout."""
from pathlib import Path
import gc,inspect,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
BASE=ROOT/'artifacts/stvg_temporal_marginal_train256_v1';PREV=ROOT/'artifacts/stvg_temporal_marginal_i2_256_v1';B0=ROOT/'artifacts/stvg_temporal_hypothesis256_v1';T2=ROOT/'artifacts/stvg_temporal_t2_vs_t4_v1'
def verify():
    for p,h in read(BASE/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    if (BASE/'ACTIVE_EXECUTION.json').exists():assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']
def init():
    from scripts.stvg_temporal_marginal_train_fit256_v1 import CFG
    assert not BASE.exists();BASE.mkdir();(BASE/'fresh_prefix').mkdir()
    assert read(PREV/'FINAL_COMPLETION.json')['count']==256 and read(PREV/'ROOT_SCALAR_READBACK.json')['status']=='pass'
    for name in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','INPUT_BARRIER.json','NO_UPDATE_CONTROL_BARRIER.json','TEACHER_BARRIER.json','TEACHER_ROOT_READBACK.json','SIGNAL_COMPLETION.json']:(BASE/name).write_bytes((PREV/name).read_bytes())
    rr=read(BASE/'ROSTER_LOCK.json')['records'];assert len(rr)==256 and {d:len({r['parent_sha256']for r in rr if r['dataset']==d})for d in ['hc2','vidstg']}=={'hc2':128,'vidstg':128}
    assert CFG==read(PREV/'METHOD_LOCK.json')['config'];old=read(PREV/'METHOD_LOCK.json');write(BASE/'METHOD_LOCK.json',dict(status='locked_before_new_fits',count=256,per_domain=128,only_change='candidate prior/Teacher target/Student likelihood all use exact merged physical interval probability',I1_I2_inputs_and_Teacher_bytes_unchanged=True,config=CFG,original_optimizer_source_sha256=sha(ROOT/'scripts/stvg_evidence_temporal_fit_v1.py'),final_decoder='unchanged actual FP32 offset MAP merge; NOT marginal MAP',control='unchanged separate Source marginal MAP no-update',missing_prefix_cases=old['missing_prefix_cases'],new_Teacher_calls=0,new_Spatial_fits=0,external_Actionness=False,CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),GT_read=False))
    names=['stvg_temporal_marginal_train256_v1.py','stvg_temporal_marginal_train_fit256_v1.py','continue_stvg_temporal_marginal_train256_v1.py','score_stvg_temporal_marginal_train256_v1.py','stvg_temporal_marginal_i2_256_v1.py','stvg_temporal_marginal_i2_select256_v1.py','stvg_temporal_hypothesis_fit32_v1.py','stvg_temporal_semantic_view_fit32_v1.py','stvg_evidence_temporal_fit_v1.py','stvg_temporal_semantic_grounder32_v1.py','stvg_temporal_semantic_grounder32_v2.py','stvg_factorial_native256_v1.py','stvg_table1_temporal_adapter_v1.py','run_decota_paper_main_v1.py','stvg_pipeline_speed_v1.py','launch_stvg_instance_identity_v1.py']
    paths=[ROOT/'scripts'/n for n in names]+[ROOT/'protocols/stvg_temporal_marginal_train256_v1.md',ROOT/'methods/CURRENT_METHOD.json']+[BASE/n for n in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','INPUT_BARRIER.json','NO_UPDATE_CONTROL_BARRIER.json','TEACHER_BARRIER.json','TEACHER_ROOT_READBACK.json','SIGNAL_COMPLETION.json','METHOD_LOCK.json']]+[PREV/n for n in ['PRIMARY_GLOBAL_PREDICTION_BARRIER.json','POSTSEAL_RESULTS.json']]+[B0/n for n in ['PRIMARY_GLOBAL_PREDICTION_BARRIER.json','POSTSEAL_RESULTS.json']]+[T2/n for n in ['GLOBAL_PREDICTION_BARRIER.json','POSTSEAL_RESULTS.json']]
    write(BASE/'EXECUTION_RUNTIME.json',dict(status='locked',count=256,pins={str(p.relative_to(ROOT)):sha(p)for p in paths},GT_read=False));write(BASE/'STATUS.json',dict(status='prepared',count=256,GT_read=False),mutable=True)
def attach():
    import numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_marginal_train_fit256_v1 as f
    verify();sys.addaudithook(guard);jm={r['case_id']:r for r in read(BASE/'INPUT_BARRIER.json')['jobs']};tm={r['case_id']:r for r in read(BASE/'TEACHER_BARRIER.json')['files']};records=[];pixels=0
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=r['probe_alias'];c,_=load_case(r);jr=jm[cid];tr=tm[cid];assert sha(ROOT/jr['path'])==jr['sha256']and sha(ROOT/tr['path'])==tr['sha256'];j=read(ROOT/jr['path']);t=read(ROOT/tr['path']);assert t['input_sha256']==jr['sha256']and j['source_case_sha256']==c['source_case_sha256']
        for h in j['selection']['hypotheses']:assert sha(ROOT/h['video_path'])==h['video_sha256'];pixels+=1
        off=f.native.offsets(c,_hidden_from_source_case(c));pre=f.prepare(off,j['selection']['hypotheses'],[e['reward']for e in t['evidence']]);assert abs(pre['Native_interval_probabilities'][1]-j['selection']['I2_marginal_mass'])<1e-12
        records.append(dict(case_id=cid,status='pass',I1_I2_source_video_query_feedback_exact=True,Native_interval_probabilities=pre['Native_interval_probabilities'].tolist(),Native_conditional=pre['Native_conditional'].tolist(),Teacher_conditional=pre['q'].tolist(),no_information=pre['no_information'],GT_read=False));del c,off;gc.collect()
    write(BASE/'FIXED_EVIDENCE_READBACK.json',dict(status='pass',count=256,pixel_hash_checks=pixels,all_I1_I2_and_feedback_reused_exact=True,new_Teacher_calls=0,records=records,GT_read=False));write(BASE/'ATTACH_COMPLETION.json',dict(status='complete',count=256,GT_read=False))
def fit():
    import numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_marginal_train_fit256_v1 as f
    verify();sys.addaudithook(guard);assert read(BASE/'ATTACH_COMPLETION.json')['count']==256;jm={r['case_id']:r for r in read(BASE/'INPUT_BARRIER.json')['jobs']};tm={r['case_id']:r for r in read(BASE/'TEACHER_BARRIER.json')['files']};files=[];targets=[];checks=0
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=r['probe_alias'];c,_=load_case(r);j=read(ROOT/jm[cid]['path']);assert sha(ROOT/tm[cid]['path'])==tm[cid]['sha256'];t=read(ROOT/tm[cid]['path']);off=f.native.offsets(c,_hidden_from_source_case(c));hs=j['selection']['hypotheses'];z=f.fit(off,hs,[e['reward']for e in t['evidence']]);checks+=f.audit(off,z);view=list(c['Native_interval'])if z['no_information']else hs[int(np.argmax(z['Teacher_conditional']))]['interval_frames']
        rec=save_pt(BASE/'targets'/(cid+'.pt'),dict(view_intervals=dict(HYP=view),marginal_no_update_interval=j['selection']['marginal_no_update_interval'],GT_read=False));rec['case_id']=cid;targets.append(rec);rec=save_pt(BASE/'fits'/(cid+'.pt'),dict(case_id=cid,dataset=r['dataset'],fits=dict(HYP=z),source_case_sha256=c['source_case_sha256'],GT_read=False));rec['case_id']=cid;files.append(rec);write(BASE/'FIT_STATUS.json',dict(status='running',done=len(files),total=256,count=256,GT_read=False),mutable=True);print('MARGINAL_TRAIN_FIT',len(files),flush=True);del c,off,z;gc.collect()
    write(BASE/'FIT_BARRIER.json',dict(status='complete',count=256,files=files,independent_checks=checks,GT_read=False));write(BASE/'TARGET_BARRIER.json',dict(status='complete',count=256,files=targets,GT_read=False))
def readout():
    from scripts import stvg_temporal_marginal_i2_256_v1 as prev
    src=inspect.getsource(prev.readout);ns=dict(prev.__dict__);ns.update(BASE=BASE,verify=verify);exec(compile(src,__file__,'exec'),ns);ns['readout']()
if __name__=='__main__':globals()[sys.argv[1]]()
