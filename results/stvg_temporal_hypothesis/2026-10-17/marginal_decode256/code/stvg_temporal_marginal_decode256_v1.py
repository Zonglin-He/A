"""Isolated decoder-only CPU measurement from exact sealed actual B2 head logits."""
from pathlib import Path
import gc,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
BASE=ROOT/'artifacts/stvg_temporal_marginal_decode256_v1';PREV=ROOT/'artifacts/stvg_temporal_marginal_train256_v1'
def verify():
    for p,h in read(BASE/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    if (BASE/'ACTIVE_EXECUTION.json').exists():assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']
def init():
    assert not BASE.exists();BASE.mkdir();assert read(PREV/'FINAL_COMPLETION.json')['count']==256 and read(PREV/'ROOT_SCALAR_READBACK.json')['status']=='pass'
    for name in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','INPUT_BARRIER.json','NO_UPDATE_CONTROL_BARRIER.json','TEACHER_BARRIER.json','TEACHER_ROOT_READBACK.json','FIT_BARRIER.json']:(BASE/name).write_bytes((PREV/name).read_bytes())
    for old,new in [('PRIMARY_GLOBAL_PREDICTION_BARRIER.json','UPSTREAM_ACTUAL_HEAD_BARRIER.json'),('PRIMARY_ROOT_READBACK.json','UPSTREAM_ACTUAL_HEAD_ROOT.json')]:(BASE/new).write_bytes((PREV/old).read_bytes())
    rr=read(BASE/'ROSTER_LOCK.json')['records'];assert len(rr)==256 and {d:len({r['parent_sha256']for r in rr if r['dataset']==d})for d in ['hc2','vidstg']}=={'hc2':128,'vidstg':128}
    write(BASE/'METHOD_LOCK.json',dict(status='locked_before_new_decode',count=256,only_change='final physical-interval MAP after exact summation over latent states',fit_and_every_upstream_input_unchanged=True,config=read(PREV/'METHOD_LOCK.json')['config'],probability_precision='actual integrated FP32 head logits; float64 legal-grid normalization/summation exactly as Source marginal control',tie_rule='sorted physical row-major first maximum',new_models=0,new_Teacher_calls=0,new_fits=0,final_parameters='exact saved B2 theta; no reoptimization',no_information='zero adaptation; same marginal decoder, exact no-update control',primary='B_MARGINAL_DECODE_minus_MARGINAL_NO_UPDATE',readout_contrast='B_MARGINAL_DECODE_minus_B_MARGINAL_TRAIN',GT_read=False))
    names=['stvg_temporal_marginal_decode256_v1.py','stvg_temporal_marginal_decode_util256_v1.py','test_stvg_temporal_marginal_decode256_v1.py','continue_stvg_temporal_marginal_decode256_v1.py','score_stvg_temporal_marginal_decode256_v1.py','stvg_temporal_semantic_view_fit32_v1.py']
    paths=[ROOT/'scripts'/n for n in names]+[ROOT/'protocols/stvg_temporal_marginal_decode256_v1.md',ROOT/'methods/CURRENT_METHOD.json',PREV/'POSTSEAL_RESULTS.json',PREV/'ROOT_SCALAR_READBACK.json',PREV/'EXECUTION_RUNTIME.json']+list(BASE.glob('*.json'))
    write(BASE/'EXECUTION_RUNTIME.json',dict(status='locked',count=256,pins={str(p.relative_to(ROOT)):sha(p)for p in paths},GT_read=False));write(BASE/'STATUS.json',dict(status='prepared',count=256,GT_read=False),mutable=True)
def decode():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_semantic_view_fit32_v1 as native,stvg_temporal_marginal_decode_util256_v1 as f
    verify();sys.addaudithook(guard);torch.set_num_threads(2)
    for p,h in read(PREV/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    assert read(BASE/'UPSTREAM_ACTUAL_HEAD_ROOT.json')['all_original_query_integrated_head_outputs_exact'];fm={r['case_id']:r for r in read(BASE/'FIT_BARRIER.json')['files']};pm={r['case_id']:r for r in read(BASE/'UPSTREAM_ACTUAL_HEAD_BARRIER.json')['files']};ctrl={r['case_id']:r for r in read(BASE/'NO_UPDATE_CONTROL_BARRIER.json')['records']};files=[];records=[];bins=0;err=0.;zeros=0;changed=0
    def load(r):assert sha(ROOT/r['path'])==r['sha256'];return torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)
    for row in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=row['probe_alias'];c,_=load_case(row);fr=fm[cid];pr=pm[cid];fit=load(fr);p=load(pr);assert fit['source_case_sha256']==p['source_case_sha256']==c['source_case_sha256']and p['fit_sha256']==fr['sha256'];z=fit['fits']['HYP'];off=native.offsets(c,_hidden_from_source_case(c));iv,logits=native.native_decode(off,z['theta']);assert iv==z['final_interval']==p['arms']['HYP_STUDENT']['interval']and all(torch.equal(a,b)for a,b in zip(logits,z['final_logits']));assert np.array_equal(p['arms']['HYP_STUDENT']['boxes'],c['native_boxes_xyxy_normalized'])
        out=[];maxerror=0.
        for zs in [c['zero']['logits'],z['final_logits']]:
            dec,value,ids,M,gs=f.decode(c['records'],zs);D=f.independent_cdf(gs,ids);de=float(np.max(abs(M-D)));assert de<1e-12 and abs(float(D.sum())-1)<1e-12 and D.min()>-1e-12;assert float(D.max())-float(D[ids.index(dec[0]),ids.index(dec[1]-1)])<1e-12;maxerror=max(maxerror,de);bins+=M.size;out.append((dec,value))
        assert out[0][0]==ctrl[cid]['interval']==p['arms']['MARGINAL_NO_UPDATE']['interval'];err=max(err,maxerror)
        if z['no_information']:
            assert z['optimizer_steps']==0 and np.count_nonzero(z['theta'])==0 and all(torch.equal(a,b)for a,b in zip(z['final_logits'],c['zero']['logits']));assert out[0]==out[1];zeros+=1
        changed+=out[1][0]!=out[0][0];r=save_pt(BASE/'predictions'/(cid+'.pt'),dict(case_id=cid,dataset=row['dataset'],arms=dict(B_MARGINAL_DECODE=dict(boxes=np.asarray(c['native_boxes_xyxy_normalized']).copy(),interval=out[1][0]),MARGINAL_NO_UPDATE=p['arms']['MARGINAL_NO_UPDATE']),source_case_sha256=c['source_case_sha256'],reused_B2_fit_sha256=fr['sha256'],actual_integrated_head_prediction_sha256=pr['sha256'],reused_full_logits_previously_bit_exact_actual_head=True,new_GPU_forwards=0,new_fits=0,GT_read=False));r['case_id']=cid;files.append(r);records.append(dict(case_id=cid,dataset=row['dataset'],Source_marginal_interval=out[0][0],adapted_marginal_interval=out[1][0],Source_mode_mass=out[0][1],adapted_mode_mass=out[1][1],differs_from_Source_marginal=out[1][0]!=out[0][0],differs_from_B2_Native_decoder=out[1][0]!=iv,no_information=z['no_information'],previous_optimizer_steps=z['optimizer_steps'],all_logits_and_theta_and_Source_boxes_exact=True,independent_CDF_max_error=maxerror));write(BASE/'READOUT_STATUS.json',dict(status='running',done=len(files),total=256,count=256,GT_read=False),mutable=True);del c,off,fit,z,p;gc.collect()
        if len(files)%32==0:print('MARGINAL_DECODE',len(files),flush=True)
    assert zeros==17;write(BASE/'DECODER_READBACK.json',dict(status='pass',count=256,records=records,independent_CDF_bins=bins,max_probability_error=err,no_information_exact_no_update=zeros,changed_from_same_decoder_no_update=changed,new_GPU_forwards=0,new_Teacher_calls=0,new_fits=0,GT_read=False));write(BASE/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=256,files=files,GT_read=False));write(BASE/'PRIMARY_ROOT_READBACK.json',dict(status='pass',count=256,Source_marginal_control_exact=256,all_full_probability_tables_independently_checked=True,all_actual_prior_head_outputs_reused_exact=True,all_spatial_boxes_unchanged=True,GT_read=False))
if __name__=='__main__':globals()[sys.argv[1]]()
