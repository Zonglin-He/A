"""Actual pre-fusion seed masking and complete transformer recomputation."""
from common75 import *
import torch,time,argparse
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
from interface66 import replay,capture
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');ap.add_argument('--held',action='store_true');a=ap.parse_args();assert not(a.engineering and a.held);p=read(H/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H/'experiment/CAPTURE_CODE_LOCK.json'));rr=rows();views=['M3'] if a.held else ['M1','M2']
 if a.engineering:rr=[min([r for r in rr if r['platform']==pl],key=lambda r:hashlib.sha256(('R75eng|'+r['id']).encode()).hexdigest()) for pl in ['drone','quad']]
 if a.held:assert read(ROOT/'crossplatform/round75/fit336_attempt001/SEALED.json')['n']==336
 elif not a.engineering:assert read(H/'engineering_capture_attempt001/SEALED.json')['state'].startswith('PASS_')
 out=H/'engineering_capture_attempt001' if a.engineering else ROOT/('crossplatform/round75/held336_attempt001' if a.held else 'crossplatform/round75/masked336_attempt001');assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();m=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(m);start=time.perf_counter();calls=0;fullcalls=0;ss=read(ROOT/p['prefusion_source_root']/'SEALED.json');stats=[]
 try:
  for i,r in enumerate(rr):
   sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(ROOT/p['prefusion_source_root']/rel)==ss['files'][rel];z=original(r);got={};proof=[]
   if a.engineering:
    pts,ix,_=sample_points(r);fresh,_=capture(m,pts,ix,r['caption']);fullcalls+=1
    for k in ['h','query','tokens','boxes','scores','vis0','lang0','pos0','query_points_sample_inds']+[f'position{j}' for j in range(6)]:assert np.array_equal(fresh[k],z[k]),('full_source_bridge',k)
    y=replay(m,z);calls+=1
    for k in ['h','query','tokens']:assert np.array_equal(y[k],z[k]),('zero_replay',k)
   for v in views:
    mask=mask_for(r,z,v);y=replay(m,z,mask);calls+=1;assert y['proof']['max_blocked_attention']==0. and y['proof']['visual_attention_calls']==12 and y['proof']['masked_query_keys_n']==0
    assert not np.array_equal(y['h'],z['h']);assert not np.array_equal(y['tokens'],z['tokens']);rec=dict(y['proof'],view=v,source_query_anchor_seeds_retained=True,masked_h_diff_L2=float(np.linalg.norm(y['h']-z['h'])),fused_text_diff_L2=float(np.linalg.norm(y['tokens']-z['tokens'])),semantic_loss_text_will_use_complete_reference_not_this_dynamic_text=True)
    if a.engineering:
     vv=z['vis0'].copy();vv[:,mask]=np.float32(999);other=replay(m,z,mask,alternate_vis0=vv);calls+=1;assert np.array_equal(other['h'],y['h']) and np.array_equal(other['tokens'],y['tokens']);rec['masked_value_leakage_test_exact']=True
    for k in ['h','query','tokens']:got[v+'_'+k]=y[k]
    got[v+'_mask']=mask;proof.append(rec)
   np.savez_compressed(out/rel,**got);dump(out/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'views':proof,'reference_text_fixed_downstream':True,'source_box_geometry_unchanged':True});stats+=proof
   if(i+1)%32==0:print('R75_MASKED',views,i+1,len(rr),calls,flush=True)
  assert fingerprint(m)==fp;cost={'n':len(rr),'new_cross_encoder_six_decoder_suffix_forwards':calls,'new_full_source_forwards':fullcalls,'source_unchanged':True,'M3_used_for_fitting':False,'masked_seed_count':int(mask.sum()),'original_seeds':len(mask),'source_query_anchors':256,'all_blocked_visual_key_attention_zero':True,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COUNTS.json',cost);seal(out,'PASS_R75_PRE_FUSION_MASKED_RECOMPUTED_QUERY_AND_DYNAMIC_TEXT_NO_GT',n=len(rr));print(cost,flush=True)
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'suffix_completed':calls,'full_completed':fullcalls,'no_scoring':True});raise
if __name__=='__main__':main()
