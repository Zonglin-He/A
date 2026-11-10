"""Same ABC full-source forwards, capture decoder0 prefix and fixed raw anchors."""
import argparse,sys,time,torch
from common62 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,source_capture,fingerprint,PREFIX_KEYS
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--engineering',action='store_true');a=ap.parse_args();p=read(a.plan);pin(p);rows=read(ROOT/p['manifest']);rows=engineering_rows(rows) if a.engineering else rows;out=ROOT/('research/round62/engineering_capture_attempt001' if a.engineering else p['capture_root']);assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();model=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(model);calls=0;start=time.perf_counter()
 if not a.engineering:assert read(ROOT/'research/round62/engineering_combo_attempt001/RESULT.json')['state'].startswith('PASS_')
 try:
  for i,row in enumerate(rows):
   sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';AB=ROOT/p['AB_root'];C=ROOT/p['C_root'];assert sha(AB/rel)==read(AB/'SEALED.json')['files'][rel] and sha(C/rel)==read(C/'SEALED.json')['files'][rel];ab=load(AB/rel);cc=load(C/rel);case=read(AB/'cases_PRIVATE'/(sid+'.json'));seedC=int(hashlib.sha256(('R57_student_C_v1|'+sid+'|'+str(row['seed'])).encode()).hexdigest()[:16],16);payload={}
   offsets=model.tokenizer([row['caption']+' . not mentioned'],return_offsets_mapping=True)['offset_mapping'][0];payload['token_offsets']=np.array(offsets,np.int64)
   for v,seed,old,pre in [('A',row['seed'],ab,'A_'),('B',case['B_seed'],ab,'B_'),('C',seedC,cc,'')]:
    points,indices,_=sample_points(dict(row,seed=seed));first={}
    def before(_m,args,kwargs):
     assert len(args)==6 and kwargs['detected_feats'] is None
     for key,val in zip(PREFIX_KEYS,args):
      if val is not None:first[key]=val.detach().clone().cpu().numpy()
    hook=model.decoder[0].register_forward_pre_hook(before,with_kwargs=True)
    try:z,receipt=source_capture(model,points,indices,row['caption'],retain_prefix=True);calls+=1
    finally:hook.remove()
    for key in ['boxes','scores','h','query','tokens','token_mask','point_indices']:assert np.array_equal(z[key],old[pre+key]),('same_original_ABC',v,key)
    for key,val in first.items():payload[v+'_'+key]=val
    for key in ['h','boxes','scores','query','tokens','token_mask','query_anchor_xyz','query_points_sample_inds','point_indices']:payload[v+'_'+key]=z[key]
    assert len(offsets)==len(z['tokens'])
   np.savez_compressed(out/rel,**payload)
   if (i+1)%32==0:print('R62_PREFIX0',i+1,len(rows),calls,flush=True)
  assert fingerprint(model)==fp;cost={'n':len(rows),'new_full_source_forwards':calls,'source_unchanged':True,'all_ABC_original_outputs_and_pointindices_exact':True,'source_fingerprint':fp,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COSTS.json',cost);seal(out,'R62_SOURCE_FIRST_DECODER_PREFIX_ANCHOR_LINEAGE_NO_GT',**cost);print(cost,flush=True)
 except BaseException as error:dump(out/'FAILED.json',{'type':type(error).__name__,'error':str(error),'forwards_completed':calls,'automatic_retry':False});raise
if __name__=='__main__':main()
