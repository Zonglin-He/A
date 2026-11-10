"""Recover missing B/C decoder prefix for the exact old point samplings."""
import argparse,sys,time,torch
from common61 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,source_capture,fingerprint
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--engineering',action='store_true');a=ap.parse_args();p=read(a.plan);pin(p);rows=read(ROOT/p['manifest']);rows=engineer(rows) if a.engineering else rows;out=ROOT/('research/round61/engineering_capture_attempt001' if a.engineering else p['capture_root']);assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();model=load_source(read(ROOT/p['source_plan']));model.eval().requires_grad_(False);torch.set_grad_enabled(False);sourcefp=fingerprint(model);start=time.perf_counter();calls=0
 np.savez_compressed(out/'DECODER_AND_MLP_PARAMETERS_PRIVATE.npz',**{n:v.detach().cpu().numpy() for n,v in model.state_dict().items() if n.startswith(('decoder.5.','contrastive_align_projection_image.'))})
 if not a.engineering:assert read(ROOT/'research/round61/engineering_replay_attempt001/RESULT.json')['state'].startswith('PASS_')
 try:
  for i,row in enumerate(rows):
   sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';AB=ROOT/p['AB_root'];C=ROOT/p['C_root'];assert sha(AB/rel)==read(AB/'SEALED.json')['files'][rel];assert sha(C/rel)==read(C/'SEALED.json')['files'][rel];ab=load(AB/rel);cc=load(C/rel);case=read(AB/'cases_PRIVATE'/(sid+'.json'));prefix={}
   # A native prefix already exists and matches the fixed source A slot set.
   root=ROOT/p['A_prefix_root'];r='cases_PRIVATE/'+sid+'.json';assert sha(root/r)==read(root/'SEALED.json')['files'][r];acase=read(root/r);file=acase['subject']['file'];assert sha(root/file)==read(root/'SEALED.json')['files'][file]==acase['subject']['sha256'];aa=load(root/file)
   for key in ['h','boxes','scores','query','tokens','token_mask']:assert np.array_equal(aa[key],ab['A_'+key]),('A',key)
   arrays={'A':aa}
   seedC=int(hashlib.sha256(('R57_student_C_v1|'+sid+'|'+str(row['seed'])).encode()).hexdigest()[:16],16)
   for v,seed,old,pre in [('B',case['B_seed'],ab,'B_'),('C',seedC,cc,'')]:
    points,ix,_=sample_points(dict(row,seed=seed));z,receipt=source_capture(model,points,ix,row['caption'],retain_prefix=True);calls+=1
    for key in ['boxes','scores','h','query','tokens','token_mask','point_indices']:assert np.array_equal(z[key],old[pre+key]),('SOURCE_BRIDGE',v,key)
    arrays[v]=z
   for v,z in arrays.items():
    for key in ['prefix_query','prefix_vis','prefix_lang','prefix_pos','prefix_text_padding','h','tokens','token_mask']:prefix[v+'_'+key]=z[key]
   prefix['source_boxes']=ab['A_boxes'];prefix['source_scores']=ab['A_scores'];np.savez_compressed(out/rel,**prefix)
   if (i+1)%32==0:print('R61_EXACT_PREFIX',i+1,len(rows),calls,flush=True)
  assert fingerprint(model)==sourcefp and calls==2*len(rows);cost={'n':len(rows),'new_full_source_forwards':calls,'source_unchanged':True,'source_fingerprint':sourcefp,'all_original_ABC_boxes_scores_h_tokens_point_indices_exact':True,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COSTS.json',cost);seal(out,'R61_ORIGINAL_ABC_PREFIX_BRIDGES_EXACT_NO_GT',**cost);print(cost,flush=True)
 except BaseException as error:dump(out/'FAILED.json',{'type':type(error).__name__,'error':str(error),'entered_full_source_forwards':calls,'automatic_retry':False});raise
if __name__=='__main__':main()
