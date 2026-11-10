"""Three matched policies, two fresh views each, one frozen grounder."""
import argparse,sys,time,torch
from common65 import *
import sampling65 as s
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,source_capture,fingerprint

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args();p=read(ROOT/'research/round65/experiment/EXPERIMENT_SPEC_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);rows=[min([r for r in rows if r['platform']==pl],key=lambda r:hashlib.sha256(('R65engineering|'+r['id']).encode()).hexdigest()) for pl in ['drone','quad']] if a.engineering else rows;out=ROOT/('research/round65/engineering_capture_attempt001' if a.engineering else p['new_capture_root']);assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir()
 if not a.engineering:assert read(ROOT/'research/round65/engineering_capture_attempt001/SEALED.json')['n']==2
 model=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(model);start=time.perf_counter();calls=0;status=[];prep=ROOT/p['prepared_root'];ps=read(prep/'SEALED.json')
 try:
  for i,r in enumerate(rows):
   sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(prep/rel)==ps['files'][rel];prepared=load(prep/rel);rec=read(prep/'cases_PRIVATE'/(sid+'.json'));assert sha(r['pointpath'])==rec['raw_sha256'];path=Path(r['pointpath']);raw=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,np.float32).reshape(-1,4);x=np.array(raw[:,:3],np.float32,copy=True)
   if r['platform']=='drone':x[:,2]+=1.8
   old=load(ROOT/p['capture_root']/rel,['A_boxes','A_scores','A_h','A_point_indices']);d={'targets':prepared['target_hypothesis_indices'],'refs':prepared['reference_hypothesis_indices'],'focus_union':prepared['focus_indices'],'context_union':prepared['context_indices']};payload={};stats={}
   if a.engineering:
    ix=old['A_point_indices'];bridge,_=source_capture(model,x[ix],ix,r['caption']);calls+=1
    for k in ['boxes','scores','h']:assert np.array_equal(bridge[k],old['A_'+k]),('engineering_A_bridge',k)
   for mode in p['sampling_modes']:
    for v in p['new_views']:
     key=mode+'_'+v;ix=prepared[key+'_point_indices'];extra={}
     def save(_m,_i,o):extra['seed_xyz']=o['fp2_xyz'][0].detach().cpu().numpy().copy()
     hook=model.register_forward_hook(save)
     try:z,_=source_capture(model,x[ix],ix,r['caption']);calls+=1
     finally:hook.remove()
     for k in ['boxes','scores','h','query','tokens','token_mask','point_indices']:payload[key+'_'+k]=z[k]
     payload[key+'_seed_xyz']=extra['seed_xyz'];stats[key]=s.counts(ix,old['A_point_indices'],d,len(x),extra['seed_xyz'],old['A_boxes'])
   np.savez_compressed(out/rel,**payload);dump(out/'cases_PRIVATE'/(sid+'.json'),stats);status.append(sid)
   if (i+1)%32==0:print('CAPTURE65',i+1,len(rows),calls,flush=True)
  assert fingerprint(model)==fp;cost={'n':len(rows),'new_full_source_forwards':calls,'source_unchanged':True,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COSTS.json',cost);seal(out,'R65_MATCHED_POLICY_SOURCE_CAPTURES_NO_GT',n=len(rows));print(cost,flush=True)
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed':len(status),'calls':calls});raise
if __name__=='__main__':main()
