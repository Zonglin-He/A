"""Current complete R95 on unchanged R97 source models/inputs/candidates."""
from core98 import *
import argparse
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['waymo','drone','quad'],required=True);ap.add_argument('--target',choices=['waymo','drone','quad'],required=True);ap.add_argument('--engineering',action='store_true');a=ap.parse_args();src,tgt=a.source,a.target;assert src!=tgt
 if not a.engineering:
  for f,s in jread(H98/'experiment/CODE_LOCK.json')['code_sha256'].items():assert sha(R98/f)==s,f
 setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);rows=rows97(tgt,'target');rows=[rows[0],rows[len(rows)//2]] if a.engineering else rows;O=B98/(('engineering_' if a.engineering else '')+src+'_to_'+tgt);O.mkdir(exist_ok=True)
 for p in ['samples_PRIVATE','records_PRIVATE','features_PRIVATE','states_PRIVATE','evidence_PRIVATE']:(O/p).mkdir(exist_ok=True)
 start=time.perf_counter();ssha=hashlib.sha256(b''.join(n.encode()+v.tobytes() for n,v in S.items())).hexdigest();complete=0
 for nr,r in enumerate(rows):
  out=O/'samples_PRIVATE'/(r['id']+'.npz')
  if out.exists():assert str(load(out)['source_state_sha256'])==ssha;continue
  oldroot=B97/(src+'_to_'+tgt);old=load(oldroot/'samples_PRIVATE'/(r['id']+'.npz'));assert str(old['source_state_sha256'])==ssha
  with torch.no_grad():z0,g,receipt=capture98(base,r,src)
  assert np.array_equal(old['source_boxes'],z0['boxes']) and np.array_equal(old['point_indices'],z0['point_indices']);rt,info=role97(base,m,r,z0,g,tensor_prefix(z0),tensor_data(g,r));scores,records,states,features,evidence,cost=complete_case98(base,m,S,r,z0,g,rt,old)
  for name,st in states.items():np.savez_compressed(O/'states_PRIVATE'/(r['id']+'_'+name+'.npz'),**st)
  np.savez_compressed(O/'features_PRIVATE'/(r['id']+'.npz'),**features);np.savez_compressed(O/'evidence_PRIVATE'/(r['id']+'.npz'),**evidence);jdump(O/'records_PRIVATE'/(r['id']+'.json'),{'source':src,'target':tgt,'readout_status':cost,'input_receipt':receipt,'route':info,'fits':records});np.savez_compressed(out,arm_names=np.array(ARMS98),scores=scores,queries=scores.argmax(1),source_boxes=z0['boxes'],source_state_sha256=np.array(ssha),anno_index=np.array(r['anno_index']));complete+=1
  if (nr+1)%32==0 or a.engineering:print('RUN98',src,tgt,nr+1,len(rows),'seconds',round(time.perf_counter()-start,1),'case_seconds',round(cost['seconds'],3),flush=True)
 assert fingerprint(base)==fp;final=state(m);assert all(np.array_equal(v,final[k]) for k,v in S.items());files=sorted(p for p in O.rglob('*') if p.is_file() and p.name!='SEALED.json');assert len(list((O/'samples_PRIVATE').glob('*.npz')))==len(rows);jdump(O/'SEALED.json',{'state':'R98_COMPLETE_R95_ENDPOINTS_SEALED_BEFORE_TARGET_SCORES','source':src,'target':tgt,'n':len(rows),'arms':ARMS98,'primary':'lm_FC','source_state_sha256':ssha,'source100_checkpoint_sha256':source97(src)['checkpoint_sha256'],'same_original_A256_R97':True,'no_source_training':True,'no_target_GT_for_update':True,'FINAL_access':False,'new_predictions':complete,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'files':{str(p.relative_to(O)):sha(p) for p in files}});print('RUN98_COMPLETE',src,tgt,len(rows),round(time.perf_counter()-start,1),flush=True)
if __name__=='__main__':main()
