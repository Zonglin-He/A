"""A/B source prefix/SA2 captures only; all AB fits precede C access."""
import sys,time,torch,argparse
from common69 import *
from method69 import F,G
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args();H=ROOT/'research/round69';p=read(H/'experiment/DESIGN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);rows=[min([r for r in rows if r['platform']==pl and read(ROOT/p['cohort_map'])[r['id']]==cohort],key=lambda r:hashlib.sha256(('R69eng|'+r['id']).encode()).hexdigest()) for cohort in ['VAL1280','TRAIN1024'] for pl in ['drone','quad']] if a.engineering else rows
out=H/'engineering_capture_attempt001' if a.engineering else ROOT/p['AB_capture_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();m=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(m);start=time.perf_counter();calls=0
for i,r in enumerate(rows):
 cl=G.category(r['caption']);payload={'category_available':np.array(cl is not None)}
 if cl is not None:lang,_=F.class_language(m,cl)
 for v,seed in [('A',r['seed']),('B',bseed(r))]:
  xyz,ix,_=sample_points(dict(r,seed=seed));got={}
  def pre(_m,args):got['SA2_xyz']=args[0].detach().clone().cpu().numpy();got['SA2_feat']=args[1].detach().clone().cpu().numpy()
  hk=m.backbone_net.SA_modules[2].register_forward_pre_hook(pre)
  try:z,_=F.capture(m,xyz,ix,r['caption']);calls+=1
  finally:hk.remove()
  z.update(got);z.update(lang if cl is not None else {'class_lang0':z['lang0'],'class_textpad':z['textpad'],'class_token_mask':z['token_mask']});assert np.array_equal(z['SA2_xyz'],z['xyz'])
  if a.engineering:
   y=F.replay(m,z)
   for k in ['h','query','tokens']:assert np.array_equal(z[k],y[k])
  for k,value in z.items():payload[v+'_'+k]=value
 np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),**payload)
 if(i+1)%64==0:print('CAPTURE_AB69',i+1,len(rows),calls,flush=True)
assert fingerprint(m)==fp;dump(out/'COSTS.json',{'n':len(rows),'full_source_forwards':calls,'C_read_or_used':False,'source_unchanged':True,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())});seal(out,'R69_A_B_SOURCE_PREFIXES_NO_C_NO_LABELS',n=len(rows));print({'n':len(rows),'source_calls':calls})
