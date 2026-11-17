"""Independent source-channel batch shift and native reader verification."""
from common139 import *
import argparse,numpy as np,torch
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round135'));from audit135 import read135,tensors;sys.argv=argv
def main():
 check();assert all(sha(R/p)==h for p,h in rd(H/'experiment/BATCH_DIRECT_CODE_LOCK.json')['files'].items());torch.set_num_threads(4);root=B/('engineering_batch_direct' if args.engineering else 'batch_direct');seal=rd(root/'SEALED.json');W={s:tensors(dict(np.load(R/'crossplatform/round135'/(s+'_SOURCE_MODEL_PRIVATE.npz')))) for s in ['waymo','drone','quad']};n=0;casefits=0;mx=0.;saved={}
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for D in sorted((root/'batches_PRIVATE').iterdir()):
  rec=rd(D/'record.json');src,tgt=rec['source'],rec['target'];name=src+'_to_'+tgt;diff=[]
  for sid in rec['members_PRIVATE']:
   z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));f=np.load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));diff.append((f['B_vis'][f['valid_B']]-z['vis0'][0][f['valid_B']]).mean(0))
  expected=np.asarray(np.mean(diff,axis=0)*.5,np.float32);actual=np.load(D/'batch_shift_PRIVATE.npz')['shift'];assert np.array_equal(expected,actual) and rec['parameters_changed']==0
  for sid in rec['members_PRIVATE']:
   a=np.load(D/(sid+'.npz'));feat=tensors({k:a[k] for k in ['h','u','late','local','own','text','relation_text','mask']});rt=tensors(dict(np.load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'))));p=read135(feat,rt,W[src]);err=float(abs(p-a['probability']).max());mx=max(mx,err);assert err<5e-6;saved[(src,tgt,sid,rec['order'])]=a['probability'];casefits+=1
  n+=1
  if n%64==0:print('AUDIT_BATCH_DIRECT139',n,casefits,flush=True)
 if not args.engineering:
  count=0
  for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
   src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(root/'samples_PRIVATE'/name/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));old=np.load(R/'crossplatform/round136/method'/name/'samples_PRIVATE'/(sid+'.npz'));p0=dict(zip(old['arm_names'],old['scores']))['Frozen_A'];assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1))
   for order in [0,1,2]:assert np.array_equal(ps['batch_direct_order'+str(order)],saved.get((src,tgt,sid,order),p0))
   count+=1
  assert count==1798 and casefits==764*3
 out={'status':'PASS','n_batch':n,'case_fits':casefits,'source_shift_exact':True,'probability_max_CPU_error':mx,'no_parameters_updated':True,'FULL_encoder_not_CPU_regenerated':True,'FINAL_access':False};dump(H/('ENGINEERING_BATCH_DIRECT_AUDIT.json' if args.engineering else 'BATCH_DIRECT_AUDIT.json'),out);print(out,flush=True)
if __name__=='__main__':main()
