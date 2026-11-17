"""Matched batch-context feature correction, no model parameter updates."""
from common139 import *
sys.path.insert(0,str(R/'research/round136'));from optim136 import *
from common139 import R,H,B,rd,sha,dump,check,utc
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 check();assert all(sha(R/p)==h for p,h in rd(H/'experiment/BATCH_DIRECT_CODE_LOCK.json')['files'].items());reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');parts=rd(H/'BATCH_PARTITIONS_PRIVATE.json');index={(e['source'],e['target'],e['row']['id']):e for e in reg};root=B/('engineering_batch_direct' if args.engineering else 'batch_direct');root.mkdir(parents=True,exist_ok=False);outall={};n=0
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);selected=[p for p in parts if p['source']==src]
  if args.engineering:selected=[p for p in selected if p['order']==0][:1]
  for p in selected:
   name=src+'_to_'+p['target'];D=root/'batches_PRIVATE'/p['batch_PRIVATE'];D.mkdir(parents=True);diff=[]
   for sid in p['members_PRIVATE']:
    a=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));f=load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));valid=f['valid_B'];assert valid.any();diff.append((f['B_vis'][valid]-a['vis0'][0][valid]).mean(0))
   common=np.asarray(np.mean(diff,axis=0)*.5,np.float32);np.savez_compressed(D/'batch_shift_PRIVATE.npz',shift=common)
   for sid in p['members_PRIVATE']:
    e=index[(src,p['target'],sid)];z0,z,d,route,ev,groups,old=inputs136(e);used=np.ascontiguousarray(z0['vis0'][0]+common[None],np.float32)
    with torch.no_grad():lp,a,_=live135(m,base,z,d,torch.as_tensor(used,device='cuda'),route)
    prob=lp.exp().cpu().numpy();feat={k:a[k].cpu().numpy() for k in FEATURES};np.savez_compressed(D/(sid+'.npz'),probability=prob,**feat);outall[(src,p['target'],sid,p['order'])]=prob
   dump(D/'record.json',{'source':src,'target':p['target'],'order':p['order'],'members_PRIVATE':p['members_PRIVATE'],'batch_size':p['n'],'shift_L2':float(np.linalg.norm(common)),'parameters_changed':0});n+=1
   if args.engineering or n%32==0:print('BATCH_DIRECT139',n,src,flush=True)
  assert fingerprint(base)==fp and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if not args.engineering:
  for e in reg:
   src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;old=np.load(R/'crossplatform/round136/method'/name/'samples_PRIVATE'/(sid+'.npz'));p0=dict(zip(old['arm_names'],old['scores']))['Frozen_A'];scores={f'batch_direct_order{k}':outall.get((src,tgt,sid,k),p0).copy() for k in [0,1,2]};D=root/'samples_PRIVATE'/name;D.mkdir(parents=True,exist_ok=True);a=np.stack(list(scores.values()));np.savez_compressed(D/(sid+'.npz'),scores=a,arm_names=np.array(list(scores)),queries=a.argmax(1),source_boxes=old['source_boxes'])
 dump(root/'SEALED.json',{'batch_count':n,'n':None if args.engineering else 1798,'parameters_changed':0,'GT_used':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('BATCH_DIRECT139_COMPLETE',n,flush=True)
if __name__=='__main__':main()
