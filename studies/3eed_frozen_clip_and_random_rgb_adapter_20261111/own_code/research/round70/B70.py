"""No CLIP: random RGB predictor fitted only to disjoint source point features."""
from common70 import *
from models70 import *
import sys,time,argparse
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
sys.path.insert(0,str(ROOT/'research/round66'));import interface66 as F

def train_model(patches,target,fit,held,wrong=False):
 torch.manual_seed(0);m=RGBAdapter().cuda();init=snapshot(m);opt=torch.optim.Adam(m.parameters(),lr=.001);x=torch.tensor(patches,device='cuda',dtype=torch.float32)/127.5-1;t=torch.tensor(target,device='cuda');f=torch.tensor(fit,device='cuda');held_t=torch.tensor(held,device='cuda');order=np.roll(np.arange(len(patches)),1) if wrong else np.arange(len(patches));trainx=x[torch.tensor(order,device='cuda')];records=[];preds={}
 with torch.no_grad():initial_fit=float(((m(trainx[f])-t[f])**2).sum(1).mean());initial_held=float(((m(x[held_t])-t[held_t])**2).sum(1).mean());random=m(x[held_t]).cpu().numpy()
 for step in range(1,33):
  batch=f if len(f)<=128 else f[((torch.arange(128,device='cuda')+(step-1)*128)%len(f))];opt.zero_grad(set_to_none=True);loss=((m(trainx[batch])-t[batch])**2).sum(1).mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if step in [3,32]:
   with torch.no_grad():preds[step]=m(x[held_t]).cpu().numpy();records.append({'step':step,'fit_loss':float(((m(trainx[f])-t[f])**2).sum(1).mean()),'held_loss':float(((m(x[held_t])-t[held_t])**2).sum(1).mean()),'parameter_L2':float(sum(np.sum((v-init[k])**2) for k,v in snapshot(m).items())**.5)})
 with torch.no_grad():wrongtest=m(x[torch.tensor(np.roll(np.arange(len(patches)),1),device='cuda')][held_t]).cpu().numpy()
 return preds,random,wrongtest,snapshot(m),{'initial_fit_loss':initial_fit,'initial_held_loss':initial_held,'updates':records,'parameters':sum(v.numel() for v in m.parameters())}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;p=read(H/'experiment/B_PLAN_LOCK.json');pin(p);rr=rows();rr=[next(r for r in rr if r['platform']==pl) for pl in ['drone','quad']] if args.engineering else rr;out=H/'engineering_B2_attempt001' if args.engineering else ROOT/p['B_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();m=load_source(read(ROOT/p['source_plan']));m.eval().requires_grad_(False);fp=fingerprint(m);start=time.perf_counter();calls=eligible=steps=0
 for i,r in enumerate(rr):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';z=prefix(r);a=load(ROOT/p['prepared_root']/rel);scores={k:z['scores'].copy() for k in ['B_random0','B_step3','B_step32','B_wrongtest32','B_wrongtrain32','B_NW','B_NW_wrong','B_mean']};zero=F.replay(m,z);calls+=1
  for k in ['h','query','tokens']:assert np.array_equal(z[k],zero[k]),('zero_replay',i,k)
  rec={'eligible':bool(a['B_eligible']),'zero_source_replay_exact':True,'CLIP_inputs_or_model_used':False}
  if a['B_eligible']:
   eligible+=1;ii=a['seed_indices'];features=z['vis0'][0,ii];target=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-6);fit=a['fit_patch_indices'];held=a['held_patch_indices'];assert not set(fit)&set(held);pred,random,wrongtest,state,realrec=train_model(a['patches'],target,fit,held);wrong,_,_,_,wrongrec=train_model(a['patches'],target,fit,held,True);steps+=64;mean=np.broadcast_to(target[fit].mean(0),(len(held),288)).copy();mean/=np.maximum(np.linalg.norm(mean,axis=1,keepdims=True),1e-6);pp={'B_random0':random,'B_step3':pred[3],'B_step32':pred[32],'B_wrongtest32':wrongtest,'B_wrongtrain32':wrong[32],'B_NW':NW(a['patches'],fit,held,target),'B_NW_wrong':NW(a['patches'],fit,held,target,True),'B_mean':mean};norms={}
   for arm,value in pp.items():
    vis=edit_vis(z,a,value);norms[arm]=float(np.linalg.norm(vis-z['vis0'])/max(np.linalg.norm(z['vis0']),1e-6))
    with torch.no_grad():y=F.replay(m,z,alternate_vis0=vis)
    scores[arm]=y['p'];calls+=1
   rec.update(real=realrec,wrong_training=wrongrec,feature_relative_change=norms,fit_seed_count=len(fit),held_seed_count=len(held),modified_query_anchors=int(np.isin(z['query_points_sample_inds'],ii[held]).sum()));np.savez_compressed(out/'samples_PRIVATE'/(sid+'_adapter.npz'),**state);np.savez_compressed(out/'samples_PRIVATE'/(sid+'_held_predictions.npz'),**pp)
  np.savez_compressed(out/rel,arm_names=np.array(list(scores)),scores=np.stack(list(scores.values())));dump(out/'cases_PRIVATE'/(sid+'.json'),rec)
  if(i+1)%16==0:print('B70',i+1,336,eligible,calls,flush=True)
 assert fingerprint(m)==fp and all(x.grad is None for x in m.parameters());dump(out/'COUNTS.json',{'n':len(rr),'eligible':eligible,'CNN_gradient_steps':steps,'encoder_decoder_replays':calls,'source_unchanged':True,'CLIP_or_external_vision_model_loaded':False,'source_GT_or_target_GT_used_for_fitting':False,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())});seal(out,'R70_B_RANDOM_RGB_EPISODIC_POINT_DISTILL_HELD_SEEDS_ALL_NEGATIVES',n=len(rr));print(read(out/'COUNTS.json'))
