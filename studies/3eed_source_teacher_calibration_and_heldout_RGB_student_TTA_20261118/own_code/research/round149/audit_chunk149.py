"""Independent native CPU verification of a sealed temporary chunk, before task GT.

Native final features are kept until PASS; only new chunk-owned temporary
feature files may then be removed. States/predictions/traces and hash receipts stay.
"""
from common149 import *
import numpy as np,torch,argparse
from torch.nn import functional as F
ap=argparse.ArgumentParser();ap.add_argument('--chunk',required=True);args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round135'));from audit135 import read135,tensors;sys.argv=argv
from common149 import R,H,B,rd,sha,dump,check,utc

def main():
 check();torch.set_num_threads(4);p=Path(args.chunk).resolve();assert p.is_relative_to(B) and p.name.endswith('.json');ch=rd(p);root=R/ch['root'];assert root.is_relative_to(B);n=0;nf=0;mxp=0.;mxs=0.;mxf=0.;mxo=0.;mxb=0.;process=[];native_hashes={}
 for rel,h in ch['files'].items():assert sha(root/rel)==h,(rel,'chunkhash')
 W=None;last=None
 for row in ch['rows']:
  src,tgt,sid=row['source'],row['target'],row['id_PRIVATE'];name=src+'_to_'+tgt;D=root/name;fpath=D/'features_TEMP_PRIVATE'/(sid+'.npz');a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));oparr=np.load(R/'crossplatform/round148/method'/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(oparr['arm_names'],oparr['scores']));assert np.array_equal(a['source_boxes'],oparr['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1));assert np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9)
  for k,v in op.items():assert np.array_equal(v,ps[k])
  ev=np.load(B/'evidence_PRIVATE'/name/(sid+'.npz'));assert bool(ev['same_teacher']);assert np.array_equal(ps['cal_A_TTA'],ps['plain_A_TTA']) and np.array_equal(ps['cal_A_wrong_TTA'],ps['plain_A_wrong_TTA']) and np.array_equal(ps['cal_AB_TTA'],ps['norm_TTA']) and np.array_equal(ps['cal_AB_wrong_TTA'],ps['norm_wrong_TTA']);assert np.array_equal(ps['cal_A_direct'],ev['direct_A']) and np.array_equal(ps['cal_AB_direct'],ev['direct_AB']) and np.array_equal(ps['cal_B'],ev['cal_B']);assert np.array_equal(ps['cal_FA_FB_mean'],(ps['norm_Aimage']+ps['cal_B'])/2) and np.array_equal(ps['cal_FAB_FB_mean'],(ps['norm_joint_zero']+ps['cal_B'])/2)
  fitrec=rd(D/'records_PRIVATE'/(sid+'.json'));ff=np.load(fpath);ss=np.load(D/'states_PRIVATE'/(sid+'.npz'));ass=ev['assignment'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];scope,C=ev['S'],ev['C']
  if last!=src:W=tensors(dict(np.load(R/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz'))));last=src
  z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));b=np.load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));P=np.where(b['valid_B'][:,None],(z['vis0'][0]+b['B_vis'])/2,z['vis0'][0]);X=torch.from_numpy(P.copy());radius=.01*float(torch.sqrt(sum(v.double().square().sum() for k,v in W.items() if k.startswith('point_adapter.'))));rt=tensors(dict(np.load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'))))
  for arm in ['plain_A_TTA','plain_A_wrong_TTA']:
   fit=fitrec['fits'][arm]
   if fit.get('reused'):
    key=fit.get('parent_arm', 'plain_A_TTA' if 'wrong' in arm and bool(ev['eligible']) else 'Frozen_A');assert np.array_equal(ps[arm],ps[key]);continue
   nf+=1;delta=np.zeros(4896,np.float32);assert abs(fit['radius']-radius)<1e-10 and len(fit['trace'])==3
   for j,t in enumerate(fit['trace'],1):
    g=ss[arm+'_gradient'+str(j)]
    if t['accepted']:delta=(delta-np.float32(radius)*g/np.float32(t['gradient_L2'])*np.float32(t['fraction'])).astype(np.float32)
    assert t['after']<=t['before']+1e-8
    for tr in t['trials']:assert tr['accepted']==(tr['objective']<=tr['bound'])
   err=float(abs(delta-ss[arm+'_delta']).max());mxs=max(mxs,err);assert err<1e-7;feat=tensors({k:ff[arm+'_final_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']});prob=read135(feat,rt,W);err=float(abs(prob-ps[arm]).max());mxp=max(mxp,err);assert err<5e-6;hidden=F.gelu(F.linear(X,W['point_adapter.down.weight'],W['point_adapter.down.bias']));base=X+F.linear(hidden,W['point_adapter.up.weight'],W['point_adapter.up.bias']);adapt=X+F.linear(hidden,W['point_adapter.up.weight']+torch.from_numpy(delta[:4608].reshape(288,16)),W['point_adapter.up.bias']+torch.from_numpy(delta[4608:]));idx=ff[arm+'_probe_indices'];assert np.array_equal(idx,np.linspace(0,1023,32).astype(np.int64));err=float(abs(adapt.numpy()[idx]-ff[arm+'_adapted_A_probe']).max());mxf=max(mxf,err);assert err<2e-5
   if ch['engineering']:
    err=float(abs(adapt.numpy()-ff[arm+'_adapted_A']).max());mxf=max(mxf,err);assert err<2e-5
   else:assert arm+'_adapted_A' not in ff.files
   assert fit['actual_F_input_calls']>0 and fit['actual_F_input_sha256']==hashlib.sha256(P.tobytes()).hexdigest() and fit['student_RGB_banks']==1
   keep=float((F.normalize(adapt,dim=-1)-F.normalize(base,dim=-1)).square().sum(-1).mean()/2);lp=torch.from_numpy(np.log(np.maximum(ps[arm],1e-300)));lm=torch.stack([lp[g].logsumexp(0)-np.log(len(g)) for g in gs]);partial=float(lm[scope].logsumexp(0)-lm[C].logsumexp(0)) if (scope&~C).any() else 0.;gm=torch.stack([lp[g].max() for g in gs]);tar=torch.from_numpy(ev['plain_wrong_target'] if 'wrong' in arm else ev['plain_target']);rank=float((tar*(tar.clamp_min(1e-300).log()-gm[C].log_softmax(0))).sum());task=partial+rank;t=fit['trace'][-1];err=max(abs(task-t['task_after']),abs(keep-t['keep_after']),abs(task+keep-t['after']));mxo=max(mxo,err);assert err<2e-6
   process.append(dict(source=src,target=tgt,id_PRIVATE=sid,arm=arm,parameter_L2=float(np.linalg.norm(delta.astype(float))),accepted_steps=sum(t['accepted'] for t in fit['trace']),trials=sum(len(t['trials']) for t in fit['trace']),task_after=task,keep_after=keep,finite_difference=fit['finite_difference'],student_RGB_banks=1))
  if not row['has_B']:
   for k in ['plain_A_TTA','plain_A_wrong_TTA','cal_A_TTA','cal_A_wrong_TTA','cal_AB_TTA','cal_AB_wrong_TTA']:assert np.array_equal(ps[k],ps['Frozen_A'])
  assert fitrec['zero_bridge_error']<1e-7;native_hashes[str(fpath.relative_to(root))]=sha(fpath);n+=1
 out=dict(status='PASS',n=n,newfits=nf,max_CPU_probability_error=mxp,max_state_error=mxs,max_pointfeature_error=mxf,max_objective_error=mxo,full1024point_keep_independently_reconstructed=True,engineering_full_feature_compared=ch['engineering'],fullTransformer_gradients_not_CPU_regenerated=True,teacher_and_evidence_fixed_before_updates=True,GT_read=False,chunk_manifest_sha256=sha(p),files=ch['files'],native_features_sha256=native_hashes,rows=ch['rows'],process=process,utc=utc());dump(p.with_name(p.stem+'_AUDIT.json'),out);print('CHUNK149_AUDIT',n,nf,mxp,mxs,mxf,mxo,flush=True)
if __name__=='__main__':main()
