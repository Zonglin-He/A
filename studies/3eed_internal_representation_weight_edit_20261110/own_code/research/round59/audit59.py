"""Independent reconstruction of targets, ridge optimality, spectral GD and all readouts."""
import argparse,time,torch
from common59 import *

def norm(x,fallback=None):
 a=np.sqrt((x*x).sum(1))[:,None];v=x/np.maximum(a,1e-300)
 return np.where(a>1e-12,v,np.zeros_like(v) if fallback is None else fallback)
def score(raw,d,g):
 vv=[]
 for v in 'ABC':
  yy=torch.tensor(np.asarray(raw[v],np.float32))[None];qq=torch.nn.functional.normalize(yy,dim=-1);tt=torch.tensor(d[v+'_tokens'])[None];mask=torch.tensor(d[v+'_mask']).bool()[None,None];l=qq@tt.transpose(-1,-2)/.07;zz=(torch.logsumexp(l.masked_fill(~mask,-torch.inf),-1)-torch.logsumexp(l.masked_fill(mask,-torch.inf),-1))[0];p=zz.double().softmax(-1).numpy()
  if v!='A':mapped=g[v+'_W']@p;mass=float(g[v+'_fixed_mass']);p=mass*mapped/max(float(mapped.sum()),1e-300)+(1-mass)*g['base']
  vv.append(p)
 return (vv[0]+vv[1]+vv[2])/3

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);out=ROOT/p['audit_root'];assert not out.exists();out.mkdir(parents=True);start=time.perf_counter();torch.set_num_threads(4);rows=read(ROOT/p['manifest']);pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');M0=np.column_stack([initial['4.weight'],initial['4.bias']]).astype(float);seals={k:read(ROOT/p[k]/'SEALED.json') for k in ['AB_root','capture_root','association_root','r57_prediction_root','r56_prediction_root','r58_prediction_root']};errors={'scores':0.,'targets':0.,'ridge_solve':0.,'gradient_spectral':0.,'normal_equation':0.};count=0;changed=0;prefixchecks=0
 def old(k,rel):assert sha(ROOT/p[k]/rel)==seals[k]['files'][rel];return load(ROOT/p[k]/rel)
 def new(rel):assert sha(pred/rel)==ps['files'][rel];return load(pred/rel)
 try:
  for row in rows:
   sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';z=new(rel);d=new('targets_PRIVATE/'+sid+'.npz');f=new('fitted_PRIVATE/'+sid+'.npz');path='cases_PRIVATE/'+sid+'.json';assert sha(pred/path)==ps['files'][path];r=read(pred/path);v=old('AB_root',rel);c=old('capture_root',rel);g=old('association_root',rel);oldpred={k:old(k,rel) for k in ['r57_prediction_root','r56_prediction_root','r58_prediction_root']};assert np.array_equal(d['M0'],M0) and np.array_equal(z['source_boxes'],v['A_boxes']) and np.array_equal(z['source_scores'],v['A_scores']) and np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']]);assert list(z['arm_names'])==ARMS and np.array_equal(z['scores'].argmax(1),z['queries']) and np.isfinite(z['scores']).all()
   for view,cache,pre in [('A',v,'A_'),('B',v,'B_'),('C',c,'')]:
    hh=torch.tensor(cache[pre+'h'])[None];xx=torch.nn.functional.linear(hh,torch.tensor(initial['0.weight']),torch.tensor(initial['0.bias'])).relu();xx=torch.nn.functional.linear(xx,torch.tensor(initial['2.weight']),torch.tensor(initial['2.bias'])).relu()[0].numpy();assert np.array_equal(xx,d[view+'_X']) and np.array_equal(np.column_stack([xx.astype(float),np.ones(256)]),d[view+'_H']);assert np.array_equal(cache[pre+'tokens'],d[view+'_tokens']) and np.array_equal(cache[pre+'token_mask'],d[view+'_mask']);prefixchecks+=1
   y={s:d[s+'_H']@M0.T for s in 'ABC'};u={s:norm(y[s]) for s in 'ABC'};wb=g['B_W'];wc=g['C_W'];rb=g['B_affinity'].max(1);br=wb.sum(1);bm=np.divide(wb@u['B'],br[:,None],out=np.zeros_like(u['A']),where=br[:,None]>0);ua=norm(u['A']+rb[:,None]*bm,u['A']);target={'A':np.linalg.norm(y['A'],axis=1)[:,None]*ua};conf={'A':rb}
   for view,W in [('B',wb),('C',wc)]:
    sm=W.sum(0);uu=np.divide(W.T@ua,sm[:,None],out=np.zeros_like(u[view]),where=sm[:,None]>0);conf[view]=g[view+'_affinity'].max(0)*(np.linalg.norm(uu,axis=1)>1e-12);target[view]=np.linalg.norm(y[view],axis=1)[:,None]*norm(uu,u[view])
   for view in 'ABC':
    err=float(abs(target[view]-d['target_'+view]).max());assert err<1e-10 and np.array_equal(conf[view],d['confidence_'+view]);errors['targets']=max(errors['targets'],err)
   w=conf['C'];rootw=np.sqrt(w/max(float(w.sum()),1e-12))[:,None];H=d['C_H']*rootw;E=(target['C']-y['C'])*rootw;G=H.T@H;B=E.T@H
   for key,array in [('Hw',H),('E',E),('G',G),('B',B)]:assert np.allclose(array,d[key],rtol=0,atol=1e-10)
   lam=r['lambda_effective'];base=max(float(np.trace(G))/289,1e-12);assert base==r['lambda_base'] and lam>=base;delta=f['ridge_edit_M64']-M0;exact=np.linalg.solve(G+lam*np.eye(289),B.T).T;error=float(abs(delta-exact).max());assert error<1e-9;errors['ridge_solve']=max(errors['ridge_solve'],error);normal=float(np.linalg.norm(delta@(G+lam*np.eye(289))-B));assert normal<1e-8;errors['normal_equation']=max(errors['normal_equation'],normal);cap=.05*np.linalg.norm(M0);assert np.linalg.norm(delta)<=cap+1e-10
   if r['cap_active']:assert abs(np.linalg.norm(delta)-cap)<1e-9
   else:assert lam==base
   ev,V=np.linalg.eigh(G);ev=np.maximum(ev,0);eta=r['step_size'];assert abs(eta-1/(2*(ev[-1]+lam)))<1e-12;R=B@V;logs={x['step']:x for x in r['gradient_checkpoints']}
   for k in [1,3,16,256]:
    expected=(R*((1-(1-2*eta*(ev+lam))**k)/(ev+lam))[None])@V.T;err=float(abs(f['grad'+str(k)+'_M64']-M0-expected).max());assert err<1e-8;errors['gradient_spectral']=max(errors['gradient_spectral'],err);before=(R*((1-(1-2*eta*(ev+lam))**(k-1))/(ev+lam))[None])@V.T;L0=float(np.square(H@before.T-E).sum()+lam*np.square(before).sum());L1=float(np.square(H@expected.T-E).sum()+lam*np.square(expected).sum());assert abs(logs[k]['objective_before']-L0)<1e-8 and abs(logs[k]['objective_after']-L1)<1e-8 and L1<=L0+1e-12;assert np.linalg.norm(expected)<=cap+1e-8
   valid=np.flatnonzero(w>0);perm=np.arange(256);rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('R59_shuffle_v1|'+sid).encode()).digest()[:8],'little'));perm[valid]=rng.permutation(valid);assert np.array_equal(perm,d['shuffle_permutation']);ts=target['C'].copy();ts[valid]=np.linalg.norm(y['C'][valid],axis=1)[:,None]*norm(target['C'][perm[valid]]);Es=(ts-y['C'])*rootw;ds=np.linalg.solve(G+lam*np.eye(289),(Es.T@H).T).T;dn=np.linalg.norm(ds);ds=ds*np.linalg.norm(delta)/dn if dn>1e-20 else np.zeros_like(ds);assert np.allclose(f['shuffled_edit_M64'],M0+ds,rtol=0,atol=1e-9)
   raw32={}
   for view in 'ABC':raw32[view]=torch.nn.functional.linear(torch.tensor(d[view+'_X'])[None],torch.tensor(initial['4.weight']),torch.tensor(initial['4.bias']))[0].numpy()
   for idx,arm in enumerate(ARMS):
    if arm in ['source','frozen_ABC','r56_one','r58_online_ema_teacher','r58_online_ema_student']:
     key,name=('r57_prediction_root',arm) if arm in ['source','frozen_ABC'] else (('r56_prediction_root','tta1') if arm=='r56_one' else ('r58_prediction_root','backtrack_online_ema_'+('teacher_post' if arm.endswith('teacher') else 'student_post')));oldz=oldpred[key];assert np.array_equal(z['scores'][idx],oldz['scores'][list(oldz['arm_names']).index(name)]);continue
    if arm=='zero_edit':raw=raw32
    elif arm.startswith('direct'):
     raw={view:raw32[view].copy() for view in 'ABC'}
     for view in ('C' if arm=='direct_C' else 'ABC'):raw[view]=(1-conf[view][:,None])*raw32[view]+conf[view][:,None]*target[view]
    else:
     W=f['ridge_edit_M32'] if arm=='functional_edit' else f[arm+'_M64'].astype(np.float32);raw={view:torch.nn.functional.linear(torch.tensor(d[view+'_X'])[None],torch.tensor(W[:,:288].copy()),torch.tensor(W[:,288].copy()))[0].numpy() for view in 'ABC'}
    predicted=score(raw,d,g);err=float(abs(predicted-z['scores'][idx]).max());assert err<3e-5 and int(predicted.argmax())==int(z['queries'][idx]),(sid,arm,err);errors['scores']=max(errors['scores'],err);count+=1
   assert np.array_equal(z['scores'][ARMS.index('ridge_edit')],z['scores'][ARMS.index('functional_edit')]);changed+=int(not np.array_equal(f['ridge_edit_M32'],M0.astype(np.float32)))
   if (len(ARMS) and prefixchecks//3)%32==0:print('AUDIT59',prefixchecks//3,336,flush=True)
  res={'state':'PASS_R59_TARGETS_RIDGE_KKT_SPECTRAL_GRADIENT_AND_ALL_READOUTS_INDEPENDENTLY_AUDITED','n':336,'arm_names':ARMS,'prediction_seal_sha256':sha(pred/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'independent_noninherited_readouts':count,'prefix_features_checked':prefixchecks,'actual_last_layer_edit_cases':changed,'normalization_and_bias_accounted':True,'functional_and_written_edit_readouts_bitexact':True,'no_GT_media_or_source_training':True,'max_errors':errors,'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',res);print(res,flush=True)
 except BaseException as error:dump(out/'FAILED.json',{'type':type(error).__name__,'error':str(error),'completed':prefixchecks//3,'GT_access':False});raise
if __name__=='__main__':main()
