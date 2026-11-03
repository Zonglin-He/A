"""Independent CPU replay, fixed-box bridge, EMA recurrence and stream-state audit."""
import argparse,time,torch
from scipy.special import softmax,logsumexp
from common57 import *

def head(arr):
 m=torch.nn.Sequential(torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,64));m.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in arr.items()});return m.eval().requires_grad_(False)
def dist(m,v,view,g):
 pre='' if view=='C' else view+'_';h=torch.from_numpy(v[pre+'h'])[None];tokens=torch.from_numpy(v[pre+'tokens'])[None];mask=v[pre+'token_mask']
 with torch.no_grad():q=torch.nn.functional.normalize(m(h),dim=-1);l=(q@tokens.transpose(-1,-2)/.07)[0].numpy().astype(float)
 z=logsumexp(l[:,mask],axis=-1)-logsumexp(l[:,~mask],axis=-1);p=softmax(z)
 if view=='A':return p
 pp=g[view+'_W']@p;mass=float(g[view+'_fixed_mass']);return mass*pp/max(pp.sum(),1e-300)+(1-mass)*g['base']
def ensemble(m,v,c,g,views):return np.mean([dist(m,c if x=='C' else v,x,g) for x in views],axis=0)
def independently_raw_affinity(a,b,xyz):
 aa=np.stack([((abs(xyz-x[:3])<=x[3:]/2).all(1)&bool((x[3:]>0).all())) for x in a]);bb=np.stack([((abs(xyz-x[:3])<=x[3:]/2).all(1)&bool((x[3:]>0).all())) for x in b]);shared=aa.astype(np.float32)@bb.astype(np.float32).T;u=aa.sum(1)[:,None]+bb.sum(1)[None]-shared;j=np.divide(shared,u,out=np.zeros_like(shared),where=u>0);l=np.maximum(a[:,None,:3]-a[:,None,3:]/2,b[None,:,:3]-b[None,:,3:]/2);h=np.minimum(a[:,None,:3]+a[:,None,3:]/2,b[None,:,:3]+b[None,:,3:]/2);inter=np.maximum(h-l,0).prod(-1);vol=np.maximum(a[:,3:],0).prod(1)[:,None]+np.maximum(b[:,3:],0).prod(1)[None]-inter;i=np.divide(inter,vol,out=np.zeros_like(inter),where=vol>0);A=np.sqrt(np.maximum(j*i,0)).astype(np.float32);A[shared<4]=0;return A

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);out=ROOT/p['audit_root'];assert not out.exists();out.mkdir(parents=True);start=time.perf_counter();torch.set_num_threads(4);rows=read(ROOT/p['manifest']);pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');assert ps['arm_names']==ARMS and ps['optimizer_steps_total_completed']==12096;rootAB=ROOT/p['AB_root'];rootC=ROOT/p['capture_root'];rootG=ROOT/p['association_root'];seals={str(r):read(r/'SEALED.json') for r in [rootAB,rootC,rootG,pred]};initial=load(rootAB/'SOURCE_MATCHING_MLP_PRIVATE.npz');source=head(initial);names=list(initial);maxerr=0.;replayed=0;rawchecks=0;emachecks=0;stats=[];rec_byid={};prepost={};probe={min([r for r in rows if r['platform']==pl],key=lambda r:hashlib.sha256((salt+r['id']).encode()).hexdigest())['id'] for pl in ['drone','quad'] for salt in ['audit57a','audit57b']}
 def member(root,rel):assert sha(root/rel)==seals[str(root)]['files'][rel];return load(root/rel)
 try:
  for row in rows:
   sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';v=member(rootAB,rel);c=member(rootC,rel);g=member(rootG,rel);s=member(pred,rel);f=member(pred,'fitted_PRIVATE/'+sid+'.npz');recpath='cases_PRIVATE/'+sid+'.json';assert sha(pred/recpath)==ps['files'][recpath];rec=read(pred/recpath);rec_byid[sid]=rec
   assert np.array_equal(s['source_boxes'],v['A_boxes']) and np.array_equal(s['source_scores'],v['A_scores']) and np.array_equal(s['selected_original_boxes'],v['A_boxes'][s['queries']]);assert list(s['arm_names'])==ARMS and np.isfinite(s['scores']).all() and np.array_equal(s['scores'].argmax(1),s['queries'])
   for view in ['B','C']:
    A=g[view+'_affinity'];W=A.astype(float)/(1+A.astype(float).sum(0,keepdims=True));assert np.array_equal(W,g[view+'_W']) and (W.sum(0)<1).all();rawdist=dist(source,c if view=='C' else v,view,dict(g,**{view+'_W':np.eye(256),view+'_fixed_mass':np.array(1.)}));assert abs(float((W@rawdist).sum())-float(g[view+'_fixed_mass']))<3e-6
   if sid in probe:
    path=Path(row['pointpath']);raw=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,dtype=np.float32).reshape(-1,4);ids=np.unique(np.concatenate([v['A_point_indices'],v['B_point_indices'],c['point_indices']]));assert np.array_equal(ids,g['union_raw_indices']);expected_C=np.random.default_rng(seed(row)).choice(len(raw),16384,replace=len(raw)<16384);assert np.array_equal(expected_C,c['point_indices']);xyz=raw[ids,:3].astype(np.float32).copy()
    if row['platform']=='drone':xyz[:,2]+=1.8
    for view,b in [('B',v['B_boxes']),('C',c['boxes'])]:assert np.array_equal(independently_raw_affinity(v['A_boxes'].astype(float),b.astype(float),xyz.astype(float)),g[view+'_affinity']);rawchecks+=1
   for index,arm in enumerate(ARMS):
    if arm=='source':assert np.array_equal(s['scores'][index],v['A_scores']);continue
    m=source if arm in ['frozen_AB','frozen_ABC'] else head({n:f[arm+'__'+n] for n in names});views='AB' if arm=='frozen_AB' else 'ABC';rr=ensemble(m,v,c,g,views);err=float(abs(rr-s['scores'][index]).max());assert err<=3e-5,(arm,err);assert int(rr.argmax())==int(s['queries'][index]),('top1',sid,arm);maxerr=max(maxerr,err);replayed+=1
   for kind in ['fixed','ema']:
    tr=rec['episodic'][kind]['trace'];assert len(tr)==16 and all(x['gradient_norm']>=0 and x['target_requires_grad'] is False for x in tr);assert all(x['teacher_step_L2']==0 for x in tr) if kind=='fixed' else all(x['teacher_step_L2']>0 for x in tr)
   for n in names:
    expected=np.float32(.99)*initial[n]+np.float32(.01)*f['epi_ema_student_1__'+n];assert np.max(abs(expected-f['epi_ema_teacher_1__'+n]))<1e-7;emachecks+=1
   for k in ['fixed_A','fixed_AB','ema_A','ema_AB']:
    for who in ['student','teacher']:
     if who=='teacher' and k.startswith('fixed'):continue
     for tim in ['pre','post']:
      hh=hashlib.sha256()
      for n in names:hh.update(n.encode()+f[f'online_{k}_{who}_{tim}__'+n].tobytes())
      assert hh.hexdigest()==rec['online'][k][who+'_'+tim+'_hash']
   prepost[sid]={k:{tim:{n:f[f'online_{k}_student_{tim}__'+n].copy() for n in names} for tim in ['pre','post']} for k in ['fixed_A','fixed_AB','ema_A','ema_AB']}
   for aug in ['A','AB']:
    for n in names:
     pre=f[f'online_ema_{aug}_teacher_pre__'+n];post=f[f'online_ema_{aug}_teacher_post__'+n];spost=f[f'online_ema_{aug}_student_post__'+n];expected=np.float32(.99)*pre+np.float32(.01)*spost;assert np.max(abs(expected-post))<1e-7;emachecks+=1
   stats.append({'platform':row['platform'],'epi_fixed_first_gradient':rec['episodic']['fixed']['trace'][0]['gradient_norm'],'epi_ema_first_gradient':rec['episodic']['ema']['trace'][0]['gradient_norm'],'B_unknown':1-float(g['B_fixed_mass']),'C_unknown':1-float(g['C_fixed_mass'])})
   if len(stats)%32==0:print('AUDIT57',len(stats),336,replayed,flush=True)
  orders=read(pred/'ONLINE_ORDERS_PRIVATE.json')
  for pl,stream in orders.items():
   expected=sorted([r['id'] for r in rows if r['platform']==pl],key=lambda sid:hashlib.sha256(('R57_online_order_v1|'+sid).encode()).hexdigest());assert stream==expected
   for k in ['fixed_A','fixed_AB','ema_A','ema_AB']:
    previous=None
    for i,sid in enumerate(stream):
     rec=rec_byid[sid]['online'][k];assert rec['stream_index']==i and rec['optimizer_step']==i+1
     if previous is None:
      for n in names:assert np.array_equal(prepost[sid][k]['pre'][n],initial[n])
     else:
      assert rec['student_pre_hash']==previous['student_post_hash'] and rec['teacher_pre_hash']==previous['teacher_post_hash']
     previous=rec
  controls=read(pred/'IDENTICAL_INPUT_ZERO_GRADIENT_CONTROLS.json');assert len(controls)==4 and max(x['gradient_norm'] for x in controls)<1e-6
  result={'state':'PASS_R57_INDEPENDENT_CPU_REPLAY_EMA_RECURRENCE_STREAM_RESET_AND_FIXED_BOX_AUDIT','n':336,'arm_names':ARMS,'prediction_seal_sha256':sha(pred/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'all_fitted_endpoint_replays':replayed,'CPU_max_absolute_probability_error':maxerr,'CPU_argmax_disagreements':0,'raw_affinity_matrices_exact':rawchecks,'EMA_parameter_tensor_recurrences':emachecks,'optimizer_steps':12096,'no_task_labels_models_or_images':True,'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);dump(out/'ALL_ASSOCIATION_AND_GRADIENT_SCALARS_PUBLIC_SAFE.json',stats);print(result,flush=True)
 except BaseException as e:dump(out/'FAILED.json',{'type':type(e).__name__,'error':str(e),'completed':len(stats),'task_labels_accessed':False});raise
if __name__=='__main__':main()
