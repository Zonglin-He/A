"""Independent functional reconstruction, chronological memory and R58 replay.
Does not import the R64 producer or memory implementation; no task labels.
"""
import argparse,hashlib,sys,time,torch
from common64 import *
sys.path.insert(0,str(ROOT/'research/round58'));import independent58 as old

def norm(a):
 n=np.linalg.norm(a,axis=-1,keepdims=True);return np.divide(a,n,out=np.zeros_like(a),where=n>1e-12)
def raw_forward(h,w):
 x=torch.tensor(h)
 for k in ['0','2']:x=torch.nn.functional.linear(x,torch.tensor(w[k+'.weight']),torch.tensor(w[k+'.bias'])).relu()
 y=torch.nn.functional.linear(x,torch.tensor(w['4.weight']),torch.tensor(w['4.bias'])).numpy();a=np.column_stack([x.numpy().astype(float),np.ones(256)]);b=np.column_stack([w['4.weight'],w['4.bias']]).astype(float);return y,a@b.T

def readout(raw,z):
 probs=[]
 for v in 'ABC':
  x=torch.tensor(np.asarray(raw[v],np.float32));q=torch.nn.functional.normalize(x,dim=-1);m=torch.tensor(z[v+'_token_mask']).bool();l=q@torch.tensor(z[v+'_tokens']).T/.07;s=torch.logsumexp(l.masked_fill(~m[None],-torch.inf),-1)-torch.logsumexp(l.masked_fill(m[None],-torch.inf),-1);p=s.double().softmax(-1).numpy()
  if v!='A':
   p=z[v+'_W']@p;mass=float(z[v+'_fixed_mass']);p=mass*p/max(float(p.sum()),1e-300)+(1-mass)*z['base']
  probs.append(p)
 return sum(probs)/3

def check_prepared(r,z,cap,g,w,subject,cfg):
 raw={};raw64={}
 for v in 'ABC':raw[v],raw64[v]=raw_forward(cap[v+'_h'],w)
 qs={v:norm(raw64[v]) for v in 'ABC'};W=g['B_W'];m=W.sum(1);mapped=np.divide(W@qs['B'],m[:,None],out=np.zeros((256,64)),where=m[:,None]>0);a=norm(qs['A']+g['B_affinity'].max(1)[:,None]*mapped);target={'A':np.linalg.norm(raw64['A'],axis=1)[:,None]*a};rho={'A':g['B_affinity'].max(1)}
 for v in 'BC':
  W=g[v+'_W'];m=W.sum(0);u=np.divide(W.T@a,m[:,None],out=np.zeros((256,64)),where=m[:,None]>0);valid=np.linalg.norm(u,axis=1)>1e-12;unitu=norm(u);unitu[~valid]=qs[v][~valid];target[v]=np.linalg.norm(raw64[v],axis=1)[:,None]*unitu;rho[v]=g[v+'_affinity'].max(0)*valid
 offs=cap['token_offsets'];content=(offs[:,1]>offs[:,0])&(offs[:,0]<len(r['caption']))&(offs[:,1]<=len(r['caption']));context=content&~subject;err=0.
 for v in 'ABC':
  fused=(1-rho[v][:,None])*raw[v]+rho[v][:,None]*target[v];q=torch.nn.functional.normalize(torch.tensor(raw[v]),dim=-1).numpy();qf=torch.nn.functional.normalize(torch.tensor(fused.astype(np.float32)),dim=-1).numpy();tk=cap[v+'_tokens'];s=norm(tk[subject].astype(float).mean(0)) if subject.any() else np.zeros(64);c=norm(tk[context].astype(float).mean(0)) if context.any() else s.copy()
  for key,val in [(v+'_fused_raw',fused),(v+'_q',q),(v+'_subject_key',s),(v+'_context_key',c)]:err=max(err,float(abs(z[key]-val).max()));assert np.allclose(z[key],val,atol=2e-12,rtol=0),key
  assert np.array_equal(z[v+'_tokens'],tk) and np.array_equal(z[v+'_token_mask'],cap[v+'_token_mask'])
  if v=='A':assert np.array_equal(z['delta'],qf.astype(float)-q.astype(float))
 mapped={}
 for v in 'BC':
  assert np.array_equal(z[v+'_W'],g[v+'_W']) and z[v+'_fixed_mass']==g[v+'_fixed_mass'];m=g[v+'_W'].sum(1);mapped[v]=norm(np.divide(g[v+'_W']@z[v+'_q'].astype(float),m[:,None],out=np.zeros((256,64)),where=m[:,None]>0))
 ab=np.sum(z['A_q']*mapped['B'],1);ac=np.sum(z['A_q']*mapped['C'],1);bc=np.sum(mapped['B']*mapped['C'],1);ge=np.minimum(g['B_affinity'].max(1),g['C_affinity'].max(1));rel=ge*np.maximum(bc,0);valid=(ge>=cfg['geometry_affinity_min'])&(ab>=cfg['source_cross_view_cos_min'])&(ac>=cfg['source_cross_view_cos_min'])&(bc>=cfg['source_cross_view_cos_min'])&(np.linalg.norm(z['delta'],axis=1)>cfg['delta_nonzero_min'])&bool(subject.any());assert np.array_equal(z['eligible'],valid) and np.array_equal(z['reliability'],rel);assert str(z['physical_frame_sha'])==hashlib.sha256((r['pointpath']+'|'+r['imagepath']+'|'+r['metapath']).encode()).hexdigest();assert np.array_equal(z['source_boxes'],cap['A_boxes']) and np.array_equal(z['source_scores'],cap['A_scores']);return err

def box_iou(a,b):
 extent=np.maximum(np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2)-np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2),0);i=extent.prod();d=np.maximum(a[3:],0).prod()+np.maximum(b[3:],0).prod()-i;return i/d if d>0 else 0.
def entries(z,c,index,pos):
 ix=list(np.flatnonzero(z['eligible']));ix.sort(key=lambda j:(-float(z['reliability'][j]),int(j)));chosen=[];out=[]
 for j in ix:
  if any(box_iou(z['source_boxes'][j].astype(float),z['source_boxes'][k].astype(float))>=c['within_input_duplicate_box_IoU'] and z['A_q'][j].astype(float)@z['A_q'][k].astype(float)>=c['within_input_duplicate_feature_cos'] for k in chosen):continue
  chosen.append(j);out.append(dict(q=z['A_q'][j].astype(float),s=z['A_subject_key'],c=z['A_context_key'],d=z['delta'][j],r=float(z['reliability'][j]),uid=index*256+int(j),time=pos,frame=str(z['physical_frame_sha'])))
  if len(out)==c['per_input_write_cap']:break
 return out

def write(pool,items,c):
 replace=evict=0
 for e in items:
  dup=None
  for k,h in enumerate(pool):
   ratio=np.linalg.norm(e['d'])/max(float(np.linalg.norm(h['d'])),1e-12)
   if e['q']@h['q']>=c['historical_duplicate_feature_cos'] and e['s']@h['s']>=c['historical_duplicate_text_cos'] and e['c']@h['c']>=c['historical_duplicate_text_cos'] and norm(e['d'])@norm(h['d'])>=c['historical_duplicate_delta_cos'] and c['historical_duplicate_delta_norm_ratio_min']<=ratio<=c['historical_duplicate_delta_norm_ratio_max']:dup=k;break
  if dup is not None:
   if e['r']<=pool[dup]['r']:continue
   pool.pop(dup);replace+=1
  pool.append(e)
  if len(pool)>c['capacity']:pool.pop(0);evict+=1
 return replace,evict

def retrieve(pool,z,v,c,kind,salt,pos):
 n=len(pool);out=np.zeros((256,64));trace={'pool_n':n,'matched_candidates':0,'selected_edges':0,'max_history_time':-1,'shuffle_possible':n>1,'median_correction_norm':0.}
 if not pool:return out,trace
 assert all(e['time']<pos for e in pool)
 Q=np.array([e['q'] for e in pool]);S=np.array([e['s'] for e in pool]);C=np.array([e['c'] for e in pool]);D=np.array([e['d'] for e in pool]);R=np.array([e['r'] for e in pool]);a=z[v+'_q'].astype(float)@Q.T;b=z[v+'_subject_key']@S.T;d=z[v+'_context_key']@C.T;ok=(a>=c['feature_cos_min'])&np.array([e['frame']!=str(z['physical_frame_sha']) for e in pool])[None];k=c['beta']*(a-1)
 if kind!='no_explicit_text':ok&=(b[None]>=c['subject_cos_min'])&(d[None]>=c['context_cos_min']);k+=c['beta']*((b-1)+(d-1))[None]
 if kind=='shuffled_values' and n>1:
  offset=1+int(hashlib.sha256(('R64_value_shuffle_v1|'+salt).encode()).hexdigest()[:12],16)%(n-1);perm=np.roll(np.arange(n),offset);assert np.all(perm!=np.arange(n));D=D[perm];R=R[perm]
 k+=np.log(np.clip(R,1e-300,None));used=[];uids=[]
 for i in range(256):
  ids=np.where(ok[i])[0];ids=ids[np.argsort(-k[i,ids],kind='stable')[:c['topk']]]
  if not len(ids):continue
  w=np.exp(k[i,ids]-max(k[i,ids]));w/=sum(w);out[i]=w@D[ids];used.append(i);uids.extend(pool[j]['uid'] for j in ids)
 us=set(uids);trace.update(matched_candidates=len(used),selected_edges=len(uids),max_history_time=max((e['time'] for e in pool if e['uid'] in us),default=-1),median_correction_norm=float(np.median(np.linalg.norm(out[used],axis=1))) if used else 0.,selected_key_UID_sha256=hashlib.sha256(np.asarray(uids,np.int64).tobytes()).hexdigest());return out,trace

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);rows=read(ROOT/p['manifest']);index={r['id']:i for i,r in enumerate(rows)};by={r['id']:r for r in rows};orders=read(ROOT/p['orders']);pred=ROOT/p['prediction_root'];sealpred=read(pred/'SEALED.json');prep=ROOT/p['prepared_root'];sp=read(prep/'SEALED.json');assert sealpred['n']==336;arms=[o+'__'+a for o in p['order_names'] for a in p['per_order_arms']];assert arms==sealpred['arm_names'];data={};saved={};cases={};w=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');roles={r['id_PRIVATE']:np.array(r['mask'],bool) for r in read(ROOT/'research/round62/role_diagnostics_attempt001/MASKS_PRIVATE.json') if r['role']=='subject'};start=time.perf_counter();errors={'prepared':0.,'probability':0.,'R58_KL':0.};cnt={'prepared':0,'memory_readouts':0,'R58_original_reused':0,'R58_independent_updates':0,'active_memory_inputs':0,'shuffled_different_Top1':0};statuses=read(pred/'READOUT_STATUS_PRIVATE.json');status={r['id']:r['arm_statuses'] for r in statuses}
 for rel,h in sealpred['files'].items():assert sha(pred/rel)==h
 for r in rows:
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(prep/rel)==sp['files'][rel];z=load(prep/rel);cap=load(ROOT/p['capture_root']/rel);g=load(ROOT/p['association_root']/rel);errors['prepared']=max(errors['prepared'],check_prepared(r,z,cap,g,w,roles[sid],p['memory']));data[sid]=z;saved[sid]=load(pred/rel);cases[sid]=read(pred/'cases_PRIVATE'/(sid+'.json'));cnt['prepared']+=1;assert saved[sid]['arm_names'].tolist()==arms and np.array_equal(saved[sid]['queries'],saved[sid]['scores'].argmax(1));assert np.array_equal(saved[sid]['selected_original_boxes'],z['source_boxes'][saved[sid]['queries']])
 def compare(sid,order,name,val):
  arm=order+'__'+name;z=saved[sid];expected=z['scores'][arms.index(arm)];err=float(abs(expected-val).max());errors['probability']=max(errors['probability'],err);assert err<1e-11,(order,name,err);assert val.argmax()==expected.argmax(),(order,name,'Top1')
 for order,streams in orders.items():
  for pl,stream in streams.items():
   pool=[]
   if order!='original':s,t,opt=old.create(w)
   for pos,sid in enumerate(stream):
    z=data[sid];r=cases[sid][order];assert r['stream_index']==pos and r['pool_before']==len(pool)
    for name,key in [('source','source_prob'),('frozen_ABC','ABC'),('r59_direct_all','R59'),('reset_each_input','R59')]:compare(sid,order,name,z[key])
    for kind in ['memory','shuffled_values','no_explicit_text']:
     raw={};match=0
     for v in 'ABC':
      d,tr=retrieve(pool,z,v,p['memory'],kind,order+'|'+pl+'|'+str(pos),pos);actual=r['memory'][kind][v];assert tr.keys()==actual.keys()
      for k in tr:
       if isinstance(tr[k],float):assert abs(tr[k]-actual[k])<1e-11
       else:assert tr[k]==actual[k],(kind,v,k)
      raw[v]=z[v+'_fused_raw'].copy();hit=np.linalg.norm(d,axis=1)>0;raw[v][hit]+=p['memory']['alpha']*np.linalg.norm(raw[v][hit],axis=1)[:,None]*d[hit];match+=tr['matched_candidates']
     compare(sid,order,kind,readout(raw,z));cnt['memory_readouts']+=1;assert status[sid][order+'__'+kind]['fallback']==(match==0)
     if kind=='memory':cnt['active_memory_inputs']+=int(match>0)
    if saved[sid]['queries'][arms.index(order+'__memory')]!=saved[sid]['queries'][arms.index(order+'__shuffled_values')]:cnt['shuffled_different_Top1']+=1
    ee=entries(z,p['memory'],index[sid],pos);rep,ev=write(pool,ee,p['memory']);assert (r['write_proposals'],r['replacements'],r['evictions'],r['pool_after'])==(len(ee),rep,ev,len(pool))
    if order=='original':
     prev=load(ROOT/p['r58_root']/'samples_PRIVATE'/(sid+'.npz'))
     for who in ['student','teacher']:
      for when in ['pre','post']:compare(sid,order,'r58_'+who+'_'+when,prev['scores'][list(prev['arm_names']).index('backtrack_online_ema_'+who+'_'+when)])
     cnt['R58_original_reused']+=1
     oldstatuses=read(ROOT/p['r58_root']/'READOUT_STATUS_PRIVATE.json');oldstatus={v['id']:v['arm_statuses'] for v in oldstatuses}[sid]
     for who in ['student','teacher']:
      for when in ['pre','post']:assert status[sid][order+'__r58_'+who+'_'+when]==oldstatus['backtrack_online_ema_'+who+'_'+when]
    else:
     ab=load(ROOT/p['AB_root']/'samples_PRIVATE'/(sid+'.npz'));c=load(ROOT/p['C_root']/'samples_PRIVATE'/(sid+'.npz'));g=load(ROOT/p['association_root']/'samples_PRIVATE'/(sid+'.npz'));d=old.inputs(ab,c,g)
     with torch.no_grad():compare(sid,order,'r58_student_pre',old.mean_prediction(s,d).numpy());compare(sid,order,'r58_teacher_pre',old.mean_prediction(t,d).numpy())
     sourcehash=old.digest(old.params(w));before={'student':old.digest(s),'teacher':old.digest(t)};rr=old.independently_step(s,t,opt,d,'ema','backtrack');errors['R58_KL']=max(errors['R58_KL'],old.compare_step(r['R58_step'],rr));cnt['R58_independent_updates']+=1
     for who,state in [('student',s),('teacher',t)]:
      for when,h in [('pre',before[who]),('post',old.digest(state))]:
       assert status[sid][order+'__r58_'+who+'_'+when]=={'unknown':False,'fallback':h==sourcehash,'no_parameter_update':h==sourcehash}
     with torch.no_grad():compare(sid,order,'r58_student_post',old.mean_prediction(s,d).numpy());compare(sid,order,'r58_teacher_post',old.mean_prediction(t,d).numpy())
    if (pos+1)%64==0:print('AUDIT64',order,pl,pos+1,flush=True)
 out=ROOT/'research/round64/independent_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R64_INDEPENDENT_PREPARATION_MEMORY_CHRONOLOGY_AND_R58_NUMERICS','n':336,'arm_names':arms,'counts':cnt,'errors':errors,'no_GT_in_runtime_or_audit':True,'read_before_write_all_streams':True,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_no_update_rows_retained':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);print(result)
if __name__=='__main__':main()
