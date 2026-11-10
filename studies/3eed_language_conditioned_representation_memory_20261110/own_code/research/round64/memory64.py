"""Language-conditioned correction memory. No labels, layers or optimizer."""
import numpy as np,hashlib
def unit(x):return x/max(float(np.linalg.norm(x)),1e-12)
def overlap(a,b):
 low=np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2);high=np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2);inter=np.maximum(high-low,0).prod();u=np.maximum(a[3:],0).prod()+np.maximum(b[3:],0).prod()-inter;return float(inter/u) if u>0 else 0.
def proposals(z,cfg,source_index,position):
 ids=np.flatnonzero(z['eligible']);ids=sorted(ids,key=lambda i:(-float(z['reliability'][i]),int(i)));chosen=[];result=[]
 for i in ids:
  if any(overlap(z['source_boxes'][i].astype(float),z['source_boxes'][j].astype(float))>=cfg['within_input_duplicate_box_IoU'] and z['A_q'][i].astype(float)@z['A_q'][j].astype(float)>=cfg['within_input_duplicate_feature_cos'] for j in chosen):continue
  chosen.append(i);result.append({'q':z['A_q'][i].astype(float).copy(),'s':z['A_subject_key'].copy(),'c':z['A_context_key'].copy(),'d':z['delta'][i].copy(),'r':float(z['reliability'][i]),'uid':source_index*256+int(i),'time':position,'frame':str(z['physical_frame_sha'])})
  if len(result)>=cfg['per_input_write_cap']:break
 return result
def insert(pool,items,cfg):
 replaced=evicted=0
 for e in items:
  dup=None
  for j,old in enumerate(pool):
   ratio=np.linalg.norm(e['d'])/max(float(np.linalg.norm(old['d'])),1e-12)
   if e['q']@old['q']>=cfg['historical_duplicate_feature_cos'] and e['s']@old['s']>=cfg['historical_duplicate_text_cos'] and e['c']@old['c']>=cfg['historical_duplicate_text_cos'] and unit(e['d'])@unit(old['d'])>=cfg['historical_duplicate_delta_cos'] and cfg['historical_duplicate_delta_norm_ratio_min']<=ratio<=cfg['historical_duplicate_delta_norm_ratio_max']:dup=j;break
  if dup is not None:
   if e['r']<=pool[dup]['r']:continue
   pool.pop(dup);replaced+=1
  pool.append(e)
  if len(pool)>cfg['capacity']:pool.pop(0);evicted+=1
 return replaced,evicted
def correction(pool,z,view,cfg,kind,salt,position):
 n=len(pool);out=np.zeros((256,64));meta={'pool_n':n,'matched_candidates':0,'selected_edges':0,'max_history_time':-1,'shuffle_possible':n>1,'median_correction_norm':0.}
 if not n:return out,meta
 assert max(e['time'] for e in pool)<position
 Q=np.stack([e['q'] for e in pool]);S=np.stack([e['s'] for e in pool]);C=np.stack([e['c'] for e in pool]);D=np.stack([e['d'] for e in pool]);R=np.array([e['r'] for e in pool]);frames=np.array([e['frame'] for e in pool]);cos=z[view+'_q'].astype(float)@Q.T;ss=z[view+'_subject_key']@S.T;cc=z[view+'_context_key']@C.T;valid=(cos>=cfg['feature_cos_min'])&(frames[None]!=str(z['physical_frame_sha']))
 if kind!='no_explicit_text':valid&=(ss[None]>=cfg['subject_cos_min'])&(cc[None]>=cfg['context_cos_min'])
 kernel=cfg['beta']*(cos-1)
 if kind!='no_explicit_text':kernel+=cfg['beta']*((ss-1)+(cc-1))[None]
 if kind=='shuffled_values' and n>1:
  shift=1+int(hashlib.sha256(('R64_value_shuffle_v1|'+salt).encode()).hexdigest()[:12],16)%(n-1);perm=np.roll(np.arange(n),shift);assert (perm!=np.arange(n)).all();D=D[perm];R=R[perm]
 kernel+=np.log(R.clip(1e-300));used=[];uids=[]
 for i in range(256):
  ids=np.flatnonzero(valid[i]);ids=ids[np.argsort(-kernel[i,ids],kind='stable')[:cfg['topk']]]
  if not len(ids):continue
  weights=np.exp(kernel[i,ids]-kernel[i,ids].max());weights/=weights.sum();out[i]=weights@D[ids];used.append(i);uids.extend(pool[j]['uid'] for j in ids)
 used_uid_set=set(uids)
 meta.update(matched_candidates=len(used),selected_edges=len(uids),max_history_time=max((pool[j]['time'] for j in range(n) if pool[j]['uid'] in used_uid_set),default=-1),median_correction_norm=float(np.median(np.linalg.norm(out[used],axis=1))) if used else 0.,selected_key_UID_sha256=hashlib.sha256(np.array(uids,np.int64).tobytes()).hexdigest())
 return out,meta
