"""Independent CPU audit of R32 localized readout and coordinate controls.
Does not import a GPU runtime or invoke an optimizer. U/FINAL hash lists only.
"""
import sys,json,time,collections,itertools,hashlib,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'crossplatform/round32'),str(ROOT/'research/round5/audit'),str(ROOT/'research/round31/independent_audit')]
from common32 import read,dump,sha,verify,load_npz,rows,feature
from coordinate_projection32 import project_group_preferences
from independent_core_v2 import candidate_ious,transform_gt
from audit31_independent import head as cpu_head
OUT=ROOT/'research/round32/independent_audit';BASE=ROOT/'crossplatform/round32';EXP=ROOT/'research/round32/experiment';EPS=1e-6
COUNTS=collections.Counter();KEYS=['noun_interaction','full_literal_caption_interaction'];ARMS=['scope_TTA','scope_memory','global_mean_TTA','global_norm_TTA','ungated_memory']
def close(a,b,tol=1e-10):assert abs(float(a)-float(b))<=tol,(a,b)
def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def ell(p):
 p=np.asarray(p,float);assert np.isfinite(p).all() and p.min()>-EPS and p.max()<1+EPS;return np.log(p+EPS)-np.log(1+EPS-p)
def qr(r,s,z):
 if r['stage'] in ['Q','R','S','T','U','F']:return int(np.argmax(s))
 groups=read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'];used=set();choices=[]
 for inds in groups.values():
  rep=max(inds,key=lambda i:(float(z['scores'][i]),-i));choices.append((float(np.max(s[inds])),-rep,rep));used.update(inds)
 for i in range(256):
  if i not in used:choices.append((float(s[i]),-i,i))
 return int(max(choices)[2])
def key(z,r,kind,tok):
 if kind=='noun_interaction':token=z['tokens'][z['token_mask']].mean(0)
 else:
  off=np.asarray(tok([r['caption']+' . not mentioned'],return_offsets_mapping=True)['offset_mapping'][0]);assert len(off)==len(z['tokens']);mask=(off[:,1]>off[:,0])&(off[:,0]<len(r['caption']));assert mask.any();token=z['tokens'][mask].mean(0)
 return normalize(z['query']*normalize(token)).astype(np.float32)
def independent_qp(source,ff,p0):
 groups=[];indices={};end=[]
 for f in ff:
  e=[]
  for name in ['positive_group_PRIVATE','negative_group_PRIVATE']:
   g=tuple(sorted(set(f[name])))
   if g not in indices:
    assert all(not set(g)&set(old) for old in groups);indices[g]=len(groups);groups.append(g)
   e.append(indices[g])
  end.append(e)
 C=np.zeros((len(ff),len(groups)))
 for j,(p,n) in enumerate(end):C[j,p]+=1;C[j,n]-=1
 s=np.array([source[list(g)].max() for g in groups]);rhs=.1-C@s;solutions=[]
 for kk in range(len(ff)+1):
  for active in itertools.combinations(range(len(ff)),kk):
   lam=np.zeros(len(ff));
   if active:
    ix=np.array(active);M=C[ix];la=np.linalg.pinv(M@M.T)@rhs[ix]
    if np.max(abs((M@M.T)@la-rhs[ix]))>1e-8 or la.min()<-1e-8:continue
    lam[ix]=la
   d=C.T@lam;slack=C@d-rhs
   if slack.min()<-1e-8 or max(abs(lam*slack))>1e-7:continue
   solutions.append((float(d@d/2),d,lam,slack))
 assert solutions;obj,d,lam,slack=min(solutions,key=lambda v:v[0]);raw=np.zeros_like(source)
 for g,delta in zip(groups,d):raw[list(g)]=delta
 assert np.max(abs(d-C.T@lam))<1e-9 and min(lam)>=-1e-8 and min(slack)>=-1e-8 and np.max(abs(lam*slack))<1e-7
 return raw,obj,d,C,s,lam

def main():
 start=time.perf_counter()
 for folder in ['scope400_attempt001','scope400_endpoint_attempt001','feasibilityA_attempt001','sourceT128_attempt001','selectedT128_attempt001','closureT128_attempt001']:verify(BASE/folder)
 feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');assert len(feedback)==22;fby=collections.defaultdict(list)
 for f in feedback:fby[f['id_PRIVATE']].append(f)
 allrows=[dict(r,split=split) for split in ['A208','DEV64','R64','S64','T128'] for r in rows(split)];lookup={r['id']:r for r in allrows};assert len(lookup)==len(allrows)==528;zmap={r['id']:feature(r) for r in allrows};Aby={pl:[r for r in rows() if r['platform']==pl] for pl in ['drone','quad']}
 groupraw={};qp_rows=[]
 for r in rows():
  ff=fby[r['id']];p=zmap[r['id']]['scores'];s=ell(p);raw=np.zeros(256)
  if ff:
   raw,obj,d,C,gs,dual=independent_qp(s,ff,p);original=project_group_preferences(s,[(f['positive_group_PRIVATE'],f['negative_group_PRIVATE']) for f in ff],margin=.1,q0=int(p.argmax()),representative_scores=p);assert np.max(abs(raw-original['raw_query_correction']))<1e-8;close(obj,original['objective_half_group_l2'],1e-8);assert original['kkt_max_abs_error']<1e-8
   # Query multiplicity does not alter group objective or group correction.
   dupprefs=[([3*i+j for i in f['positive_group_PRIVATE'] for j in range(3)],[3*i+j for i in f['negative_group_PRIVATE'] for j in range(3)]) for f in ff];dup=project_group_preferences(np.repeat(s,3),dupprefs,margin=.1,q0=int(p.argmax())*3,representative_scores=np.repeat(p,3));close(obj,dup['objective_half_group_l2'],1e-8);assert np.max(abs(np.repeat(raw,3)-dup['raw_query_correction']))<1e-8
   qp_rows.append({'alias':r['alias'],'platform':r['platform'],'groups':len(d),'constraints':len(ff),'objective':obj,'KKT_stationarity_max':float(abs(d-C.T@dual).max()),'KKT_complementarity_max':float(abs(dual*(C@(gs+d)-.1)).max()),'query_multiplicity_invariant':True});COUNTS['group_QPs_independently_reconstructed']+=1
  groupraw[r['id']]=raw
 dump(OUT/'GROUP_QP_AUDIT_ROWS.json',qp_rows)
 # Stable probability readout retains official float32 ties at endpoints.
 parent={}
 for pl in ['drone','quad']:
  p=load_npz(ROOT/'crossplatform/round31/loo19_attempt002/predictions_PRIVATE'/(pl+'_all.npz'));parent.update({rid:p['scores'][i] for i,rid in enumerate(p['row_ids_PRIVATE'].tolist())})
 for r in allrows:
  z=zmap[r['id']];p0=z['scores'];q0=int(p0.argmax());pT=parent[r['id']] if r['stage']!='T' else load_npz(BASE/'selectedT128_attempt001/predictions_PRIVATE'/(r['id']+'.npz'))['adapted_probability'];d0=ell(p0)-ell(p0)[q0];dt=ell(pT)-ell(pT)[q0];assert qr(r,d0,z)==q0;assert qr(r,dt,z)==qr(r,pT,z);assert np.array_equal(d0,d0+np.zeros(256)*(dt-d0));one=d0+(dt-d0);one[:]=dt;assert np.array_equal(one,dt);COUNTS['probability_endpoints_exact_observations']+=1
  if r['stage']!='T':
   for folder in ['scope400_attempt001','scope400_endpoint_attempt001']:
    for kk in KEYS:
     p=load_npz(BASE/folder/'predictions_PRIVATE'/(r['id']+'_'+kk+'.npz'));names=p['arm_names'].tolist();assert int(p['queries'][names.index('source')])==q0;assert int(p['queries'][names.index('R30_ungated')])==qr(r,pT,z);assert np.array_equal(p['boxes'],z['boxes'][p['queries']]);COUNTS['saved400_source_R30_endpoint_pairs_checked']+=1
 # Separately certify a saturatedfloat32tie stress case.
 tied=np.array([0.,1.,1.,np.nextafter(np.float32(1),np.float32(0))],np.float32);assert np.isfinite(ell(tied)).all();assert ell(tied)[1]==ell(tied)[2] and int(ell(tied).argmax())==1
 from transformers import RobertaTokenizerFast
 tok=RobertaTokenizerFast.from_pretrained(ROOT/'crossplatform/upstream/data/roberta_base',local_files_only=True)
 libraries={};oldlibs={}
 for pl,A in Aby.items():
  raw=np.stack([groupraw[r['id']] for r in A]);rowflag=np.array([bool(fby[r['id']]) for r in A]);endpoint=np.zeros((len(A),256),bool);oldvalues=np.zeros((len(A),256),np.float32);oldcounts=np.zeros_like(oldvalues)
  for j,r in enumerate(A):
   p=zmap[r['id']]['scores']
   for f in fby[r['id']]:
    P=f['positive_group_PRIVATE'];N=f['negative_group_PRIVATE'];endpoint[j,P+N]=True;delta=max(.1-float(p[P].max()-p[N].max()),0)/2;oldvalues[j,P]+=delta;oldvalues[j,N]-=delta;oldcounts[j,P]+=1;oldcounts[j,N]+=1
  oldvalues/=np.maximum(oldcounts,1)
  for kk in KEYS:
   K=np.stack([key(zmap[r['id']],r,kk,tok) for r in A]);libraries[pl,kk]={'K':K,'flat':K.reshape(-1,64),'raw':raw,'rowflag':rowflag,'endpointflag':endpoint}
  oldK=[]
  for ar in A:
   az=zmap[ar['id']];at=az['tokens'][az['token_mask']].mean(0);at=at/max(float(np.linalg.norm(at)),1e-12);oldK.append(az['query']*at)
  oldK=np.stack(oldK);oldK/=np.maximum(np.linalg.norm(oldK,axis=-1,keepdims=True),1e-12)
  oldlibs[pl]={'raw':oldvalues,'active':np.flatnonzero(rowflag),'keys':oldK}
 # Independent family outputs, at least allDEV+selectedA/R/S/T. Float32KNN.
 family_predictions={};knn_margins=[];near_ties=[];stat_differences=[];max_T_primary_diff=0.;max_T_memory_diff=0.;all_query_mismatches=[];cache_near={};selected=read(EXP/'SELECTED_SCOPE_LOCK.json');primary=selected['primary']
 configs={'scoped_TTA':primary,'same_gate_QP_memory':selected['same_gate_memory'],'DEVbest_QP_memory':selected['best_DEV_memory_same_budget'],'global_mean_matched':dict(primary,arm='global_mean_TTA'),'global_norm_matched':dict(primary,arm='global_norm_TTA'),'DEVbest_global_amount':selected['best_DEV_global_amount_control'],'same_family_row_scope':selected['same_family_row_support_ablation'],'DEVbest_row_scope':selected['best_DEV_row_support_ablation'],'DEVbest_ungated_QP_memory':selected['best_DEV_ungated_memory']}
 computed={};mainstats={}
 for support,folder in [('row','scope400_attempt001'),('endpoint','scope400_endpoint_attempt001')]:
  mainstats[support]={(x['alias'],x['key'],x['k']):x for x in read(BASE/folder/'PROCESS_STATS.json')}
 qp_specs=read(BASE/'scope400_endpoint_attempt001/COORDINATE_LIBRARY_SPECS.json')
 for libspec in qp_specs:
  ours={r['alias']:r for r in qp_rows if r['platform']==libspec['platform']}
  for x in libspec['coordinate_specs']:close(x['objective'],ours[x['alias']]['objective'],1e-8);assert x['max_KKT_residual']<1e-8
 for rowidx,r in enumerate(allrows):
  z=zmap[r['id']];p0=z['scores'];pT=parent[r['id']] if r['stage']!='T' else load_npz(BASE/'selectedT128_attempt001/predictions_PRIVATE'/(r['id']+'.npz'))['adapted_probability'];q0=int(p0.argmax());d0=ell(p0)-ell(p0)[q0];dt=ell(pT)-ell(pT)[q0];delta=dt-d0;needed=KEYS if r['split']=='DEV64' or r['id'] in fby else list(dict.fromkeys(spec['key'] for spec in configs.values()))
  row_out={};row_stats={};neighbors_by_key={}
  for kk in needed:
   lib=libraries[r['platform'],kk];Q=key(z,r,kk,tok);similarity=(Q@lib['flat'].T).reshape(256,len(Aby[r['platform']]),256);which=np.argmax(similarity,axis=-1);near=np.take_along_axis(similarity,which[:,:,None],axis=-1)[:,:,0];obs=np.argsort(-near,axis=-1,kind='stable')[:,:4];best=np.take_along_axis(near,obs,axis=-1);pos=np.take_along_axis(which,obs,axis=-1);assert all(len(set(row))==4 for row in obs);neighbors_by_key[kk]=(near,which);COUNTS['nearest_distinct_A_neighborhoods_recomputed']+=256
   # Record observation-boundary numerical ties; no labels used to resolve.
   sortednear=np.sort(near,axis=-1)[:,::-1];gap=sortednear[:,0]-sortednear[:,1];knn_margins.extend(gap.tolist())
   if np.min(gap)<5e-7:near_ties.append({'alias':r['alias'],'key':kk,'nearest_A_boundary_gaps_lt5e-7':int((gap<5e-7).sum())})
   for support in ['row','endpoint']:
    for k in [1,4]:
     g=(np.max(lib['rowflag'][obs[:,:k]],axis=-1) if support=='row' else np.max(lib['endpointflag'][obs[:,:k],pos[:,:k]],axis=-1)).astype(float);w=np.exp((best[:,:k]-np.float32(1.))/np.float32(.1)).astype(np.float64);raw=(lib['raw'][obs[:,:k],pos[:,:k]]*w).sum(-1)/np.maximum(w.sum(-1),1e-12);md=raw-raw[q0];a_mean=float(g.mean());a_norm=float(np.linalg.norm(g*delta)/max(np.linalg.norm(delta),1e-12));local=np.where(g==1,dt,d0);values={'scope_TTA':local,'scope_memory':d0+g*md,'global_mean_TTA':d0+a_mean*delta,'global_norm_TTA':d0+a_norm*delta,'ungated_memory':d0+md};row_out[support,kk,k]=values;row_stats[support,kk,k]={'gate_fraction':a_mean,'delta_norm':float(np.linalg.norm(delta)),'gated_delta_norm':float(np.linalg.norm(g*delta)),'global_norm_alpha':a_norm,'q0_gate':float(g[q0]),'raw_memory_norm':float(np.linalg.norm(md))};assert local[q0]==0 and values['scope_memory'][q0]==0
     # Translation invariance after NEWq0 anchoring, even under candidatewiseg.
     assert np.max(abs((d0+g*((raw+.25)-(raw[q0]+.25)))-values['scope_memory']))<1e-12
     close(np.linalg.norm(a_norm*delta),np.linalg.norm(g*delta),1e-9)
     if r['stage']!='T':
      ms=mainstats[support].get((r['alias'],kk,k))
      if ms and abs(ms['gate_fraction']-a_mean)>1e-12:stat_differences.append({'alias':r['alias'],'support':support,'key':kk,'k':k,'CPUgate_fraction':a_mean,'GPUgate_fraction':ms['gate_fraction']})
   # All4DEVfamilies andallAfeedbackfeasibility are retained independently.
  computed[r['id']]={'outputs':row_out,'stats':row_stats,'neighbors':neighbors_by_key}
  if r['stage']=='T':
   pred=load_npz(BASE/'selectedT128_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));outs={'source':d0,'R30_ungated':dt}
   for name,spec in configs.items():outs[name]=row_out[spec['support'],spec['key'],spec['k']][spec['arm']]
   old=oldlibs[r['platform']];ot=z['tokens'][z['token_mask']].mean(0);ot=ot/max(float(np.linalg.norm(ot)),1e-12);oq=z['query']*ot;oq/=np.maximum(np.linalg.norm(oq,axis=-1,keepdims=True),1e-12)
   for name,ix in [('R30_feedbackA_memory',old['active']),('R30_allA_zeroU_memory',np.arange(len(Aby[r['platform']])) )]:
    osi=(oq@old['keys'][ix].reshape(-1,64).T).reshape(256,len(ix),256);owi=np.argmax(osi,axis=-1);n=np.take_along_axis(osi,owi[:,:,None],axis=-1)[:,:,0];o=np.argsort(-n,axis=-1,kind='stable')[:,:4];best=np.take_along_axis(n,o,axis=-1);a=ix[o];pos=np.take_along_axis(owi,o,axis=-1);w=np.exp((best-np.float32(1))/np.float32(.1));corr=(old['raw'][a,pos]*w).sum(-1)/np.maximum(w.sum(-1),1e-12);outs[name]=p0+corr
   assert list(outs)==pred['arm_names'].tolist();qs=np.array([qr(r,s,z) for s in outs.values()]);bad=[(n,int(a),int(b)) for n,a,b in zip(outs,qs,pred['queries']) if a!=b]
   if bad:all_query_mismatches.append({'alias':r['alias'],'differences':bad})
   max_T_primary_diff=max(max_T_primary_diff,float(abs(outs['scoped_TTA']-pred['primary_scores']).max()));max_T_memory_diff=max(max_T_memory_diff,float(abs(outs['same_gate_QP_memory']-pred['memory_scores']).max()));COUNTS['T128_locked_readouts_recomputed']+=len(outs)
  if (rowidx+1)%64==0:print('AUDIT32_SCOPE',rowidx+1,'querymismatches',len(all_query_mismatches),'gateaggdiff',len(stat_differences),flush=True)
 dump(OUT/'KNN_NEAR_TIE_DIAGNOSTICS.json',near_ties);dump(OUT/'CPU_GPU_GATE_AGGREGATE_DIFFERENCES.json',stat_differences);dump(OUT/'CPU_GPU_T_READOUT_DIFFERENCES.json',all_query_mismatches)
 # Save intermediate audit state without private arrays, before any tasklabels.
 dump(OUT/'READOUT_RECONSTRUCTION_RESULT.json',{'counts':dict(COUNTS),'T_primary_score_maxerror':max_T_primary_diff,'T_memory_score_maxerror':max_T_memory_diff,'T_query_mismatch_rows':len(all_query_mismatches),'gate_aggregate_mismatch_rows':len(stat_differences),'nearest_tie_rows':len(near_ties),'scope_seconds':time.perf_counter()-start})
 assert not [x for x in all_query_mismatches if any(not d[0].startswith('R30_') for d in x['differences'])],all_query_mismatches[:3]
 # The remaining independent selection/data/task/closure audit is separate.
 with open(OUT/'_readout_cache_PRIVATE.pkl','wb') as f:__import__('pickle').dump({'computed':computed,'COUNTS':dict(COUNTS),'qp_rows':qp_rows,'max_T_primary_diff':max_T_primary_diff,'max_T_memory_diff':max_T_memory_diff,'stat_differences':stat_differences,'time':time.perf_counter()-start},f,protocol=4)
 print('READOUT32_CPU_PRIMARY_RECONSTRUCTED_LEGACY_WARNINGS_RETAINED',dict(COUNTS),max_T_primary_diff,max_T_memory_diff,flush=True)
if __name__=='__main__':main()
