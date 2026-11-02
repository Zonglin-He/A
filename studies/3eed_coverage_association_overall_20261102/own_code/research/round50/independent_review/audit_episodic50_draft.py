"""UNEXECUTED independent post-seal DEV336 matching-head contract audit.

Only frozen 184960-parameter head forwards may use CUDA, under a dedicated
root grant. No optimizer, full grounder, detector, VLM, original RGB/XYZ file,
annotation container, task score, or protected FINAL artifact is read/run.
Producer episodic/R50 algorithm functions are never imported as references.
"""
from pathlib import Path
import collections,hashlib,itertools,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'research/round50/experiment'
PLAN=EXP/'EPISODIC_INDEPENDENT_AUDIT_PLAN_LOCK.json'
AUTH=EXP/'ROOT_INDEPENDENT_EPISODIC_AUDIT_AUTHORIZATION.json'
OUT=Path(__file__).resolve().parent/'episodic336_audit_attempt001'
ARMS=['source','literal_A_wideG','group_A_wideG','sameinfo_consensus0','sameinfo_consensus1','sameinfo_consensus3']
STATE_SHAPES={'0.weight':(288,288),'0.bias':(288,), '2.weight':(288,288),'2.bias':(288,), '4.weight':(64,288),'4.bias':(64,)}
SEALS={};HASHES={}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 p=Path(p)
 if p not in HASHES:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  HASHES[p]=h.hexdigest()
 return HASHES[p]
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def local(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 assert 'FINAL384' not in rel and 'round46/final' not in rel and 'full5912_gpu_inputs' not in rel
 return ROOT/p
def seal(folder):
 folder=Path(folder)
 if folder not in SEALS:SEALS[folder]=read(folder/'SEALED.json')
 return SEALS[folder]
def member(folder,rel):
 folder=Path(folder);assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
 p=folder/rel;assert sha(p)==seal(folder)['files'][rel],('sealed_member',rel)
 return p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy(order='K') for k in z.files}
def exact(a,b,label):
 a=np.asarray(a);b=np.asarray(b)
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label
def state_ok(s):
 assert set(s)==set(STATE_SHAPES)
 assert sum(x.size for x in s.values())==184960
 for k,shape in STATE_SHAPES.items():assert s[k].shape==shape and s[k].dtype==np.float32 and np.isfinite(s[k]).all()
def box_iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);sa=np.maximum(a[:,3:],1e-6);sb=np.maximum(b[:,3:],1e-6)
 lo=np.maximum(a[:,None,:3]-sa[:,None]/2,b[None,:,:3]-sb[None]/2);hi=np.minimum(a[:,None,:3]+sa[:,None]/2,b[None,:,:3]+sb[None]/2)
 inter=np.maximum(hi-lo,0).prod(-1)
 return inter/np.maximum(sa.prod(-1)[:,None]+sb.prod(-1)[None]-inter,1e-12)
def groups(boxes,scores):
 overlaps=box_iou(boxes,boxes);members=[];representatives=[]
 for q in np.argsort(-scores,kind='stable'):
  group=next((i for i,r in enumerate(representatives) if overlaps[q,r]>=.5),None)
  if group is None:representatives.append(int(q));members.append([]);group=len(members)-1
  members[group].append(int(q))
 reps=np.asarray(representatives,int);membership=np.empty(256,np.int64)
 for i,g in enumerate(members):membership[g]=i
 return members,reps,membership
def matching(anchor,other):
 from scipy.optimize import linear_sum_assignment
 iou=box_iou(anchor,other)
 scale=np.maximum((np.maximum(anchor[:,3:],.1)[:,None]+np.maximum(other[:,3:],.1)[None])/2,.1)
 centre=np.mean(np.abs(anchor[:,None,:3]-other[None,:,:3])/scale,-1)
 size=np.mean(np.abs(np.log(np.maximum(anchor[:,None,3:],.05)/np.maximum(other[None,:,3:],.05))),-1)
 cost=1-iou+centre+.25*size;rr,cc=linear_sum_assignment(cost)
 ok=(iou[rr,cc]>=.1)|((centre[rr,cc]<.5)&(size[rr,cc]<.7))
 diag={'assigned':len(rr),'accepted':int(ok.sum()),'mean_accepted_IoU':float(iou[rr[ok],cc[ok]].mean()) if ok.any() else None}
 return rr[ok],cc[ok],diag
def group_scores(s,g):return np.asarray([np.max(s[x]) for x in g])
def choose_group(s,reps):return int(max(range(len(reps)),key=lambda j:(float(s[j]),-int(reps[j]))))
def unique_mask(valid,q):
 values=np.flatnonzero(valid[int(q)])
 return int(values[0]) if len(values)==1 else None
def structure(v,validA,validB,FA,FB):
 ga,geometry_ra,ma=groups(v['A_boxes'],v['A_scores']);gb,rb,mb=groups(v['B_boxes'],v['B_scores'])
 # Geometry/matching/identity retain original source representatives. Output
 # representatives retain the original G winner within each frozen group.
 ra=np.asarray([min(g,key=lambda q:(-float(FA[q]),int(q))) for g in ga],int)
 rr,cc,diag=matching(v['A_boxes'][geometry_ra],v['B_boxes'][rb]);mapping={int(a):int(b) for a,b in zip(rr,cc)}
 q0=int(v['A_scores'].argmax());qG=int(FA.argmax());negative=int(ma[q0]);positive=int(ma[qG])
 mn=unique_mask(validA,geometry_ra[negative]);mp=unique_mask(validA,geometry_ra[positive]);gs=group_scores(FA,ga);gap=float(gs[positive]-gs[negative])
 pre=bool(positive!=negative and mn is not None and mp is not None and mn!=mp and gap>0)
 reason='A_EXACT_TIE_IS_NOT_A_STRICT_PREFERENCE' if positive!=negative and gap==0 else 'NO_DISTINCT_UNIQUE_A_INSTANCE_PREFERENCE';accepted=False
 if pre:
  reason='UNKNOWN_A_GROUP_NOT_MATCHED_IN_B'
  if negative in mapping and positive in mapping:
   nb,pb=mapping[negative],mapping[positive];reason='UNKNOWN_B_MASK_IDENTITY_MISMATCH'
   if unique_mask(validB,rb[nb])==mn and unique_mask(validB,rb[pb])==mp:
    sb=group_scores(FB,gb);accepted=bool(sb[pb]>sb[nb])
    reason='ACCEPTED_LOCAL_TWO_VIEW_PREFERENCE' if accepted else 'G_B_DOES_NOT_SUPPORT_SAME_LOCAL_DIRECTION'
 return {'groupsA':ga,'groupsB':gb,'repsA':ra,'geometry_repsA':geometry_ra,'repsB':rb,'membershipA':ma,'membershipB':mb,
  'match_A':rr,'match_B':cc,'mapping':mapping,'matching_diagnostics':diag,
  'q0':q0,'qG':qG,'source_group':negative,'positive_group':positive,'strict_A_group_gap':gap,
  'pre_A_condition':pre,'accepted':accepted,'reason':reason,'q0_group_has_B_mapping':negative in mapping,
  'mask_source_A':mn,'mask_positive_A':mp}
def old_read(s,w,pool,e,pr):
 V=np.zeros(256);P=np.zeros(256);R=np.zeros(256);active=w.sum(1)>0
 if len(pool):
  cue=w@e[pool];relation=w@pr[pool];q=int(s.argmax())
  anchor=cue[q] if active[q] else float(np.median(e[pool]));ar=relation[q] if active[q] else np.median(pr[pool],axis=0)
  V[active]=np.clip((cue[active]-anchor)/2,-2,2);P[active]=np.clip(relation[active,0]-ar[0],-2,2);R[active]=np.clip(relation[active,1]-ar[1],-2,2)
 return s.astype(float)+2*V+2*P+2*R,np.stack([V,P,R])
def consensus(pa,pb,ca,cb,st):
 fa=np.asarray(pa,float)+2*ca[0]+2*ca[1]+2*ca[2];fb=np.asarray(pb,float)+2*cb[0]+2*cb[1]+2*cb[2]
 if not st['pre_A_condition'] or not st['q0_group_has_B_mapping']:return fa,int(fa.argmax()),False
 aa=group_scores(fa,st['groupsA']);bb=group_scores(fb,st['groupsB']);q=st['source_group'];qB=st['mapping'][q]
 da=aa-aa[q];db=bb-bb[qB];out=da.copy()
 for a,b in st['mapping'].items():out[a]=(da[a]+db[b])/2
 return out[st['membershipA']],int(st['repsA'][choose_group(out,st['repsA'])]),True
def rect_iou(a,b):
 inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
 union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter
 return float(inter/union) if union>0 else 0.
def B_geometry(row,boxes,pool,own,stage,AC,AF,RF,CameraSpec,project):
 sid=row['id'];case=read(member(own,'cases_PRIVATE/'+sid+'.json'))
 frame=load(member(own,'frames_PRIVATE/'+case['frame_key_PRIVATE']+'.npz'))
 stagecase=read(member(stage,'cases_PRIVATE/'+sid+'.json'));mp=local(stagecase['mask_cache_path_PRIVATE'])
 assert sha(mp)==case['frame_mask_cache_sha256']==stagecase['mask_cache_sha256'];ins=load(mp)
 if row['stage']=='A':fr=AF[AC[sid]['frame_alias']]
 else:
  found=[r for r in RF.values() if all(r.get(k)==row[k] for k in ('imagepath','pointpath','metapath'))];assert len(found)==1;fr=found[0]
 assert all(fr[k]==row[k] for k in ('imagepath','pointpath','metapath')) and fr['point_sha256']==case['raw_XYZ_sha256']
 fields={k:fr[k] for k in ('image_intrinsic','image_extrinsic')}
 if fr['camera_distortion'] is not None:fields['camera_distortion']=fr['camera_distortion']
 camera=CameraSpec.from_camera_fields(row['platform'],fields,missing_distortion_policy='assume_zero_declared')
 H,W=map(int,ins['original_hw']);points=frame['visible_points_model'];owner=frame['point_owner'];depth=frame['depth_camera'];M=len(ins['class_ids'])
 support=np.zeros((256,M),int);coverage=np.zeros((256,M));overlap=np.zeros((256,M));ids=[]
 for m in range(M):
  indices=np.flatnonzero(owner==m)
  if len(indices)>=4:
   d=depth[indices];lo,hi=np.quantile(d,[.25,.75]);spread=max(hi-lo,.1);indices=indices[(d>=lo-1.5*spread)&(d<=hi+1.5*spread)]
  ids.append(indices)
 signs=np.asarray(list(itertools.product((-1,1),repeat=3)))
 for q,b in enumerate(boxes):
  if (b[3:]<=0).any():continue
  corners=np.asarray(b[:3],float)+signs*np.asarray(b[3:],float)/2;pr=project(corners,camera,(W,H));pv=bool((pr['depth']>.1).all() and np.isfinite(pr['uv']).all());rect=np.r_[pr['uv'].min(0),pr['uv'].max(0)] if pv else None
  for m,indices in enumerate(ids):
   n=int(np.count_nonzero((np.abs(points[indices]-b[:3])<=b[3:]/2+.1).all(1)));support[q,m]=n;coverage[q,m]=n/max(len(indices),1)
   if pv:overlap[q,m]=rect_iou(rect,ins['boxes_xyxy'][m])
 valid=(support>=3)&(coverage>=.15)&(overlap>=.1)&(boxes[:,3:]>0).all(1)[:,None]
 quality=np.where(valid,coverage*np.sqrt(np.maximum(overlap,0)),0.)
 weights=quality[:,pool].copy() if len(pool) else np.zeros((256,0));weights/=np.maximum(weights.sum(1,keepdims=True),1e-12)
 return weights,valid,support,coverage,overlap
def head_forward(state,h,tokens,mask,torch):
 F=torch.nn.functional
 with torch.no_grad():
  x=torch.tensor(h,device='cuda')[None]
  for i in (0,2,4):
   x=F.linear(x,torch.tensor(state[str(i)+'.weight'],device='cuda'),torch.tensor(state[str(i)+'.bias'],device='cuda'))
   if i<4:x=F.relu(x)
  q=F.normalize(x,dim=-1);t=torch.tensor(tokens,device='cuda')[None];m=torch.tensor(mask,device='cuda')[None,None]
  logits=q@t.transpose(-1,-2)/.07;p=(logits.softmax(-1)*m).sum(-1)[0]
  z=(torch.logsumexp(logits.masked_fill(~m,-torch.inf),-1)-torch.logsumexp(logits.masked_fill(m,-torch.inf),-1))[0]
 assert torch.isfinite(p).all() and torch.isfinite(z).all()
 return p.cpu().numpy(),z.cpu().numpy()
def cpu64(state,h,tokens,mask):
 x=np.asarray(h,np.float64)
 for i in (0,2,4):
  x=x@state[str(i)+'.weight'].astype(np.float64).T+state[str(i)+'.bias'].astype(np.float64)
  if i<4:x=np.maximum(x,0)
 x/=np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12);l=x@np.asarray(tokens,np.float64).T/.07
 def lse(a):
  m=a.max(-1,keepdims=True);return m[:,0]+np.log(np.exp(a-m).sum(-1))
 z=lse(l[:,mask])-lse(l[:,~mask]);m=l.max(-1,keepdims=True);p=np.exp(l-m);p/=p.sum(-1,keepdims=True)
 return p[:,mask].sum(-1),z
def gaps_loss(zA,zB,st,torch):
 p,n=st['positive_group'],st['source_group'];pb,nb=st['mapping'][p],st['mapping'][n]
 zz=[torch.tensor(x,device='cuda') for x in (zA,zB)]
 gaps=torch.stack([zz[0][st['groupsA'][p]].max()-zz[0][st['groupsA'][n]].max(),zz[1][st['groupsB'][pb]].max()-zz[1][st['groupsB'][nb]].max()])
 return gaps.cpu().numpy(),float(torch.nn.functional.softplus(-gaps/.1).mean())

def main():
 start=time.perf_counter();auth=read(AUTH);plan=read(PLAN)
 assert auth['state']=='ROOT_AUTHORIZED_R50_INDEPENDENT_DEV_EPISODIC_TINYHEAD_AUDIT_NO_GT'
 assert auth['program_sha256']==sha(__file__) and auth['audit_plan_sha256']==sha(PLAN)
 assert plan['n']==336 and plan['arms']==ARMS and plan['no_GT_full_grounder_VLM_detector_or_optimizer'] is True
 assert plan['resource_mode']=='GPU_TINY_HEAD_FORWARD_ONLY_NO_FIT'
 for rel,h in {**plan['programs_sha256'],**plan['inputs_sha256']}.items():assert sha(local(rel))==h
 cap,fit,closure=[local(plan[k]) for k in ('capture_root','fit_root','closure_root')]
 for key in ('capture_plan','fit_plan','closure_plan','existing_DEV_prediction_audit'):
  assert plan[key] in plan['inputs_sha256']
 for folder in (cap,fit,closure):
  key=(folder/'SEALED.json').relative_to(ROOT).as_posix();assert key in plan['inputs_sha256']
 assert auth['prediction_seal_sha256']==sha(fit/'SEALED.json')
 cplan=read(local(plan['capture_plan']));fplan=read(local(plan['fit_plan']));clplan=read(local(plan['closure_plan']))
 assert fplan['n']==336 and fplan['arm_names']==ARMS and fplan['steps']==[1,3] and fplan['lr']==.0005 and fplan['weight_decay']==0 and fplan['tau']==.1
 assert local(clplan['capture_root'])==cap and local(clplan['fit_root'])==fit and clplan['steps']==[1,3] and clplan['max_grounder_forwards']==8 and clplan['n_per_platform']==1
 assert clplan['B_sampling_salt']==cplan['B_sampling_salt']
 assert seal(cap)['n']==seal(fit)['n']==336 and seal(fit)['arm_names']==ARMS
 assert seal(fit)['plan_sha256']==sha(local(plan['fit_plan'])) and seal(cap)['plan_sha256']==sha(local(plan['capture_plan']))
 assert seal(closure).get('plan_sha256',sha(local(plan['closure_plan'])))==sha(local(plan['closure_plan']))
 assert seal(fit)['source_grounder_forwards_in_this_stage']==seal(fit)['new_visual_calls']==0
 source_plan=read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json')
 assert source_plan['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 assert seal(cap)['source_fingerprint']==source_plan['source_model_state_fingerprint'] and seal(cap)['source_unchanged'] is True
 assert read(local(plan['existing_DEV_prediction_audit']))['state'].startswith('PASS_')
 rows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');assert len(rows)==336 and len({r['id'] for r in rows})==336
 prep=ROOT/'crossplatform/round50/prepared336_attempt001';previous=ROOT/'crossplatform/round50/predicted336_attempt001'
 own=ROOT/'crossplatform/round49/ownership336_attempt001';stage=ROOT/'crossplatform/round49/stage336_attempt001'
 canonical=ROOT/'crossplatform/round24/capture336_attempt001'
 for folder in (prep,previous,own,stage,canonical):assert (folder/'SEALED.json').relative_to(ROOT).as_posix() in plan['inputs_sha256']
 AC={r['id']:r for r in read(ROOT/'research/round17/input/A_CASES_PRIVATE.json')};AF=read(ROOT/'research/round17/input/A_FRAME_INDEX_PRIVATE.json');RF=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json')
 for rel in ['crossplatform/round7/camera_projection.py','research/round17/input/A_CASES_PRIVATE.json','research/round17/input/A_FRAME_INDEX_PRIVATE.json','research/round12/data_audit/FRAME_INDEX_PRIVATE.json','research/round24/input/ALL336_PRIVATE.json','research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json']:
  assert rel in {**plan['programs_sha256'],**plan['inputs_sha256']}
 sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import CameraSpec,project_model_points
 import torch
 assert torch.__version__==plan['runtime_versions']['torch'] and np.__version__==plan['runtime_versions']['numpy']
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 assert 0<=plan['tiny_head_p_max_abs_bound']<=2e-5 and 0<=plan['tiny_head_z_max_abs_bound']<=2e-4
 assert 0<=plan['trace_scalar_abs_bound']<=2e-5
 source=load(member(cap,'SOURCE_MATCHING_MLP_PRIVATE.npz'));state_ok(source)
 projection=load(member(canonical,'projection_source_PRIVATE.npz'))
 for k in source:exact(source[k],projection['image.'+k],('original_source_MLP',k))
 caprecords=read(member(cap,'RECEIPTS_PRIVATE.json'));fitrecords=read(member(fit,'RECEIPTS_PRIVATE.json'))
 capby={x['id_PRIVATE']:x for x in caprecords};fitby={x['id_PRIVATE']:x for x in fitrecords}
 assert len(caprecords)==len(capby)==len(fitrecords)==len(fitby)==336 and set(capby)==set(fitby)=={r['id'] for r in rows}
 assert not OUT.exists();OUT.mkdir(parents=True);resultrows=[];stored={};counts=collections.Counter();active=None
 try:
  for i,row in enumerate(rows):
   sid=row['id'];active=f'E{i+1:04d}';v=load(member(cap,'samples_PRIVATE/'+sid+'.npz'));cr=read(member(cap,'cases_PRIVATE/'+sid+'.json'))
   z=load(member(fit,'samples_PRIVATE/'+sid+'.npz'));r=read(member(fit,'cases_PRIVATE/'+sid+'.json'));a=load(member(prep,'arrays_PRIVATE/'+sid+'.npz'));old=load(member(previous,'samples_PRIVATE/'+sid+'.npz'));original=load(member(canonical,'samples_PRIVATE/'+sid+'.npz'))
   assert cr==capby[sid] and r==fitby[sid]
   assert cr['id_PRIVATE']==r['id_PRIVATE']==sid and cr['platform']==r['platform']==row['platform']
   assert cr['source_checkpoint_sha256']==source_plan['source_checkpoint_sha256'] and cr['new_grounder_forwards']==2 and cr['no_tasklabels_or_annotated_class_span'] is True
   exact(v['A_boxes'],original['boxes'],'canonical_A_boxes');exact(v['A_scores'],original['scores'],'canonical_A_native')
   exact(v['A_boxes'],old['source_boxes'],'R50_DEV_source_boxes');exact(v['A_scores'],old['source_scores'],'R50_DEV_source_native')
   textsha=hashlib.sha256((row['caption']+' . not mentioned').encode()).hexdigest();assert textsha==cr['actual_model_text_sha256']
   assert cr['A_seed']==row['seed'];bseed=int(hashlib.sha256((cplan['B_sampling_salt']+'|'+sid+'|'+str(row['seed'])).encode()).hexdigest()[:16],16);assert cr['B_seed']==bseed
   for j,name in enumerate(('A','B')):
    receipt=cr['views'][j];assert receipt['view']==('A_original_seed' if name=='A' else 'B_resampled_same_coordinates')
    nraw=receipt['raw_point_count'];assert type(nraw) is int and nraw>0
    exact(v[name+'_raw_point_count'],np.asarray(nraw,np.int64),'captured_raw_point_count')
    indices=np.random.default_rng(row['seed'] if name=='A' else bseed).choice(nraw,16384,replace=nraw<16384).astype(np.int64)
    exact(indices,v[name+'_point_indices'],'sampling_RNG_choice_'+name)
    m=v[name+'_token_mask'];assert m.dtype==np.bool_ and m.shape==(len(v[name+'_tokens']),) and m.any() and (~m).any() and len(m)<=256
    assert v[name+'_h'].shape==(256,288) and v[name+'_tokens'].shape[1]==64
    for key in ('h','tokens','query','scores','boxes'):assert v[name+'_'+key].dtype==np.float32 and np.isfinite(v[name+'_'+key]).all()
    if name=='A':exact(m,original['token_mask'],'canonical_GTfree_proxy_mask')
    else:exact(m,v['A_token_mask'],'same_text_sampling_only_mask')
    callname=sid+'_'+receipt['view'];call=read(member(cap,'calls_PRIVATE/'+callname+'.json'))
    assert call['state']=='COMPLETED' and call['model_forward_entered'] is True and call['actual_text_sha256']==textsha
    assert call['source_checkpoint_sha256']==source_plan['source_checkpoint_sha256']
    assert call['raw_point_count']==nraw
    assert call['point_indices_sha256']==hashlib.sha256(indices.tobytes()).hexdigest()
    assert call['point_sample_sha256']==receipt['point_sample_sha256']==str(v[name+'_point_sha256'])
    rawpath=member(cap,'raw_views_PRIVATE/'+callname+'.npz');assert sha(rawpath)==call['raw_file_sha256'];raw=load(rawpath)
    for key in ('boxes','scores','h','query','tokens','token_mask','point_indices'):exact(raw[key],v[name+'_'+key],'durable_capture_raw_'+name+'_'+key)
    exact(raw['raw_point_count'],v[name+'_raw_point_count'],'durable_raw_point_count')
   assert str(v['A_point_sha256'])!=str(v['B_point_sha256'])
   pool=a['wide_pool'];e=old['image_e'];pr=old['raw_mask_PR'];FA,ca=old_read(v['A_scores'],a['wide_old_weights'],pool,e,pr)
   bi,validB,S,C,I=B_geometry(row,v['B_boxes'],pool,own,stage,AC,AF,RF,CameraSpec,project_model_points)
   for key,value in [('B_weights',bi),('B_legacy_support',S),('B_legacy_coverage',C),('B_projection_iou',I)]:exact(z[key],value,key)
   FB,cb=old_read(v['B_scores'],bi,pool,e,pr);exact(z['A_fixed_VPR'],ca,'A_fixed_G');exact(z['B_fixed_VPR'],cb,'B_fixed_G')
   idx=old['arm_names'].tolist().index('wideentry_oldW');exact(FA,old['scores'][idx],'literal_A_G_unchanged')
   st=structure(v,a['legacy_valid'],validB,FA,FB)
   for key in ('q0','qG','source_group','positive_group','strict_A_group_gap','pre_A_condition','accepted','reason','q0_group_has_B_mapping','mask_source_A','mask_positive_A','matching_diagnostics'):assert r[key]==st[key],('frozen_structure',key)
   for name in ('A','B'):
    assert r['groups'+name+'_PRIVATE']==st['groups'+name]
    exact(z[name+'_membership'],st['membership'+name],'group_membership');exact(z[name+'_reps'],st['reps'+name],'group_representatives')
   exact(z['geometry_repsA'],st['geometry_repsA'],'unchanged_source_geometry_representatives')
   for key in ('match_A','match_B'):exact(z[key],st[key],'Hungarian_correspondence')
   gpu=[];gpu_diagnostics=[];cpu_diagnostics=[];nonzero={};bytechanges={};logodds_changed={};replays={}
   for step in (0,1,3):
    state=source if step==0 or not st['accepted'] else load(member(fit,'states_PRIVATE/'+sid+'_s'+str(step)+'.npz'));state_ok(state)
    if step:
     actual=any(not np.array_equal(state[k],source[k]) for k in source)
     bytechange=any(state[k].tobytes()!=source[k].tobytes() for k in source)
     counts['saved_head_states_with_any_source_byte_difference']+=bytechange
     counts['nonzero_parameter_element_count']+=sum(int(np.count_nonzero(state[k]!=source[k])) for k in source)
     bytechanges[step]=bytechange
     assert r['nonzero_parameter_updates'][str(step)]==actual;nonzero[step]=actual
     if not st['accepted']:assert not (fit/'states_PRIVATE'/f'{sid}_s{step}.npz').exists()
    pp=[];zz=[]
    for name in ('A','B'):
     savedp=z[f'{name}_p_s{step}'];savedz=z[f'{name}_z_s{step}']
     if step and not bytechanges[step]:
      exact(savedp,z[f'{name}_p_s0'],'unchanged_head_p_equals_source');exact(savedz,z[f'{name}_z_s0'],'unchanged_head_z_equals_source')
     if step and not st['accepted']:
      exact(savedp,z[f'{name}_p_s0'],'rejected_p_copied');exact(savedz,z[f'{name}_z_s0'],'rejected_z_copied')
      gp,gz=gpu[0][0 if name=='A' else 1]
     else:
      gp,gz=head_forward(state,v[name+'_h'],v[name+'_tokens'],v[name+'_token_mask'],torch);counts['tiny_head_CUDA_forwards']+=1
      cp,cz=cpu64(state,v[name+'_h'],v[name+'_tokens'],v[name+'_token_mask'])
      cpu_diagnostics.append({'step':step,'view':name,'p_max_abs_error':float(np.max(abs(cp-savedp))),'z_max_abs_error':float(np.max(abs(cz-savedz))),'native_query_differs':int(cp.argmax())!=int(savedp.argmax())})
     ep=float(np.max(abs(gp-savedp)));ez=float(np.max(abs(gz-savedz)))
     assert ep<=plan['tiny_head_p_max_abs_bound'] and ez<=plan['tiny_head_z_max_abs_bound'] and int(gp.argmax())==int(savedp.argmax()),'tiny_head_GPU_replay_measurement_fail'
     gpu_diagnostics.append({'step':step,'view':name,'p_max_abs_error':ep,'z_max_abs_error':ez,
      'p_values_not_bitexact':int(np.count_nonzero(gp!=savedp)),'z_values_not_bitexact':int(np.count_nonzero(gz!=savedz)),
      'native_query_exact':True,'source_forward_reused_for_verified_rejected_state':bool(step and not st['accepted'])})
     counts['tiny_head_p_values_checked']+=256;counts['tiny_head_z_values_checked']+=256
     pp.append(gp);zz.append(gz)
    gpu.append([(pp[j],zz[j]) for j in (0,1)])
    actualscores,actualq,override=consensus(z[f'A_p_s{step}'],z[f'B_p_s{step}'],ca,cb,st)
    replayed,replayedq,_=consensus(pp[0],pp[1],ca,cb,st);assert actualq==replayedq,'GPU_replayed_final_consensus_query_changed'
    replays[step]=(actualscores,actualq,override)
    if step:logodds_changed[step]=any(not np.array_equal(z[f'{name}_z_s{step}'],z[f'{name}_z_s0']) for name in ('A','B'))
    if step and not bytechanges[step]:
     exact(actualscores,replays[0][0],'unchanged_head_final_score_equals_s0');assert actualq==replays[0][1]
    if step and st['accepted']:
     outputchanged=any(not np.array_equal(z[f'{name}_p_s{step}'],z[f'{name}_p_s0']) for name in ('A','B'));assert outputchanged==r['nonzero_native_output_changes'][str(step)]
     counts['nonzero_s'+str(step)]+=nonzero[step]
   p0A=z['A_p_s0'];p0B=z['B_p_s0']
   for j,name in enumerate(('A','B')):
    err=float(np.max(abs(z[name+'_p_s0']-v[name+'_scores'])));assert err<=fplan['initial_head_native_max_abs_bound'] and int(z[name+'_p_s0'].argmax())==int(v[name+'_scores'].argmax())
    assert err==r['initial_head_max_abs_errors'][j]
   assert int((p0A.astype(float)+2*ca[0]+2*ca[1]+2*ca[2]).argmax())==int(FA.argmax())
   _,capturedq,_=consensus(v['A_scores'],v['B_scores'],ca,cb,st);assert capturedq==replays[0][1]
   gs=group_scores(FA,st['groupsA']);groupq=int(st['repsA'][choose_group(gs,st['repsA'])])
   assert groupq==int(FA.argmax()),'fixed_G_group_representative_must_preserve_literal_G_query'
   queries=np.asarray([int(v['A_scores'].argmax()),int(FA.argmax()),groupq,*[replays[s][1] for s in (0,1,3)]])
   allscores=np.stack([v['A_scores'].astype(float),FA,gs[st['membershipA']],*[replays[s][0] for s in (0,1,3)]])
   exact(z['scores'],allscores,'six_native_and_consensus_score_vectors');exact(z['queries'],queries,'six_final_queries');exact(z['selected_original_boxes'],v['A_boxes'][queries],'all_six_canonical_A_boxes')
   exact(z['source_boxes'],v['A_boxes'],'fixed_source_pool');exact(z['source_scores'],v['A_scores'],'fixed_source_scores')
   exact(z['query_override'],np.asarray([False,False,True,*[replays[s][2] for s in (0,1,3)]]),'query_override_contract');assert z['arm_names'].tolist()==ARMS
   assert queries[0]==old['queries'][0] and queries[1]==old['queries'][idx]
   assert r['steps']==(3 if st['accepted'] else 0) and len(r['trace'])==r['steps']
   if st['accepted']:
    assert [t['step'] for t in r['trace']]==[1,2,3]
    for trace in r['trace']:
     assert trace['step'] in (1,2,3) and np.isfinite(trace['gradient_L2']) and trace['gradient_L2']>=0 and trace['gradient_nonzero']==(trace['gradient_L2']>0)
     assert np.isfinite(trace['loss_before']) and all(np.isfinite(trace['gaps_before']))
    for step in (1,3):
     tr=r['trace'][step-1];gg,ll=gaps_loss(z[f'A_z_s{step}'],z[f'B_z_s{step}'],st,torch)
     assert abs(tr['loss_post']-ll)<=plan['trace_scalar_abs_bound'] and np.max(abs(np.asarray(tr['gaps_post'])-gg))<=plan['trace_scalar_abs_bound']
     state=load(member(fit,'states_PRIVATE/'+sid+'_s'+str(step)+'.npz'));d64=float(np.sqrt(sum(np.square(state[k].astype(np.float64)-source[k].astype(np.float64)).sum() for k in state)))
     assert np.isfinite(tr['parameter_delta_L2']) and (tr['parameter_delta_L2']>0)==nonzero[step]
     assert abs(tr['parameter_delta_L2']-d64)<=plan['parameter_norm_abs_bound']+plan['parameter_norm_relative_bound']*d64
    for pre,at in ((1,0),(2,1)):
     gg,ll=gaps_loss(z[f'A_z_s{at}'],z[f'B_z_s{at}'],st,torch);tr=r['trace'][pre-1]
     assert abs(tr['loss_before']-ll)<=plan['trace_scalar_abs_bound'] and np.max(abs(np.asarray(tr['gaps_before'])-gg))<=plan['trace_scalar_abs_bound']
   counts['accepted']+=st['accepted'];counts['optimizer_steps']+=r['steps'];counts['six_arm_score_values_bitexact']+=6*256;counts['six_selected_boxes_bitexact']+=6
   stored[sid]={'row':row,'case':r,'structure':st,'nonzero':nonzero}
   resultrows.append({'row':active,'platform':row['platform'],'accepted':st['accepted'],'reason':st['reason'],
    'nonzero_s1':nonzero[1],'nonzero_s3':nonzero[3],'all_six_scores_queries_original_boxes_exact':True,
    'nonzero_exactlogodds_s1':logodds_changed[1],'nonzero_exactlogodds_s3':logodds_changed[3],
    'sampling_seed_indices_independently_replayed':True,'CPU64_mathematical_forward_diagnostics':cpu_diagnostics,
    'GPU_native_forward_replay_diagnostics':gpu_diagnostics})
   if (i+1)%16==0:print('INDEPENDENT_EPISODIC50',i+1,flush=True)
  assert seal(cap)['grounder_forwards']==672 and len(list((cap/'calls_PRIVATE').glob('*.json')))==672
  assert seal(fit)['actual_optimizer_steps']==counts['optimizer_steps']==3*counts['accepted']
  selected=read(member(closure,'SELECTED_BEFORE_FORWARD_PRIVATE.json'));expected=[];missing=[]
  for platform in ('drone','quad'):
   eligible=[x for x in stored.values() if x['row']['platform']==platform and x['structure']['accepted'] and x['nonzero'][1] and x['nonzero'][3]]
   if eligible:expected.append(min(eligible,key=lambda x:hashlib.sha256((clplan['selection_salt']+x['row']['id']).encode()).hexdigest())['row']['id'])
   else:missing.append(platform)
   assert selected['eligible_counts'][platform]=={'accepted':sum(x['row']['platform']==platform and x['structure']['accepted'] for x in stored.values()),'nonzero_s1_and_s3':len(eligible)}
  assert [x['id_PRIVATE'] for x in selected['rows']]==expected and selected['no_nonzero_accepted_episode_platforms']==missing
  assert seal(closure)['n']==len(expected)
  closure_forward_checks=0
  if expected:
   result=read(member(closure,'RESULT_PRIVATE.json'));assert result['outside_projection_unchanged'] is True and result['source_fully_restored'] is True
   assert result['new_grounder_forwards']==seal(closure)['grounder_forwards']==4*len(expected)<=8
   for sid in expected:
    full=load(member(closure,sid+'_FULL_CLOSURE_PRIVATE.npz'));v=load(member(cap,'samples_PRIVATE/'+sid+'.npz'));z=load(member(fit,'samples_PRIVATE/'+sid+'.npz'));st=stored[sid]['structure']
    for step in (1,3):
     for name in ('A','B'):
      exact(full[f'{name}_s{step}_all256_boxes'],v[name+'_boxes'],'installed_full_frozen_boxes')
      p=full[f'{name}_s{step}_native_p'];cached=z[f'{name}_p_s{step}'];assert np.max(abs(p-cached))<=clplan['cached_full_max_abs_bound'] and int(p.argmax())==int(cached.argmax());closure_forward_checks+=1
     scores,q,_=consensus(full[f'A_s{step}_native_p'],full[f'B_s{step}_native_p'],z['A_fixed_VPR'],z['B_fixed_VPR'],st)
     index=ARMS.index('sameinfo_consensus'+str(step));assert q==z['queries'][index];exact(v['A_boxes'][q],z['selected_original_boxes'][index],'installed_final_box');exact(scores,full[f's{step}_final_consensus_scores'],'installed_final_score_replay')
  else:assert seal(closure)['state']=='R50_EPISODIC_CLOSURE_NOT_AVAILABLE_NO_NONZERO_ACCEPTED_STATES' and seal(closure)['grounder_forwards']==0
  verified_no_update=counts['saved_head_states_with_any_source_byte_difference']==0
  no_update_permitted=bool(verified_no_update and plan.get('allow_verified_no_parameter_update_branch') is True)
  assert bool(expected) or no_update_permitted, 'No adapted closure: a root-locked verified-no-update exception is required before scoring'
  closure_status='ACTUAL_INSTALLED_ADAPTED_HEADS_VERIFIED' if expected else 'NOT_APPLICABLE_VERIFIED_NO_PARAMETER_UPDATE'
 except BaseException as exc:
  dump(OUT/'PARTIAL_ROWS.json',resultrows);dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'active_anonymous_row':active,'completed_n':len(resultrows),'no_GT_optimizer_or_automatic_retry':True});raise
 dump(OUT/'ANONYMOUS_NUMERIC_AND_UPDATE_ROWS.json',resultrows)
 result={'state':'PASS_R50_INDEPENDENT_EPISODIC_NUMERIC_AND_CONTRACT_AUDIT_NO_GT','n':336,'arms':6,'no_GT_read':True,
  'prediction_seal_sha256':sha(fit/'SEALED.json'),'capture_seal_sha256':sha(cap/'SEALED.json'),'closure_seal_sha256':sha(closure/'SEALED.json'),
  'counts':dict(counts),'installed_closure_requirement_satisfied':bool(expected),
  'installed_closure_state':closure_status,'actual_adapted_closure_verified':bool(expected),
  'nonzero_parameter_element_count':counts['nonzero_parameter_element_count'],
  'all_saved_heads_source_bytes_identical':verified_no_update,
  'all_no_update_outputs_equal_s0':verified_no_update,
  'verified_all_saved_heads_source_bytes_equal':verified_no_update,
  'verified_no_parameter_update_branch_explicitly_root_permitted':no_update_permitted,
  'TTA_effect_actually_tested':counts['nonzero_parameter_element_count']>0,
  'no_update_branch_must_stop_and_cannot_advance_to_full_TTA':no_update_permitted,
  'actual_adapted_closure_episodes':len(expected),'actual_full_grounder_closure_forward_checks':closure_forward_checks,
  'platforms_without_nonzero_closure_candidates':missing,'no_nonzero_state_not_falsely_called_adapted_closure':True,
  'new_full_grounder_VLM_detector_or_optimizer_calls_by_auditor':0,'only_184960_parameter_head_forward_replayed':True,
  'no_author_R50_algorithm_as_reference':True,'independent_groups_Hungarian_B_geometry_oldG_and_final_readout':True,
  'CPU64_diagnostic_queries_never_replace_actual_GPU_winners':True,
  'six_score_vectors_exact_is_readout_arithmetic_from_recorded_p_not_a_claim_of_byteexact_CUDA_head_outputs':True,
  'GPU_head_p_max_abs_bound':plan['tiny_head_p_max_abs_bound'],'GPU_head_z_max_abs_bound':plan['tiny_head_z_max_abs_bound'],
  'raw_point_count_from_capture_receipt_not_rawXYZ_reread':True,
  'gradient_trace_finite_nonzero_flags_checked_but_raw_gradient_arrays_not_independently_recomputed':True,
  'step3_prestate_absent_so_its_preupdate_gradient_and_loss_not_independently_recomputed':True,
  'all_nonmatching_full_model_unchanged_assertions_hash_bound_not_full_model_reloaded_by_auditor':True,
  'sealed_visible_point_coordinate_cache_and_original_camera_registry_used_to_recompute_B_geometry':True,
  'no_protected_FINAL_specific_artifacts_read':True,'program_sha256':sha(__file__),'plan_sha256':sha(PLAN),'seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result);dump(OUT/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}});print(json.dumps(result))

if __name__=='__main__':main()
