"""Only registered cases: native attention, local loss maps and diagnostic head swaps."""
from core84 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H84/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H84/'experiment/CODE_LOCK.json'));base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);restore(m,S);O=B84/'capture33_attempt001';O.mkdir(parents=True);(O/'samples_PRIVATE').mkdir();records=[];start=time.perf_counter();last={};modelcalls=0
def attnhook(_,args,out):last['seed']=out[1].detach().clone()
def localhook(module,args,out):
 a=local_attention84(module,args);last['rgb']=a.detach().clone();valid=(args[2]>=0)&(args[3]>0);expected=module.o((a[:,:,None]*module.v(args[1])[args[2].clamp_min(0)]).sum(1))*valid.any(1)[:,None];assert float(abs(expected-out).max())<1e-6
hs=[base.decoder[5].cross_v.register_forward_hook(attnhook),m.image_to_point.register_forward_hook(localhook)]
try:
 for ci,c in enumerate(read(H84/'experiment/CASES33_PRIVATE.json')):
  r=c['row'];sid=r['id'];z0,z,d=item80(r);old=load(B82/'prediction2640_attempt001/samples_PRIVATE'/(sid+'.npz'));refs=dict(zip(old['arm_names'],old['scores']));Yroot=ROOT/('research/round55/score_dev336_attempt001' if sid in DEV_IDS else 'research/round69/score_all2304_attempt001');yy=next(x['candidate_IoUs_PRIVATE'] for x in read(Yroot/'ALL_CANDIDATE_IOUS_PRIVATE.json') if x['id_PRIVATE']==sid);yy=np.array(yy);chosen={k:int(refs[v].argmax()) for k,v in REF84.items()};chosen['oracle']=int(yy.argmax());keep=np.array(sorted(set(chosen.values())));qrank={};pair=[];arrays={'boxes':z0['boxes'],'xyz':z0['xyz'][0],'selected_query_indices':keep,'candidate_ious':yy,'uv_grid':d['uv_grid'].cpu().numpy() if 'uv_grid' in d else np.array([])}
  # Projection coordinates are saved separately in existing graph metadata.
  g=load(casepath(r) if sid in DEV_IDS else ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(sid+'.npz'));arrays.update(uv_grid=g['uv_grid'],visible=g['visible'],cell=g['cell'],depth_valid=g['depth_valid'],p2i=g['p2i'],point_indices=z0['point_indices'])
  with torch.no_grad():
   h0,u0=original84(base,z);modelcalls+=1;arrays['source_attention']=last['seed'][0,keep].cpu().numpy();orig=score84(base.contrastive_align_projection_image,h0,u0,z['token_mask'].bool()).cpu().numpy();bridge=int(orig.argmax())==chosen['source'];assert bridge;assert 'h' not in z0 or float(abs(h0.cpu().numpy()-z0['h']).max())<1e-6
   combos={'original_features_original_head':orig,'original_features_prepared_head':score84(m.match,h0,u0,z['token_mask'].bool()).cpu().numpy()};pm,im=maskpair79(r,d);hp,hi,hw=holdout81(r,d);views=list(zip(pm,im))+list(zip(hp,hi));ws=uniform_weights80(d,pm,im)+hw
   for state in STATES84:
    if state.startswith('raw'):put_patch(m,rawpatch82(r,state=='raw_wrong'))
    elif state.startswith('BT'):put_patch(m,load(B82/'prediction2640_attempt001/states_PRIVATE'/(sid+('_bt_wrong.npz' if state=='BT_wrong' else '_bt_correct.npz'))))
    pp,q,e=m.main(base,z,d,'bidirectional');modelcalls+=1;assert np.max(abs(pp.cpu().numpy()-refs[REF84[state]]))<1e-10;arrays[state+'_seed_attention']=last['seed'][0,keep].cpu().numpy();A=last['rgb'];idx=d['p2i'];rgbmass=[]
    for mass in last['seed'][0,keep]:
     v=torch.zeros(1536,device='cuda');v.scatter_add_(0,idx.clamp_min(0).flatten(),(mass[:,None]*A*(idx>=0)).flatten());rgbmass.append(v.cpu().numpy())
    arrays[state+'_RGB_path_mass']=np.array(rgbmass);arrays[state+'_scores']=pp.cpu().numpy()
    if state=='frozen':
     combos['prepared_features_original_head']=score84(base.contrastive_align_projection_image,e['query_hidden'],e['projected_text'],z['token_mask'].bool(),e['late_RGB_delta']).cpu().numpy();combos['prepared_features_prepared_head']=pp.cpu().numpy()
    losses=[];components=[]
    for a,b,w in zip([v[0] for v in views],[v[1] for v in views],ws):
     lp,li,ld=m.local_losses(z,d,a,b,False);parts=[.1*w['P']*lp,.1*w['I']*li,.01*w['D']*ld];losses.append(float(sum(x.sum() for x in parts)));components.append(parts)
    arrays[state+'_P_maps']=torch.stack([v[0] for v in components]).cpu().numpy();arrays[state+'_RGB_maps']=torch.stack([v[1] for v in components]).cpu().numpy();arrays[state+'_D_maps']=torch.stack([v[2] for v in components]).cpu().numpy();arrays[state+'_losses']=np.array(losses);restore(m,S)
  for state in ['source']+STATES84:
   scores=refs[REF84[state]];order=np.argsort(-scores,kind='stable');rank=np.empty(256,int);rank[order]=np.arange(1,257);qrank[state]=[{'query_PRIVATE':int(q),'rank':int(rank[q]),'probability':float(scores[q]/scores.sum()),'IoU':float(yy[q])} for q in keep]
  for a in keep:
   for b in keep:
    pair.append({'query_a_PRIVATE':int(a),'query_b_PRIVATE':int(b),'axis_aligned_box_IoU':box_iou84(z0['boxes'][a],z0['boxes'][b]),'center_distance':float(np.linalg.norm(z0['boxes'][a,:3]-z0['boxes'][b,:3]))})
  hybrid={k:{'query_PRIVATE':int(v.argmax()),'IoU':float(yy[v.argmax()])} for k,v in combos.items()};rec={'case':c['case'],'chosen_PRIVATE':chosen,'ranks_PRIVATE':qrank,'pairs_PRIVATE':pair,'hybrid_PRIVATE':hybrid,'source_bridge':bridge,'new_parameter_updates':0,'same_augmentation_not_independent_view':True};dump(O/(c['case']+'_PRIVATE.json'),rec);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),**arrays);records.append({'case':c['case'],'source_original_head_IoU':hybrid['original_features_original_head']['IoU'],'original_features_prepared_head_IoU':hybrid['original_features_prepared_head']['IoU'],'prepared_features_original_head_IoU':hybrid['prepared_features_original_head']['IoU'],'prepared_features_prepared_head_IoU':hybrid['prepared_features_prepared_head']['IoU']});print('CAPTURE84',ci+1,c['case'],flush=True)
finally:
 for h in hs:h.remove()
assert fingerprint(base)==fp and all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items());dump(H84/'HEAD_FUSION_2X2_ROWS.json',records);dump(O/'COUNTS.json',{'n':33,'grounding_suffixes':modelcalls,'new_parameter_updates':0,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())});seal(O,'R84_READ_ONLY_SELECTED_CASE_ATTENTION_MAPS_HYBRID_DIAGNOSTICS',n=33);print(read(O/'COUNTS.json'))
