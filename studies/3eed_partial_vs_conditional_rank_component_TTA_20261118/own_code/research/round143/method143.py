"""Existing-loss component ablations; same R142 input/reader/Up4896 budget."""
from common143 import *
sys.path.insert(0,str(R/'research/round141'));from core141 import *
from common143 import R,H,B,rd,sha,dump,check,utc
import argparse,shutil
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
ARMS=['partial_only','partial_only_wrong_C','rank_only','rank_only_wrong']

def main():
 check();assert shutil.disk_usage(R).free>=20*1024**3;reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));root=B/('engineering' if args.engineering else 'formal');root.mkdir(exist_ok=False);n=0;fits=0;maxdecomp=0.
 if not args.engineering:assert rd(H/'ENGINEERING_AUDIT.json')['status']=='PASS'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S0=model97(base,src);fp=fingerprint(base)
  for e in [e for e in reg if e['source']==src]:
   tick=time.perf_counter();tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   old=np.load(R/'crossplatform/round142/method'/name/'samples_PRIVATE'/(sid+'.npz'));scores=dict(zip(old['arm_names'],old['scores']));means=np.load(R/'crossplatform/round142/supplementary_mean'/name/(sid+'.npz'));scores.update(dict(zip(means['arm_names'],means['scores'])));ev=load(B/'evidence_PRIVATE'/name/(sid+'.npz'));ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];scores.update(direct_partial=ev['direct_partial'],direct_wrong_partial=ev['direct_wrong_partial'],direct_rank=ev['direct_rank'],direct_wrong_rank=ev['direct_wrong_rank']);eligible=bool(ev['eligible']);ff={};ss={};records={};p0=scores['F_RGBB_only'];z0=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'))
   if eligible:
    gA,_=routegeom135(e,z0);gB,_=geo141.additional(e['A_sensor'],e['pair_PRIVATE']['B'],z0,rd(R/e['registration_path']));g=merge_graphs141([gA,gB]);d=tensor_data(g,e['row']);route=tensor_route89(load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz')));f=load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));A=torch.as_tensor(z0['vis0'][0],device='cuda');P=torch.where(torch.as_tensor(f['valid_B'],device='cuda')[:,None],(A+torch.as_tensor(f['B_vis'],device='cuda'))/2,A);z=dict(tensor_prefix(z0),vis0=P[None]);image=torch.as_tensor(gB['image'].copy(),device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255;Sg=torch.as_tensor(ev['S'],device='cuda')
   for arm in ARMS:
    ispartial=arm.startswith('partial');active=bool(ev['partial_active'] if ispartial else ev['rank_active']);identity=arm=='partial_only_wrong_C' and not bool(ev['C_wrong_effective']) or arm=='rank_only_wrong' and not bool(ev['effective_wrong'])
    if not active or identity:
     ctrl='partial_only' if arm=='partial_only_wrong_C' else 'rank_only' if arm=='rank_only_wrong' else None;scores[arm]=scores[ctrl].copy() if identity and active else p0.copy();records[arm]={'reused':True,'reason':'identity_wrong' if identity and active else 'algebraic_zero_or_noB'};continue
    Cg=torch.as_tensor(ev['C_wrong_membership'] if arm=='partial_only_wrong_C' else ev['C'],device='cuda');target=torch.as_tensor(ev['wrong_target'] if arm=='rank_only_wrong' else ev['target'],device='cuda')
    def task(lp):
     pt=taskloss(lp,groups,Sg,Cg,None,'partial') if ispartial else lp.sum()*0;gm=torch.stack([lp[torch.as_tensor(q,device='cuda')].max() for q in groups]);rank=(target*(target.clamp_min(1e-300).log()-gm[Cg].log_softmax(0))).sum() if not ispartial else lp.sum()*0;return pt+rank,{'partial':pt,'conditional_B_image_KL':rank}
    with rgb_banks141(m,[image]):zero,pred,feat,st,rec=fit136(base,m,S0,z,d,route,task,args.engineering)
    assert abs(zero-p0).max()<1e-7;scores[arm]=pred;records[arm]=rec|{'component':'partial' if ispartial else 'rank','source_Up_and_full_joint_reader_retained':True};fits+=1
    for k,v in feat.items():ff[arm+'_'+k]=v
    for k,v in st.items():ss[arm+'_'+k]=v
   decomp=None
   if eligible:
    a=ss.get('partial_only_gradient1',np.zeros(4896,np.float32));b=ss.get('rank_only_gradient1',np.zeros(4896,np.float32));parent=np.load(R/'crossplatform/round142/method'/name/'states_PRIVATE'/(sid+'.npz'))['image_B_TTA_gradient1'];na,nb=np.linalg.norm(a.astype(float)),np.linalg.norm(b.astype(float));err=float(np.linalg.norm((a.astype(float)+b.astype(float)-parent.astype(float)))/max(na+nb,1e-12));assert err<.001,(name,err);maxdecomp=max(maxdecomp,err);decomp={'normalized_sum_residual':err,'partial_gradient_norm':float(na),'rank_gradient_norm':float(nb),'cosine':float(a.astype(float)@b.astype(float)/(na*nb)) if na and nb else None}
   restore(m,S0);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S0.items());arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=z0['boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**ss);dump(D/'records_PRIVATE'/(sid+'.json'),{'eligible':eligible,'fits':records,'gradient_decomposition':decomp,'GT_read':False,'seconds':time.perf_counter()-tick,'source_only_initialization':True});n+=1
   if args.engineering or n%32==0:print('COMPONENT143',args.engineering,src,n,fits,flush=True)
   if n%32==0:assert shutil.disk_usage(R).free>=12*1024**3
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'new_component_fits':fits,'max_first_gradient_sum_relative_error':maxdecomp,'old_R142_predictions_reused_exactly':True,'GT_read':False,'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('COMPONENT143_COMPLETE',n,fits,flush=True)
if __name__=='__main__':main()
