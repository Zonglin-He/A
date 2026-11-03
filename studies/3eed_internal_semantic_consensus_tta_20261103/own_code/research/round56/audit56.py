"""Independent all-row fixed-consensus, correspondence and fitted-head audit; no GT."""
import argparse,sys,time,torch,hashlib
from scipy.special import softmax,logsumexp
from common56 import *

def score(hidden,w,tokens,mask):
 x=hidden
 for i in [0,2]:x=torch.relu(torch.nn.functional.linear(x,w[str(i)+'.weight'],w[str(i)+'.bias']))
 x=torch.nn.functional.normalize(torch.nn.functional.linear(x,w['4.weight'],w['4.bias']),dim=-1);a=(x@tokens.transpose(-1,-2)/.07)[0]
 prob=(a.softmax(-1)*mask[None]).sum(-1)
 z=torch.logsumexp(a[:,mask],-1)-torch.logsumexp(a[:,~mask],-1)
 return prob,z

def transfer(A,z,base):
 W=np.asarray(A,float)/(1+np.asarray(A,float).sum(0));mapped=W@softmax(np.asarray(z,float));u=max(0.,1-float(mapped.sum()));p=mapped+u*base;return p/p.sum(),u

def raw_affinity(boxes,other,xyz):
 a=(np.abs(xyz[None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(2)&(boxes[:,3:]>0).all(1)[:,None];b=(np.abs(xyz[None]-other[:,None,:3])<=other[:,None,3:]/2).all(2)&(other[:,3:]>0).all(1)[:,None];intersection=a.astype(float)@b.astype(float).T;union=a.sum(1)[:,None]+b.sum(1)[None]-intersection;jac=intersection.astype(np.float32)/np.maximum(union.astype(np.float32),1)
 inter=np.maximum(np.minimum(boxes[:,None,:3]+boxes[:,None,3:]/2,other[None,:,:3]+other[None,:,3:]/2)-np.maximum(boxes[:,None,:3]-boxes[:,None,3:]/2,other[None,:,:3]-other[None,:,3:]/2),0).prod(2);v=np.maximum(boxes[:,3:],0).prod(1)[:,None]+np.maximum(other[:,3:],0).prod(1)[None,:]-inter;iou=np.divide(inter,v,out=np.zeros_like(inter),where=v>0);value=np.sqrt(np.maximum(iou*jac,0));value[intersection<4]=0;return value.astype(np.float32)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);start=time.perf_counter();out=ROOT/p['output_root'];assert not out.exists();out.mkdir(parents=True);rows=read(ROOT/p['manifest']);pr=ROOT/p['prediction_root'];ps=read(pr/'SEALED.json');cs=read(ROOT/p['consensus_root']/'SEALED.json');cap=ROOT/p['capture_root'];capseal=read(cap/'SEALED.json');ref=ROOT/p['reference_root'];refseal=read(ref/'SEALED.json');lat=ROOT/p['latent_root'];ls=read(lat/'SEALED.json')
 torch.set_num_threads(4);source=read(ROOT/p['source_plan']);state=torch.load(source['source_checkpoint_path'],map_location='cpu',weights_only=True)['model'];assert sha(source['source_checkpoint_path'])==source['source_checkpoint_sha256'];pref='contrastive_align_projection_image.';weights={k[len(pref):]:v for k,v in state.items() if k.startswith(pref)};del state
 selected=set()
 for pl in ['drone','quad']:selected.update(r['id'] for r in sorted([r for r in rows if r['platform']==pl],key=lambda r:hashlib.sha256(('R56_RAW_MAPPING_AUDIT:'+r['id']).encode()).digest())[:2])
 maxerr=0.;maxteacher=0.;rawmaps=0;fits=0;gradient_checks=0;graderr=0.;nonzero={k:0 for k in ['tta1','tta3','shuffle_tta1','shuffle_tta3']}
 for row in rows:
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(pr/rel)==ps['files'][rel] and sha(ROOT/p['consensus_root']/rel)==cs['files'][rel] and sha(cap/rel)==capseal['files'][rel] and sha(lat/rel)==ls['files'][rel];pred=load(pr/rel);c=load(ROOT/p['consensus_root']/rel);s=load(cap/rel);l=load(lat/rel,['h','tokens','token_mask','boxes','scores','point_indices','input_sample_xyz']);f=load(pr/'fitted_PRIVATE'/(sid+'.npz'));rec=read(pr/'cases_PRIVATE'/(sid+'.json'));assert sha(pr/'fitted_PRIVATE'/(sid+'.npz'))==rec['fitted_sha256']
  assert np.array_equal(pred['source_boxes'],l['boxes']) and np.array_equal(pred['selected_original_boxes'],l['boxes'][pred['queries']]) and np.array_equal(pred['queries'],pred['scores'].argmax(1));assert np.array_equal(pred['scores'][0],l['scores']) and np.array_equal(pred['scores'][1],l['scores'])
  raw=softmax(s['stage_role_z'][:6,0].astype(float),axis=-1).mean(0);adjusted=s['stage_role_z'].astype(float)-np.log(s['role_masks'].sum(1))[None,:,None]
  rr=adjusted[:,0]-(logsumexp(adjusted[:,1:],axis=1)-np.log(len(adjusted[0])-1)) if len(adjusted[0])>1 else s['stage_role_z'][:,0].astype(float);role=softmax(rr[:6],axis=-1).mean(0);base=softmax(s['stage_role_z'][-1,0].astype(float));A=c['association_affinity'];sub,u=transfer(A[0],s['subject_phrase_z'],base);mappedrefs=[];meta=read(ref/'cases_PRIVATE'/(sid+'.json'));byname={k:v for k,v in meta['references'].items()};originalboxes=[s['subject_phrase_boxes']]
  for i,entity in enumerate(c['association_entities'][1:],start=1):
   info=byname[str(entity)];path=ref/info['file'];assert sha(path)==info['sha256']==refseal['files'][info['file']];v=load(path,['boxes','logodds','point_indices']);assert np.array_equal(v['point_indices'],l['point_indices']);mp,ru=transfer(A[i],v['logodds'],np.full(256,1/256));assert abs(ru-c['reference_phrase_unknown_masses'][i-1])<1e-12;mappedrefs.append(mp);originalboxes.append(v['boxes'])
  refmean=np.mean(mappedrefs,axis=0) if mappedrefs else np.full(256,1/256);phrase=softmax(np.log(sub.clip(1e-300))-np.log(refmean.clip(1e-300)));teacher=(raw+role+phrase)/3;teacher/=teacher.sum();maxteacher=max(maxteacher,float(abs(teacher-c['teacher']).max()));assert np.allclose(teacher,c['teacher'],atol=1e-14,rtol=0) and abs(u-float(c['subject_phrase_unknown_mass']))<1e-12
  perm=np.random.default_rng(int.from_bytes(hashlib.sha256(('R56_candidate_alignment_control:'+sid).encode()).digest()[:8],'little')).permutation(256);assert np.array_equal(perm,c['candidate_permutation']) and np.array_equal(c['shuffled_teacher'],c['teacher'][perm])
  static={'source':l['scores'],'zero_step':l['scores'],'proposal':s['stage_target_probability'][0],**{'decoder'+str(i):s['stage_target_probability'][i+1] for i in range(5)},'early_consensus':raw,'final_role':softmax(rr[-1]),'early_role_consensus':role,'phrase_subject':sub,'phrase_role':phrase,'semantic_consensus':teacher,'shuffled_consensus':teacher[perm]}
  for arm,values in static.items():
   err=float(abs(pred['scores'][ARMS.index(arm)]-values).max());maxerr=max(maxerr,err);assert err<1e-12 and int(pred['queries'][ARMS.index(arm)])==int(values.argmax())
  if sid in selected:
   _,ix=np.unique(l['point_indices'],return_index=True);xyz=l['input_sample_xyz'][np.sort(ix)].astype(float)
   for i,bx in enumerate(originalboxes):rebuilt=raw_affinity(l['boxes'].astype(float),bx.astype(float),xyz);assert np.array_equal(A[i],rebuilt);rawmaps+=1
  hidden=torch.from_numpy(l['h'])[None];tokens=torch.from_numpy(l['tokens'])[None];mask=torch.from_numpy(l['token_mask']);p0,z0=score(hidden,weights,tokens,mask);assert int(p0.argmax())==int(l['scores'].argmax())
  for arm in ['tta1','tta3','shuffle_tta1','shuffle_tta3']:
   w={k:torch.from_numpy(f[arm+'__'+k]) for k in weights};prob,z=score(hidden,w,tokens,mask);expected=pred['scores'][ARMS.index(arm)];err=float(abs(prob.numpy()-expected).max());maxerr=max(maxerr,err);assert err<1e-5 and int(prob.argmax())==int(expected.argmax());fits+=1;delta=float(torch.sqrt(sum((w[k]-weights[k]).square().sum() for k in weights)));nonzero[arm]+=delta>0;tr=rec['statistics']['shuffle_TTA' if arm.startswith('shuffle') else 'TTA']['trace'][0 if arm.endswith('1') else 2];assert abs(delta-tr['parameter_delta_L2'])<1e-4
   target=c['shuffled_teacher'] if arm.startswith('shuffle') else c['teacher'];kl=float((torch.from_numpy(target)*(torch.from_numpy(target).clamp_min(1e-300).log()-z.double().log_softmax(-1))).sum());assert abs(kl-tr['KL_after'])<1e-5
  if sid in selected:
   for prefix,key in [('', 'teacher'),('shuffle_','shuffled_teacher')]:
    w={k:v.clone().requires_grad_() for k,v in weights.items()};_,z=score(hidden,w,tokens,mask);t=torch.from_numpy(c[key]);L=(t*(t.clamp_min(1e-300).log()-z.double().log_softmax(-1))).sum();L.backward();gn=float(torch.sqrt(sum(v.grad.square().sum() for v in w.values())));expected=rec['statistics'][prefix+'TTA']['trace'][0]['gradient_norm'];graderr=max(graderr,abs(gn-expected));assert abs(gn-expected)<1e-4;gradient_checks+=1
 result={'state':'PASS_R56_ALL_FIXED_CONSENSUS_ORIGINAL_BOX_AND_FITTED_MATCHING_AUDIT','n':336,'arm_names':ARMS,'prediction_seal_sha256':sha(pr/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'fitted_states_checked':fits,'nonzero_updates_by_arm':nonzero,'all_correspondence_and_consensus_recomputed':True,'hash_selected_raw_association_cases':4,'raw_association_matrices_exact':rawmaps,'max_teacher_error':maxteacher,'max_score_error':maxerr,'independent_Top1_disagreements':0,'independent_first_gradient_checks':gradient_checks,'max_gradient_norm_error':graderr,'GT_reads':0,'seconds':time.perf_counter()-start,'program_sha256':sha(__file__),'plan_sha256':sha(a.plan)};dump(out/'AUDIT_RESULT.json',result);print(result,flush=True)
if __name__=='__main__':main()
