"""Initial semantic consensus and cross-description transport, no task labels."""
import argparse,time,hashlib
from scipy.special import softmax,logsumexp
from common56 import *

def affinity(final,other,xyz):
 valid0=(final[:,3:]>0).all(1);valid1=(other[:,3:]>0).all(1);a=((np.abs(xyz[None]-final[:,None,:3])<=final[:,None,3:]/2).all(-1)&valid0[:,None]).astype(np.float32);b=((np.abs(xyz[None]-other[:,None,:3])<=other[:,None,3:]/2).all(-1)&valid1[:,None]).astype(np.float32)
 shared=a@b.T;union=a.sum(1)[:,None]+b.sum(1)[None]-shared;jac=np.divide(shared,union,out=np.zeros_like(shared),where=union>0)
 lo=np.maximum(final[:,None,:3]-final[:,None,3:]/2,other[None,:,:3]-other[None,:,3:]/2);hi=np.minimum(final[:,None,:3]+final[:,None,3:]/2,other[None,:,:3]+other[None,:,3:]/2);inter=np.maximum(hi-lo,0).prod(-1);vol0=np.maximum(final[:,3:],0).prod(1);vol1=np.maximum(other[:,3:],0).prod(1);vol=vol0[:,None]+vol1[None]-inter;iou=np.divide(inter,vol,out=np.zeros_like(inter),where=vol>0)
 A=np.sqrt(np.maximum(jac*iou,0));A[shared<4]=0;return A.astype(np.float32)

def transfer(A,z,base):
 W=A.astype(float)/(1+A.astype(float).sum(0,keepdims=True));p=softmax(np.asarray(z,float));mapped=W@p;unknown=max(0.,1-float(mapped.sum()));out=mapped+unknown*base;out/=out.sum();return out,unknown

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args();p=read(args.plan);pin(p);out=ROOT/p['output_root'];assert not out.exists();out.mkdir(parents=True);rows=read(ROOT/p['manifest']);cap=ROOT/p['capture_root'];cs=read(cap/'SEALED.json');lat=ROOT/p['latent_root'];ls=read(lat/'SEALED.json');ref=ROOT/p['reference_root'];rs=read(ref/'SEALED.json');graphroot=ROOT/p['graph_root'];gs=read(graphroot/'SEALED.json');stats=[];start=time.perf_counter()
 for row in rows:
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel] and sha(lat/rel)==ls['files'][rel];c=load(cap/rel);l=load(lat/rel,['input_sample_xyz','point_indices','boxes','scores']);base=softmax(c['stage_role_z'][-1,0].astype(float));early=softmax(c['stage_role_z'][:6,0].astype(float),axis=-1).mean(0)
  counts=c['role_masks'].sum(-1);adjusted=c['stage_role_z'].astype(float)-np.log(counts)[None,:,None]
  if len(counts)>1:role=adjusted[:,0]-(logsumexp(adjusted[:,1:],axis=1)-np.log(len(counts)-1))
  else:role=c['stage_role_z'][:,0].astype(float)
  finalrole=softmax(role[-1]);earlyrole=softmax(role[:6],axis=-1).mean(0)
  _,ix=np.unique(l['point_indices'],return_index=True);xyz=l['input_sample_xyz'][np.sort(ix)].astype(float);A=affinity(l['boxes'].astype(float),c['subject_phrase_boxes'].astype(float),xyz);subject,tu=transfer(A,c['subject_phrase_z'],base)
  gp=graphroot/('attributes_PRIVATE/'+sid+'.json');assert sha(gp)==gs['files']['attributes_PRIVATE/'+sid+'.json'];graph=read(gp)['text_graph_unchanged'];rcpath=ref/'cases_PRIVATE'/(sid+'.json');assert sha(rcpath)==rs['files']['cases_PRIVATE/'+sid+'.json'];rc=read(rcpath);refmaps=[];matrices=[A];refunknown=[];names=['T'];rinputs=[]
  for ent in graph['references']:
   key=ent['entity'];meta=rc['references'][key];rp=ref/meta['file'];assert sha(rp)==meta['sha256']==rs['files'][meta['file']];z=load(rp,['boxes','logodds','point_indices']);assert np.array_equal(z['point_indices'],l['point_indices']),'Cross-phrase association must use identical rawpoint sample'
   if meta['mask']['reference_usable']:
    RA=affinity(l['boxes'].astype(float),z['boxes'].astype(float),xyz);pm,ru=transfer(RA,z['logodds'],np.full(256,1/256));refmaps.append(pm);refunknown.append(ru);names.append(key);matrices.append(RA);rinputs.append({'entity_PRIVATE':key,'reference_file_sha256':sha(rp),'unknown_mass':ru})
  refmean=np.mean(refmaps,axis=0) if refmaps else np.full(256,1/256);phr=softmax(np.log(subject.clip(1e-300))-np.log(refmean.clip(1e-300)));teacher=(early+earlyrole+phr)/3;teacher/=teacher.sum()
  perm=np.random.default_rng(int.from_bytes(hashlib.sha256(('R56_candidate_alignment_control:'+sid).encode()).digest()[:8],'little')).permutation(256);shuffle=teacher[perm]
  result={'source':c['source_scores'],'zero_step':c['source_scores'],'proposal':c['stage_target_probability'][0],**{'decoder'+str(i):c['stage_target_probability'][i+1] for i in range(5)},'early_consensus':early,'final_role':finalrole,'early_role_consensus':earlyrole,'phrase_subject':subject,'phrase_role':phr,'semantic_consensus':teacher,'shuffled_consensus':shuffle}
  (out/'samples_PRIVATE').mkdir(exist_ok=True);np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),teacher=teacher,shuffled_teacher=shuffle,candidate_permutation=perm,static_arm_names=np.asarray(list(result)),static_scores=np.stack(list(result.values())),association_affinity=np.stack(matrices),association_entities=np.asarray(names),subject_phrase_unknown_mass=np.array(tu),reference_phrase_unknown_masses=np.asarray(refunknown),role_masks=c['role_masks'],source_boxes=l['boxes'],source_scores=l['scores'])
  rec={'platform':row['platform'],'roles_in_full_sentence':len(counts),'reference_phrases_used':len(refmaps),'subject_unknown_mass':tu,'reference_unknown_mean':float(np.mean(refunknown)) if refunknown else None,'subject_queries_with_point_box_support':int((A.sum(1)>0).sum()),'mapped_candidate_pairs':int((A>0).sum()),'unique_raw_sample_points':len(xyz),'candidate_permutation_keeps_probability_multiset':bool(np.array_equal(np.sort(teacher),np.sort(shuffle))),'no_query_index_reuse_across_phrases':True};dump(out/'cases_PRIVATE'/(sid+'.json'),{**rec,'references_PRIVATE':rinputs});stats.append(rec)
  if len(stats)%32==0:print('PREPARE56',len(stats),336,flush=True)
 dump(out/'COVERAGE.json',stats);dump(out/'COSTS.json',{'n':336,'seconds':time.perf_counter()-start,'new_models':0,'GT_reads':0,'reference_cache_outputs_reused':sum(x['reference_phrases_used'] for x in stats)});seal(out,'R56_INITIAL_INTERNAL_SEMANTIC_CONSENSUS_NO_GT',n=336,plan_sha256=sha(args.plan));print('SEALED56',len(stats),flush=True)
if __name__=='__main__':main()
