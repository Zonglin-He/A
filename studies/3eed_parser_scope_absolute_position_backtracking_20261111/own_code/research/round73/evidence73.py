from common73 import *
from parser73 import patched_parse,classify,absolute_negatives
from geometry72 import supports
from constraints72 import build
from scipy.special import softmax,logsumexp
import itertools,time
sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import CameraSpec,read_camera_allowlist,project_model_points
def projection(r,boxes,wh,group):
 cam=CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),missing_distortion_policy='assume_zero_declared');corners=boxes[:,None,:3]+boxes[:,None,3:]/2*np.array(list(itertools.product([-1,1],repeat=3)))[None];pr=project_model_points(corners.reshape(-1,3),cam,wh);uv=pr['uv'].reshape(256,8,2);valid=pr['in_image'].reshape(256,8).all(1)&np.isfinite(uv).all((1,2));rect=np.c_[uv[:,:,0].min(1),uv[:,:,1].min(1),uv[:,:,0].max(1),uv[:,:,1].max(1)];ng=int(group.max()+1);gr=np.zeros((ng,4));gv=np.zeros(ng,bool)
 for g in range(ng):
  ix=group==g;gv[g]=bool(valid[ix].all());gr[g]=[rect[ix,0].min(),rect[ix,1].min(),rect[ix,2].max(),rect[ix,3].max()]
 return rect,valid,gr,gv
def main():
 p=read(H/'experiment/EVIDENCE_PLAN_LOCK.json');pin(p);out=ROOT/p['evidence_root'];assert not out.exists();out.mkdir(parents=True)
 for f in ['samples_PRIVATE','cases_PRIVATE']:(out/f).mkdir()
 start=time.perf_counter();count=0;rr=read(H/'input/MECHANISM64_PRIVATE.json');oldroot=ROOT/'crossplatform/round72/evidence2640_camera_contract_attempt001';rawroot=ROOT/'crossplatform/round72/evidence2640_attempt001';oldseal=read(oldroot/'SEALED.json');rawseal=read(rawroot/'SEALED.json')
 for r in rr:
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(oldroot/rel)==oldseal['files'][rel];old=load(oldroot/rel);relg='graphs_PRIVATE/'+sid+'.npz';assert sha(rawroot/relg)==rawseal['files'][relg];s=load(rawroot/relg);a=prep(r);z=prefix(r,('boxes','scores','tokens','query'));t=patched_parse(r['caption'],len(z['tokens']));constraint=classify(r['caption']);assert constraint['kind']=='IMAGE_ABSOLUTE';a['image_width']=s['wh'][0];tm=supports(s,z['boxes'],a,t['target']);rm=supports(s,z['boxes'],a,t['reference']);qref=None
  if t['ref_mask'].any():
   logits=z['query'].astype(float)@z['tokens'].astype(float).T/.07;m=t['ref_mask'];qref=softmax(logsumexp(logits[:,m],1)-logsumexp(logits[:,~m],1))
  fixed,_=build(a,z['boxes'],tm,rm,t,qref,sid);rect,valid,gr,gv=projection(r,z['boxes'],s['wh'],a['query_group']);direction=constraint['direction'];N,known,interval=absolute_negatives(gr,gv,gv.astype(float),s['wh'][0],direction);opp,_,_=absolute_negatives(gr,gv,gv.astype(float),s['wh'][0],'right' if direction=='left' else 'left');new=fixed['main_N']|N;wrong=fixed['main_N']|opp
  np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),old_N=old['main_N'],parse_N=fixed['main_N'],absolute_N=N,main_N=new,mismatch_N=wrong,position_known=known,query_group=a['query_group'],query_rects=rect,query_projection_valid=valid,group_rects=gr,normalized_intervals=interval,source_boxes=z['boxes'],image_wh=s['wh']);p0=z['scores'].astype(float);p0/=p0.sum();top=int(p0.argmax());rec={'id_PRIVATE':sid,'caption_PRIVATE':r['caption'],'constraint_PRIVATE':constraint,'target_request_PRIVATE':t['target'],'old_rejected_groups':int(old['main_N'].sum()),'parse_rejected_groups':int(fixed['main_N'].sum()),'position_rejected_groups':int(N.sum()),'known_projection_groups':int(known.sum()),'source_top1_rejected':bool(new[a['query_group'][top]]),'source_top1_absolute_rejected':bool(N[a['query_group'][top]]),'source_negative_mass':float(p0[new[a['query_group']]].sum()),'original_source_query_PRIVATE':top,'image_sha256':read(rawroot/'cases_PRIVATE'/(sid+'.json'))['image_sha256'],'point_sha256':read(rawroot/'cases_PRIVATE'/(sid+'.json'))['point_sha256']};dump(out/'cases_PRIVATE'/(sid+'.json'),rec);count+=int(new.any())
 dump(out/'COUNTS.json',{'n':64,'active_inputs':count,'seconds':time.perf_counter()-start,'new_full_model_or_external_calls':0,'new_GT_lookups':0});seal(out,'R73_SIMPLE_POSITION_EVIDENCE_SEALED_WITHOUT_TASK_LOOKUP',n=64);print(read(out/'COUNTS.json'))
if __name__=='__main__':main()
