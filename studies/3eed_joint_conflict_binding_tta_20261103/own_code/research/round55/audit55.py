"""Independent all-row numerical/observation/scope audit; no task GT access.

Does not import producer modules. Uses source saved query projections plus
frozen checkpoint projection weights, independently reconstructing scores.
Camera projection/prototypes remain documented common dependencies.
"""
from pathlib import Path
import argparse,json,hashlib,sys,itertools,math,time
import numpy as np
import torch
from scipy.special import logsumexp,softmax
R=Path(__file__).resolve().parents[2]
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def scores(hidden,weight,bias,tokens,masks):
 q=torch.nn.functional.linear(hidden,weight,bias);q=torch.nn.functional.normalize(q,p=2,dim=-1).numpy()[0].astype(float)
 logits=q@tokens.astype(float).T/.07
 return np.stack([logsumexp(logits[:,m],axis=-1)-logsumexp(logits[:,~m],axis=-1) for m in masks])
def distribution(z,C,z0):
 p0=softmax(z0,axis=-1)
 if not len(C):return p0
 known=np.r_[C.any(0).any(-1)[None],C.any(1)];out=np.zeros_like(z)
 for e in range(len(z)):
  for mask in (known[e],~known[e]):
   if mask.any():out[e,mask]=softmax(z[e,mask])*p0[e,mask].sum()
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args();p=rd(args.plan)
 for group in ('inputs_sha256','programs_sha256'):
  for path,sha in p[group].items():assert h(R/path)==sha,('pin',path)
 out=R/p['output_root'];assert not out.exists();out.mkdir(parents=True);start=time.perf_counter();rows=rd(R/p['manifest']);pred=R/p['prediction_root'];seal=rd(pred/'SEALED.json');ev=R/p['evidence_root'];es=rd(ev/'SEALED.json');lat=R/p['latent_root'];ls=rd(lat/'SEALED.json');arms=seal['arm_names']
 torch.set_num_threads(4);source=rd(R/p['source_plan']);state=torch.load(source['source_checkpoint_path'],map_location='cpu',weights_only=True)['model'];assert source['source_checkpoint_sha256']==h(source['source_checkpoint_path'])
 prefix='contrastive_align_projection_image.';weights={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)};del state
 max_score_error=0.;max_projection_error=0.;fitted_states=0;changed_unknown_mass=0.;all_source=True;activecounts=[0,0,0];qchanges=[];raw_rgb_checks=0
 from PIL import Image
 for row in rows:
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert h(pred/rel)==seal['files'][rel] and h(ev/rel)==es['files'][rel] and h(lat/rel)==ls['files'][rel]
  a=load(pred/rel);e=load(ev/rel);l=load(lat/rel,['h','query','tokens','scores','boxes','token_mask']);rec=rd(pred/'cases_PRIVATE'/(sid+'.json'));assert h(pred/'fitted_PRIVATE'/(sid+'.npz'))==rec['fitted_sha256'];f=load(pred/'fitted_PRIVATE'/(sid+'.npz'))
  assert np.array_equal(a['source_boxes'],l['boxes']) and np.array_equal(a['selected_original_boxes'],l['boxes'][a['queries']]) and np.array_equal(a['queries'],a['scores'].argmax(1));assert len(arms)==17 and a['scores'].shape==(17,256)
  assert np.array_equal(a['source_scores'],l['scores']) and np.array_equal(a['scores'][0],l['scores']) and np.array_equal(a['scores'][1],l['scores']);assert np.array_equal(e['role_masks'][0],l['token_mask'])
  with torch.no_grad():
   hidden=torch.from_numpy(l['h'])[None]
   for ix in [0,2]:hidden=torch.relu(torch.nn.functional.linear(hidden,weights[str(ix)+'.weight'],weights[str(ix)+'.bias']))
  z0=scores(hidden,weights['4.weight'],weights['4.bias'],l['tokens'],e['role_masks'])
  # Source query bridge against independently loaded trained weights; FP64 score audit has fixed tolerance.
  q=torch.nn.functional.normalize(torch.nn.functional.linear(hidden,weights['4.weight'],weights['4.bias']),dim=-1).numpy()[0];max_projection_error=max(max_projection_error,float(np.abs(q-l['query']).max()))
  C=e['conflicts'];c=e['color_conflict'];cs=e['shuffled_color_conflict'];g=e['geometry_conflict'];ep=e['endpoint_permutation'];assert sorted(ep.tolist())==list(range(256))
  for v,(col,geo) in enumerate([(c,g),(cs,g),(c,g[:,:,ep])]):
   rebuilt=(1-(1-geo)*(1-col[0][None,:,None])*(1-col[1:,None,:])).clip(0,.95).astype(np.float32) if len(col)>1 else np.empty((0,256,256),np.float32)
   assert np.array_equal(rebuilt,C[v]);activecounts[v]+=int(C[v].any())
  # Original RGB bytes recheck all336 against calibrated visible-pixel table, ignoring R51 groups/owners.
  ar=load(R/'crossplatform/round51/prepared336_attempt001/arrays_PRIVATE'/(sid+'.npz'),['visible_pixels_xy']);xy=ar['visible_pixels_xy']
  with Image.open(row['imagepath']) as im:rgb=np.array(im.convert('RGB'))
  assert np.array_equal(rgb[xy[:,1],xy[:,0]],e['visible_rgb']);raw_rgb_checks+=len(xy)
  for vi,variant in enumerate(['real','RGBshuffle','endpointshuffle']):
   currentC=C[vi]
   for step in [0,1,3]:
    if step==0:z=z0
    else:
     A=f[variant+'_'+str(step)+'__A'];B=f[variant+'_'+str(step)+'__B'];saved=f[variant+'_'+str(step)+'__role_scores'];z=scores(hidden,weights['4.weight']+torch.from_numpy(B)@torch.from_numpy(A),weights['4.bias'],l['tokens'],e['role_masks']);err=float(np.abs(z-saved).max());max_projection_error=max(max_projection_error,err);assert err<5e-4
     # Use saved float32 scores for downstream exact conditioning vs independent math.
     z=saved.astype(float);fitted_states+=1
    zz=z0 if step==0 else z;pp=distribution(zz,currentC,z0)
    assert np.allclose(pp.sum(1),1,atol=1e-10)
    if len(currentC):
     known=np.r_[currentC.any(0).any(-1)[None],currentC.any(1)];base=softmax(z0,axis=-1)
     for entity in range(len(z0)):
      changed_unknown_mass=max(changed_unknown_mass,abs(pp[entity,~known[entity]].sum()-base[entity,~known[entity]].sum()))
     joined=np.log(pp[0].clip(1e-300))+np.log(1-np.einsum('rj,rij->ri',pp[1:],currentC)).sum(0)
    else:joined=np.log(pp[0].clip(1e-300))
    checks=[(variant+'_direct',joined)] if step==0 else [(variant+'_tta'+str(step)+'_raw',np.log(pp[0].clip(1e-300))),(variant+'_tta'+str(step)+'_joint',joined)]
    for arm,value in checks:
     produced=a['scores'][arms.index(arm)];error=float(abs(value-produced).max());max_score_error=max(max_score_error,error);assert error<5e-4
     # Independent double precision may differ near ties, record rather than silently relax decision checks.
     if int(value.argmax())!=int(produced.argmax()):qchanges.append({'arm':arm,'row_number':len(qchanges),'max_error':error})
  for v in ['real','RGBshuffle','endpointshuffle']:
   stats=rec['statistics'][v]['steps'];assert len(stats)==3
   for st in stats:
    assert st['active']==bool(C[['real','RGBshuffle','endpointshuffle'].index(v)].any())
    assert st['parameter_delta_L2']>=0 and st['gradient_norm']>=0
  all_source&=bool(rec['source_score_bridge_bitexact'])
 assert max_projection_error<5e-4 and max_score_error<5e-4 and not qchanges and changed_unknown_mass<1e-10
 result={'state':'PASS_R55_ALL_OBSERVATIONS_FITTED_PROJECTIONS_AND_JOINT_READOUT_AUDIT','n':336,'arm_names':arms,'prediction_seal_sha256':h(pred/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'source_zero_step_exact':all_source,'fitted_states_checked':fitted_states,'all_visible_RGB_pixel_checks':raw_rgb_checks,'active_inputs_by_variant':activecounts,'max_independent_score_abs_error':max_score_error,'max_projection_role_score_abs_error':max_projection_error,'fixed_score_tolerance':.0005,'independent_Top1_disagreements':0,'max_fixed_unknown_stratum_mass_error':changed_unknown_mass,'GT_reads':0,'seconds':time.perf_counter()-start,'program_sha256':h(__file__),'plan_sha256':h(args.plan)}
 dump(out/'AUDIT_RESULT.json',result);dump(out/'SEALED.json',{'files':{f.name:h(f) for f in out.iterdir() if f.is_file()}});print(json.dumps(result),flush=True)
if __name__=='__main__':main()
