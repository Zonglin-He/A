"""Independent region identity, readout, loss and heldout-C numerical audit."""
from common74 import *
from scipy.special import softmax,logsumexp
from scipy.spatial import cKDTree
import time
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def qhead(w,h):
 x=h
 for k in ['0','2','4']:
  x=x@w[k+'.weight'].T+w[k+'.bias']
  if k!='4':x=np.maximum(x,0)
 return unit(x)
def odds(q,z):
 l=q@z['tokens'].T/.07;m=z['token_mask'];return logsumexp(l[:,m],1)-logsumexp(l[:,~m],1)
def cachedprob(z):return softmax(odds(z['query'].astype(float),{'tokens':z['tokens'].astype(float),'token_mask':z['token_mask']}))
def transport74(p,W,mass,base):
 y=W@p;return mass*y/max(y.sum(),1e-300)+(1-mass)*base
def objective74(qa,qb,qa0,qb0,e,kind):
 ra=unit(e['A_pool']@qa);rb=unit(e['B_pool']@qb);N=e['geometry_negatives'] if kind=='corr_noRGB' else e['RGB_negatives'];rb=rb[e['shuffle']] if kind=='corr_shuffled' else rb;sim=ra@rb.T/.07;allowed=N|np.eye(len(N),dtype=bool);left=logsumexp(np.where(allowed,sim,-np.inf),1)-sim.diagonal();right=logsumexp(np.where(allowed.T,sim.T,-np.inf),1)-sim.diagonal();corr=(left[N.any(1)].mean()+right[N.any(0)].mean())/2;keep=(np.square(qa.astype(float)-qa0).sum(1).mean()+np.square(qb.astype(float)-qb0).sum(1).mean())/4;return float(corr+keep)
def main():
 p=read(H/'experiment/PREDICT_PLAN_LOCK_attempt002.json');pin(p);pred=ROOT/p['prediction_root'];evroot=ROOT/p['evidence_root'];fits=ROOT/p['fit_root'];ps=read(pred/'SEALED.json');fs=read(fits/'SEALED.json');es=read(evroot/'SEALED.json');initial=load(ROOT/p['head_initial']);start=time.perf_counter();maxp=maxL=maxfeature=0.;accepted=0;different_masks=0;source_difference=0;oldroot=ROOT/'crossplatform/round74/predicted64_attempt001';oldseal=read(oldroot/'SEALED.json')
 for r in rows():
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz'
  for root,ss in [(pred,ps),(evroot,es),(oldroot,oldseal)]:assert sha(root/rel)==ss['files'][rel]
  z=ab(r);c=cview(r);ev=load(evroot/rel);got=load(pred/rel);old=load(oldroot/rel);actual=dict(zip(got['arm_names'],got['scores']));assert np.array_equal(got['queries'],got['scores'].argmax(1)) and np.array_equal(got['source_boxes'],z['A']['boxes']);assert np.array_equal(got['selected_original_boxes'],got['source_boxes'][got['queries']])
  for k,a in zip(old['arm_names'],old['scores']):assert np.array_equal(actual[k],a),('unchanged_original_readout',k)
  raw=np.fromfile(r['pointpath'],np.float32).reshape(-1,4)[:,:3].copy()
  if r['platform']=='drone':raw[:,2]+=np.float32(1.8)
  boxes=ev['region_boxes'];assert np.array_equal(boxes,z['A']['boxes'][ev['region_representatives']].astype(float));sgraph=load(ROOT/'crossplatform/round72/evidence2640_attempt001/graphs_PRIVATE'/(sid+'.npz'));geo_inside=(np.abs(raw[None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(2);visible=(np.abs(sgraph['xyz'][None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(2)
  for v in 'AB':
   ids=np.unique(z[v]['point_indices']);distance,_=cKDTree(raw[ids].astype(float)).query(z[v]['query_anchor_xyz'].astype(float),workers=1);assert distance.max()<1e-5;anchor=(np.abs(z[v]['query_anchor_xyz'][None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(2);pool=anchor/np.maximum(anchor.sum(1)[:,None],1);assert np.array_equal(pool,ev[v+'_pool'])
  shared=np.array([np.intersect1d(np.unique(z['A']['point_indices'])[geo_inside[i,np.unique(z['A']['point_indices'])]],np.unique(z['B']['point_indices'])[geo_inside[i,np.unique(z['B']['point_indices'])]]).size for i in range(len(boxes))]);assert np.array_equal(shared,ev['shared_AB_raw_counts']) and (shared>=4).all();assert not (ev['shuffle']==np.arange(len(boxes))).any()
  from region74 import geometry_labels
  noRGBlabels=geometry_labels(sgraph)
  # Original RGB components must refine the identical geometry-only graph.
  for lrgb,lg in zip(sgraph['labels'],noRGBlabels):
   assert all(np.unique(lg[lrgb==k]).size==1 for k in np.unique(lrgb))
  counts=visible.sum(1)
  for name,labels in [('RGB_negatives',sgraph['labels']),('geometry_negatives',noRGBlabels)]:
   expected=np.zeros((len(boxes),len(boxes)),bool)
   for i in range(len(boxes)):
    for j in range(i):
     lo=np.maximum(boxes[i,:3]-boxes[i,3:]/2,boxes[j,:3]-boxes[j,3:]/2);hi=np.minimum(boxes[i,:3]+boxes[i,3:]/2,boxes[j,:3]+boxes[j,3:]/2);inter=np.maximum(hi-lo,0).prod();io=inter/max(boxes[i,3:].prod()+boxes[j,3:].prod()-inter,1e-12);rawfrac=(geo_inside[i]&geo_inside[j]).sum()/max(min(geo_inside[i].sum(),geo_inside[j].sum()),1);ok=io<.05 and rawfrac<=.05 and min(counts[i],counts[j])>=4
     for lab in labels:
      hi0=np.bincount(lab[visible[i]],minlength=int(lab.max()+1));hj=np.bincount(lab[visible[j]],minlength=int(lab.max()+1));frac=np.minimum(hi0,hj).sum()/max(min(counts[i],counts[j]),1);ok=ok and frac<=.05
     expected[i,j]=expected[j,i]=ok
   assert np.array_equal(expected,ev[name])
  different_masks+=int(not np.array_equal(ev['RGB_negatives'],ev['geometry_negatives']));Q0={v:qhead(initial,zz['h']) for v,zz in [('A',z['A']),('B',z['B']),('C',c)]};base=ev['base'];p0={v:cachedprob(zz) for v,zz in [('A',z['A']),('B',z['B']),('C',c)]};frozen=(p0['A']+transport74(p0['B'],ev['B_W'],float(ev['B_fixed_mass']),base))/2;assert np.max(abs(frozen-actual['frozen_AB']))<1e-12;source_difference+=int(p0['A'].argmax()!=actual['source'].argmax());assoc=load(ROOT/'crossplatform/round69/predicted2304_attempt001/association_PRIVATE'/(sid+'.npz'),['C_W','C_fixed_mass','base']);savedaux=load(pred/'aux_PRIVATE'/(sid+'.npz'))
  for kind in ADAPT:
   relw='weights_PRIVATE/'+sid+'_'+kind+'.npz';assert sha(fits/relw)==fs['files'][relw];w=load(fits/relw);rec=read(fits/'cases_PRIVATE'/(sid+'_'+kind+'.json'));assert rec['C_view_read'] is False and rec['GT_expert_feedback_used'] is False;Q={v:qhead(w,zz['h']) for v,zz in [('A',z['A']),('B',z['B']),('C',c)]}
   for ww,record in [(Q0,'loss_before'),(Q,'loss_after')]:
    ll=objective74(ww['A'],ww['B'],Q0['A'],Q0['B'],ev,kind);err=abs(ll-rec[record]);maxL=max(maxL,err);assert err<2e-4
   if rec['accepted']:accepted+=1;assert rec['loss_after']<=rec['loss_before']-max(1e-12,1e-6*abs(rec['loss_before'])) and rec['parameter_step_L2']>0
   else:assert rec['before_sha256']==rec['after_sha256'] and rec['optimizer_before_sha256']==rec['optimizer_after_sha256']
   pp={}
   for v,zz in [('A',z['A']),('B',z['B']),('C',c)]:pp[v]=softmax(np.log(p0[v])+odds(Q[v],zz).astype(float)-odds(Q0[v],zz).astype(float))
   expected=(pp['A']+transport74(pp['B'],ev['B_W'],float(ev['B_fixed_mass']),base))/2;err=float(np.max(abs(expected-actual[kind])));maxp=max(maxp,err);assert err<5e-5;expectedC=transport74(pp['C'],assoc['C_W'],float(assoc['C_fixed_mass']),assoc['base']);err=float(np.max(abs(expectedC-actual['heldout_C_'+kind])));maxp=max(maxp,err);assert err<5e-5
   ra=unit(ev['A_pool']@Q['A']);rb=unit(ev['B_pool']@Q['B']);rc=unit(savedaux['C_pool']@Q['C']);rab=unit((ra+rb)/2);err=max(np.max(abs(rab-savedaux[kind+'_AB'])),np.max(abs(rc-savedaux[kind+'_C'])));maxfeature=max(maxfeature,float(err));assert err<2e-5
  historic=load(ROOT/'crossplatform/round71/predicted2640_attempt001'/rel);hist=dict(zip(historic['arm_names'],historic['scores']))
  for k,oldk in [('source','source'),('R59','R59'),('CLIP_reference','CLIP_direct')]:assert np.array_equal(actual[k],hist[oldk])
 out=H/'independent_audit_attempt001';assert not out.exists();out.mkdir();cert={'state':'PASS_R74_ALL64_REGION_IDENTITY_NATIVE_HEAD_LOSS_READOUT_AND_C_AUDIT','n':64,'actual_fits_audited':192,'accepted_steps':accepted,'accepted_objective_increases':0,'RGB_and_noRGB_mask_different_inputs':different_masks,'native_A_argmax_differs_from_cached_source':source_difference,'max_independent_probability_error':maxp,'max_independent_loss_error':maxL,'max_independent_C_roi_feature_error':maxfeature,'no_GT_lookup':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'initial11_readouts_unchanged_when_stronger_direct_control_added':True,'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',cert);print(cert)
if __name__=='__main__':main()
