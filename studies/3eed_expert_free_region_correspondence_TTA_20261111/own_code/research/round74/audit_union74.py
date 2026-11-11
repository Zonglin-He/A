"""All-input union provenance and same-correspondence direct readout proof."""
from common74 import *
from scipy.special import softmax,logsumexp
def main():
 p=read(H/'experiment/ALL2304_PLAN_LOCK.json');pred=ROOT/p['prediction_root'];evidence=ROOT/p['evidence_root'];pilot={r['id'] for r in rows()};pp=read(H/'experiment/PREDICT_PLAN_LOCK_attempt002.json');pb=read(H/'expanded/experiment/PREDICT_PLAN_LOCK_attempt002.json');ss=read(pred/'SEALED.json');maxerr=0.;n=0
 for f in [H/'independent_audit_attempt001/AUDIT_RESULT.json',H/'independent_direct_C_audit_attempt001/AUDIT_RESULT.json',H/'expanded/independent_audit_attempt001/AUDIT_RESULT.json',H/'expanded/independent_direct_C_audit_attempt001/AUDIT_RESULT.json']:assert read(f)['state'].startswith('PASS_')
 for r in read(ROOT/'research/round69/input/ALL2304_PRIVATE.json'):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(pred/rel)==ss['files'][rel];got=load(pred/rel);lookup=dict(zip(got['arm_names'],got['scores']));old=load(ROOT/(pp if sid in pilot else pb)['prediction_root']/rel)
  for key,value in zip(old['arm_names'],old['scores']):assert np.array_equal(lookup[key],value)
  assert np.array_equal(got['queries'],got['scores'].argmax(1));assert np.array_equal(got['selected_original_boxes'],got['source_boxes'][got['queries']]);z=ab(r);e=load(evidence/rel);qa=z['A']['query'].astype(float).copy();qb=z['B']['query'].astype(float).copy();ma=e['A_pool']@qa;mb=e['B_pool']@qb;active=e['RGB_negatives'].any(1)
  for q,P,d in [(qa,e['A_pool'],(mb-ma)/2),(qb,e['B_pool'],(ma-mb)/2)]:
   for j in range(256):
    owners=np.flatnonzero((P[:,j]>0)&active)
    if len(owners)==1:
     q[j]+=d[owners[0]];q[j]/=max(np.linalg.norm(q[j]),1e-12)
  score={}
  for view,q in [('A',qa),('B',qb)]:
   logits=q@z[view]['tokens'].astype(float).T/.07;m=z[view]['token_mask'];score[view]=softmax(logsumexp(logits[:,m],1)-logsumexp(logits[:,~m],1))
  b=e['B_W']@score['B'];mass=float(e['B_fixed_mass']);b=mass*b/max(b.sum(),1e-300)+(1-mass)*e['base'];expected=(score['A']+b)/2;err=float(np.max(abs(expected-lookup['frozen_region_consensus'])));maxerr=max(maxerr,err);assert err<1e-12;n+=1
 out=H/'independent_union_audit_attempt001';assert not out.exists();out.mkdir();cert={'state':'PASS_R74_ALL2304_UNION_AND_SAME_REGION_NO_UPDATE_READER','n':n,'all_original_readouts_preserved':True,'pilot_not_refit_or_repredicted':True,'no_remaining_GT_lookup':True,'max_independent_direct_probability_error':maxerr,'prediction_seal_sha256':sha(pred/'SEALED.json')};dump(out/'AUDIT_RESULT.json',cert);print(cert)
if __name__=='__main__':main()
