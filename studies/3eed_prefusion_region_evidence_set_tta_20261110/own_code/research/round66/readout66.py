"""Read C only after evidence and A/B fitting seals; same-evidence direct controls."""
import sys,time,torch
from common66 import *
from scipy.special import softmax
sys.path.insert(0,str(ROOT/'research/round58'));import numerics58 as n
sys.path.insert(0,str(ROOT/'research/round62'));from compute62 import cpu_head,readouts

def direct(p,e,key='E'):
 p=p.copy();C=e['tested_query_mask'];S=e['positive_query_mask']
 if not C.any():return p
 mass=float(p[C].sum())
 values=np.zeros(256)
 for j,member in enumerate(e['members']):
  if e['stable'][j]:values[member]=e[key][j]
 p[C]=mass*softmax(np.log(p[C].clip(1e-300))+values[C]);return p

def set_direct(p,e):
 out=p.copy();C=e['tested_query_mask'];S=e['positive_query_mask']
 if not C.any() or not S.any() or np.array_equal(C,S):return out
 out[C]=0;out[S]=float(p[C].sum())*p[S]/max(float(p[S].sum()),1e-300);return out

def main():
 H=ROOT/'research/round66';p=read(H/'experiment/READOUT_PLAN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);froot=ROOT/p['fit_root'];fs=read(froot/'SEALED.json');assert fs['n']==336;ev=ROOT/p['evidence_root'];es=read(ev/'SEALED.json');out=ROOT/p['prediction_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');source=n.head(initial).requires_grad_(False);torch.set_grad_enabled(False);torch.set_num_threads(4);start=time.perf_counter();statuses=[]
 for i,r in enumerate(rows):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(froot/rel)==fs['files'][rel] and sha(ev/rel)==es['files'][rel];e=load(ev/rel);ab=load(ROOT/p['AB_root']/rel);c=load(ROOT/'crossplatform/round57/capture336_attempt001'/rel);g=load(ROOT/p['association_root']/rel);prior=load(ROOT/p['r59_root']/rel);d=n.prepared(ab,c,g);m=cpu_head(load(froot/rel));result={name:prior['scores'][list(prior['arm_names']).index(key)] for name,key in [('source','source'),('frozen_ABC','frozen_ABC'),('R59','direct_all')]};result['C_source']=n.distribution(source,d,'C').numpy().copy()
  for stem,base in [('ABC',result['frozen_ABC']),('R59',result['R59']),('C',result['C_source'])]:result[stem+'_E_direct']=direct(base,e)
  for stem,base in [('R59',result['R59']),('C',result['C_source'])]:result[stem+'_set_direct']=set_direct(base,e)
  for name,key in [('R59_full_only','full_only'),('R59_category_only','category_only'),('R59_control_only','control_only')]:result[name]=direct(result['R59'],e,key)
  hs={v:(c['h'] if v=='C' else ab[v+'_h']) for v in 'ABC'};zz={v+'_'+k:(c[k] if v=='C' else ab[v+'_'+k]) for v in 'ABC' for k in ['tokens','token_mask']};result['ABC_tta1'],result['R59_tta1'],_=readouts(m,hs,zz,g);result['C_tta1']=n.distribution(m,d,'C').numpy();q=np.array([result[a].argmax() for a in ARMS]);scores=np.stack([result[a] for a in ARMS]);assert np.isfinite(scores).all();np.savez_compressed(out/rel,arm_names=np.array(ARMS),scores=scores,queries=q,source_boxes=ab['A_boxes'],source_scores=ab['A_scores'],selected_original_boxes=ab['A_boxes'][q]);fit=read(froot/'cases_PRIVATE'/(sid+'.json'));statuses.append({'id':sid,'arm_statuses':{a:{'unknown':False,'fallback':bool((a.endswith('_direct') or a in ['R59_full_only','R59_category_only','R59_control_only']) and not e['tested_query_mask'].any()),'no_parameter_update':not (a.endswith('tta1') and fit['step_L2']>0)} for a in ARMS}});dump(out/'cases_PRIVATE'/(sid+'.json'),{'source_and_feedback_fixed':True,'step_accepted':fit['accepted'],'query_choices_outside_tested_pool':{a:bool(not e['tested_query_mask'][q[j]]) for j,a in enumerate(ARMS)}})
  if (i+1)%64==0:print('READOUT66',i+1,len(rows),flush=True)
 dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'COSTS.json',{'n':336,'arms':len(ARMS),'seconds':time.perf_counter()-start,'C_read_only_after_fit_seal':True});seal(out,'R66_DIRECT_AND_REAL_MATCHING_TTA_WITH_HELDOUT_C_SEALED_BEFORE_TASK_SCORE',n=336,arm_names=ARMS)
if __name__=='__main__':main()
