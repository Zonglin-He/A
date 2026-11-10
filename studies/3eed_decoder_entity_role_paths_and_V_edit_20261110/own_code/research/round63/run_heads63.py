"""All 336 fixed frames retained; no task GT or scores read by interventions."""
import argparse,sys,time,torch
from common63 import *
from paths_heads63 import chain,cpu_head,readouts,lineage,gpu_strict
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--mode',choices=['screen','heads'],default='screen');ap.add_argument('--engineering',action='store_true');a=ap.parse_args();p=read(a.plan);pin(p);gpu_strict();rows=read(ROOT/p['manifest']);rows=engineering_rows(rows) if a.engineering else rows;out=ROOT/('research/round63/engineering_attempt001' if a.engineering else p[a.mode+'_root']);assert not out.exists();out.mkdir(parents=True)
 for f in ['samples_PRIVATE','states_PRIVATE','cases_PRIVATE','traces_PRIVATE']:(out/f).mkdir()
 model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);head=cpu_head(load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz'));arms=p[a.mode+'_arms'];start=time.perf_counter();statuses=[];bridge=0.;cap=ROOT/p['capture_root'];sealcap=read(cap/'SEALED.json');ga=ROOT/p['association_root'];sg=read(ga/'SEALED.json');sr=read(ROOT/p['r59_root']/'SEALED.json');eligible=0
 for idx,row in enumerate(rows):
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==sealcap['files'][rel];z=load(cap/rel);z['_sid']=sid;assert sha(ga/rel)==sg['files'][rel];g=load(ga/rel);assert sha(ROOT/p['r59_root']/rel)==sr['files'][rel];old=load(ROOT/p['r59_root']/rel);result={name:old['scores'][list(old['arm_names']).index(k)] for name,k in [('source','source'),('frozen_ABC','frozen_ABC'),('r59_direct_all','direct_all')]};state={};traces={};cases={};valid=False
  if a.mode=='screen':
   hs={}
   for v in 'ABC':
    h,hist,rec,tr=chain(model,z,v,trace=True);hs[v]=h;bridge=max(bridge,float(abs(h-z[v+'_h']).max()));assert abs(h-z[v+'_h']).max()<1e-4
    for l,x in tr.items():traces[v+'_L'+str(l)]=x
   zero,rz,_=readouts(head,hs,z,g);assert abs(zero-result['frozen_ABC']).max()<3e-5 and zero.argmax()==result['frozen_ABC'].argmax();assert abs(rz-result['r59_direct_all']).max()<3e-5 and rz.argmax()==result['r59_direct_all'].argmax()
  for arm in arms[3:]:
   if a.mode=='screen':l=int(arm[1]);kind=arm[3:];hh=None
   else:l=p['selected_layer'];kind=p['selected_kind'];hh=int(arm[1:])
   hs={};bb={};details={}
   for v in 'ABC':
    h,hist,rec,_=chain(model,z,v,l,kind,hh);hs[v]=h;bb[v]=hist[-1];state[arm+'_'+v+'_h']=h;state[arm+'_'+v+'_boxes']=bb[v];details[v]=rec;valid=rec['valid_roles']
   result[arm]=readouts(head,hs,z,g)[0];details['identity']=lineage(z,bb,g);cases[arm]=details
  eligible+=int(valid);scores=np.stack([result[x] for x in arms]);assert np.isfinite(scores).all();q=scores.argmax(1);np.savez_compressed(out/rel,arm_names=np.array(arms),scores=scores,queries=q,source_boxes=z['A_boxes'],source_scores=z['A_scores'],selected_original_boxes=z['A_boxes'][q]);np.savez_compressed(out/'states_PRIVATE'/(sid+'.npz'),**state)
  if traces:np.savez_compressed(out/'traces_PRIVATE'/(sid+'.npz'),**traces)
  dump(out/'cases_PRIVATE'/(sid+'.json'),cases);statuses.append({'id':sid,'arm_statuses':{arm:{'unknown':not valid if arm not in p['reference_arms'] else False,'fallback':not valid if arm not in p['reference_arms'] else False,'no_parameter_update':True} for arm in arms}})
  if (idx+1)%16==0:print('R63_'+a.mode,idx+1,len(rows),flush=True)
 assert fingerprint(model)==fp;cost={'n':len(rows),'arms':len(arms),'role_eligible_n':eligible,'source_hidden_bridge_max_abs':bridge,'source_restored_exact':True,'new_full_source_forwards':0,'new_gradient_or_fit_steps':0,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COSTS.json',cost);dump(out/'READOUT_STATUS_PRIVATE.json',statuses);seal(out,'R63_'+a.mode.upper()+'_ROLE_PATH_PREDICTIONS_SEALED_BEFORE_SCORE',n=len(rows),arm_names=arms);print(cost)
if __name__=='__main__':main()
