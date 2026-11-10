"""Only one prespecified role/head/strength; no gradients or pseudo-label fit."""
import argparse,sys,time,torch
from common63 import *
from paths_heads63 import cpu_head,readouts,lineage,gpu_strict
from target_value63 import source_branch,edit_chain
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--engineering',action='store_true');a=ap.parse_args();p=read(a.plan);pin(p);gpu_strict();config=read(ROOT/p['runtime_config']);assert config['selected_head'] is not None;model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);head=cpu_head(load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz'));out=ROOT/('research/round63/engineering_value_attempt001' if a.engineering else p['edit_root']);assert not out.exists();out.mkdir(parents=True)
 for f in ['samples_PRIVATE','states_PRIVATE','cases_PRIVATE']:(out/f).mkdir()
 cap=ROOT/p['capture_root'];cs=read(cap/'SEALED.json');hsroot=ROOT/p['heads_root'];hsseal=read(hsroot/'SEALED.json');start=time.perf_counter();statuses=[];maxsame=0.;applied=0
 rows=read(ROOT/p['manifest']);rows=engineering_rows(rows) if a.engineering else rows
 for i,row in enumerate(rows):
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel] and sha(hsroot/rel)==hsseal['files'][rel];z=load(cap/rel);z['_sid']=sid;old=load(hsroot/rel);g=load(ROOT/p['association_root']/rel);result={k:old['scores'][list(old['arm_names']).index(k)] for k in p['reference_arms']};result['head_path_knockout']=old['scores'][list(old['arm_names']).index('H'+str(config['selected_head']))];sources={};state={};cases={};caseapplied=False;valid=False
  for v in 'ABC':
   h,_,sources[v]=source_branch(model,z,v,config['selected_layer']);assert abs(h-z[v+'_h']).max()<1e-4
  for mode in ['physical','functional','opposite','shuffled','text_only','visual_only','random_path']:
   hh={};bb={};rec={}
   for v in 'ABC':
    h,hist,r=edit_chain(model,z,v,config,mode,sources[v]);hh[v]=h;bb[v]=hist[-1];rec[v]=r;state[mode+'_'+v+'_h']=h;state[mode+'_'+v+'_boxes']=bb[v];valid=r['valid_roles'];caseapplied=r.get('applied',False)
   native,fused,_=readouts(head,hh,z,g);result[mode]=native
   if mode=='physical':result['physical_R59']=fused
   rec['identity']=lineage(z,bb,g);cases[mode]=rec
  applied+=int(caseapplied);same=float(abs(result['physical']-result['functional']).max());maxsame=max(maxsame,same);assert same<3e-5 and result['physical'].argmax()==result['functional'].argmax();scores=np.stack([result[k] for k in p['edit_arms']]);q=scores.argmax(1);assert np.isfinite(scores).all();np.savez_compressed(out/rel,arm_names=np.array(p['edit_arms']),scores=scores,queries=q,source_boxes=z['A_boxes'],source_scores=z['A_scores'],selected_original_boxes=z['A_boxes'][q]);np.savez_compressed(out/'states_PRIVATE'/(sid+'.npz'),**state);dump(out/'cases_PRIVATE'/(sid+'.json'),cases)
  actualwrites={'physical','physical_R59','opposite','shuffled','text_only','visual_only'};ss={}
  for arm in p['edit_arms']:
   mode='physical' if arm=='physical_R59' else arm;rr=cases.get(mode,{});known=valid if mode in ['random_path','head_path_knockout'] else any(not rr[v].get('numeric_UNKNOWN',False) and rr[v].get('valid_roles',False) for v in 'ABC') if rr else True;changed=arm in actualwrites and any(rr[v].get('actual_changed_parameter_n',0)>0 for v in 'ABC');ss[arm]={'unknown':not known,'fallback':not known,'no_parameter_update':not changed}
  statuses.append({'id':sid,'arm_statuses':ss})
  if (i+1)%16==0:print('R63_EDIT',i+1,len(rows),flush=True)
 assert fingerprint(model)==fp;cost={'n':len(rows),'arms':len(p['edit_arms']),'role_eligible_applied_n':applied,'new_optimizer_or_fit_steps':0,'source_restored_exact':True,'max_physical_functional_score_error':maxsame,'all_physical_functional_Top1_exact':True,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COSTS.json',cost);dump(out/'READOUT_STATUS_PRIVATE.json',statuses);seal(out,'R63_TARGETED_V_ROLE_SUBSPACE_AND_COMPUTATIONAL_PATH_CONTROLS_SEALED',n=len(rows),arm_names=p['edit_arms']);print(cost)
if __name__=='__main__':main()
