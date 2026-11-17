"""Independent native reader, exact prefix/continuation states and cumulative-budget audit."""
from common133 import *
import numpy as np,torch,sys
sys.path.insert(0,str(R133/'research/round129'));from audit129 import read
def main():
 basecheck();torch.set_num_threads(4);mat=rd(B133/'MATRIX_SEALED.json');reg=rd(H133/'INPUT_REGISTRY_PRIVATE.json');index={(e['source']+'_to_'+e['target'],e['row']['id']):e for e in reg};n=0;active=0;maxp=0.;maxloss=0.;maxdelta=0.;process=[]
 for lockname in ['MODEL_CODE_LOCK','FORMAL_LOCK','EVALUATION_CODE_LOCK']:
  for p,h in rd(H133/'experiment'/(lockname+'.json'))['files'].items():assert sha(R133/p)==h,p
 for name,h in mat['seals'].items():
  D=B133/'formal'/name;assert sha(D/'SEALED.json')==h
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
  src=name.split('_')[0]
  with np.load(R133/f'crossplatform/round127/{src}_AUDIT_MODEL_PRIVATE.npz') as f:W={k[6:]:torch.from_numpy(f[k].copy()) for k in f.files if k.startswith('match.')}
  for p in sorted((D/'samples_PRIVATE').glob('*.npz')):
   e=index[(name,p.stem)];a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));rec=rd(D/'records_PRIVATE'/(p.stem+'.json'));old=np.load(R133/'crossplatform/round132/formal'/name/'samples_PRIVATE'/(p.stem+'.npz'));op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']);assert a['scores'].shape==(9,256) and np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9);assert np.array_equal(a['queries'],a['scores'].argmax(1));n+=1
   for new,prior in [('zero_F','LOG_zero'),('direct_F','LOG_direct'),('partial3_F','LOG_partial'),('wrong3_F','LOG_wrong'),('R131_partial_F','partial_F')]:assert np.array_equal(ps[new],op[prior])
   assert rec['active']==e['active']
   if not e['active']:
    for label in ['partial','wrong']:
     for step in [1,3,5]:assert np.array_equal(ps[label+str(step)+'_F'],ps['zero_F'])
    assert not (D/'states_PRIVATE'/(p.stem+'.npz')).exists();continue
   active+=1;ff=np.load(D/'features_PRIVATE'/(p.stem+'.npz'));ss=np.load(D/'states_PRIVATE'/(p.stem+'.npz'));oldss=np.load(R133/e['provider_states']);oldff=np.load(R133/e['provider_features']);radius=rec['radius'];assert radius==e['provider_record']['parameter_radius'];assert np.array_equal(ff['fixed_late'],oldff['fixed_late']) and np.array_equal(ff['fixed_r'],oldff['fixed_r']) and np.array_equal(ff['mask'],oldff['mask'])
   for label in ['partial','wrong']:
    oldkey='wrong_base' if e['reference_original_R131'] and label=='wrong' else label;oldgkey='partial' if label=='wrong' and rec['identity_control'] else oldkey;newgkey='partial' if label=='wrong' and rec['identity_control'] else label;trace=rec['trajectories'][label];assert len(trace)==5 and trace[:3]==e['provider_record']['traces'][oldkey];assert np.array_equal(ss[label+'3'],oldss[oldkey]);delta=np.zeros(4896,np.float32);path=0.;accepted_radius_sum=0.;accepted=0;trials=0
    continuity=abs(trace[3]['before']-trace[2]['after']);maxloss=max(maxloss,continuity);assert continuity<1e-8
    for i,t in enumerate(trace,1):
     g=oldss[oldgkey+'_step'+str(i)+'_gradient'] if i<=3 else ss[newgkey+'_step'+str(i)+'_gradient'];previous=delta.copy()
     if t['accepted']:delta=(delta+(-np.float32(radius)*g/np.float32(t['gradient_L2']))*np.float32(t['fraction'])).astype(np.float32);accepted_radius_sum+=radius*t['fraction'];accepted+=int(t['accepted'])
     path+=float(np.linalg.norm((delta-previous).astype(np.float64)));trials+=len(t['trials']);assert t['after']<=t['before']+1e-8
     if i in [1,3,5]:
      error=float(abs(delta-ss[label+str(i)]).max());maxdelta=max(maxdelta,error);assert error<1e-7
      process.append({'source':src,'target':e['target'],'id_PRIVATE':p.stem,'label':label,'steps':i,'parameter_L2':float(np.linalg.norm(ss[label+str(i)].astype(np.float64))),'path_length':path,'accepted_radius_sum':accepted_radius_sum,'nominal_radius_budget':i*radius,'accepted_steps':accepted,'line_search_trials':trials,'backward_evaluations_if_independently_run':i,'full_reader_evaluations_if_independently_run':2*i+trials+1,'objective_after':t['after'],'task_after':t['task_after'],'feature_keep_after':t['feature_keep_after'],'identity_wrong':bool(label=='wrong' and rec['identity_control'])})
    for step in [1,5]:
     expected=read(ff[label+str(step)+'_h'],ff[label+str(step)+'_u'],ff['mask'],ff['fixed_late'],ff['fixed_r'],W);error=float(abs(expected-ps[label+str(step)+'_F']).max());maxp=max(maxp,error);assert error<5e-6
   if rec['identity_control']:
    for step in [1,3,5]:assert np.array_equal(ps['partial'+str(step)+'_F'],ps['wrong'+str(step)+'_F']) and np.array_equal(ss['partial'+str(step)],ss['wrong'+str(step)])
 assert n==1798 and active==646;out={'status':'PASS_ALL_NINE_ARMS_REUSED3_EXACT_CONTINUATION_READER_AND_STEP_BUDGETS','n':n,'active':active,'inactive':n-active,'max_CPU_probability_error':maxp,'max_reconstructed_parameter_error':maxdelta,'max_objective_discontinuity_at3to4':maxloss,'GT_read':False,'FINAL_access':False};dump(H133/'NATIVE_AUDIT.json',out);dump(H133/'PROCESS_ROWS_PRIVATE.json',process);print(out)
if __name__=='__main__':main()
