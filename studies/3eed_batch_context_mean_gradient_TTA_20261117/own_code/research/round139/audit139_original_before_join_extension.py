"""CPU replay of batch deltas/first gradients/native scores/mean objectives."""
from common139 import *
import argparse,numpy as np,torch
from torch.nn import functional as F
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round135'));from audit135 import read135,tensors;sys.argv=argv
sys.path.insert(0,str(R/'research/round129'));import support129 as U
def main():
 check();torch.set_num_threads(4);root=B/('engineering' if args.engineering else 'formal');seal=rd(root/'SEALED.json');weights={s:tensors(dict(np.load(R/'crossplatform/round135'/(s+'_SOURCE_MODEL_PRIVATE.npz')))) for s in ['waymo','drone','quad']};nb=0;nr=0;mxp=0.;mxg=0.;mxd=0.;mxo=0.;process=[]
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for D in sorted((root/'batches_PRIVATE').iterdir()):
  rec=rd(D/'record.json');src,tgt=rec['source'],rec['target'];name=src+'_to_'+tgt;W=weights[src];ss=np.load(D/'state.npz');delta=np.zeros(4896,np.float32);kind='wrong_C_B_A' if rec['wrong'] else 'C_B_A';gg=[];nr+=len(rec['members_PRIVATE']);nb+=1
  for sid in rec['members_PRIVATE']:gg.append(np.load(R/'crossplatform/round136/method'/name/'states_PRIVATE'/(sid+'.npz'))[kind+'_gradient1'])
  expected=np.stack(gg).mean(0);err=float(abs(expected-ss['gradient1']).max());mxg=max(mxg,err);assert err<1e-5
  for j,t in enumerate(rec['trace'],1):
   g=ss['gradient'+str(j)]
   if t['accepted']:delta=(delta-np.float32(rec['radius'])*g/np.float32(t['gradient_L2'])*np.float32(t['fraction'])).astype(np.float32)
   assert t['after']<=t['before']+1e-8
   for p in t['trials']:assert p['accepted']==(p['objective']<=p['bound'])
  err=float(abs(delta-ss['delta']).max());mxd=max(mxd,err);assert err<1e-7;losses=[]
  for sid,case in zip(rec['members_PRIVATE'],rec['case_final_PRIVATE']):
   assert sid==case['id_PRIVATE'];a=np.load(D/(sid+'.npz'));feat=tensors({k:a[k] for k in ['h','u','late','local','own','text','relation_text','mask']});rt=tensors(dict(np.load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'))));p=read135(feat,rt,W);err=float(abs(p-a['probability']).max());mxp=max(mxp,err);assert err<5e-6
   ev=np.load(R/'crossplatform/round136/evidence_PRIVATE'/name/(sid+'.npz'));ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];lp=torch.from_numpy(np.maximum(a['probability'],1e-300)).log();partial=float(U.taskloss(lp,groups,torch.from_numpy(ev['S']),torch.from_numpy(ev['C']),None,'partial'));gm=torch.stack([lp[g].max() for g in groups]);target=torch.from_numpy(ev['wrong_target'] if rec['wrong'] else ev['target']);rank=float((target*(target.clamp_min(1e-300).log()-gm[ev['C']].log_softmax(0))).sum());X=torch.from_numpy(np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'))['vis0'][0].copy());hidden=F.gelu(F.linear(X,W['point_adapter.down.weight'],W['point_adapter.down.bias']));source=X+F.linear(hidden,W['point_adapter.up.weight'],W['point_adapter.up.bias']);adapt=X+F.linear(hidden,W['point_adapter.up.weight']+torch.from_numpy(delta[:4608].reshape(288,16)),W['point_adapter.up.bias']+torch.from_numpy(delta[4608:]));keep=float((F.normalize(adapt,dim=-1)-F.normalize(source,dim=-1)).square().sum(-1).mean()/2);loss=partial+rank+keep;err=max(abs(partial-case['partial']),abs(rank-case['rank']),abs(keep-case['keep']),abs(loss-case['objective']));mxo=max(mxo,err);assert err<2e-6;losses.append(loss)
  err=abs(float(np.mean(losses))-rec['trace'][-1]['after']);mxo=max(mxo,err);assert err<2e-6;process.append({'source':src,'target':tgt,'order':rec['order'],'wrong':rec['wrong'],'batch_size':rec['batch_size'],'batch_PRIVATE':rec['batch_PRIVATE'],'accepted_steps':sum(t['accepted'] for t in rec['trace']),'trials':sum(len(t['trials']) for t in rec['trace']),'parameter_L2':float(np.linalg.norm(delta.astype(float))),'initial_mean_loss':rec['trace'][0]['before'],'final_mean_loss':rec['trace'][-1]['after'],'individual_loss_increases':sum(a>b+1e-10 for a,b in zip(rec['trace'][-1]['per_case_after_PRIVATE'],rec['trace'][0]['per_case_before_PRIVATE']))})
  if nb%32==0:print('AUDIT139',nb,nr,flush=True)
 if not args.engineering:
  count=0;eligible=set()
  for part in rd(H/'BATCH_PARTITIONS_PRIVATE.json'):
   for sid in part['members_PRIVATE']:eligible.add((part['source'],part['target'],sid))
  for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
   src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(root/'samples_PRIVATE'/name/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));old=np.load(R/'crossplatform/round136/method'/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1));assert np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9)
   for k in ['Frozen_A','Frozen_F','AB_mean','direct_C_B_A','C_B_A','wrong_C_B_A']:assert np.array_equal(ps[k],op[k])
   if (src,tgt,sid) not in eligible:
    for order in [0,1,2]:
     for kind in ['correct','wrong']:assert np.array_equal(ps[kind+'_order'+str(order)],ps['Frozen_A'])
   count+=1
  assert count==1798 and nr==764*6
 out={'status':'PASS','engineering':args.engineering,'batch_fits':nb,'case_fits':nr,'max_first_gradient_error_vs_prior_source_state_mean':mxg,'max_delta_error':mxd,'max_CPU_native_probability_error':mxp,'max_CPU_objective_error':mxo,'singleton_native_bridge_passed':True,'source_restore_and_onlyUp4896_native_asserted':True,'full_encoder_later_gradients_not_CPU_regenerated':True,'FINAL_access':False};dump(H/('ENGINEERING_AUDIT.json' if args.engineering else 'AUDIT.json'),out);dump(H/('ENGINEERING_PROCESS_ROWS_PRIVATE.json' if args.engineering else 'PROCESS_ROWS_PRIVATE.json'),process);print(out,flush=True)
if __name__=='__main__':main()
