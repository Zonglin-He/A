"""Independent fixed formula, actual positional MLP and complete final-head checks."""
from common144 import *
import argparse,numpy as np,torch
from torch.nn import functional as F
from scipy.spatial import cKDTree
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round135'));from audit135 import read135,tensors;sys.argv=argv

def embed(x,W):
 x=torch.from_numpy(x.copy()).transpose(1,2);q='position_embedding_head.';x=F.conv1d(x,W[q+'0.weight'],W[q+'0.bias']);x=F.batch_norm(x,W[q+'1.running_mean'],W[q+'1.running_var'],W[q+'1.weight'],W[q+'1.bias'],training=False,eps=1e-5);x=F.relu(x);return F.conv1d(x,W[q+'3.weight'],W[q+'3.bias']).transpose(1,2).contiguous().numpy()
def main():
 check();torch.set_num_threads(4);reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));root=B/('engineering' if args.engineering else 'formal');seal=rd(root/'SEALED.json');met={(e['source'],e['target'],e['id_PRIVATE']):e for e in rd(H/'POSITION_ROWS_PRIVATE.json')};srcrows=rd(R/'research/round140/SOURCE_RAW_GEOMETRY_METADATA_ROWS_PRIVATE.json');hh={s:float(np.median([e['z10'] for e in srcrows if e['source']==s])) for s in ['waymo','drone','quad']};n=nb=0;mxp=mxpos=0.
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for src in ['waymo','drone','quad']:
  W=tensors(dict(np.load(R/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz'))));PW=tensors(dict(np.load(root/(src+'_POSITION_MODULE_PRIVATE.npz'))))
  for e in [e for e in reg if e['source']==src]:
   tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name;a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));old=np.load(R/'crossplatform/round142/method'/name/'samples_PRIVATE'/(sid+'.npz'));assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1));rec=rd(D/'records_PRIVATE'/(sid+'.json'));assert rec['parameters_updated']==0
   for arm,p in zip(old['arm_names'],old['scores']):assert np.array_equal(p,ps[arm])
   means=np.load(R/'crossplatform/round142/supplementary_mean'/name/(sid+'.npz'))
   for arm,p in zip(means['arm_names'],means['scores']):assert np.array_equal(p,ps[arm])
   if e['has_B']:
    nb+=1;p=np.load(D/'position_PRIVATE'/(sid+'.npz'));z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));m=met[src,tgt,sid];delta=m['delta'];assert delta==hh[src]-m['q10'] and np.array_equal(p['physical_xyz'],z['xyz']) and np.array_equal(p['original_positions'],np.stack([z['position'+str(i)] for i in range(6)]));rt=tensors(dict(np.load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'))));ff=np.load(D/'features_PRIVATE'/(sid+'.npz'))
    for arm,sign in [('aligned_reference',1),('opposite_reference',-1)]:
     x=z['xyz'].copy();x[:,:,2]+=sign*delta;assert np.array_equal(x,p[arm+'_pos_input']);expected=embed(x,PW);err=float(abs(expected-p[arm+'_pos0']).max());mxpos=max(mxpos,err);assert err<1e-4
     pp=p['original_positions'].copy();pp[:,:,:,2]+=sign*delta;assert np.array_equal(pp,p[arm+'_positions']);a0=tensors({k:ff[arm+'_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']});prob=read135(a0,rt,W);err=float(abs(prob-ps[arm]).max());mxp=max(mxp,err);assert err<5e-6
    assert np.array_equal(ps['half_zero_aligned'],(ps['F_RGBB_only']+ps['aligned_reference'])/2)
   else:
    for arm in ['aligned_reference','opposite_reference','half_zero_aligned']:assert np.array_equal(ps[arm],ps['Frozen_A'])
   n+=1
   if n%128==0:print('AUDIT144',n,nb,flush=True)
 assert n==len(reg);out={'status':'PASS','n':n,'B':nb,'max_CPU_positional_MLP_error':mxpos,'max_CPU_full_reader_error':mxp,'all_references_nobox_noB_exact':True,'parameter_updates':0,'fullTransformer_not_independently_recomputed_CPU':True,'FINAL_access':False};dump(H/('ENGINEERING_AUDIT.json' if args.engineering else 'AUDIT.json'),out);print(out,flush=True)
if __name__=='__main__':main()
