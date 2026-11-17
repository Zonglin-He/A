"""Independent source prediction/readout and registration-support audit, no labels."""
from common132 import *
import numpy as np,torch,sys
sys.path.insert(0,str(R132/'research/round129'));from audit129 import read
def main():
 torch.set_num_threads(4);reg=rd(H132/'SOURCE_CALIBRATION_INPUTS_PRIVATE.json');seal=rd(B132/'SOURCE_PREDICTIONS_SEALED.json');maximum=0.;n=0
 for p,h in rd(H132/'experiment/SOURCE_EXECUTION_LOCK.json')['files'].items():assert sha(R132/p)==h
 for src,h in seal['seals'].items():
  D=B132/'source_calibration_PRIVATE'/src;assert sha(D/'SEALED.json')==h
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
  with np.load(R132/f'crossplatform/round127/{src}_AUDIT_MODEL_PRIVATE.npz') as a:W={k[6:]:torch.from_numpy(a[k].copy()) for k in a.files if k.startswith('match.')}
  for e in [e for e in reg if e['entry']['source']==src]:
   entry=e['entry'];sid=entry['row']['id'];f=np.load(D/(sid+'.npz'));rec=rd(D/(sid+'.json'));z=np.load(R132/entry['files']['prefix']['path']);xyz=z['xyz'][0];unique,inv=np.unique(xyz,axis=0,return_inverse=True);valid=np.zeros(len(unique),bool);np.logical_or.at(valid,inv,f['valid_B']);boxes=z['boxes'];inside=(abs(unique[None]-boxes[:,None,:3])<=np.maximum(boxes[:,None,3:6],1e-6)/2).all(2);ass=f['assignment'];groups=[np.flatnonzero(ass==j) for j in range(ass.max()+1)];sup=np.array([inside[g].any(0) for g in groups]);na=sup.sum(1);nb=(sup&valid[None]).sum(1);assert np.array_equal(na,f['n_A']) and np.array_equal(nb,f['n_B']);S=(na>=4)&(nb>=4)&(nb/np.maximum(na,1)>=.5);assert np.array_equal(S,f['S']);assert rec['B_available']==bool(f['valid_B'].any())
   for v in ['A','B']:
    p=read(f[v+'_h'],f[v+'_u'],f['mask'],f['fixed_late'],f['fixed_r'],W);err=float(abs(p-f['p'+v]).max());maximum=max(maximum,err);assert err<5e-6
   n+=1
 assert n==len(reg);dump(H132/'SOURCE_NATIVE_AUDIT.json',{'status':'PASS_SOURCE_CPU_READOUT_REGISTERED_SUPPORT','n':n,'max_probability_error':maximum,'source_labels_read':False,'target_inputs_read':False});print(n,maximum)
if __name__=='__main__':main()
