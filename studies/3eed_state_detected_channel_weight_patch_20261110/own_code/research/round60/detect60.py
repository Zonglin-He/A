"""Unlabeled internal-state detector pass, no prediction/task metric selection."""
import argparse,time,torch
from common60 import *
import mechanism60 as m

def get_inputs(row,p):
 sid=row['id'];root=ROOT/p['feature_root'];ss=read(root/'SEALED.json');rel='targets_PRIVATE/'+sid+'.npz';assert sha(root/rel)==ss['files'][rel]
 # Read only original prefix activations/text and source parameters, never R59
 # target vectors, fitted parameters, task labels, or previous per-case gains.
 f=load(root/rel,[v+'_'+key for v in 'ABC' for key in ['X','tokens','mask']]+['M0'])
 a=ROOT/p['association_root'];s=read(a/'SEALED.json');rel='samples_PRIVATE/'+sid+'.npz';assert sha(a/rel)==s['files'][rel];g=load(a/rel)
 return f,g

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);out=ROOT/p['detection_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();rows=read(ROOT/p['manifest']);start=time.perf_counter();allg={m:[] for m in MECHANISMS};diags=[]
 try:
  for i,row in enumerate(rows):
   f,g=get_inputs(row,p);M=f['M0'];gates,diag=m.detect(f,g,M[:,:288],M[:,288]);np.savez_compressed(out/'samples_PRIVATE'/(row['id']+'.npz'),**{z+'_gate':gates[z] for z in MECHANISMS},**diag)
   for z in MECHANISMS:allg[z].append(gates[z])
   diags.append({'platform':row['platform'],'support_mass':diag['support_mass'],'supported_anchor_count':diag['supported_anchor_count'],**{z+'_columns':int(np.count_nonzero(gates[z])) for z in MECHANISMS}})
  # A single shared fixed patch is label-free target calibration, explicitly
  # different from the current-input dynamic method.
  fixed={}
  for z in MECHANISMS:
   mean=np.mean(allg[z],0);idx=np.argsort(-mean,kind='stable')[:m.TOPK];fixed[z]=np.zeros(288);fixed[z][idx]=mean[idx]
  np.savez_compressed(out/'FIXED_PATCHES_PRIVATE.npz',**fixed)
  donor=np.arange(len(rows))
  for pl in ['drone','quad']:
   idx=np.array([i for i,r in enumerate(rows) if r['platform']==pl]);rng=np.random.default_rng(60301 if pl=='drone' else 60302);order=rng.permutation(idx);donor[order]=np.roll(order,1)
  assert all(i!=j for i,j in enumerate(donor));dump(out/'PATCH_DONORS_PRIVATE.json',donor.tolist())
  summary={'state':'R60_TWO_DETERMINISTIC_DETECTORS_FROZEN_BEFORE_TASK_SCORING','n':len(rows),'mechanisms':MECHANISMS,'revisions':0,'task_labels_read':False,'optimizer_calls':0,'solver_calls':0,'seconds':time.perf_counter()-start,'nonzero_patch_n':{z:sum(np.count_nonzero(q)>0 for q in allg[z]) for z in MECHANISMS},'median_support_count':float(np.median([d['supported_anchor_count'] for d in diags])),'median_selected_column_count':{z:float(np.median([np.count_nonzero(q) for q in allg[z]])) for z in MECHANISMS}}
  dump(out/'DETECTION_SUMMARY.json',summary);seal(out,summary['state'],n=len(rows),plan_sha256=sha(a.plan));print(summary,flush=True)
 except BaseException as error:dump(out/'FAILED.json',{'type':type(error).__name__,'error':str(error)});raise
if __name__=='__main__':main()
