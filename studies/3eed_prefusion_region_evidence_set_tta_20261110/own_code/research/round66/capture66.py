"""Fresh pre-fusion source captures A/B only; no C/external outputs/GT."""
import argparse,sys,time,torch
from common66 import *
import interface66 as f,groups66 as g
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args();H=ROOT/'research/round66';p=read(H/'experiment/DESIGN_LOCK.json');rows=read(ROOT/p['manifest']);rows=[min([r for r in rows if r['platform']==pl],key=lambda r:hashlib.sha256(('R66engineering|'+r['id']).encode()).hexdigest()) for pl in ['drone','quad']] if a.engineering else rows;out=ROOT/('research/round66/engineering_capture_attempt001' if a.engineering else p['capture_root']);assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();model=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(model);start=time.perf_counter();calls=0;bridges=[]
 for i,row in enumerate(rows):
  sid=row['id'];old=load(ROOT/p['AB_root']/'samples_PRIVATE'/(sid+'.npz'));case=read(ROOT/p['AB_root']/'cases_PRIVATE'/(sid+'.json'));phrase=g.category(row['caption']);payload={'category_available':np.array(phrase is not None)}
  if phrase is not None:cl,crec=f.class_language(model,phrase)
  for v,seed in [('A',row['seed']),('B',case['B_seed'])]:
   points,indices,_=sample_points(dict(row,seed=seed));z,_=f.capture(model,points,indices,row['caption']);calls+=1
   for k in ['h','query','tokens','token_mask','boxes','scores','point_indices']:assert np.array_equal(z[k],old[v+'_'+k]),('source_bridge',v,k)
   if phrase is not None:z.update(cl)
   else:z.update(class_lang0=z['lang0'],class_textpad=z['textpad'],class_token_mask=z['token_mask'])
   if a.engineering:
    zero=f.replay(model,z)
    for k in ['h','query','tokens']:assert np.array_equal(zero[k],z[k]),('zero_fixed_interface',v,k,float(abs(zero[k]-z[k]).max()))
    # Real region, matched pool-external control, and leakage/attention checks.
    ids=np.argsort(-z['scores'],kind='stable');group=[np.array([j]) for j in ids[:4]];mm=g.masks(z['xyz'][0],z['boxes'],group,list(range(4)));chosen=next(((m,c) for m,c,ok in mm if ok),None)
    if chosen:
     mask,control=chosen;alter=z['vis0'].copy();alter[:,mask]=np.float32(1234.5);one=f.replay(model,z,mask);two=f.replay(model,z,mask,alternate_vis0=alter);assert np.array_equal(one['h'],two['h']) and np.array_equal(one['tokens'],two['tokens']);cc=f.replay(model,z,control);cat=f.replay(model,z,mask,category=True);bridges.append({'view':v,'region_seeds':int(mask.sum()),'control_seeds':int(control.sum()),'blocked_key_mass':one['proof']['max_blocked_attention'],'masked_value_replacement_exact':True,'full_category_effect_nonzero':float(abs(one['p']-cat['p']).max())})
   for k,val in z.items():payload[v+'_'+k]=val
  np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),**payload)
  if (i+1)%32==0:print('CAPTURE66',i+1,len(rows),flush=True)
 assert fingerprint(model)==fp;cost={'n':len(rows),'full_source_forwards':calls,'source_unchanged':True,'zero_interface_bitexact':a.engineering,'engineering_region_checks':bridges,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COSTS.json',cost);seal(out,'R66_A_B_PREFUSION_CAPTURE_NO_C_SUPERVISION',n=len(rows));print(cost)
if __name__=='__main__':main()
