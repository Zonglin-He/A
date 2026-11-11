"""One frozen model, original ABC, SA3-only inference; no task labels."""
import argparse,sys,time,torch
from common67 import *
import sampling67 as s
sys.path.insert(0,str(ROOT/'research/round66'));import interface66 as inter
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args();H=ROOT/'research/round67';p=read(H/'experiment/DESIGN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);rows=[min([r for r in rows if r['platform']==pl],key=lambda r:hashlib.sha256(('R67eng|'+r['id']).encode()).hexdigest()) for pl in ['drone','quad']] if a.engineering else rows
 if not a.engineering:assert read(H/'engineering_attempt001/SEALED.json')['n']==2
 out=H/'engineering_attempt001' if a.engineering else ROOT/p['capture_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);torch.set_grad_enabled(False);start=time.perf_counter();count=0
 for i,r in enumerate(rows):
  sid=r['id'];ab=load(ROOT/p['AB_root']/'samples_PRIVATE'/(sid+'.npz'));c=load(ROOT/p['C_root']/'samples_PRIVATE'/(sid+'.npz'));seeds={'A':r['seed'],'B':read(ROOT/p['AB_root']/'cases_PRIVATE'/(sid+'.json'))['B_seed'],'C':seed(r)};payload={};stats={}
  for v in 'ABC':
   pts,inds,_=sample_points(dict(r,seed=seeds[v]));got={}
   def save(_m,aa):got['xyz']=aa[0].detach().clone();got['feat']=aa[1].detach().clone()
   hk=model.backbone_net.SA_modules[2].register_forward_pre_hook(save)
   try:z,_=inter.capture(model,pts,inds,r['caption']);count+=1
   finally:hk.remove()
   old={k:(c[k] if v=='C' else ab[v+'_'+k]) for k in ['h','query','tokens','token_mask','boxes','point_indices']}
   for k in old:assert np.array_equal(z[k],old[k]),('source_bridge',v,k)
   assert np.array_equal(got['xyz'].cpu().numpy(),z['xyz']);choices,st=s.choices(model.backbone_net.SA_modules[2],got['xyz'],got['feat'],sid+'|'+v);payload[v+'_SA2_xyz']=got['xyz'].cpu().numpy();payload[v+'_SA2_feat']=got['feat'].cpu().numpy()
   for k,x in z.items():payload[v+'_'+k]=x
   for mode in ['native','random','feature']:
    vis,ix=s.suffix(model,got['xyz'],got['feat'],None if mode=='native' else choices[mode]);rep=inter.replay(model,z,alternate_vis0=vis.cpu().numpy())
    if mode=='native':
     assert np.array_equal(vis.cpu().numpy(),z['vis0']),('native_backbone_suffix',v)
     for k in ['h','query','tokens']:assert np.array_equal(rep[k],z[k]),('zero_bridge',v,k)
    for k in ['h','query','tokens']:payload[mode+'_'+v+'_'+k]=rep[k]
    payload[mode+'_'+v+'_SA3_indices']=ix[0].cpu().numpy()
   stats[v]=st
  np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),**payload);dump(out/'cases_PRIVATE'/(sid+'.json'),stats)
  if (i+1)%16==0:print('RUN67',i+1,len(rows),count,flush=True)
 assert fingerprint(model)==fp;cost={'n':len(rows),'full_source_forwards':count,'suffix_replays':count*3,'source_unchanged':True,'all_native_suffix_and_fixed_decoder_bridges_bitexact':True,'all_original_1024_seed_positions_preserved':True,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'parameter_updates':0};dump(out/'COSTS.json',cost);seal(out,'R67_FIXED_SA2_SEEDS_AND_A256_SA3_POLICY_CAPTURES_NO_GT',n=len(rows));print(cost)
if __name__=='__main__':main()
