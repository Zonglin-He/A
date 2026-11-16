"""Fixed architecture repair validation on the existing R109 expanded sensor cohort.
No per-input training, loss search or parameter selection on this cohort.
"""
from pathlib import Path
import sys
R117=Path(__file__).resolve().parents[2];H117=R117/'research/round117';B117=R117/'crossplatform/round117'
sys.path.insert(0,str(R117/'research/round114'));from core114 import *
sys.path.insert(0,str(R117/'research/round109'));from core109 import extra109,H109,B109
REF117=['original_source','prepared_frozen','real_BBF_fusedABC','real_BBF_ensemble','real_softtoken_geometric_fusedABC','real_softtoken_arithmetic_fusedABC']
NEW117=['native_'+v for v in ['A','C','fusedABC']]+[s+'_'+v for s in ['rOnly','lateOnly'] for v in ['A','fusedABC']]+[s+'_'+v for s in ['repaired','fixed_calibration','reverse_calibration'] for v in ['A','C','fusedABC']]+['repaired_ABC_mean']
ARMS117=REF117+NEW117;PRIMARY117='repaired_fusedABC';assert len(NEW117)==17 and len(ARMS117)==23

def calibration117():
 v=jread(R117/'crossplatform/round115/MATRIX_SEALED.json')['source_A_only_calibration'];return {x['source']:x['mean_delta_gate'] for x in v}

def extract117(base,m,S,src,r,anchors,pairs,engineering=False):
 name=src+'_to_'+r['platform'];sid=r['id'];ar=anchors[str(Path(r['metapath']).resolve())];assert ar['A']['eligible_for_extra_media'];restore(m,S)
 with torch.no_grad():z0,g,proof=capture98(base,r,src)
 old=np.load(B109/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(z0['boxes'],old['source_boxes']);nativeA=native_scores(torch.as_tensor(z0['query'],device='cuda'),torch.as_tensor(z0['tokens'],device='cuda'),torch.as_tensor(z0['token_mask'],device='cuda').bool()).double().softmax(0);err=float(abs(nativeA.cpu().numpy()-op['original_source']).max());assert err<5e-6
 z=tensor_prefix(z0);d=tensor_data(g,r);restore(m,S);rt,role=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);observations={};obsproof={};obarrays={};prior=jread(B109/name/'records_PRIVATE'/(sid+'.json'))
 # R109 did not save the canonical prefix/aligned PointNet arrays. Extract ONLY missing prerequisites.
 for slot in ['B','C']:
  pid=ar[slot+'_pair'];v,pf=extra109(base,ar['A'],pairs[pid],z0,'real',slot) if pid else (None,{'status':'no_'+slot+'_candidate'});observations[slot]=v;obsproof[slot]=pf;want=prior['families']['real'];assert (v is not None)==want[slot+'_available']
  if v is not None:
   assert all(pf[k]==want['support'][slot][k] for k in ['point_sha256','sample_sha256','registration_sha256','aligned_seed_n']);obarrays.update({slot+'_'+k:a.cpu().numpy() for k,a in v.items()})
 zz={'A':z,'B':prefix106(z,[observations['B']],False) if observations['B'] else z,'C':prefix106(z,[observations['C']],False) if observations['C'] else z,'fusedABC':prefix106(z,[x for x in observations.values() if x is not None],True)};checks={'source_original_probability_error':err,'original_boxes_fixed':True,'role_from_original_full_prepared_A':True,'C_available':observations['C'] is not None,'B_available':observations['B'] is not None}
 if engineering:
  with np.load(B109/name/'features_PRIVATE'/(sid+'.npz')) as ff,torch.no_grad():
   for view in ['A','fusedABC']:
    restore(m,S);f=feature106(m,base,zz[view],d,route,S,[]);assert np.array_equal(f['h'].cpu().numpy(),ff['real_'+view+'_h']) and np.array_equal(f['u'].cpu().numpy(),ff['real_'+view+'_u']);er=float(abs(f['p'].cpu().numpy()-ff['real_'+view+'_p']).max());assert er<1e-7;checks['old_prepared_'+view+'_hidden_bitexact']=True;checks['old_prepared_'+view+'_prob_error']=er
 return z0,z,d,zz,route,obsproof,obarrays,dict(proof,**checks),role

def evaluate117(base,m,S,src,r,z0,d,zz,route,calibration):
 anchor=initial113(m,S,True);features={};out={};cache={};checks=[]
 with torch.no_grad():
  for view,vz in zz.items():
   restore(m,anchor);f=feature91(m,base,vz,d,route,anchor);p,e,_=scorer91(m,f,route,anchor,'R76','full_disc');cache[view]=f
   if view=='A':assert np.array_equal(f['h'].cpu().numpy(),z0['h']) and np.array_equal(f['u'].cpu().numpy(),z0['tokens'])
   if view!='B':
    q=F.normalize(m.match(f['h'][None])[0],dim=-1);native=native_scores(q,f['u'],f['mask'].bool()).double().softmax(0);out['native_'+view]=native.cpu().numpy();out['repaired_'+view]=p.cpu().numpy()
    if view!='C':
     out['rOnly_'+view]=(native.log()+e['residual'].double()).softmax(0).cpu().numpy();out['lateOnly_'+view]=e['native_scores'].softmax(0).cpu().numpy()
    for k in ['h','u','late','mask']:features['native_'+view+'_'+k]=f[k].cpu().numpy()
    features['native_'+view+'_r']=e['residual'].cpu().numpy()
  out['repaired_ABC_mean']=torch.stack([scorer91(m,cache[v],route,anchor,'R76','full_disc')[0] for v in ['A','B','C']]).mean(0).cpu().numpy()
  for label,sgn in [('fixed_calibration',1),('reverse_calibration',-1)]:
   restore(m,anchor);m.gate_I.copy_(torch.as_tensor(anchor['gate_I'],device='cuda')+sgn*calibration)
   for view in ['A','C','fusedABC']:
    f=cache[view];aa={k:(f[k][None].cpu().numpy() if k in ['h','u'] else f[k].cpu().numpy()) for k in ['h','u','mask']};aa['r']=features['native_'+view+'_r'];p,late=fast114(m,zz[view],d,aa);out[label+'_'+view]=p.cpu().numpy();features[label+'_'+view+'_late']=late.cpu().numpy()
 restore(m,S);freeze91(m);assert set(out)==set(NEW117);return out,features,checks
