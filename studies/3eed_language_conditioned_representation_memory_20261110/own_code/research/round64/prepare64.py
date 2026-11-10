"""Frozen R59 source features, residuals and text keys; no task labels."""
import argparse,sys,time,torch
from common64 import *
sys.path.insert(0,str(ROOT/'research/round62'));from compute62 import cpu_head,readouts,unit
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);torch.set_grad_enabled(False);head=cpu_head(load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz'));out=ROOT/p['prepared_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();roles={}
 for r in read(ROOT/'research/round62/role_diagnostics_attempt001/MASKS_PRIVATE.json'):roles.setdefault(r['id_PRIVATE'],{})[r['role']]=np.asarray(r['mask'],bool)
 start=time.perf_counter();eligible=0;maxbridge=0.;rows=read(ROOT/p['manifest']);seals={k:read(ROOT/p[k]/'SEALED.json') for k in ['capture_root','association_root','r59_root']}
 for idx,row in enumerate(rows):
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz'
  for key in seals:assert sha(ROOT/p[key]/rel)==seals[key]['files'][rel]
  z=load(ROOT/p['capture_root']/rel);g=load(ROOT/p['association_root']/rel);old=load(ROOT/p['r59_root']/rel);hs={v:z[v+'_h'] for v in 'ABC'};abc,r59,parts=readouts(head,hs,z,g);expected=old['scores'][list(old['arm_names']).index('direct_all')];e=float(abs(r59-expected).max());maxbridge=max(maxbridge,e);assert e==0. and r59.argmax()==expected.argmax();qsrc={};payload={};subject=roles[sid]['subject'];offsets=z['token_offsets'];content=(offsets[:,1]>offsets[:,0])&(offsets[:,0]<len(row['caption']))&(offsets[:,1]<=len(row['caption']));context=content&~subject
  for v in 'ABC':
   raw=parts['raw'][v];fused=(1-parts['confidence'][v][:,None])*raw+parts['confidence'][v][:,None]*parts['targets'][v];qsrc[v]=torch.nn.functional.normalize(torch.tensor(raw),dim=-1).numpy();qf=torch.nn.functional.normalize(torch.tensor(np.asarray(fused,np.float32)),dim=-1).numpy();tokens=z[v+'_tokens'];ts=unit(tokens[subject].astype(float).mean(0)) if subject.any() else np.zeros(64);tc=unit(tokens[context].astype(float).mean(0)) if context.any() else ts.copy();payload.update({v+'_q':qsrc[v],v+'_fused_raw':fused,v+'_tokens':tokens,v+'_token_mask':z[v+'_token_mask'],v+'_subject_key':ts,v+'_context_key':tc})
   if v!='A':payload[v+'_W']=g[v+'_W'];payload[v+'_fixed_mass']=g[v+'_fixed_mass']
   if v=='A':payload['delta']=qf.astype(float)-qsrc[v].astype(float)
  mapped={}
  for v in 'BC':
   mass=g[v+'_W'].sum(1);mapped[v]=unit(np.divide(g[v+'_W']@qsrc[v].astype(float),mass[:,None],out=np.zeros((256,64)),where=mass[:,None]>0))
  cosA_B=np.sum(qsrc['A']*mapped['B'],1);cosA_C=np.sum(qsrc['A']*mapped['C'],1);cosB_C=np.sum(mapped['B']*mapped['C'],1);geom=np.minimum(g['B_affinity'].max(1),g['C_affinity'].max(1));confidence=geom*np.maximum(cosB_C,0);cfg=p['memory'];good=(geom>=cfg['geometry_affinity_min'])&(cosA_B>=cfg['source_cross_view_cos_min'])&(cosA_C>=cfg['source_cross_view_cos_min'])&(cosB_C>=cfg['source_cross_view_cos_min'])&(np.linalg.norm(payload['delta'],axis=1)>cfg['delta_nonzero_min'])&bool(subject.any());eligible+=int(good.sum());payload.update(eligible=good,reliability=confidence,source_boxes=z['A_boxes'],source_scores=z['A_scores'],base=g['base'],source_prob=old['scores'][list(old['arm_names']).index('source')],ABC=abc,R59=r59,subject_available=np.array(bool(subject.any())),context_available=np.array(bool(context.any())),physical_frame_sha=np.array(hashlib.sha256((row['pointpath']+'|'+row['imagepath']+'|'+row['metapath']).encode()).hexdigest()));np.savez_compressed(out/rel,**payload)
  if (idx+1)%64==0:print('PREPARE64',idx+1,336,flush=True)
 dump(out/'COSTS.json',{'n':336,'candidate_eligible_entries_before_dedup':eligible,'R59_max_bridge_error':maxbridge,'new_full_source_forwards':0,'seconds':time.perf_counter()-start});seal(out,'R64_FROZEN_SOURCE_KEYS_AND_R59_REPRESENTATION_RESIDUALS_NO_GT',n=336);print(read(out/'COSTS.json'))
if __name__=='__main__':main()
