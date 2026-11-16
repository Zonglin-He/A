"""Independent contrast algebra and all source-preserving output replay; no GT."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round102';B=R/'crossplatform/round102';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def soft(z):z=z-z.max();return np.exp(z)/np.exp(z).sum()
def main():
 matrix=rd(B/'MATRIX_SEALED.json');n=values=0;maximum=0.;references=10
 for name,h in matrix['directions'].items():
  O=B/name;assert sha(O/'SEALED.json')==h
  for p,hashcode in rd(O/'SEALED.json')['files'].items():assert sha(O/p)==hashcode
  src,_,tgt=name.split('_');rows=rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');refseal=rd(R/'crossplatform/round101'/name/'SEALED.json')
  for row in rows:
   sid=row['id'];out=load(O/'samples_PRIVATE'/(sid+'.npz'));sf=R/'crossplatform/round101'/name/'samples_PRIVATE'/(sid+'.npz');assert sha(sf)==refseal['files']['samples_PRIVATE/'+sid+'.npz'];old=load(sf);assert np.array_equal(out['source_boxes'],old['source_boxes']);p=dict(zip(out['arm_names'],out['scores']));ref=dict(zip(old['arm_names'],old['scores']));assert list(out['arm_names'])==matrix['arms'];assert all(np.array_equal(p[k],ref[k]) for k in matrix['arms'][:references]);e=load(R/'crossplatform/round98'/name/'evidence_PRIVATE'/(sid+'.npz'));contrast=np.stack([np.mean(e['correct']-w,axis=0) for w in e['wrong']]);contrast-=contrast.mean(axis=1)[:,None];both=contrast[0]*contrast[1]>0;raw=np.sign(contrast[0])*np.minimum(np.abs(contrast[0]),np.abs(contrast[1]));raw[~both]=0.;common=raw-raw.mean();mean=contrast.mean(0);ratio=np.linalg.norm(common)/max(np.linalg.norm(mean),1e-30);perm=np.random.default_rng(int(hashlib.sha256(('R94querycontrol|'+sid).encode()).hexdigest()[:16],16)).permutation(256);shifts=[contrast[0],contrast[1],common,mean*ratio,common[perm]];frozen=load(R/'crossplatform/round98'/name/'samples_PRIVATE'/(sid+'.npz'));cc=dict(zip(frozen['arm_names'],frozen['scores']))['frozen_C'];pf=ref['prepared_frozen'];saved=load(O/'evidence_PRIVATE'/(sid+'.npz'));assert np.array_equal(saved['common_delta'],common) and np.array_equal(saved['contrasts'],contrast)
   for label,d in zip(['single0','single1','signcommon','scaledmean','shufflecommon'],shifts):
    F=soft(np.log(pf)+d);C=soft(np.log(cc)+d)
    for v,q in [('F',F),('C',C),('FC',(F+C)/2)]:
     err=float(abs(q-p[label+'_'+v]).max());maximum=max(maximum,err);assert err<1e-12 and q.argmax()==p[label+'_'+v].argmax();values+=1
   n+=1
 assert n==768 and values==11520;(H/'NATIVE_AUDIT.json').write_text(json.dumps({'state':'PASS_R102_ALL_CONTRASTS_SOURCE_BYTES_AND_11520_NEW_READOUTS','n':n,'new_readouts':values,'max_probability_error':maximum,'no_GT':True,'no_model_or_parameter_update':True,'FINAL_access':False},indent=2)+'\n');print('AUDIT102_PASS',n,values)
if __name__=='__main__':main()
