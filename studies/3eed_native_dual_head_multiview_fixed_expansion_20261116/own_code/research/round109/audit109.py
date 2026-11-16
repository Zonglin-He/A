"""Independent CPU native replay, sensor provenance, all negatives and fallbacks."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
R=Path(__file__).resolve().parents[2];H=R/'research/round109';B=R/'crossplatform/round109';sys.path[:0]=[str(R/'research/round108'),str(R/'research/round105')];from replay108 import ce_cpu108
from audit105 import replay
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 torch.set_num_threads(4);matrix=rd(B/'MATRIX_SEALED.json');assert matrix['n']==1536 and len(matrix['arms'])==33;assert rd(H/'BRIDGE.json')['n']==6;n=ends=alias=0;mx=0.;coverage=[];paths={'waymo':R/'crossplatform/round91/trained_source_attempt001/states_PRIVATE/R76_full_disc_e1.npz','drone':R/'crossplatform/round97/drone/SELECTED_SOURCE_ONLY.npz','quad':R/'crossplatform/round97/quad/SELECTED_SOURCE_ONLY.npz'};sources=rd(R/'research/round97/experiment/DESIGN_LOCK.json')['sources'];anchor={x['A']['metapath']:x for x in rd(H/'ANCHORS_PRIVATE.json')};pairs={x['pair_PRIVATE']:x for x in rd(H/'PAIRS_PRIVATE.json')};regseal=rd(B/'registration_PRIVATE/SEALED.json')
 for p,h in regseal['files'].items():assert sha(B/'registration_PRIVATE'/p)==h
 soft=lambda x:np.exp(x-x.max())/np.exp(x-x.max()).sum()
 def pool(p,q,sid):
  perm=np.random.default_rng(int(hashlib.sha256(('R94querycontrol|'+sid).encode()).hexdigest()[:16],16)).permutation(256);return {'softtoken':q,'softtoken_geometric':soft((np.log(p)+np.log(q))/2),'softtoken_arithmetic':(p+q)/2,'softtoken_shuffled_geometric':soft((np.log(p)+np.log(q[perm]))/2)}
 for name,hs in matrix['directions'].items():
  src,_,tgt=name.split('_');O=B/name;assert sha(O/'SEALED.json')==hs
  for p,h in rd(O/'SEALED.json')['files'].items():assert sha(O/p)==h
  S=dict(np.load(paths[src]));c=next(x for x in sources if x['source']==src);assert sha(c['checkpoint_path'])==c['checkpoint_sha256'];st=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True);orig={k[len('prediction_heads.5.sem_cls_scores_head.'):]:v.numpy() for k,v in st['model'].items() if k.startswith('prediction_heads.5.sem_cls_scores_head.')};CE=dict(np.load(B/(src+'_CE_WEIGHTS_PRIVATE.npz')));assert set(CE)==set(orig) and all(np.array_equal(v,orig[k]) for k,v in CE.items());del st
  rows=rd(H/f'input/TARGET_{tgt.upper()}256_PRIVATE.json');dev={str(Path(r['metapath']).resolve()) for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')};assert not dev&{str(Path(r['metapath']).resolve()) for r in rows};counts={'source':src,'target':tgt,'n':len(rows),'protected_alias_fallback':0,'families':{k:{'B_available':0,'C_available':0,'any_extra':0} for k in ['real','randomA','wrong']}}
  for r in rows:
   sid=r['id'];out=np.load(O/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(out['arm_names'],out['scores']));old=np.load(R/'crossplatform/round100'/name/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(old['arm_names'],old['scores']));rec=rd(O/'records_PRIVATE'/(sid+'.json'));ar=anchor[str(Path(r['metapath']).resolve())];n+=1;assert np.array_equal(out['source_boxes'],old['source_boxes']) and np.array_equal(out['queries'],out['scores'].argmax(1));assert all(np.array_equal(p[k],po[k]) for k in ['original_source','prepared_frozen','frozen_four','direct_FC','joint3_FC']);assert not rec['fits']
   def ck(a,b):
    nonlocal ends,mx
    e=float(abs(a-b).max());mx=max(mx,e);assert e<5e-6 and a.argmax()==b.argmax(),(name,sid,e);ends+=1
   if rec['protected_alias_fallback']:
    alias+=1;counts['protected_alias_fallback']+=1;assert not rec['families'] and not (O/'features_PRIVATE'/(sid+'.npz')).exists();assert not ar['A']['eligible_for_extra_media'];assert all(np.array_equal(v,po['prepared_frozen']) for k,v in p.items() if k not in po);continue
   with np.load(O/'features_PRIVATE'/(sid+'.npz')) as z:
    fs={k:{v:{f:z[k+'_'+v+'_'+f] for f in ['h','u','late','mask','r','p']} for v in ['A','B','C','fusedABC']} for k in ['real','randomA','wrong']}
   for k,vv in fs.items():
    probs={v:replay(f,S) for v,f in vv.items()}
    for v,f in vv.items():ck(probs[v],f['p'])
    ck(probs['A'],po['prepared_frozen'])
    for v in ['B','C','fusedABC']:ck(probs[v],p[k+'_BBF_'+v])
    ck(np.mean([probs[v] for v in ['A','B','C']],0),p[k+'_BBF_ensemble']);f=vv['fusedABC'];q,_=ce_cpu108(f['h'],f['mask'],CE)
    for a,x in pool(probs['fusedABC'],q,sid).items():ck(x,p[k+'_'+a+'_fusedABC'])
    fam=rec['families'][k];counts['families'][k]['any_extra']+=fam['B_available'] or fam['C_available']
    for slot in ['B','C']:
     counts['families'][k][slot+'_available']+=fam[slot+'_available'];pid=ar[slot+'_pair'];pf=fam['support'][slot]
     if pid and k!='wrong':
      regpath=B/'registration_PRIVATE'/(pid+'.json');reg=rd(regpath)
      if fam[slot+'_available']:
       sensor=pairs[pid]['B'] if k=='real' else ar['A'];assert reg['geometry_cycle_supported'] and pf['registration_sha256']==sha(regpath);assert pf['point_sha256']==reg['B_sensor' if k=='real' else 'A_sensor']['point_sha256'];assert pf['aligned_seed_n']>0 and pf['sample_points']==16384
      elif not reg['geometry_cycle_supported']:assert pf['status']==reg['status']
     if k=='wrong':assert fam[slot+'_available']==rec['families']['real'][slot+'_available']
   f=fs['real']['A'];q,_=ce_cpu108(f['h'],f['mask'],CE)
   for a,x in pool(f['p'],q,sid).items():ck(x,p[a+'_A'])
  coverage.append(counts);print('AUDIT109',name,flush=True)
 dump(H/'NATIVE_AUDIT.json',{'state':'PASS_NATIVE_CPU_ALL_FROZEN_HEADS_CHECKPOINTS_A256_REFERENCES_SENSOR_PROVENANCE_FALLBACKS','n':n,'arms':33,'endpoints':ends,'max_CPU_native_probability_error':mx,'protected_alias_fallback':alias,'new_bridge_cases':6,'reused_R108_full_native_head_engineering':12,'GT_used':False,'FINAL_access':False});dump(H/'COVERAGE_SUMMARY.json',{'groups':coverage,'all_1536_retained':True});print('AUDIT109_PASS',n,ends)
if __name__=='__main__':main()
