"""Independent graph/target/cache association and all native endpoint replay."""
from core86 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source
start=time.perf_counter();P=B86/'prediction366_attempt001';ss=read(P/'SEALED.json');E=B86/'evidence366_attempt002';es=read(E/'SEALED.json');oldseal=read(B85/'prediction366_attempt002/SEALED.json');base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));torch.set_num_threads(4);m=Fusion85(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=load(ROOT/'crossplatform/round76/trained_source_attempt002/states_PRIVATE/bidirectional_e06.npz');restore(m,S);maxerror=0.;targeterror=0.;endpoints=0;qualifications=0
with torch.no_grad():
 for nr,r in enumerate(read(H86/'experiment/ALL366_PRIVATE.json')):
  sid=r['id'];file=P/'samples_PRIVATE'/(sid+'.npz');assert sha(file)==ss['files'][file.relative_to(P).as_posix()];a=load(file);assert np.isfinite(a['scores']).all() and np.array_equal(a['queries'],a['scores'].argmax(-1));z0,z,d=item80(r);assert np.array_equal(z0['boxes'],a['source_boxes']) and np.array_equal(a['selected_original_boxes'],a['source_boxes'][a['queries']]);oldfile=B85/'prediction366_attempt002/samples_PRIVATE'/(sid+'.npz');assert sha(oldfile)==oldseal['files'][oldfile.relative_to(B85/'prediction366_attempt002').as_posix()];old=load(oldfile)
  evfile=E/'samples_PRIVATE'/(sid+'.npz');assert sha(evfile)==es['files'][evfile.relative_to(E).as_posix()];ev=load(evfile);meta=read(E/'cases_PRIVATE'/(sid+'.json'));factors=[{'kind':k,'support':s,'counter':c} for k,s,c in zip(meta['factor_kinds'],ev['support'],ev['counter'])]
  doc=read(ROOT/'crossplatform/round48/parse336_attempt001/dependencies_PRIVATE'/(sid+'.json')) if sid in DEV_IDS else read(ROOT/'crossplatform/round83/roles2304_attempt001/cases_PRIVATE'/(sid+'.json'))['document'];assert relations86(r['caption'],doc,meta['graph'])==meta['graph']['relations'];raw_ev=load(B85/'observations366_attempt001/samples_PRIVATE'/(sid+'.npz'));fcheck,_=factors86(meta['graph'],old['entity_scores'][[e['source_entity_index'] for e in meta['graph']['entities']]],raw_ev);assert len(fcheck)==len(factors)
  for x,y in zip(fcheck,factors):assert np.array_equal(x['support'],y['support']) and np.array_equal(x['counter'],y['counter'])
  restore(m,S);frozen=m.main(base,z,d,'bidirectional')[0]
  for b in ['S','F']:
   s0=torch.tensor(a[b+'_s0'],device='cuda');prior=s0.softmax(-1).cpu().numpy();pi,eligible,logs,known=target86(prior,factors,ev['labels'],sid);wrong,we,_,_=target86(prior,factors,ev['labels'],sid,True);targeterror=max(targeterror,float(abs(pi-a[b+'_pi']).max()));assert targeterror<1e-14 and np.array_equal(eligible,a[b+'_eligible']) and np.array_equal(we,a[b+'_mismatch_eligible']);assert np.array_equal(pi[~known],prior[~known]);assert all(x['unknown_probability_error']==0 for x in logs)
   for name,raw,el in [('direct',pi,eligible),('mismatch_direct',wrong,we)]:
    for rd,prob in [('raw',raw),('guard',guarded86(raw,el))]:assert float(abs(prob-a['scores'][ARMS86.index(b+'_'+name+'_'+rd)]).max())<1e-14
   for v,ks in [('candidate',[1,3]),('joint',[1,3]),('mismatch',[1,3]),('sum',[1])]:
    for k in ks:
     stem=b+'_'+v+str(k);sf=P/'states_PRIVATE'/(sid+'_'+stem+'.npz');assert sha(sf)==ss['files'][sf.relative_to(P).as_posix()];put_patch(m,load(sf));current=m.main(base,z,d,'bidirectional')[0];raw=anchored85(current,frozen,s0).cpu().numpy();el=we if v=='mismatch' else eligible
     for rd,prob in [('raw',raw),('guard',guarded86(raw,el))]:maxerror=max(maxerror,float(abs(prob-a['scores'][ARMS86.index(stem+'_'+rd)]).max()))
     assert maxerror<1e-10 and el[int(guarded86(raw,el).argmax())];qualifications+=1;endpoints+=1;restore(m,S)
  if (nr+1)%64==0:print('AUDIT86',nr+1,flush=True)
assert all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items());O=H86/'independent_native_audit_attempt001';O.mkdir();result={'state':'PASS_R86_ALL366_NATIVE_ENDPOINTS_DEPENDENCY_GRAPH_TRISTATE_TARGETS_UNKNOWN_MASS_AND_QUALIFIED_READOUT','n':366,'arm_names':ARMS86,'saved_endpoint_replays':endpoints,'qualified_readout_checks':qualifications,'max_probability_or_score_error':maxerror,'max_target_error':targeterror,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start,'new_task_scores':0};dump(O/'AUDIT_RESULT.json',result);print(result)
