"""Independent CPU replay of every new endpoint plus source provenance/seal invariants."""
from core111 import *
def main():
 torch.set_num_threads(4);matrix=jread(B111/'MATRIX_SEALED.json');assert matrix['n']==768 and len(matrix['arms'])==75;assert jread(H111/'ENGINEERING.json')['n']==12;n=ends=alias=mx=0;sup=0;reuse=0
 for src in ['waymo','drone','quad']:
  c=source97(src);assert sha(c['checkpoint_path'])==c['checkpoint_sha256'];v=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True);S={k.removeprefix('contrastive_align_projection_image.'):t.numpy() for k,t in v['model'].items() if k.startswith('contrastive_align_projection_image.')};del v;assert all(np.array_equal(S[k],v) for k,v in load(B111/(src+'_HEAD_NATIVE_PRIVATE.npz')).items());prep=load(R111/('crossplatform/round91/trained_source_attempt001/states_PRIVATE/R76_full_disc_e1.npz' if src=='waymo' else 'crossplatform/round97/'+src+'/SELECTED_SOURCE_ONLY.npz'));P={k.removeprefix('match.'):v for k,v in prep.items() if k.startswith('match.')};assert all(np.array_equal(P[k],v) for k,v in load(B111/(src+'_HEAD_PREPARED_PRIVATE.npz')).items())
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=B111/name;assert sha(D/'SEALED.json')==matrix['directions'][name]
   for f,h in jread(D/'SEALED.json')['files'].items():assert sha(D/f)==h
   for r in jread(R111/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
    sid=r['id'];out=load(D/'samples_PRIVATE'/(sid+'.npz'));old=load(R111/'crossplatform/round110'/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(out['arm_names'],out['scores']));po=dict(zip(old['arm_names'],old['scores']));rec=jread(D/'records_PRIVATE'/(sid+'.json'));n+=1;assert np.array_equal(out['source_boxes'],old['source_boxes']) and np.array_equal(out['queries'],out['scores'].argmax(1));assert all(np.array_equal(ps[k],v) for k,v in po.items());assert rec['no_TTA'] and not rec['fits']
    if rec['protected_alias_fallback']:
     alias+=1;assert all(np.array_equal(ps[k],po['prepared_frozen']) for k in ARMS111 if k not in po);continue
    z0=load(D/'prefix_PRIVATE'/(sid+'.npz'));O=load(R111/'crossplatform/round110/original_PRIVATE'/name/(sid+'.npz'));assert all(np.array_equal(z0[k],O[k]) for k in ['h','query','tokens','point_indices']);assert np.array_equal(z0['boxes'],out['source_boxes']);cache=load(D/'features_PRIVATE'/(sid+'.npz'));prepared=load(B106/name/'features_PRIVATE'/(sid+'.npz'));oldrec=jread(B106/name/'records_PRIVATE'/(sid+'.json'));reuse+=rec['readout_amended_cached_state_reuse']
    def ck(a,b):
     nonlocal mx,ends
     e=float(abs(a-b).max());mx=max(mx,e);assert e<5e-6 and int(a.argmax())==int(b.argmax()),(name,sid,e);ends+=1
    pp,_=cpu111(O['h'],O['tokens'],O['mask'],S);ck(pp,O['p']);expected={}
    for k in FAMILIES111:
     vv={}
     for view in ['B','C','fusedABC']:
      p,q=cpu111(cache[k+'_'+view+'_h'],cache[k+'_'+view+'_tokens'],cache[k+'_'+view+'_mask'],S);vv[view]=p;ck(p,cache[k+'_'+view+'_p']);expected['native_'+k+'_'+view]=p
     expected['native_'+k+'_ensemble']=(pp+vv['B']+vv['C'])/3;assert all(rec['families'][k][f]==oldrec['families'][k][f] for f in ['B_available','C_available']);sup+=rec['families'][k]['C_available']
    expected['sourceA_prepared_head']=cpu111(O['h'],O['tokens'],O['mask'],P)[0]
    for view,key in [('A','preparedA'),('fusedABC','preparedF')]:
     for head,state in [('native',S),('prepared',P)]:expected[key+'_'+head+'_head_base']=cpu111(prepared['real_'+view+'_h'],prepared['real_'+view+'_u'],prepared['real_'+view+'_mask'],state)[0]
    assert len(expected)==17
    for k,p in expected.items():ck(p,ps[k])
   print('AUDIT111',name,flush=True)
 jdump(H111/'NATIVE_AUDIT.json',{'state':'PASS_ALL_SEALS_ORIGINAL_PREFIX_SOURCE_HEAD_AUTHORITIES_NEW_ENDPOINTS_CPU_REPLAY_OLD_REFERENCES_FIXEDBOXES_ALL_FALLBACKS','n':n,'new_arms':17,'all_arms':75,'CPU_head_endpoints':ends,'max_CPU_probability_error':mx,'protected_alias_fallback':alias,'reused_failed_attempt_rows_without_backbone_or_decoder':reuse,'actual_C_flags_all_families':sup,'native_engineering':12,'GT_used':False,'FINAL_access':False});print('AUDIT111_PASS',n,ends,mx)
if __name__=='__main__':main()
