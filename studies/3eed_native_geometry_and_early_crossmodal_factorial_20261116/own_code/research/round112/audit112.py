"""Independent CPU head replay, full native equality and exact static edit-state audit."""
from core112 import *
def main():
 torch.set_num_threads(4);matrix=jread(B112/'MATRIX_SEALED.json');assert matrix['n']==768 and len(matrix['arms'])==89 and jread(H112/'ENGINEERING.json')['n']==12;n=ends=alias=0;mx=0.
 for src in ['waymo','drone','quad']:
  S=load(R112/('crossplatform/round91/trained_source_attempt001/states_PRIVATE/R76_full_disc_e1.npz' if src=='waymo' else 'crossplatform/round97/'+src+'/SELECTED_SOURCE_ONLY.npz'));head=load(B112/(src+'_HEAD_PRIVATE.npz'));priorhead=load(B111/(src+'_HEAD_NATIVE_PRIVATE.npz'));assert all(np.array_equal(head[k],v) for k,v in priorhead.items());assert jread(B112/(src+'_SOURCE_MODULE_SHA256_PRIVATE.json'))['learned_source_module_state_sha256']==hashlib.sha256(b''.join(k.encode()+v.tobytes() for k,v in sorted(S.items()))).hexdigest()
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=B112/name;assert sha(D/'SEALED.json')==matrix['directions'][name]
   for p,h in jread(D/'SEALED.json')['files'].items():assert sha(D/p)==h
   for r in jread(R112/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
    sid=r['id'];out=load(D/'samples_PRIVATE'/(sid+'.npz'));old=load(B111/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(out['arm_names'],out['scores']));po=dict(zip(old['arm_names'],old['scores']));rec=jread(D/'records_PRIVATE'/(sid+'.json'));n+=1;assert all(np.array_equal(ps[k],v) for k,v in po.items());assert np.array_equal(out['source_boxes'],old['source_boxes']) and np.array_equal(out['queries'],out['scores'].argmax(1));assert not rec['TTA_run'] and not rec['fits']
    if rec['protected_alias_fallback']:
     alias+=1;assert not rec['component_edits'] and all(np.array_equal(ps[k],po['prepared_frozen']) for k in NEW112);continue
    assert rec['same_wrong_visible_mask'] and rec['frozen_route_after_fullR91_before_edits'];assert rec['source_inputs']['no_new_sensor_read'];assert sha(B111/name/'prefix_PRIVATE'/(sid+'.npz'))==rec['source_inputs']['R111_prefix_sha256'] and sha(B111/name/'observations_PRIVATE'/(sid+'.npz'))==rec['source_inputs']['R111_observations_sha256'];f=load(D/'features_PRIVATE'/(sid+'.npz'));orig=load(R112/'crossplatform/round110/original_PRIVATE'/name/(sid+'.npz'));native=np.load(B111/name/'features_PRIVATE'/(sid+'.npz'))
    for label in NEW112:
     variant,binding,view=label.split('_',2);meta=rec['component_edits'][label];keys=[]
     if variant in ['noPA','neither']:keys+=['point_adapter.up.weight','point_adapter.up.bias']
     if variant in ['noEarlyRGB','neither']:keys+=['gate_P']
     assert set(keys)==set(meta['changed_tensor_names']);cnt=sum(int(np.count_nonzero(S[k])) for k in keys);norm=float(np.sqrt(sum(np.square(S[k].astype(float)).sum() for k in keys)));assert cnt==meta['changed_values'] and abs(norm-meta['parameter_L2'])<1e-12
     h,u,mask,late,residual=[f[label+'_'+k] for k in ['h','u','mask','late','r']];p=cpu112(h,u,mask,late,residual,head);e=float(abs(p-ps[label]).max());mx=max(mx,e);assert e<5e-6 and int(p.argmax())==int(ps[label].argmax());ends+=1;assert np.array_equal(f[label+'_p'],ps[label]);nh=orig['h'] if view=='A' else native['real_fusedABC_h'];nu=orig['tokens'] if view=='A' else native['real_fusedABC_tokens'];hm=float(np.sqrt(np.square((h-nh).astype(float)).mean()));um=float(np.sqrt(np.square((u-nu).astype(float)).mean()));assert abs(hm-meta['hidden_RMS_from_native'])<1e-12 and abs(um-meta['text_RMS_from_native'])<1e-12
     if variant=='neither':assert np.array_equal(h,nh) and np.array_equal(u,nu) and hm==um==0
    native.close()
   print('AUDIT112',name,flush=True)
 jdump(H112/'NATIVE_AUDIT.json',{'state':'PASS_FULL_SEALS_OLD75_REFS_CPU_NEW14_SOURCE_EDIT_SCOPE_NORMS_BOTHBYPASSED_NATIVE_ALL_INPUTS_FALLBACKS','n':n,'arms':89,'new_CPU_endpoints':ends,'max_CPU_probability_error':mx,'protected_alias_fallback':alias,'source_model_actual_functional_engineering':12,'TTA_run':False,'GT_used':False,'FINAL_access':False});print('AUDIT112_PASS',n,ends,mx)
if __name__=='__main__':main()
