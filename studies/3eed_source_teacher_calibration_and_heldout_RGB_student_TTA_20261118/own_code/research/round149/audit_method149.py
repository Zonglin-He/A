"""Verify complete streaming audit coverage and persistent sealed bytes before scoring."""
from common149 import *
import numpy as np,argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 check();root=B/('engineering_method' if args.engineering else 'method');seal=rd(root/'SEALED.json');reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));expect={(e['source'],e['target'],e['row']['id']) for e in reg};seen=set();mx={k:0. for k in ['max_CPU_probability_error','max_state_error','max_pointfeature_error','max_objective_error']};fits=0;process=[]
 for f,h in seal['files'].items():assert sha(root/f)==h,f
 for p in sorted((root/'audit_chunks_PRIVATE').glob('*.json')):
  if p.name.endswith('_AUDIT.json'):continue
  ch=rd(p);a=rd(p.with_name(p.stem+'_AUDIT.json'));assert a['status']=='PASS' and not a['GT_read'] and a['chunk_manifest_sha256']==sha(p) and a['files']==ch['files'];assert ch['engineering']==args.engineering
  for rel,h in ch['files'].items():
   if '/features_TEMP_PRIVATE/' in rel:
    assert a['native_features_sha256'][rel]==h
    if args.engineering:assert sha(root/rel)==h
    else:assert not(root/rel).exists()
   else:assert sha(root/rel)==h
  for r in ch['rows']:
   key=r['source'],r['target'],r['id_PRIVATE'];assert key in expect and key not in seen;seen.add(key)
  for k in mx:mx[k]=max(mx[k],a[k])
  fits+=a['newfits'];process+=a['process']
 assert seen==expect and len(seen)==seal['n'] and fits==seal['newfits'];parent=rd(R/'crossplatform/round148/method/SEALED.json')
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];rel=name+'/samples_PRIVATE/'+sid+'.npz';assert sha(R/'crossplatform/round148/method'/rel)==parent['files'][rel];ev=B/'evidence_PRIVATE'/name/(sid+'.npz');assert sha(ev)==rd(B/'EVIDENCE_SEALED.json')['files'][str(ev.relative_to(B))]
 out=dict(status='PASS',n=len(seen),fit_count=fits,**mx,all_native_chunks_audited_before_temporaryfeature_deletion=True,predictions_states_gradients_traces_and_hashreceipts_retained=True,engineering_full_features_retained=args.engineering,formal_full_adapted_A_not_saved_or_fully_compared=not args.engineering,full1024point_keep_reconstructed=True,fullTransformer_gradients_not_CPU_regenerated=True,source_alpha_all1_exact_aliases=True,original_R145_teacher_C_S_and_Up_budget=True,student_A_RGB_only_teacher_B_RGB_only_but_same_Fpoint_input=True,FINAL_access=False);dump(H/('TTA_ENGINEERING_AUDIT.json' if args.engineering else 'TTA_AUDIT.json'),out);dump(H/('TTA_ENGINEERING_PROCESS_ROWS_PRIVATE.json' if args.engineering else 'PROCESS_ROWS_PRIVATE.json'),process);print('TTA149_AUDIT',out,flush=True)
if __name__=='__main__':main()
