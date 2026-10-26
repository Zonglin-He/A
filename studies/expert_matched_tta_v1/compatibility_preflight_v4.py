"""All historical receipt schemas, actual saved readouts, no inference/labels."""
import sys,importlib.util,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_expert_compat_common',ROOT/'common.py');c=importlib.util.module_from_spec(s);sys.modules['common']=c;s.loader.exec_module(c)
def run():
 import numpy as np,torch
 from stvg.readout import prediction
 from stvg.temporal import probabilities,mass_table
 from stvg.current import map_interval
 torch.set_num_threads(1);c.validate();c.rc.gt_guard();sys.path.insert(0,str(c.R/'studies/tta_full_v1'));import full_common as full
 index={ds:{}for ds in c.COUNTS};schema={};start=c.time.perf_counter()
 for ds,n in c.COUNTS.items():
  seal=c.read(c.T/f'{ds}_sealed.json');assert seal['queries']==n
  for key,e in seal['files'].items():
   assert not e['GT_read']
   for m in ['Tent','SAR']:
    t=e['traces'][m];assert t['official_source_unmodified']and t['reset_exact']and len(t['steps'])==5
   if 'all_readouts_cold_exact'in e:assert e['all_readouts_cold_exact']and all('logits_changed'in e['traces'][m]for m in ['Tent','SAR']);continue
   assert e['reused_completed'];assert 'logits_changed'not in e['traces']['Tent']and 'logits_changed'not in e['traces']['SAR']
   for f,h in [('path','sha256'),('logits_path','logits_sha256'),('receipt_path','receipt_sha256')]:assert c.sha(c.T/e[f])==e[h]
   log=torch.load(c.T/e['logits_path'],map_location='cpu',weights_only=False,mmap=True);native=log.get('native_logits')
   if native is None:
    context,_=full.payload_context(ds,key);assert context['frame_ids']==log['frame_ids'];native=[x['native_logits']for x in context['offsets']]
   changes={}
   with np.load(c.T/e['path'])as z:
    assert np.array_equal(z['Source'],log['Source'])
    for m in ['Tent','SAR']:
     aft=log['adapted_logits'][m];iv=prediction(aft,torch.as_tensor(log['Source']),log['records'],log['frame_ids'])['physical_interval'];_,q=mass_table(probabilities(log['records'],aft));assert np.array_equal(z[m+'Native_interval'],iv)and np.array_equal(z[m+'JointMAP_interval'],map_interval(log['frame_ids'],q));changes[m]=any(not torch.equal(a,b)for a,b in zip(aft,native))
   index[ds][key]=dict(sha256=e['sha256'],logits_sha256=e['logits_sha256'],receipt_sha256=e['receipt_sha256'],actual_saved_native_logits_readouts_exact=True,all_three_files_SHA_exact=True,logits_changed=changes)
  assert len(index[ds])==128;schema[ds]=dict(queries=n,legacy_reused=128,current_receipts=n-128)
 c.atomic(c.D/'legacy_schema_compatibility.json',index)
 rec=dict(status='passed_all12205_receipt_schemas_and256_actual_legacy_logits_native_joint_readouts',datasets=schema,new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0,elapsed_wall_seconds=c.time.perf_counter()-start,worker=c.identity(),code_sha256=c.sha(Path(__file__)))
 c.atomic(c.D/'compatibility_preflight_v4_receipt.json',rec);c.atomic(c.P/'compatibility_preflight_v4_receipt.json',rec);print(rec['status'])
if __name__=='__main__':run()
