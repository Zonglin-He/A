import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,json,hashlib,time
R=Path.cwd();sys.path.insert(0,str(R/'studies/shared_corrected_roi_CG_v1'))
from cg_common import *
from cg_core import replay,checked_parent
from stvg.runtime import guard
sys.addaudithook(guard)
import torch,numpy as np
torch.set_num_threads(1);p=validate();ds='hc2';records=[]
for rp in sorted((D/ds/'receipts').glob('*.json')):
 e=read(rp)
 if e['legacy_raw_reused']:continue
 key=rp.stem;entry=committed(ds,key);old,oe=checked_parent(ds,key);c=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);ac=e['acquisition'];assert c['proof']==e['proof'] and c['input_binding']==e['input_binding'] and not c['GT_read'] and not e['GT_read']
 assert e['new_expert']==0 and ac['new_expert']==0 and ac['actual_context_exact'] and ac['Source_checkpoint_before_teacher']
 assert e['proof']['Source_and_native_logits_and_q0_exact'] and e['proof']['old_profile_all_arrays_and_pool_exact'] and e['proof']['new_E_ROI_actual_raw_grid_pool']
 assert e['proof']['first_aux_DP_in_selected_path'] is False and e['proof']['final_DP_retained'] and c['first_aux_DP_in_selected_path'] is False
 assert np.array_equal(c['Source'],old['Source']) and np.array_equal(c['q0'],old['q0'])
 tp=Path(ac['teacher_path']);tm=read(tp.with_suffix('.json'));assert sha(tp)==ac['teacher_sha256']==ac['old_acquisition']['teacher_sha256']==tm['sha256']
 assert sha(tp.with_suffix('.json'))==ac['teacher_receipt_sha256'] and tm['key']==ac['teacher_key']==ac['old_acquisition']['teacher_key']
 raw=Path(ac['capture_path']);meta=raw.with_suffix('.json');checkpoint=read(meta);attempt=read(raw.parent/'attempt.json');disposed=read(raw.parent/'disposed.json')
 assert not raw.exists() and checkpoint['sha256']==ac['capture_sha256']==disposed['raw_sha256']
 assert checkpoint['Source_forwards']==1 and checkpoint['before_teacher_or_ROI_pool'] and not checkpoint['GT_read'] and attempt['before_Source']
 assert checkpoint['worker']==e['worker']==attempt['worker'] and checkpoint['input_binding']==e['input_binding']
 assert disposed['prediction_sha256']==e['sha256'] and disposed['compact_sha256']==e['compact_sha256'] and disposed['receipt_sha256']==sha(rp) and disposed['all_arrays_cold_replay_exact']
 assert meta.stat().st_mtime_ns<=rp.stat().st_mtime_ns<=(raw.parent/'disposed.json').stat().st_mtime_ns
 a=replay(c);b=replay(c)
 with np.load(D/e['path'])as z:assert set(a)==set(z.files) and all(np.array_equal(a[k],z[k])and np.array_equal(a[k],b[k])for k in a)
 hist=fixed(parent.D/f'CG_{ds}_sealed.json')['files'][key]
 with np.load(parent.D/hist['path'])as z:assert np.array_equal(a['HistoricalTwoDP'],z['FullMAP'])and np.array_equal(a['HistoricalTwoDP_interval'],z['FullMAP_interval'])
 records.append(dict(key=key,query_identity_sha256=hashlib.sha256(key.encode()).hexdigest(),receipt_sha256=sha(rp),compact_sha256=e['compact_sha256'],prediction_sha256=e['sha256'],Source_checkpoint_sha256=sha(meta),original_source_payload_sha256=ac['capture_sha256'],teacher_payload_sha256=ac['teacher_sha256'],teacher_receipt_sha256=ac['teacher_receipt_sha256'],disposed_sha256=sha(raw.parent/'disposed.json'),worker=e['worker'],all_array_keys_exact=len(a)))
 if len(records)==2:break
assert len(records)==2
private=dict(status='root_verified_two_actual_new_CG_Source_commits',fingerprint=p['fingerprint'],script_sha256=sha(Path(__file__)),cases=records,all_actualSource_checkpoint_before_teacher_and_pool=True,all_original_available_Source_native_logits_q0_pool_exact_recorded_before_disposal=True,all_actual_E_ROI_pool_runtime_proofs=True,all_compact_arrays_root_cold_replayed_twice_exact=True,all_owned_raw_disposal_after_commit=True,new_Source=0,new_expert=0,new_model=0,new_optimizer=0,new_GT_scoring=0,scope='Root inspects SHA-bound actual paid capture receipts and cold replays retained sufficient states. Actual raw-grid pooling and native-logit checks occurred in frozen worker before only owned raw disposal; root does not claim a second raw-grid repool after disposal. No performance metric or prefix accuracy read.',time=time.time())
atomic(D/'actual_new_Source_bridge_root_receipt.json',private)
public={k:v for k,v in private.items()if k!='cases'};public['case_count']=2;public['private_receipt_sha256']=sha(D/'actual_new_Source_bridge_root_receipt.json');public['cases']=[{k:v for k,v in row.items()if k!='key'}for row in records];atomic(P/'actual_new_Source_bridge_root_receipt.json',public);print('Two actual paid CG Source commits: checkpoints/context/proofs/all cold arrays/disposal root-verified; no new Source/model/GT')
