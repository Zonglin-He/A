"""Additional independent input/parent bindings, cost export and final scalar curves.

Report-only addition: original frozen scientific code and committed arrays stay intact.
"""
from subset_common import *
import argparse
def unit_export(ds,name):
 import numpy as np,torch
 from stvg.decode import decode
 from stvg.teacher_bridge import evidence_key
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);validate();root=unit(ds,name);audit=read(root/'reporting_receipt.json');assert audit['status']=='verified_whole128_all_arrays_inputs_teacher_SHA_means';seal=read(root/'sealed.json');started=time.perf_counter();proof=[];totals={};teacher_keys={};parents=[]
 def add(prefix,value):
  if isinstance(value,dict):
   for k,v in value.items():add(prefix+'.'+str(k),v)
  elif isinstance(value,(int,float))and not isinstance(value,bool):totals[prefix]=totals.get(prefix,0.)+value
 for i,row in enumerate(rows(ds),1):
  e=seal['files'][row['id']];frames,ids=decode(row['metadata'],ds);view,meta=operators.apply(frames,ids,row['metadata'],ds,condition(name));assert hashlib.sha256(frames.tobytes()).hexdigest()==e['original_RGB_sha256']and hashlib.sha256(view.tobytes()).hexdigest()==e['transformed_RGB_sha256']and meta==e['transform'];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];c=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);assert list(ids)==list(c.get('frame_ids',ids))
  if e['kind']=='clean_identity':assert np.array_equal(view,frames);clean(ds,row)
  else:
   te=e['teacher'];assert sha(te['path'])==te['sha256']and sha(te['receipt'])==te['receipt_sha256'];rp=read(te['receipt']);key,ctx=evidence_key(view[c['positions8']],row['metadata']['caption'],positions=c['positions8'],frame_ids=ids,dataset=ds,fingerprint=fixed(parent.OLD/'shifts_plan.json')['runtime_fingerprint']);assert rp['key']==key and rp['context']==ctx and rp['sha256']==te['sha256'];teacher_keys[key]=dict(payload_sha256=te['sha256'],receipt_sha256=te['receipt_sha256'])
   if e['kind']=='legacy_prefix':
    old,oe=old_entry(ds,row['id'],condition(name)['name']);assert all(e['proof'][x]==oe[y]for x,y in [('old_prediction_sha256','sha256'),('old_compact_sha256','compact_sha256'),('old_receipt_sha256','receipt_sha256')]);assert np.array_equal(c['Source'],old['Source'])and np.array_equal(c['q0'],old['native_prior'])
  proof.append(dict(index=i,metadata_content_sha256=digest(row),prediction_sha256=e['sha256'],receipt_sha256=e['receipt_sha256'],original_RGB_sha256=e['original_RGB_sha256'],corrupted_RGB_sha256=e['transformed_RGB_sha256'],actual_transform_and_teacher_context_rederived=True))
  # Compatibility and selected timings are explicitly separate nested segments.
  for k in ['appearance_prefix_and_checkpoint_seconds','compatibility_and_selected_finish_seconds','selected_finish_seconds']:add(k,e['cost'].get(k))
  if 'capture'in e['cost']:add('capture',e['cost']['capture'])
  del frames,view,c;gc.collect()
 private=dict(status='verified_all128_actual_RGB_transform_parent_Source_teacher_context_bindings',queries=128,files=proof,teacher_payloads=teacher_keys,reporting_receipt_sha256=sha(root/'reporting_receipt.json'),worker=identity(),seconds=time.perf_counter()-started,new_GT=0,new_model=0,new_Source=0,new_expert=0,code_sha256=sha(Path(__file__)));atomic(root/'root_input_audit_v2.json',private)
 folder=P/name/ds;atomic(folder/'root_input_audit_v2.json',{k:v for k,v in private.items()if k not in ['files','teacher_payloads']});atomic(folder/'nested_stage_costs_v2.json',dict(actual_worker=seal['worker'],query_counts=seal['counts'],whole_prediction_seconds=seal['seconds'],nested_totals_do_not_sum=totals,unique_exact_teacher_payloads=len(teacher_keys),teacher_load_aggregate_scope='ReferDINO_load_seconds is a component of capture; resident teacher process startticks not separately recorded by this frozen interface, not invented. Current Source model load recorded per unit/PID/start. Original acquisition time remains nonzero; new-expert actual counts explicit.',original_cached_acquisition_nonzero=True,uncached_deployment=False,original_root_export_elapsed_seconds=None,missing_clock_not_zero=True))
 atomic(root/'root_supplement_receipt_v2.json',dict(status='complete_actual_all128_input_bindings_nested_cost_export',input_audit_sha256=sha(root/'root_input_audit_v2.json'),cost_sha256=sha(folder/'nested_stage_costs_v2.json'),code_sha256=sha(Path(__file__)),seconds=time.perf_counter()-started));atomic(folder/'root_supplement_receipt_v2.json',read(root/'root_supplement_receipt_v2.json'));print(ds,name,'root supplement complete',flush=True)
def figures():
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 validate();reports={}
 for ds in COUNTS:
  for c in conditions():
   root=unit(ds,c['id']);assert read(root/'root_export_receipt.json')['status']=='root_complete_signed_coverage_cost_analysis_pending_publication';assert read(root/'root_supplement_receipt_v2.json')['status']=='complete_actual_all128_input_bindings_nested_cost_export';reports[ds,c['id']]=read(P/c['id']/ds/'report.json')
 fig,axes=plt.subplots(2,5,figsize=(18,7.2),sharex=True)
 for j,family in enumerate(operators.FAMILIES):
  for i,ds in enumerate(COUNTS):
   ax=axes[i,j]
   for pair,color,label in [('FullMAP-Source','#1266a0','Current - corrupt Source'),('FullMAP-CleanFullMAP','#b14825','Current - clean Current')]:
    vals=[reports[ds,f'{family}_p{x:g}'.replace('.','p')]['paired'][pair]['v']for x in [2.5,5.,10.]];means=np.array([v['mean']for v in vals]);lo=np.array([v['CI95'][0]for v in vals]);hi=np.array([v['CI95'][1]for v in vals]);ax.errorbar([2.5,5,10],means,yerr=np.stack([means-lo,hi-means]),marker='o',color=color,label=label,capsize=3)
   ax.axhline(0,color='gray',linewidth=.7);ax.set_title(ds+' / '+family);ax.set_xticks([2.5,5,10]);ax.grid(alpha=.2);ax.set_xlabel('Physical coverage (%)')
   if j==0:ax.set_ylabel('m_vIoU difference (pp)')
 axes[0,0].legend(fontsize=8);fig.suptitle('Fixed128 parents/domain: simulated physical bursts, pointwise paired-parent95% CIs\nCoverage is not equal physical magnitude across families; settings unchanged',fontsize=12);fig.tight_layout(rect=(0,0,1,.92));out=D/'figures';out.mkdir(parents=True,exist_ok=True);png=out/'corruption_subset_severity_v2.png';svg=out/'corruption_subset_severity_v2.svg';fig.savefig(png,dpi=150);fig.savefig(svg);plt.close(fig);atomic(out/'render_receipt_v2.json',dict(PNG_sha256=sha(png),SVG_sha256=sha(svg),actual_root_view_image_still_required_before_public_SVG=True,queries_per_point=128,parents_per_point=128,all18x2_root_exports_checked=True));print(png,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['unit','figures']);p.add_argument('--dataset',choices=COUNTS);p.add_argument('--condition');a=p.parse_args();unit_export(a.dataset,a.condition)if a.command=='unit'else figures()
