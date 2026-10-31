"""R13 sealed9arm exactmerge/no-label audit, never reads target annotations."""
from pathlib import Path
import argparse,collections,hashlib,json
import numpy as np
B=Path(__file__).resolve().parents[3]
ARMS=('source','A_full_consensus','A_full_order0','A_noun_consensus','A_deranged_consensus','B_inequality_full','B_inequality_noRGB','B_inequality_half','B_old_surface_equality')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify(folder):
 s=read(folder/'SEALED.json')
 for rel,h in s['files'].items():
  p=Path(rel);assert not p.is_absolute() and '..' not in p.parts;z=(folder/p).resolve();assert z.is_relative_to(folder.resolve()) and sha(z)==h
 return s

def main(a):
 assert sha(a.composite/'SEALED.json')==a.expected_seal_sha256
 s=verify(a.composite);p=read(a.plan);assert sha(a.plan)==s['plan_sha256'] and p['stage']=='R13_combine_sealed_direct_arms' and not p['GT_permitted']
 for rel,h in p['program_sha256'].items():assert sha(B/rel)==h
 for k in ('manifest','cases'):assert sha(p[k])==p[k+'_sha256']==s[k+'_sha256']
 A=Path(p['association_dir']);G=Path(p['geometry_dir']);sa=verify(A);sg=verify(G);assert sha(A/'SEALED.json')==p['association_seal_sha256']==s['association_seal_sha256'];assert sha(G/'SEALED.json')==p['geometry_seal_sha256']==s['geometry_seal_sha256']
 aaudit=read(a.association_audit);gaudit=read(a.geometry_audit);assert aaudit['state']=='PASS_R13_FORMAL116_ASSOCIATION_UNLABELLED_AUDIT' and aaudit['association_seal_sha256']==sha(A/'SEALED.json');assert gaudit['state']=='PASS_R13_ACTUAL116_FIXEDQ0_GEOMETRY_UNLABELLED_AUDIT' and gaudit['prediction_seal_sha256']==sha(G/'SEALED.json')
 assert s['state']=='sealed_R13_all116_9_direct_arms_before_GT' and s['n']==s['n_samples']==116 and s['n_arms']==9 and tuple(s['arm_names'])==ARMS and s['case_role_counts']=={'new_content108':108,'regression8':8}
 assert s['all_A_original_candidate_boxes'] and s['all_B_fixed_source_q0'] and s['all_direct_no_neural_update']
 rows=read(p['manifest']);cases=read(p['cases']);trace=read(a.composite/'TRACE_PRIVATE.json');assert len(rows)==len(cases)==len(trace)==116;assert {f.name for f in a.composite.glob('*.npz')}=={r['id']+'.npz' for r in rows}
 assert {str(f.relative_to(a.composite)) for f in a.composite.rglob('*') if f.is_file()}==set(s['files'])|{'SEALED.json'}
 counts=collections.defaultdict(collections.Counter);statuses=collections.defaultdict(collections.Counter);checks=0
 for row,case,tr in zip(rows,cases,trace):
  assert row['id']==case['id']==tr['id'];alias=case['case_alias'];role='new_content108' if case['role']=='new_content' else 'regression8';tag=role+'|'+row['platform'];counts[tag]['n']+=1
  ap=A/'cases'/(alias+'.npz');gp=G/(row['id']+'.npz');aj=read(A/'cases'/(alias+'.json'))
  with np.load(ap,allow_pickle=False) as z:x={k:z[k].copy() for k in z.files}
  with np.load(gp,allow_pickle=False) as z:y={k:z[k].copy() for k in z.files}
  with np.load(a.composite/(row['id']+'.npz'),allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
  q=int(y['q_source']);pool=y['source_boxes_raw6'];assert int(x['source_q0'])==q==int(o['q_source'])==int(y['source_text_scores256'].argmax());assert np.array_equal(pool,x['source_boxes_raw6']) and np.array_equal(x['source_text_scores256'],y['source_text_scores256'])
  assert str(o['sample_id'])==row['id'] and str(o['case_alias'])==alias and str(o['case_role'])==role;assert tuple(o['arm_names'])==ARMS and np.array_equal(o['source_boxes_raw6'],pool) and np.array_equal(o['source_text_scores256'],y['source_text_scores256'])
  expectedq=np.r_[q,x['arm_query_index'][[2,1,3,4]],np.full(4,q)].astype(np.int64);expectedbox=np.vstack([pool[q],x['arm_boxes_raw6'][[2,1,3,4]],y['arm_boxes_raw6'][1:]])
  assert expectedbox.dtype==np.float32 and expectedbox.shape==(9,6) and np.isfinite(expectedbox).all();assert np.array_equal(o['arm_query_index'],expectedq) and np.array_equal(o['arm_boxes_raw6'],expectedbox);assert np.array_equal(o['arm_boxes_raw6'][:5],pool[expectedq[:5]]) and np.array_equal(o['arm_boxes_raw6'][7],pool[q]+np.float32(.5)*(o['arm_boxes_raw6'][5]-pool[q]))
  ev=expectedbox.copy();ev[:,3:]=np.maximum(ev[:,3:],np.float32(1e-6));assert o['arm_boxes_eval6'].dtype==np.float32 and np.array_equal(o['arm_boxes_eval6'],ev)
  expectedstatus=['source',aj['consensus']['full']['readout_status'],aj['full_order0_raw']['readout_status'],aj['consensus']['noun']['readout_status'],aj['consensus']['deranged']['readout_status']]+y['arm_status'][1:].tolist();assert o['arm_status'].tolist()==expectedstatus
  active=np.zeros((9,6),bool);active[1:5]=expectedbox[1:5]!=pool[q];active[5:]=y['active_coordinate_mask'][1:];assert np.array_equal(o['active_coordinate_mask'],active) and np.array_equal(o['actual_changed_axes'],expectedbox!=pool[q])
  assert str(o['association_member_sha256'])==sha(ap)==tr['A_member_sha256'] and str(o['geometry_member_sha256'])==sha(gp)==tr['B_member_sha256'];assert tr['case_role']==role and tr['arm_query_index']==expectedq.tolist() and tr['arm_status']==expectedstatus
  sp=B/case['source_dir']/'SEALED.json';assert read(sp)['source_checkpoint_sha256']==s['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
  for j,name in enumerate(ARMS):counts[tag][name+'|query_changed']+=bool(expectedq[j]!=q);counts[tag][name+'|box_changed']+=bool((expectedbox[j]!=pool[q]).any());counts[tag][name+'|center_changed']+=bool((expectedbox[j,:3]!=pool[q,:3]).any());counts[tag][name+'|size_changed']+=bool((expectedbox[j,3:]!=pool[q,3:]).any());statuses[tag+'|'+name][expectedstatus[j]]+=1
  checks+=16
 out={'state':'PASS_R13_ACTUAL116_NINEARM_COMPOSITE_UNLABELLED_AUDIT','composite_seal_sha256':sha(a.composite/'SEALED.json'),'plan_sha256':sha(a.plan),'auditor_sha256':sha(Path(__file__)),'association_seal_sha256':sha(A/'SEALED.json'),'geometry_seal_sha256':sha(G/'SEALED.json'),'association_audit_sha256':sha(a.association_audit),'geometry_audit_sha256':sha(a.geometry_audit),'n':116,'n_arms':9,'finite_merge_checks':checks,'all_seal_members_programs_exact':True,'all9arms_match_original_separate_A_B_outputs_bitexact':True,'new108_regression8_roles_separate':True,'counts':{k:dict(v) for k,v in counts.items()},'status_histograms':{k:dict(v) for k,v in statuses.items()},'size_change_scope':'A sizechange selects another existingproposal; B sizechange is inequality/sourceprior result. These are not pooled as sizelearning.','no_GT_index_taskscore_new_task_IoU_VLM_G_optimization_model_GPU_or_predictionschanges':True,'issues':[]}
 assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'checks':checks,'sha256':sha(a.output)}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for x in ('composite','plan','association-audit','geometry-audit','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--expected-seal-sha256',required=True);main(p.parse_args())
