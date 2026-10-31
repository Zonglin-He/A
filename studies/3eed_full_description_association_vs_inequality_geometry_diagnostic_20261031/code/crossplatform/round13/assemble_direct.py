"""Combine separately sealed R13 identity and fixed-query geometry arms; no GT."""
from pathlib import Path
import argparse,json,sys
import numpy as np
D=Path(__file__).resolve().parent;sys.path[:0]=[str(D),str(D.parent/'round7/experiment')]
from input_io import B,read,sha,verify_seal
from runner_helpers import eval_boxes
ARMS=('source','A_full_consensus','A_full_order0','A_noun_consensus','A_deranged_consensus','B_inequality_full','B_inequality_noRGB','B_inequality_half','B_old_surface_equality')
def main(a):
 assert sha(a.plan)==a.plan_sha256;p=read(a.plan);assert p['stage']=='R13_combine_sealed_direct_arms' and not p['GT_permitted']
 for rel,h in p['program_sha256'].items():assert sha(B/rel)==h
 for key in ('manifest','cases'):assert sha(p[key])==p[key+'_sha256']
 rows=read(p['manifest']);cases=read(p['cases']);assert len(rows)==len(cases)==116
 A=Path(p['association_dir']);G=Path(p['geometry_dir']);sa=verify_seal(A);sg=verify_seal(G)
 assert sha(A/'SEALED.json')==p['association_seal_sha256'] and sha(G/'SEALED.json')==p['geometry_seal_sha256']
 assert sa['state']=='sealed_R13_all116_frozen_association6calls_before_GT' and sa['n_samples']==116 and sa['n_model_calls']==696 and sa['model_frozen_parameters_buffers_unchanged']
 assert sg['n']==116 and sg['all_q_fixed'] and sg['no_neural_update_or_GPU_or_GT']
 assert not a.output.exists();a.output.mkdir(parents=True);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes());files={};tr=[]
 for row,case in zip(rows,cases):
  assert row['id']==case['id'];alias=case['case_alias'];ap=A/'cases'/(alias+'.npz');gp=G/(row['id']+'.npz');aj=read(A/'cases'/(alias+'.json'))
  with np.load(ap,allow_pickle=False) as z:aa={k:z[k].copy() for k in z.files}
  with np.load(gp,allow_pickle=False) as z:gg={k:z[k].copy() for k in z.files}
  assert str(aa['sample_id'])==str(gg['sample_id'])==row['id']
  assert np.array_equal(aa['source_boxes_raw6'],gg['source_boxes_raw6']) and np.array_equal(aa['source_text_scores256'],gg['source_text_scores256'])
  q=int(gg['q_source']);assert q==int(aa['source_q0']);pool=gg['source_boxes_raw6']
  aq=aa['arm_query_index'];qs=np.r_[q,aq[[2,1,3,4]],np.full(4,q)].astype(np.int64)
  boxes=np.vstack([pool[q],aa['arm_boxes_raw6'][[2,1,3,4]],gg['arm_boxes_raw6'][1:]])
  assert boxes.shape==(9,6) and boxes.dtype==np.float32 and np.isfinite(boxes).all()
  assert np.array_equal(boxes[:5],pool[qs[:5]]) and np.array_equal(boxes[7],pool[q]+np.float32(.5)*(boxes[5]-pool[q]))
  st=['source',aj['consensus']['full']['readout_status'],aj['full_order0_raw']['readout_status'],aj['consensus']['noun']['readout_status'],aj['consensus']['deranged']['readout_status']]+gg['arm_status'][1:].tolist()
  role='new_content108' if case['role']=='new_content' else 'regression8';active=np.zeros((9,6),bool);active[1:5]=boxes[1:5]!=pool[q];active[5:]=gg['active_coordinate_mask'][1:]
  name=row['id']+'.npz';np.savez_compressed(a.output/name,sample_id=np.array(row['id']),case_alias=np.array(alias),case_role=np.array(role),arm_names=np.array(ARMS),arm_query_index=qs,arm_boxes_raw6=boxes,arm_boxes_eval6=eval_boxes(boxes),arm_status=np.array(st),source_boxes_raw6=pool,source_text_scores256=gg['source_text_scores256'],q_source=np.array(q),actual_changed_axes=boxes!=pool[q],active_coordinate_mask=active,association_member_sha256=np.array(sha(ap)),geometry_member_sha256=np.array(sha(gp)))
  files[name]=sha(a.output/name);tr.append({'id':row['id'],'case_alias':alias,'case_role':role,'platform':row['platform'],'scene_PRIVATE':row['scene'],'A_member_sha256':sha(ap),'B_member_sha256':sha(gp),'source_q0':q,'arm_query_index':qs.tolist(),'arm_status':st,'all_A_original_candidate_boxes':True,'all_B_fixed_source_q0':True})
 (a.output/'TRACE_PRIVATE.json').write_text(json.dumps(tr,indent=2)+'\n')
 for name in ('registered_plan.json','TRACE_PRIVATE.json'):files[name]=sha(a.output/name)
 source_sha=read(B/cases[0]['source_dir']/'SEALED.json')['source_checkpoint_sha256']
 assert all(read(B/case['source_dir']/'SEALED.json')['source_checkpoint_sha256']==source_sha for case in cases)
 s={'state':'sealed_R13_all116_9_direct_arms_before_GT','n':116,'n_samples':116,'n_arms':9,'source_checkpoint_sha256':source_sha,'arm_names':ARMS,'case_role_counts':{'new_content108':108,'regression8':8},'all_A_original_candidate_boxes':True,'all_B_fixed_source_q0':True,'all_direct_no_neural_update':True,'association_seal_sha256':sha(A/'SEALED.json'),'geometry_seal_sha256':sha(G/'SEALED.json'),'plan_sha256':sha(a.plan),'manifest_sha256':p['manifest_sha256'],'cases_sha256':p['cases_sha256'],'program_sha256':p['program_sha256'],'files':files}
 (a.output/'SEALED.json').write_text(json.dumps(s,indent=2)+'\n');print('R13_COMPOSITE_SEALED',sha(a.output/'SEALED.json'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
