"""Actual T2-clock Full, old sealed matched baselines, postseal GT only."""
from pathlib import Path
import copy,gc,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_paper_t2_b_v1 import BASE,CORE,T2,NATIVE,read,put,sha,verify,configure

def main():
 import numpy as np,torch
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
 from scripts.score_stvg_spatial_controls256_v1 import validate_barrier
 from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
 verify();b=BASE/'arms/B/trials/G32';root=read(b/'ROOT_READBACK.json');bar=read(b/'GLOBAL_PREDICTION_BARRIER.json');assert root['status']=='pass' and root['count']==bar['count']==256
 files={f['case_id']:f for f in bar['files']};nb=read(NATIVE/'arms/B/trials/G32/GLOBAL_PREDICTION_BARRIER.json');native={f['case_id']:f for f in nb['files']};old=validate_barrier('B');missing={r['case_id'] for r in read(ROOT/'artifacts/stvg_spatial_caption_repair_v1/DIAGNOSIS.json')['records'] if r['cohort']=='core256'}
 for f in list(files.values())+list(native.values()):assert sha(ROOT/f['path'])==f['sha256']
 rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[];usage=[];invariants=[]
 for rec in rr:
  cid=rec['probe_alias'];case=cases[cid];d=torch.load(ROOT/files[cid]['path'],map_location='cpu',weights_only=False);n=torch.load(ROOT/native[cid]['path'],map_location='cpu',weights_only=False);o=torch.load(ROOT/old[cid]['path'],map_location='cpu',weights_only=False);tc=next(t['selected_interval'] for t in read(BASE/'TEMPORAL_LOCK.json')['records'] if t['case_id']==cid);assert d['arms']['T']['interval']==d['arms']['TS_M2']['interval']==tc and np.array_equal(d['arms']['F']['boxes'],n['arms']['F']['boxes']);e=copy.deepcopy(n['arms']['S_M2']);e['interval']=list(tc)
  preds={'Frozen':d['arms']['F'],'T2_only':d['arms']['T'],'S_B_Native':n['arms']['S_M2'],'Full_T2_B':d['arms']['TS_M2'],'T2_NativeB_evidence_diagnostic':e,'Full_T4_B_captionfixed':(n if cid in missing else o)['arms']['TS_M2']};common=[w['source_grid_position'] for w in read(BASE/'spatial/E1/proposals'/(cid+'.json'))['windows']];vals={}
  for name,p in preds.items():
   v,_=metric._score_student_arm(p,case,targets[cid],set(common),official,independent);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>=.3),R_vIoU_at_0p5=float(v['m_vIoU']>=.5));rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=v));vals[name]=v
  assert vals['S_B_Native']['m_tIoU']==vals['Frozen']['m_tIoU'];assert vals['Full_T2_B']['m_tIoU']==vals['T2_only']['m_tIoU']==vals['T2_NativeB_evidence_diagnostic']['m_tIoU'];assert vals['S_B_Native']['m_sIoU_dense_GT']==vals['T2_NativeB_evidence_diagnostic']['m_sIoU_dense_GT'];f=d['fits']['M2'];usage.append(dict(case_id=cid,dataset=rec['dataset'],actual_fit=f is not None,accepted_steps=f['optimizer_steps'] if f else 0,proposed_steps=f['proposed_optimizer_steps'] if f else 0,no_fit_reason='no_admitted_sparse_reference' if f is None else None,GT_read=False));invariants.append(dict(case_id=cid,Full_tIoU_equals_T2=True,SNative_tIoU_equals_Frozen=True,full_known_GT_geometry_support_shared=True,Native_evidence_composition_is_readout_diagnostic_not_new_fit=True));del d,n,o;gc.collect()
 summary,paired=subset_aggregate(rows);put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=256,per_domain=128,rows=rows,summary=summary,paired=paired,usage=usage,invariants=invariants,GT_bindings=bindings,GT_read=True,development_not_new_confirmation=True,T2_chosen_for_simplicity_not_noninferiority_proven=True,old_negative_confirmation_preserved=True,CURRENT_changed=False,no_midrun_GT_or_rule_change=True));put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=256,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True));print('FROZEN_PAPER_T2_B256_POSTSEAL_COMPLETE')
if __name__=='__main__':main()
