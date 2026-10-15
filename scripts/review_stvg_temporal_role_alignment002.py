"""Independent pre-GT probability, binding and cached512 replay readback."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_role_alignment002 import *
def main(phase):
 import torch,numpy as np
 from scripts import stvg_table1_temporal_adapter_v1 as ta
 from scripts.stvg_evidence_temporal_fit_v1 import fit
 check_runtime();sys.addaudithook(guard);torch.set_num_threads(2);b=BASE/phase;tb=read(b/'TEACHER_GLOBAL_BARRIER.json');sb=read(b/'STUDENT_GLOBAL_BARRIER.json');assert tb['count']==28 and sb['count']==64 and tb['GT_read']is sb['GT_read']is False
 teacher={};files=[];prob_error=theta_error=0.;jobs={j['job_id']:j for j in read(BASE/'P1/TEACHER_JOB_LOCK.json')['records']};records={r['probe_alias']:r for r in roster()}
 for x in tb['files']:
  p=ROOT/x['path'];assert sha(p)==x['sha256'];z=torch.load(p,map_location='cpu',weights_only=False);cid=z['case_id'];rec=records[cid];assert is_question(rec)and len(z['slots'])==len(load_case(rec)[0]['temporal_candidates']) and z['GT_read']is False and z['runtime_sha256']==sha(BASE/'RUNTIME.json');old=p0_responses(rec)
  for i,row in enumerate(z['slots']):
   j=jobs.get(cid+'_C'+str(i));assert(row is None)==(j is None)
   if row is None:continue
   assert row['source_case_sha256']==j['source_case_sha256']and row['candidate_id']==j['candidate_id']and row['P0_teacher_sha256']==j['teacher_sha256']and row['query_sha256']==j['query_sha256']and row['P0_caption_sha256']==j['caption_sha256']and not row['GT_read']
   logits=np.asarray(row['label_logits'],float);p0=np.exp(logits-logits.max());p0/=p0.sum();prob_error=max(prob_error,float(np.max(np.abs(p0-np.asarray(row['probabilities'])))));assert prob_error<2e-7
   if row['reward']is None:assert row['caption_status']=='empty_description'
   else:assert abs(row['reward']-(p0[0]-p0[1]))<3e-7
   cap=old[i]['raw_caption']if phase=='P1'else row['raw_caption'];assert row['caption_sha256']==digest(cap)and row['alignment_text_sha256']==digest(alignment_prompt(query(rec),cap))
   assert row['metadata']==old[i]['metadata']
  teacher[cid]=z;files.append(x)
 assert set(teacher)=={r['probe_alias']for r in roster()if is_question(r)}
 for x in sb['files']:
  p=ROOT/x['path'];assert sha(p)==x['sha256'];z=torch.load(p,map_location='cpu',weights_only=False);cid=z['case_id'];rec=records[cid];case,cp=load_case(rec);assert rec['dataset']=='vidstg'and z['source_case_sha256']==case['source_case_sha256']and z['source_file_sha256']==sha(cp)and z['GT_read']is False and z['runtime_sha256']==sha(BASE/'RUNTIME.json')
  for a in ['F','T',phase]:assert np.array_equal(z['arms'][a]['boxes'],case['native_boxes_xyxy_normalized'])
  assert z['arms']['T']['interval']==p0_interval(rec)and z['arms']['F']['interval']==case['Native_interval']
  if is_question(rec):
   rr=[ta._alignment_reward(c['canonical_id'],r)if r is not None else None for c,r in zip(case['temporal_candidates'],teacher[cid]['slots'])];assert rr==z['rewards']
   if any(r is not None for r in rr):
    f=fit(case,ta._hidden_from_source_case(case),case['temporal_candidates'],rr,variant='offset_trust');assert f['final_interval']==z['fit']['final_interval']==z['arms'][phase]['interval']and f['active_parameters']==512
    er=float(torch.max(torch.abs(f['path'][-1]['theta']-z['fit']['path'][-1]['theta'])));theta_error=max(theta_error,er);assert er<1e-6
    old=p0_responses(rec);rewards=[ta._alignment_reward(c['canonical_id'],r)for c,r in zip(case['temporal_candidates'],old)];baseline=fit(case,ta._hidden_from_source_case(case),case['temporal_candidates'],rewards,variant='offset_trust');assert baseline['final_interval']==p0_interval(rec),'P0 exact replay mismatch'
   else:assert z['fit']is None and z['arms'][phase]['interval']==z['arms']['T']['interval']
  else:assert z['fit']is None and z['arms'][phase]['interval']==z['arms']['T']['interval']
  files.append(x)
 write(b/'ROOT_READBACK.json',dict(status='pass',scope='all64 predictions, probability/source/input bindings and actual same-interface CPU512 replay; not independent backward or new full GPU readout',count=64,question_count=28,probability_max_abs_error=prob_error,theta_replay_max_abs_error=theta_error,runtime_sha256=sha(BASE/'RUNTIME.json'),student_barrier_sha256=sha(b/'STUDENT_GLOBAL_BARRIER.json'),GT_read=False,CPU_only=True))
 for p in [BASE/'RUNTIME.json',b/'INPUT_LOCK.json',b/'TEACHER_JOB_LOCK.json',b/'TEACHER_GLOBAL_BARRIER.json',b/'STUDENT_GLOBAL_BARRIER.json',b/'ROOT_READBACK.json']:files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
 write(b/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=64,question_count=28,files=files,runtime_sha256=sha(BASE/'RUNTIME.json'),scorer_sha256=sha(ROOT/'scripts/measure_stvg_temporal_role_alignment002.py'),GT_read=False))
 print(phase,'root64 pass',prob_error,theta_error)
if __name__=='__main__':main(sys.argv[1])
