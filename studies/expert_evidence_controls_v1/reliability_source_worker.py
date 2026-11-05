"""Exactly72 new source-domain ReferDINO queries; reuse every Source state."""
from common import *
import traceback,select,gc
import reliability_math as math
from reliability_signals import signals
C=D/'reliability';OLD=R/'outputs/spatial_temporal_six_v1/source_validation'
def check_source():
 p=read(C/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
def source_cases(target):return fixed(C/'correct_source_teacher_inventory.json')['domains'][target]['cases']
def cohort_row(target,key):return next(r for r in fixed(OLD.parent/'source_validation_cohort_v1.json')[target]['rows']if r['id']==key)
def raw_inputs(target,case):
 from stvg.decode import decode
 from stvg.teacher_bridge import evidence_key
 row=cohort_row(target,case['id']);p=Path(case['native_Source_path']);assert sha(p)==case['native_Source_sha256'];native=torch.load(p,map_location='cpu',weights_only=False,mmap=True);assert not native['GT_read']
 frames,ids=decode(row['metadata'],case['source_domain']);assert ids==native['frame_ids']and hashlib.sha256(frames.tobytes()).hexdigest()==case['native_RGB_sha256'];key,context=evidence_key(frames[case['positions8']],row['metadata']['caption'],positions=case['positions8'],frame_ids=ids,fingerprint=case['context']['fingerprint'],dataset=case['source_domain']);assert key==case['evidence_key']and context==case['context'];return row,native,frames
def foreign_with_teacher(who):
 import importlib.util
 spec=importlib.util.spec_from_file_location('_source_reliability_protected_gate',R/'studies/expert_subset_followups_v1/protected_service_gate_runtime_v2.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate);proof=paused_owner();allowed={p['pid']for p in proof}
 if who is not None:
  st=Path(f"/proc/{who['pid']}/stat").read_text().split();cmd=Path(f"/proc/{who['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode();assert int(st[21])==who['startticks']and '-m studies.expert_evidence_controls_v1.reliability_teacher_entry --serve'in cmd;allowed.add(who['pid'])
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).splitlines():
  if not line.strip():continue
  pid,name=line.split(',',1);pid=int(pid)
  if pid not in allowed and pid!=os.getpid():gate.identity_from_proc(pid,name.strip())
def acquire():
 plan=check_source();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'and not(C/'source_worker.json').exists();lock=lease();start=time.perf_counter();torch.set_num_threads(1)
 from stvg.runtime import guard
 sys.addaudithook(guard);atomic(C/'source_worker.json',dict(**identity(),plan_fingerprint=plan['fingerprint']));counts=dict(new_Source=0,new_ReferDINO_queries=0,new_optimizer=0,committed=0);process=None;who=None;allfiles={};log=None
 try:
  for target in ['hc2','vidstg']:
   files={};first=[];dest=C/'source'/target
   for case in source_cases(target):
    resource(start);foreign_with_teacher(who);tick=time.perf_counter();key=case['id'];row,native,frames=raw_inputs(target,case);folder=dest/'paid'/key;assert not folder.exists();folder.mkdir(parents=True)
    binding=dict(source_case=case,native_Source_sha256=case['native_Source_sha256'],correct_native_Source_positions8=case['positions8'],GT_read=False);atomic(folder/'attempt.json',dict(before_new_expert=True,worker=identity(),input_binding=binding,new_Source=0,GT_read=False));request=folder/'request_RGB.pt';save(request,dict(frames=frames[case['positions8']],caption=row['metadata']['caption']));output=folder/'raw_native_expert.pt';assert not output.exists()
    if process is None:
     log=(C/'teacher_server.log').open('xb');process=subprocess.Popen([str(PYTHON),'-B','-m','studies.expert_evidence_controls_v1.reliability_teacher_entry','--serve'],cwd=R,env=os.environ.copy(),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1);who=identity(process.pid);atomic(C/'teacher_server.json',dict(**who,argv=['-m','studies.expert_evidence_controls_v1.reliability_teacher_entry','--serve'],backend='originalcudnnTF32True matmulFalse Torch4'))
    foreign_with_teacher(who);counts['new_ReferDINO_queries']+=1;assert counts['new_ReferDINO_queries']<=72
    atomic(folder/'actual_request_start.json',dict(teacher=who,request_SHA=sha(request),logical_query_counter=counts['new_ReferDINO_queries'],new_Source=0,GT_read=False));call=time.perf_counter();process.stdin.write(json.dumps(dict(input=str(request),output=str(output)))+'\n');process.stdin.flush();ready,_,_=select.select([process.stdout],[],[],180);assert ready,'Paid teacher timeout: preserve request/log/state, never retry';line=process.stdout.readline();assert line,'Teacher exited after request, preserve paid state';ack=json.loads(line);assert ack.get('ok'),ack
    wall=time.perf_counter()-call;assert output.exists();rawsha=sha(output)
    # Original native payload already persisted by teacher server BEFORE any
    # parent selection/ROI/geometric/numerical compatibility validation.
    atomic(folder/'expert_checkpoint.json',dict(raw_native_path=str(output.relative_to(D)),sha256=rawsha,checkpoint_before_parent_validation=True,input_binding=binding,teacher=who,request_wall_seconds=wall,GT_read=False))
    teacher=torch.load(output,map_location='cpu',weights_only=False,mmap=True);assert teacher['native_same_track_all8']and teacher['nested_mask_all_false']and int(np.argmax(np.asarray(teacher['native_query_scores'])))==int(teacher['selected_query'])
    arrays,compact=math.build(native,teacher,target,case['positions8']);feature=signals(native['Source'],native['native_logits'],native['records'],native['frame_ids'],case['positions8'],teacher,fixed(R/'configs/default.json')['datasets'][target]['E_calibration'])
    cp=folder/'compact.pt';save(cp,compact);again=math.replay(torch.load(cp,map_location='cpu',weights_only=False,mmap=True));assert set(arrays)==set(again)and all(np.array_equal(v,again[k])for k,v in arrays.items());out=dest/'predictions'/(key+'.npz');out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb')as f:np.savez_compressed(f,**arrays)
    with np.load(out)as z:assert all(np.array_equal(v,z[k])for k,v in arrays.items())
    record=dict(path=str(out.relative_to(D)),sha256=sha(out),compact_path=str(cp.relative_to(D)),compact_sha256=sha(cp),teacher_path=str(output.relative_to(D)),teacher_sha256=rawsha,expert_checkpoint_sha256=sha(folder/'expert_checkpoint.json'),request_path=str(request.relative_to(D)),request_sha256=sha(request),input_binding=binding,signals=feature,new_Source=0,new_ReferDINO_queries=1,new_optimizer=0,teacher=who,worker=identity(),request_wall_seconds=wall,native_expert_cost=teacher['cost'],model_load_seconds=teacher['model_load_seconds'],query_active_wall_seconds=time.perf_counter()-tick,GT_read=False,plan_fingerprint=plan['fingerprint']);rp=dest/'receipts'/(key+'.json');atomic(rp,record);files[key]=dict(record,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));first.append(key);counts['committed']+=1
    if len(first)==2:
     atomic(dest/'bridge_input.json',dict(keys=first.copy(),worker=identity(),plan_fingerprint=plan['fingerprint']));path=dest/'first2_bridge.log';t=time.perf_counter()
     with path.open('xb')as f:child=subprocess.Popen([str(PYTHON),'-B',str(S/'reliability_study.py'),'source_bridge','--dataset',target],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT);childwho=identity(child.pid);code=child.wait()
     atomic(dest/'bridge_actual_exit.json',dict(worker=childwho,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-t,log_sha256=sha(path)));assert code==0,'Preserve all paid teacher state on bridge failure'
    atomic(C/'source_progress.json',dict(target_model=target,source_domain=case['source_domain'],completed_in_domain=len(files),total_in_domain=36,counts=counts,worker=identity(),GT_read=False));del native,frames,teacher,arrays,compact,again;gc.collect()
   assert len(files)==36;atomic(dest/'sealed.json',dict(files=files,queries=36,GT_read=False,source_domain=source_cases(target)[0]['source_domain'],plan_fingerprint=plan['fingerprint']));allfiles[target]=files
  atomic(C/'source_sealed.json',dict(status='all72_correct_source_policy_expert_Full_predictions_signals_sealed_before_labels',domains={t:dict(queries=36,seal_sha256=sha(C/'source'/t/'sealed.json'))for t in allfiles},counts=counts,GT_read=False,worker=identity()));print('72correct-source expert queries sealed, 0newSource',flush=True)
 except BaseException:atomic(C/'source_failure.json',dict(error=traceback.format_exc(),counts=counts,worker=identity(),teacher=who,all_paid_request_native_reply_plan_and_log_preserved=True));raise
 finally:
  if process is not None:
   if process.poll()is None:
    try:process.stdin.write('{"stop":true}\n');process.stdin.flush();process.wait(timeout=30)
    except (BrokenPipeError,subprocess.TimeoutExpired):process.terminate();process.wait(timeout=15)
   atomic(C/'teacher_actual_exit.json',dict(worker=who,actual_subprocess_wait=True,exit_code=process.returncode,log_sha256=sha(C/'teacher_server.log')))
  if log:log.close()
  lock.close()
if __name__=='__main__':acquire()
