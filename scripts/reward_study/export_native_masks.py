"""Diagnostic evidence export only: identical native teacher, no adaptation or GT."""
import os,sys,json,time,hashlib,random,datetime
from pathlib import Path
import numpy as np,torch
ROOT=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve();sys.path.insert(0,str(ROOT))
from stvg.runtime import setup,lease
from stvg.decode import decode
from stvg.teacher import SingleRefer
from stvg.teacher_worker import deny_detector
T=ROOT/'outputs/layout_reward_dev';O=T/'research/native_masks'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,z):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(z,indent=2)+'\n');os.replace(t,p)
def main():
 setup();sys.addaudithook(deny_detector);lock=lease();scope=read(T/'scope_hc2/scope.json');rows=[json.loads(s)for s in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if s.strip()];rows=[r for r in rows if r['id']in scope['ids']];assert len(rows)==128
 fingerprint=sha(__file__);write(O/'run.json',dict(status='running',queries=128,GT_read=False,diagnostic_only=True,script_sha256=fingerprint,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 torch.manual_seed(42);np.random.seed(42);teacher=SingleRefer();rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate());original=teacher.model.infer;captured={}
 def infer(*args,**kwargs):
  result=original(*args,**kwargs);samples=args[0];captured.update(raw_out={k:v.detach().float().cpu()for k,v in result.items()},nested_mask_all_false=not bool(samples.mask.any()),nested_mask_shape=list(samples.mask.shape),nested_tensor_shape=list(samples.tensors.shape));return result
 teacher.model.infer=infer;files={}
 for i,r in enumerate(rows,1):
  key=r['id'];dest=O/(key+'.pt');receipt=dest.with_suffix('.json');cp=T/'student_cache'/(key+'.pt');cc=read(cp.with_suffix('.json'));assert sha(cp)==cc['sha256']
  if receipt.exists():
   z=read(receipt);assert z['script_sha256']==fingerprint and z['sha256']==sha(dest);files[key]=dict(path=dest.name,sha256=z['sha256']);continue
  cache=torch.load(cp,map_location='cpu',weights_only=False);pos=cache['planned_positions'];expected=cache['teacher'];frames,ids=decode(r['metadata'],r['dataset']);observed=frames[pos];del frames
  evidence=read(T/'teacher_cache'/(expected['cache_key']+'.json'));assert hashlib.sha256(np.ascontiguousarray(observed).tobytes()).hexdigest()==evidence['context']['RGB_sha256']
  torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3]);captured.clear();begin=time.perf_counter();z=teacher(observed,r['metadata']['caption']);assert torch.equal(z['boxes'],expected['boxes']) and z['selected_query']==expected['selected_query'],'Native teacher changed'
  assert captured['nested_mask_all_false'];out=dict(**captured,teacher=z,selected_query=z['selected_query'],input_shape=z['input_shape'],positions8=pos,frame_ids=ids,source_cache_sha256=sha(cp),GT_read=False)
  tmp=dest.with_suffix('.pt.tmp');torch.save(out,tmp);os.replace(tmp,dest);write(receipt,dict(sha256=sha(dest),script_sha256=fingerprint,source_cache_sha256=out['source_cache_sha256'],old_RGB_sha256=evidence['context']['RGB_sha256'],native_selected_boxes_bit_exact=True,GT_read=False,seconds=time.perf_counter()-begin));files[key]=dict(path=dest.name,sha256=sha(dest))
  write(O/'progress.json',dict(completed=i,total=128));print(json.dumps(dict(completed=i,seconds=time.perf_counter()-begin,mask_shape=list(out['raw_out']['pred_masks'].shape))),flush=True);del out,cache,z,observed;captured.clear()
 assert len(files)==128;write(O/'sealed.json',dict(status='sealed_native_evidence_before_GT',queries=128,GT_read=False,script_sha256=fingerprint,scope_sha256=sha(T/'scope_hc2/scope.json'),files=files));write(O/'run.json',dict(status='completed_sealed',queries=128,GT_read=False,script_sha256=fingerprint,diagnostic_only=True));lock.close()
if __name__=='__main__':main()
