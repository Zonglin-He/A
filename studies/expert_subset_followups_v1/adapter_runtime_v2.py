"""Original adapter numerics; OnlineRefer vendor import version guards only."""
from adapter_v1 import *
import adapter_v1 as original
def construct(expert):
 audit=None
 if expert=='onlinerefer_r50':
  from online_compat_v1 import prepare
  audit=prepare(D/'vendor/OnlineRefer')
 model,tok,receipt=original.construct(expert)
 if audit is not None:receipt['import_only_compatibility']=audit
 receipt['original_adapter_code_sha256']=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()
 return model,tok,receipt
if __name__=='__main__':
 import argparse,os,torch
 p=argparse.ArgumentParser();p.add_argument('expert',choices=EXPERTS);q=p.parse_args();torch.set_num_threads(1);t=time.perf_counter();model,tok,rec=construct(q.expert);assert not torch.cuda.is_initialized();rec.update(actual_model_CPU_construction=1,actual_GPU_or_forward=0,elapsed_seconds=time.perf_counter()-t,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pid=os.getpid());(D/(q.expert+'_CPU_model_doctor.json')).write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
