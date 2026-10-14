"""One prelocked cyclic reward shuffle; CPU shared-readout replay, no teacher/GT."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.stvg_temporal_caption_query_v1 import *
from scripts.stvg_temporal_shared_readout_v1 import fit

def main():
 sys.addaudithook(inference_guard);torch.set_num_threads(2);folder=BASE/'reward_control001';assert read(BASE/'STUDENT_GLOBAL_BARRIER.json')['count']==8;lock=read(folder/'LOCK.json');assert lock['source_sha256']==sha(Path(__file__))
 jobs=read(BASE/'CAPTION_JOB_LOCK.json')['records'];rows=[]
 for rec in roster():
  case,cat=catalog(rec);prior=BASE/'student'/f"{rec['case_id']}.pt";z=torch.load(prior,map_location='cpu',weights_only=False);local=[j for j in jobs if j['case_id']==rec['case_id']];r=[read(BASE/'teacher'/f"{rec['case_id']}_{j['candidate']['canonical_id']}_alignment.json")['reward'] for j in local]
  shifted=r[-1:]+r[:-1];result=fit(case,z['temporal_hidden'],[j['candidate'] for j in local],shifted);dst=folder/'outputs'/f"{rec['case_id']}.pt";dst.parent.mkdir(exist_ok=True);assert not dst.exists();torch.save(dict(fit=result,source_student_sha256=sha(prior),GT_read=False,reward_permutation='fixed_cyclic_right_shift_one'),dst);rows.append(dict(case_id=rec['case_id'],path=str(dst.relative_to(ROOT)),sha256=sha(dst),optimizer_steps=result['optimizer_steps'],GT_read=False))
 write(folder/'GLOBAL_BARRIER.json',dict(status='complete',records=rows,count=8,optimizer_steps=sum(r['optimizer_steps'] for r in rows),new_teacher_calls=0,new_model_forwards=0,CPU_only=True,GT_read=False,time=time.time()))
if __name__=='__main__':main()
