"""Correct-checkpoint eight-frame teacher cache inventory, no model or labels."""
from common import *
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
 from stvg.runtime import guard
 from stvg.decode import decode
 from stvg.readout import prediction
 from stvg.positions import observation_positions
 from stvg.teacher_bridge import evidence_key
 sys.addaudithook(guard);C=R/'outputs/spatial_temporal_six_v1/source_validation';cohort=read(C.parent/'source_validation_cohort_v1.json');dest=D/'reliability';assert not(dest/'correct_source_teacher_inventory.json').exists()
 roots=[R.parent/'stvg-opd/outputs/boxfixed_full_v1/teacher_cache',R.parent/'stvg-mainline/outputs/tastvg_paper_v1/teacher_cache',F/'new_current_expert_cache']
 fingerprint=next(iter(fixed(D/'references.json')['hc2'].values()))['input_binding']['context']['fingerprint'];domains={};start=time.perf_counter()
 for target,cc in cohort.items():
  ds=cc['source_domain'];cases=[];seal=read(C/target/'sealed.json')
  for row in cc['rows']:
   resource(start);key=row['id'];e=seal['files'][key];assert sha(C/e['native_path'])==e['native_sha256'];n=torch.load(C/e['native_path'],map_location='cpu',weights_only=False,mmap=True)
   ids=n['frame_ids'];native=prediction(n['native_logits'],torch.as_tensor(n['Source']),n['records'],ids);pos=observation_positions(ids,native['indices']);frames,actualids=decode(row['metadata'],ds);assert actualids==ids
   evidence,context=evidence_key(frames[pos],row['metadata']['caption'],positions=pos,frame_ids=ids,fingerprint=fingerprint,dataset=ds);hits=[]
   for root in roots:
    p=root/(evidence+'.pt');mp=p.with_suffix('.json')
    if mp.exists():
     m=read(mp);assert m['context']==context and m['key']==evidence and p.exists()and sha(p)==m['sha256'];hits.append(dict(payload_path=str(p),payload_sha256=m['sha256'],receipt_path=str(mp),receipt_sha256=sha(mp)))
   cases.append(dict(id=key,source_domain=ds,target_model=target,native_Source_path=str(C/e['native_path']),native_Source_sha256=e['native_sha256'],positions8=pos,Source_native_interval=native['physical_interval'],evidence_key=evidence,context=context,existing_hits=hits,native_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),GT_read=False))
   atomic(dest/'source_inventory_progress.json',dict(target_model=target,completed=len(cases),total=36,cache_hits=sum(bool(x['existing_hits'])for x in cases),new_Source=0,new_expert=0,new_GT=0))
  domains[target]=dict(source_domain=ds,total=len(cases),cache_hits=sum(bool(x['existing_hits'])for x in cases),cases=cases)
 atomic(dest/'correct_source_teacher_inventory.json',dict(status='actual_correct_native_Source_policy_RGB_context_cache_checked',domains=domains,roots_checked=[str(x)for x in roots],new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_GT=0,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start));print({t:dict(total=v['total'],cache_hits=v['cache_hits'])for t,v in domains.items()},flush=True)
if __name__=='__main__':main()
