"""Fixed PANEL16 location diagnostic; same norm, correct/wrong, no optimizer."""
import os,sys,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import *
from scripts.desta3d_v3_privileged_ptd_qualification import B1,B1_SHA,ADAPTER_SHA,equal
ARMS=['original']+[f'{b}_{l}_{m}' for b in ['event','spatial'] for l in ['late','early'] for m in ['correct','wrong']]
OLD=OUT/'oracle001'

def register(name):
 assert sha(B1)==B1_SHA;verify_seal(OLD)
 assert read(OUT/'actuation_span001/ROOT_DECISION.json')['status']=='completed_independently_audited'
 assert all(x['registered_fixed_endpoint_success'] for x in read(OUT/'actuation_span001/ROOT_DECISION.json')['cases'])
 pre=OUT/'LOCATION_CPU_PREFLIGHT.json';assert read(pre)['status']=='passed';check_pins(read(pre)['pins'])
 rows=read(PANEL/'INPUTS.json');assert len(rows)==len({r['source'] for r in rows})==16
 paths=[Path(__file__),B1,pre,ROOT/'protocols/desta3d_v3_location_probe_v1.md',PANEL/'SOURCE_RECORDS.json',
  OLD/'PREDICTIONS_SEAL.json',OUT/'actuation_span001/ROOT_DECISION.json',OUT/'actuation_span001/ROOT_EQUIVALENT_LATENT_MAGNITUDE.json',
  ROOT/'tests/test_desta3d_v3_location_probe.py',ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/model.safetensors']
 register_base(name,dict(stage='source_norm_matched_early_late',phase_seconds=1800,maximum_new_bytes=16*2**30,
  seed=20260927,alpha=.25,arms=ARMS,adapter_sha=ADAPTER_SHA,checkpoint_sha=B1_SHA,source_label_sha=sha(PANEL/'SOURCE_RECORDS.json'),
  locations=['postFiLM before local reader','postReader/LN/SiLU before out_proj'],
  norm='Each source/branch fixed target from original correct late FP32 merger difference; no outcome scaling',
  primary='(early_correct - early_wrong) - (late_correct - late_wrong); event tIoU / spatial sIoU',
  representation='QR reachability does not imply small reader changes; current norm stays old tiny scale',target_input=False),rows,paths)

def run(name):
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';d=OUT/name
 with allocation(d) as (cfg,guard):
  import torch,gc
  from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,inputs_for
  from scripts.desta3d_v2_source_fit import prediction_record
  from scripts.desta3d_v2_reference_audit_cached_v3 import details
  from scripts.desta3d_v2_tta8_recovery_v2 import observe_time
  from scripts.desta3d_v2_p0 import adapter_sha256
  from scripts.desta3d_v3_latent_oracle import injection_effect
  from vg_tta.desta3d_v2 import Desta3DAdapterV2
  from vg_tta.desta3d_v2_ptd import capture_stock_fields
  from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
  from vg_tta.desta3d_v2_prediction_contract import validate_prediction
  from vg_tta.exact_frame_decode_audit_v2 import decode
  from vg_tta.desta3d_v3_location_probe import mask_at_location,match_delta
  from vg_tta.desta3d_v3_free_actuation import inject_delta
  from contextlib import nullcontext
  torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
  torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
  pr=processor_load();model=model_load().eval().requires_grad_(False)
  import model.ptd_generation as pg
  a=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
  a.load_state_dict(torch.load(B1,weights_only=False,map_location='cpu')['adapter']);a.set_train_stage('frozen')
  assert adapter_sha256(a)==ADAPTER_SHA
  def load(p):return torch.load(p,weights_only=False,map_location='cpu')
  for i,row in enumerate(read(d/'INPUTS.json')):
   guard();ep=d/'episodes'/f'{i:02}';ep.mkdir(parents=True);old=OLD/'episodes'/f'{i:02}'
   masks=load(old/'ORACLE_MASKS.pt');wrong=load(old/'WRONG_MASKS.pt');oldbase=load(old/'original.pt')
   frames,ids=decode(row['input']);prompt,pre=inputs_for(row,pr,frames)
   f=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
   support={k:tensor_sha(v) for k,v in f.items() if isinstance(v,torch.Tensor)}
   support.update({k:tensor_sha(prompt[k]) for k in ['input_ids','pixel_values_videos','video_grid_thw']})
   assert support==oldbase['support'] and equal(pre,oldbase['preprocess'])
   write(ep/'INPUT.json',dict(key=row['key'],source=row['source'],support=support,preprocess=pre,frame_ids=ids,
       negative_control=wrong['diagnostic'],source_GT_for_masks=True,old_mask_hashes={p.name:sha(p) for p in [old/'ORACLE_MASKS.pt',old/'WRONG_MASKS.pt']}))
   def af():
    with torch.no_grad():return a(f['visual_grid'],f['query_tokens'],query_mask=f['query_mask'],frame_times=f['frame_times'])
   base=af();base_tokens={b:base['updated_tokens_'+b] for b in ['event','spatial']};targets={};normlogs={}
   for arm in ARMS:
    guard();delta=None;branch=None
    if arm!='original':
     branch,loc,which=arm.split('_');ob='temporal' if branch=='event' else 'spatial';mm=masks if which=='correct' else wrong[ob]
     with mask_at_location(a,branch,mm[branch],loc,alpha=cfg['alpha']):candidate=af()
     other='event' if branch=='spatial' else 'spatial';assert torch.equal(candidate['updated_tokens_'+other],base_tokens[other])
     raw0=candidate['updated_tokens_'+branch]-base_tokens[branch]
     if loc=='late' and which=='correct':targets[branch]=float(raw0.double().norm())
     raw,delta,log=match_delta(base_tokens[branch],candidate['updated_tokens_'+branch],targets[branch]);normlogs[arm]=log
     torch.save(dict(raw=raw.cpu(),matched=delta.cpu(),branch=branch,location=loc,mask=mm[branch],norm=log),ep/(arm+'_DELTA.pt'))
     cm=inject_delta(a,branch,delta)
    else:cm=nullcontext()
    with cm:result,td=observe_time(pg,pr,len(ids),lambda:decode_shared_reference_two_pass(model,pr,prompt,a,f))
    p=prediction_record(result,row,pre,ADAPTER_SHA);p.update(arm=arm,readout=details(result),time_distribution=td,support=support,
        source_GT_for_masks=True,decoder_GT_prefix=False,target_read=False,optimizer_steps=0)
    torch.save(p,ep/(arm+'.pt'));validate_prediction(p,len(ids))
    keys=['key','source','frame_ids','positions','boxes_cxcywh','geometry_valid','interval','format_ok','preprocess','adapter_sha','event_completion','spatial_completion','time_distribution','readout']
    if arm=='original':
     checks={k:equal(p[k],oldbase[k]) for k in keys};write(ep/'BASELINE_REPLAY.json',checks);assert all(checks.values())
     native_base={b:result[b+'_injection']['updated_tokens'].detach().cpu() for b in ['event','spatial'] if result[b+'_injection'] is not None}
    else:
     effects={b:injection_effect(result[b+'_injection']['updated_tokens'],native_base[b]) for b in native_base if result[b+'_injection'] is not None}
     torch.save(effects,ep/(arm+'_CAST.pt'))
     if branch=='spatial':
      assert equal(p['time_distribution'],oldbase['time_distribution']) and p['event_completion']==oldbase['event_completion']
     if loc=='late' and which=='correct':
      oldp=load(old/('temporal.pt' if branch=='event' else 'spatial.pt'))
      checks={k:equal(p[k],oldp[k]) for k in keys};write(ep/(arm+'_OLD_REPLAY.json'),checks);assert all(checks.values()),checks
    assert adapter_sha256(a)==ADAPTER_SHA and all(not x.requires_grad and x.grad is None for x in a.parameters())
    assert all(not x.requires_grad and x.grad is None for x in model.parameters())
    print('LOCATION',i+1,16,arm,p['format_ok'],flush=True)
    del result,p,delta;gc.collect()
   write(ep/'COMPLETE.json',dict(key=row['key'],arms=9,norms=normlogs,optimizer_steps=0,exact_state=True))
   assert all(tensor_sha(f[k])==v for k,v in support.items() if k in f)
   del f,prompt,base,base_tokens,native_base,candidate,raw,raw0;gc.collect();torch.cuda.empty_cache()
   assert sum(p.stat().st_size for p in d.rglob('*') if p.is_file())<cfg['maximum_new_bytes']
  write(d/'COMPLETE.json',dict(status='source_location_predictions_complete_not_scored',queries=16,parents=16,predictions=144,
       seal_sha=seal(d,144),optimizer_steps=0,target_read=False,GPU_peak_allocated=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);p.add_argument('--name',required=True);a=p.parse_args();globals()[a.action](a.name)
