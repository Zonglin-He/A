"""Bounded HC temporal development and a new immutable full run; never alters v1."""
import argparse,json,hashlib,os,sys,time,subprocess,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PROVIDER=ROOT.parent/'stvg-opd';OUT=ROOT/'outputs/temporal_hc_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)
def identity():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def provider_check():
 p=read(PROVIDER/'outputs/boxfixed_full_v1/plan.json');assert p['fingerprint']=='2ebd134dfd055c8e17e01859b706d56d187b15b8907a8129c49669539d84ea6e';assert read(PROVIDER/'outputs/boxfixed_full_v1/state.json')['status']=='complete_published';assert all(sha(PROVIDER/f)==h for f,h in p['sources'].items());return p

def prepare():
 p=provider_check();rows=[json.loads(s)for s in (PROVIDER/p['jobs']['hc2']['manifest']).read_text().splitlines()];groups={}
 for r in rows:groups.setdefault(str(r['metadata']['source']),[]).append(r)
 rank=lambda x:hashlib.sha256(('HC-temporal-v2-20261022|'+str(x)).encode()).hexdigest()
 parents=sorted(groups,key=rank)[:96];selected=[min(groups[k],key=lambda r:rank(r['id']))for k in parents]
 assert len(selected)==len({r['metadata']['source']for r in selected})==96
 manifest=OUT/'panel.jsonl';manifest.parent.mkdir(parents=True,exist_ok=True)
 if manifest.exists():assert manifest.read_text()==''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in selected)
 else:manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in selected))
 protocol=dict(status='locked_before_panel_labels',user_authorization='2026-10-22 HC2 time tuning and new full evaluation',dataset='hc2',panel_manifest_sha256=sha(manifest),selection='SHA256 fixed seed on input parent IDs, one SHA-ranked query/parent; first64 tune, next32 validation. Never selected by full outcomes.',tune_parents=64,validation_parents=32,seed=20261022,exposure='All official full outcomes have already been evaluated; this is disclosed target-domain development and a post-selection full rerun, not a new independent test.',axes=dict(endpoint_temperature=[.75,1.,1.25,1.5],layer_slope=[-1.,0.,1.,2.],length_power=[-.5,0.,.5,1.]),maximum_trials=18,selection_rule='Primary tIoU. On tune require at least+.3pp over old Global6, fullv loss <=.1pp, unobservedspace loss <=.1pp, severe-v vs Source <=old+2. Choose by tIoU then vIoU; no seed selection. At most3 neighboring refinements,1 combination,3 leave-one-axis attribution controls within18 total. Root reviews each stage.',validation_rule='One locked profile on32 parent-disjoint validation; require tIoU >=old+.2pp and >Source, fullv >=old-.1pp, unobservedspace >=old-.1pp, severe-v <=old+2; no retuning on validation. If it fails, preserve old profile and report failure, do not relabel as solved.',fixed='Frozen Source, input RGB/grids/precision, one same ReferDINO/eight observations, ERef gamma/calibration, complete-box propagation, spatial score and original MBR. No448/replay/actioness/training/new teacher.',source_provider_plan_sha256=sha(PROVIDER/'outputs/boxfixed_full_v1/plan.json'),source_provider_runtime_fingerprint=p['runtime_fingerprint'],source_provider_frozen_hashes=p['sources'],labels_sha256=p['jobs']['hc2']['labels_sha256'],provider_model_and_teacher_reuse='Original unmodified native224 capture provider and exact original evidence runtime fingerprint. New temporal predictions have their own plan/code fingerprint. No old resume receipt is accepted as a new result.')
 path=OUT/'protocol.json'
 if path.exists():assert read(path)==protocol
 else:atomic(path,protocol)
 print(json.dumps(dict(status='prepared',queries=96,protocol_sha256=sha(path))))

def lean(g):
 keys=['E','frames','frame_ids','positions','positions8','observed','confidence','reliable','valid','candidate_quality','candidate_origin','metadata','projection','guard','link','boxes','seed_valid','source_anchors','side_available']
 return {k:g[k]for k in keys}

def capture():
 # Frozen provider imports are explicitly separate from the maintained readout.
 sys.path.insert(0,str(PROVIDER))
 import numpy as np,torch
 from stvg.runtime import setup,lease,load_source
 from stvg.predictor import Predictor
 from stvg.visual_cache import VisualCache
 from stvg.inputs import validate_input
 from stvg.worker import resource_guard
 from scripts.boxfixed_tuning.cache import RGBCache,ReusingTeacher,binding,from_native
 from scripts.tube_fusion.core import head_logits
 plan=provider_check();protocol=read(OUT/'protocol.json');assert sha(OUT/'panel.jsonl')==protocol['panel_manifest_sha256'];rows=[json.loads(s)for s in(OUT/'panel.jsonl').read_text().splitlines()];dest=OUT/'captures';assert not(dest/'sealed.json').exists()
 setup();lock=lease();model=Predictor('hc2',check=resource_guard);expert=ReusingTeacher(OUT/'teacher_cache',plan['runtime_fingerprint'],previous=[PROVIDER/'outputs/boxfixed_full_v1/teacher_cache']);model.teacher_service=expert;rgb=RGBCache();visual=VisualCache(4*2**30);index=read(PROVIDER/'outputs/boxfixed_full_v1/capture_index.json')['files'];files={};begin=time.perf_counter()
 atomic(dest/'worker.json',identity());atomic(dest/'run.json',dict(status='running',GT_read=False,**identity()))
 try:
  for i,r in enumerate(rows):
   resource_guard();key=f'P{i+1:03d}';path=dest/(key+'.pt');receipt=dest/(key+'.json');start=time.perf_counter()
   if receipt.exists():
    old=read(receipt);assert old['protocol_sha256']==sha(OUT/'protocol.json')and sha(path)==old['sha256'];files[key]=old;continue
   b=binding(r,plan['runtime_fingerprint']);reuse=False;cost={}
   if b in index:
    q=PROVIDER/index[b]['path'];assert sha(q)==index[b]['sha256'];cap=torch.load(q,map_location='cpu',weights_only=False);reuse=True
   else:
    frames,ids,rc=rgb.get(r['metadata'],'hc2');validate_input(frames,ids,r['metadata'])
    if model.source is None:model.source=load_source(model.config)
    torch.manual_seed(20260920);np.random.seed(20260920)
    with visual.context(model.source,frames,ids):native=model.capture(frames,ids,r['metadata'],temporal_mode='native224',export_replay=True)
    tc=native['temporal_cache'];head=model.source.temp_embed.cuda();logits,hcost=head_logits(head,[v['hidden']for v in tc['offsets']],[v['native_logits']for v in tc['offsets']]);head.cpu();cap=from_native(native,logits);cost=dict(native['cost']);del native,tc,logits,frames
   original=PROVIDER/'outputs/boxfixed_full_v1/hc2/predictions'/(r['id']+'.npz')
   with np.load(original,allow_pickle=False)as z:
    assert np.array_equal(z['Source'],cap['Source'])and np.array_equal(z['E_Ref'],cap['E'])and np.array_equal(z['Source_interval'],cap['native_interval']);reference=z['Final'].copy();reference_interval=z['Final_interval'].copy()
   cap['geometry']=lean(cap['geometry']);cap.update(fps=float(r['metadata']['fps']),reference=reference,reference_interval=reference_interval,GT_read=False)
   path.parent.mkdir(parents=True,exist_ok=True);torch.save(cap,path)
   rec=dict(path=path.name,sha256=sha(path),protocol_sha256=sha(OUT/'protocol.json'),GT_read=False,complete_capture_hit=reuse,expert_forwards=cost.get('ReferDINO_query_forwards',0),capture_cost=cost,seconds=time.perf_counter()-start)
   atomic(receipt,rec);files[key]=rec;atomic(dest/'progress.json',dict(completed=i+1,total=96,seconds=time.perf_counter()-begin,**identity()));print(json.dumps(dict(completed=i+1,total=96,seconds=rec['seconds'],expert_forwards=rec['expert_forwards'])),flush=True);del cap;gc.collect();torch.cuda.empty_cache()
  provider_check();atomic(dest/'sealed.json',dict(files=files,queries=96,GT_read=False,protocol_sha256=sha(OUT/'protocol.json')));atomic(dest/'run.json',dict(status='completed_sealed',seconds=time.perf_counter()-begin,**identity()))
 except BaseException:atomic(dest/'failure.json',dict(traceback=traceback.format_exc(),completed=len(files)));raise
 finally:expert.close();lock.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','capture']);a=p.parse_args();globals()[a.command]()
