"""New HC1902 temporal-profile queue, independent of the completed v1 run."""
import argparse,json,hashlib,os,sys,time,gc,subprocess,traceback,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PROVIDER=ROOT.parent/'stvg-opd';TUNE=ROOT/'outputs/temporal_hc_v3';RUN=ROOT/'outputs/hc_temporal_full_v3_exploratory';PUBLIC=ROOT/'results/hc_temporal_full_v3_exploratory'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def identity(pid=None):
 pid=os.getpid()if pid is None else pid;return dict(pid=pid,startticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
def alive(x):
 try:return identity(x['pid'])=={k:x[k]for k in ['pid','startticks']}
 except (FileNotFoundError,ProcessLookupError):return False
def check():
 p=read(RUN/'plan.json');assert sha(TUNE/'exploratory_full_authorization.json')==p['exploratory_extension_sha256'];assert all(sha(ROOT/f)==h for f,h in p['code'].items());assert all(sha(PROVIDER/f)==h for f,h in p['provider_sources'].items());assert all(sha(PROVIDER/f)==h for f,h in p['assets'].items());assert sha(PROVIDER/p['manifest'])==p['manifest_sha256'];assert sha(TUNE/'selected.json')==p['selection_sha256'];assert sha(ROOT/'outputs/temporal_hc_v2/captures/sealed.json')==p['prior_capture_seal_sha256'];assert sha(ROOT/'outputs/temporal_hc_v2/panel.jsonl')==p['prior_capture_manifest_sha256'];assert sha(TUNE/'validation'/p['selected_name']/'score_receipt.json')==p['validation_receipt_sha256'];return p
def prepare():
 if (RUN/'plan.json').exists():return check()
 old=read(PROVIDER/'outputs/boxfixed_full_v1/plan.json');assert read(PROVIDER/'outputs/boxfixed_full_v1/state.json')['status']=='complete_published';selected=read(TUNE/'selected.json');v=read(ROOT/'results/temporal_hc_v3/validation'/f'{selected["name"]}.json');assert not v['eligible']and not v['gates']['t_above_Source']and all(v['gates'][k]for k in ['t_gain','v_protection','unobserved_protection','severe_protection']);card=read(TUNE/'exploratory_full_authorization.json');assert card['candidate_frozen']and card['default_unchanged']and card['selection_sha256']==sha(TUNE/'selected.json')and card['validation_report_sha256']==sha(ROOT/'results/temporal_hc_v3/validation'/f'{selected["name"]}.json')
 code={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),ROOT/'tools/temporal_hc_v3.py',ROOT/'tools/temporal_hc_readout_trial.py',ROOT/'configs/default.json',*sorted((ROOT/'stvg').glob('*.py'))]}
 p=dict(status='exploratory_full_not_validation_passed',validation_eligible=False,default_unchanged=True,exploratory_extension_sha256=sha(TUNE/'exploratory_full_authorization.json'),dataset='hc2',queries=1902,params=selected['params'],selected_name=selected['name'],selection_sha256=sha(TUNE/'selected.json'),validation_receipt_sha256=sha(TUNE/'validation'/selected['name']/'score_receipt.json'),code=code,provider_plan_sha256=sha(PROVIDER/'outputs/boxfixed_full_v1/plan.json'),provider_sources=old['sources'],provider_runtime_fingerprint=old['runtime_fingerprint'],assets=old['assets'],manifest=old['jobs']['hc2']['manifest'],manifest_sha256=old['jobs']['hc2']['manifest_sha256'],labels=old['jobs']['hc2']['labels'],labels_sha256=old['jobs']['hc2']['labels_sha256'],old_seal_sha256=sha(PROVIDER/'outputs/boxfixed_full_v1/hc2/sealed.json'),panel_seal_sha256=sha(TUNE/'captures/sealed.json'),prior_capture_seal_sha256=sha(ROOT/'outputs/temporal_hc_v2/captures/sealed.json'),prior_capture_manifest_sha256=sha(ROOT/'outputs/temporal_hc_v2/panel.jsonl'),protocol_sha256=sha(TUNE/'protocol.json'),methods=['Source','Previous','Candidate'],GT_read=False,scope='Exploratory HC2 target development rerun. Original Source validation gate failed and remains failed; this is not a passed-validation promotion or independent test. Only HC2 rerun. Frozen native source/expert/geometry; changed explicitly recorded output-only posterior and MBR-utility parameters. Original Global6/MBR retained as Previous. Exact old provider and teacher key retained with explicit provenance; new predictions have independent plan and receipts. All new compact logits/geometry saved for future output-only verification. Historical full outcomes exposed, no retuning on this full result.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(RUN/'plan.json',p)
 for rel in code:
  q=RUN/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/rel).read_bytes())
 atomic(PUBLIC/'protocol.json',{k:v for k,v in p.items()if k not in ['manifest','labels']});print(json.dumps(dict(status='prepared',fingerprint=p['fingerprint'],queries=1902)));return check()
def worker():
 sys.path.insert(0,str(PROVIDER));import numpy as np,torch
 from stvg.runtime import setup,lease,load_source
 from stvg.predictor import Predictor
 from stvg.visual_cache import VisualCache
 from stvg.inputs import validate_input
 from stvg.worker import resource_guard
 from stvg.spatial import SpatialGeometry,evidence_dp
 from stvg.temporal import final_mbr
 from scripts.boxfixed_tuning.cache import RGBCache,ReusingTeacher,binding,from_native
 from scripts.tube_fusion.core import head_logits
 spec=importlib.util.spec_from_file_location('stvg.temporal_readout',ROOT/'stvg/temporal_readout.py');profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
 spec=importlib.util.spec_from_file_location('hc_v3_capture',ROOT/'tools/temporal_hc_v3.py');capture_utils=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture_utils)
 p=check();setup();lock=lease();old=read(PROVIDER/'outputs/boxfixed_full_v1/plan.json');rows=[json.loads(s)for s in(PROVIDER/p['manifest']).read_text().splitlines()];index=read(PROVIDER/'outputs/boxfixed_full_v1/capture_index.json')['files'];panelrows=[json.loads(s)for s in(TUNE/'panel.jsonl').read_text().splitlines()];panelseal=read(TUNE/'captures/sealed.json')
 panel={r['id']:panelseal['files'][f'P{i+1:03d}']for i,r in enumerate(panelrows)};priorroot=ROOT/'outputs/temporal_hc_v2';priorrows=[json.loads(s)for s in(priorroot/'panel.jsonl').read_text().splitlines()];priorseal=read(priorroot/'captures/sealed.json');priorpanel={r['id']:priorseal['files'][f'P{i+1:03d}']for i,r in enumerate(priorrows)};model=Predictor('hc2',check=resource_guard);expert=ReusingTeacher(RUN/'teacher_cache',p['provider_runtime_fingerprint'],previous=[PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',TUNE/'teacher_cache']);model.teacher_service=expert;rgb=RGBCache();visual=VisualCache(4*2**30);files={};recent=[];begin=time.perf_counter();atomic(RUN/'worker.json',dict(**identity(),fingerprint=p['fingerprint']))
 try:
  for i,r in enumerate(rows):
   resource_guard();key=r['id'];rp=RUN/'receipts'/(key+'.json');dest=RUN/'predictions'/(key+'.npz');cp=RUN/'readout_cache'/(key+'.pt')
   if rp.exists():
    rec=read(rp);assert rec['fingerprint']==p['fingerprint']and rec['input_binding']==binding(r,p['provider_runtime_fingerprint'])and sha(dest)==rec['prediction_sha256']and sha(cp)==rec['capture_sha256'];files[key]=dict(prediction=dest.name,prediction_sha256=sha(dest),receipt=rp.name,receipt_sha256=sha(rp),capture=cp.name,capture_sha256=sha(cp));continue
   assert not dest.exists()and not cp.exists(),'Unbound output exists; preserve for recovery, never overwrite'
   tick=time.perf_counter();b=binding(r,p['provider_runtime_fingerprint']);cost={};reuse='none'
   if key in panel:
    item=panel[key];src=TUNE/'captures'/item['path'];assert sha(src)==item['sha256'];cap=torch.load(src,map_location='cpu',weights_only=False);reuse='panel'
   elif key in priorpanel:
    item=priorpanel[key];src=priorroot/'captures'/item['path'];assert sha(src)==item['sha256'];cap=torch.load(src,map_location='cpu',weights_only=False);reuse='v2_bound_capture'
   elif b in index:
    item=index[b];src=PROVIDER/item['path'];assert sha(src)==item['sha256'];cap=torch.load(src,map_location='cpu',weights_only=False);reuse='v1_bound_capture'
   else:
    frames,ids,rc=rgb.get(r['metadata'],'hc2');validate_input(frames,ids,r['metadata'])
    if model.source is None:model.source=load_source(model.config)
    torch.manual_seed(20260920);np.random.seed(20260920)
    with visual.context(model.source,frames,ids):native=model.capture(frames,ids,r['metadata'],temporal_mode='native224',export_replay=True)
    tc=native['temporal_cache'];head=model.source.temp_embed.cuda();logits,hcost=head_logits(head,[v['hidden']for v in tc['offsets']],[v['native_logits']for v in tc['offsets']]);head.cpu();cap=from_native(native,logits);cost=dict(native['cost']);cost['head']=hcost;cost['RGB']=rc;del native,tc,logits,frames
   oldpath=PROVIDER/'outputs/boxfixed_full_v1/hc2/predictions'/(key+'.npz')
   with np.load(oldpath,allow_pickle=False)as z:
    assert np.array_equal(z['Source'],cap['Source'])and np.array_equal(z['E_Ref'],cap['E'])and np.array_equal(z['Source_interval'],cap['native_interval']);previous=z['Final'].copy();previous_interval=z['Final_interval'].copy()
   cap['geometry']=capture_utils.lean(cap['geometry']);cap['fps']=float(r['metadata']['fps']);baseq=profile.posterior(cap['records'],cap['logits'],cap['frame_ids'],profile.BASE);assert np.array_equal(profile.decode(cap['frame_ids'],baseq,profile.BASE)['interval'],previous_interval);gbase=SpatialGeometry.__new__(SpatialGeometry);gbase.__dict__.update(cap['geometry']);assert np.array_equal(evidence_dp(gbase.for_interval(previous_interval))['final'].numpy(),previous);q=profile.posterior(cap['records'],cap['logits'],cap['frame_ids'],p['params']);span=profile.decode(cap['frame_ids'],q,p['params'])['interval'];g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);out=evidence_dp(g.for_interval(span))
   dest.parent.mkdir(parents=True,exist_ok=True);cp.parent.mkdir(parents=True,exist_ok=True);torch.save(cap,cp);np.savez(dest,Source=cap['Source'],Source_interval=cap['native_interval'],Previous=previous,Previous_interval=previous_interval,Candidate=out['final'].numpy(),Candidate_interval=span,frame_ids=cap['frame_ids'],positions8=cap['positions8'])
   rec=dict(fingerprint=p['fingerprint'],input_binding=b,prediction_sha256=sha(dest),capture_sha256=sha(cp),capture_reuse=reuse,capture_cost=cost,params=p['params'],query_wall_seconds=time.perf_counter()-tick,visual=visual.stats(),RGB_hits=rgb.hits,RGB_misses=rgb.misses,Source448_calls=0,extra_decoder_calls=0,updates=0,GT_read=False)
   atomic(rp,rec);files[key]=dict(prediction=dest.name,prediction_sha256=sha(dest),receipt=rp.name,receipt_sha256=sha(rp),capture=cp.name,capture_sha256=sha(cp));recent.append(rec['query_wall_seconds']);recent=recent[-100:];atomic(RUN/'progress.json',dict(completed=i+1,total=1902,recent_seconds=float(np.mean(recent)),remaining_estimate_seconds=float(np.mean(recent))*(len(rows)-i-1),**identity(),visual=visual.stats(),RGB_hits=rgb.hits,RGB_misses=rgb.misses,GT_read=False));print(json.dumps(dict(completed=i+1,total=1902,seconds=rec['query_wall_seconds'],reuse=reuse)),flush=True);del cap,q,out;gc.collect();torch.cuda.empty_cache()
  check();atomic(RUN/'sealed.json',dict(queries=1902,files=files,fingerprint=p['fingerprint'],GT_read=False));atomic(RUN/'run.json',dict(status='completed_sealed',seconds=time.perf_counter()-begin,**identity()))
 finally:expert.close();lock.close()
def controller(resume=False):
 import fcntl
 p=prepare();lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (RUN/'state.json').exists():
  old=read(RUN/'state.json');assert resume and old['fingerprint']==p['fingerprint']and old['status']!='complete_published';assert not any(alive(x)for x in old.get('children',[]))
 state=dict(status='running',fingerprint=p['fingerprint'],controller=identity(),children=[]);atomic(RUN/'state.json',state)
 try:
  for command,stage in [('worker','running'),('score','scoring')]:
   if command=='worker'and(RUN/'sealed.json').exists():continue
   with(RUN/(command+'.log')).open('ab')as log:
    child=subprocess.Popen([sys.executable,'-B',__file__,command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,**({'CUDA_VISIBLE_DEVICES':''}if command=='score'else{})));state.update(status=stage,children=[dict(**identity(child.pid),role=command)]);atomic(RUN/'state.json',state);code=child.wait();assert code==0,f'{command} exit{code}; preserve all outputs'
   state['children']=[];atomic(RUN/'state.json',state)
  state['status']='awaiting_publication';atomic(RUN/'state.json',state)
 except BaseException:state.update(status='failed',error=traceback.format_exc());atomic(RUN/'state.json',state);raise
 finally:lock.close()
def score():
 os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');sys.path.insert(0,str(ROOT));import numpy as np,torch
 from tools.temporal_hc_readout_trial import metric
 p=check();seal=read(RUN/'sealed.json');assert seal['queries']==1902 and seal['fingerprint']==p['fingerprint']and seal['GT_read']is False
 if(RUN/'score_receipt.json').exists():assert read(RUN/'score_receipt.json')['seal_sha256']==sha(RUN/'sealed.json');return
 for key,e in seal['files'].items():assert sha(RUN/'predictions'/e['prediction'])==e['prediction_sha256']and sha(RUN/'receipts'/e['receipt'])==e['receipt_sha256']and sha(RUN/'readout_cache'/e['capture'])==e['capture_sha256']
 label=PROVIDER/p['labels'];assert sha(label)==p['labels_sha256'];truth={r['id']:r for r in [json.loads(s)for s in label.read_text().splitlines()]if r['dataset']=='hc2'};rows=[json.loads(s)for s in(PROVIDER/p['manifest']).read_text().splitlines()];values=[];error=0.;costs=[];checks=0
 for i,r in enumerate(rows):
  e=seal['files'][r['id']];v={}
  with np.load(RUN/'predictions'/e['prediction'],allow_pickle=False)as z:
   for arm in ['Source','Previous','Candidate']:v[arm],err=metric(z[arm],r,truth[r['id']],z[arm+'_interval'].tolist(),z['positions8']);error=max(error,err);checks+=1
   v['Candidate_fixed_space'],err=metric(z['Previous'],r,truth[r['id']],z['Candidate_interval'].tolist(),z['positions8']);error=max(error,err);checks+=1
  values.append(dict(parent=str(truth[r['id']]['parent']),values=v));costs.append(read(RUN/'receipts'/e['receipt']))
  if(i+1)%100==0:atomic(RUN/'score_progress.json',dict(completed=i+1,total=1902,CPU_only=True))
 from stvg.evaluation import finite_mean,main_table
 means=main_table({a:{f:finite_mean([r['values'][a][f]for r in values])for f in values[0]['values'][a]}for a in ['Source','Previous','Candidate','Candidate_fixed_space']});oldreport=read(PROVIDER/'results/boxfixed_full_v1/hc2/report.json');assert all(abs(means[a][f]-oldreport['means'][b][f])<1e-12 for a,b in [('Source','Source'),('Previous','Final')]for f in ['m_vIoU','m_tIoU','m_sIoU_unobserved_actual']);parents=sorted({r['parent']for r in values});lookup={x:i for i,x in enumerate(parents)};weights=np.random.default_rng(20261022).multinomial(len(parents),[1/len(parents)]*len(parents),size=10000);paired={}
 for b in ['Previous','Source']:
  paired['Candidate-'+b]={}
  for f in ['v','t','unobserved_actual_sIoU','observed_actual_sIoU','boundary_MAE_seconds']:
   scale=1 if f.endswith('seconds')else 100;sums=np.zeros(len(parents));counts=sums.copy();d=[]
   for r in values:
    x,y=r['values']['Candidate'][f],r['values'][b][f]
    if x is None or y is None:continue
    z=scale*(x-y);d.append(z);j=lookup[r['parent']];sums[j]+=z;counts[j]+=1
   d=np.array(d);den=weights@counts;boot=(weights@sums)[den>0]/den[den>0];paired['Candidate-'+b][f]=dict(mean=float(d.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),increased=int((d>1e-10).sum()),decreased=int((d < -1e-10).sum()),unchanged=int((abs(d)<=1e-10).sum()),declined_over5pp=int((d < -5).sum())if scale==100 else None)
 report=dict(status='completed_full_official_evaluation',queries=1902,parents=len(parents),params=p['params'],means=means,paired=paired,proof=dict(seal_sha256=sha(RUN/'sealed.json'),labels_sha256=sha(label),all_predictions_sealed_before_GT=True,baseline_full_interval_and_boxes_bitwise_recovered_per_query=True,official_dense_checks=checks,max_error=error,CPU_only=True),exposure=p['scope'],efficiency=dict(mean_query_wall_seconds=float(np.mean([r['query_wall_seconds']for r in costs])),capture_reuse=sum(r['capture_reuse']!='none'for r in costs),expert_forwards=sum(r['capture_cost'].get('ReferDINO_query_forwards',0)for r in costs),Source448_calls=0,updates=0,cost_scope='Current reused-evidence run wall only. Original expert/capture generation cost excluded; not a cold deployment speedup.'))
 atomic(PUBLIC/'report.json',report);fields=['query','parent','Source_v','Previous_v','Candidate_v','Source_t','Previous_t','Candidate_t','Candidate_minus_Previous_v_pp','Candidate_minus_Previous_t_pp','severe_vs_Previous','severe_vs_Source'];pm={x:f'P{i+1:03d}'for i,x in enumerate(parents)}
 for off in range(0,len(values),1000):
  lines=['\t'.join(fields)]
  for i,r in enumerate(values[off:off+1000],off+1):
   v=r['values'];dv=100*(v['Candidate']['v']-v['Previous']['v']);dt=100*(v['Candidate']['t']-v['Previous']['t']);x=[f'H{i:05d}',pm[r['parent']]]+[v[a][f]for f in ['v','t']for a in ['Source','Previous','Candidate']]+[dv,dt,dv < -5,100*(v['Candidate']['v']-v['Source']['v']) < -5];lines.append('\t'.join(str(z)for z in x))
  (PUBLIC/f'signed_{off//1000+1:02d}.tsv').write_text('\n'.join(lines)+'\n')
 atomic(RUN/'metrics.json',dict(report,private_rows=values));atomic(RUN/'score_receipt.json',dict(status='scored',seal_sha256=sha(RUN/'sealed.json'),public_files={str(f.relative_to(ROOT)):sha(f)for f in PUBLIC.glob('*')if f.suffix in ['.json','.tsv']}));print(json.dumps(dict(means=means,paired=paired)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','worker','score']);p.add_argument('--resume',action='store_true');a=p.parse_args();controller(a.resume)if a.command=='run'else globals()[a.command]()
