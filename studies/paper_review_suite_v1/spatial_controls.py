"""Coupled spatial controls and actual parameter sensitivity, cached native grids."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,json,hashlib,argparse,types,inspect,copy,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools'),str(Path(__file__).parent)]
from full import RUN,PROVIDER,read,sha,atomic,check,identity
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1'
ARMS=['Main','BoxSeeds','OneWayAffinity','PastAnchorOnly','MaskExtent','Corrupt25','Corrupt50','Gamma0','GammaHalf','GammaOne','Context1','Context2','Link0','LinkHalf','ObsQuarter','ObsThreeQuarter']

def altered_class():
 import stvg.spatial as m
 src=inspect.getsource(m.SpatialGeometry);assert src.count('at*ta.T*seeds')==1;changed=src.replace('at*ta.T*seeds','at*seeds');scope=dict(vars(m));exec(compile(changed,__file__,'exec'),scope);return scope['SpatialGeometry']

def frozen():
 p=read(D/'spatial_controls_plan.json');assert sha(Path(__file__))==p['code_sha256']and sha(D/'selection.json')==p['selection_sha256'];check();return p

def predict():
 import numpy as np,torch
 import torch.nn.functional as F
 from stvg.runtime import guard
 from stvg.spatial import SpatialGeometry,evidence_dp
 from stvg._spatial_math import occupancy,EPS
 from stvg.representation import extract
 from stvg.prototype import one_step
 from stvg.temporal import final_mbr
 from stvg.consensus import encode,estimate,apply
 from stvg.predictor import output_writeback
 p=frozen();torch.set_num_threads(1);sys.addaudithook(guard);selection=read(D/'selection.json');support=read(D/'supports_sealed.json');files={};started=time.perf_counter();oneway=altered_class();atomic(D/'spatial_controls_worker.json',identity())
 for ds,groups in selection.items():
  files[ds]={};cfg=read(R/'configs/default.json')['datasets'][ds]
  for stage,rows in groups.items():
   files[ds][stage]={}
   for i,row in enumerate(rows,1):
    key=row['id'];e=support['files'][ds][stage][key];assert sha(D/e['shared_path'])==e['shared_sha256'];shared=torch.load(D/e['shared_path'],map_location='cpu',weights_only=False);cap=torch.load(shared['parent_capture'],map_location='cpu',weights_only=False);assert sha(shared['parent_capture'])==shared['parent_capture_sha256'];ids=list(cap['frame_ids']);old=cap['geometry'];pos=list(map(int,cap['positions8']));native=np.asarray(old['native_boxes']);raw=shared['spatial_features'];valid=shared['spatial_valid'];h,w=shared['spatial_grid_shape'];seeds=old['side_maps'][pos,0].clone();assert seeds.shape==(8,h,w)
    def make_evidence(seed):return types.SimpleNamespace(features=raw,valid=valid,frame_ids=torch.tensor(ids),positions=pos,height=h,width=w,seeds=seed,observed=old['observed'].clone())
    # Original seed reconstruction is a strict no-op before any altered geometry.
    gcheck=SpatialGeometry(make_evidence(seeds),cap['E'],native)
    for field in ['boxes','features','u','side_available','link']:assert torch.equal(getattr(gcheck,field),old[field]),field
    points=encode(native)-encode(cap['Source'])[pos];cal=cfg['E_calibration'];est=estimate([x[None]for x in points],[np.ones(1)]*8,cal['scale'],cal['truncation'],1/16,False);unscaled=apply(cap['Source'],est['delta']);arrays=dict(Source=np.asarray(cap['Source']),frame_ids=np.asarray(ids),positions8=np.asarray(pos),Source_interval=np.asarray(cap['native_interval']))
    for arm in ARMS:
     g=SpatialGeometry.from_cached_geometry(old);context=1.5;link=.25;observed=.5
     if arm=='BoxSeeds':
      boxseed=occupancy(torch.as_tensor(native),h,w)[0]*valid[pos].reshape(8,h,w);g=SpatialGeometry(make_evidence(boxseed),cap['E'],native)
     elif arm=='OneWayAffinity':g=oneway(make_evidence(seeds),cap['E'],native)
     elif arm=='PastAnchorOnly':
      # Disable only FUTURE-time anchor on unobserved frames; not a one-way affinity claim.
      off=~g.observed;g.side_available[off,1]=False;g.boxes[off,2]=g.E[off];g._build_scores()
     elif arm=='MaskExtent':
      parent=read(RUN/ds/'sealed.json')['files'][key]
      with np.load(RUN/ds/parent['path'])as z:mb=z['mask_boxes'];available=z['mask_available']
      for side in [0,1]:
       use=(~g.observed).numpy();g.side_available[use,side]=torch.as_tensor(available[use,side]);g.side_reference_boxes[use,side]=torch.as_tensor(mb[use,side]);g.boxes[use,side+1]=torch.as_tensor(np.where(available[use,side,None],mb[use,side],np.asarray(g.E)[use]))
      occ=occupancy(g.boxes,h,w)[0]*valid.reshape(len(ids),1,h,w);pool=torch.bmm(occ.flatten(-2),raw.double())/occ.flatten(-2).sum(-1).clamp_min(EPS)[...,None];g.features=F.normalize(pool,dim=-1,eps=1e-12);g.association_scores=torch.einsum('tkd,tsd->tks',g.features,g.side_prototypes).clamp(0,1);g._build_scores()
     elif arm.startswith('Corrupt'):
      count=2 if arm=='Corrupt25'else 4;order=sorted(range(8),key=lambda j:hashlib.sha256(f'20261024:{ds}:{key}:anchor:{j}'.encode()).hexdigest());changed=native.copy();ss=seeds.clone()
      for j in order[:count]:changed[j,0]=1-changed[j,0];ss[j]=ss[j].flip(-1)
      pp=encode(changed)-encode(cap['Source'])[pos];ee=estimate([x[None]for x in pp],[np.ones(1)]*8,cal['scale'],cal['truncation'],1/16,False);E=output_writeback(cap['Source'],apply(cap['Source'],ee['delta']),cfg['initialization_gamma']);g=SpatialGeometry(make_evidence(ss),E,changed)
     elif arm.startswith('Gamma'):
      gamma={'Gamma0':0,'GammaHalf':.5,'GammaOne':1}[arm];g=SpatialGeometry(make_evidence(seeds),output_writeback(cap['Source'],unscaled,gamma),native)
     elif arm.startswith('Context'):context={'Context1':1,'Context2':2}[arm]
     elif arm=='Link0':link=0
     elif arm=='LinkHalf':link=.5
     elif arm=='ObsQuarter':observed=.25
     elif arm=='ObsThreeQuarter':observed=.75
     def bank(iv):
      b=g.for_interval(iv);b.score_pair*=link/.25
      obs=b.event&b.observed;unobs=b.event&~b.observed;weights=observed*obs.double()/max(1,int(obs.sum()))+(1-observed)*b.supported.double()*unobs/max(1,int(unobs.sum()));b.score_unary=weights[:,None]*b.u;return b
     aux=evidence_dp(bank([ids[0],ids[-1]+1]))['final'].numpy();expanded=aux.copy();expanded[:,2:]*=context/1.5;features,_=extract(dict(frame_ids=ids,native_local_grid=shared['native_local_grid']),expanded);q,stats=one_step(features,cap['native_prior']);iv=final_mbr(ids,q)['interval'];boxes=evidence_dp(bank(iv))['final'].numpy();arrays[arm]=boxes;arrays[arm+'_interval']=np.asarray(iv)
     if arm=='Main':
      assert np.array_equal(features,cap['content_features'])
      with np.load(D/e['path'])as original:assert np.array_equal(boxes,original['DPDP'])and np.array_equal(iv,original['DPDP_interval'])
    dest=D/'spatial_controls'/ds;dest.mkdir(parents=True,exist_ok=True);out=dest/(key+'.npz');assert not out.exists();np.savez_compressed(out,**arrays);files[ds][stage][key]=dict(path=str(out.relative_to(D)),sha256=sha(out),Main_bitwise=True,GT_read=False);atomic(D/'spatial_controls_progress.json',dict(dataset=ds,stage=stage,completed=i,total=len(rows),GT_read=False));print(json.dumps(dict(dataset=ds,stage=stage,completed=i)),flush=True)
 atomic(D/'spatial_controls_sealed.json',dict(files=files,queries=384,GT_read=False,fingerprint=p['fingerprint'],seconds=time.perf_counter()-started,new_model=0,new_expert=0))

def score():
 import numpy as np,torch
 from full_score import metric,cluster
 torch.set_num_threads(1);p=frozen();seal=read(D/'spatial_controls_sealed.json');assert seal['queries']==384 and not seal['GT_read'];checks=0;maxerr=0.
 for ds,groups in read(D/'selection.json').items():
  j=read(RUN/'plan.json')['jobs'][ds];f=PROVIDER/j['labels'];assert sha(f)==j['labels_sha256'];gt={x['id']:x for x in map(json.loads,f.read_text().splitlines())if x['dataset']==ds};old={x['id']:x for x in read(RUN/ds/'metrics.json')['private_rows']}
  for stage,records in groups.items():
   rows=[]
   for r in records:
    key=r['id'];e=seal['files'][ds][stage][key];assert sha(D/e['path'])==e['sha256'];g=gt[key];vals={'Source':old[key]['values']['Source']}
    with np.load(D/e['path'])as z:
     for a in ARMS:vals[a],err=metric(z[a],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err)
    assert vals['Main']==old[key]['values']['Full'];rows.append(dict(id=key,parent=str(g['parent']),duration_group=old[key]['duration_group'],values=vals))
   arms=list(rows[0]['values']);fields=['v','t','observed_actual_sIoU','unobserved_actual_sIoU'];pairs=[(a,'Main')for a in ARMS if a!='Main'];report=dict(dataset=ds,stage=stage,queries=len(rows),means={a:{f:float(np.mean([x['values'][a][f]for x in rows if x['values'][a][f]is not None]))for f in fields}for a in arms},paired={a+'-'+b:cluster(rows,a,b)for a,b in pairs},fingerprint=p['fingerprint'],parameter_selection=False,definitions=p['definitions']);pub=P/'spatial_controls'/ds/stage;atomic(pub/'report.json',report);atomic(D/'spatial_controls'/ds/(stage+'_metrics.json'),dict(report,private_rows=rows))
   with(pub/'signed.tsv').open('w')as f:
    w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query']+[a+'_'+k for a in arms for k in fields]);w.writerows([[f'Q{i:03d}']+[x['values'][a][k]for a in arms for k in fields]for i,x in enumerate(rows,1)])
 atomic(D/'spatial_controls_score_receipt.json',dict(checks=checks,max_error=maxerr,all384sealed_before_GT=True));print('Spatial controls scored',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['predict','score']);args=a.parse_args();predict()if args.command=='predict'else score()
