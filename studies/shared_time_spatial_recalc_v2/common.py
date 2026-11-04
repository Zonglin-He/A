"""Fresh three-path spatial comparison at one immutable Full interval."""
from pathlib import Path
import os,sys,json,time,hashlib,functools,shutil,csv,itertools,importlib.util
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/shared_time_spatial_recalc_v2';P=R/'results/shared_time_spatial_recalc_v2';CHECK=R/'outputs/spatial_table_recheck_v1'
sys.path[:0]=[str(R/'studies/reviewer_key_controls_v1'),str(R),str(R/'tools')]
import common_v1 as old
import numpy as np,torch
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.initialization import initialize
from stvg.predictor import output_writeback
read=old.read;sha=old.sha;atomic=old.atomic;digest=old.digest;identity=old.identity;fixed=functools.lru_cache(None)(read)
spec=importlib.util.spec_from_file_location('plain_raw8',R/'studies/pure_linear_standalone_v1/pure.py');plain=importlib.util.module_from_spec(spec);spec.loader.exec_module(plain)
COUNTS={'hc2':1902,'vidstg':10303};ARMS=['PureExpertLinear','SharedCorrection','Full'];PAIRS=[('Full','PureExpertLinear'),('Full','SharedCorrection')]
@functools.lru_cache(None)
def rows(ds):return old.selected.records(ds)
def validate():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for g in ['code_sha256','input_sha256']:
  for f,h in p[g].items():assert sha(R/f)==h,f
 return p
def resources(start):
 assert time.perf_counter()-start<72*3600 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30

def independent_E(source,pos,raw,cfg):
 # Independent exhaustive singleton residual consensus, original tie ordering.
 enc=lambda a:np.column_stack([a[:,:2],np.log(a[:,2:])]);res=enc(raw)-enc(source)[pos];scale=np.asarray(cfg['E_calibration']['scale']);cut=cfg['E_calibration']['truncation'];proposals=[]
 for assign in itertools.product([-1,0],repeat=8):
  take=[res[i]for i,x in enumerate(assign)if x==0];delta=np.mean(take,axis=0)if take else np.zeros(4);dist=np.square((res-delta)/scale).sum(1);sel=tuple(0 if x<cut else -1 for x in dist);proposals.append((float(np.minimum(cut,dist).sum()),sel,assign,delta))
 proposals.sort(key=lambda z:z[:3]);delta=proposals[0][3];gamma=cfg['initialization_gamma'];z=enc(source)+gamma*delta;return np.column_stack([z[:,:2],np.exp(z[:,2:])])

def numpy_dp(bank):
 u=bank.score_unary.numpy();pair=bank.score_pair.numpy();valid=bank.valid.numpy().copy();valid[~bank.active.numpy(),1:]=False;v=np.where(valid[0],u[0],-np.inf);backs=[]
 for t in range(1,len(u)):
  z=(v[:,None]+pair[t-1])+u[t,None,:];a=np.argmax(z,axis=0);v=np.where(valid[t],z[a,np.arange(3)],-np.inf);backs.append(a)
 a=int(np.argmax(v));acts=[a]
 for prev in reversed(backs):a=int(prev[a]);acts.append(a)
 acts=np.array(acts[::-1]);return bank.boxes.numpy()[np.arange(len(acts)),acts]

def xyxy(b,W,H):
 b=np.asarray(b,float);return np.column_stack([b[:,0]-b[:,2]/2,b[:,1]-b[:,3]/2,b[:,0]+b[:,2]/2,b[:,1]+b[:,3]/2])*[W,H,W,H]

def derive(ds,row,reference=False):
 c,proof=old.source(ds,row);key=row['id'];inv=fixed(CHECK/'teacher_inventory.json')[ds][key];assert sha(inv['payload'])==inv['payload_sha256']and sha(inv['receipt'])==inv['receipt_sha256'];receipt=read(inv['receipt']);ctx=receipt['context'];raw=torch.load(inv['payload'],map_location='cpu',weights_only=False,mmap=True)['boxes'].numpy().astype(float);ids=np.asarray(c['ids']);pos=np.asarray(c['pos'],int);assert ctx['frame_ids']==ids.tolist()and ctx['positions']==pos.tolist()and ctx['caption']==row['metadata']['caption'];g=SpatialGeometry.from_cached_geometry(c['geometries']['main']);assert np.array_equal(raw,g.native_boxes.numpy());source=np.asarray(c['Source'],float);cfg=fixed(R/'configs/default.json')['datasets'][ds]
 if reference:E=independent_E(source,pos,raw,cfg)
 else:
  unscaled,_=initialize(source,pos,raw,cfg['E_calibration']);E=output_writeback(source,unscaled,cfg['initialization_gamma'])
 err=float(np.max(abs(E-c['E'])));assert err<2e-12
 if not reference:assert np.array_equal(E,c['E'])
 e=fixed(CHECK/ds/'sealed.json')['files'][key];assert sha(CHECK/e['path'])==e['sha256']
 with np.load(CHECK/e['path'])as z:iv=z['Full_interval'].copy();oldfull=z['Full'].copy();assert np.array_equal(ids,z['frame_ids'])and np.array_equal(pos,z['positions8'])
 assert ids[0]<=iv[0]<iv[1]<=ids[-1]+1;bank=g.for_interval(iv);full=numpy_dp(bank)if reference else evidence_dp(bank)['final'].numpy();assert np.array_equal(full,oldfull)
 m=row['metadata'];a=0 if ds=='hc2'else int(m['start_frame']);b=int(m['frame_count'])if ds=='hc2'else int(m['end_frame']);dense=np.arange(a,b);rawxy=xyxy(raw,m['width'],m['height']);pure=plain.interpolate(ids[pos],rawxy,dense)
 if reference:
  again=np.empty_like(pure)
  for j,t in enumerate(dense):
   if t<=ids[pos[0]]:again[j]=rawxy[0]
   elif t>=ids[pos[-1]]:again[j]=rawxy[-1]
   else:
    k=int(np.searchsorted(ids[pos],t,side='right')-1);w=float(t-ids[pos[k]])/float(ids[pos[k+1]]-ids[pos[k]]);again[j]=rawxy[k]+w*(rawxy[k+1]-rawxy[k])
  assert np.max(abs(pure-again))<2e-10;pure=again
 arrays=dict(frame_ids=ids,positions8=pos,shared_interval=iv,dense_input_ids=dense,PureExpertLinear_xyxy=pure,SharedCorrection=E,Full=full,raw_expert_xyxy=rawxy)
 return arrays,dict(parent=proof,raw_expert_payload_sha256=inv['payload_sha256'],raw_expert_receipt_sha256=inv['receipt_sha256'],input_RGB_context_sha256=ctx['RGB_sha256'],current_Full_reference_sha256=e['sha256'],raw8physicalobservations_exact=True,shared_interval_exact=True,E_reconstructed_from_raw_experts_and_original_Source=True,Full_final_DP_reexecuted=True,transport_fit='Unchanged original exact candidate/unary/link sufficient-state reused; no unavailable full feature grid assumed or new model/geometry refit claimed.',method_anchor_contract='Original Full soft E/expert choices unchanged; pure anchors remain raw expert. No hardanchor fix.',GT_read=False)
