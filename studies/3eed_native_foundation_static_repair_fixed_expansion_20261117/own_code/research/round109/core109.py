"""Fixed R106/R108 experiment, registration path made explicit for expansion."""
from pathlib import Path
import sys
R109=Path(__file__).resolve().parents[2];H109=R109/'research/round109';B109=R109/'crossplatform/round109';sys.path.insert(0,str(R109/'research/round108'));from core108 import *
FAMILIES109=['real','randomA','wrong'];PRIMARY109='real_softtoken_geometric_fusedABC'
REF109=['original_source','prepared_frozen','frozen_four','direct_FC','joint3_FC']
FAMILY_ARMS109=['BBF_B','BBF_C','BBF_ensemble','BBF_fusedABC','softtoken_fusedABC','softtoken_geometric_fusedABC','softtoken_arithmetic_fusedABC','softtoken_shuffled_geometric_fusedABC']
ARMS109=REF109+[k+'_'+v for k in FAMILIES109 for v in FAMILY_ARMS109]+[p+'_A' for p in POLICIES108]
def extra109(base,A,pair,z0,kind,slot):
 regpath=B109/'registration_PRIVATE'/(pair['pair_PRIVATE']+'.json');reg=jread(regpath)
 if not reg['geometry_cycle_supported']:return None,{'status':reg['status']}
 sensor=pair['B'] if kind=='real' else A;assert sensor['eligible_for_extra_media'];actual=sha(sensor['pointpath']);expected=reg['B_sensor' if kind=='real' else 'A_sensor']['point_sha256'];assert actual==expected;raw=geo.points(sensor);assert np.isfinite(raw).all();rng=np.random.default_rng(int(hashlib.sha256(('R106sample|'+pair['pair_PRIVATE']+'|'+kind+'|'+slot).encode()).hexdigest()[:16],16));ids=rng.choice(len(raw),16384,replace=len(raw)<16384);pts=np.asarray(raw[ids],float)
 if kind=='real':pts=geo.transform_points(pts,np.asarray(reg['forward']['T_raw_B_to_A']));fullraw=geo.transform_points(np.asarray(raw,float),np.asarray(reg['forward']['T_raw_B_to_A']))
 else:fullraw=np.asarray(raw,float)
 M=geo.raw_to_model_transform(geo.camera(A));pts=np.ascontiguousarray(geo.transform_points(pts,M),np.float32);full=geo.transform_points(fullraw,M)
 with torch.no_grad():out=base.backbone_net(torch.as_tensor(pts,device='cuda')[None]);v=out['fp2_features'][0].T;xyz=out['fp2_xyz'][0].cpu().numpy();aligned,valid,proof=interpolation106(z0['xyz'][0],xyz,v,full)
 # Unsupported canonical points retain the immutable source A feature.
 aa=torch.as_tensor(z0['vis0'][0],device='cuda');aligned=torch.where(valid[:,None],aligned,aa);proof.update(status='usable' if valid.any() else 'no_seed_alignment',kind=kind,sample_points=16384,point_sha256=actual,sample_sha256=hashlib.sha256(pts.tobytes()).hexdigest(),registration_sha256=sha(regpath));return ({'vis':aligned,'valid':valid} if valid.any() else None),proof
