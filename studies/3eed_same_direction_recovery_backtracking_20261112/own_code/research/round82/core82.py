"""R80 fixed-direction acceptance; completed R81 correct searches are reused."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round81'))
from core81 import *
H82=ROOT/'research/round82';B82=ROOT/'crossplatform/round82'
GRID82=[.5/2**k for k in range(9)]
ARMS82=['source','frozen_bi','frozen_direct3','R80_raw_correct','R80_raw_wrong','bt_correct','bt_wrong','frozen_ABC','R59']
def put_patch(m,patch):
 with torch.no_grad():
  for n,v in m.tta_parameters():v.copy_(torch.as_tensor(patch[n],device='cuda'))
def patch82(m):return {n:v.detach().cpu().numpy().copy() for n,v in m.tta_parameters()}
def aux82(m,z,d,pm,im,ws,wrong=False):
 vals=[]
 for a,b,w in zip(pm,im,ws):
  P,I,D=m.local_losses(z,d,a,b,wrong);vals.append(.1*(w['P']*P).sum()+.1*(w['I']*I).sum()+.01*(w['D']*D).sum())
 return torch.stack(vals).mean()
def trial82(m,base,z,d,refp,refq,pm,im,ws,wrong):
 pp,q,_=m.main(base,z,d,'bidirectional');keep=(refp*(refp.clamp_min(1e-300).log()-pp.clamp_min(1e-300).log())).sum()+.1*(q-refq).square().sum(-1).mean()/2;aux=aux82(m,z,d,pm,im,ws,wrong)
 return pp,float(aux.double()+keep),float(aux),float(keep)
def old81(r):return B81/('dev336_attempt001' if r['id'] in DEV_IDS else 'extension2304_attempt001')
def rawpatch82(r,wrong):
 sid=r['id'];arm='uniform_wrong_sgd1' if wrong else 'uniform_sgd1';folder=(ROOT/'crossplatform/round79/prediction464_attempt001' if sid in DEV_IDS and not wrong else B80/'prediction2640_attempt001');return load(folder/'states_PRIVATE'/(sid+'_'+arm+'.npz'))
def search82(m,base,z,d,S,refp,refq,r,g,before,pm,im,ws,wrong):
 direction=-(1/(g.norm()+1e-6)).clamp_max(1)*g;records=[];reuse=False;maxerr=0.;pp=refp;alpha=0.;L=before;aux=before;keep=0.
 # A completed search is reused only with robust trace margins and an actual
 # native-R80-gradient/state/objective parity check. Near-boundary cases replay.
 if not wrong:
  old=read(old81(r)/'records_PRIVATE'/(r['id']+'.json'))['all__ALL'];trace=old['attempts'];safe=old['accepted'] and all(t['objective']-t['bound']>1e-7 for t in trace[:-1])
  if safe:
   a=old['alpha'];saved=load(old81(r)/'states_PRIVATE'/(r['id']+'_all__ALL.npz'));set_step(m,S,a*direction);native=patch82(m);maxerr=max(float(np.max(abs(native[n]-saved[n]))) for n in saved);assert maxerr<=1e-7
   put_patch(m,saved)
   with torch.no_grad():v,lv,av,kv=trial82(m,base,z,d,refp,refq,pm,im,ws,False)
   bound=before+1e-4*float(g.double()@(a*direction).double());reuse=np.isfinite(lv) and lv<=bound+1e-10
   if reuse:
    pp,L,aux,keep,alpha=v,lv,av,kv,a;records=[{**t,'reused_R81_trace':True} for t in trace];assert np.max(abs(pp.cpu().numpy()-load(old81(r)/'samples_PRIVATE'/(r['id']+'.npz'))['scores'][list(load(old81(r)/'samples_PRIVATE'/(r['id']+'.npz'))['arm_names']).index('all__ALL')]))<1e-10
 if not reuse:
  restore(m,S)
  if float(g.norm())>1e-12:
   for a in GRID82:
    delta=a*direction;set_step(m,S,delta)
    with torch.no_grad():v,lv,av,kv=trial82(m,base,z,d,refp,refq,pm,im,ws,wrong)
    bound=before+1e-4*float(g.double()@delta.double());ok=np.isfinite(lv) and lv<=bound+1e-10;records.append({'alpha':a,'objective':lv,'bound':bound,'accepted':bool(ok),'reused_R81_trace':False})
    if ok:pp,L,aux,keep,alpha=v,lv,av,kv,a;break
  if alpha==0:restore(m,S)
 actual=torch.cat([(v-torch.as_tensor(S[n],device='cuda')).detach().flatten() for n,v in m.tta_parameters()]);patch=patch82(m)
 with torch.no_grad():hp,hi,hw=holdout81(r,d);held=float(aux82(m,z,d,hp,hi,hw,False))
 record={'wrong_binding':wrong,'alpha':alpha,'beta':alpha/.5,'accepted':alpha>0,'gradient_norm':float(g.norm()),'parameter_step_L2':float(actual.norm()),'objective_before':before,'objective_after':L,'auxiliary_after':aux,'preservation_after':keep,'heldout_true_auxiliary_after':held,'query_changed':int(pp.argmax())!=int(refp.argmax()),'reused_completed_R81_correct_search':reuse,'max_native_direction_state_parity_error':maxerr,'attempt_count':len(records),'attempts':records}
 restore(m,S);return pp.detach().cpu().numpy(),patch,record
