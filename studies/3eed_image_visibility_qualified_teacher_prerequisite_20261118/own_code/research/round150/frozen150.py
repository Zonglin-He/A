"""New frozen teacher-routing outputs only. All native features come from pinned R145."""
from visibility150 import *
import collections
sys.path.insert(0,str(R/'research/round136'));from prepare136 import direct
from common150 import R,H,B,rd,sha,dump,check,utc

def main():
 check();out=B/'frozen';out.mkdir(exist_ok=False);evdir=B/'visibility_PRIVATE';evdir.mkdir(exist_ok=False);reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');parent=rd(R/'crossplatform/round149/method/SEALED.json')['files'];n=0;counts=collections.Counter()
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;D=out/name;D.mkdir(exist_ok=True);a=np.load(R/'crossplatform/round149/method'/name/'samples_PRIVATE'/(sid+'.npz'));rel=name+'/samples_PRIVATE/'+sid+'.npz';assert sha(R/'crossplatform/round149/method'/rel)==parent[rel];ps=dict(zip(a['arm_names'],a['scores']));old=np.load(R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz'));scores={};vals={k:old[k] for k in ['assignment','S','C','pA','pB','eligible']};gs=[np.flatnonzero(old['assignment']==g) for g in range(int(old['assignment'].max()+1))]
  if bool(old['eligible']):
   z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));bb=np.load(R/'crossplatform/round145/image'/name/'banks_PRIVATE'/(sid+'.npz'));g,nA,nB,nn=supports(z['xyz'][0],z['boxes'],old['assignment'],bb['norm_Aimage_visible'],bb['norm_Bimage_visible']);assert all(np.array_equal(x,y) for x,y in zip(g,gs));v=evidence150(old['pA'],old['pB'],gs,old['S'],nA,nB,src+'|'+sid);vals.update(v,n_total=nn);counts['RGB_scope_inputs_ge2']+=int(v['S_rgb'].sum()>=2);counts['old_S_groups']+=int(old['S'].sum());counts['new_S_groups']+=int(v['S_rgb'].sum());counts['wrong_equal_effective']+=int(np.max(abs(v['alpha_equal'][v['S_rgb']]-v['alpha_wrong_equal'][v['S_rgb']]),initial=0)>1e-12)
   for suffix,key in [('equal','alpha_equal'),('count','alpha_count'),('wrong_equal','alpha_wrong_equal'),('wrong_count','alpha_wrong_count'),('global_equal','alpha_global_equal'),('global_count','alpha_global_count')]:scores['vis_'+suffix]=route_prob(old['pA'],old['pB'],gs,v['S_rgb'],v[key])
   for suffix in ['equal','wrong_equal','global_equal','count']:
    p=scores['vis_'+suffix];scores['vis_'+suffix+'_Cteacher']=force_C(p,gs,old['C']);tar=np.array([p[g].max() for g in gs])[old['C']];tar/=tar.sum();scores['vis_'+suffix+'_oldC_direct']=direct(ps['norm_joint_zero'],gs,old['S'],old['C'],tar)
   scores['old_Cteacher']=force_C(old['pB'],gs,old['C'])
  else:
   vals.update(S_rgb=np.zeros_like(old['S']),nA=np.zeros(len(gs),int),nB=np.zeros(len(gs),int),n_total=np.zeros(len(gs),int))
   for suffix in ['equal','count','wrong_equal','wrong_count','global_equal','global_count']:scores['vis_'+suffix]=ps['Frozen_A'].copy()
   for suffix in ['equal','wrong_equal','global_equal','count']:
    scores['vis_'+suffix+'_Cteacher']=ps['Frozen_A'].copy();scores['vis_'+suffix+'_oldC_direct']=ps['Frozen_A'].copy()
   scores['old_Cteacher']=ps['Frozen_A'].copy()
  arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=a['source_boxes']);ed=evdir/name;ed.mkdir(exist_ok=True);np.savez_compressed(ed/(sid+'.npz'),**vals);n+=1
  if n%128==0:print('FROZEN150',n,flush=True)
 dump(out/'SEALED.json',dict(n=n,models=0,GT_read=False,FINAL_access=False,parent_seal_sha256=sha(R/'crossplatform/round149/method/SEALED.json'),files={str(p.relative_to(out)):sha(p) for p in out.rglob('*.npz')}));dump(B/'VISIBILITY_SEALED.json',dict(n=n,GT_read=False,files={str(p.relative_to(B)):sha(p) for p in evdir.rglob('*.npz')}));dump(H/'FROZEN_PREPARATION.json',dict(counts)|dict(n=n,new_models=0,labels_read=False));print('FROZEN150_SEALED',n,dict(counts),flush=True)
if __name__=='__main__':main()
