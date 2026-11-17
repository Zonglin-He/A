"""All finite source teacher predictions sealed before any new label access."""
from head149 import *
def main():
 check();torch.set_num_threads(4);reg=rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json');seal=rd(R/'crossplatform/round145/source/SEALED.json');out=B/'source_head';out.mkdir(exist_ok=False);n=0;nb=0
 for src in ['waymo','drone','quad']:
  W={k:torch.from_numpy(v.copy()) for k,v in dict(np.load(R/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz'))).items() if k.startswith('match.')};D=out/src;D.mkdir()
  for e in [e for e in reg if e['entry']['source']==src]:
   sid=e['entry']['row']['id'];p=R/'crossplatform/round145/source'/src/(sid+'.npz');j=p.with_suffix('.json');assert sha(p)==seal['files'][src+'/'+p.name] and sha(j)==seal['files'][src+'/'+j.name];a=np.load(p);has=rd(j)['has_B'];feat={k:a['B_'+k] for k in ['h','u','late','mask']};b=base_logits(feat,W);alphas=rd(H/'experiment/DESIGN_LOCK.json')['source_alpha_candidates'];pred=np.stack([calibrated(a['pB'],b,x) for x in alphas]);np.savez_compressed(D/p.name,base=b,p=pred,alphas=alphas);n+=1;nb+=has
 dump(out/'SEALED.json',dict(n=n,B=nb,GT_read=False,new_full_network_forwards=0,matching_head_replays=n,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*.npz')}));print('SOURCE149_SEALED',n,nb,flush=True)
if __name__=='__main__':main()
