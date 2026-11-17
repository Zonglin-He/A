"""Lock all crop recipes and raw-pixel provenance before any new model/readout."""
from crop151 import *
import collections

def main():
 check();out=B/'recipes_PRIVATE';out.mkdir(exist_ok=False);reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');eng=[];rows=[];n=0;counts=collections.Counter()
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;recipes=[]
  if e['has_B']:
   z=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));ev=load(R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz'));gg,pp,images=target_geometry(e,z);recipes=select_recipes(z,ev,gg,pp,images);counts['B']+=1;counts['regions']+=len(recipes);counts['A_regions']+=sum(r['view']==0 for r in recipes);counts['B_regions']+=sum(r['view']==1 for r in recipes)
  d=out/name;d.mkdir(exist_ok=True);dump(d/(sid+'.json'),dict(recipes=recipes,source_parameters_updated=0,target_GT_read=False));rows.append(dict(source=src,target=tgt,id_PRIVATE=sid,regions=len(recipes),A_regions=sum(r['view']==0 for r in recipes),B_regions=sum(r['view']==1 for r in recipes)));n+=1
  if n%128==0:print('PREP151',n,flush=True)
 for src,tgt in sorted({(e['source'],e['target']) for e in reg}):
  rr=[e for e in reg if (e['source'],e['target'])==(src,tgt)]
  for active in [True,False]:
   for e in rr:
    re=rd(out/(src+'_to_'+tgt)/(e['row']['id']+'.json'))['recipes']
    if bool(re)==active:eng.append(e);break
 dump(H/'PREPARATION.json',{k:int(v) for k,v in counts.items()}|dict(n=n,new_models=0,GT_read=False));dump(H/'RECIPE_ROWS_PRIVATE.json',rows);dump(H/'ENGINEERING_INPUTS_PRIVATE.json',eng);dump(B/'RECIPES_SEALED.json',dict(n=n,GT_read=False,files={str(p.relative_to(B)):sha(p) for p in out.rglob('*.json')}));print('PREP151_COMPLETE',n,dict(counts),flush=True)
if __name__=='__main__':main()
