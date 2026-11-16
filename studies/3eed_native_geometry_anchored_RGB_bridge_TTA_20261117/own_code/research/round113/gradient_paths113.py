"""Postseal descriptive gradient audit: did global bypass change the auxiliary task?
No new model, optimization, task scoring or target selection.
"""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round113';B=R/'crossplatform/round113';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert (B/'MATRIX_SEALED.json').exists();rows=[]
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  src,_,tgt=name.split('_');schema=rd(B/(src+'_THETA_SCHEMA_PRIVATE.json'));mask=np.concatenate([np.full(n,k=='gate_I' or k.startswith('point_to_image.'),bool) for k,n in zip(schema['names'],schema['sizes'])]);assert mask.sum()==45313
  for f in (B/name/'records_PRIVATE').glob('*.json'):
   rec=rd(f)
   if rec['protected_alias_fallback']:continue
   with np.load(B/name/'fits_PRIVATE'/(f.stem+'.npz')) as z:a=z['gradient_full'][mask].astype(float);b=z['gradient_anchor'][mask].astype(float)
   na,nb=float(np.linalg.norm(a)),float(np.linalg.norm(b));cos=float(a@b/(na*nb)) if min(na,nb)>0 else None
   rows.append({'source':src,'target':tgt,'id_PRIVATE':f.stem,'bridge_full_aux_gradient_norm':na,'bridge_bypassed_aux_gradient_norm':nb,'bridge_aux_direction_cosine':cos,'full_auxiliary':rec['gradients']['full']['auxiliary_before'],'bypassed_auxiliary':rec['gradients']['anchor']['auxiliary_before'],'not_task_gradient_alignment':True})
 summary=[]
 for src in ['all','waymo','drone','quad']:
  rr=[x for x in rows if src=='all' or x['source']==src];cc=[x['bridge_aux_direction_cosine'] for x in rr if x['bridge_aux_direction_cosine'] is not None];summary.append({'source':src,'n':len(rr),'defined_cosines':len(cc),'mean_cosine':float(np.mean(cc)) if cc else None,'median_cosine':float(np.median(cc)) if cc else None,'negative_cosines':sum(x<0 for x in cc),'mean_full_auxiliary':float(np.mean([x['full_auxiliary'] for x in rr])),'mean_bypassed_auxiliary':float(np.mean([x['bypassed_auxiliary'] for x in rr])),'not_grounding_direction_correctness':True})
 (H/'AUXILIARY_PATH_GRADIENTS_PRIVATE.json').write_text(json.dumps(rows,indent=2)+'\n');(H/'AUXILIARY_PATH_GRADIENT_SUMMARY.json').write_text(json.dumps({'postseal_descriptive_only':True,'no_new_models_fits_or_task_scores':True,'rows':summary},indent=2)+'\n');print('GRADIENT_PATH113',len(rows),flush=True)
if __name__=='__main__':main()
