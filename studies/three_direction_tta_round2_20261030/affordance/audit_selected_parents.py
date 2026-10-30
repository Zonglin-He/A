"""Coordinate-only cross-split near-duplicate audit after immutable cohort selection, before scores."""
import pathlib,json,itertools
import numpy as np
from scipy.spatial import cKDTree
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parent

def rep(path):
 x=np.load(path,allow_pickle=False).astype(np.float64);x=x-x.mean(0);x/=np.linalg.norm(x,axis=1).max();e,v=np.linalg.eigh(x.T@x/len(x));return x@v[:,::-1]

def main():
 selected={}
 for tag in ['dev','confirm']:
  rows=json.loads((ROOT/f'cohorts/{tag}_inference.json').read_text());meta=json.loads((ROOT/f'cohorts/{tag}_metadata.json').read_text());selected[tag]=[{'id':r['id'],'object':m['object_class'],'pca':rep(r['geometry']),'axis_ambiguous':m['pca_axis_near_degenerate']} for r,m in zip(rows,meta)]
 old=json.loads((BASE/'data/inference_manifest.json').read_text());oldmeta=json.loads((BASE/'data/scoring_manifest.json').read_text());selected['old']=[{'id':r['id'],'object':m['object_class'],'pca':rep(r['geometry'])} for r,m in zip(old,oldmeta)]
 matches=[];rows=[]
 for left,right in [('dev','confirm'),('old','dev'),('old','confirm')]:
  count=0;closest=None
  for a in selected[left]:
   ta=cKDTree(a['pca'])
   for b in selected[right]:
    if a['object']!=b['object']:continue
    best=(float('inf'),float('inf'))
    for signs in itertools.product([-1,1],repeat=3):
     bb=b['pca']*np.array(signs);d=np.r_[ta.query(bb,k=1)[0],cKDTree(bb).query(a['pca'],k=1)[0]];score=(float(np.sqrt((d*d).mean())),float(np.quantile(d,.95)))
     if score<best:best=score
    count+=1
    if closest is None or best<(closest['rms'],closest['q95']):closest={'rms':best[0],'q95':best[1],'left_id':a['id'],'right_id':b['id'],'object':a['object']}
    if best[0]<=.01 and best[1]<=.025:matches.append({'left_split':left,'right_split':right,'left_id':a['id'],'right_id':b['id'],'rms':best[0],'q95':best[1]})
  rows.append({'left':left,'right':right,'all_same_object_category_pairs_audited':count,'closest_pair':closest})
 result={'kind':'coordinate_only_no_model_or_GT_score','passed_no_cross_split_near_matches_under_declared_rule':not matches,'matches':matches,'all_pairs_audit':rows,'selected_axis_ambiguities':{k:sum(x['axis_ambiguous'] for x in selected[k]) for k in ['dev','confirm']},'physical_instance_provenance_verified':False,'limitations':'Only normalized PCA/sign-aligned point-set distance with RMS<=.01 and q95<=.025 is ruled out across selected sets within same native object category. Original mesh identity, different surface samplings, and PCA-degenerate rotation ambiguities can still escape.'}
 (ROOT/'COHORT_SELECTED_PROVENANCE_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert not matches,'Cohort near-overlap detected; do not execute locked confirm or silently replace parents'
if __name__=='__main__':main()
