"""Intact full-reader source-own A/B predictions; never opens source labels."""
from common140 import *
sys.path.insert(0,str(R/'research/round136'));from optim136 import *
from common140 import R,H,B,rd,sha,dump,check,utc
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 check();reg=rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json');pairs={p['pair_PRIVATE']:p for p in rd(R/'research/round132/SOURCE_PAIRS_PRIVATE.json')};root=B/('source_engineering' if args.engineering else 'source');root.mkdir(exist_ok=False);n=0
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);D=root/src;D.mkdir();selected=[e for e in reg if e['entry']['source']==src]
  if args.engineering:
   selected=[e for e in selected if rd(R/'crossplatform/round132/source_calibration_PRIVATE'/src/(e['entry']['row']['id']+'.json'))['B_available']][:1]
  for e in selected:
   tick=time.perf_counter();entry=e['entry'];r=entry['row'];sid=r['id'];restore(m,S);freeze91(m)
   assert e['A']['eligible_for_extra_media'] and not e['A']['protected_image_content_alias'];z0=load(R/entry['files']['prefix']['path']);g=load(R/entry['files']['geometry']['path']);g['query_anchor_indices']=z0['query_points_sample_inds'];z=tensor_prefix(z0);d=tensor_data(g,r);rt=load(R/entry['files']['route']['path']);route=tensor_route89(rt)
   old=np.load(R/'crossplatform/round132/source_calibration_PRIVATE'/src/(sid+'.npz'));sel=rd(R/'crossplatform/round132/selections_PRIVATE'/('SOURCE_'+e['A']['frame_PRIVATE']+'.json')) if e['pair_PRIVATE'] else None
   obs,proof=extra132(base,e['A'],pairs[e['pair_PRIVATE']],z0,sel['selected_registration']) if sel and sel['selected_rank'] else (None,{'status':'no_source_B_registration'})
   A=z['vis0'][0];P=A if obs is None else obs['vis'];valid=np.zeros(1024,bool) if obs is None else obs['valid'].cpu().numpy();assert np.array_equal(valid,old['valid_B']);assert torch.equal(P[~torch.as_tensor(valid,device='cuda')],A[~torch.as_tensor(valid,device='cuda')]);out={'valid_B':valid,'B_vis':P.cpu().numpy()}
   with torch.no_grad():
    for tag,points in [('A',A),('B',P)]:
     lp,a,_=live135(m,base,z,d,points,route);out['p'+tag]=lp.exp().cpu().numpy();npfeatures(out,tag,a)
   geo=U129.geometry(z0['xyz'][0],z0['boxes'],valid);scope=U129.qualify(geo,4,.5);assert np.array_equal(geo['assignment'],old['assignment']) and np.array_equal(scope,old['S']);out.update(S=scope,assignment=geo['assignment']);np.savez_compressed(D/(sid+'.npz'),**out);dump(D/(sid+'_route.npz.json'),{'original_source_route_sha256':entry['files']['route']['sha256'],'source_route_is_frozen_prepared_pipeline_not_reestimated_from_GT':True});dump(D/(sid+'.json'),{'has_B':obs is not None,'source':src,'seconds':time.perf_counter()-tick,'source_state_fingerprint_unchanged':all(np.array_equal(v,state(m)[k]) for k,v in S.items()),'source_or_target_GT_read':False,'registration_proof':proof});assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());n+=1
   if args.engineering or n%16==0:print('SOURCE140',src,n,flush=True)
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'source_or_target_GT_read':False,'source_training':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('SOURCE140_COMPLETE',n,flush=True)
if __name__=='__main__':main()
