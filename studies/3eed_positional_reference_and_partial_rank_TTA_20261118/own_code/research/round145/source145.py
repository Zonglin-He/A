"""Conditional phase2: source-own image-reader calibration predictions, no GT."""
from common145 import *
sys.path.insert(0,str(R/'research/round141'))
from core141 import *
from common145 import R,H,B,rd,sha,dump,check,utc
from position145 import normalize145,savepos145
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()

def checksource():
 check()
 for p,h in rd(H/'experiment/SOURCE_CODE_LOCK.json')['files'].items():assert sha(R/p)==h,p

def main():
 checksource();reg=rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json');pairs={p['pair_PRIVATE']:p for p in rd(R/'research/round132/SOURCE_PAIRS_PRIVATE.json')};oldseal=rd(R/'crossplatform/round140/source/SEALED.json')['files'];root=B/('source_engineering' if args.engineering else 'source');root.mkdir(exist_ok=False);n=0;nb=0;forward=0;mx=0.;coords={(q['source'],q['id_PRIVATE']):q for q in rd(H/'SOURCE_POSITION_ROWS_PRIVATE.json')}
 if not args.engineering:assert rd(H/'SOURCE_ENGINEERING_AUDIT.json')['status']=='PASS'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);np.savez_compressed(root/(src+'_POSITION_MODULE_PRIVATE.npz'),**{k:v.detach().cpu().numpy() for k,v in base.pos_embed.state_dict().items()});D=root/src;D.mkdir();selected=[e for e in reg if e['entry']['source']==src]
  if args.engineering:selected=[e for e in selected if rd(R/'crossplatform/round140/source'/src/(e['entry']['row']['id']+'.json'))['has_B']][:1]
  for e in selected:
   tick=time.perf_counter();entry=e['entry'];r=entry['row'];sid=r['id'];restore(m,S);freeze91(m)
   for tag in ['prefix','geometry','route']:assert sha(R/entry['files'][tag]['path'])==entry['files'][tag]['sha256']
   for ext in ['.npz','.json']:assert sha(R/'crossplatform/round140/source'/src/(sid+ext))==oldseal[src+'/'+sid+ext]
   z0=load(R/entry['files']['prefix']['path']);gA=load(R/entry['files']['geometry']['path']);gA['query_anchor_indices']=z0['query_points_sample_inds'];rt=load(R/entry['files']['route']['path']);old=np.load(R/'crossplatform/round140/source'/src/(sid+'.npz'));hasB=rd(R/'crossplatform/round140/source'/src/(sid+'.json'))['has_B'];A=torch.as_tensor(z0['vis0'][0],device='cuda');valid=old['valid_B'];P=torch.where(torch.as_tensor(valid,device='cuda')[:,None],(A+torch.as_tensor(old['B_vis'],device='cuda'))/2,A);z=dict(tensor_prefix(z0),vis0=P[None]);delta=coords[src,sid]['delta'];assert coords[src,sid]['has_B']==hasB;z=normalize145(base,z,delta);out={'valid_B':valid,'assignment':old['assignment'],'S':old['S'],'F_vis':P.cpu().numpy()};savepos145(out,z,z0,delta);banks={};record={'has_B':hasB,'source':src,'source_or_target_GT_read':False,'FINAL_access':False,'source_parameters_updated':0,'delta':delta};assert hasB==bool(valid.any())
   sensor=e['A'];assert sensor['eligible_for_extra_media'] and not sensor['protected_image_content_alias']
   for kind in ['point','image']:assert sha(sensor[kind+'path'])==sensor[kind+'_sha256']
   if hasB:
    pair=pairs[e['pair_PRIVATE']];sel=rd(R/'crossplatform/round132/selections_PRIVATE'/('SOURCE_'+e['A']['frame_PRIVATE']+'.json'));assert sel['selected_rank'] and sel['selected_pair_PRIVATE']==e['pair_PRIVATE'];regp=R/sel['selected_registration'];rr=rd(regp);assert rr['geometry_cycle_supported'];b=pair['B'];assert b['eligible_for_extra_media'] and not b['protected_image_content_alias']
    for kind in ['point','image']:assert sha(b[kind+'path'])==b[kind+'_sha256']
    gB,pr=geo141.additional(e['A'],b,z0,rr);record.update(registration_path=sel['selected_registration'],registration_sha256=sha(regp),B_visible_points=int(gB['visible'].sum()));nb+=1
   else:gB=gA
   for tag,g in [('A',gA),('B',gB)]:
    p,a,gg=read_bank141(base,m,z,[g],rt,r);out['p'+tag]=p;forward+=1
    for key,val in a.items():out[tag+'_'+key]=val
    if tag=='B':
     for key in ['p2i','p2i_weight','i2p','i2p_weight','visible']:banks[key]=gg[key]
   if not hasB:assert np.array_equal(out['pA'],out['pB']) and abs(out['pA']-old['pA']).max()<1e-7
   if args.engineering:
    p,_,_=read_bank141(base,m,tensor_prefix(z0),[gA],rt,r);err=float(abs(p-old['pA']).max());mx=max(mx,err);assert err<1e-7;record['original_A_reader_bridge_error']=err;forward+=1
   assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());np.savez_compressed(D/(sid+'.npz'),**out);np.savez_compressed(D/(sid+'_graph.npz'),**banks);record.update(seconds=time.perf_counter()-tick,source_state_fingerprint_unchanged=True);dump(D/(sid+'.json'),record);n+=1
   if args.engineering or n%16==0:print('SOURCE145',args.engineering,src,n,flush=True)
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'B_inputs':nb,'complete_reader_forwards':forward,'max_original_A_bridge_error':mx,'source_or_target_GT_read':False,'source_training':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('SOURCE145_COMPLETE',n,nb,forward,flush=True)

if __name__=='__main__':main()
