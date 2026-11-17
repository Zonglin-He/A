"""Original B unchanged; qualify only already-selected late B for original missing-B cases."""
from common146 import *
import numpy as np,collections,copy
from scipy.spatial import cKDTree
sys.path.insert(0,str(R/'crossplatform/round7'));from camera_projection import raw_to_model_points
sys.path.insert(0,str(R/'research/round129'));import support129 as U

def main():
 check();parent=rd(R/'research/round145/INPUT_REGISTRY_PRIVATE.json');cand={x['A']['frame_PRIVATE']:x for x in rd(R/'research/round132/TARGET_B3_CANDIDATES_PRIVATE.json')};oldpos={(x['source'],x['target'],x['id_PRIVATE']):x for x in rd(R/'research/round144/POSITION_ROWS_PRIVATE.json')};sourceH=rd(R/'research/round144/PREPARATION.json')['source_own_median_q10'];matrix=rd(R/'crossplatform/round132/MATRIX_SEALED.json');content={x['content_group_PRIVATE']:x for x in rd(R/'research/round131/CONTENT_GROUPS_PRIVATE.json')};reg=[];pos=[];pins={};cnt=collections.Counter();maxnear=0.;cache={};B.mkdir(exist_ok=True)
 for original in parent:
  e=copy.deepcopy(original);src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;selpath=R/'crossplatform/round132/selections_PRIVATE'/('TARGET_'+e['frame_PRIVATE']+'.json');sel=rd(selpath);e['original_has_B']=e['has_B'];e['backfilled']=bool(not e['has_B'] and sel['selected_rank'] in [2,3]);pm=copy.deepcopy(oldpos[src,tgt,sid]);cnt['n']+=1
  if e['backfilled']:
   chosen=next(x for x in cand[e['frame_PRIVATE']]['candidates'] if x['rank']==sel['selected_rank']);pair=chosen['pair'];assert pair['pair_PRIVATE']==sel['selected_pair_PRIVATE'] and sel['no_GT_or_semantic_or_score_selection'];assert all(not x['geometry_cycle_supported'] for x in sel['attempts'] if x['rank']<sel['selected_rank']);regp=R/sel['selected_registration'];rr=rd(regp);assert rr['geometry_cycle_supported'];attempt=next(x for x in sel['attempts'] if x['rank']==sel['selected_rank']);assert sha(regp)==attempt['sha256'] and attempt['geometry_cycle_supported']
   for tag,sensor in [('A',pair['A']),('B',pair['B'])]:
    assert sensor['eligible_for_extra_media'] and not sensor['protected_image_content_alias'];g=content[sensor['content_group_PRIVATE']];assert not g['protected_content_group'] and not g['old_development_content']
    for kind in ['point','image']:assert sha(sensor[kind+'path'])==sensor[kind+'_sha256'];assert rr[tag+'_sensor']['point_sha256']==sensor['point_sha256']
   assert pair['A']['frame_PRIVATE']==e['A_sensor']['frame_PRIVATE'] and pair['A']['point_sha256']==e['A_sensor']['point_sha256'] and pair['A']['image_sha256']==e['A_sensor']['image_sha256'];root=R/'crossplatform/round132/formal'/name;sp=root/'SEALED.json';assert sha(sp)==matrix['seals'][name];seal=rd(sp)['files'];f=root/'features_PRIVATE'/(sid+'_B3.npz');assert sha(f)==seal['features_PRIVATE/'+sid+'_B3.npz'];old=np.load(f);assert old['B_vis'].shape==(1024,288) and old['valid_B'].shape==(1024,) and old['valid_B'].any();pred=root/'samples_PRIVATE'/(sid+'.npz');assert sha(pred)==seal['samples_PRIVATE/'+sid+'.npz'];z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));assert np.array_equal(np.load(pred)['source_boxes'],z['boxes']);assert np.array_equal(old['B_vis'][~old['valid_B']],z['vis0'][0,~old['valid_B']]);gg=U.geometry(z['xyz'][0],z['boxes'],old['valid_B']);ss=U.qualify(gg,4,.5)
   for k in ['assignment','n_A','n_B']:assert np.array_equal(gg[k],old[k])
   assert np.array_equal(ss,old['S']);out=B/'support_PRIVATE'/name/(sid+'.npz');out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,assignment=gg['assignment'],S=ss,n_A=gg['n_A'],n_B=gg['n_B'],coverage_B=gg['coverage_B']);A=e['A_sensor'];key=A['pointpath']+'|'+hashlib.sha256(z['point_indices'].tobytes()).hexdigest()+'|'+tgt+'|'+str(e['row'].get('suffix'))
   if key not in cache:
    fp=Path(A['pointpath']);raw=np.load(fp,allow_pickle=False) if fp.suffix=='.npy' else np.fromfile(fp,np.float32).reshape(-1,4);xyz=raw_to_model_points(raw[z['point_indices'],:3],tgt,int(e['row'].get('suffix') or 0));cache[key]=(float(np.quantile(xyz[:,2],.1)),xyz)
   q,xyz=cache[key];near=float(cKDTree(xyz).query(z['xyz'][0])[0].max());maxnear=max(maxnear,near);assert near<1e-4;pm.update(has_B=True,q10=q,delta=sourceH[src]-q,source_reference=sourceH[src],max_seed_input_coordinate_error=near);e.update(has_B=True,pair_PRIVATE=pair,registration_path=str(regp.relative_to(R)),registration_sha256=sha(regp),backfill_features_path=str(f.relative_to(R)),backfill_features_sha256=sha(f),selected_B_rank=sel['selected_rank']);cnt['backfilled']+=1;cnt['new_scope_qualifies']+=bool(ss.sum()>=2)
   for p in [selpath,regp,f,sp,pred,out]:pins[str(p.relative_to(R))]=sha(p)
  else:cnt['unchanged']+=1
  cnt['effective_B']+=e['has_B'];cnt['remaining_noB']+=not e['has_B'];reg.append(e);pos.append(pm)
 dump(H/'INPUT_REGISTRY_PRIVATE.json',reg);dump(H/'POSITION_ROWS_PRIVATE.json',pos);eng=[]
 for src,tgt in sorted({(e['source'],e['target']) for e in reg}):
  for kind in ['backfilled','original_has_B','remaining_noB']:
   found=next((e for e in reg if (e['source'],e['target'])==(src,tgt) and (not e['has_B'] if kind=='remaining_noB' else e[kind])),None)
   if found:eng.append(found)
 dump(H/'ENGINEERING_INPUTS_PRIVATE.json',eng);out=dict(cnt)|{'new_physical_frames':len({e['frame_PRIVATE'] for e in reg if e['backfilled']}),'new_sampledraws':len(cache),'max_seed_bridge_error':maxnear,'old_B_not_replaced':True,'GT_read':False,'source_kappas_fixed_R145':True,'new_source_training':False,'FINAL_access':False};dump(H/'PREPARATION.json',out)
 for p in [H/'INPUT_REGISTRY_PRIVATE.json',H/'POSITION_ROWS_PRIVATE.json',H/'ENGINEERING_INPUTS_PRIVATE.json',H/'PREPARATION.json']:pins[str(p.relative_to(R))]=sha(p)
 dump(H/'INPUTS_SEAL.json',{'files':pins,'GT_read':False});print(out,flush=True)
if __name__=='__main__':main()
