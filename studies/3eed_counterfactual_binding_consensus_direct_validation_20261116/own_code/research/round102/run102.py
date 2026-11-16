"""No labels, models or weights: fixed cache contrast aggregation and prediction seal."""
from pathlib import Path
import json,hashlib,numpy as np,argparse
R=Path(__file__).resolve().parents[2];H=R/'research/round102';B=R/'crossplatform/round102';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def dump(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def soft(x):e=np.exp(x-np.max(x));return e/e.sum()
def main():
 global B
 p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
 if args.full:B=B/'full'
 refs=['original_source','prepared_frozen','binding_direct','direct_C','direct_FC','direct_four','lm_FC','joint3_FC','kl3_FC','kl3_shuffle_FC'];methods=['single0','single1','signcommon','scaledmean','shufflecommon'];arms=refs+[x+'_'+v for x in methods for v in ['F','C','FC']];seals={};n=0;stats=[]
 if args.full:refs=refs[:-2];arms=refs+[x+'_'+v for x in methods for v in ['F','C','FC']]
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;O=B/name;(O/'samples_PRIVATE').mkdir(parents=True,exist_ok=True);(O/'evidence_PRIVATE').mkdir(exist_ok=True);rows=rd(R/f'research/round97/input/TARGET_{tgt.upper()}_PRIVATE.json') if args.full else rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')
   for r in rows:
    sid=r['id'];ev=load(R/'crossplatform/round98'/name/'evidence_PRIVATE'/(sid+'.npz'));old=load(R/'crossplatform'/('round100' if args.full else 'round101')/name/'samples_PRIVATE'/(sid+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']));delta=(ev['correct'][None]-ev['wrong']).mean(1);delta-=delta.mean(1,keepdims=True);mean=delta.mean(0);assert np.max(abs(mean-ev['delta']))<1e-12;common=np.where(delta[0]*delta[1]>0,np.sign(delta[0])*np.minimum(abs(delta[0]),abs(delta[1])),0.);common-=common.mean();scale=np.linalg.norm(common)/max(np.linalg.norm(mean),1e-30)
    # Reuse the original fixed shuffle permutation, independently of task labels.
    perm=np.random.default_rng(int(hashlib.sha256(('R94querycontrol|'+r['id']).encode()).hexdigest()[:16],16)).permutation(256)
    shifts={'single0':delta[0],'single1':delta[1],'signcommon':common,'scaledmean':mean*scale,'shufflecommon':common[perm]};scores={k:oldp[k] for k in refs};pF=oldp['prepared_frozen'];old98=load(R/'crossplatform/round98'/name/'samples_PRIVATE'/(sid+'.npz'));pC=dict(zip(old98['arm_names'],old98['scores']))['frozen_C'];assert np.max(abs(soft(np.log(pF)+mean)-oldp['binding_direct']))<1e-12 and np.max(abs((soft(np.log(pF)+mean)+soft(np.log(pC)+mean))/2-oldp['direct_FC']))<1e-12
    for label,d in shifts.items():
     F=soft(np.log(pF)+d);C=soft(np.log(pC)+d)
     for v,z in [('F',F),('C',C),('FC',(F+C)/2)]:scores[label+'_'+v]=z
    arr=np.stack([scores[k] for k in arms]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-12)
    if args.full and (B.parent/name/'samples_PRIVATE'/(sid+'.npz')).exists():
     dev=load(B.parent/name/'samples_PRIVATE'/(sid+'.npz'));dp=dict(zip(dev['arm_names'],dev['scores']));assert all(np.array_equal(scores[k],dp[k]) for k in arms)
    np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(arms),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1),anno_index=old['anno_index']);np.savez_compressed(O/'evidence_PRIVATE'/(sid+'.npz'),contrasts=delta,mean_delta=mean,common_delta=common,shuffled_delta=shifts['shufflecommon']);stats.append({'source':src,'target':tgt,'id_PRIVATE':sid,'same_sign_fraction':float((delta[0]*delta[1]>0).mean()),'common_to_mean_L2_ratio':float(scale),'contrast_cosine':float(delta[0]@delta[1]/max(1e-30,np.linalg.norm(delta[0])*np.linalg.norm(delta[1]))),'single_source_top1_agrees':int(scores['single0_F'].argmax())==int(scores['single1_F'].argmax())});n+=1
   dump(O/'SEALED.json',{'n':len(rows),'files':{str(f.relative_to(O)):sha(f) for f in sorted(O.rglob('*.npz'))},'primary':'signcommon_FC','no_labels':True,'no_model_forward':True});seals[name]=sha(O/'SEALED.json');print('RUN102',name,flush=True)
 dump(B/'MATRIX_SEALED.json',{'n':n,'arms':arms,'directions':seals,'primary':'signcommon_FC'});dump(H/('FULL_CONTRAST_STATS_PRIVATE.json' if args.full else 'CONTRAST_STATS_PRIVATE.json'),stats);print('RUN102_COMPLETE',n,flush=True)
if __name__=='__main__':main()
