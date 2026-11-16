"""Post-seal full DEV mechanism replay. No new predictions selected or scored."""
from core101 import *
def main():
 assert jread(H101/'SCALAR_AUDIT.json')['n']==768;out=[];summary=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);head=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;known={r['id_PRIVATE']:r for r in jread(H101/'scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'))}
   for r in jread(R101/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
    sid=r['id'];ff=load(R101/'crossplatform/round100'/name/'features_PRIVATE'/(sid+'.npz'));views=[{k:torch.as_tensor(ff[str(i)+'_'+k],device='cuda') for k in ['h','u','late','mask']} for i in range(3)];ev=load(B98/name/'evidence_PRIVATE'/(sid+'.npz'));r0=torch.as_tensor(ev['r0'],device='cuda');delta=torch.as_tensor(ev['delta'],device='cuda');target=(torch.as_tensor(ev['correct'],device='cuda')+r0+delta).softmax(1);cc=load(B98/name/'features_PRIVATE'/(sid+'.npz'));C={k:torch.as_tensor(cc['C_'+k],device='cuda') for k in ['h','u','late','mask']};kt=known[sid];z={'source':src,'target':tgt,'id_PRIVATE':sid,'IoU_delta_KL_vs_L2':kt['arms']['kl3_FC']['IoU']-kt['arms']['joint3_FC']['IoU'],'IoU_delta_KL_vs_direct':kt['arms']['kl3_FC']['IoU']-kt['arms']['direct_FC']['IoU']};qs={}
    for label,st in [('source',head),('L2',load(R101/'crossplatform/round100'/name/'states_PRIVATE'/(sid+'_joint3.npz'))),('KL',load(B101/name/'states_PRIVATE'/(sid+'_kl3.npz')))]:
     put94(m,st)
     with torch.no_grad():
      p=probabilities101(m,views,r0);c=read94(m,C,r0);fc=(p[0]+c)/2;diff=p.log()-target.log();diff-=diff.mean(1,keepdim=True);z[label+'_fit_KL']=float(loss101(p,target));z[label+'_fit_L2']=float(diff.square().mean()/2);qs[label]=(int(p[0].argmax()),int(c.argmax()),int(fc.argmax()))
     if label in ['L2','KL']:
      prefix='joint3' if label=='L2' else 'kl3';assert qs[label]==tuple(kt['arms'][prefix+'_'+v]['query_PRIVATE'] for v in ['F','C','FC'])
    z.update(F_top1_changed=qs['KL'][0]!=qs['L2'][0],C_top1_changed=qs['KL'][1]!=qs['L2'][1],FC_top1_changed=qs['KL'][2]!=qs['L2'][2]);out.append(z)
   print('DIAG101',name,flush=True)
  del m,base;torch.cuda.empty_cache()
 for label,rr in [('all',out),('KL_improves',[r for r in out if r['IoU_delta_KL_vs_L2']>1e-12]),('KL_harms',[r for r in out if r['IoU_delta_KL_vs_L2']< -1e-12])]:
  summary.append({'group':label,'n':len(rr),**{k:float(np.mean([r[k] for r in rr])) if rr else None for k in ['source_fit_KL','L2_fit_KL','KL_fit_KL','source_fit_L2','L2_fit_L2','KL_fit_L2']},'FC_query_changes':sum(r['FC_top1_changed'] for r in rr),'FC_only_confidence_switches':sum(r['FC_top1_changed'] and not r['F_top1_changed'] and not r['C_top1_changed'] for r in rr),'KL_fits_KL_better_n':sum(r['KL_fit_KL']<r['L2_fit_KL'] for r in rr)})
 D=H101/'mechanism';D.mkdir(exist_ok=False);jdump(D/'ALL_ROWS_PRIVATE.json',out);jdump(D/'SUMMARY.json',{'state':'POSTSEAL_NATIVE_HEAD_METRIC_AND_DECISION_DIAGNOSTIC','n':len(out),'new_task_scores':0,'new_method_predictions':0,'groups':summary});print(summary)
if __name__=='__main__':main()
