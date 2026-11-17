"""Separate B3 acquisition OR original-B log-gap, never combined."""
from core132 import *
def original132(e):
 name=e['source']+'_to_'+e['target'];sid=e['row']['id'];D=R132/'crossplatform/round131/formal'/name
 with np.load(D/'samples_PRIVATE'/(sid+'.npz')) as a:scores=dict(zip(['zero','direct','partial','wrong'],a['scores']));boxes=a['source_boxes'].copy()
 return D,scores,boxes
def target_case132(base,m,S,e,kind,anchors,pair_index,engineering=False):
 D,old,boxes=original132(e);sid=e['row']['id'];name=e['source']+'_to_'+e['target'];a=anchors[e['frame_PRIVATE']];sel=rd(B132/'selections_PRIVATE'/('TARGET_'+e['frame_PRIVATE']+'.json'));scopepath=B132/'log_scopes_PRIVATE'/name/(sid+'.npz')
 if kind=='B3' and sel['selected_rank'] in [None,1]:return old,None,None,{'reference_original':True,'reason':'first_B_unchanged_or_no_alternative_pass','selected_rank':sel['selected_rank']},boxes
 if kind=='LOG':
  with np.load(scopepath) as f:logscope={k:f[k].copy() for k in f.files}
 if kind=='LOG' and bool(logscope['same_labels']):return old,None,None,{'reference_original':True,'reason':'log_labels_identical','selected_rank':1},boxes
 with np.load(D/'features_PRIVATE'/(sid+'.npz')) as f:oldf={k:f[k].copy() for k in f.files}
 if kind=='LOG' and not bool(logscope['active']):
  ff={'fixed_late':oldf['fixed_late'],'fixed_r':oldf['fixed_r'],'mask':oldf['mask'],'pA':oldf['pA'],'pB':oldf['pB'],'A_h':oldf['teacher_A_h'],'A_u':oldf['teacher_A_u'],'B_h':oldf['teacher_B_h'],'B_u':oldf['teacher_B_u'],'zero_h':oldf['zero_F_h'],'zero_u':oldf['zero_F_u'],**{k:oldf[k] for k in ['n_A','n_B','valid_B']},**{k:logscope[k] for k in ['S','C','wrong_C','assignment','teacher_conditional']}}
  for label in ['partial','wrong']:ff[label+'_h']=ff['zero_h'];ff[label+'_u']=ff['zero_u']
  states={k:np.zeros(4896,np.float32) for k in ['partial','wrong']};scores={k:old['zero'].copy() for k in ['zero','direct','partial','wrong']};return scores,ff,states,{'reference_original':False,'no_new_GPU':True,'reason':'log_C_equals_S_or_no_supported_scope','partial_eligible':False,'scope_eligible':bool(logscope['S'].sum()>=2),'B_available':bool(oldf['valid_B'].any()),'effective_shuffle':False,'identity_control':False,'parameter_radius':0.,'traces':{'partial':[],'wrong':[]},'selected_rank':1},boxes
 restore(m,S);z0=load(D/'prefix_PRIVATE'/(sid+'.npz'));assert np.array_equal(z0['boxes'],boxes);z=tensor_prefix(z0);d=geomdata125(e,z0);late=torch.as_tensor(oldf['fixed_late'],device='cuda');res=torch.as_tensor(oldf['fixed_r'],device='cuda')
 if kind=='B3':pid=sel['selected_pair_PRIVATE'];pair=pair_index[pid];regpath=sel['selected_registration']
 else:pid=a['B_pair'];pair=pair_index[pid];regpath=str((R132/'crossplatform/round131/registration_PRIVATE'/(pid+'.json')).relative_to(R132))
 obs,proof=extra132(base,a['A'],pair,z0,regpath);A=z['vis0'][0];B=A if obs is None else obs['vis'];valid=torch.zeros(1024,device='cuda',dtype=torch.bool) if obs is None else obs['valid'];FAB=torch.where(valid[:,None],(A+B)/2,A);start123(m,S);ff={'fixed_late':late.cpu().numpy(),'fixed_r':res.cpu().numpy(),'mask':z['token_mask'].cpu().numpy(),'B_vis':B.cpu().numpy()};probs={};errors={}
 with torch.no_grad():
  for tag,P in [('A',A),('B',B),('zero',FAB)]:
   lp,h,u=body127(m,base,z,d,P,late,res,True);p=lp.exp().cpu().numpy();ff[tag+'_h']=h.cpu().numpy();ff[tag+'_u']=u.cpu().numpy();probs[tag]=p
   if tag=='A' or kind=='LOG':
    ref=oldf['pA'] if tag=='A' else (oldf['pB'] if tag=='B' else old['zero']);err=float(abs(p-ref).max());errors[tag]=err;assert err<1e-7,(kind,tag,err);assert p.argmax()==ref.argmax()
 sg=support_geometry(z0['xyz'][0],boxes,valid.cpu().numpy());cal=rd(R132/'research/round129/SOURCE_CALIBRATION.json');support,C,T,_=partial_sets(probs['A'],probs['B'],sg,cal)
 if kind=='LOG':
  assert np.array_equal(support,logscope['S']) and np.array_equal(sg['assignment'],logscope['assignment']);C=logscope['C'];T=logscope['teacher_conditional'];probs['zero']=old['zero']
 ff.update(pA=probs['A'],pB=probs['B']);views={'features':ff,'zero':probs['zero'],'salt':e['source']+'|'+sid};scores,ff,states,rec=fit132(base,m,S,z0,d,late,res,A,B,valid,views,sg,support,C,T,engineering);rec.update(reference_original=False,selected_rank=sel['selected_rank'] if kind=='B3' else 1,support=proof,old_bridge_errors=errors,kind=kind);return scores,ff,states,rec,boxes
