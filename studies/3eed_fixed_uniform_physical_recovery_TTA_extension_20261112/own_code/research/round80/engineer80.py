"""Two real extension inputs and two DEV inputs: exact R79 operator parity."""
from core80 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'))
from model51 import load_source,fingerprint
p=read(H80/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H80/'experiment/CODE_LOCK.json'));base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);out=[]
rr=[next(r for r in target_rows() if r['platform']==pl) for pl in ['drone','quad']]+[next(r for r in allrows80()[336:] if r['platform']==pl) for pl in ['drone','quad']]
for i,r in enumerate(rr):
 z0,z,d=item80(r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im)
 with torch.no_grad():refp,refq,_=m.main(base,z,d,'bidirectional')
 old,od=old80(r);assert np.array_equal(old['source_boxes'],z0['boxes']);parity=None
 for wrong in [False,True]:
  g,delta,rec=gradient79(m,base,z,d,refp,refq,pm,im,ws,wrong);assert g.isfinite().all() and delta.norm()>0;set_step(m,S,delta)
  with torch.no_grad():pp,_,_=m.main(base,z,d,'bidirectional')
  if not wrong and r['id'] in DEV_IDS:
   parity=float(abs(pp.cpu().numpy()-od['uniform_sgd1']).max());assert parity<1e-10
   saved=load(ROOT/'crossplatform/round79/prediction464_attempt001/states_PRIVATE'/(r['id']+'_uniform_sgd1.npz'));assert all(np.array_equal(v.detach().cpu().numpy(),saved[n]) for n,v in m.tta_parameters())
  restore(m,S)
 out.append({'row':i+1,'platform':r['platform'],'R79_uniform_probability_parity':parity,'wrong_step_L2':rec['raw_proposal_L2'],'point_targets':int((pm[0]&d['visible']).sum()),'RGB_targets':int(im[0].sum()),'depth_targets':int(d['depth_valid'].sum())})
assert fingerprint(base)==fp;O=H80/'engineering_attempt001';O.mkdir();dump(O/'RESULT.json',{'state':'PASS_R80_REAL_DEV_AND_EXTENSION_OPERATOR_PARITY','rows':out,'GT_read':False,'source_training':False});print(read(O/'RESULT.json'))
