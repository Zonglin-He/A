"""Real-input loss parity, analytical zero paths, and label-free accepted updates."""
from core81 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source
p=read(H81/'experiment/DESIGN_LOCK.json');pin(p);base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);rr=[next(r for r in target_rows() if r['platform']==pl) for pl in ['drone','quad']];records=[]
for j,r in enumerate(rr):
 z0,z,d=item80(r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im)
 with torch.no_grad():refp,refq,_=m.main(base,z,d,'bidirectional');parts=parts81(m,z,d,pm,im,ws);orig=torch.stack([m.auxiliary(z,d,'bidirectional',a,b)[0] for a,b in zip(pm,im)]).mean();assert float(abs(sum(parts.values())-orig))<1e-7
 gg,initial=gradients81(m,z,d,pm,im,ws)
 for scope in ['point_adapter','point_to_image','image_to_point']:assert float(scope_gradient(m,gg,scope,'D').norm())==0
 assert float(scope_gradient(m,gg,'image_to_point','RGB').norm())==0
 assert torch.equal(scope_gradient(m,gg,'all','D'),scope_gradient(m,gg,'image_adapter','D'))
 for scope,sig in [('all','ALL'),('image_adapter','D'),('point_adapter','P')]:
  pp,patch,rec=step81(m,base,z,d,S,refp,refq,r,scope,sig,gg,initial,pm,im,ws);assert np.isfinite(pp).all();assert not rec['accepted'] or rec['objective_after']<=rec['objective_before']+1e-10;records.append({'row':j+1,**{k:v for k,v in rec.items() if k!='attempts'}})
O=H81/'engineering_attempt001';O.mkdir();dump(O/'RESULT.json',{'state':'PASS_R81_REAL_INPUT_NATIVE_LOSS_ZERO_GRADIENT_PATHS_AND_ACCEPTANCE','records':records,'GT_read':False});print(read(O/'RESULT.json'))
