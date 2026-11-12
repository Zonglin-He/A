from core82 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source
p=read(H82/'experiment/DESIGN_LOCK.json');pin(p);base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);rows=[]
for pl in ['drone','quad']:
 r=next(r for r in target_rows() if r['platform']==pl);_,z,d=item80(r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im)
 with torch.no_grad():refp,refq,_=m.main(base,z,d,'bidirectional')
 for wrong in [False,True]:
  g,delta,ini=gradient79(m,base,z,d,refp,refq,pm,im,ws,wrong);set_step(m,S,delta);rp=rawpatch82(r,wrong);error=max(float(abs(patch82(m)[n]-rp[n]).max()) for n in rp);assert error<1e-7;restore(m,S);_,patch,rec=search82(m,base,z,d,S,refp,refq,r,g,ini['objective_before'],pm,im,ws,wrong);assert not rec['accepted'] or rec['objective_after']<=rec['objective_before']+1e-10;rows.append({'platform':pl,'wrong_binding':wrong,'raw_parity_error':error,'alpha':rec['alpha'],'reuse':rec['reused_completed_R81_correct_search'],'gradient_norm':rec['gradient_norm']})
O=H82/'engineering_attempt001';O.mkdir();dump(O/'RESULT.json',{'state':'PASS_R82_REAL_INPUT_R80_DIRECTION_PARITY_AND_ACCEPTANCE','records':rows,'GT_read':False});print(read(O/'RESULT.json'))
