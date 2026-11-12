"""Freeze full-context native role supports and all observations before adaptation."""
from core88 import *
from routing88 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();start=time.perf_counter();base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));m=Fusion88(base.contrastive_align_projection_image).cuda().eval();S=load(ROOT/'crossplatform/round76/trained_source_attempt002/states_PRIVATE/bidirectional_e06.npz');restore(m,S);m.requires_grad_(False);fp=fingerprint(base);rows=read(H88/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[next(r for r in rows if r['platform']==p) for p in ['drone','quad']]
O=B88/('engineering_evidence_attempt001' if args.engineering else 'evidence336_attempt001');(O/'samples_PRIVATE').mkdir(parents=True,exist_ok=False);(O/'records_PRIVATE').mkdir();stats=[]
for nr,r in enumerate(rows):
 z0,z,d=item80(r);a=[]
 hook=base.cross_encoder.layers[-1].cross_layer.cross_lv.register_forward_hook(lambda _m,_a,o:a.append(o[1].detach().cpu().numpy()[0]))
 with torch.no_grad():p,q,ex=m.main(base,z,d,'bidirectional')
 hook.remove();assert len(a)==1;old=load(B87/'prediction336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(p.cpu().numpy(),old['scores'][old['arm_names'].tolist().index('frozen_bi')]);assert np.array_equal(z0['xyz'][0,z0['query_points_sample_inds']],z0['query_anchor_xyz'])
 graph=read(B87/'role_inventory336_attempt003/cases_PRIVATE'/(r['id']+'.json'))['graph'];doc=read(ROOT/'crossplatform/round48/parse336_attempt001/dependencies_PRIVATE'/(r['id']+'.json'));roles,off=token_roles88(base,r,z0,graph,doc);g=load(casepath(r));zz={**z0,'frozen_probability':p.cpu().numpy()};route,info=build88(r,zz,g,graph,roles,a[0]);np.savez_compressed(O/'samples_PRIVATE'/(r['id']+'.npz'),**route,native_role_attention=a[0],source_boxes=z0['boxes'],token_offsets_PRIVATE=off);dump(O/'records_PRIVATE'/(r['id']+'.json'),{'graph':graph,'counts':info});stats.append({'row':'DEV'+str(nr+1).zfill(4),'platform':r['platform'],**info})
 if (nr+1)%64==0:print('PREPARE88',nr+1,flush=True)
assert fp==fingerprint(base);dump(O/'COUNTS.json',{'n':len(rows),'native_suffix_calls':len(rows),'seconds':time.perf_counter()-start,'task_GT_read':False});seal(O,'R88_CONTEXTUAL_ROLE_SPATIAL_HYPOTHESES_SEALED',n=len(rows));dump(H88/('ENGINEERING_EVIDENCE_ROWS.json' if args.engineering else 'EVIDENCE_PREPARATION_ROWS.json'),stats);print(read(O/'COUNTS.json'))
