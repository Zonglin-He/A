"""GT vs anti-GT vs +/- orthogonal: no external spatial expert."""
import sys,argparse,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.run_st_causal_audit_v2 import OUT,plan
from scripts.decota_matrix_common_v1 import read,write,save,load,sha
from scripts.run_decota_refine_v1 import student,configure
from vg_tta.st_causal_audit_v2 import TubeReplay,TAReplay,center_roi
from vg_tta.matched_direction_diagnostics_v1 import spatial_directions
from vg_tta.dense_expansion_data_v1 import decode_raw
from vg_tta.foreground_runtime import state_digest
DEST=OUT/'gt_direction_focus'


def pixel_directions(pred,gt,width,height):
    p=np.array(pred,float);g=np.array(gt,float);scale=np.array([width,height],float);d=(g-p)*scale
    vv={'GT_direction':d,'anti_GT':-d,'orthogonal_plus':np.stack([-d[:,1],d[:,0]],1),'orthogonal_minus':np.stack([d[:,1],-d[:,0]],1)}
    vv={k:v/scale for k,v in vv.items()};alpha=np.ones(len(p))
    for v in vv.values():
        for j in range(2):
            pos=v[:,j]>0;neg=v[:,j]<0
            alpha[pos]=np.minimum(alpha[pos],(1-p[pos,j])/v[pos,j]);alpha[neg]=np.minimum(alpha[neg],-p[neg,j]/v[neg,j])
    alpha=np.clip(alpha,0,1);alpha[alpha<1]*=1-1e-12
    centers={k:p+alpha[:,None]*v for k,v in vv.items()};dd={k:(c-p)*scale for k,c in centers.items()};norms={k:np.linalg.norm(v,axis=1) for k,v in dd.items()}
    assert all(np.allclose(n,norms['GT_direction']) for n in norms.values())
    assert np.allclose((dd['GT_direction']*dd['orthogonal_plus']).sum(1),0,atol=1e-8)
    return centers,dict(alpha=alpha.tolist(),actual_displacement_pixels=norms['GT_direction'].tolist(),coordinate_system='original image pixels; common feasible scale')


def prepare():
    p=plan();rr=[r for r in p['rows']['hc_to_vid'] if r['diagnostic_GT']['full_track_centers'] is not None];assert len(rr)==8
    cells=[dict(gain=g,fraction=f,stage='all',step=1.,coordinates='normalized') for g in [2.,4.,8.] for f in [.125,.25,.5]]
    cells +=[dict(gain=4.,fraction=.25,stage=s,step=1.,coordinates='normalized') for s in ['early','middle','late']]
    cells +=[dict(gain=4.,fraction=.25,stage='all',step=a,coordinates='normalized') for a in [.25,.5]]
    cells +=[dict(gain=4.,fraction=.25,stage='all',step=1.,coordinates='pixels')]
    write(DEST/'lock.json',dict(rows=rr,parent_sha256=sha(OUT/'lock.json'),cells=cells,backbones=['tubedetr','tastvg'],
        primary='GT - mean(anti, orthogonal+, orthogonal-) at gain4 fraction.25 step1 all; GT-anti and GT-orth separate',
        exploratory='all other strengths/regions/stages; never choose best to claim significance',
        all_frames_identical_token_budget=True,no_expert=True,GT_diagnostic_only=True,formal_TTA=False,
        new_model='2026 PTD pending weights; apply same GT/anti/orth controls, not reflection/random substitutes',
        zero_controls=['native replay','explicit float-mask gain1','unhooked repeat'],created_unix=time.time()))


def run(b):
    p=plan();f=read(DEST/'lock.json');configure();assert f['parent_sha256']==sha(OUT/'lock.json');g='hc_to_vid'
    m=student(read(ROOT/'artifacts/decota_refine_v1/lock.json'),b,g);state=state_digest(m);receipts=[]
    for row in f['rows']:
        q=row['input'];dest=DEST/'runs'/b/f'{row["ordinal"]:03d}.pt'
        if not dest.exists():
            start=time.perf_counter();raw,ids=decode_raw(q);rr=TubeReplay(m,raw,ids,q) if b=='tubedetr' else TAReplay(m,raw,ids,q,row['student_subject']['subject']);base=rr.base
            old=load(OUT/'runs'/b/g/f'{row["ordinal"]:03d}.pt')['predictions']['native'];assert torch.equal(base['boxes'],old['boxes']) and torch.equal(base['logits'],old['logits'])
            pred=base['boxes'][:,:2].numpy();target=np.array(row['diagnostic_GT']['full_track_centers']);directions,da=spatial_directions(pred,target);pd,pa=pixel_directions(pred,target,q['width'],q['height'])
            out={};audits={}
            for i,c in enumerate(f['cells']):
                dc=directions if c['coordinates']=='normalized' else pd
                for direction,full in dc.items():
                    centers=pred+c['step']*(full-pred);rois=[center_roi(centers[pos],grid,c['fraction']) for pos,grid in zip(rr.offsets,rr.grids)]
                    assert all(len(set(roi.sum(1).tolist()))==1 for roi in rois)
                    z=rr.run(dict(roi=rois,gain=c['gain'],stage=c['stage']));out[f'{i:02d}_{direction}']=z
                    audits[f'{i:02d}_{direction}']=dict(centers=centers.tolist(),counts=[r.sum(1).tolist() for r in rois])
            rois=[center_roi(pred[pos],grid) for pos,grid in zip(rr.offsets,rr.grids)]
            out['float_mask_gain1']=rr.run(dict(roi=rois,gain=1.))
            out['native_center_gain4']=rr.run(dict(roi=rois,gain=4.))
            out['exact_GT_unmatched']=rr.run(dict(roi=[center_roi(target[pos],grid) for pos,grid in zip(rr.offsets,rr.grids)],gain=4.))
            repeated=rr.run();assert torch.equal(repeated['boxes'],base['boxes']) and torch.equal(repeated['logits'],base['logits'])
            save(dest,dict(native=base,predictions=out,query=row,normalized_geometry=da,pixel_geometry=pa,
                per_arm_audit=audits,no_expert=True,GT_diagnostic_only=True,no_op_float_delta={k:float((out['float_mask_gain1'][k]-base[k]).abs().max()) for k in ['boxes','logits']},
                code_sha256=sha(__file__),protocol_sha256=sha(DEST/'lock.json'),seconds=time.perf_counter()-start))
            print('GT_DIRECTION',b,row['ordinal'],len(out),round(time.perf_counter()-start,2),flush=True);del rr,out,raw;gc.collect();torch.cuda.empty_cache()
        receipts.append(dict(path=str(dest),sha256=sha(dest),ordinal=row['ordinal']))
    assert state_digest(m)==state;write(DEST/f'barrier_{b}.json',dict(receipts=receipts,full_state_unchanged=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);ap.add_argument('--backbone',choices=['tubedetr','tastvg']);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    else:run(a.backbone)
