"""Independent saved-mask readback, then postseal geometric diagnosis/views.

No new teacher selection, parameter fit, or label-derived reference is made.
"""
from pathlib import Path
import collections,gc,hashlib,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_reference_geometry_audit_v1 import BASE,LITE,read,put,sha,verify

def root():
    import numpy as np
    from scripts.stvg_weightfree_spatial_v1 import bbox
    verify();bar=read(BASE/'GEOMETRY_PREDICTION_BARRIER.json');assert bar['status']=='complete'and bar['count']==12 and bar['GT_read']is False
    assert bar['roster_sha256']==sha(BASE/'ROSTER_LOCK.json')and bar['runtime_sha256']==sha(BASE/'RUNTIME.json')
    rows=[]
    for f in bar['files']:
        p=ROOT/f['path'];assert sha(p)==f['sha256'];d=read(p);assert d['GT_read']is False
        hist=LITE/'reclocked_inputs/new_crop/proposals'/f"{f['case_id']}.json";assert sha(hist)==d['historical_proposal_sha256'];old=read(hist)
        for w,ow in zip(d['windows'],old['windows']):
            assert w['physical_frame_id']==ow['center_physical_frame_id']
            oldpath=ROOT/ow['mask_archive_path'];assert sha(oldpath)==ow['mask_archive_sha256']
            with np.load(oldpath,allow_pickle=False)as a:masks=np.asarray(a['masks']).copy()
            mid=ow['physical_frame_ids'].index(w['physical_frame_id']);assert len(w['candidates'])==len(ow['tracks'])-1
            for j,(c,oc)in enumerate(zip(w['candidates'],ow['tracks'][1:])):
                assert c['candidate_id']==oc['candidate_id']and c['proposal_rank']==oc['proposal_rank']and c['historical_component_box']==oc['center_box']
                if 'mask_path'in c:
                    ap=ROOT/c['mask_path'];assert sha(ap)==c['mask_sha256']
                    with np.load(ap,allow_pickle=False)as a:component=a['component'];full=a['full_foreground']
                    assert bbox(component)==c['replayed_component_box']and bbox(full)==c['replayed_full_foreground_box']
                    exact=np.array_equal(component,masks[j,mid])and bbox(component)==oc['center_box'];assert bool(exact)==c['anchor_mask_and_box_exact']
                    assert np.all((component==0)|(full!=0))
                else:assert not c['anchor_mask_and_box_exact']
            assert w['full_pool_exact']==all(c['anchor_mask_and_box_exact']and c['rank_order_exact']and c['score_exact']for c in w['candidates']) or not w['full_pool_exact']
        rows.append(dict(case_id=d['case_id'],dataset=d['dataset'],stratum=d['stratum'],window_count=len(d['windows']),full_pool_exact=sum(w['full_pool_exact']for w in d['windows']),selected_anchor_count=sum(len(w['selected_candidate_ids'])for w in d['windows']),selected_anchor_exact=sum(c['anchor_mask_and_box_exact']for w in d['windows']for c in w['candidates']if c['candidate_id']in w['selected_candidate_ids'])))
    put(BASE/'ROOT_GEOMETRY_READBACK.json',dict(status='pass',count=12,rows=rows,GT_read=False,CPU_only=True,barrier_sha256=sha(BASE/'GEOMETRY_PREDICTION_BARRIER.json'),scope='all returned component/full masks rechecked against bbox/subset and historical exact masks; not a second selective-search execution or proof of semantic identity'))
    print(json.dumps({'status':'pass','rows':rows}))

def diagnosis():
    # Exact same OpenCV ABI as the input producer.
    sys.path.insert(0,str(ROOT/'artifacts/stvg_weightfree_spatial_v1/deps'))
    import cv2,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import run_stvg_table1_source_v1 as source
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    from scripts.diagnose_stvg_lite_confirmation_spatial_v1 import iou
    assert read(BASE/'ROOT_GEOMETRY_READBACK.json')['status']=='pass'
    records=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in records}
    # The complete twelve-source pre-label barrier and root are prerequisites.
    targets,bindings=truth._truths_after_seal(cases)
    outputs=[];allanchors=[];font=ImageFont.truetype('DejaVuSans.ttf',15)
    for rec in records:
        cid=rec['probe_alias'];case=cases[cid];replay=read(BASE/'replay'/f'{cid}.json');old=read(LITE/'reclocked_inputs/new_crop/proposals'/f'{cid}.json');request=read(LITE/'reclocked_inputs/requests'/f'{cid}.json')
        _,plan=source._validate_roster_record(rec);query=plan['input']['caption'];target=metric._truth_map(targets[cid]);width,height=case['input']['width'],case['input']['height']
        wrows=[];canvas=Image.new('RGB',(1280,1020),'white');draw=ImageDraw.Draw(canvas)
        import textwrap
        qlines=textwrap.wrap(query,150)
        draw.text((10,8),f'{cid} {rec["dataset"]} {rec["diagnostic_stratum"]}  [private Query + GT view, never publish]',font=font,fill='black')
        for k,line in enumerate(qlines[:3]):draw.text((10,32+k*18),line,font=font,fill='black')
        draw.text((10,92),'RED historical R0/mask | PURPLE full foreground R1 | ORANGE raw rectangle R2 | WHITE Source | GREEN GT',font=font,fill='black')
        for wi,(w,ow)in enumerate(zip(replay['windows'],old['windows'])):
            fid=w['physical_frame_id'];local=next(r['local_frame_index']for r in request['frame_map']if r['physical_frame_id']==fid)
            cap=cv2.VideoCapture(request['video_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,local);ok,bgr=cap.read();cap.release();assert ok
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert hashlib.sha256(rgb.tobytes()).hexdigest()==ow['rgb_sha256_by_physical_frame'][str(fid)]
            gt=(np.asarray(target[fid])/np.array([width,height,width,height])).tolist()if fid in target else None
            chosen=[c for c in w['candidates']if c['candidate_id']in w['selected_candidate_ids']]
            known=[]
            for c in chosen:
                boxes={'R0':c['historical_component_box'],'Source':w['source_box']}
                if c['anchor_mask_and_box_exact']:boxes['R1']=c['replayed_full_foreground_box']
                if w['reconstructed_rectangle_qualified']and c['anchor_mask_and_box_exact']:boxes['R2']=c['replayed_rectangle_box']
                metrics={name:iou(box,gt)if gt is not None and box is not None else None for name,box in boxes.items()}
                row=dict(case_id=cid,dataset=rec['dataset'],stratum=rec['diagnostic_stratum'],window_slot=w['slot'],candidate_id=c['candidate_id'],exact_replayed_anchor=c['anchor_mask_and_box_exact'],full_pool_exact=w['full_pool_exact'],known_GT_anchor=gt is not None,reference_iou=metrics,component_count=c.get('component_count'),full_to_component_pixel_ratio=c.get('full_foreground_pixels',0)/max(1,c.get('component_pixels',0))if 'mask_path'in c else None,identity='pending_visual_review',GT_read=True)
                allanchors.append(row);known.append(row)
            wrows.append(dict(slot=w['slot'],historical_selected_candidate_ids=w['selected_candidate_ids'],full_pool_exact=w['full_pool_exact'],known_GT_anchor=gt is not None,selected=known))
            panel=Image.fromarray(rgb).resize((640,400));pd=ImageDraw.Draw(panel)
            # Original masks are shown, even when controlled replay differs.
            if chosen:
                with np.load(ROOT/ow['mask_archive_path'],allow_pickle=False)as a:masks=a['masks']
                for c in chosen:
                    m=masks[c['candidate_id']-2,ow['physical_frame_ids'].index(fid)]
                    tint=cv2.resize(m,(640,400),interpolation=cv2.INTER_NEAREST)>0
                    pixels=np.asarray(panel).copy();pixels[tint]=(.6*pixels[tint]+.4*np.array([255,0,0])).astype(np.uint8);panel=Image.fromarray(pixels);pd=ImageDraw.Draw(panel)
                    for box,color,label in [(c['historical_component_box'],'red','R0'),(c.get('replayed_full_foreground_box'),'magenta','R1 REPLAY'),(c.get('replayed_rectangle_box'),'orange','R2 REPLAY')]:
                        if box is not None:
                            xx=[box[0]*640,box[1]*400,box[2]*640,box[3]*400];pd.rectangle(xx,outline=color,width=3);pd.text((xx[0],max(0,xx[1]-18)),label,font=font,fill=color,stroke_width=1,stroke_fill='black')
            for box,color in [(w['source_box'],'white'),(gt,'lime')]:
                if box is not None:pd.rectangle([box[0]*640,box[1]*400,box[2]*640,box[3]*400],outline=color,width=2)
            x=(wi%2)*640;y=130+(wi//2)*440;canvas.paste(panel,(x,y));draw=ImageDraw.Draw(canvas)
            draw.text((x+5,y+402),f'window {w["slot"]}, selected {w["selected_candidate_ids"]}; exact pool={w["full_pool_exact"]}; GT={gt is not None}',font=font,fill='black')
        dest=BASE/'private_views'/f'{cid}.png';dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();canvas.save(dest)
        outputs.append(dict(case_id=cid,dataset=rec['dataset'],stratum=rec['diagnostic_stratum'],windows=wrows,view_sha256=sha(dest)))
        print('POSTSEAL',cid,'anchors',sum(len(w['selected'])for w in wrows),flush=True)
    put(BASE/'POSTSEAL_GEOMETRY_DIAGNOSIS.json',dict(status='complete_pending_identity_review',count=12,records=outputs,anchors=allanchors,GT_read=True,new_fit=False,new_teacher=False,GT_bindings=bindings,barrier_sha256=sha(BASE/'GEOMETRY_PREDICTION_BARRIER.json'),root_sha256=sha(BASE/'ROOT_GEOMETRY_READBACK.json'),scope='exact GT observation positions only; no labels used for reconstruction or selection; nonexact historical intermediates excluded from R1/R2 geometry comparisons'))

if __name__=='__main__':{'root':root,'diagnosis':diagnosis}[sys.argv[1]]()
