"""Finite instance-evidence study, setup and exact historical pool diagnosis.

No historical proposals are regenerated. Labels are postseal diagnostics only.
"""
from pathlib import Path
import ast,collections,copy,gc,hashlib,json,math,os,sys,textwrap,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
BASE=ROOT/'artifacts/stvg_instance_evidence32_v1'
OLD=ROOT/'artifacts/stvg_spatial_reference_geometry_v1'
LITE=ROOT/'artifacts/stvg_lite_event_support_v1'
SEARCH=ROOT/'artifacts/stvg_soft_ep_coordinate128_v1'
def put(p,d):
    assert not Path(p).exists(),str(p);write(p,d)
def setup():
    assert not BASE.exists();BASE.mkdir(parents=True)
    diagnostic=read(OLD/'ROSTER_LOCK.json')['records'];full=read(SEARCH/'ROSTER_LOCK.json')['records'];chosen=[]
    for ds in ('hc2','vidstg'):
        part=[r for r in diagnostic if r['dataset']==ds];have={r['parent_sha256']for r in part}
        candidates=sorted([r for r in full if r['dataset']==ds and r['parent_sha256']not in have],key=lambda r:(r['parent_sha256'],r['query_key_sha256']))
        chosen.extend(part+candidates[:16-len(part)])
    assert len(chosen)==32 and len({r['parent_sha256']for r in chosen})==32
    confirm=read(LITE/'confirmation/ROSTER_LOCK.json')['records'];assert not {r['parent_sha256']for r in chosen}&{r['parent_sha256']for r in confirm}
    put(BASE/'ROSTER_LOCK.json',dict(status='locked',count=32,records=chosen,GT_read=False,selection='the 12 already reviewed stratified development cases plus ten additional parent/query-SHA-ordered development sources per domain; no new metric selection',all_development_exposed=True))
    put(BASE/'DESIGN_LOCK.json',dict(status='locked',count=32,domain_counts={'hc2':16,'vidstg':16},sequence=['complete historical pool12 audit','corrected Temporal-Lite32 baseline','one evidence-unit change chosen from coverage diagnosis','E0/E1/E2 fixed32 true Student decode','conditional frozen confirmation and cross-backbone qualification'],
        temporal='original eight448 candidate pixels/Prompt/intervals; isolated fractional-time correction; same shared-candidate/no-L2/full-grid512 L-BFGS',
        spatial='same episodic1792 interface; E0/E1 response-guided Lite; E2 original validated OPD with exactly E1 references',
        lr={'hc2':.0045,'vidstg':.027},new_Teacher=False,new_optimization_search=False,prompt_search=False,bootstrap=False,CURRENT_change=False,official_fullset=False,
        pool12_scope='all actually shown Source and nonSource candidates, cached box/mask/RGB/ID bindings; no intermediate regeneration; GT only offline geometry diagnostics',
        conditional_branch='if available complete nonSource instances are missing, one sparse low-cost proposal modification; if present but missed, one explicit scene-instance binding modification; do not run both or retry variants',
        confirmation_rule='only one selected recipe, shared across domains and locked before confirmation; no confirmation-driven parameter/algorithm search; declare previous project exposure',GT_read=False))
    pins=[Path(__file__),ROOT/'scripts/stvg_qwen_uniform_clock002.py',ROOT/'scripts/stvg_temporal_lite_fit_v1.py',ROOT/'scripts/stvg_table1_temporal_adapter_v1.py',ROOT/'scripts/stvg_spatial_lite_fit_v1.py',LITE/'FINAL_RECIPE_LOCK.json',SEARCH/'ROSTER_LOCK.json',OLD/'ROSTER_LOCK.json',LITE/'reclocked_inputs/TEACHER_JOB_LOCK.json',LITE/'reclocked_inputs/CANDIDATE_INPUT_BARRIER.json']
    put(BASE/'RUNTIME.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in pins},GT_read=False))
    print(json.dumps({'status':'locked','count':32,'ids':[r['probe_alias']for r in chosen]}))
def verify():
    for rel,h in read(BASE/'RUNTIME.json')['pins'].items():assert sha(ROOT/rel)==h,rel
def pool12():
    sys.path.insert(0,str(ROOT/'artifacts/stvg_weightfree_spatial_v1/deps'))
    import cv2,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import run_stvg_table1_source_v1 as source
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    from scripts.diagnose_stvg_lite_confirmation_spatial_v1 import iou
    verify();rr=read(OLD/'ROSTER_LOCK.json')['records'];files=[];pools={}
    jobs={j['job_id']:j for j in read(LITE/'reclocked_inputs/TEACHER_JOB_LOCK.json')['records']}
    oldbar=read(LITE/'reclocked_inputs/CANDIDATE_INPUT_BARRIER.json');bindings={f['path']:f['sha256']for f in oldbar['files']}
    for r in rr:
        cid=r['probe_alias'];p=LITE/'reclocked_inputs/new_crop/proposals'/f'{cid}.json';pool=read(p);assert sha(p)==bindings[str(p.relative_to(ROOT))]
        for w in pool['windows']:
            mp=ROOT/w['mask_archive_path'];assert sha(mp)==w['mask_archive_sha256'];job=jobs[f'{cid}_temporal_new_crop_{w["slot"]}']
            assert [t['candidate_id']for t in w['tracks']]==[t['candidate_id']for t in job['candidate_bindings']]
            for video in job['videos']:assert sha(ROOT/video['path'])==video['sha256']
            files.append(dict(case_id=cid,slot=w['slot'],proposal_sha256=sha(p),mask_sha256=sha(mp),job_signature=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),candidate_count=len(w['tracks']),GT_read=False))
        pools[cid]=pool
    put(BASE/'POOL_INPUT_BARRIER.json',dict(status='complete',count=12,windows=len(files),records=files,GT_read=False,no_proposal_regeneration=True,original_barrier_sha256=sha(LITE/'reclocked_inputs/CANDIDATE_INPUT_BARRIER.json')))
    cases={r['probe_alias']:load_case(r)[0]for r in rr};targets,gtbind=truth._truths_after_seal(cases);results=[];font=ImageFont.truetype('DejaVuSans.ttf',15)
    for rec in rr:
        cid=rec['probe_alias'];case=cases[cid];pool=pools[cid];target=metric._truth_map(targets[cid]);req=read(LITE/'reclocked_inputs/requests'/f'{cid}.json');assert sha(Path(req['video_path']))==req['canonical_video_sha256'];_,plan=source._validate_roster_record(rec)
        canv=Image.new('RGB',(1200,2500),'white');draw=ImageDraw.Draw(canv);query=plan['input']['caption'];draw.text((8,5),cid+' / '+rec['dataset']+' / private whole-pool review',font=font,fill='black')
        for k,line in enumerate(textwrap.wrap(query,145)):draw.text((8,25+k*17),line,font=font,fill='black')
        for wi,w in enumerate(pool['windows']):
            fid=w['center_physical_frame_id'];li=next(x['local_frame_index']for x in req['frame_map']if x['physical_frame_id']==fid);cap=cv2.VideoCapture(req['video_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,li);ok,bgr=cap.read();cap.release();assert ok
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert hashlib.sha256(rgb.tobytes()).hexdigest()==w['rgb_sha256_by_physical_frame'][str(fid)]
            gt=(np.asarray(target[fid])/np.array([case['input']['width'],case['input']['height']]*2)).tolist()if fid in target else None
            known=[];mid=w['physical_frame_ids'].index(fid)
            with np.load(ROOT/w['mask_archive_path'],allow_pickle=False)as a:masks=a['masks']
            job=jobs[f'{cid}_temporal_new_crop_{wi}'];response=read(LITE/'reclocked_inputs/teacher'/f'{job["job_id"]}.json');selected=response['parsed'].get('selected_candidate_id')
            ybase=100+wi*595;scene=Image.fromarray(rgb).resize((480,320));sd=ImageDraw.Draw(scene)
            if gt:sd.rectangle([gt[0]*480,gt[1]*320,gt[2]*480,gt[3]*320],outline='lime',width=3)
            canv.paste(scene,(0,ybase));draw.text((5,ybase+325),f'window {wi}, selected {selected}; green=offline GT',font=font,fill='black')
            H,W=rgb.shape[:2]
            for j,t in enumerate(w['tracks']):
                box=t['center_box'];v=iou(box,gt)if box is not None and gt is not None else None
                known.append(dict(candidate_id=t['candidate_id'],role='Source'if t['candidate_id']==1 else'proposal',presence=t['presence'],selected=t['candidate_id']==selected,GT_iou=v,box_available=box is not None,identity_status='unknown_pending_region_review'))
                x=480+(j%3)*240;y=ybase+(j//3)*195;tile=Image.new('RGB',(236,190),(28,28,28));td=ImageDraw.Draw(tile)
                if box is not None:
                    a,b,c,d=box;dx=(c-a)*.15;dy=(d-b)*.15;xx=max(0,int((a-dx)*W));yy=max(0,int((b-dy)*H));xx2=min(W,math.ceil((c+dx)*W));yy2=min(H,math.ceil((d+dy)*H))
                    im=Image.fromarray(rgb)
                    if j:
                        mask=cv2.resize(masks[j-1,mid],(W,H),interpolation=cv2.INTER_NEAREST)>0;pix=rgb.copy();pix[mask]=(.6*pix[mask]+.4*np.array([255,0,0])).astype(np.uint8);im=Image.fromarray(pix)
                    im=im.crop((xx,yy,xx2,yy2));im.thumbnail((230,160));tile.paste(im,((236-im.width)//2,25));td=ImageDraw.Draw(tile)
                td.text((4,3),f'ID {t["candidate_id"]}'+(' SOURCE'if j==0 else'')+(' SELECTED'if t['candidate_id']==selected else''),font=font,fill='white');canv.paste(tile,(x,y))
            valid=[z for z in known if z['GT_iou']is not None];non=[z for z in valid if z['role']=='proposal'];best=max(non,key=lambda z:z['GT_iou'])if non else None
            results.append(dict(case_id=cid,dataset=rec['dataset'],stratum=rec['diagnostic_stratum'],slot=wi,candidates=known,known_GT=gt is not None,best_nonSource_iou=best['GT_iou']if best else None,Source_iou=valid[0]['GT_iou']if valid else None,GT_read=True))
        p=BASE/'private_pool_views'/f'{cid}.png';p.parent.mkdir(exist_ok=True);canv.save(p);print('POOL',cid,flush=True)
    put(BASE/'POOL12_POSTSEAL_DIAGNOSIS.json',dict(status='complete_pending_visual_review',count=12,window_count=48,records=results,GT_read=True,GT_bindings=gtbind,barrier_sha256=sha(BASE/'POOL_INPUT_BARRIER.json'),scope='all exact displayed historical candidate IDs; Source and nonSource coverage separate; IoU quality is not independent semantic identity proof; no label-derived reference or Oracle fit'))
    print(json.dumps({'known_windows':sum(w['known_GT']for w in results),'nonSource_iou_ge_0_5':sum(w['best_nonSource_iou']is not None and w['best_nonSource_iou']>=.5 for w in results),'source_iou_ge_0_5':sum(w['Source_iou']is not None and w['Source_iou']>=.5 for w in results)}))
if __name__=='__main__':{'setup':setup,'pool12':pool12}[sys.argv[1]]()
