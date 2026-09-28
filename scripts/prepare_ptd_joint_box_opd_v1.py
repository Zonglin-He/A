"""Source/one-query selection and scorer-only labels; no model outcomes used."""
import collections,hashlib,json,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,status,sha,load
from scripts.prepare_official_dense_pool_v2 import hc_source_id,frame_ids_from_segment
from scripts.prepare_stvg_fullscale_v1 import HC,SEVEN,_probe,xywh
OUT=ROOT/'artifacts/ptd_joint_box_opd_v1';DATA=ROOT/'data/ptd_joint_box_opd_v1'
def digest(s):return hashlib.sha256(s.encode()).hexdigest()

def select():
    old=read(ROOT/'artifacts/ptd_dense_teacher_audit_v1/INPUTS.json');excluded={r['source'] for r in old}
    pool=read(ROOT/'artifacts/spatial_consolidation_roles_v1/LOCK.json');selection=[]
    for co,n in [('hcstvg1_test',50),('vidstg_test',128)]:
        rr=sorted([r for r in pool['rows'].values() if r['cohort']==co and r['source'] not in excluded],key=lambda r:digest('joint-box-v1-source|'+r['source']))
        assert len(rr)>=n
        for j,r in enumerate(rr[:n]):selection.append(dict(**r,split='development' if j<32 else 'confirmation',engineering=j<4,official_split='test',domain='HC' if co.startswith('hc') else 'Vid'))
    ann=read(HC/'train.json');parents=collections.defaultdict(list)
    for name in ann:parents[hc_source_id(name)].append(name)
    for j,source in enumerate(sorted(parents,key=lambda s:digest('joint-box-v1-source|'+s))[:78]):
        name=min(parents[source],key=lambda s:digest('joint-box-v1-query|'+s))
        selection.append(dict(key=f'hcstvg1_train:{j:06d}',cohort='hcstvg1_train',source=source,name=name,split='confirmation',engineering=False,official_split='train',domain='HC'))
    assert len(selection)==256 and len({r['source'] for r in selection})==256 and not {r['source'] for r in selection}&excluded
    train_witness=read(ROOT/'artifacts/c1_fresh_confirmation_v1/EXECUTED_EXPOSURE_WITNESSES.json')['witnesses']['HC1_official_train']
    for r in selection:
        r['historically_exposed']=True
        r['exposure_witness']=train_witness[r['source']] if r['official_split']=='train' else dict(path=r['path'],sha256=r['sha256'],evidence='historical TA cached model episode')
    write(OUT/'SELECTION.json',dict(rows=selection,time=time.time(),excluded_old16=sorted(excluded),outcome_selection=False,
        pool_sha=sha(ROOT/'artifacts/spatial_consolidation_roles_v1/LOCK.json'),HC_train_annotation_sha=sha(HC/'train.json'),
        authorized_HC_train_supplement=True,HC_test_available_after_old8=50,counts=dict(development=64,confirmation=192,HC_train_confirmation=78,HC_test_confirmation=18,Vid_test_confirmation=96)))
    return selection

def media(selection):
    chosen=[r for r in selection if r['official_split']=='train'];names={r['name'] for r in chosen}
    paths=subprocess.check_output(['rg','--files','data','external','artifacts'],text=True,cwd=ROOT).splitlines();found={}
    for f in paths:
        if Path(f).name in names:
            p=ROOT/f
            if p.is_file():found.setdefault(p.name,str(p))
    missing=names-set(found);print('HC_MEDIA existing',len(found),'extract',len(missing),flush=True)
    if missing:
        DATA.mkdir(parents=True,exist_ok=True)
        with open(OUT/'intake_extract.log','ab') as log:
            proc=subprocess.Popen([str(SEVEN),'x','-so',str(HC/'video(5660)/v1.zip'),'v1.tgz'],stdout=subprocess.PIPE,stderr=log)
            try:
                with tarfile.open(fileobj=proc.stdout,mode='r|gz') as tar:
                    for member in tar:
                        name=Path(member.name).name
                        if not member.isfile() or name not in missing:continue
                        dest=DATA/name;tmp=dest.with_suffix('.extracting')
                        with tar.extractfile(member) as src,open(tmp,'wb') as f:
                            while block:=src.read(8<<20):f.write(block)
                        assert tmp.stat().st_size==member.size;tmp.replace(dest);found[name]=str(dest)
                        status(OUT/'INTAKE_STATUS.json',dict(stage='extracting',found=len(found),total=len(names)))
                        if len(found)==len(names):break
                # All requested members have been consumed; stop the decompressor intentionally.
            finally:
                proc.stdout.close()
                if proc.poll() is None:proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
    assert set(found)==names
    write(OUT/'HC_MEDIA.json',dict(paths=found))
    return found

def prepare():
    selection=read(OUT/'SELECTION.json')['rows'] if (OUT/'SELECTION.json').exists() else select()
    found=read(OUT/'HC_MEDIA.json')['paths'] if (OUT/'HC_MEDIA.json').exists() else media(selection)
    if (OUT/'OFFICIAL_MEDIA_VERIFICATION.json').exists():
        found={name:v['path'] for name,v in read(OUT/'OFFICIAL_MEDIA_VERIFICATION.json')['members'].items()}
    if (OUT/'INTAKE_QUERY_AMENDMENT.json').exists():
        amend=read(OUT/'INTAKE_QUERY_AMENDMENT.json')
        selection=[{**r,'name':amend['replacement']} if r.get('name')==amend['original'] else r for r in selection]
    rawann=read(HC/'train.json');oldlabels=read(ROOT/'artifacts/stvg_fullscale_diagnostics_v1/labels_diagnostic_only.json');labels={};rows=[];issues=[];vidor=None
    for ordinal,r in enumerate(selection):
        if r['official_split']=='test':
            assert sha(r['path'])==r['sha256'];parent=load(r['path']);q=parent['input'];lab=oldlabels[r['key']]
            provenance=dict(parent_file=r['path'],parent_sha=r['sha256'])
        else:
            p=found[r['name']];ann=rawann[r['name']];m=_probe(p)
            assert (m['width'],m['height'])==(ann['width'],ann['height'])
            assert ann['img_num']-m['frame_count'] in [0,1,2]
            ids=frame_ids_from_segment(frame_count=m['frame_count'],fps=m['fps'],start_frame=0,end_frame=m['frame_count'],max_frames=200)
            start=ann['st_frame']-1;interval=[start,start+len(ann['bbox'])];assert 0<=start<interval[1]<=m['frame_count']
            track={start+j:xywh(box,m['width'],m['height']) for j,box in enumerate(ann['bbox'])}
            valid=[interval[0]<=fid<interval[1] and fid in track and min(track[fid][2:])>0 for fid in ids]
            lab=dict(interval=interval,boxes=[track[fid] if ok else [0.,0.,0.,0.] for fid,ok in zip(ids,valid)],valid=valid,annotation=r['name'])
            q=dict(caption=ann['caption'],source=r['source'],original_video_id=Path(p).stem,video_path=p,video_sha256=sha(p),frame_ids=ids,start_frame=0,end_frame=m['frame_count'],index=ordinal,kind='hcstvg',**m)
            provenance=dict(parent_file=None,parent_sha=None)
        if len(q['frame_ids'])<32:
            # The inherited 5fps grid can be short although actual media has >=32 distinct frames.
            # Keep source/query, sample actual segment pixels, and rebuild scorer-only GT on that grid.
            from scripts.prepare_vidstg_wrong_domain_support import load_vidor_annotations,trajectory_for_target
            if vidor is None:vidor=load_vidor_annotations(ROOT/'downloads/vidor/validation-annotation.zip',None)
            ids=np.linspace(q['start_frame'],q['end_frame']-1,32).round().astype(int).tolist();assert len(set(ids))==32
            target=lab['official_annotation']['target_id'];track=trajectory_for_target(vidor[r['source']],target)
            track={int(fid):xywh(box['bbox'],q['width'],q['height']) for fid,box in track.items()}
            valid=[lab['interval'][0]<=fid<lab['interval'][1] and fid in track and min(track[fid][2:])>0 for fid in ids]
            lab={**lab,'boxes':[track[fid] if ok else [0.,0.,0.,0.] for fid,ok in zip(ids,valid)],'valid':valid}
            q={**q,'frame_ids':ids};issues.append(dict(key=r['key'],issue='inherited_5fps_grid_under32',resolution='32 distinct measured segment frames; source/query unchanged; GT sidecar rebuilt from official trajectory'))
        pos=np.unique(np.linspace(0,len(q['frame_ids'])-1,min(32,len(q['frame_ids']))).round().astype(int)).tolist()
        assert len(pos)==32
        rows.append(dict(key=r['key'],source=r['source'],cohort=r['cohort'],domain=r['domain'],split=r['split'],official_split=r['official_split'],engineering=r['engineering'],ordinal=ordinal,
            input={**q,'frame_ids':[q['frame_ids'][j] for j in pos]},parent_positions=pos,parent_frame_ids=q['frame_ids'],**provenance,
            historically_exposed=True,exposure_witness=r['exposure_witness'],checkpoint_training_overlap='HC official train included in PTD training; not unseen test' if r['official_split']=='train' else 'same mixed-domain checkpoint; no new training-coverage claim'))
        labels[r['key']]=lab
        print('PREPARE',ordinal,r['key'],flush=True)
    assert len({r['input']['video_sha256'] for r in rows})==256
    write(OUT/'LABELS_SCORER_ONLY.json',labels);write(OUT/'INPUTS.json',rows)
    write(OUT/'INTAKE_COMPLETE.json',dict(time=time.time(),count=256,input_sha=sha(OUT/'INPUTS.json'),labels_sha=sha(OUT/'LABELS_SCORER_ONLY.json'),selection_sha=sha(OUT/'SELECTION.json'),GT_predictor=False,grid_adjustments=issues))
    status(OUT/'INTAKE_STATUS.json',dict(state='completed',count=256))
if __name__=='__main__':prepare()
