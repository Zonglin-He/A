"""Official full test intake. New outputs only; never modify source archives.

Captions and questions are retained separately. GT is a diagnostic/scoring
sidecar, never part of the predictor input. No metric-based sample selection.
"""
import argparse, concurrent.futures, hashlib, json, re, shutil, subprocess, sys, tarfile, time, zipfile
from fractions import Fraction
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha
from scripts.prepare_official_dense_pool_v2 import frame_ids_from_segment, hc_source_id
OUT = ROOT/'artifacts/stvg_fullscale_diagnostics_v1'
DATA = ROOT/'data/stvg_fullscale_v1'
INTAKE = ROOT/'downloads/dataset_intake_20260910'
HC = INTAKE/'hcstvg/datasets/HC-STVG1/HC-STVG(v1-5660)'
SEVEN = ROOT/'.cache/fullscale-7zip/usr/lib/7zip/7z'


def hc_extract():
    target = read(HC/'test.json'); dest = DATA/'hcstvg1_test/video'; dest.mkdir(parents=True, exist_ok=True)
    receipt = OUT/'hc_archive_receipt.json'
    if receipt.exists():
        r = read(receipt)
        assert all((dest/k).is_file() and sha(dest/k)==v['sha256'] for k,v in r['members'].items())
        return
    if shutil.disk_usage(ROOT).free < 80*2**30: raise RuntimeError('Insufficient safe disk headroom')
    archive = HC/'video(5660)/v1.zip'; found = {}; names = set()
    # The split ZIP contains a single v1.tgz. Stream it; do not duplicate 38 GB.
    log = open(OUT/'hc_extraction_stderr.log','a')
    p = subprocess.Popen([str(SEVEN),'x','-so',str(archive),'v1.tgz'],stdout=subprocess.PIPE,stderr=log)
    try:
        with tarfile.open(fileobj=p.stdout,mode='r|gz') as tar:
            for member in tar:
                name=Path(member.name).name
                if not member.isfile() or name not in target: continue
                if name in names: raise ValueError('Duplicate archive basename: '+name)
                names.add(name); path=dest/name; stream=tar.extractfile(member)
                h=hashlib.sha256(); count=0
                # Existing files are verified against the entire official stream.
                tmp=path.with_suffix('.extracting'); handle=None if path.exists() else open(tmp,'xb')
                try:
                    for block in iter(lambda:stream.read(8<<20),b''):
                        h.update(block);count+=len(block)
                        if handle:handle.write(block)
                finally:
                    if handle:handle.close()
                assert count==member.size
                if path.exists():assert sha(path)==h.hexdigest()
                else:tmp.replace(path)
                found[name]=dict(member=member.name,bytes=count,sha256=h.hexdigest())
                if len(found)%100==0:
                    status(OUT/'hc_extract_progress.json',dict(done=len(found),total=len(target),unix=time.time()))
                    print('HC_EXTRACT',len(found),len(target),flush=True)
        # Drain the outer stream to ensure the ZIP CRC is checked as well.
        while p.stdout.read(8<<20):pass
        code=p.wait(); assert code==0,code
    finally:
        if p.poll() is None:p.terminate();p.wait()
        log.close()
    assert set(found)==set(target),(len(found),len(target))
    write(receipt,dict(archive=str(archive),outer_crc_verified=True,members=found,
        volumes=[dict(path=str(x),bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(archive.parent.iterdir()) if x.suffix in ['.zip'] or re.fullmatch(r'\.z\d+',x.suffix)]))


def vid_extract():
    rows=read(ROOT/'external/VidSTG-Dataset/annotations/test_annotations.json'); sources={r['vid'] for r in rows}
    dest=DATA/'vidstg_test/video';dest.mkdir(parents=True,exist_ok=True); receipt=OUT/'vid_archive_receipt.json'
    if receipt.exists():
        r=read(receipt);assert all((dest/(k+'.mp4')).is_file() and sha(dest/(k+'.mp4'))==v['sha256'] for k,v in r['members'].items());return
    archive=ROOT/'downloads/vidor/validation-video.zip';found={}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            key=Path(info.filename).stem
            if key not in sources or not info.filename.endswith('.mp4'):continue
            assert key not in found
            path=dest/(key+'.mp4');tmp=path.with_suffix('.extracting');h=hashlib.sha256()
            handle=None if path.exists() else open(tmp,'xb')
            try:
                with z.open(info) as stream:
                    for block in iter(lambda:stream.read(8<<20),b''):
                        h.update(block)
                        if handle:handle.write(block)
            finally:
                if handle:handle.close()
            if path.exists():assert sha(path)==h.hexdigest()
            else:tmp.replace(path)
            found[key]=dict(member=info.filename,bytes=info.file_size,CRC=info.CRC,sha256=h.hexdigest())
    assert set(found)==sources,(len(found),len(sources))
    write(receipt,dict(archive=str(archive),archive_sha256=sha(archive),members=found))


def _probe(path):
    p=subprocess.run([str(ROOT/'.conda/tubedetr/bin/ffprobe'),'-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,avg_frame_rate,nb_frames,duration','-of','json',str(path)],capture_output=True,text=True,check=True)
    d=json.loads(p.stdout)['streams'][0]
    if not str(d.get('nb_frames','')).isdigit():
        counted=subprocess.run([str(ROOT/'.conda/tubedetr/bin/ffprobe'),'-v','error','-select_streams','v:0','-count_frames',
            '-show_entries','stream=nb_read_frames','-of','json',str(path)],capture_output=True,text=True,check=True)
        d['nb_frames']=json.loads(counted.stdout)['streams'][0]['nb_read_frames']
    fps=float(Fraction(d['avg_frame_rate']));n=int(d['nb_frames'])
    return dict(width=int(d['width']),height=int(d['height']),fps=fps,frame_count=n,duration=float(d.get('duration',n/fps)))


def probe(path):
    cache=OUT/'media_probes'/(hashlib.sha256(str(path).encode()).hexdigest()+'.json')
    if cache.exists():return read(cache)
    try:r=_probe(path)
    except Exception as e:
        r=dict(error=str(e),stderr=getattr(e,'stderr',None),bytes=Path(path).stat().st_size)
    write(cache,r);return r


def xywh(box,width,height):
    x,y,w,h=map(float,box)
    # Match official box clipping; retain invalid annotations as explicit QA.
    x1=max(0,min(width,x));y1=max(0,min(height,y));x2=max(0,min(width,x+w));y2=max(0,min(height,y+h))
    return [(x1+x2)/2/width,(y1+y2)/2/height,(x2-x1)/width,(y2-y1)/height]


def exposure_sources():
    sources=set();scanned=[]
    def visit(x):
        if isinstance(x,dict):
            if isinstance(x.get('source'),str):sources.add(x['source'])
            for k,v in x.items():
                if k not in ('diagnostic_GT','boxes','protected_pins','pins'):visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    for path in sorted((ROOT/'artifacts').glob('*/lock.json')):
        if path.parent==OUT or path.stat().st_size>60*2**20:continue
        visit(read(path));scanned.append(dict(path=str(path),sha256=sha(path)))
    return sources,scanned


def prepare():
    import numpy as np
    from scripts.prepare_vidstg_wrong_domain_support import load_vidor_annotations,trajectory_for_target
    if (OUT/'lock.json').exists():print('EXISTING_LOCK',flush=True);return
    hc=read(HC/'test.json');vr=read(ROOT/'external/VidSTG-Dataset/annotations/test_annotations.json')
    raw=load_vidor_annotations(ROOT/'downloads/vidor/validation-annotation.zip',None)
    hrec=read(OUT/'hc_archive_receipt.json');vrec=read(OUT/'vid_archive_receipt.json')
    paths=[DATA/'hcstvg1_test/video'/k for k in hc]+[DATA/'vidstg_test/video'/(k+'.mp4') for k in sorted(vrec['members'])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: meta=dict(zip(map(str,paths),pool.map(probe,paths)))
    exposed,scanned=exposure_sources();rows={'hcstvg1_test':[],'vidstg_test':[]};labels={};issues=[];metadata_discrepancies=[]
    def add(cohort,q,gt,track,fulltrack,kind,annotation,input_issue=None):
        n=len(rows[cohort]);q.update(index=n,kind='hcstvg' if cohort.startswith('hc') else 'vidstg')
        key=f'{cohort}:{n:06d}'; ids=q['frame_ids'];truth=np.zeros((len(ids),4));valid=[];missing=[]
        for j,fid in enumerate(ids):
            inside=gt[0]<=fid<gt[1];box=track.get(fid)
            good=inside and box is not None and box[2]>0 and box[3]>0
            valid.append(good)
            if good:truth[j]=box
            elif inside:missing.append(fid)
        complete=all(f in fulltrack and fulltrack[f][2]>0 and fulltrack[f][3]>0 for f in ids)
        centers=[fulltrack[f][:2] for f in ids] if complete else None
        issue=[]
        if missing:issue.append('missing_or_degenerate_event_boxes')
        if not any(valid):issue.append('no_valid_GT_box_on_fixed_grid')
        if len(ids)<4:issue.append('fewer_than_four_input_positions')
        if issue:issues.append(dict(key=key,reasons=issue,missing=missing))
        labels[key]=dict(interval=gt,boxes=truth.tolist(),valid=valid,full_track_centers=centers,
            event_mask=[gt[0]<=f<gt[1] for f in ids],missing_event_frames=missing,official_annotation=annotation)
        rows[cohort].append(dict(ordinal=n,key=key,input=q,query_type=kind,
            historical_source_listed=q['source'] in exposed,diagnostic_issues=issue,input_unavailable=input_issue))
    for name,a in hc.items():
        path=DATA/'hcstvg1_test/video'/name;m=meta[str(path)]
        input_issue=None
        if 'error' in m:
            input_issue=m;m=dict(width=a['width'],height=a['height'],frame_count=a['img_num'],fps=a['img_num']/20.,duration=20.)
        elif m['width']!=a['width'] or m['height']!=a['height'] or m['frame_count']!=a['img_num']:
            discrepancy=dict(video=name,actual=m,annotation={k:a[k] for k in ['width','height','img_num']})
            if m['width']==a['width'] and m['height']==a['height'] and a['img_num']-m['frame_count'] in [1,2]:
                # Publisher img_num overcounts an unannotated tail. All actual
                # pixels retain zero-based identity; no GT frames are trimmed.
                assert a['st_frame']-1+len(a['bbox'])<=m['frame_count']
                metadata_discrepancies.append({**discrepancy,'resolution':'use measured media horizon; all GT frames retained, no temporal resampling shift'})
            else:
                input_issue=dict(reason='official_annotation_media_metadata_mismatch',**discrepancy)
                m=dict(width=a['width'],height=a['height'],frame_count=a['img_num'],fps=a['img_num']/20.,duration=20.)
        ids=frame_ids_from_segment(frame_count=m['frame_count'],fps=m['fps'],start_frame=0,end_frame=m['frame_count'],max_frames=200)
        start=a['st_frame']-1;gt=[start,start+len(a['bbox'])];assert 0<=gt[0]<gt[1]<=m['frame_count']
        track={start+i:xywh(b,m['width'],m['height']) for i,b in enumerate(a['bbox'])}
        q=dict(caption=a['caption'],source=hc_source_id(name),original_video_id=Path(name).stem,video_path=str(path),
            video_sha256=hrec['members'][name]['sha256'],frame_ids=ids,start_frame=0,end_frame=m['frame_count'],**m)
        add('hcstvg1_test',q,gt,track,track,'caption',dict(file=str(HC/'test.json'),key=name),input_issue)
    tracks={}
    for ai,a in enumerate(vr):
        path=DATA/'vidstg_test/video'/(a['vid']+'.mp4');m=meta[str(path)]
        assert m['width']==a['width'] and m['height']==a['height'],a['vid']
        start=a['used_segment']['begin_fid'];end=min(a['used_segment']['end_fid']+1,m['frame_count'])
        ids=frame_ids_from_segment(frame_count=m['frame_count'],fps=a['fps'],start_frame=start,end_frame=end,max_frames=200)
        gt=[a['temporal_gt']['begin_fid'],min(a['temporal_gt']['end_fid'],end)]
        assert start<=gt[0]<gt[1]<=end
        for field,kind in [('captions','caption'),('questions','question')]:
            for qi,s in enumerate(a[field]):
                tk=(a['vid'],s['target_id'])
                if tk not in tracks:
                    tr=trajectory_for_target(raw[a['vid']],s['target_id'])
                    tracks[tk]={int(k):xywh(v['bbox'],m['width'],m['height']) for k,v in tr.items()}
                q=dict(caption=s['description'],source=a['vid'],original_video_id=a['vid'],video_path=str(path),
                    video_sha256=vrec['members'][a['vid']]['sha256'],frame_ids=ids,start_frame=start,end_frame=end,**m)
                add('vidstg_test',q,gt,tracks[tk],tracks[tk],kind,dict(file='VidSTG test_annotations.json',annotation_index=ai,field=field,query_index=qi,target_id=s['target_id']))
    write(OUT/'labels_diagnostic_only.json',labels)
    pins=read(ROOT/'artifacts/st_causal_backbone_audit_v2/lock.json')['protected_pins']
    assert all(sha(ROOT/f)==h for f,h in pins.items())
    count={}
    for c,rr in rows.items():
        count[c]=dict(queries=len(rr),sources=len({r['input']['source'] for r in rr}),
            captions=sum(r['query_type']=='caption' for r in rr),questions=sum(r['query_type']=='question' for r in rr),
            listed_sources=len({r['input']['source'] for r in rr if r['historical_source_listed']}),
            full_track_queries=sum(labels[r['key']]['full_track_centers'] is not None for r in rr),
            full_track_sources=len({r['input']['source'] for r in rr if labels[r['key']]['full_track_centers'] is not None}),
            query_issues=sum(bool(r['diagnostic_issues']) for r in rr),unavailable_inputs=sum(r['input_unavailable'] is not None for r in rr))
    write(OUT/'DATA_QA.json',dict(counts=count,issues=issues,metadata=meta,metadata_discrepancies=metadata_discrepancies,exposure_scanned_locks=scanned,
        exposure_scope='Conservative listed/reserved sources in prior top-level locks, not proof of globally untouched data',
        all_official_test_queries_retained=True,train_not_pooled=True,media_probe_all_passed=not any(r['input_unavailable'] for rr in rows.values() for r in rr)))
    write(OUT/'lock.json',dict(version='stvg_fullscale_diagnostics_v1',created=time.time(),rows=rows,counts=count,
        labels=str(OUT/'labels_diagnostic_only.json'),labels_sha256=sha(OUT/'labels_diagnostic_only.json'),protected_pins=pins,
        backbones=['tubedetr','tastvg','ptd'],legacy_checkpoint_mapping={'vidstg_test':'hc_to_vid','hcstvg1_test':'vid_to_hc'},
        source_models='Old TubeDETR/TA-STVG HC-STVG2 -> VidSTG; VidSTG -> NEW HC-STVG1. PTD mixed training, not transfer.',
        protocol='Full official test descriptive replication; no retuning on these labels; not globally untouched confirmation',
        frame_grid='metadata-only 5fps capped200; original zero-based frames; PTD uniform max64 subset',
        GT_conventions='HC-STVG1 st_frame-1, end exclusive = start+box count; VidSTG original begin/end exclusive (prior TubeDETR protocol); native-official endpoint sensitivities separately',
        main_resolution=224,corruption=False,no_expert_in_causal_intervention=True,seeds=[20260910,20260911,20260912],
        primary=['time GT all keys minus native and matched random fixed-GT sIoU','space GT vs anti and two orthogonal controls at gain4 fraction.25 all layers'],
        scoring='fixed-grid corrected vIoU, sIoU on valid GT frames, physical tIoU; query and source means; source cluster CI; captions/questions separate',
        dependent_clips_not_independent=True,method_unchanged=True))
    print('PREPARED',json.dumps(count,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['hc','vid','prepare']);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    {'hc':hc_extract,'vid':vid_extract,'prepare':prepare}[a.stage]()
