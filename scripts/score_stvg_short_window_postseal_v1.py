"""Sealed fullTube metrics and separate same-position candidate geometry diagnostic."""
import sys,json,statistics,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
from scripts.review_stvg_sparse_r1_v2 import _metric_functions,score_tube_xyxy
from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
from scripts.score_stvg_motion_guided_postseal_v1 import _per_frame_iou,_subset_delta
from vg_tta.tastvg_oracle_event5_v1 import box_iou
B=R/'artifacts/stvg_short_event_tracklets_v1';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();ROLES=('local_R1','associated_R1')

def _need(condition, message):
 if not condition: raise RuntimeError(message)

def _preflight_no_gt():
 """Bind the sealed student, its independent readback, and teacher inputs before GT."""
 _need(not torch.cuda.is_initialized(), 'postseal reader must remain CPU-only')
 from scripts.review_stvg_short_window_student_math_v1 import _preflight as math_preflight
 gate=math_preflight()
 expected_ids=tuple([f'hc2_{i:03d}' for i in range(4)]+[f'vidstg_{i:03d}' for i in range(4)])
 # The math reader verifies exact student/source/feature/reference/runtime pins and
 # every student-result receipt before deserialization. Extend that gate to the
 # raw tracking and teacher-output artifacts consumed by this diagnostic.
 trackbar=read(B/'GLOBAL_TRACKLET_BARRIER.json')
 _need(trackbar.get('schema')=='stvg-short-event-tracklet-barrier-v1' and trackbar.get('status')=='complete'
       and trackbar.get('count')==8 and trackbar.get('case_ids')==list(expected_ids)
       and trackbar.get('GT_read') is False and trackbar.get('GPU_used') is False
       and trackbar.get('model_loaded') is False and trackbar.get('teacher_called') is False,
       'tracklet barrier is not the exact sealed no-GT catalog')
 trackfiles=trackbar.get('files');case_records=trackbar.get('case_records')
 _need(isinstance(trackfiles,list) and len(trackfiles)==16 and isinstance(case_records,list)
       and len(case_records)==8, 'tracklet barrier does not bind exactly eight case files')
 trackfile_by_path={};trackdoc_by_id={};trackrecord_by_id={};expected_file_sha={}
 for row in case_records:
  _need(isinstance(row,dict), 'malformed tracklet case binding')
  cid=row.get('case_id');_need(cid in expected_ids and cid not in trackrecord_by_id, 'unknown or duplicate tracklet case binding')
  jsonrel=f'artifacts/stvg_short_event_tracklets_v1/tracklets/{cid}.json'
  npzrel=f'artifacts/stvg_short_event_tracklets_v1/private_rgb/{cid}.npz'
  _need(row.get('GT_read') is False and row.get('path')==jsonrel
        and row.get('rgb_npz_path')==npzrel, f'tracklet case paths differ for {cid}')
  _need(isinstance(row.get('sha256'),str) and isinstance(row.get('rgb_npz_sha256'),str),
        f'tracklet case hashes are missing for {cid}')
  expected_file_sha[jsonrel]=row['sha256'];expected_file_sha[npzrel]=row['rgb_npz_sha256']
  trackrecord_by_id[cid]=row
 _need(set(trackrecord_by_id)==set(expected_ids), 'tracklet barrier omits a fixed case')
 for row in trackfiles:
  _need(isinstance(row,dict), 'malformed tracklet file receipt')
  rel=row.get('path');_need(rel in expected_file_sha and rel not in trackfile_by_path,
                            'tracklet file path is unknown or duplicated')
  p=R/rel;_need(p.is_file() and p.stat().st_size==row.get('bytes')
                 and sha(p)==row.get('sha256')==expected_file_sha[rel],
                 f'tracklet barrier file bytes differ: {rel}')
  trackfile_by_path[rel]=row
 _need(set(trackfile_by_path)==set(expected_file_sha), 'tracklet barrier does not bind exact JSON/NPZ files')
 trackfile_by_id={cid:trackfile_by_path[f'artifacts/stvg_short_event_tracklets_v1/tracklets/{cid}.json']
                  for cid in expected_ids}
 for cid,row in trackrecord_by_id.items():
  _need(row.get('sha256')==trackfile_by_id[cid]['sha256'],f'tracklet JSON receipt differs for {cid}')
  doc=read(R/row['path'])
  _need(doc.get('schema')=='stvg-short-event-tracklet-run-v1' and doc.get('status')=='complete'
        and doc.get('case_id')==cid and doc.get('GT_read') is False
        and doc.get('model_loaded') is False and doc.get('GPU_used') is False,
        f'tracklet case is not sealed no-GT metadata for {cid}')
  _need(isinstance(doc.get('centers'),list) and len(doc['centers'])==4
        and isinstance(doc.get('tracklets'),list), f'tracklet case schema is incomplete for {cid}')
  manifest_centers=gate['manifest_by_id'][cid].get('centers')
  _need(isinstance(manifest_centers,list) and len(manifest_centers)==4,
        f'manifest center catalog is incomplete for {cid}')
  expected_centers={int(x['slot']):int(x['center_physical_frame_id']) for x in manifest_centers}
  actual_centers={int(x['slot']):int(x['center_physical_frame_id']) for x in doc['centers']}
  _need(expected_centers==actual_centers and set(expected_centers)=={0,1,2,3},
        f'tracklet centers do not match the frozen manifest for {cid}')
  trackdoc_by_id[cid]=doc;trackrecord_by_id[cid]=row
 _need(set(trackdoc_by_id)==set(expected_ids), 'tracklet case binding is incomplete')

 jobpath=B/'TEACHER_JOB_LOCK.json';tbpath=B/'TEACHER_GLOBAL_BARRIER.json'
 jobsdoc=read(jobpath);tb=read(tbpath)
 _need(jobsdoc.get('schema')=='stvg-short-window-teacher-jobs-v1' and jobsdoc.get('status')=='locked'
       and jobsdoc.get('count')==64 and jobsdoc.get('GT_read') is False
       and jobsdoc.get('model_or_gpu_called') is False and len(jobsdoc.get('records',[]))==64,
       'teacher job lock is not the exact no-GT 64-request lock')
 inputrel=jobsdoc.get('teacher_input_barrier_path')
 _need(inputrel=='artifacts/stvg_short_event_tracklets_v1/CANDIDATE_INPUT_BARRIER.json',
       'teacher job lock input barrier path changed')
 inputpath=R/inputrel;_need(inputpath.is_file() and sha(inputpath)==jobsdoc.get('teacher_input_barrier_sha256'),
                             'teacher input barrier bytes do not match the locked SHA')
 inputdoc=read(inputpath)
 _need(inputdoc.get('status')=='locked' and inputdoc.get('GT_read') is False
       and inputdoc.get('model_or_gpu_called') is False
       and inputdoc.get('tracklet_barrier_sha256')==sha(B/'GLOBAL_TRACKLET_BARRIER.json'),
       'teacher input barrier is not bound to the no-GT tracklet barrier')
 _need(tb.get('status')=='complete' and tb.get('count')==64 and tb.get('GT_read') is False
       and tb.get('job_lock_sha256')==sha(jobpath)
       and tb.get('input_barrier_sha256')==sha(inputpath),
       'teacher output barrier does not bind the exact locked requests')
 jobs={};responses={};file_rows=tb.get('files')
 _need(isinstance(file_rows,list) and len(file_rows)==64,'teacher barrier must bind exactly 64 outputs')
 for job in jobsdoc['records']:
  _need(isinstance(job,dict) and job.get('GT_read') is False and job.get('model_or_gpu_called') is False,
        'teacher job row is not declared no-GT')
  jid=job.get('job_id');cid=job.get('case_id');slot=job.get('center_slot');kind=job.get('kind')
  _need(cid in expected_ids and isinstance(slot,int) and slot in range(4) and kind in ('base','id_rotation'),
        'teacher job identity is malformed')
  _need(jid==f'{cid}_window{slot:02d}_{kind}', 'teacher job ID does not match case/window/control')
  _need(jid not in jobs, 'duplicate teacher job ID')
  _need(job.get('dataset')==gate['manifest_by_id'][cid].get('dataset')
        and job.get('source_case_sha256')==gate['manifest_by_id'][cid].get('source_case_sha256')
        and job.get('tracklet_barrier_sha256')==sha(B/'GLOBAL_TRACKLET_BARRIER.json')
        and job.get('tracklet_case_sha256')==trackrecord_by_id[cid].get('sha256')
        and job.get('rgb_npz_sha256')==trackrecord_by_id[cid].get('rgb_npz_sha256'),
        f'teacher job source binding differs for {jid}')
  center=gate['manifest_by_id'][cid]['centers'][slot]
  _need(job.get('center_physical_frame_id')==center.get('center_physical_frame_id'),
        f'teacher job physical center differs for {jid}')
  track_center=trackdoc_by_id[cid]['centers'][slot]
  _need(set(job['candidate_order'])==set(track_center.get('candidate_ids',[])),
        f'teacher candidate pool differs from the locked tracklets for {jid}')
  _need(isinstance(job.get('candidate_order'),list)
        and all(isinstance(x,int) and not isinstance(x,bool) for x in job['candidate_order'])
        and isinstance(job.get('allowed_output_ids'),list)
        and len(job['candidate_order'])==len(set(job['candidate_order']))
        and all(isinstance(x,int) and not isinstance(x,bool) for x in job['allowed_output_ids'])
        and len(job['allowed_output_ids'])==len(set(job['allowed_output_ids']))
        and isinstance(job.get('visible_id_to_candidate'),dict)
        and set(job['allowed_output_ids'])=={0,*[int(x) for x in job['visible_id_to_candidate']]}
        and set(job['visible_id_to_candidate'].values())==set(job['candidate_order']),
        f'teacher candidate mapping malformed for {jid}')
  jobs[jid]=job
 _need(len(jobs)==64, 'teacher jobs do not cover 64 unique identities')
 for cid in expected_ids:
  for slot in range(4):
   _need(f'{cid}_window{slot:02d}_base' in jobs and f'{cid}_window{slot:02d}_id_rotation' in jobs,
         f'teacher jobs omit a locked base/rotation pair for {cid}')
 output_by_job={}
 for row in file_rows:
  _need(isinstance(row,dict), 'malformed teacher output receipt')
  rel=row.get('path');_need(isinstance(rel,str) and rel.startswith('artifacts/stvg_short_event_tracklets_v1/teacher/')
                              and rel.endswith('.json') and '..' not in Path(rel).parts,
                              'teacher output path is outside the locked scope')
  p=R/rel;_need(p.is_file() and p.stat().st_size==row.get('bytes') and sha(p)==row.get('sha256'),
                       f'teacher output bytes differ: {rel}')
  d=read(p);jid=d.get('job_id')
  _need(jid in jobs and jid not in output_by_job, 'teacher output has unknown or duplicate job identity')
  j=jobs[jid]
  _need(d.get('status')=='actual_video_candidate_verification' and d.get('GT_read') is False
        and d.get('case_id')==j['case_id'] and d.get('dataset')==j['dataset']
        and d.get('kind')==j['kind'] and d.get('center_slot')==j['center_slot']
        and d.get('source_case_sha256')==j['source_case_sha256']
        and d.get('prompt_sha256')==j['prompt_sha256']
        and d.get('candidate_bindings')==j['candidate_bindings']
        and d.get('input_videos')==j['videos'], f'teacher output identity/input binding differs for {jid}')
  parsed=d.get('parsed');_need(isinstance(parsed,dict) and parsed.get('status') in ('valid','invalid'),
                              f'teacher parse state malformed for {jid}')
  visible=parsed.get('visible_id');chosen=parsed.get('selected_candidate_id')
  if parsed['status']=='valid':
   if visible in (None,0):
    _need(chosen is None, f'abstaining teacher parse has a candidate for {jid}')
   else:
    _need(isinstance(visible,int) and not isinstance(visible,bool)
          and visible in j['allowed_output_ids']
          and str(visible) in j['visible_id_to_candidate']
          and chosen==j['visible_id_to_candidate'][str(visible)],
          f'teacher selected candidate does not match visible-ID mapping for {jid}')
  else:
   _need(chosen is None, f'invalid teacher parse unexpectedly has a selected candidate for {jid}')
  output_by_job[jid]={'job_id':jid,
                      'parsed':{'status':parsed['status'],'visible_id':visible,
                                'selected_candidate_id':chosen},
                      'output_truncated_at_cap':d.get('output_truncated_at_cap')}
 _need(set(output_by_job)==set(jobs), 'teacher output set differs from exact job lock')
 # Keep only candidate IDs and parse-status metadata for later scoring. Teacher
 # prompts, query text, raw responses, video paths, and full binding records are
 # discarded after their hash/identity checks.
 jobs={jid:{'job_id':j['job_id'],'case_id':j['case_id'],'dataset':j['dataset'],
            'kind':j['kind'],'center_slot':j['center_slot'],
            'candidate_order':list(j['candidate_order'])} for jid,j in jobs.items()}

 # Anchor-level checks prevent stale/hand-edited supervision from reaching the
 # candidate diagnostics; source-grid position is bound through physical IDs.
 refidx=gate['reference_by_id']
 for cid in expected_ids:
  ref=refidx[cid];track=trackdoc_by_id[cid];manifest=gate['manifest_by_id'][cid]
  center_by_physical={int(c['center_physical_frame_id']):c for c in manifest['centers']}
  tids={t['tracklet_id']:t for t in track['tracklets']}
  _need(len(tids)==len(track['tracklets']), f'duplicate tracklet IDs for {cid}')
  for role in ROLES:
   refrole=ref['references'][role];anchors=refrole['anchors'];seen=set()
   for a in anchors:
    physical=a['physical_frame_id'];_need(physical in center_by_physical, f'anchor is not at a locked center for {cid}/{role}')
    center=center_by_physical[physical];_need(a['position']==center['source_grid_position'],
                                              f'anchor position is not source-grid position for {cid}/{role}')
    slot=center['slot'];job=jobs[f'{cid}_window{slot:02d}_base'];out=output_by_job[job['job_id']]
    receipt_sha=next(r['sha256'] for r in file_rows if Path(r['path']).stem==job['job_id'])
    _need(a['teacher_job_id']==job['job_id'] and out['parsed']['status']=='valid'
          and out.get('output_truncated_at_cap') is False
          and a['teacher_output_sha256']==receipt_sha
          and a['selected_candidate_id']==out['parsed']['selected_candidate_id']
          and a.get('candidate_id')==a['selected_candidate_id']
          and a.get('is_Source') is False and a['selected_candidate_id']!=1,
          f'anchor is not bound to a valid non-Source base response for {cid}/{role}')
    tid=a['tracklet_id'];_need(tid in tids and tids[tid]['candidate_id']==a['candidate_id'],
                                f'anchor tracklet/candidate binding differs for {cid}/{role}')
    frame=next((f for f in tids[tid]['frames'] if f['physical_frame_id']==physical),None)
    _need(frame is not None and frame['valid'] is True and frame['box_xyxy_norm'] is not None,
          f'anchor has no valid tracklet box at its locked center for {cid}/{role}')
    box=np.asarray(frame['box_xyxy_norm'],dtype=np.float64);cxcywh=np.asarray(a['box_cxcywh'],dtype=np.float64)
    expected=np.asarray([(box[0]+box[2])/2,(box[1]+box[3])/2,box[2]-box[0],box[3]-box[1]])
    _need(np.isfinite(expected).all() and np.allclose(cxcywh,expected,rtol=0,atol=1e-12),
          f'anchor geometry differs from its locked tracklet center for {cid}/{role}')
    _need(physical not in seen, f'duplicate physical center anchor for {cid}/{role}');seen.add(physical)
 return {**gate,'trackdoc_by_id':trackdoc_by_id,'jobs_by_id':jobs,'responses_by_job':output_by_job,
         'teacher_file_rows':file_rows,'math_readback_path':B/'ROOT_STUDENT_MATH_READBACK.json'}

def _validate_math_readback(gate):
 """Exact provenance check at the first GT-loader boundary."""
 path=gate['math_readback_path'];_need(path.is_file(), 'ROOT independent math readback is missing')
 report=read(path);prov=report.get('provenance')
 _need(report.get('schema')=='stvg-short-window-student-independent-math-readback-v1'
       and report.get('status')=='pass' and report.get('GT_read') is False
       and report.get('labels_read') is False and report.get('model_loaded') is False
       and report.get('GPU_initialized') is False and report.get('video_or_RGB_read') is False
       and report.get('pixels_independently_redecoded') is False,
       'ROOT independent student math readback is not a CPU/no-GT pass')
 _need(report.get('cases')==8 and report.get('fit_count')==gate['expected_fit_count']
       and report.get('optimizer_steps')==gate['barrier'].get('optimizer_steps'),
       'ROOT math readback counts differ from current sealed student barrier')
 _need(isinstance(prov,dict), 'ROOT math readback provenance is missing')
 for key,digest in gate['hashes'].items():
  _need(prov.get(key)==digest, f'ROOT math readback has stale provenance: {key}')
 expected_files={cid:sha(path) for cid,path in gate['case_paths'].items()}
 _need(prov.get('student_global_barrier_files_sha256')==expected_files,
       'ROOT math readback does not bind every sealed student result file')
 runtime_pins={**gate['runtime'].get('pins',{}),**gate['runtime'].get('protected',{})}
 _need(prov.get('student_source_sha256')==runtime_pins.get('scripts/run_stvg_short_window_student_v1.py')
       and prov.get('reader_source_sha256')==sha(R/'scripts/review_stvg_short_window_student_math_v1.py')
       and prov.get('reused_cpu_fit_auditor_sha256')==sha(R/'scripts/review_stvg_scene_object_id_spatial_math_v1.py'),
       'ROOT math readback source provenance differs from locked runtime')
 return report

def main():
 torch.set_num_threads(2);gate=_preflight_no_gt()
 bar=gate['barrier'];refs=gate['reference_by_id'];man=gate['manifest_by_id']
 outputs={};cases={}
 for cid in [f'hc2_{i:03d}' for i in range(4)]+[f'vidstg_{i:03d}' for i in range(4)]:
  z=torch.load(gate['case_paths'][cid],map_location='cpu',weights_only=False)
  _need(z.get('case_id')==cid and z.get('GT_read') is False,f'student payload identity/GT flag differs for {cid}')
  outputs[cid]=z;mc=man[cid];cases[cid]=load_source_case(R/mc['source_case_path'])
 _need(set(outputs)==set(refs)==set(man),'sealed student/source catalogs differ')
 # No truth loader is reachable before this exact ROOT math provenance gate.
 _validate_math_readback(gate)
 from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
 truth,gtbindings=_truths_after_seal(cases);official,independent,_=_metric_functions();metrics=[];geometry=[];windows=[];movement=[]
 jobs_by_id=gate['jobs_by_id'];jobs=list(jobs_by_id.values());responses=gate['responses_by_job'];tracks_by_id=gate['trackdoc_by_id']
 for cid in sorted(outputs):
  z=outputs[cid];c=cases[cid];fid=list(map(int,c['frame_ids']));fmap={f:i for i,f in enumerate(fid)};tr=truth[cid];denseF=_per_frame_iou(z['arms']['F'],fid,c['width'],c['height'],c['dataset'],tr['truth'],tr['span']);scored={}
  expected={'F','T',*ROLES,*['T_'+x for x in ROLES]};_need(set(z['arms'])==expected,f'student arm schema differs for {cid}')
  for arm in ['F','T',*ROLES,*['T_'+x for x in ROLES]]:
   a=z['arms'][arm];q=score_tube_xyxy(a['boxes'],fid,c['width'],c['height'],a['interval'],tr['truth'],tr['span'],c['dataset'],official=official,independent=independent);scored[arm]=q
  for arm,q in scored.items():
   baseline='T'if arm.startswith('T_')or arm=='T'else'F';role=arm.removeprefix('T_');sup={int(a['physical_frame_id'])for a in refs[cid]['references'].get(role,{}).get('anchors',[])};d=_per_frame_iou(z['arms'][arm],fid,c['width'],c['height'],c['dataset'],tr['truth'],tr['span'])
   metrics.append(dict(case_id=cid,dataset=c['dataset'],arm=arm,metrics=q['metrics'],official_independent_max_abs_error=q['official_independent_max_abs_error'],baseline=baseline,delta_vIoU_pp=100*(q['metrics']['m_vIoU']-scored[baseline]['metrics']['m_vIoU']),supervised_geometry=_subset_delta(denseF,d,sup,selection_status='fixed_actual_supervision_GT_known'),unsupervised_geometry=_subset_delta(denseF,d,set(map(int,denseF.fids))-sup,selection_status='other_GT_known_frames'),actual_supervision_count=len(sup)))
  tracks=tracks_by_id[cid];tidx={t['tracklet_id']:t for t in tracks['tracklets']}
  for w in tracks['centers']:
   center=int(w['center_physical_frame_id']);slot=int(w['slot']);basejob=jobs_by_id[f'{cid}_window{slot:02d}_base'];swjob=jobs_by_id[f'{cid}_window{slot:02d}_id_rotation'];baseout=responses[basejob['job_id']];swout=responses[swjob['job_id']];chosen=baseout['parsed']['selected_candidate_id'];swchosen=swout['parsed']['selected_candidate_id'];display=set(basejob['candidate_order']);vals={}
   for tid in w['tracklet_ids']:
    t=tidx[tid];a=next(x for x in t['frames']if x['physical_frame_id']==center);known=center in tr['truth'];valid=a['valid']and a['box_xyxy_norm']is not None;v=src=None
    if known and valid:
     gt=np.asarray(tr['truth'][center],float)/[c['width'],c['height'],c['width'],c['height']];v=float(box_iou(np.asarray(a['box_xyxy_norm'])[None],gt[None])[0]);src=float(box_iou(np.asarray(z['arms']['F']['boxes'])[fmap[center]][None],gt[None])[0]);vals[t['candidate_id']]=v
    geometry.append(dict(case_id=cid,dataset=c['dataset'],slot=w['slot'],candidate_id=t['candidate_id'],displayed=t['candidate_id']in display,GT_known=known,valid_center=bool(valid),valid_tracklet_frames=sum(a['valid']for a in t['frames']),candidate_IoU=v,source_IoU=src,delta_pp=None if v is None else 100*(v-src)))
   common=center in tr['truth']and bool(display)and display<=set(vals)and 1 in vals
   oracle=max((vals[k]for k in display),default=None)if common else None;source=vals.get(1)if common else None;teacher=vals.get(chosen)if common and chosen in display else None
   parse_status=baseout['parsed']['status'];invalid=parse_status!='valid';truncated=bool(baseout.get('output_truncated_at_cap'))
   windows.append(dict(case_id=cid,dataset=c['dataset'],slot=w['slot'],common_observation_count=int(common),common_observation_available=bool(common),unknown_reason=None if common else'no_shared_GT_known_valid_observation',source_IoU=source,oracle_IoU=oracle,teacher_IoU=teacher,delta_pool_pp=None if oracle is None else 100*(oracle-source),delta_teacher_pp=None if teacher is None else 100*(teacher-source),R1_candidate=chosen,rotated_R1_candidate=swchosen,ID_invariant=None if chosen is None or swchosen is None else chosen==swchosen,selected_Source=chosen==1,abstain_or_invalid=chosen is None,parse_status=parse_status,abstained=bool(parse_status=='valid' and chosen is None),invalid_output=bool(invalid),output_truncated_at_cap=truncated,candidate_count=len(display)))
  for role,fit in z['fits'].items():
   movement.append(dict(case_id=cid,role=role,optimizer_steps=fit['optimizer_steps'],parameter_delta_norm=float(torch.cat([(fit['final_state'][n]-x).flatten().double()for n,x in fit['initial_state'].items()]).norm())))
 groups=[]
 for dataset in ['hc2','vidstg']:
  for arm in ['F','T',*ROLES,*['T_'+x for x in ROLES]]:
   rows=[q for q in metrics if q['dataset']==dataset and q['arm']==arm];ds=[q['delta_vIoU_pp']for q in rows];groups.append(dict(dataset=dataset,arm=arm,count=len(rows),mean_vIoU=statistics.mean(q['metrics']['m_vIoU']for q in rows),mean_delta_pp=statistics.mean(ds),median_delta_pp=statistics.median(ds),positive=sum(x>1e-8 for x in ds),negative=sum(x< -1e-8 for x in ds),neutral=sum(abs(x)<=1e-8 for x in ds)))
 out=dict(status='pass',scope='8 historical sources finite second-round diagnosis; not independent confirmation',GT_read=True,GT_used_for_selection=False,metric_count=len(metrics),candidate_count=len(geometry),window_count=len(windows),actual_fit_count=bar['actual_fit_count'],metrics=metrics,groups=groups,candidate_geometry=geometry,window_pool_teacher=windows,fit_movement=movement,GT_source_bindings=gtbindings,student_barrier_sha256=sha(B/'STUDENT_GLOBAL_BARRIER.json'),student_runtime_sha256=sha(B/'STUDENT_RUNTIME.json'),student_math_readback_sha256=sha(gate['math_readback_path']),teacher_job_lock_sha256=sha(B/'TEACHER_JOB_LOCK.json'),teacher_global_barrier_sha256=sha(B/'TEACHER_GLOBAL_BARRIER.json'),tracklet_barrier_sha256=sha(B/'GLOBAL_TRACKLET_BARRIER.json'),max_official_independent_error=max(x['official_independent_max_abs_error']for x in metrics),time=time.time());p=B/'ROOT_POSTSEAL_REPORT.json';_need(not p.exists(),'refusing to overwrite an existing postseal report')
 with p.open('x',encoding='utf-8') as stream:stream.write(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status='pass',groups=groups,windows=len(windows),candidates=len(geometry))))
if __name__=='__main__':main()
