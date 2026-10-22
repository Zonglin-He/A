"""Read-only temporal leakage audit; all scientific predictions remain immutable."""
import argparse,copy,gc,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stvg.runtime import sha
from stvg.tube_runner import write,digest
PUBLIC=ROOT/'results/tubedetr_temporal_leakage_audit';PRIVATE=ROOT/'outputs/tubedetr_temporal_leakage_audit'
PARENT=ROOT/'results/tubedetr_boxfixed_development/plan.json'
def read(p):return json.loads(Path(p).read_text())
def tiou(a,b):
    x=max(0,min(a[1],b[1])-max(a[0],b[0]));return x/(a[1]-a[0]+b[1]-b[0]-x)
def stats(a):return dict(mean=float(np.mean(a)),median=float(np.median(a)),q10=float(np.quantile(a,.1)),q90=float(np.quantile(a,.9)))
def prepare():
    assert not(PUBLIC/'plan.json').exists()
    write(PUBLIC/'plan.json',dict(scope='Read-only audit of frozen latest BoxFixed Global6 plus prior Strong code; not new tuning or full inference',parent_plan_sha256=sha(PARENT),old_full_final_seal_sha256=sha(ROOT/'results/tubedetr_boxfixed_development/final_seal.json'),code_sha256=sha(Path(__file__)),offline_scope='All12205 input manifests and sealed predictions; source annotation provenance; label-independent whole-input temporal baseline; repeated-exposure audit',dynamic_budget='Two already evaluated first dev128 queries, one native224 forward each if GPU lock available; no teacher, no updates. Exact label denylist and file audit.',production_changes=False,new_full_allowed=False,GT_used_only_by_offline_auditor=True))
def check():
    p=read(PUBLIC/'plan.json');assert p['code_sha256']==sha(Path(__file__))and p['parent_plan_sha256']==sha(PARENT);return read(PARENT)
def offline():
    p=check();torch.set_num_threads(1);result={}
    from tools.build_manifests import hc_grid,vid_grid,HC_SHA,VID_SHA
    from stvg.inputs import validate_metadata
    vidfile=ROOT/'data/annotations/vidstg/test_annotations.json';hcfile=ROOT/'data/annotations/tastvg_release/data/hc-stvg2/annos/test.json'
    assert sha(vidfile)==VID_SHA and sha(hcfile)==HC_SHA
    vr=read(vidfile);hr=read(hcfile)
    for ds,d in p['datasets'].items():
        manifest=ROOT/d['full_manifest'];lab=ROOT/d['full_labels'];assert sha(manifest)==d['full_manifest_sha256']and sha(lab)==d['full_labels_sha256']
        rows=[json.loads(x)for x in manifest.read_text().splitlines()];gt={x['id']:x for x in map(json.loads,lab.read_text().splitlines())};dev={x['id']for x in map(json.loads,(ROOT/d['manifest']).read_text().splitlines())};root=ROOT/'outputs/tubedetr_boxfixed_development'/ds/'full';sf=root/'sealed.json';seal=read(sf);assert len(seal['files'])==len(rows)==d['full_queries']and seal['manifest_sha256']==sha(manifest)
        for f,h in read(ROOT/'results/tubedetr_boxfixed_development'/ds/'final_seal.json')['files'].items():assert sha(ROOT/f)==h
        values=[];grids=0;sameboundary=0;contained=0;outside=0;ckeys=set();meta_keys=set();source_spans={};input_sources={};parents=set()
        hnames=list(hr)
        for row in rows:
            q=row['id'];m=row['metadata'];g=gt[q];validate_metadata(m);assert set(row)=={'dataset','id','key','metadata'};ckeys.update(row);meta_keys.update(m);parents.add(g['parent'])
            if ds=='vidstg':
                v=vr[int(q.split('-')[1])];begin,end=v['used_segment']['begin_fid'],v['used_segment']['end_fid'];assert m['start_frame']==begin and m['end_frame']==end and m['frame_count']==v['frame_count'];assert m['caption']==v[m['kind']][int(q.split('-')[-1])]['description'];grid=vid_grid(begin,end);assert g['interval']==[v['temporal_gt']['begin_fid'],v['temporal_gt']['end_fid']];input_sources['used_segment_not_temporal_gt']=True
            else:
                name=hnames[m['index']];v=hr[name];assert m['original_video_id']==name and m['frame_count']==v['img_num'];grid=hc_grid(v['img_num']);start=v['st_frame']-1;assert g['interval']==[start,start+len(v['bbox'])-1];input_sources['whole_clip_img_num_not_st_frame_or_bbox_count']=True
            assert grid==m['frame_ids'];grids+=1;support=[grid[0],grid[-1]+1];sameboundary+=support==g['interval'];contained+=g['interval'][0]>=support[0]and g['interval'][1]<=support[1];outside+=int(not(g['interval'][0]>=support[0]and g['interval'][1]<=support[1]))
            e=seal['files'][q];assert sha(root/e['path'])==e['sha256']and sha(root/e['receipt_path'])==e['receipt_sha256'];receipt=read(root/e['receipt_path']);assert receipt['request_sha256']==digest(row)
            with np.load(root/e['path'])as z:
                assert np.array_equal(z['frame_ids'],grid);native=z['Native_interval'].tolist();global6=z['Global6_interval'].tolist()
                for a in [native,global6]:assert a[0]in grid and a[1]-1 in grid and a[1]>a[0]
                from stvg.positions import observation_positions
                extent=[grid.index(native[0]),grid.index(native[1]-1)];assert observation_positions(grid,extent)==z['positions8'].tolist()
            values.append(dict(id=q,parent=g['parent'],development=q in dev,source=tiou(native,g['interval']),global6=tiou(global6,g['interval']),whole_input=tiou(support,g['interval']),native_length=(native[1]-native[0])/(support[1]-support[0]),global6_length=(global6[1]-global6[0])/(support[1]-support[0]),GT_length=(g['interval'][1]-g['interval'][0])/(support[1]-support[0])))
        dr={x['id']:x for x in read(root/'details_private.json')}
        assert max(abs(v['source']-dr[v['id']]['values']['Source']['t'])for v in values)<1e-12
        assert max(abs(v['global6']-dr[v['id']]['values']['Full']['t'])for v in values)<1e-12
        groups={}
        for name,vs in [('all',values),('dev128',[x for x in values if x['development']]),('other_queries',[x for x in values if not x['development']])]:
            groups[name]=dict(queries=len(vs),mean_tIoU_percent={k:100*float(np.mean([v[k]for v in vs]))for k in ['source','global6','whole_input']},mean_temporal_gain_pp=100*float(np.mean([v['global6']-v['source']for v in vs])),global6_length_ratio=stats([v['global6_length']for v in vs]),native_length_ratio=stats([v['native_length']for v in vs]),GT_length_ratio=stats([v['GT_length']for v in vs]),global6_longer_queries=sum(v['global6_length']>v['native_length']for v in vs),global6_exact_GT_queries=sum(abs(v['global6']-1)<1e-12 for v in vs))
        result[ds]=dict(queries=len(rows),parents=len(parents),manifest_raw_input_provenance_exact=grids,metadata_keys=sorted(meta_keys),input_sources=input_sources,GT_equals_full_sample_support_queries=int(sameboundary),GT_contained_by_sample_support_queries=int(contained),GT_not_fully_contained_queries=int(outside),observations_derived_from_native_prediction_all_pass=True,predictions_receipts_hash_verified=len(values),prediction_final_seal_sha256=sha(sf),temporal_metrics_recomputed_all_pass=True,groups=groups,development_queries_in_full=len(dev),independent_unexposed_test=False)
        write(PRIVATE/ds/'details_private.json',values)
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    write(PUBLIC/'offline_audit.json',dict(status='complete',datasets=result,old_production_code_and_full_seal_unchanged=sha(ROOT/'results/tubedetr_boxfixed_development/final_seal.json')==read(PUBLIC/'plan.json')['old_full_final_seal_sha256'],GT_for_offline_audit_only=True,whole_input_baseline_is_not_new_prediction_model=True))
    print(json.dumps(result,indent=2))
def dynamic():
    from stvg.runtime import setup,lease
    p=check();denied={str((ROOT/d[k]).resolve())for d in p['datasets'].values()for k in ['labels','full_labels']};opened=[];attempts=[]
    def trace(event,args):
        if event=='open'and args and isinstance(args[0],(str,bytes,Path)):
            path=str(Path(os.fsdecode(args[0])).resolve())
            if path in denied or '/annotations/'in path or '/annos/'in path or path.endswith('_labels.jsonl'):attempts.append(path);raise PermissionError('audit denies annotations')
            opened.append(path)
    sys.addaudithook(trace);setup();lock=lease();checks=[];torch.set_num_threads(4)
    from stvg.tubedetr import load_source,endpoint_posterior
    from stvg.tube_decode import decode
    from stvg.tensors import state_hash
    from stvg.temporal import final_mbr
    from tools.boxfixed_tube.adapter import capture
    for ds,d in p['datasets'].items():
        row=json.loads((ROOT/d['manifest']).read_text().splitlines()[0]);model=load_source(d['configuration']);before=state_hash(model.state_dict());frames,ids=decode(row['metadata'],ds);tick=time.perf_counter();cap=capture(model,frames,ids,row['metadata']['caption']);torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        root=ROOT/'outputs/tubedetr_boxfixed_development'/ds/'full';s=read(root/'sealed.json');e=s['files'][row['id']];assert sha(root/e['path'])==e['sha256']
        with np.load(root/e['path'])as z:
            for k in ['Source','Native_interval','Global6_interval','frame_ids']:assert np.array_equal(cap[k].numpy()if hasattr(cap[k],'numpy')else cap[k],z[k]),(ds,k)
        q=np.mean([endpoint_posterior(z)for z in cap['logits6']],0);assert np.array_equal(q,cap['Global6_posterior']);assert final_mbr(ids,q)['interval']==cap['Global6_interval'];assert state_hash(model.state_dict())==before
        private=PRIVATE/ds/'native_audit_logits.npz';private.parent.mkdir(parents=True,exist_ok=True);np.savez(private,logits6=cap['logits6'].numpy(),posterior=q,frame_ids=ids)
        checks.append(dict(dataset=ds,existing_query_not_new_scientific_sample=True,source_checkpoint_sha256=model.load_audit['checkpoint_sha256'],Source_Native_Global6_frame_ids_bitexact=True,six_layer_posterior_and_MBR_reconstructed=True,model_weights_unchanged=True,teacher_calls=0,decoder_calls=1,seconds=elapsed,logits_private_sha256=sha(private)));del frames,model,cap;gc.collect();torch.cuda.empty_cache()
    assert not attempts;write(PRIVATE/'read_paths_private.json',opened);write(PUBLIC/'dynamic_audit.json',dict(status='pass',checks=checks,annotation_access_attempts=0,file_open_trace=True,all_GT_label_paths_blocked=True,actual_new_native_forward_calls=2,new_teacher_calls=0,new_parameter_updates=0,new_full_predictions=0));lock.close();print(json.dumps(checks,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','offline','dynamic']);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='offline':offline()
    else:dynamic()
