"""Postseal association with historical corrections and fixed private panels.

No new Teacher, selection, box interpolation, update or online GT use.
"""
from pathlib import Path
import collections,gc,json,math,sys,textwrap
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_referent_evidence_p0_v1 import BASE,PRIOR,read,put,sha
def main():
    import numpy as np,torch
    from PIL import Image,ImageDraw,ImageFont
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    from scripts.diagnose_stvg_lite_confirmation_spatial_v1 import iou
    assert read(BASE/'GLOBAL_SELECTION_BARRIER.json')['status']=='complete';d=read(BASE/'RESPONSE_POSTSEAL_DIAGNOSIS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in rr};targets,bindings=truth._truths_after_seal(cases)
    tm={cid:metric._truth_map(targets[cid])for cid in cases};priorfiles={x['case_id']:x for x in read(PRIOR/'trials/E1/STUDENT_GLOBAL_BARRIER.json')['files']};observations=[]
    for rec in rr:
        cid=rec['probe_alias'];c=cases[cid];entry=priorfiles[cid];assert sha(ROOT/entry['path'])==entry['sha256'];old=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=False);adapted=old['arms']['TS_M2']['boxes'];positions={int(fid):i for i,fid in enumerate(c['frame_ids'])}
        for fid in sorted({x['physical_frame_id']for x in d['frame_rows']if x['case_id']==cid and x['known_GT']}):
            if fid not in positions:
                observations.append(dict(case_id=cid,dataset=rec['dataset'],physical_frame_id=fid,correction_status='off_source_grid_unknown',GT_read=True));continue
            pos=positions[fid];gt=np.asarray(tm[cid][fid])/np.array([c['input']['width'],c['input']['height']]*2);before=iou(c['native_boxes_xyxy_normalized'][pos],gt);after=iou(adapted[pos],gt);change=after-before;stratum='improved'if change>1e-6 else'worsened'if change< -1e-6 else'unchanged';occupancy={}
            for strategy in ('query','caption','consensus','disagreement','connection','query_swap'):
                vv=[x['target_occupancy']for x in d['frame_rows']if x['case_id']==cid and x['physical_frame_id']==fid and x['strategy']==strategy];assert vv and all(v is not None for v in vv);occupancy[strategy]=math.fsum(vv)/len(vv)
            observations.append(dict(case_id=cid,dataset=rec['dataset'],physical_frame_id=fid,correction_status=stratum,old_E1_minus_Source_frame_IoU_pp=100*change,occupancy=occupancy,disagreement_minus_consensus_pp=100*(occupancy['disagreement']-occupancy['consensus']),disagreement_minus_query_pp=100*(occupancy['disagreement']-occupancy['query']),GT_read=True))
    summary=[]
    for ds in ('hc2','vidstg'):
        for stratum in ('improved','worsened','unchanged','off_source_grid_unknown'):
            subset=[x for x in observations if x['dataset']==ds and x['correction_status']==stratum];source=[]
            for cid in sorted({x['case_id']for x in subset}):
                rows=[x for x in subset if x['case_id']==cid]
                if stratum!='off_source_grid_unknown':source.append(math.fsum(x['disagreement_minus_consensus_pp']for x in rows)/len(rows))
            summary.append(dict(dataset=ds,correction_status=stratum,physical_positions=len(subset),sources=len({x['case_id']for x in subset}),source_macro_disagreement_minus_consensus_pp=math.fsum(source)/len(source)if source else None))
    put(BASE/'HISTORICAL_CORRECTION_ASSOCIATION.json',dict(status='complete',count=16,observations=observations,summary=summary,GT_read=True,scope='association of fixed evidence with already completed E1 spatial geometry changes on SAME exact physical Source-grid GT positions; not causal effect of evidence selections, no interpolation/new fit'))
    records=[];font=ImageFont.truetype('DejaVuSans.ttf',13);bar=read(BASE/'RESPONSE_GLOBAL_BARRIER.json');jobs=read(BASE/'INPUT_BARRIER.json')['jobs']
    for domain in ('hc2','vidstg'):
        for rec in [r for r in rr if r['dataset']==domain][:2]:
            cid=rec['probe_alias'];job=next(j for j in jobs if j['case_id']==cid);m=read(ROOT/job['metadata_path']);key=m['candidate']['canonical_id'];info=next(r for r in bar['records']if r['case_id']==cid and r['candidate_id']==key);caption=read(ROOT/job['caption_path'])['raw_caption'];query=read(PRIOR/'temporal/requests'/f'{cid}.json')['query_text'];video=np.load(ROOT/m['input_path'],allow_pickle=False).transpose(0,2,3,1);a=np.load(ROOT/info['path'],allow_pickle=False)
            canvas=Image.new('RGB',(1240,900),'white');draw=ImageDraw.Draw(canvas);line=5
            for text in (cid+' / '+domain+' / private fixed-hash review / NO GT / merged token=two frames', 'QUERY: '+query,'CAPTION: '+caption):
                for t in textwrap.wrap(text,160):draw.text((5,line),t,font=font,fill='black');line+=16
            start=max(150,line+10);rows=['original','query','consensus','disagreement','connection']
            for row,name in enumerate(rows):
                draw.text((3,start+row*145+5),name,font=font,fill='black')
                for i in range(8):
                    image=Image.fromarray(video[i]);pix=np.array(image).copy()
                    if name!='original':
                        for ix in a['selection_'+name][i//2]:
                            y,x=divmod(int(ix),16);patch=pix[y*28:(y+1)*28,x*28:(x+1)*28];pix[y*28:(y+1)*28,x*28:(x+1)*28]=(.45*patch+.55*np.array([255,220,0])).astype(np.uint8)
                    canvas.paste(Image.fromarray(pix).resize((140,140)),(115+i*140,start+row*145))
            p=BASE/'private_views'/f'{cid}.png';p.parent.mkdir(exist_ok=True);canvas.save(p);records.append(dict(case_id=cid,dataset=domain,candidate_id=key,path=str(p.relative_to(ROOT)),sha256=sha(p),selection='first two locked SHA sources per domain, first original candidate job',GT_read=False));a.close()
    put(BASE/'PRIVATE_VISUAL_REVIEW_INPUTS.json',dict(status='complete_pending_actual_view',count=4,records=records,GT_read=False))
    print(json.dumps(summary))
if __name__=='__main__':main()
