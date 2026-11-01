"""Score sealed direct outputs; oracle strata descriptive, never a runtime gate."""
import sys,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round22'))
from common22 import *
def summarize(rows,baseline):
    out={}
    for group in ['ALL','D','W','Z','OFFERED_RECOVERABLE25','OFFERED_RECOVERABLE25_WITH_REL','ALL256_RECOVERABLE25']:
        out[group]={}
        for platform in ['drone','quad']:
            rr=[r for r in rows if r['platform']==platform and (group=='ALL' or r['phase']==group or group=='OFFERED_RECOVERABLE25' and r['offered_recoverable25'] or group=='OFFERED_RECOVERABLE25_WITH_REL' and r['offered_recoverable25'] and r['has_reference_relation'] or group=='ALL256_RECOVERABLE25' and r['all256_recoverable25'])]
            if not rr:out[group][platform]={'n':0};continue
            scenes=sorted({r['scene_PRIVATE'] for r in rr});ix=np.array([scenes.index(r['scene_PRIVATE']) for r in rr]);weights=np.eye(len(scenes),dtype=int)[np.random.default_rng(2022).integers(0,len(scenes),(10000,len(scenes)))].sum(1);counts=np.bincount(ix,minlength=len(scenes));arms={}
            for arm in rr[0]['ious']:
                v=np.array([r['ious'][arm] for r in rr]);base=np.array([r['ious'][baseline] for r in rr]);source=np.array([r['ious']['source'] for r in rr]);delta=v-base;sums=np.bincount(ix,weights=delta,minlength=len(scenes));boot=(weights@sums)/(weights@counts)*100
                arms[arm]={'n':len(rr),'scene_clusters':len(scenes),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float((v-source).mean()*100),'gain_vs_no_reference_pp':float(delta.mean()*100),'gains_vs_no_reference':int((delta>1e-10).sum()),'harms_vs_no_reference':int((delta< -1e-10).sum()),'changed_vs_no_reference':sum(r['q'][arm]!=r['q'][baseline] for r in rr),'scene_cluster95_gain_pp':list(map(float,np.quantile(boot,[.025,.975])))}
            out[group][platform]=arms
    return out
def main(args):
    lock('SCORE_LOCK.json');pred=BASE/args.pred;verify(pred);out=EXP/args.out;out.mkdir(exist_ok=False);dump(out/'PRELABEL_LOCK.json',{'prediction_seal_sha256':sha(pred/'SEALED.json'),'offline_label_cache_sha256':sha(EXP/'OFFLINE_RECOVERABILITY_PRIVATE.json'),'all208_historically_exposed':True,'oracle_strata_diagnostic_only':True,'score_program_sha256':sha(__file__),'baseline':args.baseline});oracle={r['id_PRIVATE']:r for r in read(EXP/'OFFLINE_RECOVERABILITY_PRIVATE.json')};oldscore={r['id_PRIVATE']:r for r in read(ROOT/'research/round21/experiment/scored208_attempt003/ROWS_PRIVATE.json')};rows=[]
    for r in read(ROWS):
        z=feature(r);o=oracle[r['id']]
        with np.load(pred/'predictions_PRIVATE'/(r['id']+'.npz')) as p:names=p['arm_names'].tolist();q=p['query_indices'].copy();assert np.array_equal(p['arm_boxes_raw6'],z['boxes'][q]);iou=np.array(o['candidate_ious_PRIVATE'])[q]
        values=dict(zip(names,map(float,iou)));assert abs(values['source']-oldscore[r['id']]['ious']['source'])<1e-12
        rows.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'phase':r['stage'] if r['stage'] in ['W','Z'] else 'D','scene_PRIVATE':r['scene'],'ious':values,'q':dict(zip(names,map(int,q))),'offered_recoverable25':o['offered_recoverable25'],'all256_recoverable25':o['all256_recoverable25'],'has_reference_relation':o['has_reference_relation']})
    summary=summarize(rows,args.baseline);dump(out/'ROWS_PRIVATE.json',rows);dump(out/'SUMMARY.json',summary);seal(out,'R22_all208_direct_outputs_scored_after_sealing_oracle_strata_descriptive',n=208,baseline=args.baseline);print('SCORED22',args.out,{p:{a:round(v['mIoU_percent'],6) for a,v in summary['ALL'][p].items()} for p in ['drone','quad']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pred',required=True);p.add_argument('--out',required=True);p.add_argument('--baseline',required=True);main(p.parse_args())
