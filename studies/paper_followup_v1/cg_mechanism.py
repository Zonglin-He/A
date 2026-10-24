"""GT-OFFLINE diagnosis of already scored CG development trajectories, not tuning."""
from common import *
import argparse,time

def prepare():
    freeze('cg_mechanism',dict(scope='Only previous CG development64 parents/domain; already scored and exposed. No new confirmation labels. Offline diagnostic only; no inference gate/parameter choice/extra configurations.',old_seal_sha256=sha(OLD/'cgstvg_v9_screen_sealed.json'),dev_selection_sha256=sha(D/'cg_selection.json'),oracle='Per-frame best legal candidate IoU is a diagnostic upper bound, not a realizable trajectory or method score.',questions=['shared correction bias','legal propagated candidate coverage versus selection','appearance score alignment with IoU'],new_model=0,new_expert=0),['common.py','cg_mechanism.py'])

def run():
    import numpy as np,torch,csv
    from scipy.stats import spearmanr
    from stvg.spatial import SpatialGeometry,evidence_dp
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    from stvg.consensus import encode
    from cgstvg_runtime_v9 import clip_native_boxes
    torch.set_num_threads(1);p=validate('cg_mechanism');seal=read(OLD/'cgstvg_v9_screen_sealed.json');assert sha(OLD/'cgstvg_v9_screen_sealed.json')==p['old_seal_sha256'];assert sha(D/'cg_selection.json')==p['dev_selection_sha256'];assert(OLD/'cgstvg_v9_screen_score_receipt.json').exists();started=time.perf_counter();summary={};checked=0
    def avg(v):return float(np.mean(v))if len(v)else None
    for ds,groups in read(D/'cg_selection.json').items():
        truth=labels(ds);rows=[]
        for row in groups['dev']:
            key=row['id'];e=seal['files'][ds][key];assert sha(OLD/e['path'])==e['sha256']and sha(OLD/e['capture_path'])==e['capture_sha256'];cap=torch.load(OLD/e['capture_path'],map_location='cpu',weights_only=False);geo=SpatialGeometry.from_cached_geometry(cap['method_cache']['geometry']);bank=geo.for_interval(cap['native_mbr']);replay=evidence_dp(bank);ids=cap['frame_ids'];gt={int(k):v for k,v in truth[key]['boxes'].items()};W,H=row['metadata']['width'],row['metadata']['height'];pos=set(cap['positions8']);T=len(ids)
            with np.load(OLD/e['path'])as z:
                assert np.array_equal(clip_native_boxes(replay['final'].numpy()),z['Space']);checked+=1
                source=xyxy(z['Source'],W,H);E=xyxy(z['SharedCorrection'],W,H);space=xyxy(z['Space'],W,H)
            boxes=np.maximum(xyxy(geo.boxes.numpy().reshape(-1,4),W,H),0).reshape(T,3,4);reference=np.maximum(xyxy(geo.native_boxes.numpy(),W,H),0);active=[];fixed=[];anchor=[];unary=[];quality=[];margins=[];support=[]
            for t,f in enumerate(ids):
                if f not in gt:continue
                src=box_iou(source[t],gt[f]);ee=box_iou(E[t],gt[f]);ss=box_iou(space[t],gt[f])
                if t in pos:
                    j=cap['positions8'].index(t);anchor.append([src,box_iou(reference[j],gt[f])]);continue
                fixed.append([src,ee,ss]);support.append(bool(geo.supported[t]))
                if not bank.event[t]:continue
                valid=bank.valid[t].numpy();iou=np.asarray([box_iou(b,gt[f])for b in boxes[t]]);best=float(iou[valid].max());choice=int(replay['actions'][t]);active.append([src,ee,ss,best,float(iou[choice])]);unary.extend(geo.u[t].numpy()[valid].tolist());quality.extend(iou[valid].tolist());margins.append(float(geo.u[t,choice]-geo.u[t,0]))
            fixed=np.asarray(fixed).reshape(-1,3);active=np.asarray(active).reshape(-1,5);anchor=np.asarray(anchor).reshape(-1,2);rho=float(spearmanr(unary,quality).statistic)if len(unary)>2 and np.ptp(unary)>0 and np.ptp(quality)>0 else None
            rows.append(dict(id=key,fixed_nodes=len(fixed),active_nodes=len(active),GT_anchor_nodes=len(anchor),fixed_Source=float(fixed[:,0].mean())if len(fixed)else None,fixed_E=float(fixed[:,1].mean())if len(fixed)else None,fixed_Space=float(fixed[:,2].mean())if len(fixed)else None,active_Source=float(active[:,0].mean())if len(active)else None,active_E=float(active[:,1].mean())if len(active)else None,active_Space=float(active[:,2].mean())if len(active)else None,active_best_candidate=float(active[:,3].mean())if len(active)else None,selection_gap=float((active[:,3]-active[:,4]).mean())if len(active)else None,candidate_opportunity=float((active[:,3]-active[:,1]).mean())if len(active)else None,anchor_Source=float(anchor[:,0].mean())if len(anchor)else None,anchor_expert=float(anchor[:,1].mean())if len(anchor)else None,appearance_IoU_spearman=rho,selected_unary_margin_over_E=avg(margins),supported_fraction=avg(support),shared_encoded_shift=np.median(encode(geo.E.numpy())-encode(cap['Source']),axis=0).tolist()))
        fields=[k for k,v in rows[0].items()if k not in ['id','shared_encoded_shift']];means={k:avg([r[k]for r in rows if r[k]is not None])for k in fields};summary[ds]=dict(queries=len(rows),query_equal_means=means,interpretation='All GT diagnostics offline on existing exposed DEV only. Candidate upper bound is per-frame and not a legal/global trajectory; rank correlation is descriptive, not proof of causal feature failure.',source_path_replay_bitwise=True)
        folder=P/'cg_mechanism'/ds;atomic(folder/'summary.json',summary[ds]);folder.mkdir(parents=True,exist_ok=True)
        with(folder/'all_signed.tsv').open('w')as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query']+fields+['dx','dy','dlogw','dlogh']);w.writerows([[f'Q{i:03}']+[r[k]for k in fields]+r['shared_encoded_shift']for i,r in enumerate(rows,1)])
        atomic(D/'cg_mechanism'/ds/'private_rows.json',rows)
    atomic(D/'cg_mechanism_receipt.json',dict(status='completed_offline_DEV_only',path_replay_checks=checked,summary=summary,new_model=0,new_expert=0,new_confirmation_GT=0,seconds=time.perf_counter()-started));atomic(P/'cg_mechanism/receipt.json',read(D/'cg_mechanism_receipt.json'));print(summary)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run']);z=a.parse_args();{'prepare':prepare,'run':run}[z.command]()
