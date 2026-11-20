"""Post-seal error decomposition, with separate populations and paired denominators."""
from collections import Counter
import csv
import numpy as np
from scipy.stats import spearmanr
from paths import OUT,PRIV,write,PLATFORMS
from geometry import ARMS

def values(rows,key):
    return np.array([float(r[key]) for r in rows if r.get(key) not in ('',None)],float)

def median(rows,key,absolute=False):
    v=values(rows,key)
    if not len(v):return None
    return float(np.median(abs(v) if absolute else v))

def quartiles(v):
    return np.quantile(v,[.25,.5,.75]).tolist() if len(v) else []

def main():
    with (PRIV/'diagnostic_rows.csv').open() as f:rows=list(csv.DictReader(f))
    tables={}
    for t in (*PLATFORMS,'all'):
        block=[r for r in rows if t=='all' or r['target']==t]
        subsets={'all':block,'confirmed_2d_target':[r for r in block if r['id2d']=='target'],
                 'confirmed_native_projection':[r for r in block if r['id3d_projection']=='target'],
                 'both_fit_windows':[r for r in block if min(int(r['fit_bins']),int(r['wrong_fit_bins']))>=5]}
        for name,v in subsets.items():
            ss={}
            for arm in ARMS:
                ss[arm]=dict(depth_signed_med=median(v,'depth_error_'+arm),depth_abs_med=median(v,'depth_error_'+arm,True),
                             center_error_med=median(v,'center_error_'+arm),valid_depth=len(values(v,'depth_error_'+arm)))
            tables[t+'_'+name]=dict(n=len(v),geometry=ss,id2d=dict(Counter(r['id2d'] for r in v)),
                id3d_projection=dict(Counter(r['id3d_projection'] for r in v)),
                qwen3_status=dict(Counter(r['status3'] for r in v)),qwen2_status=dict(Counter(r['status2'] for r in v)),
                real_support=sum(int(r['fit_bins'])>=5 for r in v),mismatched_support=sum(int(r['wrong_fit_bins'])>=5 for r in v),
                median_evaluable_fit_fraction_on_target=median(v,'fit_inside_gt_fraction'),
                fit_purity_eligible=len(values(v,'fit_inside_gt_fraction')),angle_degrees_med=median([r for r in v if float(r.get('gt_aspect') or 0)>=1.2],'abs_yaw_deg'),
                angle_eligible=sum(float(r.get('gt_aspect') or 0)>=1.2 for r in v),
                median_relative_extent_error_pct=median(v,'size_relative_pct'),
                median_extent_abs_m={k:median(v,'size_'+k+'_abs') for k in ('l','w','h')},
                median_native_projection_vs_annotated=median(v,'projection_agreement'),
                annotated_image_rects=sum(r['GT_image_rect']=='1' for r in v),
                scenes_with_listed_distractors=sum(int(r['available_distractors'])>0 for r in v),
                median_listed_distractors=median(v,'available_distractors'),
                native_3d_and_2d_both_target=sum(r['id2d']==r['id3d_projection']=='target' for r in v),
                target_depth_quartiles_m=quartiles(values(v,'target_optical_depth')))
    write(OUT/'diagnosis.json',tables)
    # Fixed optical-distance bands keep recording mix and target distance visible.
    # These post-seal summaries have no path back to the predicted boxes.
    bands=[]
    for target in (*PLATFORMS,'all'):
        valid=[r for r in rows if (target=='all' or r['target']==target) and r.get('target_optical_depth')]
        for low,high in ((0,10),(10,20),(20,40),(40,float('inf')),(0,float('inf'))):
            selected=[r for r in valid if low<=float(r['target_optical_depth'])<high]
            if not selected:
                continue
            depths=values(selected,'target_optical_depth')
            for arm in ('raw','ray_depth'):
                err=values(selected,'depth_error_'+arm)
                corr=None
                if len(selected)>=3 and np.ptp(depths)>0 and np.ptp(err)>0:
                    corr=float(spearmanr(depths,err).statistic)
                bands.append(dict(target=target,depth_low_m=low,depth_high_m=high if np.isfinite(high) else None,
                    n=len(selected),arm=arm,median_signed_error_m=float(np.median(err)),
                    median_absolute_error_m=float(np.median(abs(err))),
                    median_predicted_to_true_depth=float(np.median((depths+err)/depths)),
                    rank_correlation_true_depth_and_signed_error=corr))
    write(OUT/'depth_bands.json',bands)
if __name__=='__main__':main()
