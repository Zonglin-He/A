"""Post-score descriptive corruption recovery; no new selection gate or inference."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write
from scripts.run_tastvg_corruption_c0c1_v1 import OUT

def run():
    a=read(OUT/'analysis/C0_ROWS.json');b=read(OUT/'analysis/C1_ROWS.json');mapping={(r['key'],r['condition']):r for r in a};out={}
    for branch,metric in [('S','sIoU'),('T','tIoU')]:
        rr=[]
        for r in b:
            if r['condition']=='clean':continue
            base=mapping[r['key'],r['condition']];loss=-base['delta'][metric]
            if loss<=.05:continue
            gain=r['gain_'+branch];rr.append(dict(key=r['key'],ordinal=r['ordinal'],condition=r['condition'],clean_metric=base['baseline_clean'][metric],corrupted_metric=base['native'][metric],oracle_metric=r['arms'][branch+'_oracle'][metric],corruption_loss=loss,oracle_gain=gain,restores_clean=bool(gain>=loss-1e-12)))
        out[branch]=dict(cells=len(rr),parents=len({x['key'] for x in rr}),restores_clean_cells=sum(x['restores_clean'] for x in rr),mean_corruption_loss=sum(x['corruption_loss'] for x in rr)/len(rr) if rr else None,mean_oracle_gain=sum(x['oracle_gain'] for x in rr)/len(rr) if rr else None,recovered_loss_fraction_capped_at_clean=sum(min(x['oracle_gain'],x['corruption_loss']) for x in rr)/sum(x['corruption_loss'] for x in rr) if rr else None,rows=rr)
    result=dict(status='post_score_descriptive_readback',selection='all fixed C1 corrupted cells with >5pp corresponding clean-to-corrupted target damage; threshold reused from locked tail definition',not_new_gate=True,branches=out)
    f=OUT/'analysis/CORRUPTION_RECOVERY.json'
    if f.exists():assert result==read(f)
    else:write(f,result)
    tails={arm:dict(negative=sum(r['arms'][arm]['vIoU_corrected']<r['arms']['native']['vIoU_corrected']-1e-12 for r in b),worst=min(r['arms'][arm]['vIoU_corrected']-r['arms']['native']['vIoU_corrected'] for r in b)) for arm in ['S_oracle','T_oracle','TS_oracle']}
    write(OUT/'analysis/C1_MINOR_VIDEO_TAILS.json',dict(status='post_score_descriptive_readback',cells=64,arms=tails));print('Recovery and minor-tail readback completed without inference')

if __name__=='__main__':run()
