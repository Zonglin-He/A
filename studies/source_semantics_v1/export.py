"""Re-export already scored full means with explicit source/readout names. No GT/model access."""
from pathlib import Path
import json,hashlib,csv
R=Path(__file__).resolve().parents[2];P=R/'results/source_semantics_v1';D=R/'outputs/source_semantics_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs={};out={};signed=[]
for ds,count in [('hc2',1902),('vidstg',10303)]:
    path=R/'results/paper_review_suite_v1/cached'/ds/'report.json';inputs[str(path.relative_to(R))]=sha(path);r=json.loads(path.read_text());assert r['queries']==count
    keys={'SourceOriginalReadout':'SourceBox_P0Native','SourcePlusMBR':'SourceBox_P0MBR','Full':'FullBox_Soft'}
    means={name:{k:100*r['means'][key][k]for k in ['t','v','s']}for name,key in keys.items()}
    pairs=[('SourcePlusMBR','SourceOriginalReadout'),('Full','SourcePlusMBR'),('Full','SourceOriginalReadout')]
    delta={a+'-'+b:{k:means[a][k]-means[b][k]for k in ['t','v']}for a,b in pairs}
    out[ds]=dict(queries=count,parents=r['parents'],means_percent=means,delta_pp=delta)
    signed.extend([[ds,count,a,means[a]['t'],means[a]['v'],means[a]['s']]for a in keys])
code={n:sha(R/n)for n in ['external/TA-STVG/models/post_processor.py','external/TA-STVG/engine/evaluate.py','stvg/readout.py','stvg/predictor.py','stvg/temporal.py']}
result=dict(schema='source_readout_semantics_correction_v1',status='verified_existing_scored_means_only',inputs_sha256=inputs,code_sha256=code,results=out,
    semantics=dict(SourceOriginalReadout='Per-offset nativeMAP start/end and native physical hull; local fixed-input/precision protocol, not a claim of matching published in-domain paper accuracy.',SourcePlusMBR='Same frozen source boxes/logits plus OUR physical-hull posterior construction and OUR MBR decision; changed inference rule, not untouched official source.',Full='Our complete spatial and temporal feedback with MBR; Full-SourcePlusMBR isolates incremental feedback under matched readout; Full-SourceOriginalReadout is overall system gain and includes readout gains.'),
    CG_frozen_file_aliases={'SourceNative':'CG-STVG Source (official native MAP readout)','Source':'CG-STVG Source + MBR','Full':'CG-STVG + selected feedback + MBR'},
    TA_frozen_file_aliases={'Source':'TA-STVG Source + MBR','SourceBox_P0Native':'TA-STVG Source (official temporal readout, local fixed-input protocol)','SourceBox_P0MBR':'TA-STVG Source + MBR'},
    new_models=0,new_Source=0,new_expert=0,new_GT_scoring=0,old_predictions_scores_protocols_immutable=True)
P.mkdir(parents=True,exist_ok=True);D.mkdir(parents=True,exist_ok=True)
for p in [P/'comparison.json',D/'comparison.json']:p.write_text(json.dumps(result,indent=2)+'\n')
with(P/'comparison.tsv').open('w')as f:
 w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['dataset','queries','display_arm','tIoU_percent','vIoU_percent','sIoU_percent']);w.writerows(signed)
print(json.dumps(out,indent=2))
