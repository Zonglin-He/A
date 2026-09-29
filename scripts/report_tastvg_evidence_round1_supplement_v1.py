"""Append disjoint-cohort findings, interpretation and provenance to sealed readback."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_evidence_vulnerability_v1/full64_v1'

def main():
    d=OUT/'round1_readback';ss=read(d/'SUBSETS.json');rows=read(d/'ROWS.json');audit=read(d/'SUPPLEMENTAL_AUDIT.json')
    lines=['','## Disjoint pilot and extension readout','',
           'All panels are historically exposed development data; extension48 was not used to choose the attack configuration. The prelocked pilot decision only controlled whether to pay for the remaining48.','',
           '|Subset|Unique strong optimized parents|Unique strong random parents|','|---|---:|---:|']
    for name,x in ss.items():lines.append(f"|{name}|{x['unique_strong']}|{x['unique_random_strong']}|")
    lines+=['','Counts below require the same strict preservation and self-vIoU>.95; component categories overlap. TTS threshold=.01 JSD; ASA/query=.10 cosine distance; selection Jaccard<=.5.','',
       '|Subset|rho|Strong any|TTS|ASA|Query|Selection|Strong random|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,x in ss.items():
        for rho,z in x['by_rho'].items():
            t=z['types'];lines.append(f"|{name}|{rho}|{z['strong']}|{t['TTS']}|{t['ASA']}|{t['Query']}|{t['selection']}|{z['random_strong']}|")
    # Representative cases ranked only by internal evidence, without any ground truth.
    records=[]
    for r in rows:
        values=[(float(v),k,offset) for offset,m in enumerate(r['metrics']) for k,v in m.items() if v is not None and k.startswith('ASA') and k.endswith('cosine_drift')]
        v,k,offset=max(values)
        top=r['metrics'][offset][k.replace('cosine_drift','top20_iou')]
        records.append(dict(key=r['key'],cohort=r['cohort'],rho=r['rho'],component=k,offset=offset,drift=v,ASA_top20_IoU=top,self_vIoU=r['preservation']['self_vIoU'],selected_step=r['selected_step'],relative_norm=r['relative_norm']))
    records.sort(key=lambda r:r['drift'],reverse=True)
    selected=[];keys=set()
    for r in records:
        if r['key'] in keys:continue
        keys.add(r['key']);selected.append(r)
        if len(selected)==5:break
    write(d/'REPRESENTATIVE_CASES.json',selected)
    lines+=['','## Largest ASA changes (descriptive cases, selected post hoc without GT)','',
      '|Query|rho|Component/offset|Cosine drift|Top20 patch IoU|Self-vIoU|Selected step|','|---|---:|---|---:|---:|---:|---:|']
    for r in selected:lines.append(f"|{r['key']}|{r['rho']}|{r['component']}/{r['offset']}|{r['drift']:.6f}|{r['ASA_top20_IoU']:.6f}|{r['self_vIoU']:.6f}|{r['selected_step']}|")
    lines+=['','## Scope of the positive finding','',
      'The measured claim is existence of internal-evidence sensitivity with a stable final native tube under these finite latent attacks. Native correctness was not scored; stable predictions are not automatically correct predictions. Evidence drift is not by itself evidence that explanations became semantically wrong. High top20 overlap can coexist with large cosine drift because attention mass redistributes among mostly the same high-ranked patches. Report both, rather than interpreting cosine drift as wholesale relocation.',
      '', 'The random control uses the nominal radius and one direction. The optimized selection can have a smaller norm and searches 11 visited states; beating this control is a screening comparison, not proof of optimal attack efficiency or a statistically controlled ranking. Larger budgets are separate finite searches, so selected drift/strong-case counts need not be monotonic.',
      '',f"Additional independent random-control readback: {audit['random_comparisons']} quantities, max error {audit['random_max_error']:.3g}; all {audit['gradient_steps']} saved gradient norms finite, {audit['zero_gradients']} exactly zero; all six saved full-pipeline reinsertion validations passed. All16 loader-isolated pilot baselines have bitwise identical H and evidence to the original loader attempt; numerical identity does not erase the earlier incidental file exposure.",
      '', '## Constructor I/O correction and regeneration', '',
      'The initial v1 engineering run incidentally deserialized the official constructor runtime annotation dictionaries. GT values were not used in losses, selection, scoring or threshold choice; the analyst inspected dictionary schema to locate the problem. The run was interrupted with all24 completed arms/failures retained. Every scientific pilot capture/attack was regenerated in no_gt_v2 with empty constructor annotation dictionaries and a process-wide open-denial audit. Only these strict pilot artifacts were reused here; the48 new captures/144 arms used the same guard. The whole-session claim is no GT used for optimization/evaluation, not no incidental annotation file ever opened. Original118.331861s remain charged in the cumulative GPU-process total.',
      '', '## Timing and completion', '',
      'The full64 set contains192 attack arms and1920 paired-offset backwards, with480 from the sealed pilot reused exactly once and1440 new backwards. The cumulative process total includes pilot regeneration, old aborted v1, imports/loading, capture, replay checks and attack. CPU development, audits and report generation are separate; the total is not CUDA-event kernel time. All current Round1 GPU workers exited; Round2/3 were not started and PTD remains paused.']
    p=OUT/'REPORT.md';s=p.read_text().replace('no GT opened, no expert','strict scientific workers opened no GT, no expert')
    s+='\n'.join(lines)+'\n';p.write_text(s)
    print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
