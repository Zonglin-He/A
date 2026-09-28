"""CPU-only full-panel readback; outcome-selected cases explicitly diagnostic."""
import sys,re,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from vg_tta.desta3d_v3_oracle_io import read,write,sha,PANEL,verify_seal
from vg_tta.desta3d_v3_free_actuation import native_targets,margin
from scripts.desta3d_v3_privileged_ptd_qualification import B1

def run():
    old=ROOT/'artifacts/desta3d_v3/latent_oracle_v1/oracle001';out=old.parent/'diagnosis_cpu_v1'
    assert not out.exists();verify_seal(old);report=read(old/'independent_readback_v1/REPORT.json')
    labels=read(PANEL/'SOURCE_RECORDS.json');rows=read(old/'INPUTS.json');state=torch.load(B1,weights_only=False,map_location='cpu')['adapter']
    gates={b:{'stored_logit':float(state['gate_'+b]),'sigmoid':float(state['gate_'+b].sigmoid()),
              'projection_frobenius':float(state['out_proj_'+b+'.weight'].double().norm()),
              'projection_bias_norm':float(state['out_proj_'+b+'.bias'].double().norm())} for b in ['event','spatial']}
    result=[]
    for i,(row,lab) in enumerate(zip(rows,labels)):
        ep=old/'episodes'/f'{i:02}';ids=lab['frame_ids'];valid=np.array(lab['box_valid'],bool);boxes=np.array(lab['boxes_xyxy']);bb=boxes[valid]
        areas=(bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1]);centers=(boxes[:,:2]+boxes[:,2:])/2
        motion=[float(np.linalg.norm(centers[j]-centers[j-1])) for j in range(1,len(ids)) if valid[j] and valid[j-1]]
        cap=lab['caption'].lower();keywords=[x for x in ['behind','in front of','above','below','next to','beside','with','another','other','between','near'] if re.search(r'\b'+re.escape(x)+r'\b',cap)]
        case={'index':i,'key':row['key'],'source':row['source'],'caption':lab['caption'],'category':lab['category'],
              'annotation':{'mean_target_area':float(areas.mean()) if len(areas) else None,'valid_box_frames':int(valid.sum()),
              'missing_box_frames':int((~valid).sum()),'mean_normalized_center_motion_adjacent_observations':float(np.mean(motion)) if motion else None,
              'event_observation_fraction':float(np.mean(lab['event_active'])),
              'event_physical_duration_over_observed_clip':(lab['event_interval']['end_fid']-lab['event_interval']['begin_fid'])/(ids[-1]-ids[0]+1),
              'relation_lexical_proxy':keywords,'actual_relation_dependence':None,'distractor_count':None,'occlusion':None,
              'missing_reason':'Existing selected-referent records do not annotate distractors or occlusion; lexical proxy is not causal evidence'},'arms':{}}
        for arm in ['original','temporal','wrong_temporal','spatial','wrong_spatial','dual']:
            p=torch.load(ep/(arm+'.pt'),weights_only=False,map_location='cpu');entry={'metrics':report['arms'][arm]['rows'][i]['metrics'],'interval':p['interval'],'positions':p['positions'],'margins':{}}
            for branch in ['event','spatial']:
                logits=p['time_distribution']['endpoint_logits'] if branch=='event' else p['readout']['coordinate_logits']
                if logits is None:entry['margins'][branch]=None;continue
                try:t,v=native_targets(lab,p,branch)
                except ValueError as e:entry['margins'][branch]={'missing':str(e)};continue
                values=margin(logits,t,v)
                entry['margins'][branch]={'mean':float(np.mean(values)),'all':values,'target_classes':t.tolist(),'valid':v.tolist(),
                    'positive_fraction':float(np.mean(np.array(values)>0)), 'native_argmax':logits.argmax(-1).tolist()}
            case['arms'][arm]=entry
        s=case['arms'];case['spatial_delta_pp']=100*(s['spatial']['metrics']['sIoU']-s['original']['metrics']['sIoU'])
        case['spatial_correct_minus_wrong_pp']=100*(s['spatial']['metrics']['sIoU']-s['wrong_spatial']['metrics']['sIoU'])
        result.append(case)
    select={'temporal_flip':7,'spatial_largest_gain':max(range(16),key=lambda i:result[i]['spatial_delta_pp']),
            'spatial_largest_correct_below_wrong':min(range(16),key=lambda i:result[i]['spatial_correct_minus_wrong_pp'])}
    groups={}
    for label,test in [('positive',lambda d:d>1e-10),('negative',lambda d:d< -1e-10),('zero',lambda d:abs(d)<=1e-10)]:
        rr=[r for r in result if test(r['spatial_delta_pp'])]
        groups[label]={'n':len(rr),'mean_area':float(np.mean([r['annotation']['mean_target_area'] for r in rr])),
          'relation_lexical_count':sum(bool(r['annotation']['relation_lexical_proxy']) for r in rr),
          'mean_motion':float(np.mean([r['annotation']['mean_normalized_center_motion_adjacent_observations'] for r in rr]))}
    payload={'status':'completed_CPU_existing_sealed_evidence','gates':gates,'cases':result,'spatial_groups':groups,
             'selection':select,'selection_scope':'Explicit prior-outcome-selected source diagnostic, not unbiased validation',
             'GPU_used':False,'new_forward':0,'target_read':False,'source_GT_read':True,
             'pins':{str(p):sha(p) for p in [Path(__file__),B1,PANEL/'SOURCE_RECORDS.json',old/'PREDICTIONS_SEAL.json',old/'independent_readback_v1/REPORT.json']}}
    write(out/'REPORT.json',payload)
    lines=['# Existing source oracle case and margin readback','', 'All16 existing source cases; no new GPU or predictions. Selection below is deliberately outcome-exposed diagnostic.', '',str(gates),'',str(groups),'',
           '|key|correct S delta pp|correct minus wrong S pp|GT mean box area|relation lexical proxy|','|---|---:|---:|---:|---|']
    for r in result:lines.append(f"|{r['key']}|{r['spatial_delta_pp']:+.6f}|{r['spatial_correct_minus_wrong_pp']:+.6f}|{r['annotation']['mean_target_area']:.4f}|{','.join(r['annotation']['relation_lexical_proxy'])}|")
    lines+=['','Unknown distractors/occlusion and causal context dependence remain unknown. Box movement is not camera-corrected object motion. Full margins (not just argmax), all positives/negatives/ties, source provenance and label support are in REPORT.json.','',
      'For positive scalar weights, the pre-cast intervention is g W ((w-1) z); projection bias cancels. This changes already-read residual amplitude. It does not mathematically prove zero information injection: spatially varying masking itself carries support information. Correct versus wrong did not establish preferential utility in this fixed panel.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'gates':gates,'selection':select,'groups':groups}))
if __name__=='__main__':run()
