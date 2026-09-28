"""Read every final native grammar/endpoint; distinguish structural invariance."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,PANEL,read,write,sha,verify_seal

def run(name):
    import torch
    d=OUT/name;verify_seal(d);r=read(d/'independent_readback_v1/REPORT.json')
    assert read(d/'independent_readback_v1/COMPLETE.json')['report_sha']==sha(d/'independent_readback_v1/REPORT.json')
    labels={x['key']:x for x in read(PANEL/'SOURCE_RECORDS.json')};rows=read(d/'INPUTS.json');out=[];failures=[]
    sums={a:0. for a in r['arms']};same_S=0
    def load(p):return torch.load(p,map_location='cpu',weights_only=False)
    for i,row in enumerate(rows):
        ep=d/'episodes'/f'{i:02}';base=load(ep/'original.pt');registry=load(ep/'BASE_TRACE.pt')['token_ids'];lab=labels[row['key']]
        gt=lab['event_interval'];case={'index':i,'key':row['key'],'arms':{}}
        for a in r['arms']:
            p=load(ep/(a+'.pt'));iv=p['interval'];fid=p['frame_ids'];valid=iv is not None and 0<=iv[0]<=iv[1]<len(fid)
            t=0.
            if valid:
                ps,pe=fid[iv[0]],fid[iv[1]]+1;gs,ge=gt['begin_fid'],gt['end_fid']
                overlap=max(0,min(pe,ge)-max(ps,gs));t=overlap/max(1,(pe-ps)+(ge-gs)-overlap)
            sums[a]+=t/16
            raw=p['readout']['raw_blocks'];bad=[]
            if raw is not None:
                for j,block in enumerate(raw.tolist()):
                    for k,tok in enumerate(block):
                        allowed=(tok==registry['box_start']) if k==0 else ((tok==registry['box_end']) if k==5 else tok in registry['coord_id_to_value'])
                        if not allowed:bad.append(dict(block=j,slot=k,token=tok))
            same_time=torch.equal(p['time_distribution']['endpoint_logits'],base['time_distribution']['endpoint_logits']) if p['time_distribution']['endpoint_logits'] is not None else False
            z=dict(format_ok=p['format_ok'],direct_endpoint_tIoU=t,official_tIoU=r['arms'][a]['rows'][i]['metrics']['tIoU'],
                interval=p['interval'],same_time_logits=same_time,same_event_completion=p['event_completion']==base['event_completion'],
                same_reference=p['readout']['spatial_reference_token_ids']==base['readout']['spatial_reference_token_ids'],
                invalid_box_grammar=bad,geometry_invalid_count=int((~p['geometry_valid']).sum()))
            if a=='S_only':assert z['same_time_logits'] and z['same_event_completion'];same_S+=1
            if not p['format_ok']:failures.append(dict(index=i,key=row['key'],arm=a,**z))
            case['arms'][a]=z
        out.append(case)
    result=dict(status='all_native_cases_read_back',predictions=96,spatial_only_event_equal=same_S,raw_grammar_failures=failures,
        direct_endpoint_tIoU_parent_macro=sums,cases=out,scope='Direct endpoint metric ignores whole tube grammar ONLY for diagnosis; original official scores unchanged. S-only endpoint invariance is structural. All cases retained; no sample or state selection.',new_GPU=False,target_read=False)
    write(d/'ROOT_NATIVE_CASE_READBACK.json',result)
    print({'S_only_event_equal':same_S,'format_failures':len(failures),'direct_endpoint_macro':sums})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)
