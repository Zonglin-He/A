"""Independent saved-record parser and Decimal CE accounting; CPU only."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import torch
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/desta3d_v2/tta_v2'
RUN=BASE/'source_fp32_head_control_v1'
OUT=BASE/'source_prefix_support16_v2'


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''): h.update(b)
    return h.hexdigest()
def write(name,x):
    with (OUT/name).open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def parse(text, rows):
    # Delimiter-based parser independent of the audit regex implementation.
    ref=text[:text.index('<|object_ref_end|>')+len('<|object_ref_end|>')]
    times=text.split('<|time_start|>',1)[1].split('<|time_end|>',1)[0]
    endpoints=[int(t.removeprefix('<t')) for t in times.split('>') if t]
    pos=[]
    if rows:
        remainder=text.split('<|time_end|>',1)[1]
        for line in remainder.splitlines():
            if '<|box_start|>' in line:
                pos.append(int(line.split('>',1)[0][2:]))
        assert pos==list(range(endpoints[0],endpoints[1]+1))
    return ref,endpoints,pos


def main():
    tick=time.monotonic();torch.set_num_threads(2)
    r=read(OUT/'REPORT.json');source=read(RUN/'SOURCE_RECORDS.json')
    sources={s['key']:s for s in source};rows=read(RUN/'INPUTS.json')
    tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B',local_files_only=True)
    arms=('no_update','supervised','supervised_fp32');summary={a:Counter() for a in arms}
    count_checks=0;support_checks=0;ce_checks=0;maxerr=0.;changes=[]
    for i,(row,c) in enumerate(zip(rows,r['cases'])):
        assert row['key']==c['key'];s=sources[c['key']]
        ref,ends,pos=parse(s['response'],True)
        refn=len(tok.encode(ref,add_special_tokens=False));boxn=len(pos)
        expected={'event_ntp':refn+4,'event_mtp':((refn+5)//6+1)*6,
                  'spatial_ntp':boxn*7,'spatial_mtp':boxn*6,'event':refn+4+((refn+5)//6+1)*6,'spatial':boxn*13}
        for k,v in expected.items(): assert c['counts'][k]==v;count_checks+=1
        # Count families by token IDs and source span, including nulls; no audit-family helper.
        ids_time={tok.convert_tokens_to_ids(f'<t{j}>') for j in range(1,101)}
        ids_coord={tok.convert_tokens_to_ids(f'<{j}>') for j in range(1001)}
        tids=c['label_side_all_supervised_targets']
        assert sum(t['source_span']=='time' and t['token_id'] in ids_time for t in tids)==4
        assert sum(t['source_span']=='box' and t['token_id'] in ids_coord for t in tids)==boxn*8
        assert sum(t['source_span']=='box' and t['token_id'] in ids_time for t in tids)==boxn
        assert 4/expected['event']==c['endpoint_fraction_of_event_CE_denominator']
        count_checks+=4
        active=[j+1 for j,f in enumerate(s['frame_ids']) if s['event_interval']['begin_fid']<=f<s['event_interval']['end_fid']]
        assert active==pos==[j+1 for j,a in enumerate(s['event_active']) if a]
        for a in arms:
            p=torch.load(RUN/'episodes'/f'{i:02}'/(a+'.pt'),map_location='cpu',weights_only=False)
            nref,ne,np=parse(p['spatial_completion'],True)
            eref,ee,_=parse(p['event_completion'],False)
            n=c['native'][a]
            assert nref==eref==n['reference'] and ne==ee==n['time_1based']
            assert np==n['positions_1based']==[x+1 for x in p['positions']]
            assert ne==[x+1 for x in p['interval']]
            extra=list(sorted(set(np).difference(pos)));missing=list(sorted(set(pos).difference(np)))
            assert extra==n['extra_positions_1based'] and missing==n['missing_positions_1based']
            assert [s['frame_ids'][v-1] for v in extra]==n['extra_physical_frames']
            assert [s['frame_ids'][v-1] for v in missing]==n['missing_physical_frames']
            assert (nref==ref)==n['reference_exact_GT'] and (ne==ends)==n['time_exact_GT']
            summary[a].update(semantic_reference_exact_GT=int(nref==ref),time_and_frame_support_exact_GT=int(ne==ends),
                has_extra_frame_support=int(bool(extra)),has_missing_GT_frames=int(bool(missing)),
                total_native_frames=len(np),invalid_geometry_frames=int((~p['geometry_valid']).sum()))
            support_checks+=1
        ep=RUN/'episodes'/f'{i:02}';saved=read(ep/'supervised_fp32/SUMMARY.json');old=read(ep/'OLD_SUPERVISED_DUAL_CE.json')
        for dt in ('BF16','FP32'):
            for label,point in [(f'new_before_step{j}',saved['history'][j-1][dt]) for j in (1,2,3)]+[('new_final',saved[dt+'_after']),('old_final',old[dt])]:
                for b in ('event','spatial'):
                    x=point[b]
                    terms={m:Decimal(str(x[m+'_loss']))*x[m+'_count']/x['tokens'] for m in ('ntp','mtp')}
                    ce=float(sum(terms.values()));maxerr=max(maxerr,abs(ce-x['ce']))
                    assert abs(ce-x['ce'])<2e-6
                    assert abs(ce-c['CE'][dt][label][b]['CE_reconstructed'])<2e-15
                    ce_checks+=1
            for final,point in [('new_final',saved[dt+'_after']),('old_final',old[dt])]:
                start=saved['history'][0][dt];change={}
                for b in ('event','spatial'):
                    change[b]=point[b]['ce']-start[b]['ce']
                    assert change[b]==c['CE'][dt][final+'_change'][b]['CE']
                changes.append({'key':c['key'],'definition':dt,'state':final,**change,
                    'sum':math.fsum(change.values()),'opposite_branch_signs':change['event']*change['spatial']<0,
                    'native_delta_B1_pp':c['native']['supervised_fp32' if final=='new_final' else 'supervised']['existing_delta_B1_pp']})
    assert {a:dict(x) for a,x in summary.items()}==r['aggregate']['by_arm']
    assert sum(len(parse(s['response'],True)[2]) for s in source)==r['aggregate']['GT_box_rows']
    aggregates={}
    for dt in ('BF16','FP32'):
        for final in ('old_final','new_final'):
            subset=[x for x in changes if x['definition']==dt and x['state']==final]
            aggregates[dt+'_'+final]={
                'event_CE_decrease_count':sum(x['event']<0 for x in subset),
                'spatial_CE_decrease_count':sum(x['spatial']<0 for x in subset),
                'total_CE_decrease_count':sum(x['sum']<0 for x in subset),
                'opposite_branch_signs':sum(x['opposite_branch_signs'] for x in subset),
                'total_CE_decrease_and_native_v_harm':sum(x['sum']<0 and x['native_delta_B1_pp']<0 for x in subset),
                'mean_change':{k:math.fsum(x[k] for x in subset)/16 for k in ('event','spatial','sum')}}
    assert not torch.cuda.is_initialized()
    result={'status':'passed','time':time.time(),'script_sha':sha(Path(__file__)),'report_sha':sha(OUT/'REPORT.json'),
        'independent_parser':'string delimiters; independent closed counts; Decimal weighted CE from original metadata',
        'count_checks':count_checks,'native_support_checks':support_checks,'CE_checks':ce_checks,'max_saved_CE_reaggregation_error':maxerr,
        'all_aggregate_counts_exact':True,'CE_branch_changes':changes,'CE_branch_summary':aggregates,
        'new_model_calls':0,'new_labels':False,'target_data':False,'CUDA_initialized':False,
        'CPU_seconds':time.monotonic()-tick,'free_disk_bytes':shutil.disk_usage(ROOT).free}
    write('ROOT_CROSSCHECK.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='CE_branch_changes'},indent=2))


if __name__=='__main__':main()
