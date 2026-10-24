"""Final postseal SHA, cost and signed-tail audit. No model or GT scoring."""
from common import *
import time,csv

def run():
    import numpy as np,torch
    from cg_confirm import validate_all
    p=validate_all();sp=D/'cg_confirm_sealed.json';seal=read(sp);score=read(D/'cg_confirm_score_receipt.json');assert score['seal_sha256']==sha(sp)and score['checks']==978 and score['all163sealed_before_GT'];assert seal['counts']==dict(queries=163,Source_forwards=167,new_expert=163,native_bridges=4);assert not seal['GT_read'];bind=read(D/'cg_confirm_binding.json');assert sha(D/'cg_confirm_binding.json')==seal['binding_sha256']and sha(D/'cg_B_selected.json')==bind['selected_sha256']and bind['GT_confirmation_not_read'];start=time.perf_counter();checked=0;contexts=set();segments={};loads={};teacherloads=[];cost={};means={}
    selection=read(D/'cg_selection.json');old=read(OLD/'selection.json')
    for ds,files in seal['files'].items():
        rows=read(D/'cg_confirm'/ds/'metrics.json')['private_rows'];public=read(P/'cg_confirm'/ds/'report.json');assert len(rows)==len(files)==(35 if ds=='hc2'else 128);assert{r['id']for r in rows}==set(files)=={r['id']for r in selection[ds]['confirm']};oldparents={str(r['metadata']['source'])for group in old[ds].values()for r in group};assert not oldparents.intersection(str(r['metadata']['source'])for r in selection[ds]['confirm']);assert len({r['metadata']['source']for r in selection[ds]['confirm']})==len(rows)
        for arm in p['arms']:
            for field,value in public['means'][arm].items():
                v=[r['values'][arm][field]for r in rows if r['values'][arm][field]is not None];actual=float(np.mean(v))if v else None;assert(actual is None and value is None)or abs(actual-value)<1e-12
        means[ds]=public['means'];loads[ds]=read(D/f'cg_confirm_{ds}_load.json');cc=dict(queries=0,source_forwards=0,source_capture_seconds=0.,expert_new_queries=0,pipeline_seconds=0.,expert_component_seconds={});signed=[]
        for row in rows:
            key=row['id'];e=files[key];assert sha(D/e['path'])==e['sha256']and sha(D/e['capture_path'])==e['capture_sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];checked+=3;rec=read(D/e['receipt_path']);assert all(rec[k]==e[k]for k in rec);assert rec['trial']==bind['selected'][ds]
            ap=(D/e['path']).with_suffix('.acquired.pt');aj=ap.with_suffix('.json');assert sha(ap)==e['acquired_sha256']==read(aj)['sha256'];checked+=1;aq=torch.load(ap,map_location='cpu',weights_only=False,mmap=True);cap=torch.load(D/e['capture_path'],map_location='cpu',weights_only=False,mmap=True);assert np.array_equal(aq['Source_capture']['Source'],cap['Source'])and np.array_equal(aq['Source_capture']['q0'],cap['q0'])and aq['Source_capture']['frame_ids']==cap['frame_ids'];assert not cap['GT_read'];native=aq['teacher'];ck=native['cache_key'];assert ck not in contexts;contexts.add(ck);tp=D/'cg_teacher_cache'/(ck+'.pt');meta=read(tp.with_suffix('.json'));assert meta['key']==ck and sha(tp)==meta['sha256'];assert meta['context']['dataset']==ds and meta['context']['frame_ids']==cap['frame_ids']and meta['context']['positions']==cap['positions8'];assert not native['cache_hit']and native['query_forwards']==1;checked+=1;teacherloads.append(native['model_load_seconds']);assert rec['expert_cost']==native['cost']
            with np.load(D/e['path'])as z:
                assert np.array_equal(z['Source'],cap['Source'])and z['Source_interval'].tolist()==cap['native_mbr']and z['SourceNative_interval'].tolist()==cap['native_map'];assert z['frame_ids'].tolist()==cap['frame_ids']and z['positions8'].tolist()==cap['positions8'];assert all(np.isfinite(z[a]).all()for a in p['arms'])
                if ds=='hc2':assert np.allclose(z['Source'],z['SharedCorrection'],atol=1e-14,rtol=0)
            cc['queries']+=1;cc['source_forwards']+=rec['source_forwards'];cc['source_capture_seconds']+=rec['Source_seconds'];cc['expert_new_queries']+=rec['expert_new'];cc['pipeline_seconds']+=rec['pipeline_seconds']
            for k,v in rec['expert_cost'].items():cc['expert_component_seconds'][k]=cc['expert_component_seconds'].get(k,0.)+v
            worker=f"{rec['worker']['pid']}:{rec['worker']['startticks']}";w=segments.setdefault(worker,dict(queries=0,source_forwards=0,source_capture_seconds=0.,expert_new=0,pipeline_seconds=0.));w['queries']+=1;w['source_forwards']+=rec['source_forwards'];w['source_capture_seconds']+=rec['Source_seconds'];w['expert_new']+=rec['expert_new'];w['pipeline_seconds']+=rec['pipeline_seconds']
            for baseline in ['Source','TransferredFull','Space']:
                delta={f:100*(row['values']['Full'][f]-row['values'][baseline][f])if row['values']['Full'][f]is not None and row['values'][baseline][f]is not None else None for f in ['t','v','observed_actual_sIoU','unobserved_actual_sIoU']};signed.append([f'Q{len(signed)//3+1:03}',row['duration_group'],baseline,*delta.values(),delta['t']< -5,delta['v']< -5,delta['t']< -5 or delta['v']< -5])
        folder=P/'cg_confirm'/ds
        with(folder/'all_signed_severe_timegroups.tsv').open('w')as f:
            writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(['query','timegroup','baseline','delta_t_pp','delta_v_pp','delta_observed_s_pp','delta_unobserved_s_pp','severe_t','severe_v','severe_union_once']);writer.writerows(signed)
        cc['Source_model_load_seconds_once']=loads[ds]['model_load_seconds'];cc['timegroups']=public['timegroups'];cc['severe_vs_Source']=public['severe_vs_Source'];cc['cost_scope']='Source capture time includes hooks/CPU transfer and four native reference extras, not pure GPU forward. Pipeline excludes some Source model initialization, includes decode and first expert load; expert and Source times are nested inside pipeline. Do not add nested components. No cache speed comparison.';atomic(folder/'cost_tails_timegroups.json',cc);cost[ds]=cc
    assert len(contexts)==163 and sum(x>0 for x in teacherloads)==1
    result=dict(status='verified_all163_postseal',prediction_capture_receipt_acquisition_teacher_SHA_checks=checked,queries=163,unique_same_expert8_contexts=163,Source_forwards=167,ordinary_Source_queries=163,native_reference_extra_forwards=4,teacher_model_load_seconds_once=sum(teacherloads),Source_model_load_seconds_once_per_domain={ds:load['model_load_seconds']for ds,load in loads.items()},segments=segments,stage_wall_seconds=seal['seconds'],new_model=0,new_GT_scoring=0,new_expert=0,official_checks=score,selection_sha256=sha(D/'cg_selection.json'),binding_sha256=sha(D/'cg_confirm_binding.json'),seal_sha256=sha(sp),score_sha256=sha(D/'cg_confirm_score_receipt.json'),source_sha256=sha(Path(__file__)),seconds=time.perf_counter()-start)
    atomic(D/'cg_final_reporting_receipt.json',result);atomic(P/'cg_confirm/reporting_receipt.json',result);atomic(P/'cg_confirm/score_receipt.json',score);print(result)

if __name__=='__main__':run()
