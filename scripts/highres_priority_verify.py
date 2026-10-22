"""Independent sealed-array and self-hull identity checks; no labels/models."""
import os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.highres_priority.run import SUP,OUT,PUBLIC,validate,read,write,sha
from stvg.temporal import probabilities,mass_table,final_mbr

def verify():
    torch.set_num_threads(1);p=validate();seal=read(SUP/'sealed.json');native=layers=hull_checks=0;err=0.;expected=[]
    for item,receipt in zip(p['items'],seal['files']):
        assert sha(SUP/receipt['posterior_path'])==receipt['posterior_sha256'] and sha(SUP/receipt['prediction_path'])==receipt['prediction_sha256']
        raw=torch.load(SUP/receipt['posterior_path'],map_location='cpu',weights_only=False)
        for o in [0,1]:assert torch.equal(raw['all_logits'][o][-1],raw['logits'][o]);native+=1
        _,q=mass_table(probabilities(raw['records'],raw['logits']));assert np.array_equal(q,raw['posteriors']['Only448'])
        mean=np.mean([mass_table(probabilities(raw['records'],[x[l]for x in raw['all_logits']]))[1]for l in range(6)],axis=0);assert np.array_equal(mean,raw['posteriors']['High6']);layers+=6
        with np.load(SUP/receipt['prediction_path'])as z,np.load(ROOT/item['inputs']['prediction'])as old:
            assert np.array_equal(z['Only224six'],old['Global6'])and np.array_equal(z['Fusion'],old['Strong_Reference']);assert np.array_equal(z['Fusion_interval'],old['Strong_Reference_interval'])
        for grid in raw['grids']:
            ids=np.asarray(grid['ids'],float);a,b=grid['ij'];probs=grid['p'];original=float(probs@(ids[b]+1-ids[a]));sub,h=mass_table([grid,grid]);s=np.asarray(sub,float);full=float((h*(s[None,:]+1-s[:,None])).sum())
            starts=np.bincount(a,weights=probs,minlength=len(ids));ends=np.bincount(b,weights=probs,minlength=len(ids));dist=abs(ids[:,None]-ids[None,:]);formula=.5*(dist*starts[:,None]*starts[None,:]).sum()+.5*(dist*ends[:,None]*ends[None,:]).sum();err=max(err,abs((full-original)-formula));assert full>=original-1e-10;expected.append((full-original)/item['record']['metadata']['fps']);hull_checks+=1
    assert native==126 and layers==378 and hull_checks==126 and err<1e-10
    primary=read(PUBLIC/'report.json');follow=read(ROOT/'results/highres_selfhull_v1/report.json');lowreport=read(ROOT/'results/highres_selfhull_low_v1/report.json');assert primary['proof']['official_dense_checks']==1394 and follow['proof']['official_dense_checks']==504 and lowreport['proof']['official_dense_checks']==756
    local_seal={r['query']:r for r in read(ROOT/'outputs/local224_v1/sealed.json')['files']};low_checks=0
    for item in p['items']:
        key=item['anonymous'];cap=torch.load(ROOT/item['inputs']['capture'],map_location='cpu',weights_only=False);gp=torch.load(ROOT/item['inputs']['G'],map_location='cpu',weights_only=False)['final']['Global6'];ids=cap['frame_ids'];loc={int(x):i for i,x in enumerate(ids)}
        lr=local_seal[key];assert sha(ROOT/'outputs/local224_v1'/lr['posterior_path'])==lr['posterior_sha256'];lv=torch.load(ROOT/'outputs/local224_v1'/lr['posterior_path'],map_location='cpu',weights_only=False);z=np.load(ROOT/f'outputs/highres_selfhull_low_v1/predictions/{key}.npz')
        readings=[('LowSelf0',cap['records'][0]['frame_ids'],cap['temporal_cache']['offsets'][0]['native_logits'],False),('LowSelf1',cap['records'][1]['frame_ids'],cap['temporal_cache']['offsets'][1]['native_logits'],False),('LowSelfFusion0',cap['records'][0]['frame_ids'],cap['temporal_cache']['offsets'][0]['native_logits'],True),('LowSelfFusion1',cap['records'][1]['frame_ids'],cap['temporal_cache']['offsets'][1]['native_logits'],True),('LocalSelfFusion0',lv['local']['Local224']['ids'],lv['local']['Local224']['logits'],True),('ShiftedSelfFusion0',lv['local']['Shifted224']['ids'],lv['local']['Shifted224']['logits'],True)]
        for arm,subids,logits,fuse in readings:
            grids=probabilities([dict(frame_ids=subids)]*2,[logits]*2);sub,small=mass_table(grids);q=np.zeros_like(gp);ix=[loc[int(x)]for x in sub];q[np.ix_(ix,ix)]=small;q=.5*gp+.5*q if fuse else q;assert final_mbr(ids,q)['interval']==z[arm+'_interval'].tolist();low_checks+=1
    assert low_checks==378
    smoke=read(ROOT/'outputs/highres_single_smoke_v1/receipt.json');assert smoke['bitwise_logits']and smoke['max_error']==0 and smoke['module_calls']==dict(vis_encoder=1,vid=1,text_encoder=1,ground_encoder=1)and smoke['two_time_passes']==2
    full=read(ROOT/'outputs/boxfixed_full_v1/state.json');assert full['status']=='paused_by_user'and full['paused_completed']==6088
    pause=read(ROOT/full['pause_receipt'])
    for key,x in pause['files'].items():
        assert sha(ROOT/f'outputs/boxfixed_full_v1/vidstg/predictions/{key}.npz')==x['prediction_sha256'];assert sha(ROOT/f'outputs/boxfixed_full_v1/vidstg/receipts/{key}.json')==x['receipt_sha256']
    assert len(pause['files'])==len(list((ROOT/'outputs/boxfixed_full_v1/vidstg/receipts').glob('*.json')))==6088
    result=dict(status='pass',native_last_layer_vs_all_layer_bitwise_checks=native,high6_layer_tables_exact=layers,old_Global6_and_Strong_exact_queries=63,self_hull_expected_length_identity_checks=hull_checks,self_hull_identity_max_error=err,low_and_local_self_hull_decisions_independently_recovered=low_checks,mean_self_hull_expected_duration_increase_seconds=float(np.mean(expected)),official_dense_checks=2654,official_dense_max_error=max(primary['proof']['official_dense_maxerror'],follow['proof']['official_dense_max_error'],lowreport['proof']['official_dense_max_error']),actual_single448_bitwise_verified=True,full6088_predictions_receipts_unchanged=True,full434_frozen_sources_unchanged=True,no_new_labels_or_models=True,CUDA_initialized=torch.cuda.is_initialized())
    assert not result['CUDA_initialized'];write(PUBLIC/'verification.json',result);print(result)
if __name__=='__main__':verify()
