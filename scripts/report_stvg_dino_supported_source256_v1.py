"""Postaudit anonymous report figures and fixed private pixel panels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import collections,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put as immutable_put,load,sha
from scripts.run_stvg_dino_supported_source256_v1 import D,P,PRE,OLD,verify


def put(path, value):
    # Presentation-only reruns may reuse identical JSON; never overwrite a
    # changed scientific result or receipt.
    if path.exists():
        assert read(path)==value, str(path)
    else:
        immutable_put(path,value)


def main():
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    from scripts.run_decota_paper_main_v1 import frames_for
    from scripts.c1_controlled_corruption_v1 import pixelhash
    from methods.decota_final_simplified_v1.config import MethodConfig,EXPERT_SHA256
    p=verify();rs=read(P/'RESULTS.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    actual=read(P/'ACTUAL_COSTS.json');root=D/'root_review';root.mkdir(exist_ok=True)
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    exposure=read(D/'EXPOSURE_SENSITIVITY_LOCK.json');excluded=set(exposure['excluded_aliases'])
    clean=[r for r in rs['rows']if r['alias']not in excluded];sensitivity={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in clean if r['dataset']==ds]
        sensitivity[ds]=dict(parents=len(rr),queries=len(rr),
            metrics={a:{ep:float(np.mean([r['values'][a][ep]for r in rr if r['values'][a][ep]is not None]))for ep in ['v','t','s','unobserved_actual_s']}for a in p['arms']},
            E_minus={c:{ep:paired([None if r['values']['E_DINO_support'][ep]is None or r['values'][c][ep]is None else 100*(r['values']['E_DINO_support'][ep]-r['values'][c][ep])for r in rr])for ep in ['v','unobserved_actual_s']}for c in p['arms'][:2]})
    put(P/'EXPOSURE_SENSITIVITY.json',dict(status='complete',main256_unchanged=True,
        exclusion_locked_before_target_score=True,excluded_calibration_parent_count=25,
        prior_parent_overlap=read(D/'CALIBRATION_PARENT_OVERLAP_READBACK.json')['counts'],
        exact_query_input_overlap=read(D/'CALIBRATION_EXACT_QUERY_READBACK.json')['counts'],
        retained=sensitivity,selection_uses_results=False,not_independent_confirmation=True))
    fig,axs=plt.subplots(2,2,figsize=(11.5,7.5));fig.subplots_adjust(left=.09,right=.97,bottom=.15,top=.86,wspace=.28,hspace=.38)
    for i,ds in enumerate(['hc2','vidstg']):
        rr=[r for r in rs['rows']if r['dataset']==ds]
        for j,ep in enumerate(['v','unobserved_actual_s']):
            ax=axs[i,j]
            for k,arm in enumerate(['A_Top1_mean','E_DINO_support']):
                v=np.array([100*(r['values'][arm][ep]-r['values']['Source_no_correction'][ep])for r in rr if r['values'][arm][ep]is not None and r['values']['Source_no_correction'][ep]is not None]);ix=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)));lo,hi=np.quantile(v[ix].mean(1),[.025,.975]);m=v.mean()
                ax.errorbar(k,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=5,color=['#ba7739','#267a75'][k],ms=8)
            ax.axhline(0,color='.45',lw=.8);ax.set_xlim(-.6,1.6);ax.set_xticks([0,1],['A: Top1 mean','E: DINO support']);ax.set_ylabel('Change vs frozen Source (pp)');ax.set_title(f"{['HC','VidSTG'][i]} · {['full vIoU','unobserved actual-frame sIoU'][j]}",loc='left',fontsize=10);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen Source-start correction · 128 parents per target',fontsize=14)
    fig.text(.09,.035,'Fixed roster, beta = 1/16, original four DINO frames, identical final-MBR.\nPaired parent bootstrap 95% intervals; historically exposed expanded evaluation. No Student updates.',fontsize=9)
    fig.savefig(P/'EXPANDED_SOURCE.svg');fig.savefig(root/'EXPANDED_SOURCE.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4));fig.subplots_adjust(bottom=.22,wspace=.25)
    for ax,ds in zip(axs,['hc2','vidstg']):
        rr=[r for r in rs['rows']if r['dataset']==ds];x=[100*(r['values']['A_Top1_mean']['v']-r['values']['Source_no_correction']['v'])for r in rr];y=[100*(r['values']['E_DINO_support']['v']-r['values']['Source_no_correction']['v'])for r in rr]
        ax.scatter(x,y,s=18,alpha=.7,color='#267a75');lo=min(x+y+[0]);hi=max(x+y+[0]);ax.plot([lo,hi],[lo,hi],color='.5',lw=.8);ax.axvline(0,color='.75',lw=.7);ax.axhline(0,color='.75',lw=.7);ax.set_title(ds);ax.set_xlabel('A − Source full vIoU (pp)');ax.set_ylabel('E − Source full vIoU (pp)')
    fig.text(.08,.035,'Every one-query parent is shown. Diagonal: equal A/E gain. No parent is selected by score.',fontsize=9);fig.savefig(P/'ALL_PARENTS.svg');fig.savefig(root/'ALL_PARENTS.png',dpi=150);plt.close(fig)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']};views=[]
    for ds in ['hc2','vidstg']:
        r=next(r for r in p['records']if r['dataset']==ds);q=r['query_ordinal'];z=load(D/'predictions'/(r['alias']+'.pt'));row=plans[ds][q];frames,ids,_=frames_for(ds,row,collections.OrderedDict());assert ids==z['frame_ids']
        di='vidstg_to_hc2'if ds=='hc2'else'hcstvg2_to_vidstg';inp=__import__('torch').load(OLD/'inputs'/di/'clean'/f'{q:05}.pt',map_location='cpu',weights_only=False);assert pixelhash(frames)==inp['pixel_sha256']
        slots=z['planned_positions']or[0];panels=[]
        for slot,pos in enumerate(slots):
            im=Image.fromarray(frames[pos]).convert('RGB');w,h=im.size;draw=ImageDraw.Draw(im)
            for arm,color in [('Source_no_correction','#2687de'),('A_Top1_mean','#e19a32'),('E_DINO_support','#22c283')]:
                b=z['outputs'][arm][pos];xy=((b[0]-b[2]/2)*w,(b[1]-b[3]/2)*h,(b[0]+b[2]/2)*w,(b[1]+b[3]/2)*h);draw.rectangle(xy,outline=color,width=3)
            thumb=im.resize((480,max(1,round(h*480/w))));panel=Image.new('RGB',(480,thumb.height+45),'white');panel.paste(thumb,(0,45));d=ImageDraw.Draw(panel);d.text((8,5),f'{r["alias"]} slot{slot} frame{ids[pos]}',fill='black');d.text((8,24),'Blue Source | Orange A | Green E | no GT overlay',fill='black');panels.append(panel)
        height=max(im.height for im in panels);sheet=Image.new('RGB',(480*len(panels),height),'#eeeeee')
        for k,im in enumerate(panels):sheet.paste(im,(480*k,0))
        dest=root/(ds+'_fixed_input_output.png');sheet.save(dest)
        views.append(dict(alias=r['alias'],dataset=ds,planned_slots=len(z['planned_positions']),shown_frames=len(slots),pixel_sha256=inp['pixel_sha256'],panel_path=str(dest.relative_to(ROOT)),panel_sha256=sha(dest),no_new_identity_semantic_labels=True))
    put(D/'PIXEL_PANEL_GENERATION.json',dict(status='generated_pending_actual_root_view',count=sum(v['shown_frames']for v in views),views=views,original_input_pixels_exact=True,private_media_not_exported=True))
    configs={ds:MethodConfig.for_direction('vid_to_hc1'if ds=='hc2'else'hc2_to_vid')for ds in ['hc2','vidstg']}
    public=dict(status='complete',count=256,baseline='frozen Source, no inherited online state',arms=p['arms'],beta=1/16,
        backbone='TA-STVG',source_checkpoints={ds:dict(source=c.source_dataset,checkpoint_SHA256=c.checkpoint_sha256)for ds,c in configs.items()},DINO_SHA256=EXPERT_SHA256,
        Source_DINO_input_identity='All256 original inputs hash checked; Source geometry/Native interval/frames/pixels exact on reused captures, geometry/Native exact on new captures, fixed original pixel views exact',
        actual_costs=actual,source_calibration=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')['source_domains'],
        old_shared_beta_calibration_sha256=sha(ROOT/'artifacts/stvg_dino_supported_correction32_v1/BETA_CALIBRATION.json'),
        E_core_sha256=sha(ROOT/'vg_tta/stvg_dino_supported_correction32_v1.py'),same_private_roster_sha256=p['same_roster_sha256'],
        selection=read(PRE/'ROSTER_LOCK.json')['selection'],seed=20261018,excluded_tuning_parents_per_domain=32,
        original_E32_parent_overlap=read(PRE/'COHORT_EXPOSURE_READBACK.json')['overlap_with_original_E32_parents'],
        prior_source_calibration_parent_overlap=read(D/'CALIBRATION_PARENT_OVERLAP_READBACK.json')['counts'],
        prior_source_calibration_exact_query_overlap=read(D/'CALIBRATION_EXACT_QUERY_READBACK.json')['counts'],
        historically_exposed=True,not_independent_confirmation=True,GT_only_after_all256_output_seal=True,zero_Student_updates=True,zero_new_DINO_calls=True,
        old_E32_negative_gate_preserved=True,no_domain_specific_routing_or_method_promotion=True,
        old_Source_E_compatibility=read(D/'FROZEN_E_SOURCE_COMPATIBILITY_READBACK.json'),
        publication_receipt_metadata_recovery=read(D/'publication/recovery/receipt_scope002/RECOVERY.json'))
    put(P/'ACTUAL_PROTOCOL.json',public)
    put(P/'VERIFICATION.json',dict(status='pass',count=256,independent=read(P/'ROOT_INDEPENDENT_AUDIT.json'),pixel_panels_generated_pending_actual_view=True,output_barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),raw_media_GT_coordinates_and_text_excluded_from_export=True))
    (P/'README.md').write_text('# Frozen Source-start E expanded256\n\n[Report](REPORT.md) · [all signed results and strata](RESULTS.json) · [actual protocol/cost](ACTUAL_PROTOCOL.json) · [independent audit](ROOT_INDEPENDENT_AUDIT.json).\n\nSame fixed HC128/Vid128 parent roster; zero Student update or new DINO call. This explicitly changes the baseline from the cancelled inherited Before experiment to frozen Source. It does not establish online OPD incremental benefit. Historical E32 failed gate remains.\n\nPrivate dataset/cache receipts are required and not distributed. Reproduction: run_stvg_dino_supported_source256_v1.py prepare; continue_stvg_dino_supported_source256_v1.py (serial capture, predict, score, independent finalize); report_stvg_dino_supported_source256_v1.py; actual fixed-pixel/privacy review; publish_stvg_dino_supported_source256_v2.py. Use v2 for publication: v1 is preserved only because it is pinned by the original runtime, and has unused inherited receipt metadata. v2 fixes only receipt count/scope/Source-call accounting; no scientific file or prediction changes.\n')
    text=(P/'REPORT.md').read_text().split('\n\n## Historical exposure sensitivity')[0];text+='\n\n## Historical exposure sensitivity\n\nSource_no_correction means frozen Source boxes with the SAME final-MBR interval used by A/E; it is not the Native-decoder benchmark Source Only row. Unobserved actual-frame sIoU uses all known GT positions on the actual sampled decoder grid except the four planned expert slots, independent of the predicted MBR support.\n\nMetadata-only checking before new target scoring found HC18 source-train and7 source-validation calibration parent overlaps (25 total); Vid0. They are different video files, query text and sampled inputs, not exact-query reuse. Main256 is unchanged. A pre-score lock additionally reports HC103/Vid128 excluding those parents; this is exposure sensitivity, not a replacement main cohort, parameter-selection set or independent confirmation.\n\n| Exposure-excluded target | Parents | E−Source full-v pp | E−A full-v pp | E−Source unobserved s pp | E−A unobserved s pp |\n|---|---:|---:|---:|---:|---:|\n'
    for ds,g in sensitivity.items():
        vv=[g['E_minus'][c][ep]['mean']for ep,c in [('v','Source_no_correction'),('v','A_Top1_mean'),('unobserved_actual_s','Source_no_correction'),('unobserved_actual_s','A_Top1_mean')]]
        text+='| '+ds+' | '+str(g['parents'])+' | '+' | '.join(f'{v:+.6f}'for v in vv)+' |\n'
    text+='\n[All signed results](RESULTS.json) · [exposure sensitivity](EXPOSURE_SENSITIVITY.json) · [actual protocol and costs](ACTUAL_PROTOCOL.json) · [independent audit](ROOT_INDEPENDENT_AUDIT.json) · [paired intervals](EXPANDED_SOURCE.svg) · [all parents](ALL_PARENTS.svg).\n';(P/'REPORT.md').write_text(text)
    files=['scripts/run_stvg_dino_supported_expanded256_recovery002.py','scripts/continue_stvg_dino_supported_expanded256_recovery002.py','protocols/stvg_dino_supported_expanded256_recovery002.md','scripts/run_stvg_dino_supported_source256_v1.py','scripts/continue_stvg_dino_supported_source256_v1.py','scripts/finalize_stvg_dino_supported_source256_v1.py','scripts/report_stvg_dino_supported_source256_v1.py','scripts/publish_stvg_dino_supported_source256_v1.py','scripts/publish_stvg_dino_supported_source256_v2.py','protocols/stvg_dino_supported_source256_v1.md']
    files += [str((P/f).relative_to(ROOT))for f in ['RESULTS.json','REPORT.md','README.md','ROOT_INDEPENDENT_AUDIT.json','ACTUAL_COSTS.json','ACTUAL_PROTOCOL.json','VERIFICATION.json','EXPOSURE_SENSITIVITY.json','EXPANDED_SOURCE.svg','ALL_PARENTS.svg']]
    put(D/'PUBLIC_FILES.json',dict(files=files));print(json.dumps(dict(status='report_and_private_panels_generated',files=len(files),actual_root_pixel_view_pending=True)))


if __name__=='__main__':main()
