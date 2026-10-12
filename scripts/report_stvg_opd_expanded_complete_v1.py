"""Seal exact anonymous cached C scalars; no new fits, inference or GT scoring."""
import collections,csv,gzip,io,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.stvg_opd_expanded_state_common_v1 import verify,BASE,PUB
from scripts.finalize_stvg_opd_design_ablation_v1 import statistics
from scripts.audit_stvg_opd_expanded_complete_public_v1 import REL,ARM_FIELDS,run as audit
OUT=ROOT/REL
def packed(path,value,jsonl=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:
            if jsonl:
                for row in value:f.write((json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode())
            else:f.write(json.dumps(value,separators=(',',':'),allow_nan=False).encode())
def run():
    verify();root=read(BASE/'ROOT_COMPLETE_READBACK.json');scoring=read(BASE/'CPU_SCORING_COMPLETION.json')
    assert root['status']=='pass' and root['new_formal_fits']==5430 and root['parents']==905
    assert sha(PUB/'DIRECT_CONTRASTS.json')==root['results_sha256']
    OUT.mkdir(parents=True,exist_ok=False);rr=[]
    for name,count in [('E_hc2_cross_clean',1230),('E_vidstg_cross_clean',4200)]:
        p=PUB/name/'ROWS.jsonl.gz';assert sha(p)==scoring['outputs'][name]['rows_sha256']
        with gzip.open(p,'rt') as f:part=[json.loads(x) for x in f]
        assert len(part)==count;rr.extend(part)
    identity=sorted({(r['dataset'],r['source_id']) for r in rr});assert len(identity)==905
    ids={key:i for i,key in enumerate(identity)}
    public=[dict({k:v for k,v in r.items() if k not in {'source_id','query_ordinal'}},parent_id=ids[r['dataset'],r['source_id']]) for r in rr]
    packed(OUT/'ROWS.jsonl.gz',public,True)
    def anonymous(value,ds):
        if isinstance(value,dict):
            return {('parent_id' if k=='source_id' else k):(ids[ds,v] if k=='source_id' else anonymous(v,ds)) for k,v in value.items()}
        if isinstance(value,list):return [anonymous(v,ds) for v in value]
        return value
    direct=read(PUB/'DIRECT_CONTRASTS.json');raw_statistics=direct['statistics']
    direct['statistics']={n:anonymous(s,'hc2' if n=='E_hc2_cross_clean' else 'vidstg') for n,s in raw_statistics.items()}
    packed(OUT/'DIRECT_CONTRASTS.json.gz',direct)
    result={}
    for name in raw_statistics:
        part=[r for r in rr if r['stage']==name];ds=part[0]['dataset'];result[name]={}
        for arm in ['full','alpha0','alpha1']:
            chosen=[r for r in part if r['arm']==arm];result[name][arm]={}
            for order in ['pooled','order1','order2']:
                group=chosen if order=='pooled' else [r for r in chosen if r['order']==order]
                result[name][arm][order]=anonymous(statistics(group,ARM_FIELDS),ds)
    packed(OUT/'ALL_ARM_STATISTICS.json.gz',dict(status='complete_postseal_cached_scalar_diagnostics',statistics=result,
        bootstrap=10000,seed=20261011,unadjusted_CI=True,current_update_endpoint='official vIoU After minus Before; parent average of two fixed orders',
        inherited_state_endpoint_is_not_Full_minus_reset=True,new_GT_scoring_calls=0,new_model_calls=0))
    parent_table=[]
    cells={(r['dataset'],r['arm'],r['order'],r['parent_id']):r for r in public}
    for ds,pid in [(d,ids[d,s]) for d,s in identity]:
        x=dict(dataset=ds,parent_id=pid)
        for order in ['order1','order2']:
            full=cells[ds,'full',order,pid]
            for f in ['After_v','Before_v','Frozen_v']:x[order+'_'+f]=full[f]
            for a in ['alpha0','alpha1']:x[order+'_Full_minus_'+a+'_v']=full['After_v']-cells[ds,a,order,pid]['After_v']
        for f in ['After_v','Before_v','Frozen_v','Full_minus_alpha0_v','Full_minus_alpha1_v']:
            x['pooled_'+f]=(x['order1_'+f]+x['order2_'+f])/2
        parent_table.append(x)
    string=io.StringIO();writer=csv.DictWriter(string,fieldnames=list(parent_table[0]));writer.writeheader();writer.writerows(parent_table)
    with (OUT/'ALL_PARENT_CONTRASTS.csv.gz').open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:f.write(string.getvalue().encode())
    write(OUT/'DESIGN_PUBLIC.json',dict(status='complete_locked_historical_nondevelopment_expanded_state',
        targets=['hc2','vidstg'],hc2_parents=205,vidstg_parents=700,parents=905,orders=['order1','order2'],
        arms=['full','alpha0','alpha1'],new_formal_fits=5430,qualification_arrivals=24,qualification_repeat_fits=48,
        active_query_parameters=256,active_LN_parameters=1536,query_and_Adam_reset_per_query=True,
        original_Full_writeback=dict(hc2=1/16,vidstg=1/8),alpha0_writeback=0,alpha1_writeback=1,
        original_science_source_and_runtime_unchanged=True,same_original_source_pixel_expert_frame_binding=True,
        same_query_key_sampling_randomness=True,bootstrap=10000,seed=20261011,unadjusted_CI=True,
        historically_evaluated_not_fresh=True,GT_used_to_select_cohort=False,GT_only_after_all_formal_global_seal=True,
        new_DINO_calls=0,new_GT_scoring_calls_in_this_export=0,raw_GT_pixels_model_states_excluded=True,
        production_promoted=False,new_gate_created=False,new_state_intervention_launched=False,
        deployment_timing128_complete=False,LN768_complete=False,cost32_complete=False,manuscript_written=False,
        cached_fit_compute_is_not_end_to_end_deployment_timing=True,
        observation_delta_definition='legacy C scorer DenseTubeGT timeline; unavailable remains null; not P9-P0 common-U genuine-frame endpoint'))
    a=audit(ROOT);write(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json',dict(a,source_root_sha256=sha(BASE/'ROOT_COMPLETE_READBACK.json'),source_scoring_sha256=sha(BASE/'CPU_SCORING_COMPLETION.json'),time=time.time()))
    write(OUT/'PUBLIC_SCALAR_AUDIT.json',a)
    lines=['# Expanded-State: complete historical non-development result','','All 5,430 new formal fits were sealed before GT exposure: 205 HC-STVG-v2 parents and 700 VidSTG parents, three fixed retention rules, two fixed online orders. No new detector calls. This is the complete Expanded-State phase; deployment timing, LN-only and audited cost remain separate authorized work.','',
        '## Direct final-output comparisons','','Parent averages retain both online orders. Intervals use the original paired 10,000-parent bootstrap (seed 20261011) without multiplicity adjustment. Values are percentage points.','',
        '| Target | Order | Paired contrast | Difference (pp) | 95% CI | Negative parents |','|---|---|---|---:|---|---:|']
    for x in direct['table']:lines.append(f"| {x['dataset']} | {x['order']} | {x['contrast']} | {x['mean_pp']:+.3f} | [{x['ci95_pp'][0]:+.3f}, {x['ci95_pp'][1]:+.3f}] | {x['negative_parents']} / {x['parent_count']} |")
    lines+=['','## Absolute outcomes and current-update negative tails','','All entries average within parent across the two orders. A negative current-update parent means its mean official vIoU After-minus-Before is below zero; this counts arbitrarily small negative differences as well as substantial harms. The >5 pp and >20 pp columns distinguish the negative tail.','',
        '| Target | Rule | Frozen vIoU (%) | Before vIoU (%) | After vIoU (%) | Current update (pp) | Negative parents | >5 pp harm | >20 pp harm |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,arm_result in result.items():
        ds='hc2' if n=='E_hc2_cross_clean' else 'vidstg'
        for arm,by_order in arm_result.items():
            m=by_order['pooled']['metrics'];x=m['delta_current_v']
            lines.append(f"| {ds} | {arm} | {m['Frozen_v']['mean']*100:.3f} | {m['Before_v']['mean']*100:.3f} | {m['After_v']['mean']*100:.3f} | {x['mean']*100:+.3f} | {x['negative_parent_count']} / {by_order['pooled']['sources']} | {x['harm_gt5pp_sources']} | {x['harm_gt20pp_sources']} |")
    lines+=['','Full-minus-reset pooled intervals cross zero in both domains. Full-minus-full-carry is positive in VidSTG for both orders and the pooled endpoint; HC intervals cross zero. The measured effect therefore supports attenuation versus unattenuated carry in VidSTG, while a stable advantage over reset across both domains is not established.','',
        'Before-minus-Frozen describes the state entering the current query; it is not the direct Full-minus-reset contrast. The three rules intervene on an entire online retention regime. They do not isolate the causal effect of one LN state under a single fixed-query state swap, nor validate an unlabeled update selector. P9-P0 natural order/state associations and these regime comparisons must retain these distinct scopes.','',
        'The cohort excludes the predeclared development parents but has historical evaluation exposure. It is not a new untouched test set. All 5,430 anonymous scalar rows, all 905 paired parent outcomes, pooled and order-specific absolute/direct statistics, neutral values, unavailable observations and negative tails are exported. No GT-driven cohort or configuration selection was performed.','',
        'The optional observed/unobserved_iou_delta columns are the original C DenseTubeGT-timeline diagnostic; unavailable entries remain null. They do not use P9-P0\'s common-U genuine-frame definition. Cached capture/fit compute fields describe this experiment and do not substitute for the pending complete deployment timing or teacher cost measurements.','',
        'The original scientific implementation, runtime pins, accepted fits, qualification and global barrier remain unchanged. No gate, new method, causal state-intervention GPU experiment, production promotion or manuscript was created.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='pass',rows=len(public),parents=len(identity),scalar_comparisons=a['scalar_comparisons'],new_GT_scoring_calls=0)))
if __name__=='__main__':run()
