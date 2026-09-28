"""Write-once CPU audit of sealed merger gradients and fixed residuals."""
import argparse, json, os, sys, time, shutil, traceback, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from vg_tta.desta3d_v3_direction_alignment import pair_stats, sign_gate

OUT=ROOT/'artifacts/desta3d_v3/latent_oracle_v1'
DEST=OUT/'direction_alignment_v1'
CASES=[(0,7,'event'),(1,3,'spatial')]
RUNS=['actuation_full001','actuation_span001','location001','large_mask002','context001']
KEYS=['key','source','frame_ids','positions','boxes_cxcywh','geometry_valid',
      'interval','format_ok','event_completion','spatial_completion','preprocess',
      'adapter_sha','readout','time_distribution']


def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def variants(branch):
    return [('location001',f'{branch}_{loc}_{kind}') for loc in ['late','early'] for kind in ['correct','wrong']]+[
        ('large_mask002',f'{branch}_large_{kind}') for kind in ['correct','wrong']]+[
        ('context001',f'{branch}_context_{kind}') for kind in ['correct','wrong']]


def register():
    assert not DEST.exists(),'Write-once audit; never replace prior result'
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v3_direction_alignment.py',
           ROOT/'tests/test_desta3d_v3_direction_alignment.py',
           ROOT/'protocols/desta3d_v3_direction_alignment_v1.md']
    sealed={}
    for run in RUNS:
        d=OUT/run;done=read(d/'COMPLETE.json');manifest=read(d/'PREDICTIONS_SEAL.json')
        assert done['seal_sha']==sha(d/'PREDICTIONS_SEAL.json')
        paths.extend([d/'COMPLETE.json',d/'PREDICTIONS_SEAL.json',d/'CONFIG.json',d/'INPUTS.json'])
        sealed[run]=manifest['files']
    consumed=[]
    for index,panel,branch in CASES:
        for run in ['actuation_full001','actuation_span001']:
            names=['INPUT.json','NATIVE_00.pt','STEP_01.pt','FORWARD_00.pt','FINAL.pt','COMPLETE.json']
            if run=='actuation_span001':names+=['BASIS.pt']
            for n in names:consumed.append((run,f'episodes/{index:02}/{n}'))
            paths.append(OUT/run/'ROOT_DECISION.json')
        for run in ['location001','large_mask002','context001']:
            consumed.extend((run,f'episodes/{panel:02}/{n}') for n in ['INPUT.json','original.pt'])
            paths.append(OUT/run/'independent_readback_v1/REPORT.json')
        for run,arm in variants(branch):
            consumed.append((run,f'episodes/{panel:02}/{arm}_DELTA.pt'))
    for run,rel in consumed:
        p=OUT/run/rel
        assert sha(p)==sealed[run][rel],f'Original seal mismatch: {p}'
        paths.append(p)
    unique=sorted(set(paths));pins={str(p.relative_to(ROOT)):sha(p) for p in unique}
    write(DEST/'REGISTRATION.json',dict(status='registered_before_numeric_audit',time=time.time(),
        cases=[dict(index=i,panel_index=p,branch=b) for i,p,b in CASES],
        primary_gradient='stored unclipped STEP_01 full merger gradient at B1; not span parameter gradient',
        variants_per_case=8,fixed_optimized_step=30,new_GPU_seconds=0,new_model_execution=False,
        new_GT_pool=False,new_target=False,prior_GPU_seconds=42493.15652965409,cumulative_cap=None,
        CPU_threads=4,CPU_phase_seconds=900,maximum_new_bytes=32*2**20,minimum_free_bytes=8*2**30,
        branch_gate='span descent positive AND late-correct/context-correct negative outside 1e-9 arithmetic band; both branches required',
        consumed_sealed_files=len(consumed)))
    write(DEST/'LOCK.json',dict(pins=pins))
    print('REGISTERED',len(pins),'pinned files',len(consumed),'sealed raw inputs')


def run():
    start=time.monotonic();torch.set_num_threads(4)
    assert not (DEST/'STARTED.json').exists()
    pins=read(DEST/'LOCK.json')['pins']
    for p,h in pins.items():assert sha(ROOT/p)==h,p
    assert shutil.disk_usage(ROOT).free>=8*2**30
    write(DEST/'STARTED.json',dict(time=time.time(),pid=os.getpid(),device='cpu'))
    discrepancies=[]
    def load(p):return torch.load(p,map_location='cpu',weights_only=False)
    def stats(a,b):
        a,b=a.detach().cpu(),b.detach().cpu()
        r=pair_stats(a.numpy(),b.numpy())
        # Separate vectorized PyTorch reduction, not the chunked NumPy helper.
        x=a.double().flatten();y=b.double().flatten()
        nx,ny=float(x.norm()),float(y.norm());dot=float(torch.dot(x,y))
        alt=dict(x_norm=nx,y_norm=ny,dot=dot,cosine=dot/(nx*ny) if nx and ny else None)
        for k,v in r.items():
            if v is None:assert alt[k] is None
            else:
                err=abs(v-alt[k]);discrepancies.append(dict(field=k,absolute_error=err))
                assert np.isclose(v,alt[k],atol=1e-9,rtol=1e-10),(k,v,alt[k])
        return r
    try:
        result=[]
        for index,panel,branch in CASES:
            assert time.monotonic()-start<900
            full=OUT/'actuation_full001/episodes'/f'{index:02}'
            span=OUT/'actuation_span001/episodes'/f'{index:02}'
            inp=read(full/'INPUT.json');assert inp['support']==read(span/'INPUT.json')['support']
            base=load(full/'NATIVE_00.pt');sb=load(span/'NATIVE_00.pt')
            assert all(equal(base[k],sb[k]) for k in KEYS)
            for source in ['location001','large_mask002','context001']:
                ep=OUT/source/'episodes'/f'{panel:02}'
                assert read(ep/'INPUT.json')['support']==inp['support']
                other=load(ep/'original.pt')
                assert all(equal(base[k],other[k]) for k in KEYS)
                assert read(OUT/source/'CONFIG.json')['checkpoint_sha']==read(OUT/'actuation_span001/CONFIG.json')['checkpoint_sha']
            sf,ss=load(full/'STEP_01.pt'),load(span/'STEP_01.pt')
            g=sf['gradient'];gs=ss['gradient'];basis=load(span/'BASIS.pt');q=basis['basis'];scale=basis['scale']
            assert g.shape[:-1]==gs.shape[:-1] and g.shape[-1]==2560 and gs.shape[-1]==128
            assert all(torch.isfinite(v).all() for v in [g,gs,q])
            ff,fs=load(full/'FORWARD_00.pt'),load(span/'FORWARD_00.pt')
            assert ff['exact'] and fs['exact'] and equal(ff,fs)
            assert torch.equal(sf['target'],ss['target']) and torch.equal(sf['valid'],ss['valid'])
            assert torch.equal(ff['targets'],sf['target']) and torch.equal(ff['valid'],sf['valid'])
            assert sf['loss']==ss['loss']
            classes=ff['native'].shape[-1];assert classes==(len(base['frame_ids']) if branch=='event' else 152775)
            ce=float(torch.nn.functional.cross_entropy(ff['native'][ff['valid']].float(),ff['targets'][ff['valid']]))
            assert abs(ce-sf['loss'])<2e-6
            del ff,fs
            opt=load(span/'FINAL.pt');free=load(full/'FINAL.pt')
            star=opt['token_delta'];assert star.shape==g.shape and free['token_delta'].shape==g.shape
            assert read(span/'COMPLETE.json')['actual_steps']==30
            projected=(g.double()@q.double())*scale
            chain=stats(projected,gs.double());chain['relative_L2_error']=float((projected-gs.double()).norm()/gs.double().norm())
            chain['max_abs_error']=float((projected-gs.double()).abs().max())
            chain['basis_orthogonality_max_error']=float((q.double().T@q.double()-torch.eye(128,dtype=torch.float64)).abs().max())
            reconstructed=(opt['parameter'].double()@q.double().T)*scale
            reconstruction=dict(max_abs=float((reconstructed-star.double()).abs().max()),
                                relative_L2=float((reconstructed-star.double()).norm()/star.double().norm()))
            del reconstructed,projected
            star_g=stats(-g,star);star_norm=star_g['y_norm']
            directions={}
            all_dirs=[('span_step30',star),('free_step30',free['token_delta'])]
            for name,delta in all_dirs:
                descent=stats(-g,delta);align=stats(star,delta)
                directions[name]=dict(**descent,cosine_to_span=align['cosine'],
                    descent_at_span_norm=descent['dot']*star_norm/descent['y_norm'],new_native=False)
            native_reports={r:read(OUT/r/'independent_readback_v1/REPORT.json') for r in ['location001','large_mask002','context001']}
            base_metrics=read(OUT/'actuation_span001/ROOT_DECISION.json')['cases'][index]['baseline']['metrics']
            final_metrics=read(OUT/'actuation_span001/ROOT_DECISION.json')['cases'][index]['final']['metrics']
            for source,arm in variants(branch):
                raw=load(OUT/source/'episodes'/f'{panel:02}'/(arm+'_DELTA.pt'))
                delta=raw['matched'];assert raw['branch']==branch and delta.shape==g.shape
                descent=stats(-g,delta);align=stats(star,delta)
                met=native_reports[source]['arms'][arm]['rows'][panel]
                assert met['key']==base['key']
                directions[arm]=dict(**descent,cosine_to_span=align['cosine'],
                    descent_at_span_norm=descent['dot']*star_norm/descent['y_norm'] if descent['y_norm'] else None,
                    original_norm_record=raw['norm'],native_metrics=met['metrics'],
                    native_effect_pp={m:100*(met['metrics'][m]-base_metrics[m]) for m in ['tIoU','sIoU','vIoU']},
                    new_native=False)
            gate=sign_gate(star_g['dot'],directions[f'{branch}_late_correct']['dot'],directions[f'{branch}_context_correct']['dot'])
            result.append(dict(alias=f'P{panel:02}',key=base['key'],branch=branch,shape=list(g.shape),
                gradient_norm=star_g['x_norm'],gradient_dtype=str(g.dtype),gradient_clipping_was_triggered=sf['clip_triggered'],
                gradient_used='saved BEFORE clipping',baseline_native_equal_all_five_runs=True,
                support_sha=inp['support'],native_classes=classes,valid_action_positions=int(sf['valid'].sum()),
                initial_CE=ce,span_coordinate_chain=chain,span_final_reconstruction=reconstruction,
                baseline_metrics=base_metrics,span_final_metrics=final_metrics,directions=directions,
                strong_sign_diagnosis=gate))
            print(branch,'span cosine',directions['span_step30']['cosine'],'late',directions[f'{branch}_late_correct']['cosine'],
                  'context',directions[f'{branch}_context_correct']['cosine'],'gate',gate,flush=True)
        report=dict(status='completed_independently_reduced',cases=result,
                    both_branches_strong_sign_diagnosis=all(c['strong_sign_diagnosis'] for c in result),
                    numeric_checks=len(discrepancies),maximum_absolute_discrepancy=max(x['absolute_error'] for x in discrepancies),
                    new_GPU_seconds=0,new_forward=0,new_backward=0,new_optimizer=0,new_native=0,new_target=False,
                    limitations=['Two previously outcome-selected, source-training exposed cases; no population inference.',
                      'g is the initial B1 native CE gradient, not a gradient at the optimized endpoint.',
                      '30-step endpoint was reached along changing native states; initial Taylor dot is not its finite utility.',
                      'BF16 autograd is not an exact derivative of the discrete numerical program.',
                      'Low endpoint cosine alone does not establish adverse gradient direction or a root cause.',
                      'No claim of actual-reader learnability, teacher qualification or OPD success.'])
        write(DEST/'REPORT.json',report)
        write(DEST/'ROOT_NUMERIC_CROSSCHECK.json',dict(status='passed',implementation='NumPy float64 chunked dot and independent PyTorch float64 vector reductions',
              checks=len(discrepancies),max_absolute_error=report['maximum_absolute_discrepancy'],atol=1e-9,rtol=1e-10))
        lines=['# Saved direction alignment audit','',
               'CPU-only readback of two preselected source cases. No new model execution or predictions.','',
               '|Case|Direction|cos(delta, span30)|cos(-g0, delta)|-g0 dot delta|matched-span-norm dot|',
               '|---|---|---:|---:|---:|---:|']
        for c in result:
            for name,v in c['directions'].items():
                lines.append(f"|{c['alias']} {c['branch']}|{name}|{v['cosine_to_span']:.8f}|{v['cosine']:.8f}|{v['dot']:.9g}|{v['descent_at_span_norm']:.9g}|")
        lines+=['','Positive last two columns predict local CE descent; they are not measured CE/native improvements.',
                '',f"Both-branch strong sign gate: {report['both_branches_strong_sign_diagnosis']}.",'']+report['limitations']
        (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
        elapsed=time.monotonic()-start
        assert elapsed<900 and shutil.disk_usage(ROOT).free>=8*2**30
        assert sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())<32*2**20
        write(DEST/'COMPLETE.json',dict(status='completed',CPU_seconds=elapsed,new_GPU_seconds=0,
             cumulative_GPU_seconds=42493.15652965409,cap=None,report_sha=sha(DEST/'REPORT.json'),
             pinned_inputs=len(pins),original_seals_checked=True,free_bytes=shutil.disk_usage(ROOT).free))
    except BaseException as e:
        write(DEST/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),CPU_seconds=time.monotonic()-start,new_GPU_seconds=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);globals()[p.parse_args().action]()
