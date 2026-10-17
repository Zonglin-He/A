"""Independent saved-scalar audit, anonymous native-head report and actual panel."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_head_adapter32_v1 import BASE,read,write,sha
OUT=ROOT/'results/stvg_native_heads/2026-10-17/matched32'

def audit(d,rr):
    import numpy as np
    from scipy.stats import t
    idx={(r['case_id'],r['arm']):r['metrics']for r in d['rows']};checks=0
    for s in d['summary']:
        v=[r['metrics'][s['metric']]for r in d['rows']if r['dataset']==s['dataset']and r['arm']==s['arm']and r['metrics'][s['metric']]is not None]
        assert s['valid']==len(v) and s['unknown']==16-len(v)
        assert s['mean']==(math.fsum(v)/len(v)if v else None);checks+=len(v)
    for p in d['paired']:
        a,b=p['comparison'].split('_minus_');v={r['probe_alias']:100*(idx[r['probe_alias'],a][p['metric']]-idx[r['probe_alias'],b][p['metric']])for r in rr if r['dataset']==p['dataset']and idx[r['probe_alias'],a][p['metric']]is not None and idx[r['probe_alias'],b][p['metric']]is not None};n=len(v);mean=math.fsum(v.values())/n if n else None
        assert p['per_source_delta_pp']==v and p['mean_delta_pp']==mean and p['valid']==n and p['unknown']==16-n
        ci=None
        if n>1:e=float(t.ppf(.975,n-1)*np.std(list(v.values()),ddof=1)/math.sqrt(n));ci=[mean-e,mean+e]
        assert ci==p['paired_source_t95_interval_pp'] or (ci is not None and np.allclose(ci,p['paired_source_t95_interval_pp'],atol=1e-11,rtol=0))
        assert [sum(x>1e-6 for x in v.values()),sum(x< -1e-6 for x in v.values()),sum(abs(x)<=1e-6 for x in v.values())]==[p[k]for k in ['positive','negative','neutral']];checks+=n
    return checks

def report():
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    rr=read(BASE/'ROSTER_LOCK.json')['records'];d=read(BASE/'POSTSEAL_RESULTS.json');assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json');checks=audit(d,rr)
    by={(x['case_id'],x['arm']):x['metrics']for x in d['rows']}
    for r in rr:
        cid=r['probe_alias'];assert by[cid,'ORIGINAL']==by[cid,'TEMPORAL_HEAD'];assert by[cid,'SPATIAL_HEAD']==by[cid,'BOTH_HEADS'];assert by[cid,'B3_TEMPORAL']==by[cid,'NEW_TEMPORAL']
    a=dict(status='pass',count=32,checks=checks,Temporal_factor_exact_identity=True,full_output_and_parameters_qualification_sha256=sha(BASE/'QUALIFICATION_COMPLETION.json'),result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'));write(BASE/'ROOT_SCALAR_READBACK.json',a)
    aliases={r['probe_alias']:f'R{i:03d}'for i,r in enumerate(rr,1)}
    def anon(v):
        if isinstance(v,dict):return {aliases.get(k,k):anon(x)for k,x in v.items()if k!='GT_bindings'}
        if isinstance(v,list):return [anon(x)for x in v]
        return aliases.get(v,v)if isinstance(v,str)else v
    OUT.mkdir(parents=True,exist_ok=False);write(OUT/'RESULTS.json',anon(d));write(OUT/'ROOT_SCALAR_READBACK.json',a);write(OUT/'QUALIFICATION_COMPLETION.json',anon(read(BASE/'QUALIFICATION_COMPLETION.json')))
    write(OUT/'SOURCE_BINDINGS.json',dict(count=32,records=[dict(case_id=aliases[r['probe_alias']],dataset=r['dataset'],parent_sha256=r['parent_sha256'],source_checkpoint_sha256=r['source_checkpoint_sha256'])for r in rr]))
    write(OUT/'METHOD_LOCK.json',read(BASE/'METHOD_LOCK.json'));write(OUT/'COST.json',dict(new_Teacher_calls=0,new_Temporal_fits=0,new_Spatial_fits=sum(x['new_fit']for x in d['usage']),native_Temporal_parameters=512,old_Spatial_parameters=1792,native_Spatial_parameters=sorted({x['Spatial_new_parameters']for x in d['usage']}),stages=[dict(stage=p.parent.name,**read(p))for p in (BASE/'execution').glob('*/WORKER_EXIT.json')],reused_cost_not_zero=True))
    def val(ds,arm,k):return next(s['mean']for s in d['summary']if s['dataset']==ds and s['arm']==arm and s['metric']==k)
    lines=['# Fixed-evidence native head interface32','','Same exposed HC16/Vid16, cross-domain TA-STVG checkpoints; frozen R1 Direct references and B3 I1/I2/rewards unchanged. No new Teacher calls or candidate generation. No independent confirmation.','','Temporal adapter at native temp_embed input is exactly the existing256x2 residual on this backbone. This axis is an explicitly verified identity, not a distinct model experiment; all fitted weights and full-grid outputs reused. Spatial adds a zero-initialized shared bias-free256x4 residual at the actual final bbox Linear input in the last layer of the final native decoder pass. Original Query/LN parameters frozen. Original Gaussian32, domain settings, Adam/response/fullstep acceptance/rollback/early stop unchanged.','','Four cells are controlled fixed-evidence readouts: original interfaces, represented Temporal only, native Spatial only, both native heads. Spatial observations retain the sealed T2 clock for every cell; combinations use the same fixed B3 time. They are not newly generated B3-clock end-to-end Full. Separate fixed-T2-clock spatial rows isolate box-interface effects. Source marginal no-update is kept. No decoder-only increment is counted as adaptation.','','|Domain|Arm|vIoU %|dense sIoU %|tIoU %|','|---|---|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        for arm in ['Frozen','MARGINAL_NO_UPDATE','B3_TEMPORAL','ORIGINAL','TEMPORAL_HEAD','SPATIAL_HEAD','BOTH_HEADS','ORIGINAL_S_FIXED_CLOCK','NEW_S_FIXED_CLOCK']:
            lines.append('|'+ds+'|'+arm+'|'+'|'.join('unknown'if(v:=val(ds,arm,k))is None else f'{100*v:.6f}'for k in ['m_vIoU','m_sIoU_dense_GT','m_tIoU'])+'|')
    lines+=['','All source metrics, anchor/nonanchor unknowns, all signed paired deltas and nominal t95 intervals remain in RESULTS.json. No mandatory significance gate. Actual gradient/finite-difference, zero recovery, complete Source/updated native outputs and reversible nonanchor influence checked on the fixed first parent/domain; these are engineering checks, not semantic labels. The semantic512 panel is still44 model-assisted/468pending. CURRENT/PAPER unchanged.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');(OUT/'PROTOCOL.md').write_bytes((ROOT/'protocols/stvg_native_head_adapter32_v1.md').read_bytes())
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));aa=['ORIGINAL_S_FIXED_CLOCK','NEW_S_FIXED_CLOCK']
    for ax,ds in zip(axes,['hc2','vidstg']):ax.bar(['Query+LN','Native box head'],[100*val(ds,x,'m_vIoU')for x in aa]);ax.set_title(ds+' fixed T2 clock / all16');ax.set_ylabel('vIoU (%)')
    fig.tight_layout();fig.savefig(OUT/'effects.png',dpi=160);plt.close(fig)
    for rel in ['vg_tta/stvg_native_head_residual32_v1.py','scripts/stvg_native_head_adapter32_v1.py','scripts/report_stvg_native_head_adapter32_v1.py','scripts/continue_stvg_native_heads_qwen32_v1.py']:
        p=OUT/'code'/Path(rel).name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/rel).read_bytes()))
    paths={str(p.relative_to(ROOT))for p in OUT.rglob('*')if p.is_file()};verify_privacy(paths);write(BASE/'publication/matched32/EXPORT_MANIFEST.json',dict(status='complete',count=32,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]))
    render();print('HEAD_REPORT_READY',checks,flush=True)

def render():
    import numpy as np,torch
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    rr=read(BASE/'ROSTER_LOCK.json')['records'];jj=read(BASE/'INPUT_JOB_LOCK.json')['records'];bm={f['case_id']:f for f in read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['files']};fig,axes=plt.subplots(2,2,figsize=(10,7))
    for row,ds in enumerate(['hc2','vidstg']):
        rec=next(r for r in rr if r['dataset']==ds);cid=rec['probe_alias'];j=next(j for j in jj if j['case_id']==cid);assert sha(ROOT/j['image']['path'])==j['image']['sha256'];rgb=np.load(ROOT/j['image']['path'],allow_pickle=False);H,W=rgb.shape[:2];d=torch.load(ROOT/bm[cid]['path'],map_location='cpu',weights_only=False)
        for col,arm in enumerate(['ORIGINAL','SPATIAL_HEAD']):
            ax=axes[row,col];ax.imshow(rgb);x,y,u,v=map(float,d['arms'][arm]['boxes'][j['position']]);ax.add_patch(Rectangle((x*W,y*H),(u-x)*W,(v-y)*H,fill=False,color='cyan'));ax.set_title(ds+' first source / '+arm);ax.axis('off')
    fig.tight_layout();fig.savefig(BASE/'private_actual_output_panel.png',dpi=130);plt.close(fig)

if __name__=='__main__':report()
