"""Explicit configuration revision; execute original core in an isolated namespace."""
import copy,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
from scripts.decota_opd_tuning_common_v1 import verify_paper,bridge
BASE=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
OLD=ROOT/'artifacts/stvg_opd_paper_v1'
TUNING=ROOT/'artifacts/stvg_opd_hc2_coordinate_v1'
PUB=ROOT/'results/stvg_opd_paper_hc2_revision/2026-10-08'
PYTHON=ROOT/'.conda/tubedetr/bin/python'

def verify():
    verify_paper();r=read(BASE/'RUNTIME_LOCK.json')
    assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256']
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    original=read(OLD/'RUNTIME_LOCK.json')
    for f,h in original['pins'].items():
        if f=='methods/decota_spatial_opd_v1/configs.json':
            assert sha(ROOT/f)==r['selected_config_file_sha256']
        else:assert sha(ROOT/f)==h,f
    assert sha(OLD/'RUNTIME_LOCK.json')==r['original_GPU_runtime_sha256']
    assert sha(TUNING/'ROOT_CLOSING_RECEIPT.json')==r['tuning_root_receipt_sha256']
    return r

def activate():
    verify()
    import scripts.stvg_opd_paper_common_v1 as c
    c.BASE=BASE;c.PUB=PUB;c.verify=verify
    return c

def prepare():
    if (BASE/'RUNTIME_LOCK.json').exists():return verify()
    closing=read(TUNING/'ROOT_CLOSING_RECEIPT.json');assert closing['status']=='complete'
    selected=read(TUNING/'SELECTION_BARRIER.json')['config'];configs=read(ROOT/'methods/decota_spatial_opd_v1/configs.json')
    assert configs['datasets']['hc2']['config']==selected
    original=read(OLD/'DESIGN_LOCK.json');d=copy.deepcopy(original)
    assert configs['datasets']['vidstg']==read(TUNING/'PREVIOUS_CONFIGS.json')['datasets']['vidstg']
    changed=selected!=d['datasets']['hc2']['config'];d['datasets']['hc2']['config']=selected
    d.update(version='stvg_opd_paper_hc2_revision_v2',retuning=True,fixed_after_sequential_coordinate_lock=True,
        HC2_changed=changed,original_design_sha256=sha(OLD/'DESIGN_LOCK.json'),
        HC2_coordinate_selection_sha256=sha(TUNING/'SELECTION_BARRIER.json'),
        original_P0_negatives_preserved=True,confirmation_is_previously_exposed=True,
        selected_configuration_not_spliced_into_old_P1=True,time=time.time())
    write(BASE/'DESIGN_LOCK.json',d)
    # Exact already closed baseline rows are joined by the unchanged assembler.
    shutil.copy2(OLD/'P1_BASELINE_BINDING.json',BASE/'P1_BASELINE_BINDING.json')
    files=['scripts/stvg_opd_paper_hc2_revision_common_v2.py','scripts/run_stvg_opd_paper_hc2_revision_v2.py',
        'scripts/continue_stvg_opd_paper_hc2_revision_v2.py','scripts/audit_stvg_opd_revision_aliases_v2.py',
        'protocols/stvg_opd_paper_hc2_revision_v2.md','scripts/run_stvg_opd_paper_v1.py',
        'scripts/score_stvg_opd_paper_v1.py','scripts/finalize_stvg_opd_p0_v1.py',
        'scripts/score_stvg_opd_p1_v1.py','scripts/assemble_stvg_opd_table1_v1.py','scripts/finalize_stvg_opd_table1_v1.py']
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in files},design_sha256=sha(BASE/'DESIGN_LOCK.json'),
        selected_config_file_sha256=sha(ROOT/'methods/decota_spatial_opd_v1/configs.json'),
        original_GPU_runtime_sha256=sha(OLD/'RUNTIME_LOCK.json'),
        tuning_root_receipt_sha256=sha(TUNING/'ROOT_CLOSING_RECEIPT.json'),
        only_production_config_exception='HC2 selected values; original config file saved unchanged in tuning/PREVIOUS_CONFIGS.json',time=time.time()))
    return verify()

def link(path,relative):
    dest=BASE/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(dest)==sha(path)
    else:os.link(path,dest)

def reuse_stage(name):
    new=read(BASE/'DESIGN_LOCK.json');old=read(OLD/'DESIGN_LOCK.json');ds=new['stages'][name]['dataset']
    assert new['stages'][name]==old['stages'][name] and new['datasets'][ds]['config']==old['datasets'][ds]['config']
    p=OLD/'stages'/name/'PREDICTION_BARRIER.json';b=read(p)
    for f,h in {**b['files'],**b['inputs']}.items():
        source=OLD/f;rc=read(source.with_suffix('.json'));assert sha(source)==h==rc['sha256'] and not rc['GT_read']
        if f in b['files']:assert source.stat().st_size==rc['bytes']
        assert rc['time']<=b['time'];link(source,f);link(source.with_suffix('.json'),str(Path(f).with_suffix('.json')))
    bb={**b,'exact_stage_reused':True,'original_barrier_path':str(p.relative_to(ROOT)),
        'original_barrier_sha256':sha(p),'original_seal_time':b['time'],
        'original_runtime_lock_sha256':b['runtime_lock_sha256'],'runtime_lock_sha256':sha(BASE/'RUNTIME_LOCK.json'),'time':time.time()}
    write(BASE/'stages'/name/'PREDICTION_BARRIER.json',bb)
    if (OLD/'stages'/name/'CPU_COMPLETION.json').exists():
        oldpub=ROOT/'results/stvg_opd_paper/2026-10-08'/name;out=PUB/name;out.mkdir(parents=True,exist_ok=True)
        completion=read(OLD/'stages'/name/'CPU_COMPLETION.json')
        for f,h in completion['files'].items():assert sha(ROOT/f)==h
        for file in oldpub.iterdir():
            if file.is_file():shutil.copy2(file,out/file.name)
        write(BASE/'stages'/name/'CPU_COMPLETION.json',dict(status='exact_closed_stage_scalar_evidence_reused',
            original_CPU_receipt_sha256=sha(OLD/'stages'/name/'CPU_COMPLETION.json'),
            files={str(f.relative_to(ROOT)):sha(f) for f in out.iterdir() if f.is_file()},time=time.time()))
    status(BASE/'stages'/name/'STATUS.json',dict(status='sealed_exact_unchanged_stage_reused',new_GPU_fits=0,logical_arrivals=b['adapted_arrivals'],time=time.time()))

def reuse_unchanged_P1_prefix():
    d=read(BASE/'DESIGN_LOCK.json')
    if d['HC2_changed']:return
    saved=OLD/'user_hc2_coordinate_pause_20261008/EXACT_RESUME_STATE.json';x=read(saved)
    assert x['config']==d['datasets']['hc2']['config'] and x['done']==685
    proof=read(saved.parent/'OPAQUE_PREFIX_READBACK.json')
    for f,rc in proof['files'].items():
        p=ROOT/f;assert sha(p)==rc['sha256'] and p.stat().st_size==rc['bytes']
        z=__import__('scripts.decota_matrix_common_v1',fromlist=['load']).load(p)
        inp=OLD/z['input']['path'];assert sha(inp)==z['input']['sha256']
        link(p,str(p.relative_to(OLD)));link(p.with_suffix('.json'),str(p.with_suffix('.json').relative_to(OLD)))
        link(inp,str(inp.relative_to(OLD)));link(inp.with_suffix('.json'),str(inp.with_suffix('.json').relative_to(OLD)))
    write(BASE/'P1_EXACT_PREFIX_REUSE.json',dict(status='pass',arrivals=685,source_receipt_sha256=sha(saved),configuration_identical=True,time=time.time()))
