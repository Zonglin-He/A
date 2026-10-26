"""Manual figure layout; identical frozen cases and actual prediction arrays."""
import render as original
import inspect,argparse
from pathlib import Path
D=original.D/'display_v2';P=original.P/'display_v2'

def changes():
    return [
        ("fig=plt.figure(figsize=(24,15),layout='constrained')", "fig=plt.figure(figsize=(24,15))"),
        ("grid=fig.add_gridspec(7,6,width_ratios=[1.9,3,3,3,3,3],height_ratios=[1,.34,2.2,2.2,2.2,2.2,1.35])",
         "grid=fig.add_gridspec(7,6,width_ratios=[2.05,3,3,3,3,3],height_ratios=[1,.34,2.2,2.2,2.2,2.2,1.35],left=.025,right=.995,bottom=.10,top=.97,hspace=.10,wspace=.07)"),
        ("la.text(.98,.55,'Event intervals\\n+ same eight prompts',ha='right',va='center',fontsize=11)", "la.text(.95,.97,'Event intervals',ha='right',va='top',fontsize=10)"),
        ("(a['Source_interval'],'Native Source','#167fc1')", "(a['Source_interval'],'Source','#167fc1')"),
        ("(a['SourceROI_interval'],'Source-box appearance','#d88813')", "(a['SourceROI_interval'],'Source ROI','#d88813')"),
        ("'Current event: interpolation / independent / Ours'if case['kind']!='pooling_ROI'else'Corrected-box appearance (Ours)'", "'Current'if case['kind']!='pooling_ROI'else'Corrected ROI'"),
        ("(gt['interval'],'Annotated GT event','#25ad43')", "(gt['interval'],'GT','#25ad43')"),
        ("fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=11)", "fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.55,.012),ncol=3,fontsize=11)"),
    ]

def prepare():
    assert not(D/'plan.json').exists();original.validate()
    src=inspect.getsource(original.render);changed=src
    for old,new in changes():
        assert changed.count(old)==1,old;changed=changed.replace(old,new)
    reverse=changed
    for old,new in reversed(changes()):reverse=reverse.replace(new,old)
    assert reverse==src
    inputs=[original.D/'plan.json',original.D/'selection.json',original.D/'render_receipt.json',original.D/'initial_layout_inspection.json']
    p=dict(schema='failure_display_v2',scope='Only fixed manual layout, concise timeline labels and legend position. Cases/frame positions/predicted boxes/intervals/scalars unchanged; selection read from frozen original selection.json without reselection.',
           code_sha256={str(Path(__file__).relative_to(original.R)):original.sha(Path(__file__))},
           input_sha256={str(f):original.sha(f)for f in inputs},reverse_style_edits_recover_original_render_source_exact=True,
           original_render_source_sha256=original.hashlib.sha256(src.encode()).hexdigest())
    p['fingerprint']=original.digest(p);original.atomic(D/'plan.json',p)
    original.atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'})
    print(p['fingerprint'],flush=True)

def validate():
    original.validate();p=original.read(D/'plan.json');assert p['fingerprint']==original.digest({k:v for k,v in p.items()if k!='fingerprint'})
    for field in ['code_sha256','input_sha256']:
        for f,h in p[field].items():assert original.sha(original.R/f)==h,f
    return p

def select_existing():
    cases=original.read(original.D/'selection.json')
    rows={ds:{r['id']:r for r in original.records(ds)}for ds in original.COUNTS}
    truth={ds:original.selected.labels(ds)for ds in original.COUNTS}
    return cases,rows,truth

def render():
    validate();src=inspect.getsource(original.render)
    for old,new in changes():assert src.count(old)==1;src=src.replace(old,new)
    scope={**vars(original),'D':D,'P':P,'validate':validate,'select':select_existing,'__file__':__file__}
    exec(compile(src,'<failure_display_v2_manual_layout>','exec'),scope);scope['render']()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','render']);globals()[a.parse_args().command]()
