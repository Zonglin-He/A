"""Display-only off-image prediction notice; no box clipping or replacement."""
import display_v2 as previous
import inspect,argparse
from pathlib import Path
original=previous.original
D=original.D/'display_v3';P=original.P/'display_v3'

def changes():
    needle="ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_xticks([]);ax.set_yticks([])"
    notice="if box[2]<=0 or box[3]<=0 or box[0]>=w or box[1]>=h:\n                    ax.text(.98,.06,'Prediction outside image',ha='right',va='bottom',color='white',fontsize=9,transform=ax.transAxes,bbox=dict(facecolor='black',alpha=.7))\n                "+needle
    return previous.changes()+[(needle,notice)]

def prepare():
    assert not(D/'plan.json').exists();previous.validate();src=inspect.getsource(original.render);changed=src
    for old,new in changes():assert changed.count(old)==1;changed=changed.replace(old,new)
    reverse=changed
    for old,new in reversed(changes()):reverse=reverse.replace(new,old)
    assert reverse==src;compile(changed,'<display_v3>','exec')
    inputs=[previous.D/'plan.json',previous.D/'render_receipt.json',original.D/'selection.json',original.D/'display_v2_inspection.json']
    p=dict(schema='failure_display_v3',scope='Identical original cases/frame positions/arrays and v2 manual layout. Explicit text only when actual predicted xyxy is entirely outside the image, preserving original coordinates and outputs. No clipping, fallback, reselection or scoring.',
           code_sha256={str(Path(__file__).relative_to(original.R)):original.sha(Path(__file__)),str(Path(previous.__file__).relative_to(original.R)):original.sha(Path(previous.__file__))},
           input_sha256={str(f):original.sha(f)for f in inputs},reverse_display_edits_recover_original_render_exact=True)
    p['fingerprint']=original.digest(p);original.atomic(D/'plan.json',p);original.atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'});print(p['fingerprint'],flush=True)

def validate():
    previous.validate();p=original.read(D/'plan.json');assert p['fingerprint']==original.digest({k:v for k,v in p.items()if k!='fingerprint'})
    for field in ['code_sha256','input_sha256']:
        for f,h in p[field].items():assert original.sha(original.R/f)==h,f
    return p

def render():
    validate();src=inspect.getsource(original.render)
    for old,new in changes():assert src.count(old)==1;src=src.replace(old,new)
    scope={**vars(original),'D':D,'P':P,'validate':validate,'select':previous.select_existing,'__file__':__file__}
    exec(compile(src,'<failure_display_v3_off_image_notice>','exec'),scope);scope['render']()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','render']);globals()[a.parse_args().command]()
