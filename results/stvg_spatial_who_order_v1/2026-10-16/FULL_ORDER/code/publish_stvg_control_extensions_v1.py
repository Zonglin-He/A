"""Explicit per-group public manifest; no experimental actions."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.report_stvg_control_extensions_v1 import SCOPES
from scripts import publish_stvg_soft_ep_v1 as pub,publish_stvg_lite_batch001 as upload
if __name__=='__main__':
    kind,group,mode=sys.argv[1:4];base=ROOT/'artifacts'/SCOPES[kind];b=base/'publication'/group
    if mode=='close':
        assert sys.argv[4]=='--after-actual-pixel-and-ref-view'
        from scripts import close_stvg_referent_abc_publication_v1 as close
        close.BASE=base;close.main(group,sys.argv[5]);sys.exit(0)
    paths=set(json.loads((b/'EXPORT_MANIFEST.json').read_text())['paths']);pub.BASE=b;pub.expected_paths=lambda:paths;upload.BASE=b;upload.paths=lambda:paths
    if mode=='stage':pub.stage(pub.DEFAULT_CHECKOUT,None,False)
    elif mode=='refresh':
        upload.refresh();p=b/'PUBLIC_STAGE.json';d=json.loads(p.read_text());actual=set(pub.git(pub.DEFAULT_CHECKOUT,'diff','--cached','--name-only').splitlines());d.update(staged_paths=sorted(actual),staged_file_count=len(actual))
        for r in d['files']:r['staged']=r['path']in actual
        p.write_text(json.dumps(d,indent=2)+'\n');pub.verify_stage(pub.DEFAULT_CHECKOUT);(b/'build_tree_plan.py').write_bytes((ROOT/'artifacts/stvg_lite_event_support_v1/publication/batch001/build_tree_plan.py').read_bytes())
    else:pub.verify_stage(pub.DEFAULT_CHECKOUT)
