"""Only explicitly reviewed stage exports; no whole-repository staging."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import publish_stvg_soft_ep_v1 as publisher
from scripts import publish_stvg_lite_batch001 as upload
BASE=ROOT/'artifacts/stvg_referent_evidence_p0_v1'
if __name__=='__main__':
    group,mode=sys.argv[1:3];assert group in ('source_control001','response001','closing001')
    b=BASE/'publication'/group;manifest=json.loads((b/'EXPORT_MANIFEST.json').read_text());paths=set(manifest['paths'])
    publisher.BASE=b;publisher.expected_paths=lambda:paths;upload.BASE=b;upload.paths=lambda:paths
    if mode=='stage':publisher.stage(publisher.DEFAULT_CHECKOUT,None,False)
    elif mode=='refresh':
        upload.refresh();p=b/'PUBLIC_STAGE.json';d=json.loads(p.read_text());actual=set(publisher.git(publisher.DEFAULT_CHECKOUT,'diff','--cached','--name-only').splitlines());d['staged_paths']=sorted(actual);d['staged_file_count']=len(actual)
        for r in d['files']:r['staged']=r['path']in actual
        p.write_text(json.dumps(d,indent=2)+'\n');publisher.verify_stage(publisher.DEFAULT_CHECKOUT)
        src=ROOT/'artifacts/stvg_lite_event_support_v1/publication/batch001/build_tree_plan.py';(b/'build_tree_plan.py').write_bytes(src.read_bytes())
    elif mode=='verify':publisher.verify_stage(publisher.DEFAULT_CHECKOUT)
    else:raise ValueError(mode)
