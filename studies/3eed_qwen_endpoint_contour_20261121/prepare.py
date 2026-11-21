"""Freeze real and misregistered pixel pairs before running the model."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from endpoints import (HERE, OUT, PRIVATE, SPEC, TOL, ancestry, canvas, crop_rect, digest,
                       display_uv, displayed_distance, fit_view, misplaced, source_endpoints, write)
from geometry import camera


def prepare():
    OUT.mkdir(exist_ok=True)
    PRIVATE.mkdir(mode=0o700,exist_ok=True)
    panel, frozen, previous, _ = ancestry()
    observations={}
    counts=Counter()
    shown=set()
    for row in panel:
        uid=row["uid"]
        entry, saved=frozen[uid], previous[uid]
        r=dict(selected=entry["selected"],status="upstream_fallback",upstream=entry["status"],
               movable=False,pair_usable=False,mark_shift_px=None)
        if entry["selected"]:
            g=saved["geometry"]
            r["movable"]=saved["legend"] is not None and g["slack_m"]>TOL
            spec,size,transform=camera(row)
            cut=crop_rect(entry["pixel"],size)
            views=[fit_view([0,0,*size],0),fit_view(cut,1)]
            ends,err=source_endpoints(row,entry,g,size,spec,transform)
            r.update(views=views,endpoints=ends,max_source_replay_error_m=err)
            if all(e["status"]=="ok" for e in ends):
                uv=np.array([e["uv"] for e in ends])
                spacing=displayed_distance(uv,views)
                r["spacing_px"]=spacing
                if spacing>=10:
                    wrong=misplaced(uv,size,cut)
                    moves=[]
                    for v in views:
                        for x,y in zip(uv,wrong,strict=True):
                            a,b=display_uv(x,v),display_uv(y,v)
                            assert (a is None)==(b is None)
                            if a is not None:
                                moves.append(float(np.linalg.norm(b-a)))
                    assert abs(displayed_distance(wrong,views)-spacing)<1e-6
                    r.update(pair_usable=True,correct=uv.tolist(),wrong=wrong.tolist(),
                             mark_shift_px=max(moves),status="eligible")
                else:
                    r["status"]="markers_overlap"
            else:
                r["status"]="missing_visible_endpoint"
            if not r["movable"]:
                r["status"]="no_axial_movement"
            if r["status"]=="eligible" and row["target"] not in shown:
                # Deterministic private rendering check, picked without task labels.
                assert digest(row["rgb"])==row["rgb_sha"]
                with Image.open(row["rgb"]) as im:
                    demo=canvas(im.convert("RGB"),views,r["correct"])
                    demo.save(PRIVATE/("preview_"+row["target"]+".png"))
                shown.add(row["target"])
        observations[uid]=r
        counts[r["status"]]+=1
    path=PRIVATE/"observations.json"
    write(path,observations)
    summary=dict(frames=len(panel),statuses=dict(counts),selected=sum(z["selected"] for z in observations.values()),
                 movable=sum(z["movable"] for z in observations.values()),
                 locatable_in_rgb=sum(z["pair_usable"] for z in observations.values()),
                 effective_mismatch=sum(z["status"]=="eligible" and z["mark_shift_px"]>=10 for z in observations.values()))
    write(OUT/"input_seal.json",dict(**summary,observations=digest(path),protocol=digest(SPEC),
          code={k:digest(HERE/k) for k in ("endpoints.py","prepare.py")},
          preview_platforms=sorted(shown),target_labels_read=False,FINAL_access=False))
    print(summary,flush=True)


if __name__=="__main__":
    prepare()
