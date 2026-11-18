"""One episodic step on the checkpoint's native final horizontal-center parameters."""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from predict157 import HERE, ROOT, OUT, UPSTREAM, DESIGN, digest, common_component, basis, ray_first, surface_residual


def load_heads():
    registry = json.loads((ROOT/"research/round97/experiment/DESIGN_LOCK.json").read_text())
    heads = {}
    for entry in registry["sources"]:
        p = Path(entry["checkpoint_path"])
        with p.open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == entry["checkpoint_sha256"]
        state = torch.load(p, weights_only=True, map_location="cpu", mmap=True)
        assert state["epoch"] == 100 and state["trained_platforms"] == [entry["source"]]
        prefix = "prediction_heads.5.center_residual_head.net."
        weights = {k[len(prefix):]: v for k,v in state["model"].items() if k.startswith(prefix)}
        modules = []
        for _ in range(2):
            modules.extend((nn.Conv1d(288, 288, 1, bias=False), nn.BatchNorm1d(288),
                            nn.ReLU(), nn.Dropout(0.3)))
        head = nn.Sequential(*modules, nn.Conv1d(288, 3, 1))
        head.load_state_dict(weights, strict=True)
        heads[entry["source"]] = head.eval().requires_grad_(False)
    return heads


def geometry_cost(xy, b, points, origins, weights):
    """Same truncation, weights and ray misses as the selector; differentiate only XY."""
    center = torch.cat((xy, torch.as_tensor(b[2:3], dtype=torch.float64)))
    rot = torch.eye(3,dtype=torch.float64)
    rot[:2,:2] = torch.tensor(basis(b[6]))
    p = torch.tensor(points, dtype=torch.float64)
    half = torch.tensor(b[3:6]/2, dtype=torch.float64)
    cap = max(.2,min(b[3:5])/2)
    if origins is None:
        q = abs((p-center)@rot)-half
        cost = torch.linalg.vector_norm(q.clamp_min(0),dim=-1)-q.max(dim=-1).values.clamp_max(0)
    else:
        o = torch.tensor(origins, dtype=torch.float64)
        ray = p-o
        dist = torch.linalg.vector_norm(ray,dim=-1)
        unit = (ray / dist[:,None]) @ rot
        eye = (o-center)@rot
        parallel = abs(unit)<1e-12
        divisor = torch.where(parallel,torch.ones_like(unit),unit)
        t0,t1 = (-half-eye)/divisor,(half-eye)/divisor
        enter = torch.minimum(t0,t1).masked_fill(parallel,-torch.inf).amax(dim=-1)
        leave = torch.maximum(t0,t1).masked_fill(parallel,torch.inf).amin(dim=-1)
        hit = ~(parallel & (abs(eye)>half)).any(dim=-1) & (enter<=leave) & (enter>=0)
        cost = torch.where(hit,(dist-enter).abs(),torch.full_like(dist,cap))
    return torch.tensor(weights, dtype=torch.float64) @ cost.clamp_max(cap)


def main():
    from heldout157 import iter_heldout
    driver = json.loads((HERE/"experiment/TTA_RELEASE.json").read_text())["driver"]
    assert driver in ("ray","combined")
    assert digest(Path(__file__)) == json.loads((HERE/"experiment/TTA_CODE_LOCK.json").read_text())["files"]["tta157.py"]
    seal = json.loads((OUT/"PREDICTION_SEAL.json").read_text())
    dest, receipt = OUT/"TTA_PRIVATE.npz", OUT/"TTA_SEAL.json"
    assert not dest.exists() and not receipt.exists()
    assert seal["prediction_sha256"] == digest(OUT/"PREDICTIONS_PRIVATE.npz")
    registry_path = ROOT/"research/round145/INPUT_REGISTRY_PRIVATE.json"
    assert digest(registry_path) == "6df8cbc22aff3b8c945c6a50c6a23ec5c3b559d46f200618cfb91e83a220ad85"
    entries=json.loads(registry_path.read_text())
    with np.load(OUT/"PREDICTIONS_PRIVATE.npz", allow_pickle=False) as f:
        a,h = f["A"].astype(float),f["H"].astype(float)
        take, sensor, ordinal=f["choose_"+driver],f["origin_ok"],f["ordinal"]
        ray_h, surface_h=f["ray_H"],f["surface_H"]
    with np.load(UPSTREAM/"PAIR_PRIVATE.npz", allow_pickle=False) as f:
        winner=f["query"][:,0]
    directory=ROOT/"crossplatform/round131/formal"
    pins={d.name:json.loads((d/"SEALED.json").read_text())["files"] for d in directory.iterdir() if d.is_dir()}
    boxes=np.repeat(a[:,None,:],3,axis=1)
    step, param, gradnorm = np.zeros((3,len(a)))
    heads=load_heads()
    max_bridge=max_cost=max_functional=0.
    torch.set_num_threads(1)
    count=0
    for i,(points,origin,meta) in enumerate(iter_heldout()):
        assert meta["ordinal"]==i+1==ordinal[i]
        count+=1
        if not take[i]:
            continue
        e=entries[i]
        q=int(winner[i])
        src,tgt=e["source"],e["target"]
        direction=src+"_to_"+tgt
        rel="prefix_PRIVATE/"+ e["row"]["id"]+".npz"
        path=directory/direction/rel
        assert digest(path)==pins[direction][rel]
        head=heads[src]
        with np.load(path,allow_pickle=False) as prefix, torch.no_grad():
            hidden=torch.tensor(prefix["h"]).T[None].contiguous()
            feat=head[:8](hidden)
            anchor=torch.tensor(prefix["query_anchor_xyz"])
            raw=(anchor+head[8](feat)[0].T).numpy()
            max_bridge=max(max_bridge,float(abs(raw-prefix["boxes"][:,:3]).max()))
            assert np.array_equal(prefix["boxes"][q,:3], a[i,:3]) and max_bridge<5e-5
        ids,w,*_=common_component(points,a[i],h[i])[0]
        p=points[ids].astype(float)
        o=None
        if sensor[i]:
            assert origin is not None
            o=np.asarray(origin,dtype=float)
            if o.ndim==2: o=o[ids]
        check=geometry_cost(torch.tensor(h[i,:2]),h[i],p,o,w)
        max_cost=max(max_cost,abs(check.item()-(surface_h[i] if o is None else ray_h[i])))
        assert max_cost<1e-8
        # Anchor numerical replay to the original checkpoint's exact cached output.
        # The offset is constant and accounts only for CPU versus capture precision.
        f=feat[0,:,q].double()
        # Edit an episodic copy of the actual Conv1d, then invoke it again.
        final=copy.deepcopy(head[8]).double().requires_grad_(True)
        start_w=final.weight.detach().clone()
        start_b=final.bias.detach().clone()
        activ=f.reshape(1, -1, 1)
        native=final(activ)[0, :2, 0]
        base=torch.tensor(a[i,:2])-native.detach()
        xy=base+native
        loss=geometry_cost(xy,h[i],p,o,w)
        g,gw,gb=torch.autograd.grad(loss,(xy,final.weight,final.bias))
        assert not gw[2].any() and not gb[2].any()
        gradnorm[i]=float(g.norm())
        boxes[i,:,6]=h[i,6]
        displacement=h[i,:2]-a[i,:2]
        boxes[i,2,:2]+=displacement/2
        step_radius=np.linalg.norm(displacement)/2
        if gradnorm[i]>1e-12 and step_radius>0:
            eta=step_radius/(gradnorm[i]*(1+f.square().sum().item()))
            with torch.no_grad():
                final.weight[:2].add_(gw[:2],alpha=-eta)
                final.bias[:2].add_(gb[:2],alpha=-eta)
                boxes[i,0,:2]=(base+final(activ)[0,:2,0]).numpy()
            boxes[i,1,:2]-=step_radius*g.detach().numpy()/gradnorm[i]
        param[i]=((final.weight-start_w).square().sum()+(final.bias-start_b).square().sum()).sqrt().item()
        step[i]=np.linalg.norm(boxes[i,0,:2]-a[i,:2])
        max_functional=max(max_functional,float(abs(boxes[i,0]-boxes[i,1]).max()))
    assert count==len(a)==1798 and max_functional<1e-9
    tmp=dest.with_suffix(".tmp")
    with tmp.open("xb") as f:
        np.savez_compressed(f,boxes=boxes,ordinal=ordinal,arms=["one_step","box_step","interpolate"],
                            accepted=take,moved=step>1e-12,step_m=step,param_l2=param,gradient_norm=gradnorm)
    os.replace(tmp,dest)
    info=dict(status="PASS",n=count,driver=driver,accepted=int(take.sum()),nonzero_steps=int((step>1e-12).sum()),
              design_sha256=digest(DESIGN),code_sha256=digest(Path(__file__)),direct_prediction_sha256=seal["prediction_sha256"],
              prediction_sha256=digest(dest),max_native_center_replay_error_m=max_bridge,max_score_replay_error=max_cost,
              max_parameter_vs_box_output_difference_m=max_functional,parameter_count_per_input=578,
              torch_autograd_single_step=True,ordinary_only=True,GT_read=False,source_training=False)
    receipt.write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info),flush=True)


if __name__=="__main__":
    main()
