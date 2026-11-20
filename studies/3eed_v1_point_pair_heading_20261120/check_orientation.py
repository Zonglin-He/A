"""Compare angular event moments to explicit pair-by-pair quadrature."""
import hashlib
import json
from pathlib import Path

import numpy as np

from orientation import averaged_scatters, estimate


def reference(xy, length, width, angles, weights):
    rotation = np.stack((np.cos(angles), np.sin(angles)))
    a = np.abs(xy @ rotation)
    b = np.abs(xy @ np.stack((-rotation[1], rotation[0])))
    keep = ((a <= length/2) & (b <= width/2)).astype(float)
    other = ((b <= length/2) & (a <= width/2)).astype(float)
    marginal = keep @ weights
    gates = dict(shared=(keep*weights)@keep.T, independent=np.outer(marginal,marginal),
                 crossed=(keep*weights)@other.T, disk=np.ones((len(xy),len(xy))))
    i,j = np.triu_indices(len(xy), 1)
    dx,dy = (xy[i]-xy[j]).T
    pair = np.column_stack((dx**2, dx*dy, dy**2, np.ones(len(dx))))
    return {k: ((g[i,j]+g[j,i])/2)@pair for k,g in gates.items()}


def boundary_quadrature(xy,length,width):
    # Enumerate both signed lines of each edge; determine occupancy at midpoints
    # with direct Cartesian inequalities, independently of event sweep logic.
    ends = [0.,np.pi]
    for x,y in xy:
        radius = np.hypot(x,y)
        for edge in (length/2,width/2):
            if radius > edge:
                phase = np.arctan2(y,x)
                offset = np.arccos(edge/radius)
                for sign in (-1,1):
                    ends.extend([(phase+sign*offset+shift)%np.pi for shift in (0.,np.pi/2)])
    edges = np.unique(ends)
    return (edges[1:]+edges[:-1])/2, np.diff(edges)/np.pi


def main():
    root=Path(__file__).resolve().parent
    rng=np.random.default_rng(20261120)
    errors = dict(exact=0., grid=0., rotation=0.)
    for trial in range(20):
        length,width = rng.uniform(2.4,5.),rng.uniform(.8,1.35)
        phi=rng.uniform(-np.pi,np.pi,13)
        r=rng.uniform(0.,np.hypot(length,width)/2,len(phi))
        xy=np.vstack((np.column_stack((r*np.cos(phi),r*np.sin(phi))),[0.,0.],[width/2,0.]))
        measured=averaged_scatters(xy,length,width)
        boundaries, weights = boundary_quadrature(xy,length,width)
        exact=reference(xy,length,width,boundaries,weights)
        n=32768
        grid=reference(xy,length,width,(np.arange(n)+.5)*np.pi/n,np.full(n,1/n))
        angle=rng.uniform(0,np.pi)
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        turned=averaged_scatters(xy@rot.T,length,width)
        for arm,(scatter,mass) in measured.items():
            packed=np.r_[scatter,mass]
            scale=max(1.,np.max(abs(packed)))
            for name,answer in [('exact',exact),('grid',grid)]:
                errors[name]=max(errors[name],float(np.max(abs(packed-answer[arm]))/scale))
            xx,v,yy=scatter
            matrix=rot@np.array([[xx,v],[v,yy]])@rot.T
            expected=np.array([matrix[0,0],matrix[0,1],matrix[1,1],mass])
            got=np.r_[turned[arm][0],turned[arm][1]]
            errors['rotation']=max(errors['rotation'],float(np.max(abs(expected-got))/scale))
    assert errors['exact']<1e-10 and errors['rotation']<1e-10 and errors['grid']<5e-4,errors
    empty=np.empty((0,3),np.float32)
    default,metadata,_=estimate(empty,[0,0,0,4,1,1],0.)
    assert np.all(default==0.) and metadata[0]==0
    dest=root/'results'
    dest.mkdir(exist_ok=True)
    result=dict(max_normalized_error=errors,random_clouds=20,quadrature_angles=32768,
                native_fallback=True,orientation_sha256=hashlib.sha256((root/'orientation.py').read_bytes()).hexdigest())
    (dest/'geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    main()
