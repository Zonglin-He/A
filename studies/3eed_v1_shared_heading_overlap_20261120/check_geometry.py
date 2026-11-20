"""Compare analytic overlap against numerical integration and independent rotated-rectangle IoU."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad

from heading import HERE, metrics
sys.path.insert(0,str(HERE.parents[1]/'research/round6/source_validation_scoring'))
from batch_yaw_iou import many_iou


def quadrature(a,b,crossed=False):
    d=b[:3]-a[:3]
    z=max(0.,min(a[2]+a[5]/2,b[2]+b[5]/2)-max(a[2]-a[5]/2,b[2]-b[5]/2))
    angle=(np.arange(16384)+.5)*(np.pi/16384)
    shift=np.stack((np.cos(angle)*d[0]+np.sin(angle)*d[1],
                    -np.sin(angle)*d[0]+np.cos(angle)*d[1]),-1)
    size=b[3:5][::-1] if crossed else b[3:5]
    width=np.maximum(0.,np.minimum(a[3:5]/2,shift+size/2)-np.maximum(-a[3:5]/2,shift-size/2))
    inter=z*np.mean(width.prod(1))
    return inter/(a[3:6].prod()+b[3:6].prod()-inter)


def disk_numeric(a,b):
    radius=np.sqrt(np.array([a[3]*a[4],b[3]*b[4]])/np.pi)
    d=np.linalg.norm(a[:2]-b[:2])
    lo,hi=max(-radius[0],d-radius[1]),min(radius[0],d+radius[1])
    z=max(0.,min(a[2]+a[5]/2,b[2]+b[5]/2)-max(a[2]-a[5]/2,b[2]-b[5]/2))
    if hi<=lo or not z: return 0.
    cross=(radius[0]**2-radius[1]**2+d*d)/(2*d) if d>0 else lo
    def length(x):
        return 2*np.sqrt(max(0.,min(radius[0]**2-x*x,radius[1]**2-(x-d)**2)))
    inter=z*quad(length,lo,hi,points=[cross] if lo<cross<hi else [],epsabs=1e-10)[0]
    return inter/(a[3:6].prod()+b[3:6].prod()-inter)


def main():
    rng=np.random.default_rng(557903)
    boxes=np.column_stack((rng.normal(0,.7,(32,3)),np.exp(rng.normal(0,.65,(32,3))),rng.uniform(-np.pi,np.pi,32)))
    boxes[1,:]=boxes[0]
    boxes[2]=[0,0,0,5,.6,1.5,1.2]
    boxes[3]=[0,0,0,5,.6,1.5,.1]
    mats=metrics(boxes)
    assert np.array_equal(mats,mats.transpose(0,2,1))
    assert np.isfinite(mats).all() and mats.min()>=0 and mats.max()<=1
    np.testing.assert_allclose(np.diagonal(mats,axis1=1,axis2=2),1.,atol=0)
    errors=np.zeros(4)
    for i,a in enumerate(boxes):
        errors[1]=max(errors[1],np.max(abs(many_iou(boxes,a)-mats[1,:,i])))
        for j in range(i):
            b=boxes[j]
            v=[quadrature(a,b),disk_numeric(a,b),quadrature(a,b,True)]
            for k,x in zip((0,2,3),v): errors[k]=max(errors[k],abs(x-mats[k,i,j]))
    assert max(errors)<2e-7,errors
    angle=.784372
    cs,sn=np.cos(angle),np.sin(angle)
    rotated=boxes.copy()
    rotated[:,:2]=rotated[:,:2]@np.array([[cs,sn],[-sn,cs]])+[14.,-3.]
    rotated[:,2]-=2.
    rotated[:,6]+=angle
    after=metrics(rotated)
    drift=np.max(abs(after-mats),axis=(1,2))
    assert max(drift[:4])<1e-12 and drift[4]>1e-3
    assert mats[0,2,3]==1. and mats[3,2,3]<.2
    bad=boxes[:3].copy();bad[0,3]=0;bad[1,5]=-2
    np.testing.assert_array_equal(metrics(bad),np.broadcast_to(np.eye(3),(5,3,3)))
    result=dict(pairs=496,numerical_angle_nodes=16384,
                max_shared_quadrature_error=errors[0],max_measured_iou_error=errors[1],
                max_disk_quadrature_error=errors[2],max_crossed_quadrature_error=errors[3],
                rigid_pose_error=drift.tolist(),coincident_elongated_shared=float(mats[0,2,3]),
                coincident_elongated_crossed=float(mats[3,2,3]))
    path=HERE/'results';path.mkdir(exist_ok=True)
    (path/'geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    main()
