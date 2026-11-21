"""One-dimensional containment and calibrated transverse-line readout."""
import numpy as np
from common import QWEN
from geometry import pixel_rays, sensor_pixels

TOL=1.e-7

def axes(box):
    yaw=box[6] if box[3]>=box[4] else box[6]+np.pi/2
    return np.array([-np.sin(yaw),np.cos(yaw),0.]),float(min(box[3:5]))

def center_interval(a,b,width,endpoint=None):
    if endpoint is not None:
        lo,hi=endpoint
        a,b=min(a,hi),max(b,lo)
    lower,upper=b-width/2,a+width/2
    if lower>upper+TOL:
        return None
    if lower>upper:
        lower=upper=(lower+upper)/2
    return float(lower),float(upper)

def offset(interval):
    if interval is None:
        return 0.
    value=float(np.clip(0.,*interval))
    return value if abs(value)>TOL else 0.

def cone_bounds(point,axis,width,xlo,xhi):
    # Solve the undistorted pixel rectangle as linear inequalities in shift s.
    lo,hi=-float(width),float(width)
    tests=[(point[2]-.1,axis[2])]
    for j in range(2):
        tests.extend([(point[j]-xlo[j]*point[2],axis[j]-xlo[j]*axis[2]),
                      (xhi[j]*point[2]-point[j],xhi[j]*axis[2]-axis[j])])
    for intercept,slope in tests:
        if abs(slope)<1.e-12:
            if intercept<0:
                return None
        elif slope>0:
            lo=max(lo,-intercept/slope)
        else:
            hi=min(hi,-intercept/slope)
    return (float(lo),float(hi)) if hi>=lo else None

def read_endpoint(p, nc, station, pixel, spec, mat, normal, width):
    """p is in camera coordinates, station is its frozen raw width coordinate."""
    ray=pixel_rays(pixel,spec)[0]
    u=ray/np.linalg.norm(ray)
    samples=np.array(pixel)+np.array([(x,y) for x in (-4,0,4) for y in (-4,0,4)])
    rays=pixel_rays(samples,spec)
    lo,hi=rays[:,:2].min(0),rays[:,:2].max(0)
    inv=np.linalg.inv(mat[:3,:3])
    # Copying camera optical depth induces a linear interval in raw-axis position.
    corners=np.array([[i,j,1.] for i in (lo[0],hi[0]) for j in (lo[1],hi[1])])
    copied=station+((p[2]*corners-p)@inv.T)@normal
    result=dict(copy=[float(copied.min()),float(copied.max())],copy_z=float(p[2]))
    perp=nc-u*np.dot(nc,u)
    denominator=float(perp@perp)
    result['sin_angle']=float(np.sqrt(denominator)/np.linalg.norm(nc))
    if denominator<1.e-12:
        return result|dict(status='parallel')
    s=float(-perp@p/denominator)
    q=p+s*nc
    if q[2]<=.1:
        return result|dict(status='behind_camera')
    error=float(np.linalg.norm(sensor_pixels(q[None],spec)[0]-pixel))
    result.update(reprojection_px=error,nominal_z=float(q[2]),nominal_s=s)
    if error>4:
        return result|dict(status='outside_4px')
    span=cone_bounds(p,nc,width,lo,hi)
    if span is None:
        return result|dict(status='infeasible_width_or_cone')
    return result|dict(status='feasible',endpoint=[station+span[0],station+span[1]])
