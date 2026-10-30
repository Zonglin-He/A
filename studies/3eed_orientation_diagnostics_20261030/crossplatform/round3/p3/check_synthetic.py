"""Meaningful synthetic engineering checks, never data/GT metrics."""
import pathlib,json,hashlib,itertools,sys
import numpy as np
from geometry_baselines import *
P=Path(__file__).resolve().parent

def cloud(yaw=.0,center=(0,0,0),L=4,W=2,H=1):
    x,y=np.meshgrid(np.linspace(-L/2,L/2,25),np.linspace(-W/2,W/2,15));xy=np.c_[x.ravel(),y.ravel()]@rotation(yaw).T
    p=np.vstack([np.c_[xy,np.full(len(xy),z)] for z in [-H/2,H/2]])+np.asarray(center);return p

def main():
    theta=np.deg2rad(37);points=cloud(theta);src=np.array([0.,0.,0.,8.,8.,4.,0.],dtype=np.float32)
    # Enclosing/pca geometry is tested with a rectangular (not L-shaped triangular) hull.
    standard=np.array([0.,0,0,4.,2.,1.,0.],dtype=np.float32)
    crop=points.copy();checks={};diagnostics={}
    for method in METHODS:
        b,d=orientation_only_from_crop(crop,standard,method);assert np.array_equal(b[:6],standard[:6]);assert np.isfinite(b).all();checks[method]={'center_size_bitexact':True,'yaw_radians':float(b[6]),'reason':d['reason']};diagnostics[method]=d
    assert angle_difference(checks['PCA_standard_axisaware']['yaw_radians'],theta)<1e-6
    assert angle_difference(checks['minimum_area_exact']['yaw_radians'],theta)<1e-6
    shortx=standard.copy();shortx[3:5]=[2.,4.]
    for method in ['PCA_standard_axisaware','minimum_area_exact','Lshape_closeness']:
        b,_=orientation_only_from_crop(crop,shortx,method);assert angle_difference(float(b[6]),theta+np.pi/2)<np.deg2rad(1.01),(method,b)
    # One/two/collinear and duplicate-only inputs use exactsource fallback.
    degenerate=[np.array([[0.,0,0]]),np.array([[0.,0,0],[1.,0,0]]),np.c_[np.arange(12),np.arange(12)*2,np.zeros(12)],np.zeros((20,3))]
    for p in degenerate:
        for method in METHODS:
            b,d=orientation_only_from_crop(p,standard,method);assert np.array_equal(b,standard) and not d['updated']
    # Sorting/deduplication makes all methods invariant to input order/repetition.
    rng=np.random.default_rng(17);p=np.r_[points,points[:100]];shuffled=p[rng.permutation(len(p))]
    for method in METHODS:
        a,_=orientation_only_from_crop(points,standard,method);b,_=orientation_only_from_crop(shuffled,standard,method);assert np.array_equal(a,b),(method,a,b)
    # Paper/implementation are intentionally different, on an analytically solvable fixture.
    c1=np.array([0,0,2,2]);c2=np.array([0,1,0,1]);assert lshape_score(c1,c2,'closeness')==301.;assert lshape_score(c1,c2,'closeness','pythonrobotics')==400.;assert lshape_score(c1,c2,'variance')==-.25;assert lshape_score(c1,c2,'variance','pythonrobotics')==0.
    _,_,_,pd=lshape_rectangle(points[:,:2],'area');_,_,_,rd=lshape_rectangle(points[:,:2],'area','pythonrobotics');assert pd['angles_evaluated']==90 and rd['angles_evaluated']==89
    # Known two-side L fixture verifies papercloseness/variance recovera gridaligned edge.
    local=np.vstack([np.c_[np.linspace(-2,2,121),np.full(121,-1)],np.c_[np.full(61,-2),np.linspace(-1,1,61)]]);lp=np.c_[local@rotation(theta).T,np.zeros(len(local))]
    lresults={}
    for method in ['Lshape_closeness','Lshape_variance']:
        b,d=orientation_only_from_crop(lp,standard,method);lresults[method]=float(np.rad2deg(angle_difference(float(b[6]),theta)));assert lresults[method]<=1.01,lresults
    # Exactminarea vs independent GEOS oracle; random translation and anisotropy.
    oracle=[]
    try:
        import shapely
        from shapely.geometry import MultiPoint
        for seed in range(30):
            g=np.random.default_rng(seed);xy=g.normal(size=(40,2))*[3,1]+[12,-4];angle,lo,hi,d=minimum_area_rectangle(xy);reference=MultiPoint(xy).minimum_rotated_rectangle.area;error=abs(d['area_m2']-reference);assert error<1e-9*max(reference,1.),(seed,error,reference);oracle.append(error)
        oracle_info={'library':'Shapely/GEOS','version':shapely.__version__,'cases':len(oracle),'max_area_abs_error':max(oracle)}
    except ImportError:oracle_info={'library':'Shapely unavailable','cases':0}
    # Exactcropand selectedquery shared. No transform recrops pointsets.
    outputs,meta=compare_orientation(points,src,selected_query=71);digests={d['crop_sha256_float64_xyz'] for d in meta['methods'].values()};assert len(digests)==1 and meta['selected_query_unchanged']==71
    opt=diagnostics['surface_energy_orientationonly_optimization'];assert opt['selected_energy']<=min(r['initial_energy'] for r in opt['runs'])+1e-15;assert not opt['original_acceptance_rules_used']
    # Full refit is separate, actual geometrychanges exercised.
    moved=cloud(theta,center=(.7,-.4,.3));full,diag=full_refit_from_crop(moved,src,'minimum_area_exact');assert diag['center_or_size_changed'];assert np.allclose(full[:3],[.7,-.4,.3],atol=1e-6);assert np.allclose(full[3:6],[4,2,1],atol=1e-6)
    result={'state':'passed','scope':'syntheticengineeringonly; no targetcache/metaGT/model/GPU read','orientation_checks':checks,'all_source_center_size_bitexact':True,'short_x_source_axis_correspondence':True,'degenerate_cases':len(degenerate),'all_methods_permutation_duplicate_invariant':True,'paper_vs_MIT_fixture':{'paper_closeness':301,'MIT_closeness':400,'paper_variance':-.25,'MIT_variance':0,'paper_angles':90,'MIT_angles':89},'L_fixture_error_degrees':lresults,'minimum_area_independent_oracle':oracle_info,'one_shared_crop_digest':True,'fixed_selected_query':71,'geometryopt_energy_nonincrease_vsstarts':True,'separate_full_refit_changes_geometry':True,'code_sha256':hashlib.sha256((P/'geometry_baselines.py').read_bytes()).hexdigest()}
    (P/'SYNTHETIC_CHECKS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
