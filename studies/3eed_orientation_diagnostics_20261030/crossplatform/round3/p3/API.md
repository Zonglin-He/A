# P3 frozen API and outputs

All functions live in geometry_baselines.py and use NumPy/SciPy CPU only. Box format is `[cx,cy,cz,extent_x,extent_y,extent_z,yaw]`; yaw is localXextent direction, moduloπ. Inputs must be finite. No positive-size repair, labels, model or candidate selection occurs inside the estimators.

```python
from geometry_baselines import compare_orientation,orientation_only_from_crop
outputs,diagnostics=compare_orientation(points_xyz,source_box7,selected_query=source_query)
# One sourceAABB crop for all six newmethods. source_query is carried, never changed.
box7,info=orientation_only_from_crop(precomputed_same_crop,source_box7,
                                   'Lshape_closeness')
```

Methods: `PCA_standard_axisaware`, `minimum_area_exact`, `Lshape_area`, `Lshape_closeness`, `Lshape_variance`, `surface_energy_orientationonly_optimization`. All preserve source[:6] bitexact, and only overwriteyaw in the original source floatingdtype. Downstream rotation/trigonometry must first cast yaw toPythonfloat/float64, as in geometry_baselines.rotation.

`crop_source_aabb(points,source)` returns deduplicatedcrop and a float64XYZbytes SHA256. It uses native input/source dtype forAABBmembership to exactly preserve oldPCA membership; casts tofloat64 only afterward. Parent's round3/p2/CROP_PRECISION_AUDIT proves native/float64 predicate results areidentical in all54 actual cachedsamples. `orientation_only_from_crop` never recrops or reselects objects.

`lshape_score(c1,c2,criterion,variant='paper_literal')` exposes objective math. `lshape_rectangle(...,variant='pythonrobotics')` is an explicitly named MITcommon-implementation compatibility API, not an actualadditional arm. PrimaryscoredLshape methods are paperliteral; seePROVENANCE.md for whole-vectorL2edge choice, strict ties, variance convention and90v89angular endpointdifference.

`full_refit_from_crop(crop,source,method,refit_z=True)` is a separate optional ability. It changes center/extents/yaw and is markedclassical_full_refit. It is NOT included in the actual9-arm output. Withrefit_z=False only footprintcenter/size/yaw change whileZis explicitlypreserved. Degenerate fitted extents returnsource.

Actualcached run: predictions/SEALED.json (lowercase sealed.json is exactsamebytes) contains `n=54`, `arms`, NPZfilename→SHA `files`, sourcecode/plan hashes and explicitcapability groups. Each54 NPZ file contains nine7-entry arrays with keys:

1. source_yaw0
2. legacy_PCA_reference
3. PCA_standard_axisaware
4. minimum_area_exact
5. Lshape_area
6. Lshape_closeness
7. Lshape_variance
8. surface_energy_orientationonly_optimization
9. legacy_joint_adapter3_reference

The finalreference is copied exactly from oldsealedroof_constrained3; it can change query/center and must not be counted as yaw-only. LegacyPCAis copied exactly from oldseal, retaining its older min8/L_over_W>=1.5gate; standardPCA uses no data-tunedelongationgate and maps longaxis tosource dimensions.

trace.json provides sourcequery, crop SHA/support, objective/orientation/optimizer diagnostics and timings. input_index.json contains onlyGT-free sample IDs/platforms/scenes/seed/pointcachepath/hash. NumericGT, metadata paths and prior outcome scores are neitheropened norread. Scopeis the54 historicallyexposed round2dev18+confirm36union under union-trained checkpoint/sourcefirstnoun readout. These are posthoc cacheddiagnostics, notnewconfirmation orsource-onlydomainshift evidence.

Synthetic checks and exact GEOSarea oracle comparisons are engineering evidence only. Parent performs all realGTscoring after seal; this package performs no score-driven method orsample changes.
