# P3 classical geometry provenance and frozen interface

This module is standard geometry postprocessing, not a new neuralTTA method. It reads no metadata, target labels, target scores or models. Actual cached evaluation is authorized only by the parent master lock; later GT scoring is separate.

Primary paper: Xiao Zhang,Wenda Xu,Chiyu Dong,JohnM.Dolan, **Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners**, IEEEIV2017, DOI10.1109/IVS.2017.7995698. Official author-institution source: https://publications.ri.cmu.edu/efficient-l-shape-fitting-for-vehicle-detection-using-laser-scanners . PDFURL and SHA are in references/PROVENANCE.json. Paper is retained locally for audit, not licensed for reposting.

Published reference implementation: PythonRobotics, commit8c392897c7f41a6e4d9aab9cb2e9737a7a6fa031, Mapping/rectangle_fitting/rectangle_fitting.py, authorAtsushiSakai, MIT license retained in references/LICENSE_PythonRobotics.txt. Official implementation documentation https://atsushisakai.github.io/PythonRobotics/modules/3_mapping/rectangle_fitting/rectangle_fitting.html . This is a primary documented implementation based on the paper, **not paper-author official code**. No unknown unlicensed thirdparty implementation is copied.

Paper literal objectives used for the preregistered Lshape arms:

- Alg3 area: maximize minus projected boundingrectangle area.
- Alg4 closeness: for each projectionaxis, choose ONE of the whole upper/lower boundary-distance vectors by the smallerL2norm; then per point take the smaller of those two selected-axis distances, floor atd0=0.01metres, maximize sum reciprocal distances. d0=.01 comes from the MIT reference default; paper does not lock its numerical value.
- Alg5 variance: same whole-vector edge choice; E1 containsD1 whereD1<D2 and E2 containsD2 whereD2<D1, so equal-distance points enter neither. Maximize minus population variances, ddof0; empty/singleton groups contribute0. Normalization/empty conventions are explicit because paper pseudocode is underspecified there.
- Angles0..89degrees inclusive,1degree step, equivalent to paperAlg2's inclusive0 to90−delta range. Firstmaximum on ties. No GT selection of metric or angles.

The pinned PythonRobotics common implementation differs: per-point nearerparallelboundary ratherthan whole-vectorL2edge selection; E2 uses>=ties; its half-open np.arange(0,90−1,1) evaluates0..88degrees. A separately named compatibilityAPI reproduces those choices for engineering, **not a mixed or additional scored arm**. The paper-vs-code analytic fixture gives closeness301vs400 and variance−.25vs0; hence they cannot be silently called the same objective. Root's independent renderedPDF audit corroborated these differences.

Minimum-area specification: https://shapely.readthedocs.io/en/stable/reference/shapely.minimum_rotated_rectangle.html . Implementation uses Andrew convexhull and everyhull-edge orientation with exact projected extents; finite edge enumeration, no angulargrid. Objectiveis minimumenclosingrectangle area. Degenerate line/point geometry falls back to source, not a fabricated3Dvolume. Synthetic comparison toShapely2.2.0/GEOS passed30random cases with areaabsolute error<=1.07e-13.

Shared source/crop contract: source selectedquery is fixed argmax of cachedsemantic_scores. Center/size/source yaw0 are unchanged for all orientation-only arms. One common sourceAABB crop, tolerance1e-7m, unique observedXYZpoints; membershipuses nativecachedpoint/sourcebox dtype identical to historicalPCA, then the selectedcrop is cast tofloat64 for all geometry. No segmentation, candidate ranking or recropping at newyaws. Sharedmin8unique points; n1,n2, coincidentXY, collinearXY returnexactsource. All crops are hashed.

Axis correspondence: yaw describes the local sourceXextent axis. PCA's largestcovarianceeigenvector is its estimatedlongaxis; minarea/Lshape longaxis is chosen by fitted projected spans. The estimatedlongaxis maps to the larger sourceXYextent without swapping source size entries. IfsourceL<W, headingrotates bypi/2. Squareaxis ambiguity uses the equivalentaxis closest tosourceyaw. Allreported yaws aremodpi in[-pi/2,pi/2), trigonometry done infloat64; sourcecenter/size remain bitexact.

Geometry optimization strongcontrol: reuse unchanged round2.surface_teacher observations/energy on exactlythatcrop; optimize only yaw by1D boundedL-BFGS-B, fixedstarts[0,PCA,PCA+pi/2], bounds[-pi/2,pi/2], maxiter20,ftol1e-9,maxls15. No originalacceptance gate, no candidate/center/size update; best ofinitial/final iterates is retained. Surface code SHAis sealed. This is distinct from legacyjoint_adapter3, whose center/selectedquery can change; that oldsealed output is only a separate capability/reference group.

Optional full_refit_from_crop changesXYcenter/extents/yaw and optionallyZcenter/height, explicitly markedclassical_full_refit. It is exercised only in syntheticengineering, not included in the actual predeclared9-arm cachedrun. It never masquerades as orientation-only.
