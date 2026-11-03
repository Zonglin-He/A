from common55 import *
rows=read(ROOT/'crossplatform/round55/conflicts336_attempt001/COVERAGE.json')
result={'n':len(rows),'roles_histogram':{str(i):sum(r['n_roles']==i for r in rows) for i in sorted({r['n_roles'] for r in rows})},'any_conflict_by_variant':[sum(r['nonzero_pairs_by_variant'][i]>0 for r in rows) for i in range(3)],'any_geo_conflict':sum(r['geometry_nonzero_pairs']>0 for r in rows),'any_color_conflict':sum(r['color_conflict_queries']>0 for r in rows),'median_geometry_known_queries':float(np.median([r['geometry_known_queries'] for r in rows])),'median_color_stable_support_queries':float(np.median([r['color_stable_support_queries'] for r in rows])),'GT_reads':0}
dump(ROOT/'research/round55/PRE_GT_OBSERVATION_COVERAGE.json',result);print(json.dumps(result))
