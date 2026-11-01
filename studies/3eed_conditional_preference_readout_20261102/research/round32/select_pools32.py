import sys,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round31'))
from common31 import *
inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];index={x['image_file_sha256']:x for x in inv};history=list(prior.prior.rows())+rows('R64')+rows('S64')
for n in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:history+=read(ROOT/n)
ims={index[sha(r['imagepath'])]['decoded_pixel_sha256'] for r in history};pcs={index[sha(r['imagepath'])]['canonical_XYZ_sha256'] for r in history};known={x['image_sha256'] for x in read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json').values()};full={r['id']:r for r in read(ROOT/'research/round8/experiment/full5912_gpu_inputs.json')};seenI=set();seenP=set();available=[x for x in inv if x['platform'] in ['drone','quad'] and x['image_file_sha256'] not in known and x['decoded_pixel_sha256'] not in ims and x['canonical_XYZ_sha256'] not in pcs];initial_counts=collections.Counter((x['platform'],x['scene_PRIVATE']) for x in available);summary=[];dest=ROOT/'research/NIGHT_20261101/input';dest.mkdir(exist_ok=True)
for split,n,stage,offset in [('T128',64,'T',14000),('U128',64,'U',15000),('FINAL384',192,'F',16000)]:
 result=[];hashes=[]
 for pl in ['drone','quad']:
  pools=collections.defaultdict(list)
  for x in available:
   if x['platform']==pl and x['decoded_pixel_sha256'] not in seenI and x['canonical_XYZ_sha256'] not in seenP:pools[x['scene_PRIVATE']].append((hashlib.sha256(('NIGHT20261101|'+split+'|'+x['pair_group_sha256']).encode()).hexdigest(),x))
  scenes=sorted(pools);weights={s:initial_counts[pl,s] for s in scenes};total=sum(weights.values());quota={s:int(n*weights[s]/total) for s in scenes}
  for s in sorted(scenes,key=lambda s:(-(n*weights[s]/total-quota[s]),s))[:n-sum(quota.values())]:quota[s]+=1
  count=0
  for scene in scenes:
   g=sorted(pools[scene],key=lambda t:t[0]);taken=0
   while taken<quota[scene]:
    while g and (g[0][1]['decoded_pixel_sha256'] in seenI or g[0][1]['canonical_XYZ_sha256'] in seenP):g.pop(0)
    assert g,(split,pl,scene,quota[scene],taken)
    _,x=g.pop(0);r=dict(full[x['canonical_id_PRIVATE']],stage=stage,alias=f'{stage}_{pl}_{count:03d}',frame_alias=f'input_frame_{offset+len(result):05d}');result.append(r);hashes.append(x);seenI.add(x['decoded_pixel_sha256']);seenP.add(x['canonical_XYZ_sha256']);count+=1;taken+=1
  assert count==n
 dump(dest/(split+'_PRIVATE.json'),result);dump(dest/(split+'_CONTENT_HASHES_PRIVATE.json'),hashes);summary.append({'split':split,'n':len(result),'platforms':dict(collections.Counter(r['platform'] for r in result)),'scenes':{p:len({r['scene'] for r in result if r['platform']==p}) for p in ['drone','quad']},'contentdisjoint_prior656_anddeclaredhistory':True,'no_tasklabels_used':True,'scene_quotas_from_input_counts_only':True})
assert len(seenI)==len(seenP)==640;dump(ROOT/'research/NIGHT_20261101/POOL_SELECTION_RECEIPT.json',{'allocations':summary,'new_content':640,'remaining':{pl:sum(x['platform']==pl and x['decoded_pixel_sha256'] not in seenI and x['canonical_XYZ_sha256'] not in seenP for x in available) for pl in ['drone','quad']},'historicalscenes_andfullVALexposure':True,'FINAL384protected':True,'selector_sha256':sha(__file__)});print(summary)
