from common70 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['clip','A','B']);args=ap.parse_args();p=read(H/'experiment/DESIGN_LOCK.json');p['inputs_sha256'][p['prepared_root']+'/SEALED.json']=sha(ROOT/p['prepared_root']/'SEALED.json')
if args.stage=='clip':names=['clip70.py']
elif args.stage=='A':names=['A70.py','models70.py'];p['inputs_sha256'][p['clip_root']+'/SEALED.json']=sha(ROOT/p['clip_root']/'SEALED.json')
else:
 names=['B70.py','models70.py'];p['inputs_sha256'].pop('crossplatform/round8/assets/clip-vit-base-patch32/SAFE_CONVERSION_RECEIPT.json');p['programs_sha256'].pop('research/round70/clip70.py');p['B_explicit_no_CLIP_features_or_model']=True
if args.stage=='A':p['programs_sha256']['research/round70/clip70.py']=read(H/'experiment/CLIP_PLAN_LOCK.json')['programs_sha256']['research/round70/clip70.py']
for n in names:p['programs_sha256']['research/round70/'+n]=sha(H/n)
path=H/'experiment'/(args.stage.upper()+'_PLAN_LOCK.json');assert not path.exists();dump(path,p);print({'stage':args.stage,'sha256':sha(path)})
