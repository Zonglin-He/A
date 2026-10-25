"""Same three historical illustrative cases, current kappa0 arrays and scalars."""
from ns_common import *
import textwrap

def run():
 validate();q=D/'qualitative';assert not(q/'render_v2_receipt.json').exists();cases=read(PREV/'qualitative/selection.json');metrics={ds:{x['id']:x for x in read(PREV/'temporal_v2_v2'/ds/'metrics.json')['private_rows']}for ds in COUNTS}
 for c in cases:
  path,e=no_shrink(c['dataset'],c['id']);vals=metrics[c['dataset']][c['id']]['values'];c.update(prediction_path=str(path),prediction_sha256=e['sha256'],delta={k:vals['NoShrink'][k]-vals['Source'][k]if vals['NoShrink'][k]is not None and vals['Source'][k]is not None else None for k in ['t','v','unobserved_actual_sIoU']},selection='historical kappa64 '+c['selection']+' case; current kappa0 outputs')
 atomic(q/'selection.json',cases);old=load_old('_old_qual_layout',R/'studies/map_paper_completion_v1/qualitative_render_v2.py',suite());src=textwrap.dedent(inspect.getsource(old.run));oldline="validate('qualitative');q=D/'qualitative';parent=read(q/'render_receipt.json');assert sha(q/'current_MAP_qualitative.png')==parent['PNG_sha256'];cases=read(q/'selection.json');"
 newline="validate();q=D/'qualitative';parent=read(PREV/'qualitative/render_receipt.json');assert sha(PREV/'qualitative/current_MAP_qualitative.png')==parent['PNG_sha256'];cases=read(q/'selection.json');";assert src.count(oldline)==1;src=src.replace(oldline,newline).replace("z['FullMAP']","z['NoShrink']").replace("z['FullMAP_interval']","z['NoShrink_interval']").replace('current_MAP_qualitative_v2','current_no_shrink_v1').replace("sha(q/'render_receipt.json')","sha(PREV/'qualitative/render_receipt.json')").replace('same_prediction_arrays=True','same_prediction_arrays=False,current_kappa=0,same_historical_kappa64_case_selection=True').replace("'Eight expert observations: trajectory changes on unobserved frames and event boundaries'","'No shrinkage (kappa0): eight observations, trajectory changes and event boundaries'")
 def current(ds,key):p,e=no_shrink(ds,key);return p,e['sha256']
 scope={**old.run.__globals__,'D':D,'P':P,'PREV':PREV,'validate':validate,'current_MAP':current,'__file__':__file__};exec(compile(src,'<same_cases_current_no_shrink_layout>','exec'),scope);scope['run']()
if __name__=='__main__':run()
