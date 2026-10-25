"""Same three historical illustrative cases, selected shared-ROI arrays and scalars."""
from shared_common import *
import textwrap
def current(ds,key):
 ap=D/f'components_{ds}_reporting_receipt.json';assert read(ap)['status']=='verified_whole_domain';e=fixed(D/f'components_{ds}_sealed.json')['files'][key];p=D/e['path'];assert sha(p)==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];return p,e['sha256']
def run():
 validate();q=D/'qualitative';assert not(q/'render_v2_receipt.json').exists();cases=read(PREV/'qualitative/selection.json');metrics={ds:{r['id']:r for r in read(D/'components'/ds/'metrics.json')['private_rows']}for ds in COUNTS}
 for c in cases:
  path,h=current(c['dataset'],c['id']);values=metrics[c['dataset']][c['id']]['values'];c.update(prediction_path=str(path),prediction_sha256=h,selection='historical two-DP '+c['selection']+' case; current shared-corrected-box ROI',delta={k:values['FullMAP'][k]-values['Source'][k]if values['FullMAP'][k]is not None and values['Source'][k]is not None else None for k in ['t','v','unobserved_actual_sIoU']})
 atomic(q/'selection.json',cases);old=load_old('_selected_shared_ROI_qualitative_parent',R/'studies/map_paper_completion_v1/qualitative_render_v2.py',suite());src=textwrap.dedent(inspect.getsource(old.run));needle="validate('qualitative');q=D/'qualitative';parent=read(q/'render_receipt.json');assert sha(q/'current_MAP_qualitative.png')==parent['PNG_sha256'];cases=read(q/'selection.json');";assert src.count(needle)==1
 src=src.replace(needle,"validate();q=D/'qualitative';parent=read(PREV/'qualitative/render_receipt.json');assert sha(PREV/'qualitative/current_MAP_qualitative.png')==parent['PNG_sha256'];cases=read(q/'selection.json');").replace('current_MAP_qualitative_v2','current_shared_corrected_ROI_v1').replace("sha(q/'render_receipt.json')","sha(PREV/'qualitative/render_receipt.json')").replace('same_prediction_arrays=True','same_prediction_arrays=False,first_auxiliary_DP=False,final_DP=True,same_historical_case_selection=True').replace("'Eight expert observations: trajectory changes on unobserved frames and event boundaries'","'Shared-corrected-box ROI features: eight observations, final trajectory DP retained'")
 scope={**old.run.__globals__,'D':D,'P':P,'PREV':PREV,'validate':validate,'current_MAP':current,'__file__':__file__};exec(compile(src,'<same_historical_cases_selected_shared_ROI_layout>','exec'),scope);scope['run']()
 # Only signed scalars, hashes and protocols are public. All captions/media and
 # prediction/frame coordinates remain exclusively in the private output root.
 atomic(P/'qualitative/signed_scalar_cases.json',[dict(case=i+1,dataset=c['dataset'],selection='Same historical post-hoc illustrative case, no new selection',delta_pp={k:100*v if v is not None else None for k,v in c['delta'].items()},prediction_sha256=c['prediction_sha256'])for i,c in enumerate(cases)])
if __name__=='__main__':run()
