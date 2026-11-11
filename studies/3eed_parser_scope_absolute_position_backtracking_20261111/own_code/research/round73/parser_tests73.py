from parser73 import *
def main():
 tests=[]
 for form in ['left of','to the left of','on the left of','on the left side of','to the left side of']:
  s='a red car '+form+' a white truck';d=DIRECTION.search(s);a=ALLREL.search(s);assert a.span()==d.span();t=classify(s);assert t['kind']=='ENTITY_RELATIVE' and 'white' not in s[slice(*t['target_span'])];tests.append('shared_span_'+form)
 for s,d in [('The car on the left side of the image.','left'),('A person in the upper right of the view.','right'),('The car slightly to the right of the center of the view.','right')]:
  t=classify(s);assert t['kind']=='IMAGE_ABSOLUTE' and t['direction']==d;tests.append('explicit_frame')
 for s in ['The car to the left of the road.','The car behind a person on the left of the image.','The car not on the left side of the image.','A person near a car to the right of the image.']:
  assert classify(s)['kind']!='IMAGE_ABSOLUTE';tests.append('unknown_scope_or_non_image')
 rect=np.array([[70,10,90,90],[10,10,30,90],[40,10,70,90],[90,10,110,90]],float);N,K,I=absolute_negatives(rect,[1,1,1,1],[1,1,1,.8],100,'left');assert N.tolist()==[True,False,False,False];tests.append('full_interval_unknown')
 assert DIRECTION_PATTERN in ALLREL.pattern;tests.append('single_direction_source')
 return {'state':'PASS_R73_INDEPENDENTLY_IMPLEMENTED_FINITE_PARSER_AND_GEOMETRY_TESTS','tests':len(tests),'original_attachment_ZIP_not_available':True,'real_task_effect_not_tested_by_synthetic_tests':True}
if __name__=='__main__':print(main())
