import sysconfig,runpy
old="/home/wwww/visual grounding/.conda/tubedetr"
new="/home/wwww/new/crossplatform/runtime_base"
for k,v in sysconfig.get_config_vars().items():
 if isinstance(v,str):sysconfig.get_config_vars()[k]=v.replace(old,new)
runpy.run_path("setup.py",run_name="__main__")
