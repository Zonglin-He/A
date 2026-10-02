#!/usr/bin/env bash
set -euo pipefail
# The automatic system update left the loaded kernel at 580.159.03.
# Use the matching extracted userspace package for this process only.
vg_driver_lib='/home/wwww/visual grounding/.runtime/nvidia-580.159.03/extracted/usr/lib/x86_64-linux-gnu'
if [[ -f "$vg_driver_lib/libcuda.so.580.159.03" ]] && [[ "$(</proc/driver/nvidia/version)" == *580.159.03* ]]; then
  export LD_LIBRARY_PATH="$vg_driver_lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$@"
