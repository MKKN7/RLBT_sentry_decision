#!/usr/bin/env bash
# Standalone scene view: Gazebo must already be installed. No robot is included.
set -eo pipefail
rmul_bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  source /opt/ros/jazzy/setup.bash
fi
command -v gz >/dev/null || { echo 'Gazebo Harmonic not found; install the existing simulation prerequisites first.' >&2; exit 1; }
python3 "$rmul_bundle/verify_package.py"
export GZ_SIM_RESOURCE_PATH="$rmul_bundle/src/guodong_sentry_sim/models:$rmul_bundle/src${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GZ_PARTITION="${GZ_PARTITION:-rmul_field_view_$USER}"
exec gz sim -r "$rmul_bundle/standalone/field_only.sdf" "$@"
