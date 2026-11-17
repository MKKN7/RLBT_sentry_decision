#!/usr/bin/env bash
# This script is installed into the existing full repository's scripts/ directory.
set -eo pipefail
rmul_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rmul_workspace="${SENTRY_WS:-$rmul_repo/output/validation}"
source /opt/ros/jazzy/setup.bash
[[ -f "$rmul_workspace/install/setup.bash" ]] || { echo "Build missing: $rmul_workspace/install/setup.bash" >&2; exit 1; }
source "$rmul_workspace/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY=1
export GZ_PARTITION="${GZ_PARTITION:-rmul_2027_$USER}"
exec ros2 launch guodong_sentry_sim sentry_sim.launch.py world:=rmul_2027 visual_quality:=preview rviz:=true "$@"
