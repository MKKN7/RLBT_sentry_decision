#!/usr/bin/env bash
set -eo pipefail
rmul_repo=$(realpath -- "${1:?Usage: bash scripts/build_integrated.sh /path/to/guodong-sentry-sim}")
[[ -f "$rmul_repo/src/guodong_sentry_sim/package.xml" ]] || { echo 'Full simulation repository not found.' >&2; exit 1; }
source /opt/ros/jazzy/setup.bash
python3 "$rmul_repo/tools/test_rmul_field.py"
python3 "$rmul_repo/tools/test_step_field.py"
if [[ -f "$rmul_repo/tools/verify_bundle.py" && -f "$rmul_repo/manifest.json" ]]; then
  python3 "$rmul_repo/tools/verify_bundle.py"
fi
rmul_workspace="${SENTRY_WS:-$rmul_repo/output/validation}"
mkdir -p "$rmul_workspace"
cd "$rmul_workspace"
colcon build --base-paths "$rmul_repo/src" --packages-select guodong_sentry_sim
echo "Built in: $rmul_workspace"
echo "Launch: SENTRY_WS='$rmul_workspace' bash '$rmul_repo/scripts/rmul_2027.sh'"
