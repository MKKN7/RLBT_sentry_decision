# Source inside Ubuntu 22.04 / ROS 2 Humble. This relocatable overlay contains
# extracted debs, not a system installation. It starts no processes.
_rl_bt_nav2_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "RL-BT Nav2 runtime requires the existing /opt/ros/humble base" >&2
  return 1
fi
source /opt/ros/humble/setup.bash
_rl_bt_nav2_prefix="$_rl_bt_nav2_root/prefix/opt/ros/humble"
export AMENT_PREFIX_PATH="$_rl_bt_nav2_prefix${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export CMAKE_PREFIX_PATH="$_rl_bt_nav2_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export COLCON_PREFIX_PATH="$_rl_bt_nav2_prefix${COLCON_PREFIX_PATH:+:$COLCON_PREFIX_PATH}"
export PATH="$_rl_bt_nav2_prefix/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$_rl_bt_nav2_prefix/lib:$_rl_bt_nav2_root/prefix/usr/lib/x86_64-linux-gnu:$_rl_bt_nav2_root/prefix/usr/lib:$_rl_bt_nav2_root/prefix/usr/lib/llvm-14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$_rl_bt_nav2_prefix/local/lib/python3.10/dist-packages:$_rl_bt_nav2_prefix/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset _rl_bt_nav2_root _rl_bt_nav2_prefix
