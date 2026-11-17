# Project-private Nav2 runtime

This directory persists extracted Ubuntu 22.04 amd64 / ROS 2 Humble packages.
It is an overlay for the existing Humble base, not a host ROS installation.
No `apt install`, maintainer script, ROS node, or system package upgrade is
executed by the preparation script.

Prepare or reconstruct the private runtime from the host:

```bash
bash scripts/prepare_nav2_overlay.sh
```

The default container is `Combat_Sentry2026`; its private staging path is
`/tmp/rl_bt_nav2_overlay`. Source its setup inside that container before running
the project's isolated probe:

```bash
source /tmp/rl_bt_nav2_overlay/setup.bash
ros2 pkg prefix nav2_mppi_controller
```

After copying this whole directory into another Ubuntu 22.04 / Humble container,
source its `setup.bash` at the copied location. The setup derives its own prefix
from `BASH_SOURCE`; it does not depend on the staging path. The host's Python
3.11 Conda environment cannot import these Python 3.10 ROS extensions directly.

`data/reports/nav2_overlay_20261116.json` records the 46 exact package versions,
archive SHA256 and apt checksums, unchanged shared dpkg-status hash, and actual
package-discovery, action-import and `ldd` checks. Nav2 packages are 1.1.20.

The base package solver pulled some development packages as declared hard
dependencies. They are retained to reproduce the resolved dependency set. The
persisted prefix is about 84 MiB; downloaded archives are deleted only from the
task's own staging directory after extraction. `--verify-only` rechecks an
existing staged prefix and copies it to the persistent project directory.
