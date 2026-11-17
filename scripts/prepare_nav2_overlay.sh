#!/usr/bin/env bash
# Download and extract trusted Humble debs into a private prefix. Never install
# packages, run maintainer scripts, change apt sources, or start ROS nodes.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${RL_BT_NAV2_CONTAINER:-Combat_Sentry2026}"
WORK_ROOT="${RL_BT_NAV2_WORK_ROOT:-/tmp/rl_bt_nav2_overlay}"
DEST_ROOT="${RL_BT_NAV2_DEST_ROOT:-$PROJECT_ROOT/runtime/nav2}"
REPORT="${RL_BT_NAV2_REPORT:-$PROJECT_ROOT/data/reports/nav2_overlay_20261116.json}"
if [[ "$WORK_ROOT" != /tmp/rl_bt_nav2_overlay* ]]; then
  echo "Private work directory must begin /tmp/rl_bt_nav2_overlay" >&2
  exit 2
fi
mkdir -p "$DEST_ROOT" "$(dirname -- "$REPORT")"
if [[ "${1:-}" != --verify-only ]]; then
docker exec -i "$CONTAINER" python3 - "$WORK_ROOT" <<'PY'
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request

work = Path(sys.argv[1])
prefix = work / "prefix"
debs = work / "debs"
for path in (work, prefix, debs):
    path.mkdir(parents=True, exist_ok=True)
requested = ["ros-humble-" + name for name in (
    "nav2-controller", "nav2-planner", "nav2-navfn-planner",
    "nav2-mppi-controller", "nav2-bt-navigator", "nav2-behaviors",
    "nav2-lifecycle-manager", "nav2-map-server",
)]

def run(argv, **kwargs):
    result = subprocess.run(argv, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{argv!r} failed ({result.returncode}): {result.stderr}")
    return result.stdout

status_file = Path("/var/lib/dpkg/status")
status_before = hashlib.sha256(status_file.read_bytes()).hexdigest()
simulate = run(["apt-get", "--simulate", "--no-install-recommends", "install", *requested])
(work / "apt_simulation.txt").write_text(simulate)
packages = []
for line in simulate.splitlines():
    match = re.match(r"Inst (\S+)(?: \[[^]]+\])? \((\S+)", line)
    if match:
        packages.append({"package": match[1], "version": match[2]})
uris = run(["apt-get", "--print-uris", "--yes", "--download-only", "--no-install-recommends", "install", *requested])
(work / "apt_uris.txt").write_text(uris)
uri_rows = []
for line in uris.splitlines():
    if line.startswith("'"):
        row = shlex.split(line)
        uri_rows.append({"url": row[0], "filename": row[1], "bytes": int(row[2]), "apt_checksum": row[3]})
if len(uri_rows) != len(packages):
    raise RuntimeError("Dependency solver and download URI counts differ")
if sum(row["bytes"] for row in uri_rows) > 150 * 1024**2:
    raise RuntimeError("Download exceeds private runtime budget (150 MiB)")
if shutil.disk_usage(work).free < 400 * 1024**2:
    raise RuntimeError("Insufficient free space for a private overlay")

def download(item):
    package, row = item
    path = debs / row["filename"]
    expected_md5 = row["apt_checksum"].split(":", 1)[1]
    def valid():
        return path.is_file() and path.stat().st_size == row["bytes"] and hashlib.md5(path.read_bytes()).hexdigest() == expected_md5
    attempts = []
    if not valid():
        result = subprocess.run(["apt-get", "-o", "Acquire::Retries=2", "download", package["package"] + "=" + package["version"]], cwd=debs, text=True, capture_output=True)
        attempts.append({"method": "apt-get download", "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    if not valid():
        url = row["url"]
        if "/ros2/ubuntu/" in url:
            fallback = "https://packages.ros.org/ros2/ubuntu/" + url.split("/ros2/ubuntu/", 1)[1]
        elif "/ubuntu/" in url:
            fallback = "https://archive.ubuntu.com/ubuntu/" + url.split("/ubuntu/", 1)[1]
        else:
            raise RuntimeError(f"No permitted fallback for {url}")
        request = urllib.request.Request(fallback, headers={"User-Agent": "RL-BT-private-overlay/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
        attempts.append({"method": "official URL fallback", "url": fallback})
    if not valid():
        raise RuntimeError("Downloaded package failed apt checksum: " + str(path))
    metadata = run(["dpkg-deb", "--field", str(path), "Package", "Version", "Architecture"])
    actual = dict(line.split(": ", 1) for line in metadata.splitlines())
    if actual["Package"] != package["package"] or actual["Version"] != package["version"]:
        raise RuntimeError("Package identity mismatch")
    print("Downloaded " + package["package"] + " " + package["version"], flush=True)
    return {**package, **row, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "download_attempts": attempts, "architecture": actual["Architecture"]}

# Distinct archive files can download concurrently; extraction remains serial.
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    archives = list(executor.map(download, zip(packages, uri_rows)))
for package in archives:
    run(["dpkg-deb", "--extract", str(debs / package["filename"]), str(prefix)])
status_after = hashlib.sha256(status_file.read_bytes()).hexdigest()
if status_before != status_after:
    raise RuntimeError("Shared dpkg status changed during private preparation")
manifest = {
    "schema": "rl_bt_nav2_private_overlay_v1",
    "prepared_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "requested_packages": requested,
    "packages": archives,
    "private_container_root": str(work),
    "private_container_prefix": str(prefix),
    "apt_install_executed": False,
    "maintainer_scripts_executed": False,
    "apt_sources_changed": False,
    "nodes_started": False,
    "shared_dpkg_status_sha256_before": status_before,
    "shared_dpkg_status_sha256_after": status_after,
    "download_bytes": sum(p["bytes"] for p in archives),
    "extracted_regular_file_bytes": sum(p.stat().st_size for p in prefix.rglob("*") if p.is_file() and not p.is_symlink()),
    "runtime_requires": "Ubuntu 22.04 amd64 and the existing /opt/ros/humble base",
}
(work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
# Only these task-created archive files are removed; manifest retains hashes.
for package in archives:
    (debs / package["filename"]).unlink()
PY
fi
docker cp "$CONTAINER:$WORK_ROOT/prefix/." "$DEST_ROOT/prefix"
docker cp "$PROJECT_ROOT/runtime/nav2/setup.bash" "$CONTAINER:$WORK_ROOT/setup.bash"
docker exec -i "$CONTAINER" bash -s -- "$WORK_ROOT" <<'BASH'
set -e
private_root="$1"
source "$private_root/setup.bash"
python3 - "$private_root" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
checks = {}
def check(name, argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    checks[name] = {"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode or "not found" in result.stdout:
        raise RuntimeError(f"Runtime check failed: {name}: {result.stdout} {result.stderr}")
check("mppi_package_prefix", ["ros2", "pkg", "prefix", "nav2_mppi_controller"])
check("nav2_msgs_import", ["/usr/bin/python3", "-c", "import nav2_msgs; from nav2_msgs.action import NavigateToPose; print(nav2_msgs.__file__); print(NavigateToPose.Goal())"])
ros_prefix = root / "prefix/opt/ros/humble"
for name, relative in (
    ("controller_server_ldd", "lib/nav2_controller/controller_server"),
    ("planner_server_ldd", "lib/nav2_planner/planner_server"),
    ("mppi_plugin_ldd", "lib/libmppi_controller.so"),
    ("bt_navigator_ldd", "lib/nav2_bt_navigator/bt_navigator"),
    ("map_server_ldd", "lib/nav2_map_server/map_server"),
):
    check(name, ["ldd", str(ros_prefix / relative)])
manifest["verification"] = checks
manifest["verified"] = True
manifest["setup_sha256"] = hashlib.sha256((root / "setup.bash").read_bytes()).hexdigest()
(root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"verified": True, "private_prefix": str(ros_prefix), "packages": len(manifest["packages"]), "download_bytes": manifest["download_bytes"], "extracted_regular_file_bytes": manifest["extracted_regular_file_bytes"]}))
PY
BASH
docker cp "$CONTAINER:$WORK_ROOT/manifest.json" "$REPORT"
echo "Private runtime persisted at $DEST_ROOT/prefix"
echo "Manifest: $REPORT"
