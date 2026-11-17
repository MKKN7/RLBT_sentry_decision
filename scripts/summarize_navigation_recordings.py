#!/usr/bin/env python3
"""Read-only summary of the three original MuJoCo recordings, not real matches."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = '''
from pathlib import Path
import csv,hashlib,json,math,statistics
reports=[]
for p in sorted(Path('/opt/mid360_mujoco/recordings').glob('*/state.csv')):
    data=p.read_bytes(); rows=list(csv.DictReader(data.decode().splitlines()))
    times=[int(r['time_ns'])*1e-9 for r in rows]
    speed=[math.hypot(float(r['vx']),float(r['vy'])) for r in rows]
    dt=[b-a for a,b in zip(times,times[1:])]
    distance=sum(math.hypot(float(b['x'])-float(a['x']),float(b['y'])-float(a['y'])) for a,b in zip(rows,rows[1:]))
    reports.append({'container_path':str(p),'sha256':hashlib.sha256(data).hexdigest(),
        'rows':len(rows),'duration_s':times[-1]-times[0] if rows else 0,
        'strictly_increasing_time':all(v>0 for v in dt),
        'median_dt_s':statistics.median(dt) if dt else None,
        'speed_mps_median':statistics.median(speed) if speed else None,
        'speed_mps_max':max(speed) if speed else None,'xy_path_length_m':distance,
        'limits':'Simulated self odometry only; no command, opponent, task or detection labels.'})
print(json.dumps(reports))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="Combat_Sentry2026")
    args = parser.parse_args()
    result = subprocess.run(["docker", "exec", "-i", args.container, "python3", "-"],
        input=COLLECTOR, text=True, capture_output=True, check=True, timeout=30)
    records = json.loads(result.stdout)
    for row in records:
        row["source"] = f"docker://{args.container}{row['container_path']}"
    report = {"source_kind": "MuJoCo simulation recording_not_real_robot_or_match_telemetry",
              "read_only": True, "records": records,
              "used_to_calibrate_detection_or_command_latency": False,
              "reason": "Missing paired detection/command/ack timestamps and measured robot truth."}
    path = ROOT / "data/derived/navigation_recording_statistics.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"recordings": len(records), "rows": sum(r["rows"] for r in records),
                      "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
