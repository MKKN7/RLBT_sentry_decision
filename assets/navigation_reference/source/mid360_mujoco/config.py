from pathlib import Path
import copy
import os
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(path=None):
    with open(path or ROOT / "configs/simulation.yaml") as f:
        c = yaml.safe_load(f)
    c = copy.deepcopy(c)
    c['lidar']['backend']=os.environ.get('MID360_RAY_BACKEND',c['lidar'].get('backend','cpu'))
    if c['lidar']['backend'] not in ('cpu','cuda'):raise ValueError('LiDAR backend must be cpu or cuda')
    c['lidar'].setdefault('batch_steps',100)
    c['physics'].setdefault('collision_window_radius_m',4. if c['lidar']['backend']=='cuda' else 0.)
    c['physics'].setdefault('cached_forward',True)
    dt = c["physics"]["timestep"]
    if not 0 < dt <= 0.005:
        raise ValueError("physics timestep must be in (0, 5ms]")
    # Old saved configurations predate UWB; preserve their sensor behavior.
    if 'uwb' not in c:
        with open(ROOT / 'configs/simulation.yaml') as f:
            c['uwb']=yaml.safe_load(f)['uwb']
        c['uwb']['enabled']=False
    for rate in (c["imu"]["rate"], c["lidar"]["frame_rate"], c['uwb']['rate_hz']):
        steps = 1 / (dt * rate)
        if abs(steps - round(steps)) > 1e-8:
            raise ValueError("sample periods must be integer physics steps")
    if c["lidar"]["point_rate"] % c["lidar"]["frame_rate"]:
        raise ValueError("point rate must be divisible by frame rate")
    if 1_000_000_000 % c["lidar"]["point_rate"]:
        raise ValueError("point interval must be integral nanoseconds")
    if not 0 <= c["lidar"]["range_sigma"] < c["lidar"]["max_range"]:
        raise ValueError("invalid range noise")
    steps=c['lidar']['batch_steps']
    if not isinstance(steps,int) or steps<1 or round(1/(dt*c['lidar']['frame_rate']))%steps:
        raise ValueError('LiDAR batch_steps must divide the scan frame')
    if c['lidar']['backend']=='cuda' and abs(c['lidar']['point_rate']*dt-round(c['lidar']['point_rate']*dt))>1e-8:
        raise ValueError('CUDA backend requires integral rays per physics step')
    return c

