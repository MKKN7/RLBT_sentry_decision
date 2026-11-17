from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scripts.train_phase1 import snapshot_environments, restore_environments, step_environment


def test_parallel_and_snapshot_resume_preserve_environment_stream():
    from rl_bt.alpharm_task_env import AlphaRuntime, AlphaTaskEnv
    runtime = AlphaRuntime()
    envs = [AlphaTaskEnv(runtime, seed, opponent_style="capture") for seed in (39, 42)]
    saved = snapshot_environments(envs)
    serial = [env.step(1) for env in envs]
    restore_environments(envs, saved)
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(step_environment, zip(envs, (1, 1))))
    for a, b in zip(serial, parallel):
        for name in a[0]:
            np.testing.assert_array_equal(a[0][name], b[0][name])
        assert a[1:] == b[1:]
    future_state = snapshot_environments(envs)
    expected = [env.step(2) for env in envs]
    restored = [AlphaTaskEnv(runtime, 999) for _ in envs]
    restore_environments(restored, future_state)
    actual = [env.step(2) for env in restored]
    for a, b in zip(expected, actual):
        for name in a[0]:
            np.testing.assert_array_equal(a[0][name], b[0][name])
        assert a[1:] == b[1:]
