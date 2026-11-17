"""Blocking host environment whose transitions come from a live ROS session."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import numpy as np

from rl_bt.live_nav_observation import LIVE_NAV_SCENARIO_ID

ROOT=Path(__file__).resolve().parents[1]


class LiveNav2Env:
    """A fresh MuJoCo/Point-LIO/Nav2 session for each episode; no proxy motion.

    Checkpoints cannot restore physical/estimator state, so this environment
    intentionally has no exact-resume claim. The ROS runner owns cleanup and
    copies its logs after close. It must never be run concurrently in domain79.
    """
    def __init__(self,run_dir,goal=(0.,.4),wall_timeout_s=180.,decision_dt_s=.5,
                 task_timeout_s=20.,capture_terrain_input=False,simulation_seed=None,**kwargs):
        self.run_dir=Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True,exist_ok=True)
        self.goal=tuple(map(float,goal))
        self.wall_timeout_s=float(wall_timeout_s)
        self.decision_dt_s=float(decision_dt_s)
        self.task_timeout_s=float(task_timeout_s)
        self.capture_terrain_input=bool(capture_terrain_input)
        if simulation_seed is not None and (type(simulation_seed) is not int or simulation_seed < 0):
            raise ValueError("Simulation seed must be a nonnegative integer")
        self.simulation_seed=simulation_seed
        self.process=None
        self.episode_index=-1
        self.request_id=0
        self.last_response=None
        self.obs=None
        self.transitions=[]
        self.return_total=0.
        self.final_artifacts=[]

    def _reader(self,pipe):
        try:
            for line in pipe:
                try:
                    item=json.loads(line)
                except json.JSONDecodeError:
                    self.responses.put(RuntimeError("Non-JSON session stdout: "+line[:300]))
                    continue
                self.responses.put(item)
        finally:
            self.responses.put(EOFError("Live Nav2 session output closed"))

    def _rpc(self,command,timeout):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Live Nav2 session is not running")
        self.request_id+=1
        command={**command,"request_id":self.request_id}
        self.process.stdin.write(json.dumps(command,allow_nan=False)+"\n")
        self.process.stdin.flush()
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            try:item=self.responses.get(timeout=max(.01,deadline-time.monotonic()))
            except queue.Empty:break
            if isinstance(item,BaseException):raise item
            if item.get("type")=="error":raise RuntimeError("ROS session: "+json.dumps(item))
            if item.get("request_id")!=self.request_id:
                self.event_stream.write(json.dumps({"unsolicited":item})+"\n");self.event_stream.flush()
                continue
            self.event_stream.write(json.dumps({"request":command,"response":item},allow_nan=False)+"\n")
            self.event_stream.flush()
            return item
        raise TimeoutError(f"Live Nav2 {command['op']} timed out after {timeout}s")

    @staticmethod
    def _observation(response):
        source=response["observation"]
        obs={"state":np.asarray(source["state"],np.float32),
             "candidates":np.asarray(source["candidates"],np.float32),
             "mask":np.asarray(source["mask"],bool)}
        if obs["state"].shape!=(24,) or obs["candidates"].shape!=(16,12) or obs["mask"].shape!=(16,):
            raise ValueError("Live observation shape mismatch")
        if not np.isfinite(obs["state"]).all() or not np.isfinite(obs["candidates"]).all() or not obs["mask"].any():
            raise ValueError("Invalid live observation")
        return obs

    def reset(self,seed=None):
        self.close()
        self.episode_index+=1
        self.episode_dir=self.run_dir/f"episode_{self.episode_index:03d}"
        self.episode_dir.mkdir(exist_ok=False)
        self.event_stream=(self.episode_dir/"rpc.jsonl").open("w")
        self.error_stream=(self.episode_dir/"runner.stderr.log").open("w")
        self.responses=queue.Queue()
        command=[sys.executable,str(ROOT/"scripts/run_nav2_rl_session.py"),
            "--out",str(self.episode_dir/"execution"),"--wall-timeout-s",str(self.wall_timeout_s),
            "--scene","rmuc","--goal",*map(str,self.goal),"--start-footprint-prior",
            "--task-timeout-s",str(self.task_timeout_s),"--decision-dt-s",str(self.decision_dt_s)]
        if self.capture_terrain_input:
            command.append("--capture-terrain-input")
        if self.simulation_seed is not None:
            command += ["--simulation-seed",str(self.simulation_seed)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.error_stream,text=True,bufsize=1,start_new_session=True)
        self.reader=threading.Thread(target=self._reader,args=(self.process.stdout,),daemon=True)
        self.reader.start()
        self.last_response=self._rpc({"op":"reset"},self.wall_timeout_s+30.)
        self.obs=self._observation(self.last_response)
        self.return_total=0.
        self.transitions=[]
        return self.obs

    def step(self,action):
        if self.last_response is None or self.obs is None:raise RuntimeError("reset required")
        proposed=int(action)
        raw_before=self.last_response["raw"]
        response=self._rpc({"op":"step","action":proposed,
            "candidate_version":self.last_response["candidate_version"],
            "issued_at_s":self.last_response.get("issued_at_s",raw_before["now_s"])},max(30.,self.wall_timeout_s))
        obs=self._observation(response)
        raw=response["raw"]
        # Physics and sensors continue while host inference/updates run. The
        # transition starts at the previous observation, including that delay.
        duration=float(raw["now_s"]-raw_before["now_s"])
        if not np.isfinite(duration) or duration<0:raise ValueError("Invalid actual simulation duration")
        before=float(np.linalg.norm(np.asarray(raw_before["pose_xy_m"])-self.goal))
        after=float(np.linalg.norm(np.asarray(raw["pose_xy_m"])-self.goal))
        terminated=bool(response.get("terminated",False))
        truncated=bool(response.get("truncated",False))
        done=terminated or truncated
        navstatus=response.get("nav_status",raw.get("terminal_status"))
        success=bool(navstatus=="succeeded" and raw.get("mission_success",False) and after<=.2)
        veto=bool(response.get("BT_veto",raw.get("last_veto",False)))
        gamma=.99**(duration/.5)
        parts={"time":-.04*duration,"potential":(0. if terminated else -.5*after)*gamma+.5*before,
            "success":3. if success else 0.,"terminal_failure":-1. if terminated and not success else 0.,
            "veto":-.15 if veto else 0.}
        reward=float(sum(parts.values()))
        self.return_total+=reward
        info={**response,"terminated":terminated,"truncated":truncated,
            "rpc_execution_elapsed_s":response.get("elapsed_s"),
            "duration_s":duration,"elapsed_s":duration,"reward_components":parts,"BT_veto":veto,
            "proposed_action":proposed,"actual_navigation_backend":"PointLIO_Nav2_MPPI_MuJoCo",
            "goal_error_m":after,"success":success}
        provenance=dict(response.get("provenance",{}))
        provenance["real_nav2_in_loop"]=bool(provenance.get("real_nav2_in_loop",provenance.get("real_Nav2_in_loop",False))
            and provenance.get("PointLIO_estimated_odometry",False)
            and provenance.get("MPPI_controls_velocity",False)
            and not provenance.get("GT_used_as_actor_input",True))
        info["provenance"]=provenance
        if done:
            info["episode"]={"success":success,"return":self.return_total,
                "steps":len(self.transitions)+1,"time_s":float(raw.get("task_elapsed_s",0.)),
                "nav_status":navstatus,"final_estimated_goal_error_m":after,
                "execution_dir":str(self.episode_dir/"execution"),
                "actual_navigation_backend":"PointLIO_Nav2_MPPI_MuJoCo",
                "GT_policy_input":False,"physical_state_resumable":False}
        self.transitions.append({"action":proposed,"reward":reward,"info":info})
        self.obs,self.last_response=obs,response
        return obs,reward,done,info

    def close(self):
        process=self.process
        if process is None:return
        try:
            if process.poll() is None:
                try:self._rpc({"op":"close"},30.)
                except (OSError,RuntimeError,EOFError,TimeoutError):pass
                try:process.wait(timeout=40.)
                except subprocess.TimeoutExpired:
                    # Terminate the runner, whose handler cleans up its own ROS
                    # groups. Never use broad pkill or stop another container.
                    process.terminate()
                    try:process.wait(timeout=15.)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            self.final_artifacts.append({"episode_dir":str(self.episode_dir),"runner_returncode":process.returncode})
        finally:
            for pipe in (process.stdin,process.stdout):
                if pipe is not None:pipe.close()
            self.event_stream.close();self.error_stream.close()
            self.process=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
